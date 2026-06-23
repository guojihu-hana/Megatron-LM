# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Solve per-layer compute times from pipeline stage timings via linear equations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    _get_contiguous_layer_pattern,
    validate_segment_layers,
)

# E/L are pipeline bookends; M/-/L are hybrid pattern layer symbols.
LAYER_TIME_UNKNOWN_ORDER: Tuple[str, ...] = ("E", "M", "-", "*", "L")
PATTERN_LAYER_TYPES: Tuple[str, ...] = ("M", "-", "*")
TIMING_KEY_FORWARD = "forward_avg_ms_exc_1"
TIMING_KEY_BACKWARD = "backward_avg_ms_exc_1"
LayerTimeValues = Dict[str, Optional[float]]


@dataclass
class LayerEquation:
    """One stage timing equation."""

    equation_id: str
    sid: int
    partition: List[int]
    counts: Dict[str, float]
    measured_ms: float
    predicted_ms: float
    residual_ms: float


@dataclass
class IncrementalSolveResult:
    """Result of an incremental layer-time solve pass."""

    layer_times: LayerTimeValues
    equations: List[LayerEquation]
    newly_solved: List[str]
    still_undetermined: List[str]


@dataclass
class LayerTimeSystem:
    """Coefficient matrix and metadata for the layer-time linear system."""

    coefficient_matrix: np.ndarray
    equation_ids: List[str]
    stage_layer_counts: List[Dict[str, float]]
    equation_partitions: List[List[int]]
    equation_sids: List[int]
    unknown_order: Tuple[str, ...] = LAYER_TIME_UNKNOWN_ORDER

    @property
    def num_unknowns(self) -> int:
        return len(self.unknown_order)

    @property
    def num_equations(self) -> int:
        return self.coefficient_matrix.shape[0]

    def rank(self, tol: float = 1e-9) -> int:
        return int(np.linalg.matrix_rank(self.coefficient_matrix, tol=tol))


def _empty_layer_time_values() -> LayerTimeValues:
    return {name: None for name in LAYER_TIME_UNKNOWN_ORDER}


def _normalize_layer_time_values(
    values: Optional[Dict[str, Union[float, int, None]]],
) -> LayerTimeValues:
    normalized = _empty_layer_time_values()
    if not values:
        return normalized
    for name in LAYER_TIME_UNKNOWN_ORDER:
        val = values.get(name)
        normalized[name] = None if val is None else float(val)
    return normalized


def _partition_key(partition: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in partition)


def _equation_id(partition: Sequence[int], sid: int) -> str:
    return f"{list(partition)}:sid{sid}"


def load_layer_time_json(output_path: str) -> Dict:
    """Load an existing layer_times.json if present."""
    if not output_path or not os.path.isfile(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _migrate_legacy_layer_time_json(data)


def _migrate_legacy_layer_time_json(data: Dict) -> Dict:
    """Convert legacy single-partition JSON into equation_sets storage."""
    if not data:
        return data
    if "equation_sets" in data:
        return data

    legacy_equations = data.get("equations")
    legacy_partition = data.get("partition")
    if legacy_equations and legacy_partition:
        equation_sets = [{
            "partition": list(legacy_partition),
            "layout_mode": data.get("layout_mode", "unknown"),
            "pattern": data.get("pattern", ""),
            "equations": legacy_equations,
        }]
        data["equation_sets"] = equation_sets
    else:
        data.setdefault("equation_sets", [])
    return data


def get_stored_equation_sets(
    output_path: str,
    reset: bool,
) -> List[Dict]:
    """Return stored equation sets unless reset is requested."""
    if reset:
        return []
    data = load_layer_time_json(output_path)
    return list(data.get("equation_sets", []))


def _get_main_hybrid_pattern(hybrid_layer_pattern: str) -> str:
    """Return the main decoder pattern (strip MTP suffix after '/')."""
    return hybrid_layer_pattern.split(Symbols.MTP_SEPARATOR)[0]


def _canonical_layer_pattern(hybrid_layer_pattern: str) -> str:
    """Normalize pattern for comparison (strip MTP suffix and pipe markers)."""
    return _get_contiguous_layer_pattern(_get_main_hybrid_pattern(hybrid_layer_pattern))


def _patterns_are_compatible(stored_pattern: str, current_pattern: str) -> bool:
    return _canonical_layer_pattern(stored_pattern) == _canonical_layer_pattern(current_pattern)


def _filter_compatible_equation_sets(
    equation_sets: Sequence[Dict],
    hybrid_layer_pattern: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Keep only equation sets whose layer sequence matches the current pattern."""
    compatible: List[Dict] = []
    skipped: List[Dict] = []
    for equation_set in equation_sets:
        stored_pattern = equation_set.get("pattern", hybrid_layer_pattern)
        if _patterns_are_compatible(stored_pattern, hybrid_layer_pattern):
            compatible.append(equation_set)
        else:
            skipped.append(equation_set)
    return compatible, skipped


def _validate_pattern_layer_types(layer_type_list: Sequence[str]) -> None:
    unsupported = sorted(
        {ch for ch in layer_type_list if ch not in PATTERN_LAYER_TYPES}
    )
    if unsupported:
        raise ValueError(
            f"--profile-layer-time currently supports pattern layer types "
            f"{PATTERN_LAYER_TYPES} only; found unsupported types: {unsupported}."
        )


def _compute_auto_pp_partition(
    num_layers: int,
    pp_size: int,
    first_stage_layers: Optional[int] = None,
    last_stage_layers: Optional[int] = None,
) -> List[int]:
    """Mirror even/uneven PP layer splits used by select_pipeline_segment."""
    if pp_size == 1:
        return [num_layers]

    if first_stage_layers is not None or last_stage_layers is not None:
        first = first_stage_layers or 0
        last = last_stage_layers or 0
        middle_num_layers = num_layers - first - last
        middle_stages = pp_size - sum(
            1 for x in (first_stage_layers, last_stage_layers) if x is not None
        )
        if middle_stages > 0:
            if middle_num_layers % middle_stages != 0:
                raise ValueError(
                    f"Middle layers ({middle_num_layers}) must be evenly divisible "
                    f"by middle pipeline stages ({middle_stages}) for layer-time profiling."
                )
            layers_per_middle = middle_num_layers // middle_stages
        else:
            layers_per_middle = 0

        partition: List[int] = []
        for pp_rank in range(pp_size):
            is_first = first_stage_layers is not None and pp_rank == 0
            is_last = last_stage_layers is not None and pp_rank == pp_size - 1
            if is_first:
                partition.append(first)
            elif is_last:
                partition.append(last)
            else:
                partition.append(layers_per_middle)
        return partition

    if num_layers % pp_size != 0:
        raise ValueError(
            f"Number of layers ({num_layers}) must be evenly divisible by "
            f"pipeline-model-parallel-size ({pp_size}) when hybrid_layer_pattern "
            f"has no pipe ('|') separators. Add '|' separators to define explicit "
            f"stage boundaries, or adjust --num-layers / PP size."
        )
    layers_per_rank = num_layers // pp_size
    return [layers_per_rank] * pp_size


def derive_stage_layout(
    hybrid_layer_pattern: str,
    pipeline_model_parallel_size: int,
    decoder_first_pipeline_num_layers: Optional[int] = None,
    decoder_last_pipeline_num_layers: Optional[int] = None,
    octopipe_partition: Optional[Sequence[int]] = None,
) -> Tuple[List[int], int, str]:
    """Derive per-stage layer counts for the layer-time equation system."""
    if octopipe_partition is not None:
        partition = list(octopipe_partition)
        return partition, len(partition), "octopipe"

    main_pattern = _get_main_hybrid_pattern(hybrid_layer_pattern)
    if Symbols.PIPE in main_pattern:
        segments = main_pattern.split(Symbols.PIPE)
        partition = [len(validate_segment_layers(seg)) for seg in segments]
        for seg in segments:
            _validate_pattern_layer_types(validate_segment_layers(seg))
        return partition, len(partition), "pipe"

    full_pattern = _get_contiguous_layer_pattern(main_pattern)
    layer_type_list = validate_segment_layers(full_pattern)
    _validate_pattern_layer_types(layer_type_list)

    pp_size = pipeline_model_parallel_size
    partition = _compute_auto_pp_partition(
        len(layer_type_list),
        pp_size,
        decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers,
    )
    if pp_size == 1:
        return partition, 1, "single_pp"
    return partition, len(partition), "pp_even"


def build_layer_time_system(
    partition: Sequence[int],
    hybrid_layer_pattern: str,
    stage_num: Optional[int] = None,
) -> LayerTimeSystem:
    """Build Ax=b coefficient matrix from per-stage layer counts and hybrid pattern."""
    if stage_num is None:
        stage_num = len(partition)
    if len(partition) != stage_num:
        raise ValueError(
            f"partition length ({len(partition)}) must match stage_num ({stage_num})."
        )

    full_pattern = _get_contiguous_layer_pattern(hybrid_layer_pattern)
    layer_type_list = validate_segment_layers(full_pattern)
    if len(layer_type_list) != sum(partition):
        raise ValueError(
            f"Hybrid layer pattern length ({len(layer_type_list)}) does not match "
            f"partition total ({sum(partition)}): {list(partition)}."
        )

    unsupported = sorted(
        {ch for ch in layer_type_list if ch not in PATTERN_LAYER_TYPES}
    )
    if unsupported:
        raise ValueError(
            f"--profile-layer-time currently supports pattern layer types "
            f"{PATTERN_LAYER_TYPES} only; found unsupported types: {unsupported}."
        )

    layer_idx_offset = [sum(partition[0:i]) for i in range(len(partition) + 1)]
    unknown_index = {name: idx for idx, name in enumerate(LAYER_TIME_UNKNOWN_ORDER)}

    rows: List[List[float]] = []
    equation_ids: List[str] = []
    stage_layer_counts: List[Dict[str, float]] = []
    equation_partitions: List[List[int]] = []
    equation_sids: List[int] = []

    for sid in range(stage_num):
        offset = layer_idx_offset[sid]
        count = partition[sid]
        segment = layer_type_list[offset : offset + count]

        row = [0.0] * len(LAYER_TIME_UNKNOWN_ORDER)
        counts = {name: 0.0 for name in LAYER_TIME_UNKNOWN_ORDER}

        if sid == 0:
            row[unknown_index["E"]] = 1.0
            counts["E"] = 1.0
        if sid == stage_num - 1:
            row[unknown_index["L"]] = 1.0
            counts["L"] = 1.0

        for layer_char in segment:
            row[unknown_index[layer_char]] += 1.0
            counts[layer_char] += 1.0

        rows.append(row)
        equation_ids.append(_equation_id(partition, sid))
        stage_layer_counts.append(counts)
        equation_partitions.append(list(partition))
        equation_sids.append(sid)

    return LayerTimeSystem(
        coefficient_matrix=np.array(rows, dtype=np.float64),
        equation_ids=equation_ids,
        stage_layer_counts=stage_layer_counts,
        equation_partitions=equation_partitions,
        equation_sids=equation_sids,
    )


def _build_measurement_records(
    system: LayerTimeSystem,
    measurements_ms: Sequence[float],
) -> List[Dict]:
    records: List[Dict] = []
    for sid, counts, measured in zip(
        system.equation_sids,
        system.stage_layer_counts,
        measurements_ms,
    ):
        records.append({
            "sid": sid,
            "counts": counts,
            "measured_ms": float(measured),
        })
    return records


def upsert_equation_set(
    equation_sets: List[Dict],
    partition: Sequence[int],
    layout_mode: str,
    hybrid_layer_pattern: str,
    forward_records: List[Dict],
    backward_records: List[Dict],
) -> List[Dict]:
    """Insert or replace the equation set for a partition."""
    target_key = _partition_key(partition)
    new_set = {
        "partition": list(partition),
        "layout_mode": layout_mode,
        "pattern": hybrid_layer_pattern,
        "equations": {
            "forward": forward_records,
            "backward": backward_records,
        },
    }
    for idx, equation_set in enumerate(equation_sets):
        if _partition_key(equation_set["partition"]) == target_key:
            equation_sets[idx] = new_set
            return equation_sets
    equation_sets.append(new_set)
    return equation_sets


def build_combined_layer_time_system(
    equation_sets: Sequence[Dict],
    hybrid_layer_pattern: str,
    direction: str,
) -> LayerTimeSystem:
    """Stack all stored equation rows for one direction into a combined system."""
    rows: List[List[float]] = []
    equation_ids: List[str] = []
    stage_layer_counts: List[Dict[str, float]] = []
    equation_partitions: List[List[int]] = []
    equation_sids: List[int] = []

    for equation_set in equation_sets:
        stored_pattern = equation_set.get("pattern", hybrid_layer_pattern)
        if not _patterns_are_compatible(stored_pattern, hybrid_layer_pattern):
            continue
        partition = equation_set["partition"]
        system = build_layer_time_system(partition, hybrid_layer_pattern)
        records = equation_set["equations"][direction]
        record_by_sid = {int(rec["sid"]): rec for rec in records}

        for sid, row_counts, row in zip(
            system.equation_sids,
            system.stage_layer_counts,
            system.coefficient_matrix,
        ):
            record = record_by_sid.get(sid)
            if record is None:
                raise ValueError(
                    f"Missing {direction} measurement for partition={partition}, sid={sid}."
                )
            rows.append(row.tolist())
            equation_ids.append(_equation_id(partition, sid))
            stage_layer_counts.append(record.get("counts", row_counts))
            equation_partitions.append(list(partition))
            equation_sids.append(sid)

    if not rows:
        raise ValueError("No stored equations available to build a combined system.")

    combined = LayerTimeSystem(
        coefficient_matrix=np.array(rows, dtype=np.float64),
        equation_ids=equation_ids,
        stage_layer_counts=stage_layer_counts,
        equation_partitions=equation_partitions,
        equation_sids=equation_sids,
    )
    return combined


def _find_proportional_stage_pairs(
    system: LayerTimeSystem,
    rtol: float = 1e-9,
) -> List[Tuple[str, str, List[float], List[float]]]:
    pairs: List[Tuple[str, str, List[float], List[float]]] = []
    matrix = system.coefficient_matrix

    for i in range(len(system.equation_ids)):
        for j in range(i + 1, len(system.equation_ids)):
            row_i = matrix[i]
            row_j = matrix[j]
            mask = np.abs(row_i) + np.abs(row_j) > rtol
            if not np.any(mask):
                pairs.append(
                    (
                        system.equation_ids[i],
                        system.equation_ids[j],
                        row_i.tolist(),
                        row_j.tolist(),
                    )
                )
                continue
            if not np.all(mask):
                continue
            ratios = row_i[mask] / row_j[mask]
            if np.allclose(ratios, ratios[0], rtol=rtol, atol=rtol):
                pairs.append(
                    (
                        system.equation_ids[i],
                        system.equation_ids[j],
                        row_i.tolist(),
                        row_j.tolist(),
                    )
                )
    return pairs


def _identify_unappearing_unknowns(system: LayerTimeSystem) -> List[str]:
    unappearing = []
    for unknown in system.unknown_order:
        col_sum = float(np.sum(system.coefficient_matrix[:, system.unknown_order.index(unknown)]))
        if col_sum == 0.0:
            unappearing.append(unknown)
    return unappearing


def analyze_incremental_solvability(
    system: LayerTimeSystem,
    known_ms: Optional[LayerTimeValues] = None,
    tol: float = 1e-9,
) -> Tuple[List[str], List[str], List[str]]:
    """Estimate which unknowns can be solved from the combined equation system."""
    known_ms = _normalize_layer_time_values(known_ms)
    known_names = [name for name in system.unknown_order if known_ms[name] is not None]
    undetermined = {name for name in system.unknown_order if known_ms[name] is None}

    order = system.unknown_order
    A = system.coefficient_matrix

    potentially_solvable: set[str] = set()
    remaining = set(undetermined)
    while True:
        trial = set(remaining)
        dummy_x = np.zeros(len(order), dtype=np.float64)
        _propagate_isolated_unknowns(
            A,
            np.zeros(A.shape[0], dtype=np.float64),
            np.zeros(A.shape[0], dtype=np.float64),
            dummy_x,
            order,
            trial,
            [],
            tol,
        )
        newly = remaining - trial
        if not newly:
            break
        potentially_solvable.update(newly)
        remaining = trial

    undetermined = remaining
    if undetermined:
        idx = [order.index(name) for name in order if name in undetermined]
        A_red = A[:, idx]
        if A_red.size > 0:
            rank = np.linalg.matrix_rank(A_red, tol=tol)
            if rank == len(idx):
                potentially_solvable.update(undetermined)
                undetermined.clear()

    still_undetermined = sorted(
        set(_identify_unappearing_unknowns(system)) | undetermined
    )
    return known_names, sorted(potentially_solvable), still_undetermined


def check_layer_time_system_solvable(
    system: LayerTimeSystem,
    tol: float = 1e-9,
) -> Tuple[bool, int, List[str]]:
    """Return whether the combined system has a unique solution."""
    rank = system.rank(tol=tol)
    if rank == system.num_unknowns:
        return True, rank, []

    suggestions: List[str] = [
        f"Combined equation system rank is {rank}, but {system.num_unknowns} linearly "
        f"independent equations are required for a unique solution "
        f"({system.num_equations} stored equations across partitions).",
        "Add equation sets with different partitions to determine remaining layer types.",
    ]

    if system.num_equations < system.num_unknowns:
        suggestions.append(
            f"  - Need more equations (currently {system.num_equations}); "
            f"rerun profiling with additional partitions."
        )

    proportional_pairs = _find_proportional_stage_pairs(system, rtol=tol)
    for eq_i, eq_j, row_i, row_j in proportional_pairs:
        suggestions.append(
            f"  - Equations {eq_i} and {eq_j} are linearly dependent "
            f"({row_i} vs {row_j})."
        )

    for unknown in _identify_unappearing_unknowns(system):
        suggestions.append(
            f"  - Unknown '{unknown}' never appears in any stored equation."
        )

    return False, rank, suggestions


def collect_sid_timings(
    pp_timing_dir: str,
    num_sids: int,
    timing_key: str,
) -> List[float]:
    """Load per-sid timing from pp_*.json files written during training."""
    timings: Dict[int, float] = {}

    if not os.path.isdir(pp_timing_dir):
        raise FileNotFoundError(
            f"pp_timing directory not found: {pp_timing_dir}"
        )

    for fname in os.listdir(pp_timing_dir):
        if not fname.startswith("pp_") or not fname.endswith(".json"):
            continue
        path = os.path.join(pp_timing_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sid_str, stats in data.get("vp_stages", {}).items():
            sid = int(sid_str)
            if sid in timings:
                raise ValueError(
                    f"Duplicate timing entry for sid {sid} in {pp_timing_dir}."
                )
            timings[sid] = float(stats.get(timing_key, 0.0))

    missing = [sid for sid in range(num_sids) if sid not in timings]
    if missing:
        raise ValueError(
            f"Missing pp_timing measurements for sids {missing} under {pp_timing_dir}."
        )

    return [timings[sid] for sid in range(num_sids)]


def _try_isolate_single_unknown_from_row(
    row: np.ndarray,
    rhs: float,
    order: Tuple[str, ...],
    undetermined: set[str],
    tol: float,
) -> Optional[Tuple[str, float]]:
    pending = [
        (j, row[j])
        for j in range(len(order))
        if order[j] in undetermined and abs(row[j]) > tol
    ]
    if len(pending) != 1:
        return None
    j, coeff = pending[0]
    return order[j], float(rhs / coeff)


def _propagate_isolated_unknowns(
    A: np.ndarray,
    b: np.ndarray,
    b_adj: np.ndarray,
    x: np.ndarray,
    order: Tuple[str, ...],
    undetermined: set[str],
    newly_solved: List[str],
    tol: float,
) -> bool:
    changed = False

    for i in range(A.shape[0]):
        isolated = _try_isolate_single_unknown_from_row(
            A[i], b_adj[i], order, undetermined, tol
        )
        if isolated is None:
            continue
        name, value = isolated
        j = order.index(name)
        x[j] = value
        undetermined.remove(name)
        if name not in newly_solved:
            newly_solved.append(name)
        b_adj[:] = b - A @ x
        changed = True

    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            diff = A[i] - A[j]
            isolated = _try_isolate_single_unknown_from_row(
                diff, b_adj[i] - b_adj[j], order, undetermined, tol
            )
            if isolated is None:
                continue
            name, value = isolated
            col = order.index(name)
            x[col] = value
            undetermined.remove(name)
            if name not in newly_solved:
                newly_solved.append(name)
            b_adj[:] = b - A @ x
            changed = True

    return changed


def solve_layer_times_incremental(
    system: LayerTimeSystem,
    measurements_ms: Sequence[float],
    known_ms: Optional[LayerTimeValues] = None,
    tol: float = 1e-9,
) -> IncrementalSolveResult:
    """Incrementally solve layer times from a combined equation system."""
    known_ms = _normalize_layer_time_values(known_ms)
    order = system.unknown_order
    A = system.coefficient_matrix
    b = np.asarray(measurements_ms, dtype=np.float64)
    if b.shape[0] != system.num_equations:
        raise ValueError(
            f"Expected {system.num_equations} measurements, got {b.shape[0]}."
        )

    x = np.array([known_ms[name] or 0.0 for name in order], dtype=np.float64)
    undetermined = {name for name in order if known_ms[name] is None}
    newly_solved: List[str] = []

    b_adj = b - A @ x

    changed = True
    while changed:
        changed = _propagate_isolated_unknowns(
            A, b, b_adj, x, order, undetermined, newly_solved, tol
        )

    if undetermined:
        idx = [order.index(name) for name in order if name in undetermined]
        A_red = A[:, idx]
        if A_red.size > 0:
            rank = np.linalg.matrix_rank(A_red, tol=tol)
            if rank == len(idx):
                solution, _, _, _ = np.linalg.lstsq(A_red, b_adj, rcond=None)
                for col, j in enumerate(idx):
                    name = order[j]
                    x[j] = float(solution[col])
                    undetermined.remove(name)
                    if name not in newly_solved:
                        newly_solved.append(name)
                b_adj = b - A @ x

    for name in _identify_unappearing_unknowns(system):
        undetermined.add(name)

    layer_times: LayerTimeValues = {
        name: (float(x[i]) if name not in undetermined else None)
        for i, name in enumerate(order)
    }

    x_pred = x.copy()
    for name in undetermined:
        x_pred[order.index(name)] = 0.0
    predicted = A @ x_pred
    equations: List[LayerEquation] = []
    for eq_id, sid, partition, counts, measured, pred in zip(
        system.equation_ids,
        system.equation_sids,
        system.equation_partitions,
        system.stage_layer_counts,
        b,
        predicted,
    ):
        equations.append(
            LayerEquation(
                equation_id=eq_id,
                sid=sid,
                partition=partition,
                counts=counts,
                measured_ms=float(measured),
                predicted_ms=float(pred),
                residual_ms=float(measured - pred),
            )
        )

    return IncrementalSolveResult(
        layer_times=layer_times,
        equations=equations,
        newly_solved=newly_solved,
        still_undetermined=sorted(undetermined),
    )


def _equations_to_dict(eqs: List[LayerEquation]) -> List[Dict]:
    return [
        {
            "equation_id": eq.equation_id,
            "sid": eq.sid,
            "partition": eq.partition,
            "counts": eq.counts,
            "measured_ms": eq.measured_ms,
            "predicted_ms": eq.predicted_ms,
            "residual_ms": eq.residual_ms,
        }
        for eq in eqs
    ]


def _attach_solve_metadata_to_equation_sets(
    equation_sets: List[Dict],
    forward_equations: List[LayerEquation],
    backward_equations: List[LayerEquation],
) -> List[Dict]:
    """Attach latest solve predictions back to stored equation records."""
    forward_by_id = {eq.equation_id: eq for eq in forward_equations}
    backward_by_id = {eq.equation_id: eq for eq in backward_equations}

    updated_sets: List[Dict] = []
    for equation_set in equation_sets:
        updated = json.loads(json.dumps(equation_set))
        for direction, lookup in (
            ("forward", forward_by_id),
            ("backward", backward_by_id),
        ):
            new_records = []
            for record in updated["equations"][direction]:
                eq_id = _equation_id(updated["partition"], int(record["sid"]))
                solved = lookup.get(eq_id)
                new_record = dict(record)
                if solved is not None:
                    new_record["predicted_ms"] = solved.predicted_ms
                    new_record["residual_ms"] = solved.residual_ms
                new_records.append(new_record)
            updated["equations"][direction] = new_records
        updated_sets.append(updated)
    return updated_sets


def build_layer_time_result(
    model_name: str,
    hybrid_layer_pattern: str,
    system: LayerTimeSystem,
    equation_sets: List[Dict],
    forward_ms: LayerTimeValues,
    backward_ms: LayerTimeValues,
    forward_equations: List[LayerEquation],
    backward_equations: List[LayerEquation],
    forward_newly_solved: Optional[List[str]] = None,
    backward_newly_solved: Optional[List[str]] = None,
    reset: bool = False,
    prior_loaded: bool = False,
    current_partition: Optional[Sequence[int]] = None,
    skipped_equation_sets: Optional[List[Dict]] = None,
) -> Dict:
    """Build the JSON-serializable layer time profiling result."""
    complete = all(
        forward_ms[name] is not None and backward_ms[name] is not None
        for name in system.unknown_order
    )

    return {
        "model": model_name,
        "pattern": hybrid_layer_pattern,
        "unknown_order": list(system.unknown_order),
        "matrix_rank": system.rank(),
        "num_equations": system.num_equations,
        "num_partitions": len(equation_sets),
        "complete": complete,
        "reset_on_last_run": reset,
        "prior_loaded": prior_loaded,
        "current_partition": list(current_partition) if current_partition is not None else None,
        "skipped_equation_sets": skipped_equation_sets or [],
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "undetermined_forward": [
            name for name in system.unknown_order if forward_ms[name] is None
        ],
        "undetermined_backward": [
            name for name in system.unknown_order if backward_ms[name] is None
        ],
        "newly_solved_forward": forward_newly_solved or [],
        "newly_solved_backward": backward_newly_solved or [],
        "equation_sets": _attach_solve_metadata_to_equation_sets(
            equation_sets, forward_equations, backward_equations
        ),
        "combined_equations": {
            "forward": _equations_to_dict(forward_equations),
            "backward": _equations_to_dict(backward_equations),
        },
        "coefficient_matrix": {
            "rows": system.equation_ids,
            "cols": list(system.unknown_order),
            "values": system.coefficient_matrix.tolist(),
        },
    }


def prepare_layer_time_profiling_for_training(
    hybrid_layer_pattern: str,
    pipeline_model_parallel_size: int,
    output_path: str,
    reset_profiled_layer_time: bool,
    decoder_first_pipeline_num_layers: Optional[int] = None,
    decoder_last_pipeline_num_layers: Optional[int] = None,
    octopipe_partition: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict], List[int], int, str, str]:
    """Summarize stored equation sets and the partition for the current run."""
    partition, stage_num, layout_mode = derive_stage_layout(
        hybrid_layer_pattern=hybrid_layer_pattern,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        decoder_first_pipeline_num_layers=decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers=decoder_last_pipeline_num_layers,
        octopipe_partition=octopipe_partition,
    )

    prior_loaded = (not reset_profiled_layer_time) and os.path.isfile(output_path)
    equation_sets = get_stored_equation_sets(output_path, reset_profiled_layer_time)

    status_lines = [
        f"Layer time profiling: current layout={layout_mode}, partition={partition}.",
        f"  Stored equation sets: {len(equation_sets)} "
        f"(reset={reset_profiled_layer_time}, prior_loaded={prior_loaded}).",
    ]

    if equation_sets:
        compatible_sets, skipped_sets = _filter_compatible_equation_sets(
            equation_sets, hybrid_layer_pattern
        )
        if skipped_sets:
            status_lines.append(
                f"  Skipping {len(skipped_sets)} stored equation set(s) with incompatible "
                f"layer patterns."
            )
        if compatible_sets:
            try:
                combined_forward = build_combined_layer_time_system(
                    compatible_sets, hybrid_layer_pattern, "forward"
                )
                combined_backward = build_combined_layer_time_system(
                    compatible_sets, hybrid_layer_pattern, "backward"
                )
                _, potentially_forward, undetermined_forward = analyze_incremental_solvability(
                    combined_forward
                )
                _, potentially_backward, undetermined_backward = analyze_incremental_solvability(
                    combined_backward
                )
                solvable_f, rank_f, _ = check_layer_time_system_solvable(combined_forward)
                solvable_b, rank_b, _ = check_layer_time_system_solvable(combined_backward)
                status_lines.extend([
                    f"  Stored forward equations={combined_forward.num_equations}, "
                    f"rank={rank_f}/{combined_forward.num_unknowns}, "
                    f"may solve={potentially_forward}, undetermined={undetermined_forward}.",
                    f"  Stored backward equations={combined_backward.num_equations}, "
                    f"rank={rank_b}/{combined_backward.num_unknowns}, "
                    f"may solve={potentially_backward}, undetermined={undetermined_backward}.",
                ])
                if not solvable_f or not solvable_b:
                    status_lines.append(
                        "  Combined solve is not yet unique; current run will upsert this "
                        "partition and re-solve from all stored equation sets."
                    )
            except ValueError as exc:
                status_lines.append(f"  Stored equation analysis skipped: {exc}")
    else:
        status_lines.append(
            "  No stored equation sets yet; this run will create the first partition entry."
        )

    return equation_sets, partition, stage_num, layout_mode, "\n".join(status_lines)


def solve_and_dump_layer_times(
    partition: Sequence[int],
    hybrid_layer_pattern: str,
    stage_num: int,
    pp_timing_dir: str,
    output_path: str,
    model_name: str,
    layout_mode: str = "unknown",
    reset_profiled_layer_time: bool = False,
) -> str:
    """Upsert current partition equations and solve from all stored equation sets."""
    current_system = build_layer_time_system(partition, hybrid_layer_pattern, stage_num)
    prior_loaded = (not reset_profiled_layer_time) and os.path.isfile(output_path)
    equation_sets = get_stored_equation_sets(output_path, reset_profiled_layer_time)

    forward_measurements = collect_sid_timings(
        pp_timing_dir, stage_num, TIMING_KEY_FORWARD
    )
    backward_measurements = collect_sid_timings(
        pp_timing_dir, stage_num, TIMING_KEY_BACKWARD
    )

    forward_records = _build_measurement_records(current_system, forward_measurements)
    backward_records = _build_measurement_records(current_system, backward_measurements)
    equation_sets = upsert_equation_set(
        equation_sets,
        partition=partition,
        layout_mode=layout_mode,
        hybrid_layer_pattern=hybrid_layer_pattern,
        forward_records=forward_records,
        backward_records=backward_records,
    )

    compatible_sets, skipped_sets = _filter_compatible_equation_sets(
        equation_sets, hybrid_layer_pattern
    )
    if skipped_sets:
        print(
            f"Layer time profiling: skipped {len(skipped_sets)} stored equation set(s) "
            f"with incompatible layer patterns.",
            flush=True,
        )

    combined_forward = build_combined_layer_time_system(
        compatible_sets, hybrid_layer_pattern, "forward"
    )
    combined_backward = build_combined_layer_time_system(
        compatible_sets, hybrid_layer_pattern, "backward"
    )
    forward_measurements_all = [
        float(rec["measured_ms"])
        for equation_set in compatible_sets
        for rec in equation_set["equations"]["forward"]
    ]
    backward_measurements_all = [
        float(rec["measured_ms"])
        for equation_set in compatible_sets
        for rec in equation_set["equations"]["backward"]
    ]

    prior_forward, prior_backward = _empty_layer_time_values(), _empty_layer_time_values()
    forward_result = solve_layer_times_incremental(
        combined_forward, forward_measurements_all, known_ms=prior_forward
    )
    backward_result = solve_layer_times_incremental(
        combined_backward, backward_measurements_all, known_ms=prior_backward
    )

    result = build_layer_time_result(
        model_name=model_name,
        hybrid_layer_pattern=hybrid_layer_pattern,
        system=combined_forward,
        equation_sets=equation_sets,
        forward_ms=forward_result.layer_times,
        backward_ms=backward_result.layer_times,
        forward_equations=forward_result.equations,
        backward_equations=backward_result.equations,
        forward_newly_solved=forward_result.newly_solved,
        backward_newly_solved=backward_result.newly_solved,
        reset=reset_profiled_layer_time,
        prior_loaded=prior_loaded,
        current_partition=partition,
        skipped_equation_sets=skipped_sets,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return output_path
