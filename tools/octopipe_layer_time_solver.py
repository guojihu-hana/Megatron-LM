#!/usr/bin/env python3
"""Solve per-layer OctoPipe f/b/w times from a training log.

The script reads two pieces of information from a log:
  1. Hybrid layer allocation lines, keyed by OctoPipe sid/vp_stage, or a
     user-provided layer pattern split by the OctoPipe YAML partition.
  2. Final [octopipe-stage-time] profiler summaries, also keyed by sid.

For each op (f, b, w), it solves a non-negative least-squares problem:

    stage_time_ms ~= sum(count(layer_type on stage) * layer_type_time_ms)

If the log does not explicitly place E/L, E is added to the first stage and L
is added to the last stage by default.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ALLOC_RE = re.compile(
    r"HybridModel:\s+pp_rank=(?P<pp_rank>\d+)/(?:\d+),\s+"
    r"vp_stage=(?P<sid>\d+|None),\s+layers='(?P<layers>[^']*)'"
)
PROFILE_RE = re.compile(
    r"\[octopipe-stage-time\]\[rank=(?P<rank>\d+)\s+pp_rank=(?P<pp_rank>\d+)\s+"
    r"steps=(?P<steps>\d+)\s+sid=(?P<sid>\d+)\]\s+"
    r"(?P<body>.*?)(?=\[octopipe-stage-time\]|\r?\n?$)"
)
OP_RE = re.compile(
    r"(?P<op>[fbw]):count=(?P<count>\d+)\s+"
    r"sum=(?P<sum>[0-9.]+)ms\s+"
    r"avg=(?P<avg>[0-9.]+)ms\s+"
    r"min=(?P<min>[0-9.]+)ms\s+"
    r"max=(?P<max>[0-9.]+)ms"
)
CONFIG_YAML_RE = re.compile(r"\boctopipe_config_yaml\s+\.*\s*(?P<path>\S+)")
PARTITION_INLINE_RE = re.compile(r"^\s*partition:\s*(?P<value>.*?)\s*(?:#.*)?$")


@dataclass
class StageProfile:
    sums: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, op: str, count: int, total_ms: float) -> None:
        self.sums[op] += total_ms
        self.counts[op] += count

    def avg(self, op: str) -> Optional[float]:
        count = self.counts.get(op, 0)
        if count <= 0:
            return None
        return self.sums[op] / count


@dataclass
class ParsedLog:
    layers_by_sid: Dict[int, str]
    profiles_by_sid: Dict[int, StageProfile]
    warnings: List[str]
    octopipe_config_yaml: Optional[str] = None
    layer_source: str = "HybridModel log lines"


def parse_log(path: Path) -> ParsedLog:
    layers_by_sid: Dict[int, str] = {}
    profiles_by_sid: Dict[int, StageProfile] = defaultdict(StageProfile)
    warnings: List[str] = []
    octopipe_config_yaml: Optional[str] = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            config_match = CONFIG_YAML_RE.search(line)
            if config_match:
                config_path = config_match.group("path")
                octopipe_config_yaml = None if config_path == "None" else config_path

            alloc_match = ALLOC_RE.search(line)
            if alloc_match:
                sid_text = alloc_match.group("sid")
                sid = int(alloc_match.group("pp_rank") if sid_text == "None" else sid_text)
                layers = alloc_match.group("layers")
                old_layers = layers_by_sid.get(sid)
                if old_layers is not None and old_layers != layers:
                    warnings.append(
                        f"line {lineno}: sid={sid} allocation changed from "
                        f"{old_layers!r} to {layers!r}; using the latest value"
                    )
                layers_by_sid[sid] = layers
                continue

            for profile_match in PROFILE_RE.finditer(line):
                sid = int(profile_match.group("sid"))
                body = profile_match.group("body")
                found_ops = set()
                for op_match in OP_RE.finditer(body):
                    op = op_match.group("op")
                    count = int(op_match.group("count"))
                    total_ms = float(op_match.group("sum"))
                    profiles_by_sid[sid].add(op, count, total_ms)
                    found_ops.add(op)
                missing_ops = {"f", "b", "w"} - found_ops
                if missing_ops:
                    warnings.append(
                        f"line {lineno}: sid={sid} profile missing ops "
                        f"{','.join(sorted(missing_ops))}"
                    )

    return ParsedLog(dict(layers_by_sid), dict(profiles_by_sid), warnings, octopipe_config_yaml)


def expand_layer_pattern(pattern: str) -> str:
    """Expand a compact pattern like F*2M*26 into FFMMMM..."""
    layers: List[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char.isspace() or char in {",", "|"}:
            index += 1
            continue
        if char.isdigit() or char == "*":
            raise RuntimeError(f"invalid layer pattern near {pattern[index:]!r}")

        index += 1
        repeat = 1
        if index < len(pattern) and pattern[index] == "*":
            index += 1
            start = index
            while index < len(pattern) and pattern[index].isdigit():
                index += 1
            if start == index:
                raise RuntimeError(f"missing repeat count in layer pattern near {char!r}")
            repeat = int(pattern[start:index])
            if repeat <= 0:
                raise RuntimeError("layer pattern repeat count must be positive")
        layers.append(char * repeat)

    expanded = "".join(layers)
    if not expanded:
        raise RuntimeError("layer pattern expanded to an empty layer sequence")
    return expanded


def resolve_config_yaml_path(
    config_yaml: Optional[str], log_path: Path, explicit_config_yaml: Optional[Path]
) -> Path:
    if explicit_config_yaml is not None:
        return explicit_config_yaml
    if not config_yaml:
        raise RuntimeError(
            "no octopipe_config_yaml found in log; pass --octopipe-config-yaml explicitly"
        )

    raw_path = Path(config_yaml)
    if raw_path.is_absolute() or raw_path.exists():
        return raw_path

    candidates = [Path.cwd() / raw_path]
    candidates.extend(parent / raw_path for parent in [log_path.parent, *log_path.parents])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw_path


def _parse_partition_without_yaml(path: Path) -> List[int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_index, line in enumerate(lines):
        match = PARTITION_INLINE_RE.match(line)
        if not match:
            continue

        value = match.group("value").strip()
        if value:
            try:
                partition = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise RuntimeError(f"cannot parse partition in {path}: {value!r}") from exc
            if not isinstance(partition, list):
                raise RuntimeError(f"partition in {path} must be a list")
            return [int(item) for item in partition]

        partition: List[int] = []
        for block_line in lines[line_index + 1 :]:
            if block_line and not block_line[0].isspace():
                break
            stripped = block_line.strip()
            if not stripped:
                continue
            if not stripped.startswith("-"):
                break
            partition.append(int(stripped[1:].strip().split()[0]))
        if partition:
            return partition
        break

    raise RuntimeError(f"partition not found in {path}")


def read_partition_from_octopipe_yaml(path: Path) -> List[int]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8", errors="replace") as handle:
            data = yaml.safe_load(handle)
        partition = data.get("partition") if isinstance(data, dict) else None
        if partition is None:
            raise RuntimeError(f"partition not found in {path}")
        return [int(item) for item in partition]
    except ImportError:
        return _parse_partition_without_yaml(path)


def layers_by_sid_from_pattern(pattern: str, partition: Sequence[int]) -> Dict[int, str]:
    expanded = expand_layer_pattern(pattern)
    total_layers = sum(partition)
    if total_layers != len(expanded):
        raise RuntimeError(
            f"partition sum ({total_layers}) does not match expanded pattern length "
            f"({len(expanded)})"
        )

    layers_by_sid: Dict[int, str] = {}
    offset = 0
    for sid, size in enumerate(partition):
        if size < 0:
            raise RuntimeError("partition entries must be non-negative")
        layers_by_sid[sid] = expanded[offset : offset + size]
        offset += size
    return layers_by_sid


def layer_counts_by_sid(
    layers_by_sid: Mapping[int, str], add_default_el: bool = True
) -> Tuple[Dict[int, Counter], bool, bool]:
    counts = {sid: Counter(layers) for sid, layers in layers_by_sid.items()}
    if not counts or not add_default_el:
        return counts, False, False

    has_e = any("E" in counter for counter in counts.values())
    has_l = any("L" in counter for counter in counts.values())
    ordered_sids = sorted(counts)
    if not has_e:
        counts[ordered_sids[0]]["E"] += 1
    if not has_l:
        counts[ordered_sids[-1]]["L"] += 1
    return counts, not has_e, not has_l


def ordered_layer_types(counts_by_sid: Mapping[int, Counter]) -> List[str]:
    types = set()
    for counts in counts_by_sid.values():
        types.update(counts)

    preferred = ["E", "F", "M", "-", "*", "L"]
    ordered = [layer_type for layer_type in preferred if layer_type in types]
    ordered.extend(sorted(types - set(ordered)))
    return ordered


def display_layers_by_sid(
    layers_by_sid: Mapping[int, str], default_e_added: bool, default_l_added: bool
) -> Dict[int, str]:
    display = dict(layers_by_sid)
    if not display:
        return display
    ordered_sids = sorted(display)
    if default_e_added:
        display[ordered_sids[0]] = "E" + display[ordered_sids[0]]
    if default_l_added:
        display[ordered_sids[-1]] = display[ordered_sids[-1]] + "L"
    return display


def gaussian_solve(matrix: List[List[float]], vector: List[float]) -> List[float]:
    n = len(vector)
    augmented = [row[:] + [rhs] for row, rhs in zip(matrix, vector)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise RuntimeError("normal equation matrix is singular")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        pivot_value = augmented[col][col]
        for item in range(col, n + 1):
            augmented[col][item] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for item in range(col, n + 1):
                augmented[row][item] -= factor * augmented[col][item]

    return [augmented[row][n] for row in range(n)]


def fallback_lstsq(a: Sequence[Sequence[float]], y: Sequence[float]) -> Tuple[List[float], int]:
    """Small dependency-free least squares fallback via ridge normal equations."""
    rows = len(a)
    cols = len(a[0]) if rows else 0
    ata = [[0.0 for _ in range(cols)] for _ in range(cols)]
    aty = [0.0 for _ in range(cols)]

    for row, rhs in zip(a, y):
        for i in range(cols):
            aty[i] += row[i] * rhs
            for j in range(cols):
                ata[i][j] += row[i] * row[j]

    trace = sum(ata[i][i] for i in range(cols))
    ridge = max(trace, 1.0) * 1e-10
    for i in range(cols):
        ata[i][i] += ridge

    return gaussian_solve(ata, aty), min(rows, cols)


def lstsq(a: Sequence[Sequence[float]], y: Sequence[float]) -> Tuple[List[float], int]:
    try:
        import numpy as np  # type: ignore

        matrix = np.asarray(a, dtype=float)
        vector = np.asarray(y, dtype=float)
        solution, _, rank, _ = np.linalg.lstsq(matrix, vector, rcond=None)
        return solution.tolist(), int(rank)
    except ImportError:
        return fallback_lstsq(a, y)


def vector_matmul_transpose(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> List[float]:
    if not matrix:
        return []
    cols = len(matrix[0])
    result = [0.0 for _ in range(cols)]
    for row, value in zip(matrix, vector):
        for col in range(cols):
            result[col] += row[col] * value
    return result


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(value * coef for value, coef in zip(row, vector)) for row in matrix]


def solve_columns(
    matrix: Sequence[Sequence[float]], vector: Sequence[float], columns: Sequence[int]
) -> List[float]:
    if not columns:
        return [0.0 for _ in range(len(matrix[0]))]
    submatrix = [[row[col] for col in columns] for row in matrix]
    subsolution, _ = lstsq(submatrix, vector)
    solution = [0.0 for _ in range(len(matrix[0]))]
    for col, value in zip(columns, subsolution):
        solution[col] = value
    return solution


def nnls(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Tuple[List[float], int]:
    """Lawson-Hanson style non-negative least squares for small dense systems."""
    if not matrix:
        return [], 0
    cols = len(matrix[0])
    passive = [False for _ in range(cols)]
    solution = [0.0 for _ in range(cols)]
    tolerance = 1e-10
    max_outer = cols * 8
    max_inner = cols * 8

    for _ in range(max_outer):
        residual = [rhs - pred for rhs, pred in zip(vector, matvec(matrix, solution))]
        gradient = vector_matmul_transpose(matrix, residual)
        candidates = [col for col in range(cols) if not passive[col]]
        if not candidates:
            break
        chosen = max(candidates, key=lambda col: gradient[col])
        if gradient[chosen] <= tolerance:
            break

        passive[chosen] = True
        passive_columns = [col for col, is_passive in enumerate(passive) if is_passive]
        candidate_solution = solve_columns(matrix, vector, passive_columns)

        for _ in range(max_inner):
            bad_columns = [
                col for col in passive_columns if candidate_solution[col] <= tolerance
            ]
            if not bad_columns:
                break

            alpha_candidates = [
                solution[col] / (solution[col] - candidate_solution[col])
                for col in bad_columns
                if solution[col] > candidate_solution[col]
            ]
            if not alpha_candidates:
                for col in bad_columns:
                    passive[col] = False
                    solution[col] = 0.0
                passive_columns = [col for col, is_passive in enumerate(passive) if is_passive]
                candidate_solution = solve_columns(matrix, vector, passive_columns)
                continue

            alpha = min(alpha_candidates)
            solution = [
                current + alpha * (candidate - current)
                for current, candidate in zip(solution, candidate_solution)
            ]

            for col in list(passive_columns):
                if solution[col] <= tolerance:
                    passive[col] = False
                    solution[col] = 0.0
            passive_columns = [col for col, is_passive in enumerate(passive) if is_passive]
            candidate_solution = solve_columns(matrix, vector, passive_columns)

        solution = [max(0.0, value) for value in candidate_solution]

    rank = sum(passive)
    return solution, rank


def solve_op(
    op: str,
    sids: Sequence[int],
    layer_types: Sequence[str],
    counts_by_sid: Mapping[int, Counter],
    profiles_by_sid: Mapping[int, StageProfile],
    allow_negative: bool,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]], int]:
    matrix: List[List[float]] = []
    targets: List[float] = []
    used_sids: List[int] = []

    for sid in sids:
        stage_avg = profiles_by_sid[sid].avg(op)
        if stage_avg is None:
            continue
        matrix.append([float(counts_by_sid[sid].get(layer_type, 0)) for layer_type in layer_types])
        targets.append(stage_avg)
        used_sids.append(sid)

    if not matrix:
        raise RuntimeError(f"no profile data for op={op}")

    if allow_negative:
        solution, rank = lstsq(matrix, targets)
    else:
        solution, rank = nnls(matrix, targets)
    times_by_type = dict(zip(layer_types, solution))

    residuals: Dict[int, Dict[str, float]] = {}
    for sid, row, actual in zip(used_sids, matrix, targets):
        predicted = sum(value * coef for value, coef in zip(row, solution))
        residuals[sid] = {
            "actual_ms": actual,
            "predicted_ms": predicted,
            "error_ms": predicted - actual,
        }
    return times_by_type, residuals, rank


def rms(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def format_float(value: float) -> str:
    return f"{value:9.4f}"


def format_dict_float(value: float) -> str:
    value = 0.0 if abs(value) < 0.00005 else value
    return f"{value:.4f}"


def sanitize_dict_name_part(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return sanitized.upper() or "UNKNOWN"


def infer_dict_name_from_log_path(log_path: Path) -> str:
    parts = log_path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "results" and index >= 2:
            model = sanitize_dict_name_part(parts[index - 2])
            size = sanitize_dict_name_part(parts[index - 1])
            return f"_{model}_{size}"

    if len(parts) >= 3:
        model = sanitize_dict_name_part(parts[-3])
        size = sanitize_dict_name_part(parts[-2])
        return f"_{model}_{size}"
    return "_OCTOPIPE_LAYER_TIMES"


def print_python_dict(
    dict_name: str,
    combined: Mapping[str, Mapping[str, float]],
    layer_types: Sequence[str],
) -> None:
    print()
    print("python dict:")
    print(f"{dict_name} = {{")
    for key, op in (
        ("forward_ms", "f"),
        ("backward_ms", "b"),
        ("weight_ms", "w"),
    ):
        print(f'    "{key}": {{')
        for layer_type in layer_types:
            value = combined[layer_type].get(op, 0.0)
            print(f'        "{layer_type}": {format_dict_float(value)},')
        print("    },")
    print("}")


def solve_layer_times(
    parsed: ParsedLog, add_default_el: bool = True, allow_negative: bool = False
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[int, Dict[str, float]]],
    Dict[str, int],
    Dict[int, Counter],
    List[int],
    List[str],
    bool,
    bool,
]:
    counts_by_sid, default_e_added, default_l_added = layer_counts_by_sid(
        parsed.layers_by_sid, add_default_el=add_default_el
    )
    common_sids = sorted(set(counts_by_sid) & set(parsed.profiles_by_sid))
    if not counts_by_sid:
        raise RuntimeError("no HybridModel layer allocation lines found")
    if not parsed.profiles_by_sid:
        raise RuntimeError("no [octopipe-stage-time] profiler lines found")
    if not common_sids:
        raise RuntimeError("no sid exists in both layer allocation and profiler data")

    layer_types = ordered_layer_types(counts_by_sid)
    available_ops = [
        op
        for op in ("f", "b", "w")
        if any(parsed.profiles_by_sid[sid].avg(op) is not None for sid in common_sids)
    ]
    if not available_ops:
        raise RuntimeError("no f/b/w profile data found in matched stages")

    op_times: Dict[str, Dict[str, float]] = {}
    op_residuals: Dict[str, Dict[int, Dict[str, float]]] = {}
    op_ranks: Dict[str, int] = {}
    for op in available_ops:
        op_times[op], op_residuals[op], op_ranks[op] = solve_op(
            op,
            common_sids,
            layer_types,
            counts_by_sid,
            parsed.profiles_by_sid,
            allow_negative,
        )

    combined: Dict[str, Dict[str, float]] = {}
    for layer_type in layer_types:
        f = op_times.get("f", {}).get(layer_type, 0.0)
        b = op_times.get("b", {}).get(layer_type, 0.0)
        w = op_times.get("w", {}).get(layer_type, 0.0)
        combined[layer_type] = {"f": f, "b": b, "w": w, "fbw": f + b + w}

    return (
        combined,
        op_residuals,
        op_ranks,
        counts_by_sid,
        common_sids,
        layer_types,
        default_e_added,
        default_l_added,
    )


def print_text_report(
    log_path: Path,
    parsed: ParsedLog,
    combined: Mapping[str, Mapping[str, float]],
    residuals: Mapping[str, Mapping[int, Mapping[str, float]]],
    ranks: Mapping[str, int],
    counts_by_sid: Mapping[int, Counter],
    common_sids: Sequence[int],
    layer_types: Sequence[str],
    default_e_added: bool,
    default_l_added: bool,
    show_stage_report: bool,
    allow_negative: bool,
    dict_name: str,
) -> None:
    missing_profile = sorted(set(counts_by_sid) - set(parsed.profiles_by_sid))
    missing_alloc = sorted(set(parsed.profiles_by_sid) - set(counts_by_sid))

    print(f"log: {log_path}")
    print(f"layer source: {parsed.layer_source}")
    print(f"matched stages: {len(common_sids)}")
    print(f"layer types: {' '.join(layer_types)}")
    print(f"profile ops: {' '.join(op for op in ('f', 'b', 'w') if op in residuals)}")
    print(f"system: rows={len(common_sids)} unknowns={len(layer_types)}")
    if default_e_added:
        print("note: E was not present in layer allocation; added to the first sid")
    if default_l_added:
        print("note: L was not present in layer allocation; added to the last sid")
    if missing_profile:
        print(f"warning: allocation without profile sid(s): {missing_profile}")
    if missing_alloc:
        print(f"warning: profile without allocation sid(s): {missing_alloc}")
    if "w" in residuals:
        core_w = [
            combined.get(layer_type, {}).get("w", 0.0)
            for layer_type in layer_types
            if layer_type not in {"E", "L"}
        ]
        endpoint_w_warning_threshold = max(1e-3, 0.1 * max(core_w, default=0.0))
        if default_e_added and combined.get("E", {}).get("w", 0.0) <= endpoint_w_warning_threshold:
            print(
                "warning: inferred E.w is very small; current profiler may not include "
                "embedding weight-gradient work in sid-local w timing"
            )
        if default_l_added and combined.get("L", {}).get("w", 0.0) <= endpoint_w_warning_threshold:
            print(
                "warning: inferred L.w is very small; current profiler may not include "
                "head/output weight-gradient work in sid-local w timing"
            )
    for warning in parsed.warnings:
        print(f"warning: {warning}")

    print()
    print("per-layer average time (ms):")
    print(f"{'type':>6} {'f':>9} {'b':>9} {'w':>9} {'fbw':>9}")
    for layer_type in layer_types:
        item = combined[layer_type]
        print(
            f"{layer_type!r:>6} "
            f"{format_float(item['f'])} "
            f"{format_float(item['b'])} "
            f"{format_float(item['w'])} "
            f"{format_float(item['fbw'])}"
        )
    print_python_dict(dict_name, combined, layer_types)

    print()
    print("fit quality:")
    rank_label = "rank" if allow_negative else "active"
    for op in ("f", "b", "w"):
        if op not in residuals:
            continue
        errors = [item["error_ms"] for item in residuals[op].values()]
        max_abs = max((abs(value) for value in errors), default=0.0)
        print(
            f"  {op}: rows={len(residuals[op])} {rank_label}={ranks[op]}/{len(layer_types)} "
            f"rms_error={rms(errors):.4f}ms max_abs_error={max_abs:.4f}ms"
        )

    if not show_stage_report:
        return

    display_layers = display_layers_by_sid(
        parsed.layers_by_sid, default_e_added=default_e_added, default_l_added=default_l_added
    )
    print()
    print("stage fit details:")
    header = f"{'sid':>4} {'layers':>12} {'op':>2} {'actual':>9} {'pred':>9} {'err':>9}"
    print(header)
    for sid in common_sids:
        layer_summary = display_layers.get(sid, "")
        for op in ("f", "b", "w"):
            if op not in residuals:
                continue
            item = residuals[op].get(sid)
            if item is None:
                continue
            print(
                f"{sid:4d} {layer_summary!r:>12} {op:>2} "
                f"{format_float(item['actual_ms'])} "
                f"{format_float(item['predicted_ms'])} "
                f"{format_float(item['error_ms'])}"
            )


def build_json_report(
    log_path: Path,
    parsed: ParsedLog,
    combined: Mapping[str, Mapping[str, float]],
    residuals: Mapping[str, Mapping[int, Mapping[str, float]]],
    ranks: Mapping[str, int],
    counts_by_sid: Mapping[int, Counter],
    common_sids: Sequence[int],
    layer_types: Sequence[str],
    default_e_added: bool,
    default_l_added: bool,
    allow_negative: bool,
) -> Dict[str, object]:
    return {
        "log": str(log_path),
        "layer_source": parsed.layer_source,
        "matched_sids": list(common_sids),
        "layer_types": list(layer_types),
        "profile_ops": [op for op in ("f", "b", "w") if op in residuals],
        "system": {"rows": len(common_sids), "unknowns": len(layer_types)},
        "default_e_added": default_e_added,
        "default_l_added": default_l_added,
        "layer_times_ms": combined,
        "fit": {
            op: {
                "rows": len(residuals[op]),
                "rank" if allow_negative else "active": ranks[op],
                "rank_denominator" if allow_negative else "active_denominator": len(layer_types),
                "rms_error_ms": rms(item["error_ms"] for item in residuals[op].values()),
                "max_abs_error_ms": max(
                    (abs(item["error_ms"]) for item in residuals[op].values()), default=0.0
                ),
            }
            for op in ("f", "b", "w")
            if op in residuals
        },
        "stage_layer_counts": {
            str(sid): dict(counts_by_sid[sid]) for sid in sorted(counts_by_sid)
        },
        "stage_fit": {
            op: {str(sid): item for sid, item in sorted(residuals[op].items())}
            for op in ("f", "b", "w")
            if op in residuals
        },
        "warnings": parsed.warnings,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-layer-type OctoPipe f/b/w times from HybridModel layer "
            "partition and [octopipe-stage-time] summaries in a log."
        )
    )
    parser.add_argument("log", type=Path, help="training log path")
    parser.add_argument(
        "--pattern",
        help=(
            "explicit compact layer pattern to use when HybridModel allocation lines are absent, "
            "for example F*2M*26. The pattern is split by partition from octopipe_config_yaml."
        ),
    )
    parser.add_argument(
        "--octopipe-config-yaml",
        type=Path,
        help=(
            "OctoPipe config YAML to read partition from. Defaults to the octopipe_config_yaml "
            "path recorded in the log."
        ),
    )
    parser.add_argument(
        "--no-default-el",
        action="store_true",
        help="do not add E to the first stage and L to the last stage when absent",
    )
    parser.add_argument(
        "--stage-report",
        action="store_true",
        help="print per-stage actual/predicted residual details",
    )
    parser.add_argument(
        "--allow-negative",
        action="store_true",
        help="use unconstrained least squares; useful only for diagnostics",
    )
    parser.add_argument(
        "--dict-name",
        default=None,
        help="variable name for the copyable Python dict printed in text output",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    parsed = parse_log(args.log)
    try:
        if args.pattern:
            config_yaml = resolve_config_yaml_path(
                parsed.octopipe_config_yaml, args.log, args.octopipe_config_yaml
            )
            partition = read_partition_from_octopipe_yaml(config_yaml)
            parsed.layers_by_sid = layers_by_sid_from_pattern(args.pattern, partition)
            parsed.layer_source = (
                f"pattern {args.pattern!r} split by partition {partition} from {config_yaml}"
            )
        (
            combined,
            residuals,
            ranks,
            counts_by_sid,
            common_sids,
            layer_types,
            default_e_added,
            default_l_added,
        ) = solve_layer_times(
            parsed,
            add_default_el=not args.no_default_el,
            allow_negative=args.allow_negative,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        report = build_json_report(
            args.log,
            parsed,
            combined,
            residuals,
            ranks,
            counts_by_sid,
            common_sids,
            layer_types,
            default_e_added,
            default_l_added,
            args.allow_negative,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        dict_name = args.dict_name or infer_dict_name_from_log_path(args.log)
        print_text_report(
            args.log,
            parsed,
            combined,
            residuals,
            ranks,
            counts_by_sid,
            common_sids,
            layer_types,
            default_e_added,
            default_l_added,
            args.stage_report,
            args.allow_negative,
            dict_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
