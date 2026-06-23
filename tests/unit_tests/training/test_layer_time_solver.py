# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import json
import os
import tempfile

import pytest

from megatron.training.layer_time_solver import (
    _migrate_legacy_layer_time_json,
    build_combined_layer_time_system,
    build_layer_time_system,
    get_stored_equation_sets,
    solve_and_dump_layer_times,
    solve_layer_times_incremental,
    upsert_equation_set,
)


@pytest.mark.internal
class TestLayerTimeSolver:
    def test_migrate_legacy_json(self):
        legacy = {
            "partition": [13, 13, 13, 13],
            "pattern": "M" * 52,
            "equations": {
                "forward": [{"sid": 0, "counts": {"E": 1.0}, "measured_ms": 1.0}],
                "backward": [{"sid": 0, "counts": {"E": 1.0}, "measured_ms": 2.0}],
            },
        }
        migrated = _migrate_legacy_layer_time_json(legacy)
        assert len(migrated["equation_sets"]) == 1
        assert migrated["equation_sets"][0]["partition"] == [13, 13, 13, 13]

    def test_upsert_overwrites_same_partition(self):
        sets = []
        sets = upsert_equation_set(
            sets, [4, 4], "pp_even", "M" * 8,
            [{"sid": 0, "counts": {"E": 1.0}, "measured_ms": 1.0}],
            [{"sid": 0, "counts": {"E": 1.0}, "measured_ms": 2.0}],
        )
        sets = upsert_equation_set(
            sets, [4, 4], "pp_even", "M" * 8,
            [{"sid": 0, "counts": {"E": 1.0}, "measured_ms": 9.0}],
            [{"sid": 0, "counts": {"E": 1.0}, "measured_ms": 8.0}],
        )
        assert len(sets) == 1
        assert sets[0]["equations"]["forward"][0]["measured_ms"] == 9.0

    def test_combined_solve_from_multiple_partitions(self):
        pattern = "M-M*-"
        set_a = {
            "partition": [1, 1, 1, 1, 1],
            "layout_mode": "pipe",
            "pattern": pattern,
            "equations": {
                "forward": [
                    {"sid": 0, "counts": {"E": 1.0, "M": 1.0}, "measured_ms": 11.0},
                    {"sid": 1, "counts": {"-": 1.0}, "measured_ms": 2.0},
                    {"sid": 2, "counts": {"M": 1.0}, "measured_ms": 1.0},
                    {"sid": 3, "counts": {"*": 1.0}, "measured_ms": 3.0},
                    {"sid": 4, "counts": {"-": 1.0, "L": 1.0}, "measured_ms": 9.0},
                ],
                "backward": [],
            },
        }
        combined = build_combined_layer_time_system([set_a], pattern, "forward")
        result = solve_layer_times_incremental(
            combined,
            [11.0, 2.0, 1.0, 3.0, 9.0],
        )
        assert result.layer_times["E"] == pytest.approx(10.0, rel=1e-6)
        assert result.layer_times["M"] == pytest.approx(1.0, rel=1e-6)
        assert result.layer_times["-"] == pytest.approx(2.0, rel=1e-6)
        assert result.layer_times["*"] == pytest.approx(3.0, rel=1e-6)
        assert result.layer_times["L"] == pytest.approx(7.0, rel=1e-6)

    def test_json_persistence_with_equation_sets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "layer_times.json")
            pp_timing_dir = tmpdir
            partition = [13, 13, 13, 13]
            pattern = "M" * 52

            for pp_rank, timing in enumerate([100.0, 39.0, 39.0, 46.0]):
                with open(os.path.join(pp_timing_dir, f"pp_{pp_rank}.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "pp_rank": pp_rank,
                            "vp_stages": {
                                str(pp_rank): {
                                    "forward_avg_ms_exc_1": timing,
                                    "backward_avg_ms_exc_1": timing,
                                }
                            },
                        },
                        f,
                    )

            solve_and_dump_layer_times(
                partition=partition,
                hybrid_layer_pattern=pattern,
                stage_num=4,
                pp_timing_dir=pp_timing_dir,
                output_path=output_path,
                model_name="test",
                layout_mode="pp_even",
            )
            with open(output_path, "r", encoding="utf-8") as f:
                first = json.load(f)
            assert len(first["equation_sets"]) == 1
            assert first["forward_ms"]["E"] == pytest.approx(61.0, rel=1e-6)

            stored = get_stored_equation_sets(output_path, reset=False)
            assert len(stored) == 1

            for pp_rank, timing in enumerate([110.0, 35.0, 35.0, 50.0]):
                with open(os.path.join(pp_timing_dir, f"pp_{pp_rank}.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "pp_rank": pp_rank,
                            "vp_stages": {
                                str(pp_rank): {
                                    "forward_avg_ms_exc_1": timing,
                                    "backward_avg_ms_exc_1": timing,
                                }
                            },
                        },
                        f,
                    )

            solve_and_dump_layer_times(
                partition=partition,
                hybrid_layer_pattern=pattern,
                stage_num=4,
                pp_timing_dir=pp_timing_dir,
                output_path=output_path,
                model_name="test",
                layout_mode="pp_even",
            )
            with open(output_path, "r", encoding="utf-8") as f:
                second = json.load(f)
            assert len(second["equation_sets"]) == 1
            assert second["equation_sets"][0]["equations"]["forward"][0]["measured_ms"] == 110.0

    def test_incremental_solve_e_and_l_via_row_differences(self):
        partition = [13, 13, 13, 13]
        pattern = (
            "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
        )
        system = build_layer_time_system(partition, pattern)
        measurements = [29.150489002767234, 27.587347322040134, 27.718353996373185, 40.759513142133]
        result = solve_layer_times_incremental(system, measurements)
        assert result.layer_times["E"] == pytest.approx(1.563141680727099, rel=1e-6)
        assert result.layer_times["L"] == pytest.approx(13.172165820092866, rel=1e-6)
