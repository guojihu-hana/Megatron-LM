import os
import tempfile

import yaml

from octopipe.generate_inst import (
    get_octopipe_config,
    get_octopipe_config_from_yaml,
    read_octopipe_yaml,
)


def test_read_octopipe_yaml_accepts_tuple_strings():
    data = {
        "partition": [2, 1],
        "placement": [[0], [1]],
        "scheduling": [
            "(f, 0, 0, 0, 0, 10)",
            "(b, 0, 1, 1, 10, 20)",
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(data, handle)
        yaml_path = handle.name

    try:
        partition, placement, scheduling = read_octopipe_yaml(yaml_path)
        assert partition == [2, 1]
        assert placement == [[0], [1]]
        assert scheduling[0]["type"] == "f"
        assert scheduling[0]["mid"] == 0
        assert scheduling[0]["start_time"] == 0.0
        assert scheduling[1]["type"] == "b"
    finally:
        os.remove(yaml_path)


def test_yaml_config_matches_txt_config_for_nemotron_debug():
    base_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "octopipe",
        "debug_config",
        "nemotron",
    )
    base_dir = os.path.abspath(base_dir)

    from_txt = get_octopipe_config(
        partition_path=os.path.join(base_dir, "partition.txt"),
        placement_path=os.path.join(base_dir, "placement.txt"),
        results_path=os.path.join(base_dir, "result.txt"),
    )

    scheduling = []
    with open(os.path.join(base_dir, "result.txt"), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            op_token, start_str, end_str = [part.strip() for part in line.split(",")]
            parts = op_token.split("_")
            scheduling.append(
                f"({parts[0]}, {parts[1]}, {parts[2]}, {parts[3]}, {start_str}, {end_str})"
            )

    yaml_data = {
        "partition": from_txt["partition"],
        "placement": [[0], [1], [2], [3]],
        "scheduling": scheduling,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(yaml_data, handle)
        yaml_path = handle.name

    try:
        from_yaml = get_octopipe_config_from_yaml(yaml_path)
        assert from_yaml["partition"] == from_txt["partition"]
        assert from_yaml["stage_num"] == from_txt["stage_num"]
        assert from_yaml["max_chunk_num"] == from_txt["max_chunk_num"]
        assert from_yaml["sid->did"] == from_txt["sid->did"]
    finally:
        os.remove(yaml_path)
