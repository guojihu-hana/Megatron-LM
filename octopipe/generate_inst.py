import os
import ast
import math
from typing import List, Union, Dict
from collections import defaultdict, deque

def is_send(op):
    return op["op"] == "send"

def is_recv(op):
    return op["op"] == "recv"

def find_matching_send(workloads, recv_op):
    """
    Given a send op, find its matching recv op in the receiver device workload.
    """
    mid = recv_op["mid"]
    wtype = recv_op["type"]

    for op in workloads:
        if (
            op["op"] == "send"
            and op["mid"] == mid
            and op["type"] == wtype
            and op["sender_sid"] == recv_op["sender_sid"]
            and op["recver_sid"] == recv_op["recver_sid"]
        ):
            return op
    raise RuntimeError("matching recv not found")

def find_matching_recv(workloads, send_op):
    """
    Given a send op, find its matching recv op in the receiver device workload.
    """
    mid = send_op["mid"]
    wtype = send_op["type"]

    for op in workloads:
        if (
            op["op"] == "recv"
            and op["mid"] == mid
            and op["type"] == wtype
            and op["sender_sid"] == send_op["sender_sid"]
            and op["recver_sid"] == send_op["recver_sid"]
        ):
            return op
    raise RuntimeError("matching recv not found")

def find_earliest_comp(workloads, send_op):

    for op in workloads:
        if (
            op["op"] == "comp"
            and op["start_time"] >= send_op["start_time"]
        ):
            return op
    raise RuntimeError("earliest comp not found")

def legal_pair(op1, op2):
    if op1.get("sender_sid") == op2.get("sender_sid"):
        if op1.get("recver_sid") == op2.get("recver_sid"):
            if op1.get("mid") == op2.get("mid"):
                if op1.get("op") == 'send' and op2.get("op") == 'recv' or op1.get("op") == 'recv' and op2.get("op") == 'send':
                    return True
    return False


def get_pair_comm_ops(workloads, dev_a, dev_b, device_stage_mappings):
    """
    Return all send and recv ops whose sender and receiver are dev_a and dev_b.
    """
    sids_on_dev_a = device_stage_mappings[dev_a]
    sids_on_dev_b = device_stage_mappings[dev_b]
    comm_ops_a = []
    comm_ops_b = []

    for workload in workloads[dev_a]:
        if workload.get("sender_sid") in sids_on_dev_b and workload.get("recver_sid") in sids_on_dev_a:
            comm_ops_a.append(workload)
        elif workload.get("recver_sid") in sids_on_dev_b and workload.get("sender_sid") in sids_on_dev_a:
            comm_ops_a.append(workload)
    for workload in workloads[dev_b]:
        if workload.get("sender_sid") in sids_on_dev_b and workload.get("recver_sid") in sids_on_dev_a:
            comm_ops_b.append(workload)
        elif workload.get("recver_sid") in sids_on_dev_b and workload.get("sender_sid") in sids_on_dev_a:
            comm_ops_b.append(workload)

    return comm_ops_a, comm_ops_b

def reorder_pair_minimally(res, dev_a, dev_b, debug=True):
    """
    Fix deadlock between dev_a and dev_b by insertion-only reordering.
    """
    solved_dead_lock = 0
    exist_dead_lock = True
    workloads = res["workloads"]
    device_stage_mappings = res["did->sid"]
    comm_ops_a, comm_ops_b = get_pair_comm_ops(workloads, dev_a=dev_a, dev_b=dev_b, device_stage_mappings=device_stage_mappings)
    while exist_dead_lock:
        exist_dead_lock = False
        assert len(comm_ops_a) == len(comm_ops_b)
        # print_ops(res["workloads"], s_sid=[6,7], r_sid=[6,7])
        for idx in range(len(comm_ops_a)):
            op_a = comm_ops_a[idx]
            op_b = comm_ops_b[idx]
            if legal_pair(op_a, op_b):
                continue
            if is_send(op_a):
                recv = find_matching_recv(workloads[dev_b], op_a)
                w = workloads[dev_b]
                insert_pos = w.index(op_b)
                print(op_a, op_b,'---------------')
                print_ops(res['workloads'], s_sid=[1,2], r_sid=[1,2], num=65)
                w.remove(recv)
                recv['start_time'] = op_b['start_time']
                w.insert(insert_pos, recv)
                exist_dead_lock = True
                solved_dead_lock += 1
                print_ops(res['workloads'], s_sid=[1,2], r_sid=[1,2], num=65)
                if debug:
                    print(f"solve {idx}, {op_a}, {op_b}, 1")
                break
            elif is_send(op_b):
                recv = find_matching_recv(workloads[dev_a], op_b)
                w = workloads[dev_a]
                insert_pos = w.index(op_a)
                w.remove(recv)
                recv['start_time'] = op_a['start_time']
                w.insert(insert_pos, recv)
                exist_dead_lock = True
                solved_dead_lock += 1
                if debug:
                    print(f"solve {op_a}, {op_b}, 2")
                break
            elif is_recv(op_a) and is_recv(op_b):
                # case: recv + recv
                print("Potential deadlock unsolved.")
    return solved_dead_lock

def reorder_comm_ops(res, dev_a, dev_b):
    """
    Reorder communication ops for overlapping.
    """
    num_reorders = 0
    exist_dead_lock = True
    while exist_dead_lock:
        exist_dead_lock = False
        workloads = res["workloads"]
        comm_ops_a, comm_ops_b = get_pair_comm_ops(
            workloads, dev_a=dev_a, dev_b=dev_b
        )
        assert len(comm_ops_a) == len(comm_ops_b)
        # print_ops(res["workloads"], s_sid=[0,1], r_sid=[0,1])
        for idx in range(len(comm_ops_a)):
            op_a = comm_ops_a[idx]
            op_b = comm_ops_b[idx]
            if legal_pair(op_a, op_b):
                continue
            # case a: send recv
            if is_send(op_a) and is_recv(op_b):
                recv = find_matching_recv(workloads[dev_b], op_a)
                w = workloads[dev_b]
                w.remove(recv)
                insert_pos = w.index(op_b)
                w.insert(insert_pos, recv)
                exist_dead_lock = True
                num_reorders += 1
                break
            # case b: recv send
            elif is_recv(op_a) and is_send(op_b):
                recv = find_matching_recv(workloads[dev_a], op_b)
                w = workloads[dev_a]
                w.remove(recv)
                insert_pos = w.index(op_a)
                w.insert(insert_pos, recv)
                exist_dead_lock = True
                num_reorders += 1
                break
            # case c: send send
            elif is_send(op_a) and is_send(op_b):
                recv_a = find_matching_recv(workloads[dev_b], op_a)
                recv_b = find_matching_recv(workloads[dev_a], op_b)

                # move recv_a before op_b
                w_b = workloads[dev_b]
                w_b.remove(recv_a)
                pos_b = w_b.index(op_b)
                w_b.insert(pos_b, recv_a)

                # move recv_b after op_a
                w_a = workloads[dev_a]
                w_a.remove(recv_b)
                pos_a = w_a.index(op_a)
                w_a.insert(pos_a + 1, recv_b)

                exist_dead_lock = True
                num_reorders += 1
                break
            # case d: recv recv
            else:
                # explicitly do nothing
                pass
    return num_reorders

def reorder_send_recv_pairs(res):
    devices = list(res["workloads"].keys())
    num_potential_dead_locks = 1
    while num_potential_dead_locks > 0:
        num_potential_dead_locks = 0
        for i in range(len(devices)):
            for j in range(len(devices)):
                if i == j:
                    continue
                num_potential_dead_locks += reorder_pair_minimally(res, devices[i], devices[j])
        print(f"Number of solved dead locks: {num_potential_dead_locks}")


def delay_send(res, sender_did):
    sender_workloads = res["workloads"][sender_did]
    stage_device_mappings = res["sid->did"]
    num_delay = 0
    i = 0
    while i < len(sender_workloads):
        send = sender_workloads[i]
        if send["op"] != "send":
            i += 1
            continue

        t_send = send["start_time"]

        recver_sid = send["recver_sid"]
        recver_did = stage_device_mappings[recver_sid]
        recver_workloads = res["workloads"][recver_did]

        # 找到 send 落入的 receiver comp
        overlapped_comp = None
        for rw in recver_workloads:
            if rw["op"] == "comp" and rw["start_time"] < t_send < rw["end_time"]:
                overlapped_comp = rw
                break

        if overlapped_comp is None:
            i += 1
            continue

        # recv 的开始时间
        recv_start = None
        for rw in recver_workloads:
            if legal_pair(send, rw):
                recv_start = rw["start_time"]
                break

        if recv_start is None:
            i += 1
            continue

        # 收集 sender 上候选时间点
        candidates = []
        for sw in sender_workloads:
            if sw["op"] == "comp":
                for t in (sw["start_time"], sw["end_time"]):
                    if t_send <= t <= recv_start:
                        candidates.append(t)

        if not candidates:
            i += 1
            continue

        # 与 receiver comp 最近的时间点
        best_t = None
        min_dist = float("inf")
        for t in candidates:
            for ref in (overlapped_comp["start_time"], overlapped_comp["end_time"]):
                dist = abs(t - ref)
                if dist < min_dist:
                    min_dist = dist
                    best_t = t

        if best_t is None:
            i += 1
            continue

        # 执行 insertion-only 重排
        sender_workloads.pop(i)

        insert_idx = 0
        while insert_idx < len(sender_workloads):
            if sender_workloads[insert_idx]["start_time"] >= best_t:
                break
            insert_idx += 1

        sender_workloads.insert(insert_idx, send)

        # 更新时间
        send["start_time"] = best_t
        num_delay += 1
        # 重新从当前索引检查
        i = insert_idx
    return num_delay

def delay_send_for_overlap(res):
    devices = list(res["workloads"].keys())
    num_delayed_send = 0
    for i in range(len(devices)):
        num_delayed_send += delay_send(res, i)
    print(f"Number of delayed sends: {num_delayed_send}")
    return num_delayed_send

def comp_op_idx(workloads, s_idx, e_idx):
    idxs = []
    for idx, workload in enumerate(workloads):
        if s_idx <= idx <= e_idx:
            if workload['op'] == 'comp':
                idxs.append(idx)
    return idxs

def is_dependent_with_next_comp(recv, workloads):
    if recv['op'] in ('comp', 'send'):
        return False
    assert recv in workloads
    for workload in workloads[workloads.index(recv):]:
        if workload['op'] == 'comp':
            if workload['mid'] == recv['mid']:
                if workload['type'] == recv['type']:
                    if workload['sid'] == recv['recver_sid']:
                        return True
    return False

def find_earliest_insert_recv_pos(workloads, send_time):
    for workload in workloads:
        if workload['start_time'] >= send_time:
            return workloads.index(workload)
    return -1

def is_non_decreasing(workloads):
    prev = None
    for w in workloads:
        t = w["start_time"]
        if prev is not None and t < prev:
            return False
        prev = t
    return True

def advance_recv(res, recver_did):
    """
    Advance recv operations to earlier computation boundaries
    while respecting local send ordering constraints.
    """
    workloads = res["workloads"][recver_did]
    num_reorderings = 0
    i = 0
    # print_ops(res["workloads"], skip_comp=False)
    # input()
    while i < len(workloads):
        recv = workloads[i]
        if recv["op"] != "recv" or not is_dependent_with_next_comp(recv, workloads):
            i += 1
            continue

        recv_start = recv["start_time"]
        sender_sid = recv["sender_sid"]
        stage_device_mappings = res["sid->did"]
        sender_did = stage_device_mappings[sender_sid]
        sw = res["workloads"][sender_did]

        # 在 receiver 本地，找到 recv 之前最近的、发往该 sender 的 send
        send = find_matching_send(workloads=sw, recv_op=recv)
        local_send_start = send["start_time"]
        for j in range(i - 1, -1, -1):
            w = workloads[j]
            if w['op'] == 'comp':
                continue
            if w["recver_sid"] == sender_sid or w["sender_sid"] == sender_sid:
                local_send_start = max(local_send_start, w["start_time"])
                break
        
        # 收集候选 comp 时间点
        candidates = []
        for w in workloads:
            if w["op"] == "comp":
                for t in (w["start_time"], w["end_time"]):
                    if local_send_start <= t < recv_start:
                        candidates.append(t)
        
        # if recv['mid'] == 1 and recv['sender_sid'] == 7 and recv["recver_sid"] == 6:
        #     print(recv, send, local_send_start)
        #     print(candidates)
        #     print_ops(res["workloads"], skip_comp=False)
        #     input()
        if not candidates:
            i += 1
            continue

        # 选择最接近 local_send_start 的时间点
        best_t = candidates[-1]

        # insertion only 重排
        workloads.pop(i)

        insert_idx = 0
        while insert_idx < len(workloads):
            if workloads[insert_idx]["start_time"] >= best_t:
                break
            insert_idx += 1

        # print("insert idx:", insert_idx)
        # input()
        workloads.insert(insert_idx, recv)

        # 更新时间
        delta = best_t - recv["start_time"]
        recv["start_time"] += delta

        num_reorderings += 1
        # 从新位置继续扫描
        i = insert_idx + 1
        # print_ops(res["workloads"], skip_comp=False)
        # input()
        # if recv['mid'] == 1 and recv['sender_sid'] == 7 and recv["recver_sid"] == 6:
        #     print_ops(res["workloads"], skip_comp=False)
        #     input()
    return num_reorderings

def advance_recv_for_overlap(res):
    devices = list(res["workloads"].keys())
    num_reorderings = 0
    for i in reversed(range(len(devices))):
        num_reorderings += advance_recv(res, i)
    print(f"Number of advanced recvs: {num_reorderings}")

def read_partition_from_file(file_path: str) -> List[int]:
    with open(file_path, "r") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Empty partition file: {file_path}")

    try:
        data = ast.literal_eval(content)
    except Exception as e:
        raise ValueError(f"Invalid partition format: {file_path}") from e

    if not isinstance(data, list):
        raise ValueError("Partition must be a list")

    return [int(x) for x in data]

def read_placement_from_file(file_path: str) -> List[List[int]]:
    with open(file_path, "r") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Empty placement file: {file_path}")

    try:
        data = ast.literal_eval(content)
    except Exception as e:
        raise ValueError(f"Invalid placement format: {file_path}") from e

    if not isinstance(data, list):
        raise ValueError("Placement must be a 2D list")

    placement: List[List[int]] = []
    for i, dev_stages in enumerate(data):
        if not isinstance(dev_stages, list):
            raise ValueError(f"placement[{i}] must be a list")

        placement.append([int(s) for s in dev_stages])

    return placement

def read_scheduling_from_file(file_path: str) -> List[Dict]:
    scheduling = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            try:
                op_token, start_str, end_str = [x.strip() for x in line.split(",")]
            except ValueError as e:
                raise ValueError(f"Invalid scheduling line: {line}") from e

            parts = op_token.split("_")
            if len(parts) != 3:
                raise ValueError(f"Invalid op format: {op_token}")

            op_type = parts[0]
            mid = int(parts[1])
            sid = int(parts[2])

            scheduling.append(
                {
                    "op": 'comp',
                    "type": op_type,
                    "mid": mid,
                    "sid": sid,
                    "start_time": float(start_str),
                    "end_time": float(end_str),
                }
            )

    return scheduling

def build_stage_device_mappings(
    partition: List[int],
    placement: List[List[int]],
) -> tuple[Dict[int, int], Dict[int, List[int]]]:
    """
    Returns
    - stage_device_mapping: sid -> device_id
    - device_stage_mapping: device_id -> list of sid
    """

    num_stages = len(partition)

    stage_device_mapping: Dict[int, int] = {}
    device_stage_mapping: Dict[int, List[int]] = {}

    for device_id, stages in enumerate(placement):
        if not isinstance(stages, list):
            raise ValueError(f"placement[{device_id}] must be a list")

        device_stage_mapping[device_id] = []

        for sid in stages:
            if not isinstance(sid, int):
                raise ValueError(f"Invalid stage id {sid} on device {device_id}")

            if sid < 0 or sid >= num_stages:
                raise ValueError(f"Stage id {sid} out of range")

            if sid in stage_device_mapping:
                raise ValueError(f"Stage {sid} assigned to multiple devices")

            stage_device_mapping[sid] = device_id
            device_stage_mapping[device_id].append(sid)

    # Sanity check: every stage must be placed exactly once
    if len(stage_device_mapping) != num_stages:
        missing = set(range(num_stages)) - set(stage_device_mapping.keys())
        raise ValueError(f"Unplaced stages: {sorted(missing)}")

    return stage_device_mapping, device_stage_mapping

def build_stage_chunk_mappings(
    device_stage_mapping: Dict[int, List[int]],
) -> Dict[int, int]:
    """
    Returns
    - stage_device_mapping: sid -> device_id
    - device_stage_mapping: device_id -> list of sid
    """
    stage_chunk_mapping = {}
    for did, stages in device_stage_mapping.items():
        for idx, sid in enumerate(stages):
            stage_chunk_mapping[sid] = idx
    return  stage_chunk_mapping

def build_workload_exe_order(
    stage_device_mapping: Dict[int, int],
    scheduling: List[Dict],
) -> Dict[int, List[Dict]]:
    """
    workload_exe_order[device_id] is a list of workloads
    sorted by start_time in ascending order
    """

    workload_exe_order: Dict[int, List[Dict]] = {}

    for entry in scheduling:
        sid = entry["sid"]
        if sid not in stage_device_mapping:
            raise ValueError(f"Stage {sid} not found in stage_device_mapping")

        device_id = stage_device_mapping[sid]

        if device_id not in workload_exe_order:
            workload_exe_order[device_id] = []

        workload_exe_order[device_id].append(entry)

    # sort workloads on each device by start_time
    for device_id, workloads in workload_exe_order.items():
        workloads.sort(key=lambda x: x["start_time"])

    return workload_exe_order

def insert_comm_ops(
    workload_exe_order: Dict[int, List[Dict]],
    stage_device_mapping: Dict[int, int],
) -> Dict[int, List[Dict]]:
    """
    Returns workload_comp_comm_order
    Each device contains compute, send, and receive ops
    sorted by start_time
    """

    workload_comp_comm_order: Dict[int, List[Dict]] = {}

    # initialize with existing compute workloads
    for device_id, workloads in workload_exe_order.items():
        workload_comp_comm_order[device_id] = list(workloads)

    for device_id, workloads in workload_exe_order.items():
        
        new_list: List[Dict] = []

        for wid, w in enumerate(workloads):
            op = w["op"]
            op_type = w["type"]
            mid = w["mid"]
            sid = w["sid"]
            recv_time = w["start_time"]
            send_time = w["end_time"]
            
            if op != 'comp':
                continue

            if op_type == "f":
                sender_sid = sid - 1
                recver_sid = sid + 1
            elif op_type == "b":
                sender_sid = sid + 1
                recver_sid = sid - 1
            elif op_type == "w":
                # NOTE: Weight gradients do not need comm
                new_list.append(w)
                continue
            elif op_type == "r":
                # NOTE: Support recomputation
                continue
            else:
                raise ValueError(f"Unknown workload type: {op_type}")

            # recv_op: before compute, on current device
            if sender_sid in stage_device_mapping:
                recv_op = {
                    "op": "recv",
                    "type": op_type,
                    "mid": mid,
                    "sender_sid": sender_sid,
                    "recver_sid": sid,
                    "start_time": recv_time,
                }
                new_list.append(recv_op)

            # compute workload itself
            new_list.append(w)

            # send_op: after compute, on current device
            if recver_sid in stage_device_mapping:
                send_op = {
                    "op": "send",
                    "type": op_type,
                    "mid": mid,
                    "sender_sid": sid,
                    "recver_sid": recver_sid,
                    "start_time": send_time,
                }
                new_list.append(send_op)

        workload_comp_comm_order[device_id] = new_list

    return workload_comp_comm_order

def generate_pipeline_layout(partition, placement, E=0, L=-1):
    """
    Generate Megatron-LM pipeline layout string.

    Args:
        partition (list[int]):
            partition[i] is the number of transformer layers in stage i.
        placement (list[list[int]]):
            placement[d] is the list of stage ids placed on device d.
            Currently used only for validation and stage count.
        E (int):
            stage id to place embedding layer.
        L (int):
            stage id to place loss layer. Supports negative index.

    Returns:
        str: pipeline layout string.
    """
    num_stages = len(partition)

    # normalize negative index
    if E < 0:
        E += num_stages
    if L < 0:
        L += num_stages

    if not (0 <= E < num_stages):
        raise ValueError(f"Invalid E stage id {E}")
    if not (0 <= L < num_stages):
        raise ValueError(f"Invalid L stage id {L}")

    layout_stages = []

    for sid in range(num_stages):
        elems = []

        # embedding
        if sid == E:
            elems.append("E")

        # transformer layers
        num_layers = partition[sid]
        if num_layers > 0:
            elems.append(f"t*{num_layers}")

        # loss
        if sid == L:
            elems.append("L")

        if not elems:
            raise ValueError(f"Stage {sid} has no components")

        layout_stages.append(",".join(elems))

    return "|".join(layout_stages)

def find_dependent_comp_on_recver(comp, workloads):
    if comp['op'] != 'comp':
        return None, None
    
    op_type = comp["type"]
    if op_type not in ('f', 'b'):
        return None, None
    
    mid = comp["mid"]
    sid = comp["sid"]
    offset = 1 if op_type == 'f' else -1
    for wid, workload in enumerate(workloads):
        if op_type == workload['type']:
            if workload['mid'] == mid:
                if workload['sid'] == sid + offset:
                    return wid, workload

def find_time_points(workloads, stime, etime):
    time_points = set()
    for wid, w in enumerate(workloads):
        sta_time = w["start_time"]
        end_time = w["end_time"]
        if stime <= sta_time <= etime:
            time_points.add((wid, sta_time))
        if stime <= end_time <= etime:
            time_points.add((wid+1, end_time))
    time_points = list(time_points)
    time_points.sort(key=lambda x:x[1])
    return time_points

def find_insert_time_pairs(send_time_points, recv_time_points):
    insert_time_pairs = []
    for idx, (send_wid, send_time) in enumerate(send_time_points):
        for p_idx, (recv_wid, recv_time) in enumerate(recv_time_points):
            if recv_time - send_time >= 0: # NOTE: avoid recv earlier than send, maintain scheduling results
                insert_time_pairs.append((idx, p_idx, recv_time - send_time))
    insert_time_pairs.sort(key=lambda x: x[2])
    return insert_time_pairs

def overlap_aware_comm_insert(
    workload_exe_order: Dict[int, List[Dict]],
    stage_device_mapping: Dict[int, int],
) -> Dict[int, List[Dict]]:

    comm_ops = []
    for did, workloads in workload_exe_order.items():
        comm_ops.append([])
        for workload in workloads:
            comm_ops[did].append([])
        comm_ops[did].append([])

    for did, workloads in workload_exe_order.items():
    # for did in reversed(list(workload_exe_order.keys())):
        workloads = workload_exe_order[did]
        for w in workloads:
            if w['op'] != 'comp':
                continue

            op = w["op"]
            op_type = w["type"]
            mid = w["mid"]
            sid = w["sid"]
            sta_time = w["start_time"]
            end_time = w["end_time"]
            if op_type == "f":
                recver_sid = sid + 1
            elif op_type == "b":
                recver_sid = sid - 1
            elif op_type == "w":
                # NOTE: Does not need send/recv
                continue
            elif op_type == "r":
                # NOTE: Support recomputation
                continue
            else:
                raise ValueError(f"Unknown workload type: {op_type}")
            
            if recver_sid not in stage_device_mapping:
                continue
            recver_did = stage_device_mapping[recver_sid]
            recver_workloads = workload_exe_order[recver_did]

            wid, recver_w = find_dependent_comp_on_recver(w, recver_workloads)

            if wid is not None and recver_w is not None:
                send_start_min_time = end_time
                recv_start_max_time = recver_w['start_time']
                send_start_time_points = find_time_points(workloads, send_start_min_time, recv_start_max_time)
                recv_start_time_points = find_time_points(recver_workloads, send_start_min_time, recv_start_max_time)
                pairs = find_insert_time_pairs(send_time_points=send_start_time_points, recv_time_points=recv_start_time_points)
                # print(w, send_start_time_points, recv_start_time_points, pairs)
                assert len(pairs) > 0, f"No comm pairs found for {w}, recver_w:{recver_w}!"
                (send_idx, send_time), (recv_idx, recv_time) = send_start_time_points[pairs[0][0]], recv_start_time_points[pairs[0][1]]
                send_op = {
                    "op": "send",
                    "type": op_type,
                    "mid": mid,
                    "sender_sid": sid,
                    "recver_sid": recver_sid,
                    "start_time": send_time,
                }
                recv_op = {
                    "op": "recv",
                    "type": op_type,
                    "mid": mid,
                    "sender_sid": sid,
                    "recver_sid": recver_sid,
                    "start_time": recv_time,
                }
                comm_ops[did][send_idx].append(send_op)
                comm_ops[recver_did][recv_idx].append(recv_op)

    # print(comm_ops)
    # workload_comp_comm_order: Dict[int, List[Dict]] = {}
    # # initialize with existing compute workloads
    # for device_id, workloads in workload_exe_order.items():
    #     workload_comp_comm_order[device_id] = []
    #     workload_comp_comm_order[device_id].extend(comm_ops[device_id][0])
    #     for idx, workload in enumerate(workloads):
    #         workload_comp_comm_order[device_id].append(workload)
    #         workload_comp_comm_order[device_id].extend(comm_ops[device_id][idx+1])
    workload_comp_comm_order: Dict[int, List[Dict]] = {}
    # initialize with existing compute workloads
    for device_id, workloads in workload_exe_order.items():
        workload_comp_comm_order[device_id] = []
        workload_comp_comm_order[device_id].extend(sorted(comm_ops[device_id][0], key=lambda x : x['start_time']))
        for idx, workload in enumerate(workloads):
            workload_comp_comm_order[device_id].append(workload)
            workload_comp_comm_order[device_id].extend(sorted(comm_ops[device_id][idx+1], key=lambda x : x['start_time']))
    # print_ops(workload_comp_comm_order=workload_comp_comm_order, skip_comp=False)
    return workload_comp_comm_order
            

def reorder_cross_comm_pairs(res):
    pass

def get_octopipe_config(partition_path, placement_path, results_path):
    partition = read_partition_from_file(partition_path)
    placement = read_placement_from_file(placement_path)
    scheduling = read_scheduling_from_file(results_path)
    layer_idx_offset = [sum(partition[0:i]) for i in range(len(partition)+1)]  # Assuming layer index offset is based on the first stage

    assert len(partition) == sum([len(stages) for stages in placement]), "Total number of stages in placement must match length of partition"
    assert set().union(*placement) == set(range(len(partition))), "All stages must be placed exactly once in placement"

    stage_device_mapping, device_stage_mapping = build_stage_device_mappings(
        partition,
        placement,
    )

    stage_chunk_mapping = build_stage_chunk_mappings(
        device_stage_mapping
    )

    workload_exe_order = build_workload_exe_order(
        stage_device_mapping,
        scheduling,
    )

    # workload_comp_comm_order = insert_comm_ops(workload_exe_order, stage_device_mapping)
    workload_comp_comm_order = overlap_aware_comm_insert(workload_exe_order, stage_device_mapping)
    
    layout = generate_pipeline_layout(partition=partition, placement=placement)

    stage_num = len(partition)
    max_chunk_num = max([len(sids) for sids in device_stage_mapping.values()])
    padded_device_stage_mapping = {
        key: value + [-1] * (max_chunk_num - len(value))
        for key, value in device_stage_mapping.items()
    }

    res = {
        "sid->did": stage_device_mapping,
        "did->sid": device_stage_mapping,
        "sid->cid": stage_chunk_mapping,
        "comp_workloads": workload_exe_order,
        "workloads": workload_comp_comm_order,
        "layout": layout,
        "layer_idx_offset": layer_idx_offset,
        "stage_num": stage_num,
        "max_chunk_num": max_chunk_num,
        "did->padded_sids": padded_device_stage_mapping,
        "partition": partition,
    }

    return res

def print_ops(workload_comp_comm_order, skip_comp=True, s_sid:list=[], r_sid:list=[], num:int=20, s:int=0):
    # 首先找到最长的输出字符串
    max_width = 0
    for did in workload_comp_comm_order.keys():
        workloads = workload_comp_comm_order[did]
        for workload in workloads:
            if skip_comp and workload['op'] == 'comp':
                continue
            
            mid = workload['mid']
            wtype = str(workload['type']).upper()
            if workload['op'] == 'comp':
                length = len(f"C{wtype}{mid}")
            else:
                sender_sid = workload['sender_sid']
                recver_sid = workload['recver_sid']
                length = len(f"{'S' if workload['op'] == 'send' else 'R'}{wtype}{mid}{sender_sid}{recver_sid}")
            
            max_width = max(max_width, length)
    
    # 宽度加1，使输出更清晰
    width = max_width + 1
    
    COLOR_RESET = '\033[0m'
    COLOR_COMP = '\033[92m'  # 亮绿色
    COLOR_SEND = '\033[93m'  # 亮黄色
    COLOR_RECV = '\033[94m'  # 亮蓝色
    COLOR_BOLD = '\033[1m'   # 粗体

    # 打印输出
    for did in workload_comp_comm_order.keys():
        workloads = workload_comp_comm_order[did]
        prefix = " "*(did * width + 3)
        if not skip_comp:
            print(prefix, end="")
        for idx, workload in enumerate(workloads[s:s+num]):
            if skip_comp and workload['op'] == 'comp':
                continue
            
            mid = workload['mid']
            wtype = str(workload['type']).upper()
            if workload['op'] == 'comp':
                op = 'C'
                sid = workload['sid']
                color = COLOR_COMP
                output = f"{wtype}{mid}{sid}"
                print(f"{color}{idx+s} {output:<{width}}{COLOR_RESET}", end="")
            else:
                if s_sid:
                    if workload['sender_sid'] not in s_sid and workload['sender_sid'] not in r_sid:
                        continue
                if r_sid:
                    if workload['recver_sid'] not in s_sid and workload['recver_sid'] not in r_sid:
                        continue
                if workload['op'] == 'send':
                    op = 'S'
                    color = COLOR_SEND
                else:
                    op = 'R'
                    color = COLOR_RECV
                sender_sid = workload['sender_sid']
                recver_sid = workload['recver_sid']
                output = f"{op}{wtype}{mid}{sender_sid}{recver_sid}"
                print(f"{color}{idx+s} {output:<{width}}{COLOR_RESET}", end="")
        print()

def recv_forward():
    pass
def forward_step():
    pass
def send_forward():
    pass
def recv_backward():
    pass
def backward_step():
    pass
def send_backward():
    pass
def send_forward_recv_backward():
    pass
def send_backward_recv_forward():
    pass

def Pipeline_Schedule_of_1F1B(nmb_warmup, nmb_remaining):
    # Run warmup forward passes.
    for i in range(nmb_warmup):
        recv_forward() # 通信
        forward_step() # 计算
        send_forward() # 通信

    # Run 1F1B in steady state.
    for i in range(nmb_remaining):
        forward_step() # 计算
        send_forward_recv_backward() # 通信
        backward_step() # 计算
        send_backward_recv_forward() # 通信

    # Run cooldown backward passes.
    for i in range(nmb_warmup):
        recv_backward() # 通信
        backward_step() # 计算
        send_backward() # 通信
    
if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEBUG_CONFIG_DIR = os.path.join(BASE_DIR, "debug_config/asymmetric_multi_chunk")
    # DEBUG_CONFIG_DIR = os.path.join(BASE_DIR, "debug_config/single_chunk")

    partition_path = os.path.join(DEBUG_CONFIG_DIR, "partition.txt")
    placement_path = os.path.join(DEBUG_CONFIG_DIR, "placement.txt")
    results_path = os.path.join(DEBUG_CONFIG_DIR, "result.txt")
    # Example:
    res = get_octopipe_config(partition_path=partition_path,placement_path=placement_path,results_path=results_path)
    print(res["sid->did"])
    print(res["did->sid"])
    print(res["sid->cid"])
    print(res["layer_idx_offset"])
    print(res["max_chunk_num"])
    print(res["did->padded_sids"])
    # print(res["workloads"][0][:10])
    # print_ops(res['workloads'], skip_comp=False,num=20, s=70)
    # print_ops(res['workloads'], s_sid=[6, 14, 22, 30, 7, 15, 23, 31], r_sid=[6, 14, 22, 30, 7, 15, 23, 31], num=90)