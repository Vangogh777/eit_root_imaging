#!/usr/bin/env python3
"""修复 training_records/index.json 中的僵尸 running 状态。

- 检查 ps aux 中是否有对应进程存活
- 检查 meta.json 是否存在及状态
- 已死亡且无有效最终 epoch 的标记为 failed
- 已死亡但有完整记录的标记为 completed
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

RECORDS_DIR = Path(__file__).parent / "training_records"
INDEX_PATH = RECORDS_DIR / "index.json"

# 获取当前所有python训练进程的命令行
def get_running_train_cmds():
    """返回当前运行中的 train 进程的 run_id 集合"""
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
    except subprocess.TimeoutExpired:
        return set()

    running = set()
    for line in result.stdout.split("\n"):
        if "python" not in line or "train" not in line:
            continue
        if "grep" in line:
            continue
        # 尝试从命令行提取 run_id
        for part in line.split():
            for prefix in ["202606", "2026062"]:  # 2026年的 run_id
                if prefix in part and "checkpoints/" in part:
                    # 从路径中提取 run_id
                    import re
                    m = re.search(r"(202606\d{2}_\d{6}_[a-z0-9_]+)", part)
                    if m:
                        running.add(m.group(1))
    return running


def check_record_dir(run_id):
    """检查训练记录的 meta.json"""
    meta_path = RECORDS_DIR / run_id / "meta.json"
    epochs_path = RECORDS_DIR / run_id / "epochs.jsonl"
    events_path = RECORDS_DIR / run_id / "events.jsonl"

    if not meta_path.exists():
        return None, None

    with open(meta_path) as f:
        meta = json.load(f)

    # 检查最后一条 epoch 记录
    last_epoch = None
    last_event_time = None
    if epochs_path.exists():
        with open(epochs_path) as f:
            lines = f.readlines()
            if lines:
                last = json.loads(lines[-1].strip())
                last_epoch = last.get("epoch")

    if events_path.exists():
        with open(events_path) as f:
            lines = f.readlines()
            if lines:
                last = json.loads(lines[-1].strip())
                last_event_time = last.get("time")

    return meta, last_epoch


def main():
    with open(INDEX_PATH) as f:
        index = json.load(f)

    running_cmds = get_running_train_cmds()
    print(f"当前存活的训练进程 run_id: {running_cmds}")

    updated = 0
    for run in index["runs"]:
        if run["status"] != "running":
            continue

        run_id = run["run_id"]
        meta, last_epoch = check_record_dir(run_id)

        # 如果进程还在运行，跳过
        if run_id in running_cmds:
            print(f"  ✓ {run_id} — 进程存活中 (epoch {last_epoch})")
            continue

        # 进程已死
        if meta is None:
            # 连 meta.json 都没有 — 从未启动成功
            print(f"  ✗ {run_id} — 无 meta.json, 标记 failed")
            run["status"] = "failed"
            run["end_time"] = datetime.now().isoformat()
            updated += 1
        elif meta.get("status") == "completed":
            # meta 里已经是 completed，同步到 index
            print(f"  ↻ {run_id} — meta 已标记 completed, 同步")
            run["status"] = "completed"
            run["end_time"] = meta.get("end_time", run.get("end_time", ""))
            updated += 1
        elif last_epoch is not None and last_epoch > 0:
            # 有训练记录但进程死了 — 可能是 crash 或被 kill
            print(f"  ⚠ {run_id} — 进程已死, epoch={last_epoch}, 标记 failed")
            run["status"] = "failed"
            run["end_time"] = datetime.now().isoformat()
            updated += 1
        else:
            # 没有有效 epoch 记录
            print(f"  ✗ {run_id} — 无有效训练记录, 标记 failed")
            run["status"] = "failed"
            run["end_time"] = datetime.now().isoformat()
            updated += 1

    print(f"\n共更新 {updated} 条记录")

    # 备份
    backup_path = INDEX_PATH.with_suffix(".json.bak")
    with open(backup_path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"已备份到 {backup_path}")

    # 保存
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"已保存到 {INDEX_PATH}")


if __name__ == "__main__":
    main()
