#!/usr/bin/env python3
"""
Training Recorder — 训练记录器
================================
将每次训练运行的配置、逐 epoch 指标、事件记录到
project_root/training_records/ 目录，供网页展示。

用法:
    from training.recorder import TrainingRecorder
    recorder = TrainingRecorder("v2_both_512")
    recorder.save_meta({"hidden_dim": 512, ...})
    recorder.log_epoch(phase="supervised", epoch=1, loss=0.04, re=0.2235)
    recorder.log_event("checkpoint_saved", path="...")
    recorder.set_status("completed")
"""

import os, sys, json, time
from datetime import datetime
from pathlib import Path

RECORDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training_records")


def get_run_id(name: str) -> str:
    """生成运行 ID: {datetime}_{name}"""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{now}_{name}"


def load_index() -> dict:
    """加载训练索引"""
    idx_path = os.path.join(RECORDS_DIR, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            return json.load(f)
    return {"runs": []}


def save_index(index: dict):
    """保存训练索引"""
    os.makedirs(RECORDS_DIR, exist_ok=True)
    idx_path = os.path.join(RECORDS_DIR, "index.json")
    with open(idx_path, 'w') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


class TrainingRecorder:
    """训练记录器 — 在训练脚本中使用"""

    def __init__(self, name: str, run_id: str = None, records_dir: str = RECORDS_DIR):
        self.records_dir = records_dir
        self.run_id = run_id or get_run_id(name)
        self.run_dir = os.path.join(records_dir, self.run_id)
        self.name = name
        self._epoch_file = os.path.join(self.run_dir, "epochs.jsonl")
        self._event_file = os.path.join(self.run_dir, "events.jsonl")
        self._meta_file = os.path.join(self.run_dir, "meta.json")

    def save_meta(self, meta: dict):
        """保存训练配置"""
        os.makedirs(self.run_dir, exist_ok=True)
        meta["run_id"] = self.run_id
        meta["name"] = self.name
        if "start_time" not in meta:
            meta["start_time"] = datetime.now().isoformat()
        if "status" not in meta:
            meta["status"] = "running"
        with open(self._meta_file, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # 更新索引
        index = load_index()
        # 删除同 name 旧记录（防止重复）
        index["runs"] = [r for r in index["runs"] if r["run_id"] != self.run_id]
        index["runs"].append({
            "run_id": self.run_id,
            "name": self.name,
            "status": meta["status"],
            "start_time": meta["start_time"],
        })
        # 按时间倒序
        index["runs"].sort(key=lambda r: r.get("start_time", ""), reverse=True)
        save_index(index)

        # 更新 current 软链接
        current_link = os.path.join(self.records_dir, "current")
        if os.path.islink(current_link):
            os.unlink(current_link)
        try:
            os.symlink(self.run_id, current_link, target_is_directory=True)
        except (OSError, PermissionError):
            pass

    def log_epoch(self, phase: str, epoch: int, **metrics):
        """记录一个 epoch 的指标"""
        os.makedirs(self.run_dir, exist_ok=True)
        record = {
            "phase": phase,
            "epoch": epoch,
            "time": datetime.now().isoformat(),
            **metrics,
        }
        with open(self._epoch_file, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_event(self, event_type: str, **data):
        """记录事件（checkpoint、错误等）"""
        os.makedirs(self.run_dir, exist_ok=True)
        record = {
            "event": event_type,
            "time": datetime.now().isoformat(),
            **data,
        }
        with open(self._event_file, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def set_status(self, status: str):
        """更新运行状态: running | completed | failed"""
        if os.path.exists(self._meta_file):
            with open(self._meta_file) as f:
                meta = json.load(f)
        else:
            meta = {"run_id": self.run_id, "name": self.name}
        meta["status"] = status
        if status in ("completed", "failed") and "end_time" not in meta:
            meta["end_time"] = datetime.now().isoformat()
        with open(self._meta_file, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # 更新索引
        index = load_index()
        for r in index["runs"]:
            if r["run_id"] == self.run_id:
                r["status"] = status
                if "end_time" in meta:
                    r["end_time"] = meta["end_time"]
                break
        save_index(index)


def load_run_data(run_id: str, records_dir: str = RECORDS_DIR) -> dict:
    """加载一次训练运行的全部数据"""
    run_dir = os.path.join(records_dir, run_id)
    if not os.path.isdir(run_dir):
        return None

    data = {"run_id": run_id}

    # meta
    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            data["meta"] = json.load(f)
    else:
        data["meta"] = {}

    # epochs
    epoch_path = os.path.join(run_dir, "epochs.jsonl")
    data["epochs"] = []
    if os.path.exists(epoch_path):
        with open(epoch_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data["epochs"].append(json.loads(line))

    # events
    event_path = os.path.join(run_dir, "events.jsonl")
    data["events"] = []
    if os.path.exists(event_path):
        with open(event_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data["events"].append(json.loads(line))

    return data


def list_runs(records_dir: str = RECORDS_DIR) -> list:
    """列出所有训练运行"""
    index = load_index()
    # 确保每个 run 有目录
    valid = []
    for r in index.get("runs", []):
        if os.path.isdir(os.path.join(records_dir, r["run_id"])):
            valid.append(r)
    return valid
