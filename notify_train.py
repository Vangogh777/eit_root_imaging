#!/usr/bin/env python3
"""
训练完成邮件通知脚本
======================
监控训练进程，完成后发送邮件通知。

用法:
    # 方式1: 监控PID (训练已在运行)
    python notify_train.py --pid 12345 --log train_diff_v3_smoke.log

    # 方式2: 直接启动训练 + 通知 (推荐)
    python notify_train.py --log train_diff_v3_smoke.log -- \
        python train_diff_eit.py --phase conditional --epochs_cond 20 ...

    # 方式3: 后台运行
    nohup python notify_train.py --pid 12345 --log log.txt &

SMTP 配置 (环境变量):
    export SMTP_HOST=smtp.gmail.com
    export SMTP_PORT=587
    export SMTP_USER=your@gmail.com
    export SMTP_PASS=your_app_password
    export NOTIFY_TO=your@email.com
"""

import os, sys, time, argparse, subprocess, signal
from datetime import datetime
from pathlib import Path


def get_smtp_config():
    """从环境变量读取 SMTP 配置"""
    config = {
        'host': os.environ.get('SMTP_HOST', ''),
        'port': int(os.environ.get('SMTP_PORT', '587')),
        'user': os.environ.get('SMTP_USER', ''),
        'pass': os.environ.get('SMTP_PASS', ''),
        'to': os.environ.get('NOTIFY_TO', os.environ.get('SMTP_USER', '')),
    }
    if not all([config['host'], config['user'], config['pass'], config['to']]):
        print("⚠️  SMTP 未配置, 将只打印结果不发邮件")
        print("   配置方式: export SMTP_HOST/PORT/USER/PASS/NOTIFY_TO")
        return None
    return config


def get_training_summary(log_path: str, tail_lines: int = 20) -> str:
    """提取训练日志末尾和关键指标"""
    if not Path(log_path).exists():
        return "(日志文件未找到)"

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 提取所有 epoch 摘要行
    epoch_lines = [l for l in lines if 'Loss:' in l and 'Val:' in l]

    summary_parts = []
    if epoch_lines:
        summary_parts.append(f"--- 训练指标 ({len(epoch_lines)} epochs) ---")
        # 前3个 epoch
        summary_parts.append("前 3 epoch:")
        summary_parts.extend(epoch_lines[:3])
        # 最后3个 epoch
        if len(epoch_lines) > 6:
            summary_parts.append(f"... (省略 {len(epoch_lines)-6} 行) ...")
        summary_parts.append("最后 3 epoch:")
        summary_parts.extend(epoch_lines[-3:])

    # 最后几行
    summary_parts.append("\n--- 日志尾部 ---")
    summary_parts.extend(lines[-tail_lines:])

    return '\n'.join(summary_parts)


def send_email(config: dict, subject: str, body: str):
    """发送邮件通知"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart()
    msg['From'] = config['user']
    msg['To'] = config['to']
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        with smtplib.SMTP(config['host'], config['port'], timeout=30) as server:
            server.starttls()
            server.login(config['user'], config['pass'])
            server.send_message(msg)
        print(f"✅ 邮件已发送到 {config['to']}")
        return True
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return False


def monitor_pid(pid: int, log_path: str, smtp_config: dict = None):
    """监控进程直到退出, 然后发送通知"""
    start_time = datetime.now()
    print(f"📡 监控 PID {pid} (开始于 {start_time.strftime('%H:%M:%S')})")

    while True:
        try:
            # 检查进程是否还在运行
            os.kill(pid, 0)
            time.sleep(30)  # 每30秒检查一次
        except (OSError, ProcessLookupError):
            # 进程已退出
            break

    end_time = datetime.now()
    duration = end_time - start_time
    duration_str = f"{duration.total_seconds()/60:.0f} 分钟"

    # 判断结果
    log_summary = get_training_summary(log_path)

    # 检查是否正常完成
    completed = 'completed' in log_summary.lower() or '✅' in log_summary
    failed = any(kw in log_summary.lower() for kw in ['error', 'traceback', 'killed', 'oom'])

    if completed:
        status = "✅ 完成"
        subject = f"EIT 训练完成 — {duration_str}"
    elif failed:
        status = "❌ 失败"
        subject = f"EIT 训练失败 — {duration_str}"
    else:
        status = "⚠️ 结束 (状态未知)"
        subject = f"EIT 训练结束 — {duration_str}"

    body = f"""训练状态: {status}
耗时: {duration_str}
日志: {log_path}
时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}

{log_summary}
"""
    print(f"\n{'='*60}")
    print(body)

    if smtp_config:
        send_email(smtp_config, subject, body)

    return 0 if completed else 1


def run_and_monitor(cmd: list, log_path: str, smtp_config: dict = None):
    """启动训练并监控, 完成后通知"""
    print(f"🚀 启动训练: {' '.join(cmd)}")

    with open(log_path, 'w') as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    print(f"   PID: {proc.pid}, 日志: {log_path}")
    return monitor_pid(proc.pid, log_path, smtp_config)


def main():
    parser = argparse.ArgumentParser(description='训练完成邮件通知')
    parser.add_argument('--pid', type=int, help='监控已有进程的 PID')
    parser.add_argument('--log', default='train.log', help='训练日志路径')
    parser.add_argument('--tail', type=int, default=30, help='日志尾部行数')
    parser.add_argument('cmd', nargs=argparse.REMAINDER, help='训练命令 (-- 之后)')
    args = parser.parse_args()

    smtp_config = get_smtp_config()

    # 清理 cmd (去掉 -- 分隔符)
    cmd = args.cmd
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]

    if args.pid:
        sys.exit(monitor_pid(args.pid, args.log, smtp_config))
    elif cmd:
        sys.exit(run_and_monitor(cmd, args.log, smtp_config))
    else:
        print("用法: python notify_train.py --pid <PID> --log <LOG>")
        print("  或: python notify_train.py --log <LOG> -- <训练命令>")
        print("\nSMTP 配置 (环境变量):")
        print("  SMTP_HOST  SMTP_PORT  SMTP_USER  SMTP_PASS  NOTIFY_TO")
        sys.exit(1)


if __name__ == '__main__':
    main()
