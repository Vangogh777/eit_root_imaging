#!/usr/bin/env python3
"""
EIT 结果动态展示服务器
========================
每次刷新页面时自动扫描 results/ 目录，生成带最新结果的可视化页面。

用法:
    python serve_results.py [--port 8080]

然后浏览器访问 http://localhost:8080
（远程服务器需 SSH 端口转发: ssh -L 8080:localhost:8080 ubuntu@<IP>）
"""

import os, sys, json, glob, argparse, importlib.util
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from pathlib import Path
import mimetypes
import datetime as dt

# 加入 training 模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 直接导入 recorder 模块（绕过 __init__.py 中 torch 等依赖）
import importlib.util
_recorder_spec = importlib.util.spec_from_file_location(
    "training_recorder",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "recorder.py")
)
_recorder_mod = importlib.util.module_from_spec(_recorder_spec)
_recorder_spec.loader.exec_module(_recorder_mod)
list_runs = _recorder_mod.list_runs
load_run_data = _recorder_mod.load_run_data
TRAINING_RECORDS_DIR = _recorder_mod.RECORDS_DIR

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
ALLOWED_DIRS = [
    os.path.realpath(RESULTS_DIR),
    os.path.realpath(DOCS_DIR),
    os.path.realpath(TRAINING_RECORDS_DIR),
]


def load_metrics(dir_path):
    """从 validation_xxx/ 目录加载指标"""
    metrics_path = os.path.join(dir_path, "metrics.json")
    report_path = os.path.join(dir_path, "report.txt")
    summary = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                data = json.load(f)
            summary = data.get("summary", {})
        except:
            pass
    # 也尝试从 report.txt 提取一行摘要
    summary_line = ""
    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                for line in f:
                    if "RE" in line and "CC" in line and "SSIM" in line:
                        summary_line = line.strip()
                        break
        except:
            pass
    return summary, summary_line


def scan_results():
    """扫描 results/ 目录，返回所有结果组"""
    groups = []  # 每个元素: {name, path, images: [{name, path, size}], metrics, is_dir}

    # 1. 子目录（validation_xxx/ 等）
    for d in sorted(os.listdir(RESULTS_DIR)):
        dpath = os.path.join(RESULTS_DIR, d)
        if not os.path.isdir(dpath) or d.startswith('.'):
            continue
        images = []
        for fname in sorted(os.listdir(dpath)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                fpath = os.path.join(dpath, fname)
                size = os.path.getsize(fpath)
                label = fname.rsplit('.', 1)[0].replace('_', ' ').title()
                images.append({'name': fname, 'path': f'results/{d}/{fname}',
                               'label': label, 'size': size})
        metrics, summary_line = load_metrics(dpath)
        groups.append({
            'name': d,
            'path': f'results/{d}/',
            'images': images,
            'metrics': metrics,
            'summary_line': summary_line,
            'is_dir': True,
        })

    # 2. 根目录的 .png 文件
    root_images = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
            fpath = os.path.join(RESULTS_DIR, fname)
            size = os.path.getsize(fpath)
            label = fname.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
            root_images.append({'name': fname, 'path': f'results/{fname}',
                                'label': label, 'size': size})

    if root_images:
        metrics = {}
        summary_line = ""
        # 尝试从同名的 validation 目录找指标
        groups.insert(0, {
            'name': '根目录图像',
            'path': 'results/',
            'images': root_images,
            'metrics': metrics,
            'summary_line': summary_line,
            'is_dir': False,
        })

    return groups


def format_size(n):
    for unit in ['B', 'KB', 'MB']:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def format_metrics_html(metrics, summary_line):
    """将指标渲染为 HTML badge 字符串"""
    parts = []
    for key, color in [('RE', '#f59e0b'), ('CC', '#22c55e'),
                        ('SSIM', '#3b82f6'), ('PSNR', '#a855f7'),
                        ('IoU', '#ec4899'), ('Dice', '#14b8a6')]:
        if key in metrics:
            m = metrics[key]
            val = m.get('mean', 0)
            parts.append(
                f'<span class="metric" style="color:{color}">'
                f'{key} = {val:.4f}</span>'
            )
    if summary_line:
        parts.append(f'<span class="metric" style="color:#667788">📄 {summary_line[:80]}</span>')
    return ' '.join(parts)


def generate_results_html():
    """生成完整 HTML 页面"""
    groups = scan_results()

    # 统计
    total_png = sum(len(g['images']) for g in groups)
    total_dirs = sum(1 for g in groups if g['is_dir'])

    # 构建每个组的卡片
    cards_html = ""
    for g in groups:
        metrics_html = format_metrics_html(g['metrics'], g['summary_line'])

        # 取第一张图为封面
        cover_path = ""
        if g['images']:
            cover_path = g['images'][0]['path']
        else:
            cover_path = ""

        # 文字描述
        n_img = len(g['images'])
        desc = f"{n_img} 张图像"
        if g['is_dir']:
            desc += f" · 子目录 {g['name']}"
        else:
            desc += " · results/ 根目录"

        # 取指标显示在 badge
        re_val = ""
        if 'RE' in g['metrics']:
            m = g['metrics']['RE']
            re_val = f"RE = {m.get('mean', 0):.3f}"

        # 是否为新的 v2 结果
        is_v2 = 'v2' in g['name'].lower() or 'best' in g['name'].lower()

        if cover_path:
            card = f'''
        <div class="card {'v2' if is_v2 else ''}">
            <a href="/{cover_path}" target="_blank">
            <div class="img-wrap"><img src="/{cover_path}" alt="{g['name']}" loading="lazy"></div>
            <div class="info">
                <h3>{'🚀 ' if is_v2 else ''}{g['name']}</h3>
                <p>{desc}</p>
                <div class="metrics-row">{metrics_html}</div>
            </div>
            </a>
        </div>'''
        else:
            card = f'''
        <div class="card {'v2' if is_v2 else ''}">
            <a href="/{g['path']}" target="_blank">
            <div class="img-wrap" style="display:flex;align-items:center;justify-content:center;background:#0a0e17;">
                <span style="font-size:48px;opacity:0.5;">📁</span>
            </div>
            <div class="info">
                <h3>{g['name']}</h3>
                <p>{desc}</p>
                <div class="metrics-row">{metrics_html}</div>
            </div>
            </a>
        </div>'''
        cards_html += card

    # 如果该目录太多图，加一个"查看全部"链接
    for g in groups:
        if len(g['images']) > 4 and g['is_dir']:
            pass  # 已经有了链接

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EIT 实验结果总览</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: #0a0e17;
    color: #e0e8f0;
    min-height: 100vh;
}}
body::before {{
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.06) 0%, transparent 50%);
    pointer-events: none; z-index: 0;
}}
.container {{ position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 24px; }}
.header {{
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 20px; border-bottom: 1px solid rgba(59,130,246,0.15);
    margin-bottom: 28px; flex-wrap: wrap; gap: 12px;
}}
.header h1 {{ font-size: 22px; font-weight: 600; }}
.home-btn {{ display:inline-flex;align-items:center;gap:4px;padding:6px 14px;background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);border-radius:8px;color:#60a5fa;text-decoration:none;font-size:13px;transition:all 0.2s; }}
.home-btn:hover {{ background:rgba(59,130,246,0.2); }}
.header .count {{ font-size: 13px; color: #667788; }}
.refresh-info {{ font-size: 12px; color: #556677; margin-top: 4px; }}

.gallery {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 24px;
}}
.card {{
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 14px; overflow: hidden;
    transition: all 0.3s;
    backdrop-filter: blur(10px);
}}
.card:hover {{
    border-color: rgba(59,130,246,0.35);
    box-shadow: 0 8px 32px rgba(59,130,246,0.12);
    transform: translateY(-3px);
}}
.card.v2 {{
    border-color: rgba(34,197,94,0.2);
    background: rgba(15,30,20,0.8);
}}
.card.v2:hover {{
    border-color: rgba(34,197,94,0.5);
    box-shadow: 0 8px 32px rgba(34,197,94,0.1);
}}
.card a {{ text-decoration: none; color: inherit; display: block; }}
.card .img-wrap {{
    width: 100%; height: 240px;
    background: #0a0e17;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
}}
.card .img-wrap img {{
    width: 100%; height: 100%;
    object-fit: cover;
    transition: transform 0.4s;
}}
.card:hover .img-wrap img {{ transform: scale(1.04); }}
.card .info {{ padding: 16px 18px; }}
.card .info h3 {{ font-size: 15px; font-weight: 600; color: #d0d8e0; margin-bottom: 4px; }}
.card .info p {{ font-size: 12px; color: #667788; line-height: 1.5; margin-bottom: 6px; }}
.metrics-row {{ display: flex; flex-wrap: wrap; gap: 4px 10px; }}
.metric {{ font-size: 11px; padding: 2px 8px; border-radius: 6px;
           background: rgba(255,255,255,0.04); }}

.header .header-top {{ display:flex;align-items:center;gap:12px;flex-wrap:wrap; }}
@media (max-width: 768px) {{
    .gallery {{ grid-template-columns: 1fr; }}
    .card .img-wrap {{ height: 200px; }}
}}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                <a href="/" class="home-btn">← 返回论文首页</a>
                <h1>🔬 EIT 实验结果总览</h1>
            </div>
            <div class="refresh-info">自动扫描 results/ 目录 · 共 {total_png} 张图片 / {total_dirs} 个子目录 · 刷新页面即更新</div>
        </div>
        <span class="count">🕐 {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>

    <div class="gallery">
        {cards_html}
    </div>

    <div style="text-align:center;margin-top:40px;padding:20px;border-top:1px solid rgba(59,130,246,0.1);color:#556677;font-size:12px;">
        💡 把新结果图片放到 <code>results/</code> 目录下，刷新页面即可看到<br>
        子目录会自动识别为独立的结果组 · 含 <code>metrics.json</code> 的子目录会显示指标
    </div>
</div>
</body>
</html>'''
    return html


# ============ 训练记录 HTML 生成 ============

def generate_training_card():
    """生成首页训练摘要卡片 HTML"""
    runs = list_runs()
    if not runs:
        return ""

    # 当前运行（第一行 = 最新）
    latest = runs[0]
    data = load_run_data(latest["run_id"])
    meta = data.get("meta", {}) if data else {}

    sup_epochs = [e for e in (data.get("epochs") or []) if e["phase"] == "supervised"]
    unsup_epochs = [e for e in (data.get("epochs") or []) if e["phase"] == "unsupervised"]

    best_re = meta.get("best_re", "—")
    total_params = meta.get("model_params", 0)
    hidden = meta.get("hidden_dim", "—")
    status = latest["status"]
    status_icon = {"completed": "✅", "running": "🔴", "failed": "❌"}.get(status, "⚪")
    status_text = {"completed": "已完成", "running": "运行中", "failed": "失败"}.get(status, status)

    # 最新的 RE 曲线用 SVG 简单示意
    re_values = [e.get("re") for e in sup_epochs if e.get("re") is not None]
    re_chart = ""
    if re_values:
        n = len(re_values)
        h, w = 40, 200
        min_r, max_r = min(re_values), max(re_values)
        rng = max(max_r - min_r, 1e-6)
        pts = []
        for i, v in enumerate(re_values):
            x = i / max(n - 1, 1) * w
            y = h - (v - min_r) / rng * (h - 4) - 2
            pts.append(f"{x:.1f},{y:.1f}")
        poly = " ".join(pts)
        re_chart = f'''<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="margin:4px 0;">
          <polyline points="{poly}" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-linecap="round"/>
        </svg>'''

    n_sup = len(sup_epochs)
    n_total = meta.get("epochs_sup", n_sup)
    sup_pct = n_total and int(n_sup / n_total * 100) or 100

    html = f'''
<div class="section-title">
    <h2>📋 训练记录</h2>
    <div class="line"></div>
</div>
<div class="training-card" onclick="location.href='/training/'" style="cursor:pointer;">
    <div class="training-header">
        <div class="training-status">
            <span class="status-dot">{status_icon}</span>
            <span class="status-label">{status_text}</span>
            <span class="training-name">{latest["name"]}</span>
        </div>
        <div style="font-size:12px;color:#667788;">点击查看全部 →</div>
    </div>
    <div class="training-body">
        <div class="training-info">
            <div class="training-metrics">
                <div class="tm-item"><span class="tm-label">最佳 RE</span><span class="tm-value">{best_re}</span></div>
                <div class="tm-item"><span class="tm-label">参数量</span><span class="tm-value">{total_params:,}</span></div>
                <div class="tm-item"><span class="tm-label">hidden_dim</span><span class="tm-value">{hidden}</span></div>
            </div>
            {re_chart if re_chart else '<div class="tm-item"><span class="tm-label">RE 曲线</span></div>'}
        </div>
        <div class="training-progress">
            <div class="tp-bar"><div class="tp-fill" style="width:{sup_pct}%;"></div></div>
            <div class="tp-label">有监督 {n_sup}/{n_total} epoch</div>
        </div>
    </div>
</div>
<style>
.training-card {{
    background: rgba(15,23,42,0.8); border:1px solid rgba(59,130,246,0.12);
    border-radius:14px; padding:20px; margin-bottom:28px; transition:all 0.3s;
    backdrop-filter:blur(10px);
}}
.training-card:hover {{ border-color:rgba(59,130,246,0.35); transform:translateY(-2px); }}
.training-header {{ display:flex;justify-content:space-between;align-items:center;margin-bottom:14px; }}
.training-status {{ display:flex;align-items:center;gap:8px; }}
.status-dot {{ font-size:16px; }}
.status-label {{ font-size:13px;color:#94a3b8; }}
.training-name {{ font-size:14px;font-weight:600;color:#e0e8f0; }}
.training-body {{ }}
.training-info {{ display:flex;gap:24px;align-items:center;flex-wrap:wrap; }}
.training-metrics {{ display:flex;gap:16px;flex-wrap:wrap; }}
.tm-item {{ display:flex;flex-direction:column;gap:2px; }}
.tm-label {{ font-size:11px;color:#667788; }}
.tm-value {{ font-size:16px;font-weight:600;color:#e0e8f0; }}
.training-progress {{ margin-top:12px; }}
.tp-bar {{ height:4px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden; }}
.tp-fill {{ height:100%;background:linear-gradient(90deg,#3b82f6,#22c55e);border-radius:2px;transition:width 0.5s; }}
.tp-label {{ font-size:11px;color:#667788;margin-top:4px; }}
</style>'''
    return html


def inject_training_card_into_homepage(homepage_html: str) -> str:
    """在首页的 模型信息 和 实验结果 之间插入训练记录卡片"""
    card = generate_training_card()
    if not card:
        return homepage_html
    # 在 "实验结果" section 前面插入
    marker = '<!-- ===== Results Gallery ===== -->'
    if marker in homepage_html:
        return homepage_html.replace(marker, card + '\n\n    ' + marker)
    return homepage_html


BASE_STYLE = '''
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; background: #0a0e17; color: #e0e8f0; min-height: 100vh; line-height: 1.6; }
body::before { content: ""; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.06) 0%, transparent 50%), radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.06) 0%, transparent 50%), radial-gradient(ellipse at 50% 80%, rgba(6,182,212,0.04) 0%, transparent 50%); pointer-events: none; z-index: 0; }
body::after { content: ""; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background-image: linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px); background-size: 60px 60px; pointer-events: none; z-index: 0; }
.container { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 0 24px; }
.navbar { padding: 20px 0; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(59,130,246,0.15); margin-bottom: 32px; }
.navbar-left { display: flex; align-items: center; gap: 12px; }
.logo-dot { width: 10px; height: 10px; background: #3b82f6; border-radius: 50%; box-shadow: 0 0 12px rgba(59,130,246,0.5); }
.navbar-title { font-size: 15px; font-weight: 600; color: #e0e8f0; letter-spacing: 0.5px; }
.navbar-badge { font-size: 11px; padding: 3px 10px; border-radius: 12px; background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.2); }
.section-title { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding-top: 10px; }
.section-title h2 { font-size: 20px; font-weight: 600; color: #d0d8e0; white-space: nowrap; }
.section-title .line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(59,130,246,0.3), transparent); }
.nav-back { display: inline-flex; align-items: center; gap: 4px; padding: 6px 14px; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.2); border-radius: 8px; color: #60a5fa; text-decoration: none; font-size: 13px; transition: all 0.2s; }
.nav-back:hover { background: rgba(59,130,246,0.2); }
.run-card { background: rgba(15,23,42,0.8); border: 1px solid rgba(59,130,246,0.12); border-radius: 14px; padding: 16px 20px; margin-bottom: 12px; cursor: pointer; transition: all 0.3s; backdrop-filter: blur(10px); }
.run-card:hover { border-color: rgba(59,130,246,0.35); transform: translateY(-2px); }
.run-card.completed { border-color: rgba(34,197,94,0.15); }
.run-card.failed { border-color: rgba(239,68,68,0.15); }
.run-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 8px; }
.run-name { font-size: 16px; font-weight: 600; }
.run-status { font-size: 11px; padding: 2px 10px; border-radius: 10px; }
.run-status.completed { background: rgba(34,197,94,0.1); color: #22c55e; }
.run-status.running { background: rgba(59,130,246,0.1); color: #60a5fa; }
.run-status.failed { background: rgba(239,68,68,0.1); color: #ef4444; }
.run-meta { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: #8899aa; margin-top: 4px; }
.run-meta-label { color: #556677; }
.run-chart { margin-top: 6px; }
</style>
'''





def generate_training_overview() -> str:
    """生成训练总览页 /training/"""
    runs = list_runs()
    now_str = dt.datetime.now().strftime('%Y-%m-%d %H:%M')

    # 运行卡片
    cards = ""
    for r in runs:
        run_id = r["run_id"]
        data = load_run_data(run_id)
        meta = data.get("meta", {}) if data else {}
        sup_epochs = [e for e in (data.get("epochs") or []) if e["phase"] == "supervised"]
        unsup_epochs = [e for e in (data.get("epochs") or []) if e["phase"] == "unsupervised"]

        status = r.get("status", "completed")
        status_cls = status
        status_icon = {"completed": "✅", "running": "🔴", "failed": "❌"}.get(status, "⚪")

        best_re = meta.get("best_re", "—")
        params = meta.get("model_params", 0)
        hidden = meta.get("hidden_dim", "—")
        n_sup = len(sup_epochs)
        n_unsup = len(unsup_epochs)
        start = r.get("start_time", "")[:16].replace("T", " ")

        # RE 微型曲线
        re_values = [e.get("re") for e in sup_epochs if e.get("re") is not None]
        chart_svg = ""
        if re_values:
            n = len(re_values)
            h, w = 30, 120
            min_r, max_r = min(re_values), max(re_values)
            rng = max(max_r - min_r, 1e-6)
            pts = []
            for i, v in enumerate(re_values):
                x = i / max(n - 1, 1) * w
                y = h - (v - min_r) / rng * (h - 4) - 2
                pts.append(f"{x:.1f},{y:.1f}")
            chart_svg = f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polyline points="{" ".join(pts)}" fill="none" stroke="#22c55e" stroke-width="1"/></svg>'

        cards += f'''
    <div class="run-card {status_cls}" onclick="location.href='/training/{run_id}/'">
        <div class="run-card-header">
            <div><span class="run-name">{status_icon} {r["name"]}</span></div>
            <span class="run-status {status_cls}">{status}</span>
        </div>
        <div class="run-meta">
            <span class="run-meta-item"><span class="run-meta-label">🕐</span> {start}</span>
            <span class="run-meta-item"><span class="run-meta-label">最佳 RE</span> {best_re}</span>
            <span class="run-meta-item"><span class="run-meta-label">参数</span> {params:,}</span>
            <span class="run-meta-item"><span class="run-meta-label">hidden_dim</span> {hidden}</span>
            <span class="run-meta-item"><span class="run-meta-label">有监督</span> {n_sup} ep</span>
            <span class="run-meta-item"><span class="run-meta-label">无监督</span> {n_unsup} ep</span>
        </div>
        {f'<div class="run-chart">{chart_svg}</div>' if chart_svg else ''}
    </div>'''

    if not cards:
        cards = '<div style="text-align:center;padding:60px;color:#667788;">暂无训练记录</div>'
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>训练记录 - EIT</title>
{BASE_STYLE}
</head>
<body><div class="container">
    <div class="navbar">
        <div class="navbar-left">
            <span class="logo-dot"></span>
            <span class="navbar-title">EIT 训练记录</span>
        </div>
        <div>
            <a href="/" class="nav-back">&larr; 返回首页</a>
            <span class="navbar-badge" style="margin-left:8px;">{now_str}</span>
        </div>
    </div>

    <div class="section-title">
        <h2>&#128203; 训练总览</h2>
        <div class="line"></div>
    </div>

    {cards}

    <div style="text-align:center;padding:30px;color:#556677;font-size:12px;">
        &#128161; 以后运行训练脚本会自动记录到此页面
    </div>
</div></body></html>'''
    return html


def generate_training_detail(run_id: str) -> str:
    """生成单次训练详情页 /training/{run_id}/"""
    data = load_run_data(run_id)
    if not data:
        return None

    meta = data.get("meta", {})
    epochs = data.get("epochs", [])
    events = data.get("events", [])

    sup_epochs = [e for e in epochs if e["phase"] == "supervised"]
    unsup_epochs = [e for e in epochs if e["phase"] == "unsupervised"]

    status = meta.get("status", "completed")
    status_icon = {"completed": "\u2705", "running": "\U0001F534", "failed": "\u274C"}.get(status, "\u26AA")
    name = meta.get("name", run_id)
    start = meta.get("start_time", "")[:19].replace("T", " ")
    end = meta.get("end_time", "")[:19].replace("T", " ") if meta.get("end_time") else "\u2014"

    # 配置表格
    config_fields = [
        ("hidden_dim", "hidden_dim"), ("gnn_hidden", "gnn_hidden"), ("gnn_layers", "GNN \u5C42\u6570"),
        ("batch_size", "batch_size"), ("mode", "\u8BAD\u7EC3\u6A21\u5F0F"), ("epochs_sup", "\u6709\u76D1\u7763 epoch"),
        ("epochs_unsup", "\u65E0\u76D1\u7763 epoch"), ("model_params", "\u53C2\u6570\u91CF"),
    ]
    config_rows = ""
    for key, label in config_fields:
        val = meta.get(key, "\u2014")
        if key == "model_params" and isinstance(val, (int, float)):
            val = f"{int(val):,}"
        config_rows += f'<div class="cfg-item"><span class="cfg-label">{label}</span><span class="cfg-value">{val}</span></div>'

    # 生成曲线 SVG
    def make_chart(ep_list, key="re", color="#22c55e", label="RE"):
        values = [e.get(key) for e in ep_list if e.get(key) is not None]
        if not values:
            return '<div style="color:#667788;font-size:12px;">\u65E0\u6570\u636E</div>'
        n = len(values)
        h, w = 120, 500
        min_v, max_v = min(values), max(values)
        rng = max(max_v - min_v, 1e-8)
        pts = []
        for i, v in enumerate(values):
            x = i / max(n - 1, 1) * w
            y = h - (v - min_v) / rng * (h - 10) - 5
            pts.append(f"{x:.1f},{y:.1f}")
        poly = " ".join(pts)
        grid = ""
        for i in range(5):
            gy = i / 4 * h
            grid += f'<line x1="0" y1="{gy:.0f}" x2="{w}" y2="{gy:.0f}" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>'
        return f'''<svg width="100%" viewBox="0 0 {w} {h}" style="max-width:{w}px;">
          {grid}
          <polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
          <text x="{w-30}" y="15" fill="{color}" font-size="10">{label}: {values[-1]:.4f}</text>
        </svg>'''

    chart_sup = make_chart(sup_epochs, "re", "#22c55e", "RE (\u6709\u76D1\u7763)")
    loss_sup = make_chart(sup_epochs, "loss", "#3b82f6", "Loss (\u6709\u76D1\u7763)")
    chart_unsup = make_chart(unsup_epochs, "loss", "#f59e0b", "Loss (\u65E0\u76D1\u7763)")

    # 事件列表
    events_html = ""
    for ev in events:
        et = ev.get("event", "")
        etime = ev.get("time", "")[:19].replace("T", " ")
        edetail = " | ".join(f"{k}: {v}" for k, v in ev.items() if k not in ("event", "time"))
        icon = {"best_model_saved": "\U0001F3C6", "checkpoint_saved": "\U0001F4BE", "training_completed": "\u2705"}.get(et, "\U0001F4CC")
        events_html += f'<div class="ev-item">{icon} <span class="ev-time">{etime}</span> {et} {edetail}</div>'

    detail_style = '''
<style>
.detail-section { background:rgba(15,23,42,0.8);border:1px solid rgba(59,130,246,0.12);border-radius:14px;padding:20px;margin-bottom:20px;backdrop-filter:blur(10px); }
.cfg-grid { display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px; }
.cfg-item { display:flex;flex-direction:column;gap:2px; }
.cfg-label { font-size:11px;color:#667788; }
.cfg-value { font-size:14px;font-weight:600;color:#e0e8f0; }
.charts { display:grid;grid-template-columns:1fr 1fr;gap:16px; }
@media(max-width:700px){ .charts { grid-template-columns:1fr; } }
.ev-list { max-height:300px;overflow-y:auto; }
.ev-item { padding:6px 0;font-size:12px;border-bottom:1px solid rgba(255,255,255,0.04); }
.ev-time { color:#556677; }
</style>'''

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name} - EIT \u8BAD\u7EC3\u8BE6\u60C5</title>
{BASE_STYLE}
{detail_style}
</head>
<body><div class="container">
    <div class="navbar">
        <div class="navbar-left">
            <span class="logo-dot"></span>
            <span class="navbar-title">EIT \u8BAD\u7EC3\u8BE6\u60C5</span>
        </div>
        <div>
            <a href="/training/" class="nav-back">&larr; \u8BAD\u7EC3\u603B\u89C8</a>
            <a href="/" class="nav-back" style="margin-left:6px;">&larr; \u9996\u9875</a>
            <span class="navbar-badge" style="margin-left:8px;">{status_icon} {status}</span>
        </div>
    </div>

    <div style="margin-bottom:20px;">
        <span style="font-size:22px;font-weight:600;">{name}</span>
        <span style="font-size:12px;color:#667788;margin-left:12px;">{start} &rarr; {end}</span>
    </div>

    <div class="detail-section">
        <div class="section-title" style="margin-bottom:16px;padding-top:0;"><h2>&#9881;&#65039; \u8BAD\u7EC3\u914D\u7F6E</h2><div class="line"></div></div>
        <div class="cfg-grid">{config_rows}</div>
    </div>

    <div class="detail-section">
        <div class="section-title" style="margin-bottom:16px;padding-top:0;"><h2>&#128200; \u8BAD\u7EC3\u66F2\u7EBF</h2><div class="line"></div></div>
        <div class="charts">
            <div><div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">\u6709\u76D1\u7763 RE</div>{chart_sup}</div>
            <div><div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">\u6709\u76D1\u7763 Loss</div>{loss_sup}</div>
            <div><div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">\u65E0\u76D1\u7763 Loss</div>{chart_unsup}</div>
        </div>
    </div>

    <div class="detail-section">
        <div class="section-title" style="margin-bottom:16px;padding-top:0;"><h2>&#128203; \u4E8B\u4EF6\u65E5\u5FD7</h2><div class="line"></div></div>
        <div class="ev-list">{events_html if events_html else '<div style="color:#667788;font-size:12px;">暂无事件</div>'}</div>
    </div>
</div></body></html>'''
    return html


# ============ 文档浏览 (Docs Viewer) ============

def _get_docs_files():
    """扫描 docs/ 目录，返回文件列表"""
    files = []
    for dirpath, dirnames, filenames in os.walk(DOCS_DIR):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fn in filenames:
            if fn.startswith('.') or fn.endswith('.pyc'):
                continue
            fpath = os.path.join(dirpath, fn)
            rel = os.path.relpath(fpath, DOCS_DIR)
            stat = os.stat(fpath)
            ext = os.path.splitext(fn)[1].lower()
            icon = {'html': '\U0001F310', 'md': '\U0001F4DD', 'pdf': '\U0001F4C4',
                    'png': '\U0001F5BC', 'jpg': '\U0001F5BC', 'py': '\U0001F4BB',
                    'json': '\U0001F4CB', 'txt': '\U0001F4C3'}.get(ext, '\U0001F4C4')
            mtype = {'html': 'HTML', 'md': 'Markdown', 'pdf': 'PDF',
                     'png': 'Image', 'jpg': 'Image', 'py': 'Python'}.get(ext, ext[1:].upper())
            files.append({
                'name': fn,
                'path': rel,
                'url': f'/docs-view/?file={rel}',
                'raw_url': f'/docs/{rel}',
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'ext': ext,
                'icon': icon,
                'type': mtype,
            })
    # Sort by mtime descending
    files.sort(key=lambda f: f['mtime'], reverse=True)
    return files


def _simple_markdown(text: str) -> str:
    """极简 Markdown → HTML 转换"""
    import re
    lines = text.split('\n')
    out = []
    in_code = False
    in_table = False
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith('```'):
            if in_table:
                out.append('</table>')
                in_table = False
            if in_code:
                out.append('</pre>')
                in_code = False
            else:
                out.append('<pre style="background:#111827;padding:12px;border-radius:8px;overflow-x:auto;font-size:13px;">')
                in_code = True
            i += 1
            continue

        if in_code:
            out.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            i += 1
            continue

        # Tables: process BEFORE text formatting, use lookahead
        is_table_line = line.strip().startswith('|') and '|' in line
        if is_table_line:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if all(re.match(r'^[-:]+$', c) for c in cells):
                i += 1
                continue  # separator line
            # Now apply formatting to cells
            def fmt_cell(c):
                c = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', c)
                c = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', c)
                c = re.sub(r'\*(.+?)\*', r'<em>\1</em>', c)
                c = re.sub(r'`([^`]+)`', r'<code style="background:#1e293b;padding:1px 5px;border-radius:3px;color:#22c55e;">\1</code>', c)
                return c
            cells = [fmt_cell(c) for c in cells]
            tag = 'th' if not in_table else 'td'
            row = '<tr>' + ''.join(f'<{tag} style="border:1px solid rgba(255,255,255,0.1);padding:6px 12px;text-align:left;">{c}</{tag}>' for c in cells) + '</tr>'
            if not in_table:
                out.append('<table style="border-collapse:collapse;width:100%;margin:8px 0;">')
                in_table = True
            out.append(row)
            # Check if next line is also table
            if i+1 < len(lines) and not (lines[i+1].strip().startswith('|') and '|' in lines[i+1]):
                out.append('</table>')
                in_table = False
            i += 1
            continue
        elif in_table:
            out.append('</table>')
            in_table = False

        # Code blocks and headers AFTER table check
        orig_line = line

        # Headers
        line = re.sub(r'^#### (.+)', r'<h4>\1</h4>', line)
        line = re.sub(r'^### (.+)', r'<h3>\1</h3>', line)
        line = re.sub(r'^## (.+)', r'<h2>\1</h2>', line)
        line = re.sub(r'^# (.+)', r'<h1>\1</h1>', line)

        # Bold / Italic
        line = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', line)
        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        line = re.sub(r'\*(.+?)\*', r'<em>\1</em>', line)
        line = re.sub(r'`([^`]+)`', r'<code style="background:#1e293b;padding:2px 6px;border-radius:4px;font-size:0.9em;">\1</code>', line)

        # Links
        line = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2" style="color:#60a5fa;">\1</a>', line)

        # HR
        line = re.sub(r'^---+$', '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:16px 0;">', line)

        # List items
        if re.match(r'^\s*[-*+]\s', line):
            line = re.sub(r'^\s*[-*+]\s+(.+)', r'<li>\1</li>', line)
            if not out or not out[-1].startswith('<ul>'):
                out.append('<ul style="padding-left:20px;">')
            out.append(line)
            i += 1
            continue
        elif re.match(r'^\s*\d+\.\s', line):
            line = re.sub(r'^\s*\d+\.\s+(.+)', r'<li>\1</li>', line)
            if not out or not out[-1].startswith('<ol>'):
                out.append('<ol style="padding-left:20px;">')
            out.append(line)
            i += 1
            continue
        else:
            if out and (out[-1].startswith('<ul>') or out[-1].startswith('<ol>')):
                out.append('</ul>' if out[-1].startswith('<ul>') else '</ol>')

        # Paragraph
        if line.strip():
            out.append(f'<p style="margin:8px 0;">{line}</p>')
        else:
            out.append('<br>')
        i += 1

    if in_table:
        out.append('</table>')
    if in_code:
        out.append('</pre>')
    return '\n'.join(out)


def generate_docs_list(query: str = "", sort_by: str = "time") -> str:
    """生成文档列表页"""
    files = _get_docs_files()

    # Filter
    if query:
        q = query.lower()
        files = [f for f in files if q in f['name'].lower() or q in f['path'].lower()]

    # Sort
    if sort_by == 'name':
        files.sort(key=lambda f: f['name'].lower())
    else:
        files.sort(key=lambda f: f['mtime'], reverse=True)

    total = len(files)
    now = dt.datetime.now().strftime('%Y-%m-%d %H:%M')

    # File cards
    cards = ""
    for f in files:
        mtime_str = dt.datetime.fromtimestamp(f['mtime']).strftime('%m-%d %H:%M')
        size_str = format_size(f['size'])
        is_md = f['ext'] == '.md'
        is_viewable = f['ext'] in ('.md', '.html', '.pdf', '.png', '.jpg')
        href = f['url'] if is_md else f['raw_url']
        target = '' if is_md else ' target="_blank"'
        cards += f'''
        <a href="{href}"{target} style="text-decoration:none;display:block;">
        <div class="doc-card">
            <div class="doc-card-icon">{f['icon']}</div>
            <div class="doc-card-info">
                <div class="doc-card-name">{f['name']}</div>
                <div class="doc-card-path">{f['path']}</div>
            </div>
            <div class="doc-card-meta">
                <span class="doc-type">{f['type']}</span>
                <span class="doc-size">{size_str}</span>
                <span class="doc-time">{mtime_str}</span>
            </div>
        </div>
        </a>'''

    if not cards:
        cards = '<div style="text-align:center;padding:60px;color:#667788;">未找到匹配文件</div>'

    doc_list_style = '''
<style>
.doc-toolbar { display:flex;gap:12px;align-items:center;margin-bottom:24px;flex-wrap:wrap; }
.doc-search { flex:1;min-width:200px;padding:10px 16px;background:rgba(15,23,42,0.9);border:1px solid rgba(59,130,246,0.2);border-radius:10px;color:#e0e8f0;font-size:14px;outline:none; }
.doc-search::placeholder { color:#556677; }
.doc-search:focus { border-color:rgba(59,130,246,0.5); }
.doc-sort { padding:10px 16px;background:rgba(15,23,42,0.9);border:1px solid rgba(59,130,246,0.15);border-radius:10px;color:#94a3b8;font-size:13px;cursor:pointer;outline:none;white-space:nowrap; }
.doc-sort:hover { border-color:rgba(59,130,246,0.3); }
.doc-sort option { background:#0f172a;color:#e0e8f0; }
.doc-card { display:flex;align-items:center;gap:14px;padding:14px 18px;margin-bottom:6px;background:rgba(15,23,42,0.6);border:1px solid rgba(59,130,246,0.06);border-radius:12px;transition:all 0.2s; }
.doc-card:hover { background:rgba(15,23,42,0.9);border-color:rgba(59,130,246,0.25);transform:translateX(2px); }
.doc-card-icon { font-size:24px;flex-shrink:0;width:36px;text-align:center; }
.doc-card-info { flex:1;min-width:0; }
.doc-card-name { font-size:14px;font-weight:600;color:#d0d8e0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.doc-card-path { font-size:11px;color:#556677;margin-top:2px; }
.doc-card-meta { display:flex;gap:12px;align-items:center;flex-shrink:0;font-size:11px;color:#667788; }
.doc-type { padding:2px 8px;border-radius:6px;background:rgba(59,130,246,0.08);color:#60a5fa; }
.doc-size { color:#556677; }
.doc-time { color:#445566;min-width:80px;text-align:right; }
@media (max-width:600px) {
    .doc-card-meta { display:none; }
    .doc-toolbar { flex-direction:column; }
}
</style>'''

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>研究文档 - EIT</title>
{BASE_STYLE}
{doc_list_style}
</head>
<body><div class="container">
    <div class="navbar">
        <div class="navbar-left">
            <span class="logo-dot"></span>
            <span class="navbar-title">EIT 研究文档</span>
        </div>
        <div>
            <a href="/" class="nav-back">&larr; 返回首页</a>
            <span class="navbar-badge" style="margin-left:8px;">{total} 个文件</span>
        </div>
    </div>

    <div class="doc-toolbar">
        <input type="text" class="doc-search" id="search" placeholder="搜索文件名..."
               value="{query}" oninput="filterDocs()">
        <select class="doc-sort" id="sort" onchange="filterDocs()">
            <option value="time" {'selected' if sort_by=='time' else ''}>按时间排序</option>
            <option value="name" {'selected' if sort_by=='name' else ''}>按名称排序</option>
        </select>
    </div>

    <div id="doc-list">
        {cards}
    </div>

    <div style="text-align:center;padding:30px;color:#556677;font-size:12px;">
        &copy; EIT 研究项目 &middot; {now}
    </div>
</div>
<script>
function filterDocs() {{
    var q = document.getElementById('search').value;
    var s = document.getElementById('sort').value;
    window.location.href = '/docs-list/?q=' + encodeURIComponent(q) + '&sort=' + s;
}}
document.getElementById('search').addEventListener('keydown', function(e) {{
    if (e.key === 'Enter') filterDocs();
}});
</script>
</body></html>'''
    return html


def generate_docs_view(file_path: str) -> str:
    """生成文档查看页（Markdown 渲染）"""
    full_path = os.path.join(DOCS_DIR, file_path)
    full_path = os.path.normpath(full_path)
    # Security check
    if not os.path.realpath(full_path).startswith(os.path.realpath(DOCS_DIR)):
        return None
    if not os.path.exists(full_path):
        return None

    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()

    html_body = _simple_markdown(content)
    fname = os.path.basename(file_path)
    mtime = dt.datetime.fromtimestamp(os.path.getmtime(full_path)).strftime('%Y-%m-%d %H:%M')

    view_style = '''
<style>
.doc-content { max-width:860px;margin:0 auto;padding:20px; }
.doc-content h1 { font-size:26px;font-weight:700;color:#e0e8f0;margin:28px 0 16px;border-bottom:1px solid rgba(59,130,246,0.2);padding-bottom:10px; }
.doc-content h2 { font-size:20px;font-weight:600;color:#d0d8e0;margin:24px 0 12px; }
.doc-content h3 { font-size:17px;font-weight:600;color:#c8d0d8;margin:20px 0 10px; }
.doc-content h4 { font-size:15px;font-weight:600;color:#b8c0c8;margin:16px 0 8px; }
.doc-content p { margin:8px 0;line-height:1.7;color:#94a3b8; }
.doc-content a { color:#60a5fa;text-decoration:none; }
.doc-content a:hover { text-decoration:underline; }
.doc-content ul, .doc-content ol { padding-left:24px;margin:8px 0;color:#94a3b8; }
.doc-content li { margin:4px 0; }
.doc-content strong { color:#e0e8f0; }
.doc-content code { background:#1e293b;padding:2px 6px;border-radius:4px;font-size:0.9em;color:#22c55e; }
.doc-content pre { background:#111827;padding:14px 18px;border-radius:10px;overflow-x:auto;font-size:13px;margin:12px 0;border:1px solid rgba(255,255,255,0.06); }
.doc-content table { border-collapse:collapse;width:100%;margin:12px 0; }
.doc-content th, .doc-content td { border:1px solid rgba(255,255,255,0.1);padding:8px 14px;text-align:left;font-size:13px; }
.doc-content th { background:rgba(59,130,246,0.08);font-weight:600;color:#d0d8e0; }
.doc-content td { color:#94a3b8; }
.doc-content em { color:#c8d0d8; }
.doc-content hr { border:none;border-top:1px solid rgba(255,255,255,0.08);margin:20px 0; }
.doc-content blockquote { border-left:3px solid #3b82f6;padding:8px 16px;margin:12px 0;color:#8899aa;background:rgba(59,130,246,0.04);border-radius:0 8px 8px 0; }
</style>'''

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{fname} - EIT 文档</title>
{BASE_STYLE}
{view_style}
</head>
<body><div class="container">
    <div class="navbar">
        <div class="navbar-left">
            <span class="logo-dot"></span>
            <span class="navbar-title">EIT 研究文档</span>
        </div>
        <div>
            <a href="/docs-list/" class="nav-back">&larr; 文档列表</a>
            <a href="/" class="nav-back" style="margin-left:6px;">&larr; 首页</a>
            <span class="navbar-badge" style="margin-left:8px;">{mtime}</span>
        </div>
    </div>

    <div class="section-title">
        <h2>{fname}</h2>
        <div class="line"></div>
    </div>

    <div class="doc-content">
        {html_body}
    </div>
</div></body></html>'''
    return html


# ============ HTTP Handler ============

class DynamicHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # / -> homepage with training card injected
        if path == '/' or path == '/index.html':
            homepage = os.path.join(DOCS_DIR, 'index.html')
            if os.path.exists(homepage):
                with open(homepage, 'rb') as f:
                    html_content = f.read().decode('utf-8')
                html_content = inject_training_card_into_homepage(html_content)
                data = html_content.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', len(data))
                self.end_headers()
                self.wfile.write(data)
            else:
                html = generate_results_html()
                self._send_html(html)
            return

        # /training/ -> overview
        if path == '/training/' or path == '/training':
            html = generate_training_overview()
            self._send_html(html)
            return

        # /training/{run_id}/ -> detail
        if path.startswith('/training/') and path.count('/') >= 2:
            parts = path.rstrip('/').split('/')
            if len(parts) == 3:
                run_id = parts[-1]
                html = generate_training_detail(run_id)
                if html:
                    self._send_html(html)
                    return
                else:
                    self.send_error(404)
                    return

        # /results/ -> EIT results
        if path == '/results/' or path == '/results':
            html = generate_results_html()
            self._send_html(html)
            return

        # /docs-list/ -> 文档浏览器
        if path.startswith('/docs-list'):
            parsed_qs = urlparse(self.path).query if '?' in self.path else ''
            import urllib.parse
            params = urllib.parse.parse_qs(parsed_qs) if parsed_qs else {}
            query = params.get('q', [''])[0]
            sort_by = params.get('sort', ['time'])[0]
            html = generate_docs_list(query=query, sort_by=sort_by)
            self._send_html(html)
            return

        # /docs-view/ -> Markdown 文档查看
        if path.startswith('/docs-view/'):
            parsed_qs = urlparse(self.path).query if '?' in self.path else ''
            import urllib.parse
            params = urllib.parse.parse_qs(parsed_qs) if parsed_qs else {}
            file_path = params.get('file', [''])[0]
            if file_path:
                html = generate_docs_view(file_path)
                if html:
                    self._send_html(html)
                    return
            self.send_error(404)
            return

        # Static files
        local_path = None
        if path.startswith('/results/'):
            local_path = os.path.join(os.path.dirname(__file__), path.lstrip('/'))
        elif path.startswith('/docs/'):
            local_path = os.path.join(os.path.dirname(__file__), path.lstrip('/'))
        else:
            local_path = os.path.join(os.path.dirname(__file__), 'docs', path.lstrip('/'))

        if local_path:
            local_path = os.path.normpath(local_path)

        try:
            file_real = os.path.realpath(local_path)
        except:
            self.send_error(404)
            return

        allowed = False
        for adir in ALLOWED_DIRS:
            if file_real.startswith(adir):
                allowed = True
                break
        if not allowed:
            self.send_error(403)
            return

        if os.path.isdir(local_path):
            idx = os.path.join(local_path, 'index.html')
            if os.path.exists(idx):
                local_path = idx
            else:
                self.send_error(404)
                return

        if os.path.exists(local_path) and os.path.isfile(local_path):
            content_type, _ = mimetypes.guess_type(local_path)
            if content_type is None:
                content_type = 'application/octet-stream'
            with open(local_path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(data))
            if len(data) > 1024 * 1024:
                self.send_header('Cache-Control', 'public, max-age=3600')
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def _send_html(self, html: str):
        data = html.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        timestamp = self.log_date_time_string()
        if len(args) >= 3:
            sys.stderr.write(f"[{timestamp}] {args[0]} {args[1]} {args[2]}\n")
        elif len(args) >= 2:
            sys.stderr.write(f"[{timestamp}] {args[0]} {args[1]}\n")
        else:
            sys.stderr.write(f"[{timestamp}] {' '.join(str(a) for a in args)}\n")


def main():
    parser = argparse.ArgumentParser(description="EIT \u7ED3\u679C\u52A8\u6001\u5C55\u793A\u670D\u52A1\u5668")
    parser.add_argument("--port", type=int, default=8080, help="\u7AEF\u53E3\u53F7 (\u9ED8\u8BA4 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="\u76D1\u542C\u5730\u5740 (\u9ED8\u8BA4 0.0.0.0)")
    args = parser.parse_args()

    print(f"EIT server starting on http://localhost:{args.port}")
    server = HTTPServer((args.host, args.port), DynamicHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.server_close()


if __name__ == "__main__":
    main()
