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

import os, sys, json, glob, argparse, importlib.util, subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """多线程 HTTP 服务器，避免大文件下载阻塞其他请求"""
    daemon_threads = True
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




def fmt_mtime(path):
    """返回文件/目录的最后修改时间字符串"""
    try:
        ts = os.path.getmtime(path)
        return dt.datetime.fromtimestamp(ts).strftime('%m-%d %H:%M')
    except:
        return ""


def scan_results():
    """扫描 results/ 目录，返回所有结果组（含修改时间）"""
    groups = []

    # 1. 子目录
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
                               'label': label, 'size': size,
                               'mtime': fmt_mtime(fpath)})
        metrics, summary_line = load_metrics(dpath)
        dir_mtime = fmt_mtime(dpath)
        groups.append({
            'name': d,
            'path': f'results/{d}/',
            'images': images,
            'metrics': metrics,
            'summary_line': summary_line,
            'is_dir': True,
            'mtime': dir_mtime,
        })

    # 2. 根目录图片
    root_images = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
            fpath = os.path.join(RESULTS_DIR, fname)
            size = os.path.getsize(fpath)
            label = fname.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
            root_images.append({'name': fname, 'path': f'results/{fname}',
                                'label': label, 'size': size,
                                'mtime': fmt_mtime(fpath)})

    if root_images:
        groups.insert(0, {
            'name': '根目录图像',
            'path': 'results/',
            'images': root_images,
            'metrics': {},
            'summary_line': "",
            'is_dir': False,
            'mtime': "",
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
    """生成完整 HTML 页面，支持前端排序 + 灯箱画廊"""
    groups = scan_results()

    total_png = sum(len(g['images']) for g in groups)
    total_dirs = sum(1 for g in groups if g['is_dir'])

    # 构建每个组的卡片（嵌入 data- 属性用于前端排序 + 画廊）
    cards_html = ""
    for g in groups:
        metrics_html = format_metrics_html(g['metrics'], g['summary_line'])

        cover_path = g['images'][0]['path'] if g['images'] else ""

        n_img = len(g['images'])
        desc = f"{n_img} 张图像"
        if g['is_dir']:
            desc += f" · 子目录 {g['name']}"
        else:
            desc += " · results/ 根目录"

        mtime_display = ""
        if g.get('mtime'):
            mtime_display = f'<div class="mtime">🕐 {g["mtime"]}</div>'
        is_v2 = 'v2' in g['name'].lower() or 'best' in g['name'].lower()
        card_cls = 'card' + (' v2' if is_v2 else '')
        sort_time = g.get('mtime', '')
        sort_name = g['name'].lower()

        # 所有图片路径作为 JSON 嵌入 data-images
        img_list = [img['path'] for img in g['images']]
        import json
        images_json = json.dumps(img_list, ensure_ascii=False)

        card_attrs = f'data-sort-time="{sort_time}" data-sort-name="{sort_name}" data-images=\'{images_json}\''
        card_title = g['name']

        if cover_path:
            card = f'''
        <div class="{card_cls}" {card_attrs}>
            <div class="img-wrap" onclick="openGallery(this.parentElement)"><img src="/{cover_path}" alt="{g['name']}" loading="lazy"></div>
            <div class="info" onclick="openGallery(this.parentElement)">
                <h3>{'🚀 ' if is_v2 else ''}{g['name']}</h3>
                <p>{desc}</p>
                <div class="metrics-row">{metrics_html}</div>
                {mtime_display}
            </div>
        </div>'''
        else:
            card = f'''
        <div class="{card_cls}" {card_attrs}>
            <div class="img-wrap" style="display:flex;align-items:center;justify-content:center;background:#0a0e17;" onclick="openGallery(this.parentElement)">
                <span style="font-size:48px;opacity:0.5;">📁</span>
            </div>
            <div class="info" onclick="openGallery(this.parentElement)">
                <h3>{g['name']}</h3>
                <p>{desc}</p>
                <div class="metrics-row">{metrics_html}</div>
                {mtime_display}
            </div>
        </div>'''
        cards_html += card

    now_str = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')

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
.sort-bar {{ display: flex; align-items: center; gap: 8px; }}
.sort-btn {{
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 12px; border-radius: 8px; cursor: pointer;
    font-size: 12px; color: #667788; background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06); transition: all 0.2s;
    user-select: none;
}}
.sort-btn:hover {{ background: rgba(59,130,246,0.12); color: #94a3b8; }}
.sort-btn.active {{ background: rgba(59,130,246,0.15); color: #60a5fa; border-color: rgba(59,130,246,0.3); }}

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
.mtime {{ font-size: 11px; color: #556677; margin-top: 6px; }}
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
            <div class="refresh-info">自动扫描 results/ 目录 · 共 {total_png} 张图片 / {total_dirs} 个子目录</div>
        </div>
        <div class="sort-bar">
            <span class="sort-btn active" data-sort="time-desc" onclick="sortGallery('time-desc')">🕐 时间 ↓</span>
            <span class="sort-btn" data-sort="time-asc" onclick="sortGallery('time-asc')">🕐 时间 ↑</span>
            <span class="sort-btn" data-sort="name" onclick="sortGallery('name')">📋 名称</span>
            <span class="count" style="margin-left:4px;">{now_str}</span>
        </div>
    </div>

    <div class="gallery" id="gallery">
        {cards_html}
    </div>

    <div style="text-align:center;margin-top:40px;padding:20px;border-top:1px solid rgba(59,130,246,0.1);color:#556677;font-size:12px;">
        💡 把新结果图片放到 <code>results/</code> 目录下，刷新页面即可看到<br>
        子目录会自动识别为独立的结果组 · 含 <code>metrics.json</code> 的子目录会显示指标
    </div>
</div>
<script>
function sortGallery(mode) {{
    var gallery = document.getElementById('gallery');
    var cards = Array.from(gallery.children);
    var btn = document.querySelector('.sort-btn[data-sort="'+mode+'"]');
    if (!btn) return;
    document.querySelectorAll('.sort-btn').forEach(function(b) {{ b.classList.remove('active'); }});
    btn.classList.add('active');

    cards.sort(function(a, b) {{
        if (mode === 'name') {{
            return a.getAttribute('data-sort-name').localeCompare(b.getAttribute('data-sort-name'));
        }} else {{
            var ta = a.getAttribute('data-sort-time') || '';
            var tb = b.getAttribute('data-sort-time') || '';
            if (ta === tb) return 0;
            if (ta === '') return 1;
            if (tb === '') return -1;
            var cmp = ta.localeCompare(tb);
            return mode === 'time-desc' ? -cmp : cmp;
        }}
    }});

    cards.forEach(function(card) {{ gallery.appendChild(card); }});
}}
</script>
<!-- 灯箱 -->
<style>
.gal-overlay {{ display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.92);z-index:9999;cursor:default; }}
.gal-overlay.active {{ display:block; }}
.gal-img-wrap {{ position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center; }}
.gal-img {{ max-width:90vw;max-height:80vh;object-fit:contain;transition:transform 0.2s;user-select:none;border-radius:4px; }}
.gal-toolbar {{ display:flex;gap:6px;margin-top:12px;background:rgba(15,23,42,0.9);padding:8px 12px;border-radius:10px;border:1px solid rgba(255,255,255,0.1); }}
.gal-toolbar button {{ background:none;border:1px solid rgba(255,255,255,0.15);color:#94a3b8;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:14px; }}
.gal-toolbar button:hover {{ background:rgba(255,255,255,0.05); }}
.gal-counter {{ color:#94a3b8;font-size:13px;padding:4px 10px; }}
.gal-nav {{ position:fixed;top:50%;transform:translateY(-50%);font-size:40px;color:rgba(255,255,255,0.3);cursor:pointer;padding:20px 16px;user-select:none;transition:all 0.2s;z-index:10000; }}
.gal-nav:hover {{ color:rgba(255,255,255,0.8);background:rgba(255,255,255,0.05); }}
.gal-prev {{ left:10px; }}
.gal-next {{ right:10px; }}
.gal-close {{ position:fixed;top:12px;right:20px;font-size:32px;color:rgba(255,255,255,0.4);cursor:pointer;z-index:10000;transition:all 0.2s; }}
.gal-close:hover {{ color:#ef4444;transform:scale(1.15); }}
</style>
<div id="gallery-overlay" class="gal-overlay" onclick="closeGallery(event)">
    <span class="gal-nav gal-prev" onclick="navGallery(-1)">&#10094;</span>
    <span class="gal-nav gal-next" onclick="navGallery(1)">&#10095;</span>
    <span class="gal-close" onclick="closeGallery(event)">&times;</span>
    <div class="gal-img-wrap">
        <img id="gal-img" class="gal-img" src="" alt="">
        <div class="gal-toolbar">
            <span id="gal-counter" class="gal-counter"></span>
            <button onclick="galZoom(-0.2)">−</button>
            <button onclick="galZoom(0.2)">+</button>
            <button onclick="galRotate(-90)">&#8634;</button>
            <button onclick="galRotate(90)">&#8635;</button>
            <button onclick="galReset()" style="color:#f59e0b">R</button>
        </div>
    </div>
</div>
<script>
var _gal = {{ idx: 0, images: [], scale: 1, rot: 0 }};
function openGallery(card) {{
    try {{ _gal.images = JSON.parse(card.getAttribute('data-images') || '[]'); }} catch(e) {{ _gal.images = []; }}
    if (_gal.images.length === 0) return;
    _gal.idx = 0; _gal.scale = 1; _gal.rot = 0;
    showGalImage();
    document.getElementById('gallery-overlay').classList.add('active');
    document.body.style.overflow = 'hidden';
}}
function closeGallery(ev) {{
    if (ev && ev.target && ev.target.id !== 'gallery-overlay' && !ev.target.classList.contains('gal-close')) return;
    document.getElementById('gallery-overlay').classList.remove('active');
    document.body.style.overflow = '';
}}
function showGalImage() {{
    var img = document.getElementById('gal-img');
    if (_gal.images.length === 0) return;
    img.src = '/' + _gal.images[_gal.idx];
    document.getElementById('gal-counter').textContent = (_gal.idx + 1) + ' / ' + _gal.images.length;
    galReset();
}}
function navGallery(d) {{
    if (_gal.images.length === 0) return;
    _gal.idx = (_gal.idx + d + _gal.images.length) % _gal.images.length;
    showGalImage();
}}
function galZoom(d) {{ _gal.scale = Math.max(0.2, Math.min(5, _gal.scale + d)); applyGalTransform(); }}
function galRotate(d) {{ _gal.rot = (_gal.rot + d) % 360; applyGalTransform(); }}
function galReset() {{ _gal.scale = 1; _gal.rot = 0; applyGalTransform(); }}
function applyGalTransform() {{ document.getElementById('gal-img').style.transform = 'scale(' + _gal.scale + ') rotate(' + _gal.rot + 'deg)'; }}
document.addEventListener('keydown', function(ev) {{
    if (!document.getElementById('gallery-overlay').classList.contains('active')) return;
    if (ev.key === 'Escape') closeGallery(ev);
    if (ev.key === 'ArrowLeft') navGallery(-1);
    if (ev.key === 'ArrowRight') navGallery(1);
    if (ev.key === '+' || ev.key === '=') galZoom(0.2);
    if (ev.key === '-') galZoom(-0.2);
    if (ev.key === 'r') galReset();
}});
</script>
</body></html>'''
    return html


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
    diff_epochs = [e for e in (data.get("epochs") or []) if e["phase"] == "diffusion"]

    # 从 meta 或 epochs 中获取最佳 RE
    all_epochs = sup_epochs + unsup_epochs + diff_epochs
    best_re = meta.get("best_re")
    if best_re is None and all_epochs:
        re_vals = [e.get("re") for e in all_epochs if e.get("re") is not None]
        best_re = f"{min(re_vals):.4f}" if re_vals else "—"
    elif best_re is not None:
        best_re = f"{best_re:.4f}" if isinstance(best_re, float) else str(best_re)
    else:
        best_re = "—"
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
<link rel="icon" href="/docs/favicon.svg" type="image/svg+xml">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; background: #0a0e17; color: #e0e8f0; min-height: 100vh; line-height: 1.6; }
body::before { content: ""; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.06) 0%, transparent 50%), radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.06) 0%, transparent 50%), radial-gradient(ellipse at 50% 80%, rgba(6,182,212,0.04) 0%, transparent 50%); pointer-events: none; z-index: 0; }
body::after { content: ""; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background-image: linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px); background-size: 60px 60px; pointer-events: none; z-index: 0; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: rgba(15,23,42,0.6); }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(59,130,246,0.4); }
::-webkit-scrollbar-corner { background: transparent; }
/* Firefox */
* { scrollbar-width: thin; scrollbar-color: rgba(59,130,246,0.2) rgba(15,23,42,0.6); }
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
.back-to-top { position:fixed;bottom:24px;right:24px;width:40px;height:40px;background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);border-radius:50%;color:#60a5fa;font-size:20px;cursor:pointer;z-index:999;display:none;align-items:center;justify-content:center;transition:all 0.3s; }
.back-to-top:hover { background:rgba(59,130,246,0.3);transform:translateY(-2px); }
.lb-overlay { display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.92);z-index:9999; }
.lb-overlay.active { display:block; }
.lb-img { position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);max-width:90vw;max-height:85vh;object-fit:contain;transition:transform 0.2s;user-select:none; }
.lb-toolbar { position:fixed;bottom:20px;left:50%;transform:translateX(-50%);display:flex;gap:6px;z-index:10000;background:rgba(15,23,42,0.9);padding:10px 14px;border-radius:12px;border:1px solid rgba(255,255,255,0.1); }
.lb-toolbar button { background:none;border:1px solid rgba(255,255,255,0.15);color:#94a3b8;padding:6px 10px;border-radius:8px;cursor:pointer;font-size:15px;min-width:32px; }
.lb-toolbar button:hover { background:rgba(255,255,255,0.05); }
.lb-close { background:rgba(239,68,68,0.1)!important;border-color:rgba(239,68,68,0.3)!important;color:#ef4444!important;padding:6px 14px!important; }
.lb-info { position:fixed;top:16px;left:50%;transform:translateX(-50%);color:#94a3b8;font-size:13px;background:rgba(15,23,42,0.8);padding:6px 14px;border-radius:8px;z-index:10000; }
</style>
<script>
(function(){
if(document.getElementById('lb-inline'))return;
var s=document.createElement('script');s.id='lb-inline';
s.textContent='var _lb={s:1,r:0,fx:1,fy:1};\nvar e=document.createElement("div");e.id="lb";e.className="lb-overlay";e.innerHTML='+"'"+'<img class="lb-img" id="lbimg"><div class="lb-info" id="lbinfo"></div><div class="lb-toolbar"><button onclick="_lbz(-0.2)">−</button><button onclick="_lbz(0.2)">+</button><button onclick="_lbr(-90)">↺</button><button onclick="_lbr(90)">↻</button><button onclick="_lbfx()">↔</button><button onclick="_lbfy()">↕</button><button onclick="_lbreset()" style="color:#f59e0b">R</button><button class="lb-close" onclick="_lbclose()">✕</button></div>'+"'"+";document.body.appendChild(e);\nwindow._lbopen=function(src,title){document.getElementById(\"lbimg\").src=src;document.getElementById(\"lb\").classList.add(\"active\");document.getElementById(\"lbinfo\").textContent=title||\"\";document.body.style.overflow=\"hidden\";_lbreset();};\nwindow._lbclose=function(){document.getElementById(\"lb\").classList.remove(\"active\");document.body.style.overflow=\"\";};\nfunction _lbup(){var i=document.getElementById(\"lbimg\");i.style.transform=\"translate(-50%,-50%) scale(\"+_lb.s+\") rotate(\"+_lb.r+\"deg) scaleX(\"+_lb.fx+\") scaleY(\"+_lb.fy+\")\";}\nfunction _lbz(d){_lb.s=Math.max(0.2,Math.min(5,_lb.s+d));_lbup();}\nfunction _lbr(d){_lb.r=(_lb.r+d)%360;_lbup();}\nfunction _lbfx(){_lb.fx*=-1;_lbup();}\nfunction _lbfy(){_lb.fy*=-1;_lbup();}\nfunction _lbreset(){_lb.s=1;_lb.r=0;_lb.fx=1;_lb.fy=1;_lbup();}\ndocument.getElementById(\"lb\").addEventListener(\"click\",function(ev){if(ev.target===this)_lbclose();});\ndocument.getElementById(\"lb\").addEventListener(\"wheel\",function(ev){ev.preventDefault();_lbz(ev.deltaY<0?0.1:-0.1);},{passive:false});\ndocument.addEventListener(\"keydown\",function(ev){if(!document.getElementById(\"lb\").classList.contains(\"active\"))return;if(ev.key===\"Escape\")_lbclose();if(ev.key===\"+\"||ev.key===\"=\")_lbz(0.2);if(ev.key===\"-\")_lbz(-0.2);if(ev.key===\"r\")_lbreset();});\ndocument.addEventListener(\"click\",function(ev){var a=ev.target.closest(\"a\");if(!a)return;var h=a.getAttribute(\"href\");if(!h)return;if(/\\.(png|jpg|jpeg|gif|svg|webp)(\\?|$)/i.test(h)){ev.preventDefault();var t=a.querySelector(\"h4,h3\")||a.querySelector(\"[class*=\\\"title\\\"]\");_lbopen(h,t?t.textContent:\"\");}});\nvar bt=document.createElement(\"div\");bt.className=\"back-to-top\";bt.innerHTML=\"↑\";bt.title=\"\u56de\u5230\u9876\u90e8\";bt.addEventListener(\"click\",function(){window.scrollTo({top:0,behavior:\"smooth\"});});document.body.appendChild(bt);window.addEventListener(\"scroll\",function(){bt.style.display=window.scrollY>400?\"flex\":\"none\";});';
document.head.appendChild(s);
})();
</script>'''





def _compute_best_re(meta: dict, epochs: list) -> str:
    """从 meta 或 epochs 数据中计算最佳 RE，返回格式化字符串"""
    best = meta.get("best_re")
    if best is not None:
        return f"{best:.4f}" if isinstance(best, float) else str(best)
    if epochs:
        re_vals = [e.get("re") for e in epochs if e.get("re") is not None]
        if re_vals:
            return f"{min(re_vals):.4f}"
    return "—"


def generate_training_overview() -> str:
    """生成训练总览页 /training/，支持多选删除"""
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
        diff_epochs = [e for e in (data.get("epochs") or []) if e["phase"] == "diffusion"]

        status = r.get("status", "completed")
        status_cls = status
        status_icon = {"completed": "✅", "running": "🔴", "failed": "❌"}.get(status, "⚪")

        best_re = _compute_best_re(meta, data.get("epochs") or [])
        params = meta.get("model_params", 0)
        hidden = meta.get("hidden_dim", "—")
        n_sup = len(sup_epochs)
        n_unsup = len(unsup_epochs)
        n_diff = len(diff_epochs)
        start = r.get("start_time", "")[:16].replace("T", " ")

        # RE/Loss 微型曲线 — 优先用 diff 的 loss
        re_values = [e.get("re") for e in sup_epochs if e.get("re") is not None]
        if not re_values:
            re_values = [e.get("val_loss") for e in diff_epochs if e.get("val_loss") is not None]
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
    <div class="run-card {status_cls}" data-run-id="{run_id}">
        <div class="run-card-checkbox" onclick="event.stopPropagation()">
            <input type="checkbox" class="run-select" value="{run_id}" id="sel_{run_id}">
        </div>
        <div class="run-card-body" onclick="location.href='/training/{run_id}/'">
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
                <span class="run-meta-item"><span class="run-meta-label">扩散</span> {n_diff} ep</span>
            </div>
            {f'<div class="run-chart">{chart_svg}</div>' if chart_svg else ''}
        </div>
    </div>'''

    if not cards:
        cards = '<div style="text-align:center;padding:60px;color:#667788;">暂无训练记录</div>'

    toolbar_style = '''
.tr-toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; padding: 10px 16px; background: rgba(15,23,42,0.8); border: 1px solid rgba(59,130,246,0.12); border-radius: 10px; backdrop-filter: blur(10px); flex-wrap: wrap; }
.tr-toolbar label { display: flex; align-items: center; gap: 6px; font-size: 13px; color: #94a3b8; cursor: pointer; }
.tr-toolbar .tr-count { font-size: 12px; color: #556677; margin-left: auto; }
.tr-del-btn { padding: 6px 16px; border-radius: 8px; font-size: 13px; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 4px; background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.25); color: #ef4444; }
.tr-del-btn:disabled { opacity: 0.35; cursor: not-allowed; }
.tr-del-btn:hover:not(:disabled) { background: rgba(239,68,68,0.25); }
.run-card { display: flex; align-items: flex-start; gap: 0; padding: 0; overflow: hidden; }
.run-card-checkbox { display: flex; align-items: center; justify-content: center; min-width: 40px; height: 100%; padding: 16px 0 16px 12px; cursor: pointer; }
.run-card-checkbox input[type="checkbox"] { width: 16px; height: 16px; cursor: pointer; accent-color: #3b82f6; }
.run-card-body { flex: 1; padding: 16px 20px; cursor: pointer; }
/* 选中高亮 */
.run-card.selected { border-color: rgba(59,130,246,0.5); background: rgba(59,130,246,0.08); }
.confirm-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 9999; align-items: center; justify-content: center; }
.confirm-overlay.active { display: flex; }
.confirm-dialog { background: rgba(15,23,42,0.95); border: 1px solid rgba(239,68,68,0.3); border-radius: 14px; padding: 24px 32px; max-width: 420px; width: 90%; backdrop-filter: blur(10px); }
.confirm-dialog h3 { font-size: 16px; color: #ef4444; margin-bottom: 8px; }
.confirm-dialog p { font-size: 13px; color: #94a3b8; margin-bottom: 6px; line-height: 1.5; }
.confirm-dialog .confirm-ids { font-size: 11px; color: #667788; background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 8px; margin-bottom: 16px; max-height: 120px; overflow-y: auto; word-break: break-all; }
.confirm-dialog .confirm-btns { display: flex; gap: 10px; justify-content: flex-end; }
.confirm-dialog .btn-cancel { padding: 8px 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); background: transparent; color: #94a3b8; cursor: pointer; font-size: 13px; }
.confirm-dialog .btn-confirm { padding: 8px 20px; border-radius: 8px; border: none; background: rgba(239,68,68,0.8); color: #fff; cursor: pointer; font-size: 13px; font-weight: 600; }
.confirm-dialog .btn-confirm:hover { background: #ef4444; }
'''

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>训练记录 - EIT</title>
{BASE_STYLE}
<style>
/* 多选删除样式 */
.tr-toolbar {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; padding: 10px 16px; background: rgba(15,23,42,0.8); border: 1px solid rgba(59,130,246,0.12); border-radius: 10px; backdrop-filter: blur(10px); flex-wrap: wrap; }}
.tr-toolbar label {{ display: flex; align-items: center; gap: 6px; font-size: 13px; color: #94a3b8; cursor: pointer; }}
.tr-toolbar .tr-count {{ font-size: 12px; color: #556677; margin-left: auto; }}
.tr-del-btn {{ padding: 6px 16px; border-radius: 8px; font-size: 13px; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 4px; background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.25); color: #ef4444; }}
.tr-del-btn:disabled {{ opacity: 0.3; cursor: not-allowed; background: rgba(100,100,100,0.08); border-color: rgba(100,100,100,0.15); color: #666; }}
.tr-del-btn:not(:disabled) {{ box-shadow: 0 0 8px rgba(239,68,68,0.15); }}
.tr-del-btn:hover:not(:disabled) {{ background: rgba(239,68,68,0.25); }}
.run-card {{ display: flex; align-items: flex-start; gap: 0; padding: 0; overflow: hidden; }}
.run-card-checkbox {{ display: flex; align-items: center; justify-content: center; min-width: 40px; height: 100%; padding: 16px 0 16px 12px; cursor: pointer; flex-shrink: 0; }}
.run-card-checkbox input[type="checkbox"] {{ width: 16px; height: 16px; cursor: pointer; accent-color: #3b82f6; }}
.run-card-body {{ flex: 1; min-width: 0; padding: 16px 20px; cursor: pointer; }}
.run-card.selected {{ border-color: rgba(59,130,246,0.5) !important; background: rgba(59,130,246,0.08) !important; }}
.confirm-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 9999; align-items: center; justify-content: center; }}
.confirm-overlay.active {{ display: flex; }}
.confirm-dialog {{ background: rgba(15,23,42,0.95); border: 1px solid rgba(239,68,68,0.3); border-radius: 14px; padding: 24px 32px; max-width: 420px; width: 90%; backdrop-filter: blur(10px); }}
.confirm-dialog h3 {{ font-size: 16px; color: #ef4444; margin-bottom: 8px; }}
.confirm-dialog p {{ font-size: 13px; color: #94a3b8; margin-bottom: 6px; line-height: 1.5; }}
.confirm-dialog .confirm-ids {{ font-size: 11px; color: #667788; background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 8px; margin-bottom: 16px; max-height: 120px; overflow-y: auto; word-break: break-all; }}
.confirm-dialog .confirm-btns {{ display: flex; gap: 10px; justify-content: flex-end; }}
.confirm-dialog .btn-cancel {{ padding: 8px 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); background: transparent; color: #94a3b8; cursor: pointer; font-size: 13px; }}
.confirm-dialog .btn-cancel:hover {{ background: rgba(255,255,255,0.05); }}
.confirm-dialog .btn-confirm {{ padding: 8px 20px; border-radius: 8px; border: none; background: rgba(239,68,68,0.8); color: #fff; cursor: pointer; font-size: 13px; font-weight: 600; }}
.confirm-dialog .btn-confirm:hover {{ background: #ef4444; }}
</style>
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

    <div class="tr-toolbar">
        <label><input type="checkbox" id="selectAll" onchange="toggleSelectAll(this)"> 全选</label>
        <button class="tr-del-btn" id="delBtn" onclick="confirmDelete()" disabled>&#128465; 删除所选</button>
        <span class="tr-count" id="countInfo">共 {len(runs)} 条记录</span>
    </div>

    <div id="cardList">
        {cards}
    </div>

    <div style="text-align:center;padding:30px;color:#556677;font-size:12px;">
        &#128161; 勾选记录后点击"删除所选"批量清理 · 点击卡片查看详情
    </div>
</div>

<!-- 确认对话框 -->
<div class="confirm-overlay" id="confirmOverlay">
    <div class="confirm-dialog">
        <h3>&#9888;&#65039; 确认删除</h3>
        <p>以下训练记录将被永久删除，此操作不可撤销：</p>
        <div class="confirm-ids" id="confirmIds"></div>
        <div class="confirm-btns">
            <button class="btn-cancel" onclick="closeConfirm()">取消</button>
            <button class="btn-confirm" onclick="doDelete()">确认删除</button>
        </div>
    </div>
</div>

<script>
var _pendingDelete = [];

document.querySelectorAll('.run-select').forEach(function(cb) {{
    cb.addEventListener('change', function() {{
        var card = this.closest('.run-card');
        if (this.checked) {{
            card.classList.add('selected');
        }} else {{
            card.classList.remove('selected');
        }}
        updateUI();
    }});
}});

function toggleSelectAll(sel) {{
    document.querySelectorAll('.run-select').forEach(function(cb) {{
        cb.checked = sel.checked;
        var card = cb.closest('.run-card');
        if (sel.checked) {{
            card.classList.add('selected');
        }} else {{
            card.classList.remove('selected');
        }}
    }});
    updateUI();
}}

function updateUI() {{
    var checked = document.querySelectorAll('.run-select:checked');
    var btn = document.getElementById('delBtn');
    var info = document.getElementById('countInfo');
    if (checked.length > 0) {{
        btn.textContent = '[\u2716] 删除所选 (' + checked.length + ')';
        btn.disabled = false;
        info.textContent = '已选 ' + checked.length + ' / 共 ' + document.querySelectorAll('.run-select').length + ' 条';
    }} else {{
        btn.disabled = true;
        info.textContent = '共 ' + document.querySelectorAll('.run-select').length + ' 条记录';
    }}
}}

function confirmDelete() {{
    var checked = document.querySelectorAll('.run-select:checked');
    if (checked.length === 0) return;
    _pendingDelete = Array.from(checked).map(function(cb) {{ return cb.value; }});
    var names = _pendingDelete.join('\n');
    document.getElementById('confirmIds').textContent = names;
    document.getElementById('confirmOverlay').classList.add('active');
}}

function closeConfirm() {{
    document.getElementById('confirmOverlay').classList.remove('active');
    _pendingDelete = [];
}}

function doDelete() {{
    if (_pendingDelete.length === 0) return;
    fetch('/api/training/delete', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ run_ids: _pendingDelete }})
    }})
    .then(function(r) {{ return r.json(); }})
    .then(function(data) {{
        if (data.success) {{
            location.reload();
        }} else {{
            alert('\u5220\u9664\u5931\u8D25: ' + (data.error || '\u672A\u77E5\u9519\u8BEF'));
            closeConfirm();
        }}
    }})
    .catch(function(err) {{
        alert('\u7F51\u7EDC\u9519\u8BEF: ' + err);
        closeConfirm();
    }});
}}

// ESC 关闭确认框
document.addEventListener('keydown', function(ev) {{
    if (ev.key === 'Escape' && document.getElementById('confirmOverlay').classList.contains('active')) {{
        closeConfirm();
    }}
}});
</script>
</body></html>'''
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
    diff_epochs = [e for e in epochs if e["phase"] == "diffusion"]

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
        # Chart dimensions
        margin_l, margin_r, margin_t, margin_b = 55, 10, 10, 28
        cw, ch = 500, 150  # chart area
        sw, sh = cw + margin_l + margin_r, ch + margin_t + margin_b
        
        min_v, max_v = min(values), max(values)
        rng = max(max_v - min_v, 1e-8)
        
        # Y-axis: compute nice ticks
        y_ticks = []
        for i in range(5):
            tick_val = min_v + rng * i / 4.0
            y_ticks.append(tick_val)
        
        # Data points
        pts = []
        for i, v in enumerate(values):
            x = margin_l + i / max(n - 1, 1) * cw
            y = margin_t + ch - (v - min_v) / rng * ch
            pts.append(f"{x:.1f},{y:.1f}")
        poly = " ".join(pts)
        
        # Grid lines + Y labels
        grid = ""
        for i, tv in enumerate(y_ticks):
            gy = margin_t + ch - (i / 4.0) * ch
            grid += f'<line x1="{margin_l}" y1="{gy:.0f}" x2="{margin_l+cw}" y2="{gy:.0f}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>'
            grid += f'<text x="{margin_l-6}" y="{gy+4:.0f}" fill="#667788" font-size="10" text-anchor="end">{tv:.4f}</text>'
        
        # X-axis labels
        x_ticks_n = min(6, n)
        for i in range(x_ticks_n):
            epoch_i = int(i * (n - 1) / max(x_ticks_n - 1, 1))
            xx = margin_l + epoch_i / max(n - 1, 1) * cw
            grid += f'<text x="{xx:.0f}" y="{sh-6:.0f}" fill="#667788" font-size="10" text-anchor="middle">{epoch_i+1}</text>'
        
        # Axes
        grid += f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+ch}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>'
        grid += f'<line x1="{margin_l}" y1="{margin_t+ch}" x2="{margin_l+cw}" y2="{margin_t+ch}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>'
        
        # Axis labels
        grid += f'<text x="{sw/2:.0f}" y="{sh-2:.0f}" fill="#556677" font-size="11" text-anchor="middle">Epoch</text>'
        grid += f'<text x="12" y="{sh/2:.0f}" fill="#556677" font-size="11" text-anchor="middle" transform="rotate(-90,12,{sh/2:.0f})">{label}</text>'
        
        # Value label at end of curve
        grid += f'<text x="{margin_l+cw-5}" y="{margin_t+15}" fill="{color}" font-size="10" text-anchor="end">{values[-1]:.4f}</text>'
        
        return f'''<svg width="100%" viewBox="0 0 {sw} {sh}" style="max-width:{sw}px;background:rgba(0,0,0,0.1);border-radius:8px;">
          {grid}
          <polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>'''

    chart_sup = make_chart(sup_epochs, "re", "#22c55e", "RE (\u6709\u76D1\u7763)")
    loss_sup = make_chart(sup_epochs, "loss", "#3b82f6", "Loss (\u6709\u76D1\u7763)")
    chart_unsup = make_chart(unsup_epochs, "loss", "#f59e0b", "Loss (\u65E0\u76D1\u7763)")
    loss_diff = make_chart(diff_epochs, "val_loss", "#a855f7", "Val Loss (\u6269\u6563)")
    train_diff = make_chart(diff_epochs, "loss", "#8b5cf6", "Train Loss (\u6269\u6563)")

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
            <div><div style="font-size:12px;color:#a78bfa;margin-bottom:4px;">\u6269\u6563 Val Loss</div>{loss_diff}</div>
            <div><div style="font-size:12px;color:#8b5cf6;margin-bottom:4px;">\u6269\u6563 Train Loss</div>{train_diff}</div>
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
        is_header = False
        line = re.sub(r'^#### (.+)', r'<h4>\1</h4>', line)
        line = re.sub(r'^### (.+)', r'<h3>\1</h3>', line)
        line = re.sub(r'^## (.+)', r'<h2>\1</h2>', line)
        line = re.sub(r'^# (.+)', r'<h1>\1</h1>', line)
        if line != orig_line and line.startswith('<h'):
            is_header = True

        # Blockquote
        is_bq = line.startswith('> ')
        if is_bq:
            line = re.sub(r'^> (.+)', r'<blockquote style="border-left:3px solid #3b82f6;padding:8px 16px;margin:12px 0;color:#8899aa;background:rgba(59,130,246,0.04);border-radius:0 8px 8px 0;">\1</blockquote>', line)

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
            # 检查是否已在 <ul> 中（从后往前找）
            in_list = False
            for j in range(len(out)-1, -1, -1):
                if out[j].startswith('<ul'):
                    in_list = True
                    break
                elif out[j].startswith('</ul') or out[j].startswith('<ol') or out[j].startswith('</ol'):
                    break
            if not in_list:
                out.append('<ul style="padding-left:20px;">')
            out.append(line)
            i += 1
            continue
        elif re.match(r'^\s*\d+\.\s', line):
            line = re.sub(r'^\s*\d+\.\s+(.+)', r'<li>\1</li>', line)
            in_list = False
            for j in range(len(out)-1, -1, -1):
                if out[j].startswith('<ol'):
                    in_list = True
                    break
                elif out[j].startswith('</ol') or out[j].startswith('<ul') or out[j].startswith('</ul'):
                    break
            if not in_list:
                out.append('<ol style="padding-left:20px;">')
            out.append(line)
            i += 1
            continue
        else:
            # 关闭所有打开的列表（但不删除其中的 <li> 项）
            if out and (out[-1].startswith('<li')):
                # 找到最近的开列表标签并关闭
                for j in range(len(out)-1, -1, -1):
                    if out[j].startswith('<ul') or out[j].startswith('<ol'):
                        out.append('</ul>' if out[j].startswith('<ul') else '</ol>')
                        break

        # Paragraph — skip self-closing tags (headers, hrs, blockquotes, pre, tables)
        if line.strip():
            if is_header or is_bq or line.startswith('<hr') or line.startswith('<pre') or line.startswith('<table') or line.startswith('<blockquote'):
                out.append(line)
            else:
                out.append(f'<p style="margin:8px 0;">{line}</p>')
        else:
            out.append('<br>')
        i += 1

    if in_table:
        out.append('</table>')
    if in_code:
        out.append('</pre>')
    # 清理文档末尾未关闭的列表
    if out and out[-1].startswith('<li'):
        for j in range(len(out)-1, -1, -1):
            if out[j].startswith('<ul') or out[j].startswith('<ol'):
                out.append('</ul>' if out[j].startswith('<ul') else '</ol>')
                break
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
        <div class="doc-card-row">
        <a href="{href}"{target} class="doc-card-link">
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
        </a>
        <a href="{f['raw_url']}?dl=1" class="doc-dl-btn" title="下载">⬇</a>
        </div>'''

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
.doc-card-row { display:flex;align-items:center;gap:0;margin-bottom:6px; }
.doc-card-link { flex:1;min-width:0;display:flex;align-items:center;gap:14px;padding:14px 18px;text-decoration:none;background:rgba(15,23,42,0.6);border:1px solid rgba(59,130,246,0.06);border-radius:12px 0 0 12px;transition:all 0.2s; }
.doc-card-link:hover { background:rgba(15,23,42,0.9);border-color:rgba(59,130,246,0.25); }
.doc-card-icon { font-size:24px;flex-shrink:0;width:36px;text-align:center; }
.doc-card-info { flex:1;min-width:0; }
.doc-card-name { font-size:14px;font-weight:600;color:#d0d8e0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.doc-card-path { font-size:11px;color:#556677;margin-top:2px; }
.doc-card-meta { display:flex;gap:12px;align-items:center;flex-shrink:0;font-size:11px;color:#667788; }
.doc-type { padding:2px 8px;border-radius:6px;background:rgba(59,130,246,0.08);color:#60a5fa; }
.doc-size { color:#556677; }
.doc-time { color:#445566;min-width:80px;text-align:right; }
.doc-dl-btn { flex-shrink:0;display:flex;align-items:center;justify-content:center;width:44px;align-self:stretch;border-radius:0 12px 12px 0;background:rgba(59,130,246,0.06);color:#60a5fa;text-decoration:none;font-size:16px;transition:all 0.2s;border:1px solid rgba(59,130,246,0.06);border-left:none; }
.doc-dl-btn:hover { background:rgba(59,130,246,0.15);color:#93c5fd;border-color:rgba(59,130,246,0.25); }
.doc-card-row:hover .doc-card-link { background:rgba(15,23,42,0.9);border-color:rgba(59,130,246,0.25); }
.doc-card-row:hover .doc-dl-btn { border-color:rgba(59,130,246,0.25);background:rgba(59,130,246,0.1); }
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
            <a href="/docs/{file_path}?dl=1" class="nav-back" style="margin-left:6px;background:rgba(59,130,246,0.1);">⬇ 下载</a>
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


def generate_datasets_page() -> str:
    """生成训练数据集预览页"""
    shapes_info = [
        {'name': '根系 (root)', 'key': 'root', 'desc': '植物根系结构：直根型/须根型/鱼骨型，平均5.3%覆盖率'},
        {'name': '单圆 (circle)', 'key': 'circle', 'desc': '随机位置 + 随机半径 1.2~3.2cm，25%占比'},
        {'name': '椭圆 (ellipse)', 'key': 'ellipse', 'desc': '随机长/短轴 + 随机旋转角度，25%占比'},
        {'name': '双圆 (double_circle)', 'key': 'double_circle', 'desc': '两个不重叠的圆，25%占比'},
        {'name': '菱形 (diamond)', 'key': 'diamond', 'desc': '旋转菱形，随机拉伸比例，25%占比'},
    ]
    
    cards = ""
    for si in shapes_info:
        key = si['key']
        if key == 'root':
            # 根系数据集有专门的 detail 和 samples 图
            detail_img = f"/results/dataset_preview/{key}_detail.png"
            samples_img = f"/results/dataset_preview/{key}_samples.png"
        else:
            # 通用形状数据集使用 shapes_preview 中的预览图
            detail_img = f"/results/shapes_preview/preview_{key}.png"
            samples_img = f"/results/shapes_preview/preview_{key}.png"
        cards += f'''
        <div class="ds-card">
            <div class="ds-card-header">{si['name']}</div>
            <p class="ds-card-desc">{si['desc']}</p>
            <div class="ds-images">
                <a href="{detail_img}" target="_blank">
                    <img src="{detail_img}" 
                         alt="{si['name']}" loading="lazy" class="ds-img">
                </a>
                <a href="{samples_img}" target="_blank">
                    <img src="{samples_img}" 
                         alt="{si['name']} samples" loading="lazy" class="ds-img">
                </a>
            </div>
        </div>'''
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>训练数据集 - EIT</title>
{BASE_STYLE}
<style>
.ds-card {{
    background:rgba(15,23,42,0.8);border:1px solid rgba(59,130,246,0.12);
    border-radius:14px;padding:20px;margin-bottom:20px;backdrop-filter:blur(10px);
}}
.ds-card-header {{ font-size:18px;font-weight:600;color:#d0d8e0;margin-bottom:6px; }}
.ds-card-desc {{ font-size:13px;color:#8899aa;margin-bottom:14px; }}
.ds-images {{ display:flex;gap:16px;overflow-x:auto; }}
.ds-img {{ max-height:320px;border-radius:8px;border:1px solid rgba(255,255,255,0.06);transition:transform 0.2s; }}
.ds-img:hover {{ transform:scale(1.02);border-color:rgba(59,130,246,0.3); }}
.ds-stats {{ display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-bottom:24px; }}
.ds-stat {{ text-align:center;padding:16px;background:rgba(15,23,42,0.6);border:1px solid rgba(59,130,246,0.08);border-radius:12px; }}
.ds-stat-value {{ font-size:24px;font-weight:700;color:#e0e8f0; }}
.ds-stat-label {{ font-size:12px;color:#667788;margin-top:4px; }}
@media (max-width:700px) {{ .ds-images {{ flex-direction:column; }} }}
</style>
</head>
<body><div class="container">
    <div class="navbar">
        <div class="navbar-left">
            <span class="logo-dot"></span>
            <span class="navbar-title">EIT 训练数据集</span>
        </div>
        <div>
            <a href="/" class="nav-back">&larr; 返回首页</a>
        </div>
    </div>

    <div class="section-title"><h2>数据集概览</h2><div class="line"></div></div>

    <!-- 根系数据集 -->
    <div class="ds-card" style="border-color:rgba(34,197,94,0.2);background:rgba(15,30,20,0.8);">
        <div class="ds-card-header" style="color:#22c55e;">🌱 根系数据集 (Root Dataset)</div>
        <p class="ds-card-desc">植物根系电导率成像数据集，包含直根型、须根型、鱼骨型三种根系结构。电导率范围：土壤 0.01 S/m，根系 0.05 S/m。</p>
        <div class="ds-stats" style="margin-bottom:12px;">
            <div class="ds-stat"><div class="ds-stat-value">1000</div><div class="ds-stat-label">训练样本</div></div>
            <div class="ds-stat"><div class="ds-stat-value">100</div><div class="ds-stat-label">验证样本</div></div>
            <div class="ds-stat"><div class="ds-stat-value">100</div><div class="ds-stat-label">测试样本</div></div>
            <div class="ds-stat"><div class="ds-stat-value">4424</div><div class="ds-stat-label">FEM 单元</div></div>
            <div class="ds-stat"><div class="ds-stat-value">5.3%</div><div class="ds-stat-label">平均根系覆盖率</div></div>
            <div class="ds-stat"><div class="ds-stat-value">1 频率</div><div class="ds-stat-label">测量频率</div></div>
        </div>
        <div class="ds-images">
            <a href="/results/dataset_preview/root_detail.png" target="_blank">
                <img src="/results/dataset_preview/root_detail.png"
                     alt="Root dataset detail" loading="lazy" class="ds-img">
            </a>
            <a href="/results/dataset_preview/root_samples.png" target="_blank">
                <img src="/results/dataset_preview/root_samples.png"
                     alt="Root samples" loading="lazy" class="ds-img">
            </a>
        </div>
    </div>

    <!-- 通用形状数据集 -->
    <div class="section-title" style="margin-top:30px;"><h2>通用形状数据集（精细网格）</h2><div class="line"></div></div>
    <div class="ds-stats">
        <div class="ds-stat"><div class="ds-stat-value">40</div><div class="ds-stat-label">预览样本</div></div>
        <div class="ds-stat"><div class="ds-stat-value">6528</div><div class="ds-stat-label">FEM 单元</div></div>
        <div class="ds-stat"><div class="ds-stat-value">4</div><div class="ds-stat-label">形状类型</div></div>
        <div class="ds-stat"><div class="ds-stat-value">5x</div><div class="ds-stat-label">固定对比度</div></div>
        <div class="ds-stat"><div class="ds-stat-value">50/50</div><div class="ds-stat-label">边缘/中心</div></div>
        <div class="ds-stat"><div class="ds-stat-value">-40~-20 dB</div><div class="ds-stat-label">噪声范围</div></div>
    </div>
    <div class="ds-images" style="margin-bottom:20px;">
        <a href="/results/shapes_preview/preview_grid.png" target="_blank">
            <img src="/results/shapes_preview/preview_grid.png"
                 alt="All shapes preview" loading="lazy" class="ds-img" style="max-height:480px;">
        </a>
    </div>

    <div class="section-title"><h2>形状类型预览</h2><div class="line"></div></div>
    <p style="color:#667788;font-size:13px;margin-bottom:16px;">
        每行展示该形状的 <strong style="color:#94a3b8;">电导率分布 (Sigma)</strong> 和 <strong style="color:#94a3b8;">边界电压 (Voltage)</strong> (前2频率)。点击图片放大查看。
    </p>
    {cards}

    <!-- ===== 增强版 v3 数据集 ===== -->
    <div class="section-title" style="margin-top:40px;"><h2>🚀 增强版 v3 数据集（混合形状）</h2><div class="line"></div></div>
    <div class="ds-card" style="border-color:rgba(251,191,36,0.2);background:rgba(30,20,10,0.8);">
        <div class="ds-card-header" style="color:#fbbf24;">混合形状 EIT 数据集 v3</div>
        <p class="ds-card-desc">
            多样性增强版：6 种形状类型（含环形、近边界硬样本）、随机对比度 3x~10x、
            3 个系统噪声测试集。用于 ConvSpatialEIT v3 训练。
        </p>
        <div class="ds-stats" style="margin-bottom:12px;">
            <div class="ds-stat"><div class="ds-stat-value">20,000</div><div class="ds-stat-label">训练样本</div></div>
            <div class="ds-stat"><div class="ds-stat-value">500</div><div class="ds-stat-label">验证样本</div></div>
            <div class="ds-stat"><div class="ds-stat-value">2,000</div><div class="ds-stat-label">测试样本 (4组×500)</div></div>
            <div class="ds-stat"><div class="ds-stat-value">4424</div><div class="ds-stat-label">FEM 单元</div></div>
            <div class="ds-stat"><div class="ds-stat-value">6</div><div class="ds-stat-label">形状类型</div></div>
            <div class="ds-stat"><div class="ds-stat-value">3x~10x</div><div class="ds-stat-label">对比度范围</div></div>
        </div>
        <div class="ds-images">
            <a href="/results/dataset_preview_v3/all_shapes_grid.png" target="_blank">
                <img src="/results/dataset_preview_v3/all_shapes_grid.png"
                     alt="v3 all shapes" loading="lazy" class="ds-img" style="max-height:480px;">
            </a>
        </div>
    </div>

    <!-- ===== 扩散模型数据集 (Diffusion Model Dataset) ===== -->
    <div class="section-title" style="margin-top:40px;"><h2>🌀 扩散模型专用数据集 (Diffusion-Ready)</h2><div class="line"></div></div>
    <div class="ds-card" style="border-color:rgba(34,197,94,0.3);background:rgba(10,25,15,0.85);">
        <div class="ds-card-header" style="color:#4ade80;">硬边界 → 平滑边界 (Hard → Smooth Boundary)</div>
        <p class="ds-card-desc">
            原数据每样本仅 <strong style="color:#f87171;">2 个 σ 值</strong>（硬边界），扩散模型无法有效训练。
            经过 <strong style="color:#4ade80;">1 次图拉普拉斯平滑 (strength=0.10)</strong> 后，边界产生连续过渡带（14~35 个 σ 值），
            同时保持电压信号相关性 &gt; 0.97，物理一致性几乎不变。
        </p>
        <div class="ds-stats" style="margin-bottom:12px;">
            <div class="ds-stat"><div class="ds-stat-value" style="color:#f87171;">2</div><div class="ds-stat-label">Hard: unique σ values</div></div>
            <div class="ds-stat"><div class="ds-stat-value" style="color:#4ade80;">14~35</div><div class="ds-stat-label">Smooth: unique σ values</div></div>
            <div class="ds-stat"><div class="ds-stat-value">3~20%</div><div class="ds-stat-label">Transition zone</div></div>
            <div class="ds-stat"><div class="ds-stat-value" style="color:#4ade80;">&gt;0.97</div><div class="ds-stat-label">Voltage correlation</div></div>
            <div class="ds-stat"><div class="ds-stat-value">6</div><div class="ds-stat-label">Shape types</div></div>
            <div class="ds-stat"><div class="ds-stat-value">1 iter, 0.10</div><div class="ds-stat-label">Smooth params</div></div>
        </div>
        <div class="ds-images">
            <a href="/results/dataset_preview/diffusion_hard_vs_smooth_grid.png" target="_blank">
                <img src="/results/dataset_preview/diffusion_hard_vs_smooth_grid.png"
                     alt="Diffusion dataset: Hard vs Smooth all shapes" loading="lazy" class="ds-img" style="max-height:520px;">
            </a>
        </div>
    </div>

    <div class="section-title"><h2>扩散模型数据集 — 详细对比</h2><div class="line"></div></div>
    <p style="color:#667788;font-size:13px;margin-bottom:16px;">
        每个形状展示：σ 分布图 (Hard / Smooth / Difference) + 直方图 + 边界电压 + 电压一致性散点图。
    </p>
    <div class="ds-images" style="flex-wrap:wrap;gap:12px;margin-bottom:16px;">
        <a href="/results/dataset_preview/diffusion_circle_detail.png" target="_blank">
            <img src="/results/dataset_preview/diffusion_circle_detail.png"
                 alt="circle detail" loading="lazy" class="ds-img" style="max-height:280px;">
        </a>
        <a href="/results/dataset_preview/diffusion_ring_detail.png" target="_blank">
            <img src="/results/dataset_preview/diffusion_ring_detail.png"
                 alt="ring detail" loading="lazy" class="ds-img" style="max-height:280px;">
        </a>
        <a href="/results/dataset_preview/diffusion_summary_comparison.png" target="_blank">
            <img src="/results/dataset_preview/diffusion_summary_comparison.png"
                 alt="summary comparison" loading="lazy" class="ds-img" style="max-height:280px;">
        </a>
    </div>

    <div class="section-title"><h2>v3 形状类型预览</h2><div class="line"></div></div>
    <p style="color:#667788;font-size:13px;margin-bottom:16px;">
        每行展示该形状的 <strong style="color:#94a3b8;">电导率分布 (σ Map, FEM 网格)</strong> 和 <strong style="color:#94a3b8;">边界电压曲线 (208 通道)</strong>。点击图片放大查看。
    </p>

    <div class="ds-card">
        <div class="ds-card-header">圆形 (Circle)</div>
        <p class="ds-card-desc">单圆内含物，随机位置+半径，约占训练集 20%</p>
        <div class="ds-images">
            <a href="/results/dataset_preview_v3/circle_preview.png" target="_blank">
                <img src="/results/dataset_preview_v3/circle_preview.png" alt="v3 circle" loading="lazy" class="ds-img">
            </a>
        </div>
    </div>
    <div class="ds-card">
        <div class="ds-card-header">椭圆 (Ellipse)</div>
        <p class="ds-card-desc">椭圆内含物，随机长短轴+旋转，约占 18%</p>
        <div class="ds-images">
            <a href="/results/dataset_preview_v3/ellipse_preview.png" target="_blank">
                <img src="/results/dataset_preview_v3/ellipse_preview.png" alt="v3 ellipse" loading="lazy" class="ds-img">
            </a>
        </div>
    </div>
    <div class="ds-card">
        <div class="ds-card-header">双圆 (Double Circle)</div>
        <p class="ds-card-desc">两个不重叠的圆内含物，约占 18%</p>
        <div class="ds-images">
            <a href="/results/dataset_preview_v3/double_circle_preview.png" target="_blank">
                <img src="/results/dataset_preview_v3/double_circle_preview.png" alt="v3 double_circle" loading="lazy" class="ds-img">
            </a>
        </div>
    </div>
    <div class="ds-card">
        <div class="ds-card-header">环形 (Ring)</div>
        <p class="ds-card-desc">空心圆环，测试边缘检测能力，约占 15%</p>
        <div class="ds-images">
            <a href="/results/dataset_preview_v3/ring_preview.png" target="_blank">
                <img src="/results/dataset_preview_v3/ring_preview.png" alt="v3 ring" loading="lazy" class="ds-img">
            </a>
        </div>
    </div>
    <div class="ds-card">
        <div class="ds-card-header">近边界 (Near-Boundary)</div>
        <p class="ds-card-desc">内含物强制靠近桶壁（距边界 &lt; 2.5cm），EIT 最难场景，约占 15%</p>
        <div class="ds-images">
            <a href="/results/dataset_preview_v3/near_boundary_preview.png" target="_blank">
                <img src="/results/dataset_preview_v3/near_boundary_preview.png" alt="v3 near_boundary" loading="lazy" class="ds-img">
            </a>
        </div>
    </div>

    <div class="section-title" style="margin-top:30px;"><h2>v3 系统噪声测试集</h2><div class="line"></div></div>
    <p style="color:#667788;font-size:13px;margin-bottom:16px;">
        固定噪声电平的专用测试集，用于评估模型在不同噪声下的鲁棒性。
    </p>
    <div class="ds-card">
        <div class="ds-card-header" style="color:#22c55e;">低噪声测试 (-30 dB)</div>
        <p class="ds-card-desc">500 样本，固定 -30dB 加性噪声</p>
        <div class="ds-images">
            <a href="/results/dataset_preview_v3/test_low_noise_preview.png" target="_blank">
                <img src="/results/dataset_preview_v3/test_low_noise_preview.png" alt="v3 low noise" loading="lazy" class="ds-img">
            </a>
        </div>
    </div>
    <div class="ds-card">
        <div class="ds-card-header" style="color:#ef4444;">高噪声测试 (-15 dB)</div>
        <p class="ds-card-desc">500 样本，固定 -15dB 加性噪声（高噪声挑战）</p>
        <div class="ds-images">
            <a href="/results/dataset_preview_v3/test_high_noise_preview.png" target="_blank">
                <img src="/results/dataset_preview_v3/test_high_noise_preview.png" alt="v3 high noise" loading="lazy" class="ds-img">
            </a>
        </div>
    </div>
    <div class="ds-card">
        <div class="ds-card-header" style="color:#f59e0b;">近边界测试 (-25 dB)</div>
        <p class="ds-card-desc">500 样本，内含物靠近边界 + 固定 -25dB 噪声</p>
        <div class="ds-images">
            <a href="/results/dataset_preview_v3/test_near_boundary_preview.png" target="_blank">
                <img src="/results/dataset_preview_v3/test_near_boundary_preview.png" alt="v3 near_boundary noise" loading="lazy" class="ds-img">
            </a>
        </div>
    </div>
</div></body></html>'''
    return html


# ============ HTTP Handler ============

class DynamicHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # /api/git-pull -> execute git pull and return result
        if self.path == '/api/git-pull':
            try:
                result = subprocess.run(
                    ['git', 'pull'],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    capture_output=True, text=True, timeout=30
                )
                success = result.returncode == 0
                output = result.stdout + result.stderr
                if len(output) > 2000:
                    output = output[:2000] + '\n...(truncated)'
                data = json.dumps({
                    'success': success,
                    'output': output.strip(),
                    'returncode': result.returncode,
                }).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', len(data))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                data = json.dumps({
                    'success': False,
                    'output': f'Error: {str(e)}',
                }).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', len(data))
                self.end_headers()
                self.wfile.write(data)
            return

        parsed = urlparse(self.path)
        import urllib.parse
        path = urllib.parse.unquote(parsed.path)

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

        # /results/ -> EIT results (静态 HTML，和首页一样避免卡顿)
        if path == '/results/' or path == '/results':
            results_static = os.path.join(RESULTS_DIR, 'index.html')
            if os.path.exists(results_static):
                with open(results_static, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', len(data))
                self.end_headers()
                self.wfile.write(data)
            else:
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

        # /datasets/ -> 训练数据集预览
        if path == '/datasets/' or path == '/datasets':
            html = generate_datasets_page()
            self._send_html(html)
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
            # 下载模式: ?dl=1 触发 Content-Disposition
            if '?' in self.path and 'dl=1' in self.path:
                from urllib.parse import quote
                fname = os.path.basename(local_path)
                # RFC 5987: encode non-ASCII filenames for Content-Disposition
                try:
                    fname.encode('ascii')
                    self.send_header('Content-Disposition',
                                     f'attachment; filename="{fname}"')
                except UnicodeEncodeError:
                    self.send_header('Content-Disposition',
                                     f"attachment; filename*=UTF-8''{quote(fname)}")
            # Cache images for 1 hour, HTML for 5 min
            if content_type and 'image' in content_type:
                self.send_header('Cache-Control', 'public, max-age=3600')
            elif content_type and 'html' in content_type:
                self.send_header('Cache-Control', 'public, max-age=300')
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def do_POST(self):
        # /api/training/delete -> delete training records
        if self.path == '/api/training/delete':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                data = json.loads(body)
                run_ids = data.get('run_ids', [])
                if not run_ids:
                    raise ValueError("No run_ids provided")

                import shutil
                deleted = []
                for run_id in run_ids:
                    run_dir = os.path.join(TRAINING_RECORDS_DIR, run_id)
                    if os.path.isdir(run_dir):
                        shutil.rmtree(run_dir)
                        deleted.append(run_id)

                # 更新 index.json
                index = _recorder_mod.load_index()
                index["runs"] = [r for r in index["runs"] if r["run_id"] not in deleted]
                _recorder_mod.save_index(index)

                resp = json.dumps({"success": True, "deleted": deleted})
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', len(resp.encode()))
                self.end_headers()
                self.wfile.write(resp.encode())
            except Exception as e:
                resp = json.dumps({"success": False, "error": str(e)})
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', len(resp.encode()))
                self.end_headers()
                self.wfile.write(resp.encode())
            return
        self.send_error(405)

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
    server = ThreadingHTTPServer((args.host, args.port), DynamicHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.server_close()


if __name__ == "__main__":
    main()
