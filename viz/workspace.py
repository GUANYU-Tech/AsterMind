"""Render the local studio shell before the browser's first paint."""
import re
from html import escape

LINKS = (
    ('01', '场景概览', '/workspace'), ('02', '实验配置', '/configuration'),
    ('03', '在线仿真', '/live'), ('04', '实验结果', '/results'),
    ('05', '运行记录', '/experiments'),
)


_LANGUAGE_ASSETS = (b'<link rel="stylesheet" href="/assets/language.css">'
                    b'<script src="/assets/language.js" defer></script>')


def add_language_switch(content: bytes) -> bytes:
    head_end = re.search(br'</head\s*>', content, re.I)
    if head_end:
        at = head_end.start()
        return content[:at] + _LANGUAGE_ASSETS + content[at:]
    return content + _LANGUAGE_ASSETS


def decorate_document(content: bytes, path: str) -> bytes:
    if b'data-studio-rendered' in content:
        return content
    # Styles follow page-specific styles, but block initial rendering. Legacy
    # documents without explicit head/body tags receive them before main.
    styles = (b'<meta name="color-scheme" content="dark">'
              b'<style>html{background:#080f14;color-scheme:dark}</style>'
              b'<link rel="stylesheet" href="/assets/workspace.css" data-studio-theme>'
              b'<link rel="stylesheet" href="/assets/starguide.css">'
              b'<link rel="stylesheet" href="/assets/mission.css">')
    head_end = re.search(br'</head\s*>', content, re.I)
    body = re.search(br'<body\b[^>]*>', content, re.I)
    main = re.search(br'<main\b', content, re.I)
    at = head_end.start() if head_end else body.start() if body else main.start() if main else 0
    content = content[:at] + styles + content[at:]

    def root(match):
        tag = match.group().decode('utf-8')
        if re.search(r'\bclass\s*=', tag, re.I):
            tag = re.sub(r'(\bclass\s*=\s*[\"\'])', r'\1studio-shell ', tag, count=1, flags=re.I)
        else:
            tag = tag[:-1] + ' class="studio-shell">'
        return (tag[:-1] + ' data-studio-rendered>').encode('utf-8')

    content = re.sub(br'<html\b[^>]*>', root, content, count=1, flags=re.I)
    content = content.replace(b'OrbitServe', 'AsterMind 星鉴'.encode())
    if path == '/platform':
        content = content.replace(b'class="studio-shell', b'class="studio-shell portal-shell', 1)
    current_path = ('/configuration' if path in {'/physical', '/scene'} else
                    '/results' if re.match(r'/runs/[^/]+/(result|scene-replay|native-replay|runtime)', path) else
                    '/live' if path.startswith('/live/') else path)
    rows = []
    for index, (number, label, href) in enumerate(LINKS):
        current = ' aria-current="page"' if current_path == href.split('?')[0] else ''
        rows.append(f'<a href="{escape(href, quote=True)}"{current}><b aria-hidden="true">{number}</b>{label}</a>')
    caption = '太空智算系统仿真平台' if path == '/platform' else '仿真任务工作台'
    parent = {'/platform': 'https://guanyu-tech.github.io/AsterMind/', '/workspace': '/platform'}.get(path, '/workspace')
    home = ' data-project-home target="_blank" rel="noopener noreferrer"' if path == '/platform' else ''
    shell = (f'<header class="studio-topbar"><a class="studio-logo" href="{parent}"{home}>'
             '<span class="studio-mark" aria-hidden="true">✧</span>AsterMind <em>星鉴</em></a>'
             f'<span>{caption}</span><a class="studio-local" href="/platform">'
             '切换仿真工作区 ↗</a></header>'
             '<aside class="studio-sidebar" aria-label="工作台导航">'
             '<div class="studio-nav-label">完整场景仿真</div>' + ''.join(rows) +
             '<div class="studio-sidefoot"><a href="/assets/guide.html" '
             'target="_blank" rel="noopener noreferrer">技术文档 ↗</a></div></aside>').encode('utf-8')
    body = re.search(br'<body\b[^>]*>', content, re.I)
    main = re.search(br'<main\b', content, re.I)
    at = body.end() if body else main.start() if main else len(content)
    return add_language_switch(content[:at] + shell + content[at:])
