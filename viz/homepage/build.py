"""Generate the equivalent English and Chinese static project pages."""
from pathlib import Path
from html import escape
import argparse
import json
from shutil import copyfile

ROOT = Path(__file__).resolve().parent
REPORT = ROOT.parents[1] / 'doc' / 'AsterMind-V1-Tech-Report.pdf'
REPO = 'https://github.com/GUANYU-Tech/AsterMind'
PREVIEW_TAG = 'v1.0.0-preview.7'
WINDOWS_INSTALLER = f'{REPO}/releases/download/{PREVIEW_TAG}/AsterMind-v1.0-Windows-x64-Setup.exe'
COPY = {
'en': {
'lang':'en','skip':'Skip to content','menu':'Menu','nav':['Background','Framework','Interactive diagrams','Examples','About'],
'eyebrow':'GUANYU · SPACE COMPUTING RESEARCH',
'title':['Open-source simulation','for space computing systems'],
'desc':'Connect <strong>orbital dynamics, distributed compute, and communication</strong> to understand end-to-end LLM serving across satellite Compute Cells.',
'platform':'Enter simulation platform',
'actions':['Start with the background','Explore the framework','GitHub'],
'report':'Technical Report','collapse':'Collapse','close_report':'Collapse report',
'dimensions':['Orbital geometry','Distributed compute','Communication','System evaluation'],
'dimsub':['Positions & connectivity','Model, hardware & parallelism','Protocols & shared resources','Latency, throughput & traces'],
'promise':'Physical configuration · Event-driven execution · Interpretable serving behavior',
'background':['FROM CONSTELLATIONS TO COMPUTE CELLS','Space computing is<br>a system-level question.','Faster accelerators are only part of the answer. Satellites move, models span devices, and requests share network resources.'],
'challenges':[
['A moving network','Distance, visibility, and link availability evolve with orbital geometry. The physical scene defines the resources that execution can use.'],
['A distributed model','Model ranks span accelerators and satellites. Kernel dependencies, parallel deployment, and collective communication shape progress.'],
['A shared system','Input delivery, KV readiness, batch scheduling, and network contention all contribute to the time a request takes to complete.']],
'statement':'Evaluate a specified system design—from physical configuration to end-to-end serving behavior.',
'framework':['THE ASTERMIND FRAMEWORK','Configure the scenario.<br>Run the simulation.','Builder prepares configuration and physical inputs. Engine advances runtime state through a shared event timeline.'],
'builder':'Configuration & preparation','engine':'Runtime simulation',
'brows':[['Constellation & geometry','Formation · TLE / SGP4 · visibility'],['Resources & deployment','Accelerators · model ranks · Compute Cells'],['Workload & network','Requests · topology · link parameters']],
'compute':'Compute','comm':'Communication','commshort':'Protocol models + shared flows',
'runtime':['Request admission','Batch scheduling','KV readiness','Dynamic network state'],
'clock':['One global event timeline','Integer ticks · causal dependencies · completion feedback'],
'observe':'OBSERVE','metrics':['Request latency','Token timing','Throughput','Compute & communication traces'],
'journey':['FOLLOW A REQUEST','From arrival to the next token.','Follow a request from ground delivery through Compute Cell execution to token return.'],
'steps':['Deliver & admit','Schedule & compute','Communicate','Return & observe'],
'nodes':[['Ground','Request source'],['Compute Cell','Scheduler + kernels'],['Shared network','Flows + resources'],['Output','Tokens + traces']],
'stages':[
['REQUEST READINESS','A request enters only when it is ready.','Input delivery, predecessor completion, and KV readiness gate admission to the target Cell. A migration must also satisfy model, KV format, and parallel-layout compatibility.'],
['COMPUTE CELL EXECUTION','A batch becomes a dependency graph.','The scheduler selects work and manages KV state. The compute model estimates kernel execution. Dependencies determine when compute and communication operations can advance.'],
['COMMUNICATION PROGRESS','A transfer shares the path it uses.','Supported collectives can expand into dependent P2P transfers. Protocol-aware estimates and shared-resource allocation determine progress; completion releases dependent work. Intra-satellite modes use their applicable models.'],
['COMPLETION FEEDBACK','Tokens return. The system moves forward.','Outputs travel back to the request endpoint. Completion events update request and Cell state, making subsequent work eligible. Reports expose serving metrics and execution traces.']],
'play':'Play','pause':'Pause','restart':'Restart from the beginning',
'models':['INTERPRETABLE PERFORMANCE MODELING','Compute and communication.<br>Connected by dependencies.','Model execution time and data movement together to explain where the system spends time.'],
'modelcards':[
['COMPUTE','From a batch to kernel time.','Midlayer or Roofline is selected as the global compute mode. Model configuration, hardware profiles, and supported parallel deployments determine the execution path.'],
['COMMUNICATION','Protocol-aware latency.<br>Resource-aware progress.','Communication combines protocol-aware estimates with flow-level resource sharing where applicable. Cross-satellite transfers consume the capacity of their paths and shared resources.']],
'outputcards':[['TTFT','Time to first token','Trace the path from request arrival to the first returned output token.'],['TOKEN TIMING','Decode progress','Inspect output-token timing alongside scheduling, computation, and communication events.'],['THROUGHPUT','System-level output','Interpret serving throughput together with request completion and execution traces.']],
'examples':['CASE STUDIES','Three scenarios for<br>inference in orbit.','Compare satellite links, task assignment and request scheduling.'],
'cases':[
['CASE STUDY I · COMPUTE FORMATION','How should satellites in a formation connect?', 'With the same total bandwidth, direct links between satellites that often exchange data can shorten inference latency.', ('Ring', '→', 'Full mesh')],
['CASE STUDY II · CONSTELLATION ACCESS AND LOAD DISTRIBUTION','When ground sites send requests at once, how can the constellation serve more of them?', 'Relays can ease local congestion by sending tasks to cells with spare capacity.', ('Relay', '→', 'Cell')],
['CASE STUDY III · MULTI-TURN INFERENCE SCHEDULING','How can users get a faster response to a follow-up?', 'Keeping follow-ups on the same cell lets it reuse cached context and return the first token sooner in long conversations.', ('Ground', '⇄', 'Cell')]],
'start':['GET STARTED','Build on AsterMind.','Run the Windows prerelease installer or clone the source from GitHub and build it yourself.'],
'windows':'Windows installer','windowsnote':'Windows 11 x64 · Prerelease',
'guide':'Install from source','download':'Download the Windows x64 installer',
'codehead':'Clone, build and run','copy':'Copy commands','copied':'Copied','copystatus':'Setup commands copied.','copyfail':'Select code to copy','copyfailstatus':'Clipboard unavailable. Select and copy the commands below.',
'about':['ABOUT THE PROJECT','Exploring Space Computing','GUANYU and developers are building an open digital testbed for space computing.'],
'org':'GUANYU / 观宇芯算','issue':'Issues & discussion','footer':'ORBITAL GEOMETRY · COMPUTE · COMMUNICATION · SYSTEMS',
},
'zh': {
'lang':'zh-CN','skip':'跳转到正文','menu':'菜单','nav':['研究背景','整体框架','交互图解','场景示例','关于项目'],
'eyebrow':'观宇芯算 · 太空计算研究',
'title':['面向太空智算系统的','开源仿真平台'],
'desc':'连接<strong>轨道动态、分布式计算与通信</strong>，<br class="desktop-break">理解卫星计算编队中大模型推理的端到端服务性能。',
'platform':'进入仿真平台',
'actions':['了解研究背景','探索整体框架','GitHub'],
'report':'技术报告','collapse':'收起','close_report':'收起报告',
'dimensions':['轨道几何','分布式计算','通信建模','系统评估'],
'dimsub':['位置与连接关系','模型、硬件与并行部署','协议与共享资源','时延、吞吐与执行轨迹'],
'promise':'物理场景配置 · 事件驱动执行 · 可解释的推理服务行为',
'background':['从星座到计算编队','太空智算，<br>是一个系统级问题。','更快的加速器只是其中一环。卫星持续运动，模型跨设备部署，请求共同使用有限的网络资源。'],
'challenges':[
['持续变化的网络','轨道几何影响距离、可见性与链路可用性。物理场景决定仿真执行时能够使用哪些资源。'],
['分布式部署的模型','模型各个 rank 分布在加速器和卫星上，算子依赖、并行部署与集合通信共同影响执行进度。'],
['共享资源的系统','输入送达、KV 就绪、批次调度与网络竞争，共同决定请求从到达到完成所需的时间。']],
'statement':'评估一套确定的系统设计：从物理配置，一直到端到端推理服务行为。',
'framework':['ASTERMIND 整体框架','配置场景，<br>执行仿真。','Builder 准备配置与物理输入，Engine 通过统一的事件时间线推进运行时状态。'],
'builder':'配置与准备','engine':'运行时仿真',
'brows':[['星座与几何','编队 · TLE / SGP4 · 可见性'],['资源与部署','加速器 · 模型 rank · 计算 Cell'],['负载与网络','请求流 · 拓扑 · 链路参数']],
'compute':'计算','comm':'通信','commshort':'协议模型与流级资源共享',
'runtime':['请求准入','批次调度','KV 就绪','动态网络状态'],
'clock':['统一全局事件时间线','整数 tick · 因果依赖 · 完成事件反馈'],
'observe':'观测输出','metrics':['请求时延','Token 时间','吞吐量','计算与通信执行轨迹'],
'journey':['跟随一个请求','从请求到达，<br>到下一个 Token。','从地面送达请求，到计算编队执行，再到 Token 回传。'],
'steps':['送达与准入','调度与计算','通信执行','返回与观测'],
'nodes':[['地面','请求来源'],['计算 Cell','调度器与算子'],['共享网络','流与资源'],['输出','Token 与执行轨迹']],
'stages':[
['请求就绪','条件满足，请求才能进入 Cell。','输入送达、前驱完成和 KV 就绪共同约束目标 Cell 的请求准入。迁移还需满足模型、KV 格式和并行布局的兼容性要求。'],
['计算 CELL 执行','一个批次，展开为依赖图。','调度器选择执行任务并管理 KV 状态，计算模型估计算子执行时间。依赖关系决定计算与通信算子何时能够继续推进。'],
['通信进度','一次传输，共享它经过的资源。','支持的集合通信可展开为有依赖关系的 P2P 传输。协议感知估计与共享资源分配共同决定进度，完成事件释放后续任务；星内通信使用对应模式下适用的模型。'],
['完成反馈','Token 返回，系统继续推进。','输出返回请求端点，完成事件更新请求与 Cell 状态，使后续工作具备执行条件。报告呈现服务指标与计算、通信执行轨迹。']],
'play':'播放','pause':'暂停','restart':'从头播放',
'models':['可解释的性能建模','计算与通信，<br>通过依赖关系连接。','联合建模执行时间与数据传输，理解时间消耗在系统的哪些环节。'],
'modelcards':[
['计算建模','从调度批次，到算子耗时。','全局选择 Midlayer 或 Roofline 计算模式，由模型配置、硬件档案与受支持的并行部署共同决定执行路径。'],
['通信建模','协议感知的时延，<br>资源约束下的进度。','通信在适用路径上结合协议感知时延估计与流级资源共享。跨星传输消耗所经路径和共享资源的容量。']],
'outputcards':[['TTFT','首 Token 时延','跟踪从请求到达，到首个输出 Token 返回的完整路径。'],['TOKEN 时间','Decode 执行进度','结合调度、计算与通信事件，观察输出 Token 的时间信息。'],['吞吐量','系统级输出','将服务吞吐与请求完成状态、执行轨迹结合起来分析。']],
'examples':['研究案例','用三个场景，<br>研究在轨推理。','比较星间互联、任务分配与请求调度。'],
'cases':[
['案例 I · 算力编队','编队内的卫星如何互联？','总带宽相同时，让频繁通信的卫星直连可缩短推理时延。', ('环形', '→', '全连接')],
['案例 II · 星座接入与分流','多地任务同时到达，星座怎样承接更多请求？','中继星把任务分给仍有余量的编队，可缓解局部拥塞。', ('中继星', '→', '服务单元')],
['案例 III · 多轮推理调度','用户连续追问时，怎样缩短等待？','让同一编队接续对话，可复用上下文缓存，让长对话更早返回首个 Token。', ('地面站', '⇄', '编队')]],
'start':['开始使用','基于 AsterMind，开展研究。','提供Windows 上运行预发布安装包，或从 GitHub 克隆源码并构建'],
'windows':'Windows 安装包','windowsnote':'Windows 11 x64 · 预发布版',
'guide':'从源码安装','download':'下载 Windows x64 安装包',
'codehead':'克隆、构建与启动','copy':'复制命令','copied':'已复制','copystatus':'安装命令已复制。','copyfail':'选中代码复制','copyfailstatus':'剪贴板不可用，请选中并复制下方命令。',
'about':['关于项目','探索太空智算','观宇，与开发者共同构建太空智算的开放数字试验场'],
'org':'观宇芯算 / GUANYU','issue':'问题与讨论','footer':'轨道几何 · 计算 · 通信 · 系统',
}}

icons = [
'<ellipse cx="70" cy="42" rx="49" ry="18" transform="rotate(-24 70 42)"/><circle cx="70" cy="42" r="24"/><ellipse cx="70" cy="42" rx="11" ry="24"/><path d="M47 42h46"/><rect x="107" y="18" width="9" height="9" fill="currentColor"/>',
'<rect x="51" y="23" width="38" height="38" rx="6"/><rect x="61" y="33" width="18" height="18" rx="2" fill="currentColor"/><path d="M59 16v7m11-7v7m11-7v7M59 61v7m11-7v7m11-7v7M44 31h7m-7 11h7m-7 11h7m38-22h7m-7 11h7m-7 11h7"/><rect x="17" y="30" width="21" height="24" rx="4"/><rect x="102" y="30" width="21" height="24" rx="4"/>',
'<path d="M24 23h92v38H24zM70 23v38"/><g fill="currentColor"><circle cx="24" cy="23" r="4"/><circle cx="70" cy="23" r="4"/><circle cx="116" cy="23" r="4"/><circle cx="24" cy="61" r="4"/><circle cx="70" cy="61" r="4"/><circle cx="116" cy="61" r="4"/></g>',
'<path d="M26 15v52h94"/><rect x="40" y="46" width="14" height="20" rx="2"/><rect x="65" y="33" width="14" height="33" rx="2"/><rect x="90" y="18" width="14" height="48" rx="2" fill="currentColor"/><path d="m42 28 30-13 29-8"/>']
def icon(i): return '<svg viewBox="0 0 140 84" aria-hidden="true">'+icons[i]+'</svg>'
def external(url,label,cls=''): return f'<a class="{cls}" href="{url}" target="_blank" rel="noopener">{label}</a>'
def heading(a): return f'<div class="section-heading"><p class="eyebrow">{a[0]}</p><h2>{a[1]}</h2><p>{a[2]}</p></div>'
setup=f'''git clone {REPO}.git
cd AsterMind

bash scripts/install-ubuntu-deps.sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/build.sh
bash scripts/run.sh'''
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pages-dir', type=Path, help='Export a standalone GitHub Pages site to this directory.')
args = parser.parse_args()
output = args.pages_dir if args.pages_dir is not None else ROOT / 'dist'
if args.pages_dir is not None:
    if output.resolve() == (ROOT / 'dist').resolve():
        parser.error('Use a separate directory for the Pages export.')
    (output / 'assets').mkdir(parents=True, exist_ok=True)
    for name in ('style.css', 'app.js', 'language.js', 'assets/guanyu-logo.png',
                 'assets/astermind-mark.png'):
        copyfile(ROOT / 'dist' / name, output / name)
    (output / '.nojekyll').touch()

copyfile(REPORT, output / 'assets' / REPORT.name)

for lang,d in COPY.items():
    prefix='../' if lang=='en' else './'
    alternate='../' if lang=='en' else './en/'
    report_url = prefix + 'assets/AsterMind-V1-Tech-Report.pdf'
    start_href = '/platform' if args.pages_dir is None else '#start'
    company = ('<span class="company-name" lang="en">GUANYU</span>' if lang == 'en' else
               f'<img src="{prefix}assets/guanyu-logo.png" width="1903" height="784" alt="观宇芯算 GUANYU">')
    nav=''.join(f'<a href="#{anchor}">{name}</a>' for anchor,name in zip(['background','framework','modeling','examples','about'],d['nav']))
    hero=''.join(f'<div class="dimension {"evaluation" if i==3 else ""}"><span class="dimension-label">{d["dimensions"][i]}</span>{icon(i)}<h3>{d["dimensions"][i]}</h3><p>{d["dimsub"][i]}</p></div>'+ ('<span class="connector" aria-hidden="true">'+('→' if i==2 else '+')+'</span>' if i<3 else '') for i in range(4))
    challenges=''.join(f'<article>{icon(i)}<h3>{v[0]}</h3><p>{v[1]}</p></article>' for i,v in enumerate(d['challenges']))
    brows=''.join(f'<div class="config-row"><h4>{v[0]}</h4><p>{v[1]}</p></div>' for v in d['brows'])
    tabs=''.join(f'<button id="step-{i}" role="tab" aria-selected="{str(i==0).lower()}" aria-controls="journey-panel" tabindex="{0 if i==0 else -1}" data-step="{i}"><span class="stage-track" aria-hidden="true"><i></i></span><span class="step-number">{i+1:02}</span>{v}</button>' for i,v in enumerate(d['steps']))
    nodes = '<span class="flow-wire" aria-hidden="true">→</span>'.join(
        f'<button type="button" class="flow-node{" active" if i == 0 else ""}" data-node="{i}" '
        f'aria-pressed="{str(i == 0).lower()}" aria-controls="journey-panel">'
        f'{icon(i)}<span class="node-name">{name}</span><span class="node-detail">{detail}</span></button>'
        for i, (name, detail) in enumerate(d['nodes'])
    )
    models=''.join(f'<article class="model-card"><p class="eyebrow">{v[0]}</p><h3>{v[1]}</h3><p>{v[2]}</p></article>' for v in d['modelcards'])
    outputs=''.join(f'<article><span class="eyebrow">{v[0]}</span><h3>{v[1]}</h3><p>{v[2]}</p></article>' for v in d['outputcards'])
    cases = ''.join(
        f'<article class="example"><div class="mini-art" aria-hidden="true">'
        f'<span>{left}</span><i>{arrow}</i><span>{right}</span></div>'
        f'<p class="eyebrow">{label}</p><h3>{title}</h3><p>{description}</p></article>'
        for label, title, description, (left, arrow, right) in d['cases']
    )
    data={k:d[k] for k in ['stages','play','pause','copy','copied','copystatus','copyfail','copyfailstatus']}
    page=f'''<!doctype html>
<html lang="{d['lang']}"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#ffffff">
<title>AsterMind V1.0 — {"探索太空智算" if lang=="zh" else "Exploring Space Computing"}</title>
<meta name="description" content="{escape(d['about'][2])}">
<link rel="alternate" hreflang="en" href="{prefix}en/"><link rel="alternate" hreflang="zh-CN" href="{prefix}">
<link rel="icon" type="image/png" href="{prefix}assets/guanyu-logo.png"><link rel="stylesheet" href="{prefix}style.css"><script defer src="{prefix}language.js"></script><script defer src="{prefix}app.js"></script>
</head><body><a class="skip" href="#main">{d['skip']}</a>
<header class="header"><div class="nav-wrap"><a class="brand" href="#top" aria-label="{'GUANYU' if lang=='en' else '观宇芯算'} · AsterMind V1.0">{company}<span class="brand-divider"></span><span class="brand-product"><img class="brand-mark" src="{prefix}assets/astermind-mark.png" width="876" height="728" alt="">AsterMind</span></a><div class="nav-right"><nav id="nav" aria-label="{d['menu']}">{nav}</nav><a id="language" class="language" href="{alternate}" lang="{'en' if lang=='zh' else 'zh-CN'}" aria-label="{'Switch to English' if lang=='zh' else '切换到中文版'}"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/></svg>{'EN' if lang=='zh' else '中文'}</a><button id="menu" class="menu" aria-expanded="false" aria-controls="nav">{d['menu']}</button></div></div></header>
<main id="main"><section class="hero" id="top"><div class="orbital-lines" aria-hidden="true"></div><div class="container hero-content"><p class="eyebrow">{d['eyebrow']}</p><h1><span class="project-title">AsterMind <span class="version">V1.0</span></span><span>{d['title'][0]}</span><span class="silver-title">{d['title'][1]}</span></h1><p class="hero-description">{d['desc']}</p><div class="actions"><a class="button primary" href="{start_href}">{d['platform']}</a><a class="button secondary" href="#background">{d['actions'][0]}</a><a class="button secondary" href="#framework">{d['actions'][1]}</a>{external(REPO,d['actions'][2],'button secondary github')}<a class="button secondary" id="report-link" href="#pdf" aria-controls="pdf">{d['report']}</a></div><div class="dimensions">{hero}</div><p class="hero-note">{d['promise']}</p></div></section>
<section id="pdf" class="container report" aria-labelledby="report-link" tabindex="-1">
<a class="report-close" href="#report-link" aria-label="{d['close_report']}"><svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m3 10 5-5 5 5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>{d['collapse']}</a>
<template id="report-viewer"><iframe src="{report_url}" title="AsterMind V1.0 · {d['report']}"></iframe></template>
<noscript><a class="text-link" href="{report_url}">{d['report']}</a></noscript>
</section>
<section class="section container" id="background">{heading(d['background'])}<div class="challenges">{challenges}</div><div class="statement">{d['statement']}</div></section>
<section class="section muted" id="framework"><div class="container">{heading(d['framework'])}<div class="architecture"><article class="builder"><header><h3>Builder</h3><span>{d['builder']}</span></header>{brows}</article><div class="arch-bridge" aria-hidden="true">→</div><article class="engine"><header><h3>Engine</h3><span>{d['engine']}</span></header><div class="engine-models"><div><h4>{d['compute']}</h4><p>Midlayer / Roofline</p></div><div><h4>{d['comm']}</h4><p>{d['commshort']}</p></div></div><div class="runtime">{''.join(f'<span>{v}</span>' for v in d['runtime'])}</div><div class="timeline"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/></svg><div><h4>{d['clock'][0]}</h4><p>{d['clock'][1]}</p></div></div></article></div><div class="arch-output"><span class="eyebrow">{d['observe']}</span><p>{' <span aria-hidden="true">/</span> '.join(d['metrics'])}</p></div></div></section>
<section class="section container" id="modeling">{heading(d['journey'])}
<div class="journey"><div class="flow-visual">{nodes}</div>
<div class="tabs" role="tablist" aria-label="{d['journey'][0]}">{tabs}</div>
<div class="journey-caption"><div id="journey-panel" role="tabpanel" tabindex="0" aria-labelledby="step-0">
<p class="eyebrow" id="stage-kicker">{d['stages'][0][0]}</p><h3 id="stage-title">{d['stages'][0][1]}</h3><p id="stage-body">{d['stages'][0][2]}</p></div>
<div class="journey-controls" hidden>
<button type="button" id="restart-stages" aria-label="{d['restart']}" title="{d['restart']}"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 8a6.5 6.5 0 1 1-.2 4M4 3v5h5"/></svg></button>
<button type="button" id="play-stages" aria-label="{d['pause']}" title="{d['pause']}"><svg viewBox="0 0 20 20" aria-hidden="true"><path class="pause-icon" d="M7 4v12m6-12v12"/><path class="play-icon" d="m7 4 8 6-8 6z"/></svg></button>
</div></div></div></section>
<section class="section muted" id="performance"><div class="container">{heading(d['models'])}<div class="model-grid">{models}</div><div class="metrics">{outputs}</div></div></section>
<section class="section container" id="examples">{heading(d['examples'])}<div class="example-grid">{cases}</div></section>
<section class="section muted" id="start"><div class="container">{heading(d['start'])}
<div class="getting-started">
<article class="install-windows">
<div><h3>{d['windows']}</h3><p><small>{d['windowsnote']}</small></p></div>
{external(WINDOWS_INSTALLER,d['download'],'button primary')}</article>
<article class="install-source"><h3>{d['guide']}</h3>
<div class="code-panel"><div class="code-top"><span>{d['codehead']}</span><button id="copy-code">{d['copy']}</button></div><pre><code id="setup-code">{escape(setup)}</code></pre><span class="sr-only" role="status" id="copy-status"></span></div></article>
</div></div></section>
<section class="section container about" id="about">{heading(d['about'])}<p class="organization">{d['org']}</p><div class="actions">{external(REPO,'GitHub','button primary')}{external(REPO+'/issues',d['issue'],'button secondary')}</div></section></main>
<footer class="container"><span>AsterMind V1.0</span><span>{d['footer']}</span><span>GUANYU</span></footer>
<script type="application/json" id="page-copy">{json.dumps(data,ensure_ascii=False)}</script>
</body></html>'''
    path=output/('en/index.html' if lang=='en' else 'index.html')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding='utf-8', newline='\n')
print('Generated English and Chinese pages.')
