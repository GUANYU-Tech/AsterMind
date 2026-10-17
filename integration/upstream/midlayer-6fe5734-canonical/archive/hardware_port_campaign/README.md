# H200 port campaign one-off appliers archive

本目录保存 2026 年 9 月 H200 WP1–WP7 port campaign 的一次性写入脚本，仅用于追溯 `configs/hardware/nvidia_h200.yaml` 各机制参数当时的落地方式。七个 `apply_nvidia_h200_wp{1..7}_references.py` 各自把对应 work package 的 port report 数值写入硬件 YAML；数值本身、推导过程与证据保留在活动目录 `microbench/nvidia_h200/references/port_reports/`，写入器不会再运行，于 2026-10 归档。活动代码不得 import 或执行本目录内容。
