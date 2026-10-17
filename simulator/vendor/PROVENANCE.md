# 本地复用来源

`islsim/` 的 config.py、link_model.py、README 复制自参考目录的 `third_party/islsim`，物理公式未修改。调用侧临时绑定它使用的绝对 config import，避免污染全局模块。

来源目录：`D:/下载/astermind-feature-flowsim-web-integration/astermind-feature-flowsim-web-integration`。这是本地开发副本；不在此对外部项目的许可或可公开发布范围作推断。

M-L 本身没有复制或修改；默认复用 `integration/upstream/midlayer-6fe5734-canonical`，来源 commit `6fe573479c37a931744a6442e7a4b3503eec02e6`。Cell、Scheduler、原生 KV block、事件协调与 FlowSim 继续通过原接入模块调用。`provenance.py` 生成实际读入文件的哈希，便于后续与持续开发版本区分。
