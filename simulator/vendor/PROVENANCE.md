# 本地复用来源

`src/network_sim/physical/_vendor/islsim/` 的 config.py、link_model.py、README 复制自参考目录的 `third_party/islsim`，物理公式未修改。调用侧临时绑定它使用的绝对 config import，避免污染全局模块。

来源目录：`D:/下载/astermind-feature-flowsim-web-integration/astermind-feature-flowsim-web-integration`。这是本地开发副本；不在此对外部项目的许可或可公开发布范围作推断。

默认计算后端位于 `src/astermind/ArchPrism/midlayer/upstream`。快照来自本地 `midlayer_hw_model-dev` 导出，版本和校准文件来源见该目录的 `.orbitserve_snapshot.json`。`provenance.py` 生成实际接入文件的哈希。
