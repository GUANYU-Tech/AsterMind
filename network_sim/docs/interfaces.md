# 输入、输出与时钟约定（v0.2）

## 公共接口

| 接口 | 输入／结果 |
|---|---|
| Scenario.load | 配置文件，严格校验设备、连接、可选建链策略 |
| PhysicalTopology | 产生 AsterMind canonical graph、FlowSim 执行图、设备编号映射 |
| topology.view | cells / satellites / devices；配置图含候选与禁用边，不代表当前已建链 |
| network.route | 起终点、可选字节数；当前路径、资源、孤立估计；无当前路径抛 ValueError |
| network.candidate_path | 忽略可见性／捕获的结构路径；永久 enabled=false 仍排除 |
| network.submit | TransferRequest，返回 ID；动态暂不可用会等待，候选不可达则拒绝 |
| network.next_event_time_ns | 下次就绪、检查、失效、捕获、发送／到达事件时刻；无事件 None |
| network.advance_to | 单调整数 ns，返回期间新增 CompletionEvent / FailureEvent，按时刻排序、仅一次 |
| network.snapshot | 任务状态、成功／失败、资源统计、动态连接状态和变化历史 |
| network.run | 静态可不设参数；动态必须 until_ns；集成时推荐外部时钟 |
| network.close | 释放 native 状态，推荐 with |

同一共享场景只用一个单线程运行时，三类来源共同竞争。不要分别建 TP 网络和 KV 网络后期待互相竞争。

## 提交与反馈

TransferRequest 必填 id、source、target、size_bytes。可填 ready_time_ns、origin（multi_gpu / orbitserve / midlayer / test）、operation、correlation_id。起终点是物理设备 ID，非 rank / Cell ID；大小是实际 byte，非 token 数。

每次传输唯一 ID；不猜测两笔内容相同的任务是否重复。只提交前置步骤已经完成的操作，或明确给未来就绪时刻。动态状态为 scheduled → waiting_link → transmitting → propagating → completed，waiting_link 可省略；开始后断链可转 failed。等待无自动截止／取消，上层须避免无限等待。

CompletionEvent(kind=completed) 包含来源、业务操作、关联 ID、起终点、字节数、completion_time_ns 和路径。收到后才能推进依赖计算或提交 KV 迁移。

FailureEvent(kind=failed) 同样携带业务信息，另有 failure_time_ns、reason、remaining_bytes、路径。上层决定报错或重试，不当作成功；remaining_bytes 不是已确认续传偏移。每任务只通知一次终态。大跨度推进可能一次返回多个事件；有依赖时应逐事件驱动。

## 三类输入方

多 GPU 操作层负责并行组、PD／AF 角色、rank→GPU、集合通信算法及步骤依赖；网络只执行已展开点对点操作，不因 operation=all_reduce 自动展开。物理 Ring 中间 GPU 的收发步骤须上层显式提出，GPU 不是隐式路由器。

O-S 管理请求、KV 实际位置、目标 HBM 预留、迁移成功后的源释放、失败回滚。网络不改 HBM。多 GPU 分布的 KV 应拆成明确传输；请求、输出和 KV 使用同一接口。

M-L 保留本地计算访存。只把选定的跨设备传输交给外部网络；同一次操作不能重复计 M-L 通信时间，也不能由多 GPU 操作层和 M-L 重复提交。

三个真实适配器尚未编写，本版提供共同契约和独立运行模块。

## 单位、计时和边界

十进制 Gbps、整数 byte、整数 ns，最低带宽 1e-6 Gbps。native 内部保留分数 ns，公开事件向上取整。拒绝未知字段、非法值、过去时刻。

确定性最小传播延迟路径，开始后固定。孤立耗时 = 字节量×8／路径最小 Gbps + 沿途 latency_ns 总和；流水模型不在每跳串行重发整份数据。孤立估计不含竞争／建链等待，不是预约。实际以事件为准。零字节沿用 FlowSim 即刻完成；同设备任务交给本地模型。

bidirectional=true 是两个方向各有独立链路容量；Switch／终端 capacity_gbps 是经过设备的双向总共享容量。未配设备容量时只限每条边；不自动推断 NVSwitch 参数。

资源 bytes_transferred 按已结束发送的流记账；在途／发送中失败的部分字节未进入累计值。进度看 remaining_bytes / rate_gbps，多个资源计数不可相加当业务数据量。配置图不可变，动态状态归 controller。

尚无取消、优先级、指定路径、动态带宽修改、续传、包级协议、surrogate 校准。窗口和周期检查关系见 link-policy-reference.md。
