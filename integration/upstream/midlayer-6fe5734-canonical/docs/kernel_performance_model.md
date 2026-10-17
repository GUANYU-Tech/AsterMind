# Midlayer Kernel 解析性能模型

Midlayer 的 Kernel 层回答一个局部但基础的问题：给定算子形状、离散 launch 决策和目标硬件， 如何在不使用目标 kernel 执行时间拟合的前提下，预测一次调用的执行时间及其物理瓶颈。

本文描述模型的设计思想、完整求解过程和证据边界。从生产请求到算子调用的过程见 [Serving 系统设计](serving_system.md)，FlashAttention 已验证机制、失败方案与剩余误差见 [FlashAttention 解析模型精度与机制记录](flash_attention_accuracy.md)；参数的证据等级与发布要求见本文第 10 节。

第一次阅读可以沿着一条固定主线理解本文：

```text
算子形状与 launch
  → 计算操作数和各级物理流量
  → 计算资源时间与存储服务率
  → 有限并发下的实际完成率
  → worker/tile 流水
  → launch floor 与 kernel 总时间
```

这条主线刻意把“做多少工作”和“硬件多快完成”分开。前者由算子、layout 和 launch 决定， 后者由规格和独立机制微基准决定；目标 kernel 的实测时间只用于最终评价，不能倒流进公式。

## 1. 白盒解析建模

Kernel 预测不是把 shape 映射到历史时间，而是先解释 kernel 做了什么，再计算这些工作在目标硬件上需要多久：

```text
Operator shape + Launch + Hardware
                 │
                 ▼
       Compute / Memory / Sync demand
                 │
                 ▼
      Resource service and dependencies
                 │
                 ▼
       Pipeline and launch composition
                 │
                 ▼
 Duration + Bottleneck + Evidence limitations
```

这种方式的目标不只是给出一个时间，还要回答：

- 算子产生了多少计算、访存和同步工作；
- 哪个资源形成下界；
- 哪些阶段可以重叠，哪些必须串行；
- 哪些结论来自规格、源码或独立微基准；
- 哪些硬件机制尚未闭合，结果应如何降级解释。

解析模型允许误差存在。若误差无法由独立物理证据解释，应报告限制，而不是从目标数据反解一个 per-shape、per-kernel-name 或 kernel-class 时延修正。

## 2. 输入、输出与职责边界

### 2.1 模型输入

一次预测需要三类信息：

1. **算子语义**：kernel kind、shape、dtype、layout、variant 以及运行时相关的长度或路由。
2. **Launch 语义**：algorithm、tile、grid、block、split、workspace、worker 分配和静态资源占用。
3. **硬件语义**：计算单元、时钟、吞吐、存储层级、带宽、请求能力、并发容量和 launch 机制。

算子层描述“需要完成的数学工作”，launch 描述“工作如何离散映射到硬件任务”，硬件规格描述 “每种物理资源能够提供什么服务”。三者不能互相代替：shape 不能猜 tile，kernel name 不能代替完整 launch，峰值带宽也不能解释未知的请求并发。

### 2.2 模型输出

预测结果包含：

- 总执行时间；
- 各计算和存储资源的时间下界；
- pipeline 或 worker makespan；
- launch/empty/non-empty task 下界；
- 主要瓶颈；
- 支持等级和未闭合机制。

Evaluation 可以比较这些中间量与 profiler observation，但不能在外围重新实现另一套公式。 Serving 只消费单次调用的预测结果，不把请求级真值反馈给 Kernel 模型。

### 2.3 模型适用边界

Kernel 层预测一次完整调用在一组明确资源上的 device execution time。默认不包含多租户、多个互不相关 kernel 的并发、跨请求 persisting cache、数据压缩或稀疏跳过；这些机制只有在 workload 和硬件合同都显式描述时才能加入。跨 kernel 的先后与资源竞争由 Serving/执行时间线负责，不能隐式折进单 kernel 效率。

资源分区均匀、cache set 均匀或 task 工作量均匀都不是全局默认。只有地址映射、launch/source 或独立 observation 能够证明时，才可以使用均匀分配；否则必须保留不均衡 limitation，或使用明确的 per-partition/per-worker 工作量。

## 3. 从算子到资源需求

时间模型首先构造与时间无关的资源需求。需求回答“做多少工作”，硬件能力回答“以什么速率完成”；在这一步不应出现由目标执行时间反解的效率。

### 3.1 计算需求

计算工作按真实执行资源分开，例如 Matrix/Tensor/Cube、Vector、Reduction、SFU 或普通 ALU。每类资源记录：

- 语义操作数；
- compiler lowering 引入的额外操作；
- tile padding 和尾部工作；
- worker 之间的工作分配；
- 必须串行的特殊指令或完整请求。

同一个数学操作在不同硬件上可能映射到不同资源。例如 softmax 可以同时包含普通 Vector、 Exp、Div 和 reduction；不能把它们全部压成一个没有物理意义的“算力效率”。只有在源码或静态 lowering 能够闭合操作数量时，才可以使用更细的 service。

### 3.2 存储需求

存储建模区分三个容易混淆的量：

- **逻辑数据量**：算法上必须读取或写回的数据；
- **source-issued 流量**：执行程序实际发出的 load/store 请求；
- **物理层级流量**：经过 coalescing、cache、replay 和 transaction 粒度后，各层真正处理的 bytes 和 transactions。

一个典型 cache 路径可以写成：

```text
Producer → L1 → L2 → DRAM
```

但并非所有硬件都具有同样的层级，也不是所有层级都能获得完整证据。NPU 可能表现为 DMA/MTE、片上 SRAM/UB 和 GM；某些 GPU 路径只能从 L2 开始建模。模型按实际能力选择节点， 而不是为了公式对称补出不存在或未知的 cache。

每级流量应从地址映射、请求粒度、条件 miss、multicast/replay 和唯一数据 footprint 推导。 Cache hit rate 只描述请求路由，不能单独把逻辑 bytes 变成物理 traffic；当事务粒度发生变化时， 必须重新统计该层的 transaction 数和 bytes。

### 3.3 同步和中间状态

Barrier、semaphore、reducer、split scratch、workspace 和 producer/consumer buffer 会改变执行顺序和数据生命周期。它们只有在 source 或 ABI 能够证明时才进入模型。

例如 split-K 或 split-attention 可能包含主 kernel、partial workspace 和 combine kernel。 Split 数本身不能证明 partial 是否到达 DRAM、使用什么 dtype 或何时可被 reducer 消费；这些必须由 launch sequence、pointer dtype 和物理 traffic 证据分别闭合。

## 4. Compute 时间

对每类计算资源分别统计工作量。Matrix、Vector、Reduction 和 SFU 即使在同一个 kernel 中出现， 也不能先合成一个总 FLOPS，因为它们使用不同的执行管线。对资源 $k$：

$$
P_k^{\mathrm{eff}}
=N_{\mathrm{active}}P_k^{\mathrm{unit}}
\eta_{k,\mathrm{shape}}
\eta_{k,\mathrm{schedule}}
$$

$$
T_k =
\frac{OP_k}
{P_k^{\mathrm{eff}}}
$$

其中：

- $OP_k$：映射到资源 $k$ 的操作数；
- $N_{\mathrm{active}}$：当前 wave 中实际参与工作的 SM、AI Core 或其他物理单元数；
- $P_k^{\mathrm{unit}}$：单个物理单元对该 typed operation 的吞吐；
- $\eta_{k,\mathrm{shape}}$：tile padding、对齐和尾块造成的结构效率；
- $\eta_{k,\mathrm{schedule}}$：有源码、静态 schedule 或独立机制证据的发射、依赖和同步效率；
- $P_k^{\mathrm{eff}}$：这个 kernel 对资源 $k$ 实际可用的全芯片吞吐。

峰值吞吐只在操作类型、数据类型、时钟域和执行域一致时有效。Resident block/warp 增加可并发 producer 和 latency hiding，不会凭空复制一份计算 pipe。低 occupancy 也不能自动线性变成 compute efficiency；若要对指令发射或依赖收费，必须有静态 instruction schedule 或独立机制证据。

当源码能够给出逐 worker 的 typed service 序列时，应在同一 worker 内按真实依赖串行组合， 再跨 worker 取最大值。缺少 worker mapping 时保持 aggregate 下界，不能根据平均分布猜测 critical worker。

当同一物理资源包含多种 typed service，且源码已经给出逐 worker 的执行序列时，不能把这些 service 的全芯片时间各自取最大值，也不能压成一个平均效率。此时按 worker 串行求和，再取最慢 worker：

$$
T_{\mathrm{typed}}
=\max_w\sum_s
\left(
\frac{OP_{s,w}}{P_s^{\mathrm{unit}}}
+N_{s,w}\tau_s
\right)
$$

$OP_{s,w}$ 是 worker $w$ 上 service $s$ 的操作数，$N_{s,w}$ 是不可再分的离散请求次数， $\tau_s$ 是该请求由独立单机制微基准测得的服务时间。缺少逐 worker schedule 时继续使用 aggregate 下界，不能猜测指令数或 worker 分配。

如果离散请求的服务时间随请求大小变化，并且独立机制微基准给出了有物理单位的 exact service 函数 $L_s(q)$，则使用：

$$
T_{exact}=\max_w\sum_{q\in Q_w}L_s(q)
$$

它仍然要求逐 worker 的请求集合 $Q_w$ 来自 source/launch，而不是按 kernel name 查目标时延。

Occupancy 决定一个物理单元上可同时存在多少 producer，但不会复制该单元的 compute pipe。若线程、寄存器和片上存储的分配语义已知，每单元 resident task 上限为：

$$
B_{res}
=\min\left(
B_{arch},
\left\lfloor\frac{T_{cap}}{T_{task}}\right\rfloor,
\left\lfloor\frac{R_{cap}}{R_{task}}\right\rfloor,
\left\lfloor\frac{S_{cap}}{S_{task}}\right\rfloor
\right)
$$

对 GPU，活跃 warp/wave 与可生成的独立存储请求可进一步写成：

$$
W_{active}
=\min(W_{arch},B_{res}W_{task}),\qquad
Q_{workload}=W_{active}q_{warp}
$$

$B_{res}$ 和 $W_{active}$ 用于 producer、MLP 和隐藏延迟；计算吞吐的物理单元数仍是 $N_{active}$。当尚无显式 worker schedule、task 同质且一个 task 占用一个物理单元服务流时， aggregate 路径可以单独计算尾波：

$$
n_{wave}=\left\lceil\frac{N_{task}}{N_{unit}}\right\rceil,
\qquad
\eta_{tail}=\frac{N_{task}}{n_{wave}N_{unit}}
$$

此时可用 $\eta_{tail}$ 修正 aggregate full-chip service。不能把 $B_{res}$ 再乘入 $N_{unit}$： resident task 共享同一条物理 compute pipe。显式 worker makespan 已经包含尾波时，也不能再次使用 $\eta_{tail}$。

## 5. Memory 时间

Memory 时间同时受流量、服务速率、基础延迟和有限并发约束。只使用 `bytes / bandwidth` 会在小请求、低并发、cache-resident 或 transaction-rate 受限场景中丢失关键机制。

### 5.1 从逻辑数据量得到物理事务

以普通 cache 路径为例：

```text
Producer → L1 → L2 → DRAM
```

若每次 miss 产生一个等粒度的下级事务，则：

$$
N_{L1}=N_0,\qquad
N_{L2}=m_1N_0,\qquad
N_{DRAM}=m_1m_2N_0
$$

$$
D_i=N_iB_i
$$

- $N_0$ 是 producer 产生的初始物理请求数；
- $m_1$ 是“请求已经到达 L1”条件下的 L1 miss rate；
- $m_2$ 是“请求已经到达 L2”条件下的 L2 miss rate；
- $N_i$、$B_i$ 和 $D_i$ 分别是层级 $i$ 的事务数、事务大小和物理字节数。

条件 miss rate 只决定有多少请求继续向下路由。若 coalescing、拆分、多播或 replay 改变事务粒度，应由地址映射直接统计各级 $N_i$ 和 $B_i$，不能用 miss rate 乘逻辑 tensor bytes 冒充物理 traffic。DMA/MTE/TMA 路径也不必强行经过 cache；只要把真实经过的节点和流量列出来， 后面的求解方式相同。

### 5.2 路径延迟与请求生命周期

低负载 pointer-chasing 或等价机制微基准测得的是一条完整访问路径的端到端延迟。例如：

$$
L_{L1}^{0,\mathrm{test}},\quad
L_{L2}^{0,\mathrm{test}},\quad
L_{DRAM}^{0,\mathrm{test}}
$$

只有当 producer、指令、事务粒度、地址映射和返回路径一致时，才可以近似用相邻路径差分得到每一级自身的无排队增量：

$$
\Delta t_{L1}^{0}=L_{L1}^{0,\mathrm{test}}
$$

$$
\Delta t_{L2}^{0}
\approx L_{L2}^{0,\mathrm{test}}-L_{L1}^{0,\mathrm{test}},\qquad
\Delta t_{DRAM}^{0}
\approx L_{DRAM}^{0,\mathrm{test}}-L_{L2}^{0,\mathrm{test}}
$$

在当前负载下，层级 $i$ 的增量响应时间为：

$$
\Delta t_i=\Delta t_i^0+t_{q,i}
$$

于是一个初始请求平均占用 source scoreboard、load queue 或等价在途槽位的时间为：

$$
\boxed{
t_{req}
=\Delta t_{L1}
+m_1\Delta t_{L2}
+m_1m_2\Delta t_{DRAM}
}
$$

这里的权重非常重要：所有请求都支付 L1 路径，只有 L1 miss 支付 L2 路径，只有 L2 miss 继续支付 DRAM 路径。$t_{req}$ 是 source 槽位的完整生命周期，不等于任一级局部 queue entry 的生命周期。

### 5.3 Byte/s 与 request/s

先定义该级相对于初始请求的路由比例与到达率：

$$
\gamma_i=\frac{N_i}{N_0},\qquad
\lambda_i=\gamma_i\lambda_0
$$

若层级 $i$ 的 transaction size 为 $B_i$，持续 byte 带宽为 $BW_i^{\mathrm{sust}}$，请求能力为 $\Lambda_i^{\mathrm{req,max}}$，则该层的 transaction 服务能力为：

$$
\lambda_i^{\mathrm{cap}}
=\min\left(
\frac{BW_i^{\mathrm{sust}}}{B_i},
\Lambda_i^{\mathrm{req,max}}
\right)
$$

这同时检查字节数据通路和 lookup/controller 等事务前端。对应的可直接检查时间下界为：

$$
\boxed{
T_i^{\mathrm{cap}}
=\max\left(
\frac{D_i}{BW_i^{\mathrm{sust}}},
\frac{N_i}{\Lambda_i^{\mathrm{req,max}}}
\right)
=\frac{N_i}{\lambda_i^{\mathrm{cap}}}
}
$$

若 request/s 没有独立证据，该约束保持 absent，只使用已知 byte capacity；不能从峰值带宽和猜测的 cache-line size 伪造事务能力。若实测 sustained bandwidth 已经包含该事务模式的 request-rate 限制，则也不能重复收费。

若层级 $i$ 有 $N_{part,i}$ 个均匀使用的物理分区，等效单分区服务时间为：

$$
s_i
=\frac{N_{part,i}}{\lambda_i^{cap}}
=\max\left(
\frac{B_i}{BW_i^{part,sust}},
\frac{1}{\Lambda_i^{part,req,max}}
\right)
$$

$s_i$ 描述瓶颈服务端处理一个事务占用多久，不是端到端路径延迟 $\Delta t_i^0$。若请求并非均匀落入分区，必须使用实际 per-partition 到达率，不能继续使用这个平均式。

### 5.4 Source、局部 queue 与 BDP

workload 能产生的独立请求数与硬件 source slot 上限共同决定有效在途数：

$$
Q_{workload}=N_{producer}^{active}q_{producer},\qquad
Q_{src}=\min(Q_{workload},Q_{src}^{hardware})
$$

由 Little 定律，source 能维持的全芯片初始请求率为：

$$
\lambda_{src}(\lambda_0)
=\frac{N_{active}Q_{src}}{t_{req}(\lambda_0)}
$$

L1 MSHR、L2 miss queue、DMA descriptor 和 DRAM credit 不是 source slot 的别名。对每一个能独立识别的局部结构 $j$，需要分别写出分配比例 $\gamma_j^Q$、entry 生命周期 $\ell_j^Q$ 和容量 $Q_j^{max}$：

$$
Q_j^{used}=\gamma_j^Q\lambda_0\ell_j^Q,\qquad
\lambda_{Q,j}(\lambda_0)
=\frac{Q_j^{max}}{\gamma_j^Q\ell_j^Q(\lambda_0)}
$$

例如 L1 miss entry 只由 L1 miss 分配，其生命周期通常覆盖后续 L2/DRAM 路径；不能把所有初始请求都计入，也不能把同一个物理 entry 同时当成两级 queue。

请求前端本身还可能形成发射上限：

$$
\lambda_{frontend}
=N_{active}r_{frontend}^{unit}
$$

$r_{frontend}^{unit}$ 是单个活跃物理单元每秒最多生成的初始物理事务数，由 load/store issue、指令混合和每条指令产生的事务数共同决定。缺少独立证据时该约束 absent，不能从 achieved bandwidth 反推。

无排队时，source 的完整请求寿命为：

$$
t_{req}^{0}
=\Delta t_{L1}^{0}
+m_1\Delta t_{L2}^{0}
+m_1m_2\Delta t_{DRAM}^{0}
$$

把层级 $i$ 跑到服务容量所需的 source slot 和局部结构 $j$ 的 entry 数分别为：

$$
\boxed{
Q_{need,i}^{src,total}
=\frac{\lambda_i^{cap}}{\gamma_i}t_{req}^{0},
\qquad
Q_{need,j\leftarrow i}^{level}
=\gamma_j^Q
\frac{\lambda_i^{cap}}{\gamma_i}
\ell_j^{Q,0}
}
$$

这就是 bandwidth-delay product 检查：容量很高但 source slot、MSHR、descriptor 或 credit 不足时，硬件仍无法达到峰值带宽。两种作用域必须分别比较，不能用一个总在途数同时代表 source 和所有局部 queue。

### 5.5 自洽完成率

实际完成率同时受以下约束：

- 各层 byte/request 服务能力；
- producer 能产生的独立请求数；
- source 的硬件槽位；
- 各级局部 queue/credit；
- frontend 发射能力。

先把各级服务能力换算回初始请求域：

$$
\lambda_0^{cap}
=\min_{i:\gamma_i>0}
\frac{\lambda_i^{cap}}{\gamma_i}
$$

再把 frontend、source 和所有局部 queue 合成：

$$
\phi(\lambda)
=\min\left[
\lambda_{frontend},
\lambda_{src}(\lambda),
\min_j\lambda_{Q,j}(\lambda)
\right]
$$

用无排队的 $t_{req}^0$ 和 $\ell_j^{Q,0}$ 计算容量点的有限结构上限：

$$
\phi^0
=\min\left[
\lambda_{frontend},
\frac{N_{active}Q_{src}}{t_{req}^0},
\min_j
\frac{Q_j^{max}}{\gamma_j^Q\ell_j^{Q,0}}
\right]
$$

若 frontend 或某类局部结构没有独立证据，对应项从最小值中删除，等价于不让未知约束伪造一个有限上限。

若某一级明确声明使用 M/D/1 queue capability，可先由 $\rho_i=\lambda_i/\lambda_i^{cap}$ 和等效单事务服务时间 $s_i$ 计算：

$$
t_{q,i}
=\frac{\rho_i}{2(1-\rho_i)}s_i,
\qquad 0\le\rho_i<1
$$

这是带有明确适用假设的解析 queue model，不是拟合曲线。没有该 capability 的硬件不使用此式。

用无排队生命周期计算 $\phi^0$。若有限结构足以覆盖容量点，即 $\phi^0\ge\lambda_0^{cap}$，则完成率精确取容量；否则在容量点以下求闭环：

$$
\boxed{
\lambda_0^*
=
\begin{cases}
\lambda_0^{cap},
&\phi^0\ge\lambda_0^{cap},\\[4pt]
\text{满足 }\lambda=\phi(\lambda)
\text{ 的物理解},
&\phi^0<\lambda_0^{cap}.
\end{cases}
}
$$

之所以需要闭环，是因为 $\lambda_0$ 决定每一级利用率与排队，排队延长 $t_{req}$ 和局部 entry 生命周期，生命周期又反过来降低有限槽位能维持的请求率。没有 queue capability 时，queue wait 保持未知；求解只使用已有 capacity/source/credit 约束，不能默认套用 M/D/1、固定 $\rho=0.99$ 或任意截断。

稳态存储时间最终为：

$$
T_{memory}^{steady}=\frac{N_0}{\lambda_0^*}
$$

当没有可证明的 source/queue 上限时，模型退化为已有的 byte/request capacity，并明确报告证据缺口，而不是补默认并发参数。

### 5.6 首请求与短工作

短 tile 不能假设启动延迟已经被稳态吞吐摊薄。若首请求延迟为 $L_{first}$，后续请求以稳定速率排出，一个 memory stage 可以近似写成：

$$
T_{memory}^{tile}
=L_{first}+\frac{\max(N_{stage}-1,0)}{\lambda_0^*}
$$

首请求延迟必须来自相同 producer、指令和目标层级的低负载机制测量。不能用完整 kernel 的短时延残差代替，也不能在 pipeline 和 launch floor 中重复收费。

### 5.7 特定访问模式的 Memory service

有些访问模式即使具有相同 bytes，也不能由连续 streaming bandwidth 表达，例如间接 gather、强依赖链或固定事务形状。只有独立机制微基准能够隔离这种差异时，模型才可以为一部分 ingress 流量声明有类型的 memory service：

$$
T_{access,s}=\sum_i\frac{D_{logical,i}}{C_{s,i}}
$$

$C_{s,i}$ 必须绑定访问类型、dtype、事务或 tile signature、作用域和证据；它不能按 kernel name 绑定，也不能跨未知 signature 外推。同一份流量只能由一个 service 收费，typed service 也不自动改变 L2/DRAM physical bytes、仲裁或 worker pipeline。缺少匹配证据时保留通用 memory 下界和 limitation，不能选择“最接近”的容量。

### 5.8 Byte-only 退化

部分硬件 profiler 只能提供 aggregate bytes，无法闭合 transaction、first-request 或 credit。此时 path 显式使用 byte rate。若 $D_0$ 是 source bytes、$\gamma_i^B=D_i/D_0$，则：

$$
\lambda_0^{B,cap}
=\min_i\frac{BW_i^{sust}}{\gamma_i^B},\qquad
T_{memory}=\frac{D_0}{\lambda_0^{B,cap}}
$$

该分支仍能给出可靠的 aggregate 带宽下界，但不能消费 request/s ceiling、source slot、local queue 或 first-request latency，也不能自行推导 transaction-level latency hiding 或排队。如果 source/launch 另外提供了逐 worker 的 absolute-byte stage 和依赖，pipeline 仍可使用这些 byte stage；但不得从 byte-only observation 反推出请求数、请求生命周期或 queue。结果应标记为 aggregate-only，而不是用默认 transaction size 补齐公式。

### 5.9 共享读写资源

若读写最终共享同一个半双工 DRAM/NoC 资源，两条 path 不能各自计时后简单取最大值。若读、写方向容量分别为 $C_R$ 和 $C_W$，当前不需要方向顺序即可确定的下界为：

$$
\boxed{
T_{shared}^{lb}=\frac{D_R}{C_R}+\frac{D_W}{C_W}
}
$$

只有一个 common capacity $C$ 时，退化为 $(D_R+D_W)/C$。同卡复制 $B$ bytes 同时产生 source read 和 destination write，因此收费为 $B/C_R+B/C_W$，不能只计算一次逻辑 payload。

若 launch/source 能证明 controller-visible 的全局方向 epoch $\mathcal E$，才可以进一步写成：

$$
T_{shared}
=\sum_{e\in\mathcal E}
\left(
\frac{D_{R,e}}{C_R}
+\frac{D_{W,e}}{C_W}
+I_{RW}(e)\tau_{RW}
+I_{WR}(e)\tau_{WR}
\right)
$$

单个 kernel 的 read/write 比例不能证明控制器看到的 epoch 顺序，也不能从混合带宽下降反解每次方向切换常数。缺少顺序证据时，只发布前面的 service-sum 下界。

## 6. Worker、依赖与流水

资源下界描述总工作量，pipeline 描述工作如何在时间上展开。一个可执行的流水需要知道：

- task/tile 到 worker 的映射；
- 每个 task 内的 stage 和依赖；
- stage 使用的物理资源；
- producer/consumer buffer 的数量和复用时刻；
- shared resource 的仲裁；
- persistent queue 或动态领取顺序。

对一个简单的 Load–Compute–Store 流水，如果三个阶段使用独立资源且 buffer 足够，稳态启动间隔为：

$$
II=\max(T_{load},T_{compute},T_{store})
$$

若 worker $w$ 执行 $n_w$ 个 task，其 makespan 可写成：

$$
M_w=T_{fill}+(n_w-1)II+T_{drain}
$$

设备执行时间由 critical worker 决定：

$$
T_{execution}=\max_w M_w
$$

真实 pipeline 往往比三阶段模型复杂。FlashAttention 包含 K/V load、QK、softmax、PV 和 store， 其中 Matrix 资源可能被 QK/PV 共享，K/V buffer 具有非对称生命周期。MoE 的 routed rows 和 expert 分布会产生不均衡 worker。只有在 dependency、buffer 和 worker mapping 同时闭合时， 才允许把资源时间重叠；否则保留 aggregate lower bound。

以 FlashAttention 为例，同一个 KV tile 内存在依赖：

```text
QK → online softmax/update → PV
```

因此一个 tile 的依赖延迟是：

$$
L_{compute,tile}=T_{QK}+T_{softmax}+T_{PV}
$$

但相邻 tile 可以在不同资源上交错。若 QK/PV 共享 Matrix pipe，softmax 使用 Vector/SFU，且 load/store 有独立资源与足够 buffer，则稳态启动间隔为：

$$
II_{FA}
=\max\left(
T_{load},
T_{QK}+T_{PV},
T_{softmax},
T_{store}
\right)
$$

同一 tile 的依赖决定加法，同一物理资源的占用也必须相加；只有跨 tile、资源独立且 buffer 与 barrier 允许时才取最大值。`add` 或 `max` 来自执行 DAG，不来自哪种组合让误差更低。

异步搬运或双缓冲还必须满足片上容量条件：

$$
\boxed{
C_{buffer}\ge n_{buf}S_{tile}
}
$$

$C_{buffer}$ 是该流水实际可用的 SRAM/SMEM/UB 容量，$n_{buf}$ 是同时保留的 buffer 份数， $S_{tile}$ 是一份 tile 的存储量。容量不足、buffer 生命周期重叠或 barrier 不允许时，Load 与 Compute 不能按理想双缓冲重叠。

一旦显式流水已经计算尾部和 critical worker，就不能再额外乘 tail-wave efficiency 或增加一个经验 pipeline floor。不同表达方式必须守恒同一份工作，不能重复计价。

## 7. Launch 与短 kernel

完整 kernel 时间同时受到实际资源执行和 runtime task 下界约束：

$$
T_{kernel}=\max\left(
T_{empty},
T_{nonempty,min},
t_{dispatch}+T_{execution}
\right)
$$

- $T_{empty}$ 描述没有有效工作时的完整提交和调度任务；
- $T_{nonempty,min}$ 描述执行最小非空工作所需的机制下界；
- $t_{dispatch}$ 只有在能与完整 task duration 独立识别时才作为 additive 项；
- $T_{execution}$ 包含计算、存储、同步和流水 fill/drain。

如果微基准测到的是完整 empty/non-empty task duration，它应作为 max floor，而不是再与执行时间相加。Host elapsed、device event、profiler replay time 和 kernel event 属于不同计时域， 未经证明不能互换。

短 kernel 对 grid、block、task type 和 runtime 版本敏感。合法的 launch floor 必须来自与目标 timer domain 一致、与 production workload 独立的机制微基准，并明确适用范围；不能根据 operator name 选择不同经验常数。

## 8. 常见算子的实例化

### 8.1 GEMM

对 $[B,M,K]\times[B,K,N]$ 的 batched GEMM，语义 Matrix 工作量为：

$$
OP_{GEMM}=2BMNK
$$

若 launch tile 为 $(T_M,T_N,T_K)$，真正执行的工作还要考虑 $M/N/K$ padding、split 和 reducer； 不能始终把语义 FLOPs 当成实际 issued work。最低逻辑 traffic 包括 A、B 读取与 C 写回：

$$
D_{logical}=(BMK+BKN+BMN)d
$$

$d$ 是元素字节数。Cache reuse、swizzle 和 tile 重读决定它如何变成 L2/DRAM physical traffic。

若 K 方向有 $S>1$ 个 split 且 ABI 明确存在独立 reducer，令 $E=BMN$、partial 元素宽度为 $d_p$，则 main 至少产生 $SEd_p$ 的 partial，reducer 至少读取这批 partial 并写出 $Ed$ 的最终结果。partial 是否进入 DRAM、在 L2 保留还是由片上协议消费，必须由 cache policy、ABI 或 physical traffic 证据决定；不能仅从 `SP=S` 推导。

大 GEMM 常由 Matrix 或 DRAM capacity 约束；小 GEMM 更容易暴露 launch、尾波和并发不足。

### 8.2 FlashAttention

设 batch 中请求 $b$ 的实际 query/KV 长度为 $L_{Q,b}$、$L_{KV,b}$，query head 数为 $H_Q$， head dimension 为 $d$。忽略 causal mask 削去的区域时，QK 与 PV 的主要 Matrix 工作量为：

$$
OP_{FA}
\approx 4H_Qd\sum_b L_{Q,b}L_{KV,b}
$$

系数 4 来自 QK 和 PV 两次乘加各计 2 operations。实际模型不能一律使用这个全矩形近似： causal/prefix mask 应按每个 query 真正可见的 key 数求和。对 bottom-right causal，若 query 是长度为 $L_Q$ 的 KV 后缀，第 $q$ 个 query 的可见 key 数为：

$$
V_q
=\min\left(
L_{KV},
L_{KV}-L_Q+q+1
\right)
$$

并将负值截为 0。于是 QK/PV 工作量与 $\sum_q V_q$ 成正比，而不是固定的 $L_QL_{KV}$。这也是 `Q_LEN`、实际 `KV_LEN` 和 causal 语义必须作为 workload 输入的原因。

decode 时 K/V cache 的最低逻辑读取量为：

$$
D_{KV}=2\sum_b L_{KV,b}H_{KV}d\,d_t
$$

$d_t$ 是元素字节数，系数 2 表示 K 和 V。Paged、varlen、chunked prefill 和 MLA 会改变实际长度、页表访问、head 表示和 tile 数，不能只记录最大 sequence length。

split-KV 若有 main 和 combine 两个 kernel，令 $E_Q=H_Q\sum_bL_{Q,b}$、split 数为 $S$、value dimension 为 $d_v$，FP32 partial value 加 LSE 的 scratch 至少为：

$$
D_{scratch}=E_QS(d_v+1)\times4
$$

main 写、combine 读的是同一 scratch；是否穿过 DRAM 边界仍需独立证据，不能根据 allocation 存在就自动算作 DRAM traffic。

Attention 的关键不是把各阶段时间简单相加，而是根据 K/V buffer、Matrix/Vector 共享资源、 worker 分配和 split 生命周期判断哪些工作可以重叠。

### 8.3 Fused-MoE

设输入 token 数为 $T$、top-k 为 $K_{top}$，则 routed rows 总数为：

$$
R=T K_{top}=\sum_e T_e
$$

$T_e$ 是 expert $e$ 实际接收的 row 数。若专家线性层为 $[T_e,H]\times[H,I]$，其 Matrix 工作量按 expert 求和：

$$
OP_{MoE}=\sum_e 2T_eHI
$$

门控、up/down projection 和激活按真实结构分别计数。Fused-MoE 的工作量由 routed rows、 expert ownership、top-k、tile 和 rank-local 分配共同决定。权重读取通常是大块流量，激活 gather/scatter 可能受到访问模式而非纯 streaming 带宽限制。平均 tokens per expert 不能替代 $\{T_e\}$，因为 padding 后的 tile 数、critical expert 和 worker 不均衡直接影响 makespan。若真实路由未知，必须输入可复现的分布假设或 replay；该分布属于 workload，不是通过 kernel duration 拟合出的参数。

### 8.4 Vector

Normalization、RoPE、Activation、Elementwise 和 KV-cache 虽都归入 Vector，但数据流和依赖不同。对 $E$ 个元素，elementwise add 的语义 traffic 至少是两次读取和一次写回，即 $3Ed$； mul 也是相同 tensor traffic，但 typed arithmetic 不同。Normalization 包含 reduction、归一化和可能的额外 reread，不能用单次 $Ed$ streaming 代替。RoPE 包含 Q/K、位置参数和可能的 cache 路径；KV-cache 对 batch 中每个 slot 搬运 K/V，并读取 slot mapping，实际 batch 和 layout 都是工作量输入。

模型分别统计语义操作、compiler lowering、source-issued traffic 和物理层级流量。不能通过一个统一的“elements × 常数”隐藏不同 variant 的实际读写和同步。

## 9. 跨硬件统一

统一模型不是要求 GPU 和 NPU 拥有相同结构，而是要求它们使用相同的问题分解：

```text
Work demand
→ Physical resource capacities
→ Dependency and concurrency constraints
→ Launch composition
→ Duration and diagnostics
```

GPU 可以实例化 Tensor/Vector/SFU、warp producer、cache hierarchy 和 CTA；NPU 可以实例化 Cube/Vector、DMA/MTE、片上 SRAM/UB 和 AI Core task。缺少某一级资源时删除该节点；缺少某个参数时保留 limitation。硬件差异通过 capability 和证据表达，而不是在公共公式中按硬件 ID 选择完全不同的 predictor。

同一问题分解在不同架构上的典型映射为：

| 建模问题 | GPU cache 路径 | GPU async-copy 路径 | NPU DMA/SRAM 路径 |
|---|---|---|---|
| 计算 worker | SM/CU 上的 CTA/warp | SM/CU 上的 CTA/warp | AI Core 上的 task |
| 计算资源 | Tensor/Matrix、Vector、SFU | Tensor/Matrix、Vector、SFU | Cube/MXU、Vector、Scalar |
| 数据路径 | L1→L2→HBM | HBM→NoC→TMA/DMA→SMEM | GM/HBM→NoC→DMA/MTE→SRAM/UB |
| producer | resident warp/wave | async copy stage/descriptor | CopyIn/CopyOut 或 tile queue |
| 有限结构 | scoreboard、MSHR、miss queue、DRAM credit | descriptor、copy queue、NoC/DRAM credit | MTE/DMA queue、NoC/GM credit |
| 片上约束 | register、L1/SMEM、bank | SMEM buffer、bank、barrier | SRAM/UB buffer、bank、event |
| 典型重叠 | warp latency hiding | async copy + double buffer | CopyIn/Compute/CopyOut |

表格描述的是概念对应关系，不表示每款硬件都已闭合所有节点。无 L1 就删除 L1，无 transaction 证据就使用 byte-only；没有异步搬运或足够 buffer 时按依赖串行。经典 Roofline 是本模型的一个退化情形：单计算资源、单存储容量、无 transaction/queue 证据、只计算资源时间下界。

当前主要生产硬件是 NVIDIA A800、MetaX C500 和 Ascend 910B。三者可以处于不同的证据成熟度： 某个硬件可能有完整 transaction path，另一个只有 byte-only；某个算子有 worker pipeline， 另一个只有 aggregate demand。统一接口不等于统一支持等级。

## 10. 参数与证据

Production Forward 允许使用的证据分为四类：

1. **规格与架构事实**：单元数量、时钟域、理论执行宽度、容量和明确的硬件语义。
2. **源码与静态 lowering**：操作数量、tile、worker mapping、依赖、buffer 和离散 launch。
3. **独立机制微基准**：计算吞吐、带宽、基础延迟、请求率、并发容量和 launch/task 开销。
4. **非 duration observation**：实际 variant、grid/block/split、traffic、occupancy、cache hit、 pipe duty 等，用于验证或闭合资源需求。

证据的成熟度与 production 使用方式必须明确区分：

| 等级 | 含义 | Production 使用方式 |
|---|---|---|
| Official/static | 规格、架构、ISA、编译器或算子源码事实 | 可表达架构容量、工作量和离散 launch |
| Measured mechanism | 独立、可复测的单机制微基准 | 可表达带单位、带适用域的实际机制参数 |
| Observation | 真实 launch 或非 duration profiler 证据 | 可验证或补全离散 launch 与资源需求 |
| Candidate | 物理量合理，但 provenance、consumer 映射或适用域未闭合 | 不得冒充完整支持，只能保留为候选 |
| Unsupported | 没有合法证据或组合语义尚未定义 | 参数保持缺失，结果携带 limitation 或直接失败 |

证据等级描述来源成熟度，不描述预测精度。一个参数即使显著降低 MAPE，只要来源不合法，仍然不能进入 Forward；一个合法机制参数也不能因为当前数据集误差没有改善就被撤回。

不同机制应由针对性的独立实验提供，不能用一个完整 operator benchmark 同时“测出”所有参数：

| 机制实验 | 主要控制变量 | 可提供的参数或证据 |
|---|---|---|
| Matrix/Vector/SFU 独立长循环 | typed operation、依赖链、dtype、active unit | 单单元吞吐、固定 typed service、结构效率 |
| 低负载 pointer chasing | footprint、stride、目标命中层级 | 端到端路径延迟；条件成立时可做相邻层差分 |
| 分层 streaming | cache-resident/超 cache footprint、方向、足够 MLP | 各层 sustained byte capacity |
| Transaction-rate sweep | transaction size、访问类型、足够 MLP | request/s ceiling 与 byte/s/request/s 瓶颈区分 |
| Frontend issue sweep | 独立 load/store 数、每指令事务数 | $r_{frontend}^{unit}$ |
| MLP/credit sweep | producer 数、独立链数、目标层级 | source slot 或可辨识的局部 queue/credit 容量 |
| Latency-load sweep | offered load、producer 数 | 只验证解析 queue 假设，不拟合 production workload |
| SRAM/SMEM/UB sweep | stride、bank、元素宽度、buffer 数 | 片上容量、bank-pattern service 与重叠条件 |
| DMA/TMA/MTE sweep | tile size、in-flight depth | setup、bandwidth、descriptor/credit 和 overlap capability |
| Empty/minimal task | grid、block/task 类别、timer domain | empty/non-empty runtime floor；不能复刻 production operator |

若一个拐点无法区分 source slot、MSHR 或 DRAM credit，只能发布一个明确作用域的 effective constraint，不能把同一测量同时写入多个参数。

进入 production 的经验参数至少要记录：canonical 名称、物理含义、单位、硬件/软件/时钟域、适用 dtype/variant/shape、证据类型、可复测 provenance、样本与聚合方法、consumer 及组合方式、缺失或越界行为。只有离散测量点且没有物理缩放定律时必须 exact lookup，不得插值或外推。

目标 production kernel、完整算子或 Serving workload 的 duration 不得用于训练、回归、插值、阈值选择或 residual correction。Holdout、cross-validation 或改称 calibration 都不能改变这一边界。

微基准必须隔离一个有物理意义的机制，并输出带单位、适用域和 raw provenance 的参数。例如 streaming benchmark 可以测 sustained bandwidth，pointer chasing 可以测路径 latency，空 task 可以测 runtime scheduler floor；复刻完整 production 算子并把其时间写入参数表仍然是拟合。

具体数值和 raw provenance 由 hardware config、launch data、microbenchmark report 及其 manifest 维护。设计文档不保存 binary hash、case 数、实验日期或逐硬件常量，以避免把某次实现快照误当成模型原理。

## 11. Diagnostics 与支持等级

一个可解释预测应能给出：

- 每类 compute work 和时间；
- 每条 memory path 的 traffic、capacity、latency/concurrency 状态和瓶颈；
- worker/pipeline 的 critical path；
- launch floor 与资源执行的组合方式；
- 哪些参数缺失或只形成 lower bound。

Prediction contract 使用四种精确名称区分物理机制闭合程度：

- **`complete`**：该路径所需语义全部闭合，不携带 limitation；
- **`partial`**：存在 source/launch 支撑的细粒度机制，但仍有明确 limitation；
- **`aggregate_only`**：只能可靠发布 aggregate demand 或资源下界，不能把缺失的 transaction、worker 或 pipeline 语义伪装成完整模型；
- **`compatibility`**：使用面向兼容硬件或旧抽象的显式简化路径，不等同于三种生产硬件上的完整解析能力，必须携带 limitation。

`unsupported` 不是第五种返回等级。缺少必要 launch、shape 或硬件参数、无法产生有意义预测时， 模型必须在边界直接报错。

模型不能在证据不足时静默退回旧公式或默认硬件。返回数值时也必须同时暴露 limitation，避免调用者把解析下界误解为完整预测。

## 12. 验证方法

验证应沿模型链逐层进行，而不是只比较最终时间：

1. shape、tile、worker 和 launch 是否正确；
2. operations、bytes 和 transactions 是否守恒；
3. compute/memory resource demand 是否与非 duration observation 一致；
4. pipeline、buffer 和依赖是否与源码时间线一致；
5. 最终 duration 和 bottleneck 是否与独立真值一致。

每一层都应尽量比较同一个物理量：

| 模型输出 | 可比较的独立 observation | 检查的问题 |
|---|---|---|
| $N_i,D_i,\gamma_i$ | transaction/sector、physical bytes、条件 hit/miss | 地址映射、事务粒度和层级路由是否正确 |
| $L_i^{0,test},\Delta t_i^0$ | 低并发、相同 producer 的路径延迟 | 延迟差分是否成立 |
| $\lambda_i^{cap},\rho_i$ | achieved byte/s 与 request/s | 服务容量和瓶颈端口是否正确 |
| $Q_{src},Q_j^{used},\lambda_0^*$ | in-flight/queue counter、完成事务率 | 有限并发作用域是否正确 |
| $OP_k,P_k^{eff}$ | typed instruction/work、pipe duty、吞吐 | 计算 lowering 与资源映射是否正确 |
| stage、$II$、worker makespan | 时间线、barrier、async-copy interval | 依赖、buffer 和重叠是否正确 |
| $T_{kernel}$、bottleneck | 与 Forward 隔离的 device kernel time | 最终误差和瓶颈分类 |

最终时间的单样本相对误差为：

$$
e_T
=\frac{|\widehat T-T^{measured}|}{T^{measured}}
$$

数据集应至少报告 median、P90 和最大误差，并同时报告瓶颈分类、shape/launch 覆盖和 limitation； 只有平均 MAPE 会掩盖长尾和相互抵消的中间误差。

推荐消融按物理假设分组，而不是随意删除代码：

| 消融 | 对照问题 |
|---|---|
| 逻辑 bytes vs. physical transactions | coalescing、alignment 和 replay 是否必要 |
| 分层 cache 路由 vs. DRAM-only | L1/L2/DRAM 的 traffic 与瓶颈切换是否解释正确 |
| byte capacity vs. byte+request capacity | 小事务是否受 request frontend 限制 |
| 无限并发 vs. source/local finite slots | 带宽未跑满是否由 MLP/credit 解释 |
| 无 queue vs. 有证据的解析 queue | offered load 增大时 latency 是否按机制变化 |
| 串行 vs. 依赖流水 | buffer、barrier 和跨 tile overlap 是否必要 |
| aggregate vs. worker schedule | 尾波、persistent queue 和不均衡是否影响 critical worker |

消融只用于解释一项已经由独立证据确定的机制改变了哪些中间量和预测时间。不能因为某个消融版本在 evaluation case 上误差更低，就据此选择 production 公式、阈值或参数。

最终误差用于评估模型，而不是选择参数。消融可以说明某个已由独立证据确定的机制对预测的影响，但不能根据“加入后 MAPE 更低”决定是否采用该机制。

当误差来自缺失证据时，正确结果是记录 limitation、设计新的独立机制实验，或把组合标为 unsupported。解析模型的可信度来自可复测的物理链条，而不是对已有数据集的拟合程度。

## 13. 一次预测的求解顺序

把前述公式集中起来，一次完整预测按以下顺序进行：

1. 解析算子 shape、实际运行时长度、dtype、layout 和语义工作量；
2. 由 launch 冻结 variant、tile、grid/block、split、worker mapping、workspace 和资源占用；
3. 计算每类 typed compute work，以及逻辑、source-issued 和各物理层级 traffic；
4. 选择有证据的 memory path，计算 $N_i,D_i,\gamma_i$ 与各级 byte/request capacity；
5. 由 occupancy 和 source 依赖得到 $Q_{workload}$，再与硬件 slot 合成 $Q_{src}$；
6. 若 transaction、latency 和有限结构闭合，求 $t_{req}$、局部生命周期和 $\lambda_0^*$；否则明确退化为 byte-only capacity；
7. 得到 compute/memory/synchronization 的资源时间和首请求/短工作下界；
8. 按 worker、stage、buffer 和依赖组成流水，计算 critical worker makespan；
9. 与同一 timer domain 的 empty/non-empty task floor 和可识别 dispatch 组合；
10. 输出总时间、各资源时间、瓶颈、support level、使用的证据和所有 limitation。

这套顺序的目的不是要求每款硬件都提供全部参数，而是确保缺失发生在哪一步清晰可见。任何阶段都不能跳过未闭合语义，用目标 kernel duration 反推出一个让最终数值“看起来正确”的替代项。

## 14. 扩展边界

新增算子需要贯通算子语义、离散 launch、资源需求和统一预测结果四个边界，并分别验证工作量守恒、launch 完整性、资源诊断和最终时间。Serving 只能消费这条统一预测链，不能为新算子另建一套公式。

新增硬件只增加有证据的资源能力、机制参数和同接口 launch selector；缺失能力应形成 limitation 或 unsupported，不能复制一份公共公式并按硬件 ID 分叉。反过来，新增 Scheduler 或 Serving 框架只能决定请求与逻辑算子，不得进入 Kernel 层选择经验时延。这样扩展的是同一套物理模型的语义和证据覆盖，而不是逐硬件、逐场景积累互不兼容的预测器。
