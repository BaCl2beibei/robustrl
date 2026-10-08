
# 面向分布式节点容错任务迁移的可编程技术路线

下面以你提供的 Markdown 为研究依据。为便于区分来源，全文使用三种标记：

- **【原文规定】**：附件明确提出的模型、目标或公式。
- **【实现补充】**：附件未细化，但编程和实验必须确定的规则。
- **【建议修正】**：原式存在歧义，或不能直接用于含离散故障事件的仿真。

建议先完成一个**单机 CPU 可运行的、基于检查点恢复的仿真系统**，再加入鲁棒优化和神经网络训练。主线始终保持为：

```text
节点故障、网络波动、负载突变
        ↓
联合扰动模型与标称转移模型
        ↓
敏感度驱动的时变不确定集
        ↓
最坏情形下的迁移决策
        ↓
检查点传输、恢复与确认
        ↓
任务续算和集群韧性评估
```

## A. 原方案分析

### A1. 原文已经明确的内容

| 研究环节         | 原文依据             | 可直接保留的核心                                |
| ---------------- | -------------------- | ----------------------------------------------- |
| 节点可靠性       | 式（2-17）、（2-18） | 两状态故障/修复过程；至少\(k\) 个节点可用的判据 |
| 网络波动         | 式（2-19）           | 正常、拥塞两种时延成分的高斯混合                |
| 联合扰动         | 式（2-20）           | 从节点与网络联合分布生成扰动                    |
| 标称转移与敏感度 | 式（2-21）、（2-22） | 参数化转移模型；价值对参数的敏感度              |
| 不确定集         | 式（2-23）           | 沿敏感参数维度放大的椭球                        |
| 鲁棒决策         | 式（2-24）、（2-25） | 每个状态和动作下搜索不利转移，再优化策略        |

### A2. 阻碍直接编码的地方及最小修正

1. **【建议修正】下一状态不能整体写成高斯变量。** 节点是否存活、任务是否完成、迁移是否成功是离散量；时延、负载、剩余计算量是连续量。实际转移核应写成混合形式：
   $$
   P_\omega(s'\mid s,a)
   =P_\omega(z'\mid s,a)\,
   p_\omega(x'\mid z',s,a),
   $$
   其中 \(z'\) 是故障、拥塞等离散事件，\(x'\) 是连续状态。原式（2-21）的高斯分布只用于**条件连续部分**，或用于连续转移的近似模型。
2. **【建议修正】协方差重参数化不是 \(\mu+\Sigma\sigma\)。** 若 \(\Sigma\) 表示协方差，应令 \(LL^\top=\Sigma\)，再计算
   $$
   x'=\mu+L\epsilon,\qquad\epsilon\sim\mathcal N(0,I).
   $$
   直接乘 \(\Sigma\) 会得到协方差 \(\Sigma\Sigma^\top\)。PyTorch 的 `MultivariateNormal(scale_tril=L).rsample()` 与此一致。[PyTorch 分布文档](https://docs.pytorch.org/docs/stable/distributions.html)
3. **【建议修正】\(k\)-of-\(N\) 只给出“可用节点数是否足够”，不能单独推出完整的 \(P(S(t))\)。** 首版在节点独立假设下，把各节点的伯努利分布相乘；相关故障另加公共冲击变量。
4. **【建议修正】式（2-23）按梯度方向逐维相除，遇到零梯度会无定义。** 应在无约束参数空间构造带正下界轴长的椭球，并通过参数变换保证故障概率、方差等始终合法。
5. **【建议修正】式（2-24）的即时奖励如果取决于迁移是否成功、下一状态是否完成任务，就应写成**
   $$
   (\mathcal TV)(s)=
   \max_{a\in A(s)}
   \min_{\omega\in\Omega(s,a)}
   \mathbb E_{s'\sim P_\omega}
   [r(s,a,s')+\gamma V(s')].
   $$
   原式 \(r(s,a)\) 可视为奖励已对下一状态取期望的简写。
6. **【建议修正】式（2-24）允许每个 \((s,a)\) 独立选择最坏转移，而式（2-25）的一个全局 \(\omega\) 可能耦合所有状态。** 首版明确采用**状态—动作矩形不确定集** \(\Omega(s,a)\)，这样可进行鲁棒 Bellman 迭代；全局共享扰动参数作为后续更贴近物理系统的扩展实验。矩形性是该类动态规划成立的重要条件。[Iyengar，《Robust Dynamic Programming》](https://pubsonline.informs.org/doi/abs/10.1287/moor.1040.0129)
7. **【实现补充】原文没有规定任务、检查点、资源预留、故障检测和时间推进语义。** 这些必须先固定，否则即使训练出策略，也无法判断动作是否真正完成了一次迁移。

---

## B. 总体技术架构

### B1. 研究对象与运行场景

**【实现补充】节点**是一台逻辑计算机或虚拟机。每个节点有 CPU 计算速率、CPU 可分配容量、内存容量、本地检查点缓存、网络接口带宽、运行/故障状态、故障率 \(\lambda_i\)、修复率 \(\mu_i\)。首版不模拟操作系统指令，只按“计算量/有效计算速率”推进任务。

**【实现补充】任务**是可暂停的长作业，预先声明 CPU、内存需求、总计算量和截止时间。任务定期将进度写入可靠检查点存储。默认检查点已落盘且可被其他节点读取；节点故障后，未写入检查点的工作量丢失。

选择**基于检查点的恢复迁移**：

- 节点仍运行时：创建一致检查点，预留目标资源，传输检查点，目标恢复，确认后释放原节点。
- 原节点已故障时：从最近的**已提交**检查点恢复；先前检查点之后的计算要重做。
- 适用边界：任务计算状态可序列化，外部副作用具备幂等或事务保护。无检查点且不能重放的任务，不能声称“续算”。

**迁移、重启、重新调度的区别：**

| 操作     | 恢复点         | 目标节点       | 已完成计算           |
| -------- | -------------- | -------------- | -------------------- |
| 迁移     | 已提交检查点   | 通常改变       | 保留至检查点         |
| 重启     | 任务起点       | 原节点或新节点 | 全部重做             |
| 重新调度 | 按队列分配资源 | 可变           | 取决于是否携带检查点 |

**触发条件**：检测到所在节点故障；预测当前节点在任务截止前失败风险过高；预计目标节点能显著降低完成时间；持续拥塞使当前迁移/执行路径不可用。首版只实现“故障触发”和“明确的高负载阈值触发”，减少早期变量。

**观测与随机性**：调度器可见已检测节点状态、资源占用、最近一次链路测量、任务进度和检查点版本；未来故障、未来拥塞、负载突增是随机扰动。首版假设检测即时、观测准确，用完整状态验证 MDP；后续加入检测延迟和测量噪声，并用历史窗口或信念状态处理部分可观测性。

**时间**：首版仿真步长 \(\Delta=1\) 秒；每 \(5\) 秒或出现故障/迁移完成事件时做一次决策。计算推进、故障、修复、传输都在每个仿真步更新。后续可把内部推进换成离散事件队列，Gymnasium 仍负责对外提供 `reset/step`。其接口返回 `observation, reward, terminated, truncated, info`。[Gymnasium 环境接口](https://gymnasium.farama.org/api/env/)

### B2. 一个完整例子

设 \(N=4,\ k=3,\ M=3\)。任务 \(T_1\) 在节点 1 执行，最新检查点记录“已完成 60 单位”，故障发生前实际已计算 68 单位。节点 1 故障后：

1. 检测器在下一个仿真步报告故障；68 中超过检查点的 8 单位标记为需重做。
2. 候选节点 2、3、4 中，节点 2 内存不足，动作掩码排除它。
3. 鲁棒策略比较“在节点 3 恢复”“在节点 4 恢复”和“等待/重新调度”，选节点 3。
4. 迁移管理器原子性预留节点 3 的 CPU 与内存，读取已提交检查点。
5. 传输耗时由检查点字节数、有效带宽、时延及协议固定开销计算。
6. 若传输中节点 3 故障，预留释放，检查点仍保持版本 60，再次决策。
7. 若成功，节点 3 加载版本 60，恢复执行；任务达到总计算量后完成。
8. 节点 1 修复只改变未来容量，不使任务自动回迁。

这里的 \(k\)-of-\(N\) 服务可用性和“某个任务能否恢复”是**两个指标**：即使 3 个节点正常，资源不足也可能使某任务无法续算。

### B3. 端到端模块数据流

```text
配置与随机种子
  → Node/Network/Task 状态
  → CTMC 故障 + BGMM 时延 + 负载冲击
  → Cluster.step() 生成转移记录
  → 标称参数估计 ω̄
  → 价值敏感度 g
  → 椭球 Ω(s,a)
  → 内层最坏扰动搜索
  → 策略输出合法动作
  → MigrationManager 执行动作
  → 事件日志与评估指标
```

---

## C. 详细数学与算法实现

### C1. 模块一：节点、网络与负载的联合扰动

#### 1. 节点可靠性

**【原文规定】** 每个节点是“正常 \(1\) /故障 \(0\)”两状态连续时间马尔可夫过程，故障率 \(\lambda_i\)、修复率 \(\mu_i\)。

**【实现补充】** 将率统一为“每秒”。若有日志，估计
$$
\hat\lambda_i=
\frac{\text{正常}\to\text{故障次数}}{\text{累计正常秒数}},
\qquad
\hat\mu_i=
\frac{\text{故障}\to\text{正常次数}}{\text{累计故障秒数}}.
$$
没有真实数据时，配置合成参数并明确记录，不把合成值称为实测值。对很少发生的事件，先用跨节点共享先验或平滑，避免估计为零。

按固定步长精确采样两状态转移，而非简单地把 \(\lambda\Delta\) 当作概率：
$$
q=\lambda+\mu,\quad
p_{10}(\Delta)=\frac{\lambda}{q}(1-e^{-q\Delta}),\quad
p_{01}(\Delta)=\frac{\mu}{q}(1-e^{-q\Delta}).
$$
若要记录精确故障时间，正常态等待时间采样 \(T_f\sim\mathrm{Exp}(\lambda)\)，故障态采样 \(T_r\sim\mathrm{Exp}(\mu)\)。

原文式（2-17）直接计算 \(P_i(t)\)。独立异构节点时，集群可用度是泊松二项分布尾概率：
$$
A_{\rm cluster}(t)=
\sum_{\substack{B\subseteq\{1,\ldots,N\}\\|B|\ge k}}
\prod_{i\in B}P_i(t)\prod_{j\notin B}[1-P_j(t)].
$$
代码用一维动态规划，避免枚举 \(2^N\) 个子集：

```python
def k_of_n_availability(p_up: np.ndarray, k: int) -> float:
    mass = np.zeros(len(p_up) + 1)
    mass[0] = 1.0
    for p in p_up:
        nxt = np.zeros_like(mass)
        nxt[0] = mass[0] * (1 - p)
        for j in range(1, len(mass)):
            nxt[j] = mass[j] * (1 - p) + mass[j - 1] * p
        mass = nxt
    return float(mass[k:].sum())
```

复杂度 \(O(N^2)\)，如果只保留 \(k\) 附近状态可进一步优化。相关故障时，引入公共冲击 \(C_t\)：给定 \(C_t\) 后节点条件独立，对 \(C_t\) 积分或采样。此时不再把独立公式当作真实可用度。

**验证**：长期仿真的每节点正常占比应接近 \(\mu_i/(\lambda_i+\mu_i)\)；多个时刻的正常率应接近式（2-17）；独立场景中 \(\mathbf1[\sum S_i\ge k]\) 的经验均值应落在理论值的抽样置信区间内。

#### 2. 网络 BGMM 与迁移耗时

**【原文规定】**
$$
D\sim w\mathcal N(\mu_{\rm normal},\sigma_{\rm normal}^2)
 +(1-w)\mathcal N(\mu_{\rm congested},\sigma_{\rm congested}^2).
$$
这里将原文混合权重 \(\omega\) 改记为 \(w\)，避免与后文整组转移参数 \(\boldsymbol\omega\) 混淆。

**【实现补充】** 每条链路先抽网络状态 \(Z^\text{net}\in\{\text{正常},\text{拥塞}\}\)，再抽时延。独立逐步抽状态会导致拥塞不合理地频繁跳变，所以首版可使拥塞状态遵循二状态转移矩阵，稳定权重对应 BGMM 的 \(w\)。若只需验证原文的边际分布，也可先独立抽样。

高斯可能产生负时延。首版用**截断正态** \(D\mid D\ge d_{\min}>0\)，并在报告中说明这使分布不再严格等于原式（2-19）；若负值概率在所选参数下极小，也可采用拒绝重采样。迁移总耗时为
$$
T_{\rm mig}
=T_{\rm checkpoint}
+D_{ij}
+\frac{B_{\rm checkpoint}+B_{\rm metadata}}
       {b^{\rm effective}_{ij}}
+T_{\rm restore},
$$
其中 \(b^{\rm effective}_{ij}=b^{\rm nominal}_{ij}(1-c_{ij})\)，\(c_{ij}\in[0,1)\) 是拥塞损失。应明确区分传播时延 \(D\) 与带宽导致的传输时间，避免重复计算。

#### 3. 联合采样与负载

**【原文规定】** 式（2-20）采用节点与时延独立假设。

**【实现补充】** 先按独立模型获得基线结果，再设计公共冲击场景：共享电源/机架事件同时提高多个节点的故障风险，并使网络拥塞概率上升。负载变化用正常到达过程加突发批量到达过程：
$$
L_{i,t+1}
=\operatorname{clip}
\bigl(L_{i,t}
+\text{新任务资源需求}
-\text{已完成/迁出资源}
+\xi_{i,t},\,0,C_i\bigr).
$$
这里 \(\xi\) 不能让负载凭空占用实体资源；若代表背景任务，必须同时生成对应的背景资源占用记录。

| 随机量            | 首版参数                            | 来源与记录             |
| ----------------- | ----------------------------------- | ---------------------- |
| 节点故障/修复     | \(\lambda_i,\mu_i\)                 | 合成配置或事件日志估计 |
| 链路状态          | 正常/拥塞转移概率                   | 合成配置               |
| BGMM 时延         | \(w,\mu_1,\sigma_1,\mu_2,\sigma_2\) | 合成配置或网络测量拟合 |
| 有效带宽          | 标称带宽、拥塞损失                  | 链路配置和测量         |
| 任务到达/负载突变 | 到达率、突发幅度                    | 合成场景               |
| 公共冲击          | 到达率、影响节点集合                | 扩展场景               |

场景参数不能只写“正常/严重”：应保存每个场景的故障倍率、拥塞概率、带宽折减、突发任务数及持续时间。建议四级：正常；单一中等扰动；单一严重扰动；故障、拥塞和负载同时出现的复合扰动。扩展实验对比“独立训练/独立测试”“独立训练/相关测试”“相关训练/相关测试”。

```python
def sample_disturbance(state, cfg, rng):
    common = sample_common_shock(cfg, rng)
    node_next = sample_ctmc_nodes(state.nodes, cfg, common, rng)
    net_regime = sample_network_regime(state.network, cfg, common, rng)
    delays = sample_positive_bgmm(net_regime, cfg, rng)
    arrivals = sample_task_arrivals(cfg, common, rng)
    return Disturbance(node_next, net_regime, delays, arrivals)
```

网络验证用分成分布的均值、方差、拥塞占比和负时延计数；联合验证用节点故障与拥塞的经验相关系数，并确认独立基线接近零、公共冲击场景明显偏离零。

### C2. 模块二：标称转移、敏感度与时变不确定集

#### 1. 参数与数据

令 \(\boldsymbol\omega\) 包含节点故障/修复率、网络状态转移概率、BGMM 参数、带宽折减、任务到达率及公共冲击参数。**【实现补充】** 首版为控制维度，可先只让以下参数进入鲁棒内层：
$$
\boldsymbol\omega_{\rm robust}
=(\lambda_{\rm group},\mu_{\rm group},
p_{\rm congest},\mu_{\rm congested},
b_{\rm loss},p_{\rm burst}).
$$
其他参数固定为标称值。随后再扩展到分节点、分链路参数。

标称值 \(\bar{\boldsymbol\omega}\) 从合成环境训练集的**状态转移记录**估计，不能直接使用生成器的真值冒充估计结果。记录至少包含 \((s_t,a_t,s_{t+1})\)、故障事件、链路状态/时延、任务到达、迁移结果和时间戳。故障与修复率用暴露时间估计；混合网络参数用已知仿真标签估计，后续无标签数据再拟合混合模型。参数拟合与策略测试必须分开数据。

#### 2. 连续与离散变量的梯度

设连续状态 \(x'\) 的条件模型给出 \(\mu_\omega(s,a,z')\) 和 \(L_\omega(s,a,z')\)：
$$
x'=\mu_\omega+L_\omega\epsilon,\quad \epsilon\sim\mathcal N(0,I).
$$
固定价值网络参数 \(\theta\)，计算
$$
g_\omega=
\nabla_\omega
\mathbb E_{z',\epsilon}
\left[V_\theta(z',x')\right].
$$

- 连续分量：固定随机噪声 \(\epsilon\)，通过 \(x'\to\mu,L\to\omega\) 反向传播。
- 离散分量：小规模时**枚举故障/拥塞结果并按其概率加权**，概率对 \(\omega\) 可微；大规模时用条件抽样加似然比梯度，或对少数关键事件枚举、其余用采样。Gumbel-Softmax 可作训练近似，但其软状态不能直接当作真实故障状态。
- 不应对一次 `Bernoulli.sample()` 的布尔结果调用 `autograd` 并期望获得故障率梯度。
- 价值网络在算 \(g_\omega\) 时固定 \(\theta\)；要保留从价值输出至转移参数的计算图，不能对生成的 \(x'\) 执行 `detach()`。

PyTorch 风格的核心计算：

```python
# raw_center: [p]，无约束参数；V_theta 在本步骤固定
raw = raw_center.detach().clone().requires_grad_(True)
omega = to_physical_parameters(raw)             # sigmoid/softplus 等
mu, scale_tril = continuous_kernel(s, a, omega) # [B,d], [B,d,d]
eps = fixed_eps                                  # [B,d]
x_next = mu + (scale_tril @ eps[..., None]).squeeze(-1)

# 小规模：离散结果 K 个；prob:[B,K]，value:[B,K]
prob = enumerate_discrete_probabilities(s, a, omega)
value = evaluate_all_discrete_outcomes(V_theta, x_next)
objective = (prob * value).sum(-1).mean()
g_raw = torch.autograd.grad(objective, raw)[0]  # [p]
```

如果离散结果与连续条件分布相关，应对每个离散结果分别计算其条件 \(\mu,L\)，再加权；上面伪代码只展示张量路径。

#### 3. 合法、稳定的椭球

**【建议修正】** 在无约束坐标 \(u\in\mathbb R^p\) 中建集合，物理参数由映射 \(\omega=h(u)\) 给出：

- 概率、混合权重：`sigmoid(u)`，必要时映射到 \([\varepsilon,1-\varepsilon]\)。
- 故障率、修复率、标准差：`softplus(u)+ε`。
- 两个 BGMM 均值的顺序：令 \(\mu_2=\mu_1+\operatorname{softplus}(v)\)。
- 协方差：下三角 \(L\)，对角线用 `softplus+ε`，再令 \(\Sigma=LL^\top\)。

对每个标准化后的参数方向，令
$$
d_j=
d_{\min}+
(1-d_{\min})
\frac{|g_{u,j}|}{\max_\ell|g_{u,\ell}|+\varepsilon},
\quad d_{\min}>0,
$$
并定义
$$
\Omega_u(s,a)=
\left\{\bar u+
\alpha\,\operatorname{diag}(\hat\sigma_1d_1,\ldots,\hat\sigma_pd_p)z:
\|z\|_2\le1\right\}.
$$
\(\hat\sigma_j\) 是参数估计误差或预定物理尺度。这样零梯度仍有小半径，且维度量纲不同的问题得到处理。若全部梯度近零，退回由 \(\hat\sigma_j\) 决定的基础椭球，而不是声称不存在不确定性。

**\(\alpha\) 的确定**：在训练数据中拟合中心和尺度，用独立校准数据计算真实/估计参数落入椭球所需的半径分位数；再在验证集考察可靠度与成本的折中。不要在测试集上调 \(\alpha\)。附件提出可按可靠度目标或经验分位确定，这里给出了可执行步骤。

#### 4. 最坏扰动搜索

首版选**投影梯度下降 PGD**，优化球坐标 \(z\)：
$$
z^*=\arg\min_{\|z\|\le1}
\widehat{\mathbb E}\,[r(s,a,s')+\gamma V(s')].
$$
每一步用固定随机样本降低噪声，更新 \(z\leftarrow z-\eta\nabla_z J\)，然后投影到单位球。运行多个初值，中心点也作为候选，选择目标最小者。单步梯度投影更快，可作消融或在线低延迟版本；PGD 更适合主线，因为多参数之间可能相互抵消。PGD 找到的是**局部近似最坏值**，不是理论上保证的全局最小值。

时变更新以滚动窗口 \(W_t\) 拟合 \(\bar u_t,\hat\sigma_t\)，对梯度采用指数平滑：
$$
\bar g_t=\beta\bar g_{t-1}+(1-\beta)g_t.
$$
每 \(K\) 个决策周期更新一次；小样本时回退至上一窗口或全局先验。严格的非平稳 MDP 应把时间/扰动状态或参数估计加入状态；训练时也可在每回合冻结一组参数，回合之间更新。

**必要单元测试**：协方差半正定；重参数化样本的均值/协方差接近设定值；连续参数梯度与有限差分方向一致；离散枚举梯度与有限差分一致；零梯度、极端 logits 下无 NaN；PGD 输出始终在椭球内；所有物理参数合法；窗口更新只使用当前时刻以前的数据。

### C3. 模块三：鲁棒 MDP 与求解

#### 1. 状态、动作、转移、奖励

**【实现补充】** 仿真器的完整马尔可夫状态 \(s_t\) 包括：

- 每节点 8 项：运行状态、CPU/内存容量、CPU/内存占用、队列长度、故障率、修复率。
- 每条有向链路 3 项：时延、有效带宽、拥塞状态。
- 每任务 8 个标量：CPU/内存需求、剩余计算量、最近已提交检查点对应工作量、截止剩余时间、待传输字节、检查点年龄、重试次数；另有 5 维任务状态、\(N+1\) 维当前所在位置和 \(N+1\) 维迁移目标的独热编码。
- 4 个全局量：剩余仿真时间、\(k\)、公共冲击状态、负载场景状态。

固定 \(N,M\) 时，展平维度为
$$
D=8N+3N(N-1)+M(2N+15)+4.
$$
持续时间、容量、计算量分别除以配置的参考值并裁剪到合理范围；离散状态用独热编码。参数估计若影响未来转移，应另外把估计值和必要的滚动统计量加入状态。资源预留和迁移中的阶段/计时器也要进入内部状态；上式是基础观测维度，启用这些机制后需追加相应字段并固定版本号。

首版一次决策处理一个“焦点任务”：按故障优先、截止时间其次的固定规则选任务，动作 \(a\in\{0,1,\ldots,N,N+1\}\)：`0` 不迁移/继续，`i` 迁往节点 \(i\)，`N+1` 故障恢复队列或重调度。对在故障节点上的任务，“不迁移”意味着等待修复，必须承担等待成本。资源不足、目标故障、网络不可达、自身重复迁移、任务已完成的动作全部屏蔽；即使策略错误输出，也由环境拒绝并记录，而不能造成容量超额。

一次 `step(s,a)` 的顺序固定为：

```text
校验动作 → 预留资源/启动迁移
→ 抽节点、网络、负载扰动
→ 推进传输与执行 Δ 秒
→ 处理故障、修复、完成、超时事件
→ 提交/回滚迁移状态
→ 计算本步奖励与指标 → 返回 s'
```

资源分配、剩余工作量和传输字节在给定扰动后按确定性规则更新；故障、拥塞、任务到达等由随机核产生。这就明确了节点、网络和负载如何**共同影响**转移：节点故障决定能否执行/恢复，网络决定检查点抵达时间及中断风险，负载决定目标可用容量与有效计算速率，三者共同决定下一步任务状态和完成概率。

奖励建议采用有明确单位的事件计数与归一化成本：
$$
\begin{aligned}
r_t={}&
w_c\,n^{\rm complete}_t
-w_f\,n^{\rm irrecoverable}_t
-w_s\,n^{\rm newSLAviol}_t\\
&-w_m\,\frac{T^{\rm migration}_t}{T_{\rm ref}}
-w_p\,\frac{T^{\rm checkpoint+restore}_t}{T_{\rm ref}}
-w_w\,\frac{\text{未完成任务等待秒数}}{T_{\rm ref}}\\
&-w_o\,\sum_i
\left[\max(0,u_i-u_{\rm safe})\right]^2.
\end{aligned}
$$
完成、不可恢复、首次 SLA 违约只计一次；等待和迁移时间按实际秒数累计，避免每步重复罚一次完整迁移成本。可从 \(w_c=1,w_f=1,w_s=0.5,w_m=w_p=0.1,w_w=0.02\) 作为**实验初始值**，在验证集做权重网格/敏感度分析；不把这些数值称作原文给定。最终仍分别报告成功率、可用性、时延和成本，不能只用累积奖励代替研究指标。

#### 2. 第一层：小规模精确验证

先做 \(N=2\) 或 \(3\)、一个任务、离散化进度/负载/网络状态。显式构建 \(P_\omega(s'|s,a)\)，检查每一行非负且和为 1。普通值迭代：
$$
V_{n+1}(s)=\max_a\sum_{s'}P_{\bar\omega}(s'|s,a)
[r(s,a,s')+\gamma V_n(s')].
$$
鲁棒值迭代将求和替换为对 \(\Omega(s,a)\) 的最小化。若离散小环境把不确定集设为**概率单纯形上的线性约束集**，内层是线性规划，可以得到可核验的全局最坏分布；若保持原文参数椭球，则对低维参数密集网格搜索并与 PGD 对照。两种求解要明确区分。

验收要求：\(\alpha=0\) 时鲁棒值与普通值相同；扩大集合时鲁棒价值不升高；所求最坏分布合法；至少构造一个普通策略与鲁棒策略不同的可解释例子。差异不是必然出现，若某配置两策略相同，不应强行修改结果。

#### 3. 第二层：连续状态扩展

候选方法：

| 方法               | 适合处                               | 代价                       |
| ------------------ | ------------------------------------ | -------------------------- |
| 鲁棒值迭代         | 小状态精确验证                       | 状态爆炸                   |
| 鲁棒 Actor-Critic  | 可直接把最坏 Bellman 目标写进 Critic | 实现和稳定性要求较高       |
| 对抗环境训练 + PPO | 容易接入仿真、适合离散迁移动作       | 对鲁棒目标的实现是近似     |
| 纯普通 PPO         | 有用的对照                           | 没有原文要求的不确定集内层 |

**主路线：敏感度椭球 + PGD 对抗环境训练 + 掩码 PPO。** Actor、Critic 分别用两层 MLP（如 128/128、ReLU），Actor 输出 \(N+2\) 个动作 logits，非法动作设为 \(-\infty\) 后再采样；Critic 输出一个状态值。节点数变化较大时改用共享节点编码器和注意力聚合，首版固定 \(N\)。

```python
for iteration in range(num_iterations):
    update_nominal_model_from_past_rollouts()
    update_sensitivity_and_ellipsoid()

    trajectories = []
    for env in parallel_envs:
        s, _ = env.reset(seed=next_seed())
        for t in range(rollout_steps):
            mask = env.valid_action_mask()
            a, logp = masked_actor.sample(s, mask)
            u_bad = pgd_minimize_bellman_target(
                s, a, critic, nominal_u, ellipsoid, fixed_noise=True
            )
            s_next, reward, terminated, truncated, info = env.step(
                a, disturbance_parameters=to_physical_parameters(u_bad)
            )
            trajectories.append((s, a, logp, reward, s_next,
                                 terminated, truncated, info))
            s = reset_if_needed(env, s_next, terminated, truncated)

    advantages, returns = compute_gae(trajectories, critic)
    update_actor_with_ppo_clip(trajectories, advantages)
    update_critic(trajectories, returns)
    validate_on_fixed_held_out_scenarios()
```

可用初始超参数：\(\gamma=0.99\)、GAE 系数 \(0.95\)、PPO 裁剪 \(0.2\)、每批 \(1024\)–\(4096\) 步、每批 4–10 个优化轮次、PGD 3–10 步和 2–4 个初值；这些是起点，应通过验证集调整。训练先在 CPU 小集群跑通，再考虑 GPU 批量训练。

**近似的位置必须如实说明**：Critic 近似价值；PGD 近似内层最小化；有限蒙特卡洛样本近似期望；PPO 的代理目标并非直接精确求解式（2-25）；固定维度观测可能丢失部分真实系统状态。因此，小环境的精确/高精度对照不可省略。PPO 的裁剪目标与多轮小批量更新可参考其原始论文。[Schulman 等，《Proximal Policy Optimization Algorithms》](https://arxiv.org/abs/1707.06347)

在线部署于仿真中时，调度器读取当前观测，计算合法动作掩码，Actor 输出动作；迁移管理器执行。训练中的 PGD 不一定要在每次在线决策时运行，取决于延迟预算；若只用离线训练出的 Actor，需在论文中明确。

---

## D. 工程设计

### D1. 六个核心类及接口

下表的字段为首版必需字段。所有时间、容量和字节的单位在配置中固定。

| 类/文件                                        | 主要字段（类型）                                                                                                                                                                                                                             | 核心接口与输入输出                                                                                                                                                           |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Node`，`environments/entities.py`         | `id:int`, `cpu_rate:float`, `cpu_capacity:float`, `mem_capacity:float`, `cpu_used:float`, `mem_used:float`, `up:bool`, `failure_rate:float`, `repair_rate:float`, `reservations:dict[str, Reservation]`                  | `can_host(task)->bool`；`reserve(task, token)->bool`；`commit(token)->None`；`release(token)->None`；`advance_health(dt,rng)->list[Event]`                         |
| `Task`，`environments/entities.py`         | `id:int`, `cpu_need:float`, `mem_need:float`, `total_work:float`, `remaining_work:float`, `checkpoint_work:float`, `checkpoint_version:int`, `checkpoint_bytes:int`, `deadline:float`, `status:TaskStatus`, `node_id:int | None`, `migration_target:int                                                                                                                                               |
| `Network`，`environments/network.py`       | `bandwidth:ndarray[N,N]`, `delay:ndarray[N,N]`, `regime:ndarray[N,N]`, `reachable:ndarray[N,N]`                                                                                                                                      | `sample_step(dt,omega,rng)->None`；`transfer_time(src,dst,nbytes)->float`；`transfer_capacity(src,dst,dt)->float`                                                      |
| `Cluster`，`environments/cluster.py`       | `nodes:dict[int,Node]`, `tasks:dict[int,Task]`, `network:Network`, `queue:list[int]`, `time:float`, `event_log:list[Event]`                                                                                                      | `advance(dt,disturbance)->list[Event]`；`place(task,node)->bool`；`release(task)->None`；`select_focus_task()->int                                                     |
| `MigrationManager`，`migration/manager.py` | `active:dict[int,Migration]`, `next_token:int`, `timeout_s:float`                                                                                                                                                                      | `valid_targets(task,cluster)->ndarray[bool]`；`start(task_id,target_id,cluster)->MigrationResult`；`advance(dt,cluster)->list[Event]`；`abort(task_id,reason)->None` |
| `MigrationEnv`，`environments/env.py`      | `cluster:Cluster`, `manager:MigrationManager`, `scenario:Scenario`, `horizon_s:float`, `rng:Generator`                                                                                                                             | `reset(seed=None,options=None)->(obs,info)`；`step(action)->(obs,reward,terminated,truncated,info)`；`valid_action_mask()->ndarray[bool]`                              |

推荐调用关系：
`MigrationEnv.step → MigrationManager.start → DisturbanceModel.sample → Cluster.advance → MigrationManager.advance → Reward.compute → Metrics.record`。实体类不依赖 RL 算法；算法只调用环境接口，避免循环依赖。

关键更新公式：

$$
\Delta W_{\rm task}
=\Delta\cdot \text{node.cpu\_rate}\cdot
\frac{\text{allocated CPU}}{\text{cpu\_capacity}}
\cdot \text{efficiency(load)}.
$$
$$
B^{\rm remaining}_{t+1}
=\max(0,B^{\rm remaining}_t
-b^{\rm effective}_{ij}\Delta).
$$
节点故障时，将任务工作量恢复至最近已提交检查点：`remaining_work = total_work - checkpoint_work`。

迁移必须是状态机，不能在选出节点后立即修改任务位置：

```text
REQUESTED → RESERVED → CHECKPOINT_READY → TRANSFERRING
→ RESTORING → CONFIRMED
                         ↘ ABORTED
```

- `RESERVED`：目标容量计入预留，防止其他任务抢占。
- `CHECKPOINT_READY`：版本号、工作量和校验摘要固定。
- `TRANSFERRING`：每步扣除可传输字节；链路断开则暂停或超时。
- `RESTORING`：目标加载检查点，尚不对外宣称任务已恢复。
- `CONFIRMED`：仅此时更新任务唯一运行位置，再释放旧资源。
- `ABORTED`：释放目标预留，保持最近已提交检查点，允许重试。

每个任务使用唯一执行租约/迁移 token；同一版本仅允许一个已确认执行者。仿真中用状态和断言实现，真实系统则需持久化元数据、租约和幂等提交。若任务写外部系统，检查点本身不足以防止重复副作用。

### D2. 项目目录与每文件职责

建议保持一个小型 Python 包，避免一开始引入服务端组件：

```text
fault_tolerant_migration/
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   └── scenarios.yaml
├── ftm/
│   ├── environments/
│   │   ├── entities.py
│   │   ├── network.py
│   │   ├── cluster.py
│   │   ├── disturbances.py
│   │   └── env.py
│   ├── migration/
│   │   └── manager.py
│   ├── models/
│   │   ├── reliability.py
│   │   ├── nominal.py
│   │   └── policy_value.py
│   ├── uncertainty/
│   │   ├── transforms.py
│   │   ├── sensitivity.py
│   │   ├── ellipsoid.py
│   │   └── adversary.py
│   ├── algorithms/
│   │   ├── tabular.py
│   │   └── robust_ppo.py
│   ├── baselines/
│   │   └── policies.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── reports.py
│   └── main.py
├── experiments/
│   ├── collect_nominal.py
│   ├── train.py
│   └── evaluate.py
└── tests/
    ├── test_reliability.py
    ├── test_migration.py
    ├── test_uncertainty.py
    └── test_tabular.py
```

| 文件                              | 依赖、输入 → 输出                         | 开发顺序与验证                    |
| --------------------------------- | ------------------------------------------ | --------------------------------- |
| `configs/*.yaml`                | 参数、单位、场景 → 配置对象               | 第 1 步；字段范围校验             |
| `entities.py`                   | 配置 → Node/Task/Checkpoint               | 第 1 步；资源和进度不变量         |
| `network.py`                    | 参数、随机数 → 链路样本/传输量            | 第 2 步；正时延、传输量守恒       |
| `cluster.py`                    | 实体、扰动 → 事件与新状态                 | 第 1 步；容量和任务唯一性         |
| `disturbances.py`               | 状态、场景、随机数 → 联合扰动             | 第 2 步；边际频率、相关性         |
| `env.py`                        | Cluster、Manager → Gymnasium 交互结果     | 第 3 步；接口检查、种子复现       |
| `manager.py`                    | 任务、目标、集群 → 迁移状态及事件         | 第 3 步；中断、回滚、重复执行测试 |
| `reliability.py`                | \(\lambda,\mu,k,t\) → 节点/集群可用度     | 第 2 步；理论与模拟比较           |
| `nominal.py`                    | 转移日志 →\(\bar\omega\)、参数尺度        | 第 5 步；已知合成参数恢复         |
| `policy_value.py`               | 观测、掩码 → logits、\(V_\theta\)         | 第 6 步；形状与动作掩码           |
| `transforms.py`                 | 无约束参数 → 合法物理参数                 | 第 5 步；边界测试                 |
| `sensitivity.py`                | 模型、Critic、样本 → 梯度向量             | 第 5 步；有限差分对比             |
| `ellipsoid.py`                  | 中心、尺度、梯度、\(\alpha\) → 集合与投影 | 第 5 步；半径、零梯度测试         |
| `adversary.py`                  | \(s,a,V,\Omega\) → 近似最坏参数           | 第 5 步；目标不高于中心值         |
| `tabular.py`                    | 离散转移、奖励 → 普通/鲁棒价值与策略      | 第 4 步；手工算例                 |
| `robust_ppo.py`                 | 环境、模型、对抗器 → 策略权重             | 第 6 步；短训练无 NaN             |
| `policies.py`                   | 观测、掩码 → 基线动作                     | 第 4 步；所有动作合法             |
| `metrics.py`                    | 事件日志 → 指标表                         | 第 3 步；人工构造事件核算         |
| `reports.py`                    | 多种子指标 → 表和图                       | 第 7 步；汇总可重算               |
| `main.py`、`experiments/*.py` | 配置 → 数据收集/训练/评估产物             | 分阶段增加；命令可复现            |
| `tests/*.py`                    | 各模块 → 自动断言                         | 与模块同步编写                    |

### D3. 一个完整仿真回合

```python
obs, info = env.reset(seed=42)
while True:
    mask = env.valid_action_mask()
    action = policy.act(obs, mask)
    obs, reward, terminated, truncated, info = env.step(action)
    recorder.append(info["events"], reward)
    if terminated or truncated:
        break
metrics = compute_metrics(recorder.events, env.horizon_s)
```

`terminated=True` 表示全部任务完成、全部任务无法恢复等自然终止；`truncated=True` 表示达到预设时间上限。训练目标和评价都要正确区分两者。

---

## E. 分阶段编程计划与验收

| 阶段                    | 前置、任务与文件                                                                                  | 难度 | 可运行案例与完成标准                                             | 常见失败及排查                               |
| ----------------------- | ------------------------------------------------------------------------------------------------- | ---- | ---------------------------------------------------------------- | -------------------------------------------- |
| 1. 基础集群             | 无；配置、`entities.py`、`cluster.py`                                                         | 低   | 2 节点 1 任务，无故障时按预期时间完成；资源从不超额              | 单位混用；检查秒/字节/计算量                 |
| 2. 故障与网络           | 阶段 1；`reliability.py`、`network.py`、`disturbances.py`                                   | 中   | 长期正常占比和\(k\)-of-\(N\) 理论一致；时延非负，拥塞分量可识别  | 把率当概率；检查转移公式与步长               |
| 3. 迁移与环境           | 阶段 2；`manager.py`、`env.py`、`metrics.py`                                                | 中高 | 故障后从已提交检查点恢复；目标故障可回滚；固定种子得到同一事件流 | 双执行、资源泄漏；检查 token 和状态机        |
| 4. 标称 MDP 与基线      | 阶段 3；`tabular.py`、`policies.py`                                                           | 中   | 小环境 Bellman 值与手算一致；六类策略均可跑回合                  | 状态遗漏导致非马尔可夫；检查转移行和         |
| 5. 不确定集             | 阶段 4；`nominal.py`、`transforms.py`、`sensitivity.py`、`ellipsoid.py`、`adversary.py` | 高   | 梯度通过有限差分；PGD 输出合法且能找到不高于标称目标的候选       | 采样断梯度、零梯度、越界；逐层检查图和变换   |
| 6. 鲁棒策略             | 阶段 5；`policy_value.py`、`robust_ppo.py`、`train.py`                                      | 高   | 小环境鲁棒价值/策略与表格基准方向一致；短训练稳定、可保存加载    | Critic 不准或 PGD 噪声大；先固定模型和随机数 |
| 7. 综合实验             | 阶段 6；`evaluate.py`、`reports.py`                                                           | 中   | 所有场景、基线、消融、种子可批量运行；表可从原始日志重算         | 测试泄漏；检查场景、种子、参数快照           |
| 8. 真实集群验证（可选） | 阶段 7；另建适配层                                                                                | 很高 | 在小规模真实作业上测量检测、传输、恢复；与仿真误差对照           | 外部副作用、时钟不一致；先限制为可重放任务   |

第一阶段只需 NumPy 和 Python 基础功能；Gymnasium 接口可在阶段 3 接入；PyTorch 在阶段 5–6 使用。真实集群不是论文仿真结论的前置条件。

---

## F. 实验与验证方案

### F1. 对照和场景

对照策略应共用**同一批场景种子与初始任务**：

1. 不迁移：故障后等待修复或从头重启，分别注明规则。
2. 随机迁移：在合法目标中均匀选取。
3. 贪心迁移：选估计恢复完成时间最短的节点。
4. 基于负载：选剩余资源最多或预计负载最小的节点。
5. 普通 MDP/RL：同样的环境、网络容量和训练预算，但 \(\alpha=0\)。
6. 本研究方法：时变敏感度椭球 + 最坏扰动训练。

测试场景覆盖：正常、独立随机故障、集中故障、拥塞、负载突变、故障和网络同时发生、训练强度之外的扰动。集中故障测试需承认：若同时可用节点少于 \(k\)，服务不可用是模型定义下的事实，策略无法凭空恢复容量。

### F2. 指标定义与记录位置

| 指标           | 数学/代码定义                                                                                                    | 事件来源                                |
| -------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| 任务续算成功率 | \(N_{\rm recovered\ and\ completed}/N_{\rm interrupted}\)；分母为曾因故障中断且有恢复机会的任务，零分母报 `NA` | `FAILURE`, `RESTORED`, `COMPLETE` |
| 集群可用性     | \(\frac1T\int_0^T\mathbf1[\sum_iS_i(t)\ge k]dt\)；另报资源可服务性                                               | 节点状态时间线                          |
| 任务完成时间   | 对已完成任务取`complete_time-arrival_time`，同时报告未完成比例                                                 | `ARRIVAL`, `COMPLETE`               |
| SLA 违约率     | 未在截止时间前完成的任务数/到达任务数                                                                            | 截止时间、完成事件                      |
| 迁移次数       | `CONFIRMED` 次数；同时单列尝试次数                                                                             | 迁移状态机事件                          |
| 平均迁移开销   | 已确认迁移的\(\sum T_{\rm mig}/N_{\rm confirmed}\)，另报失败尝试成本                                             | 时间戳、传输字节                        |
| 资源利用率     | \(\sum_i\int CPU^{used}_i(t)dt/\sum_i\int CPU^{available}_i(t)dt\)，内存另报                                     | 周期资源快照                            |
| 极端扰动性能   | 上述指标在严重/未见过场景中的值、最差场景值和尾部分位                                                            | 场景标签与指标表                        |

事件日志建议统一字段：`episode_id, seed, scenario, time, event_type, task_id, source, target, checkpoint_version, bytes, reason`。`metrics.py` 只从日志/状态快照算指标，使评估可以独立复核。

### F3. 消融与统计

至少做四组消融，其他设置保持一致：

- \(\alpha=0\)、小、中、大半径：检验保守度与迁移成本的折中。
- 固定等轴椭球、敏感度椭球：检验梯度形状的作用。
- 固定不确定集、滚动更新：检验时变建模。
- 完整奖励、去除迁移成本、去除 SLA 项：检验目标冲突和策略变化。

训练、验证、测试使用不重叠的**场景种子和参数区间**；至少用多个独立训练种子及多个测试种子，具体数量按算力预算预先登记。报告每个指标的均值、标准差和 95% 置信区间；策略对比优先采用相同测试种子的配对差值及其区间。画成功率—迁移开销散点/帕累托图、不同故障强度曲线、迁移时间分布、资源利用时间线，并展示失败案例。绝不预设鲁棒策略一定胜出；若在正常条件下成本上升，应如实呈现。

---

## G. 最小可行版本 MVP

**第一版最低范围**：\(N=3,\ k=2,\ M=2\)；一类可检查点任务；节点独立故障/修复；正常/拥塞两态网络和正时延 BGMM；固定步长；故障触发的检查点恢复迁移；不迁移、随机、贪心三种策略；事件日志与续算成功率、可用性、迁移时间三个指标。先不训练 PyTorch。

从创建项目到第一个实验：

1. 建立上面的目录、配置解析、统一单位和随机数入口。
2. 实现 `Node/Task/Cluster`，运行无故障任务完成案例。
3. 加入 CTMC 节点事件及 BGMM 链路样本，验证经验频率。
4. 加入检查点版本、资源预留和迁移状态机，验证故障后能恢复。
5. 实现 `MigrationEnv.reset/step` 和动作掩码，固定种子跑通一回合。
6. 实现三种 MVP 策略，用**相同 20 个预先固定的场景种子**运行，保存每回合事件和指标。
7. 手工检查至少一条迁移日志：故障时间、检查点版本、丢失工作量、目标预留、传输结束、恢复、完成都能对上。
8. 再进入表格 MDP、不确定集和鲁棒训练，避免用神经网络掩盖仿真错误。

首个实验的“成功”仅表示流程正确、指标可重算，不要求某策略优于另一策略。

---

## H. 可逐项交给编程 AI 的编码任务清单

每项任务只依赖前面已完成的成果；文件路径均相对上述项目根目录。

| 顺序 | 独立任务：修改文件与主要接口                                                                                            | 必测案例                                          | 完成标准                                   |
| ---- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------ |
| H1   | `configs/*.yaml`, `pyproject.toml`：实现 `load_config(path)->Config` 与参数范围校验                               | 非法\(k>N\)、负容量、非法方差被拒绝               | 单位、种子、场景参数固定且可序列化         |
| H2   | `ftm/environments/entities.py`：`Node.can_host/reserve/release`，`Task.run/checkpoint/restore`                    | 超额预留失败；恢复后进度等于已提交版本            | 资源与进度不变量成立                       |
| H3   | `ftm/environments/cluster.py`：`Cluster.place/advance/select_focus_task`                                            | 无故障 2 节点任务按计算量完成                     | 无双重分配、无负剩余工作量                 |
| H4   | `ftm/models/reliability.py`：`node_up_probability`, `k_of_n_availability`；`disturbances.py` 的 CTMC            | 理论值与长时仿真频率对照                          | 误差落在抽样容差内                         |
| H5   | `ftm/environments/network.py`：`sample_step/transfer_time/transfer_capacity`                                        | 正常/拥塞样本；0 或低带宽；负时延防护             | 时延为正，传输量守恒                       |
| H6   | `ftm/migration/manager.py`：`valid_targets/start/advance/abort`                                                     | 目标故障、链路中断、重复`start`、检查点版本冲突 | 事务回滚后无资源泄漏或双执行               |
| H7   | `ftm/environments/env.py`, `ftm/evaluation/metrics.py`：Gymnasium 接口、奖励、事件日志                              | 同种子重复回合；手算指标                          | 回合可复现、指标可从日志重算               |
| H8   | `ftm/baselines/policies.py`, `experiments/evaluate.py`：三种 MVP 策略                                               | 所有策略面对资源不足                              | 永不选择未屏蔽的非法目标；可产出首个对照表 |
| H9   | `ftm/algorithms/tabular.py`, `tests/test_tabular.py`：普通/鲁棒值迭代与内层求解                                     | 手工 2 状态算例；\(\alpha=0\)                     | Bellman 残差达阈值、概率行合法             |
| H10  | `ftm/models/nominal.py`, `ftm/uncertainty/transforms.py`：`fit_nominal(records)`, `to_physical_parameters(raw)` | 合成已知参数拟合；极端输入                        | 参数估计可复核，输出总合法                 |
| H11  | `sensitivity.py`, `ellipsoid.py`, `adversary.py`：`value_sensitivity`, `make_ellipsoid`, `pgd_worst_case`   | 有限差分、零梯度、投影、固定样本 PGD              | 梯度正确且最坏候选不劣于中心候选           |
| H12  | `policy_value.py`, `robust_ppo.py`, `experiments/train.py`                                                        | 掩码输出、短训练、保存加载                        | 无 NaN，可复现训练曲线并输出可用策略       |
| H13  | `configs/scenarios.yaml`, `experiments/evaluate.py`, `reports.py`                                                 | 全部场景、六类策略、四类消融                      | 原始记录、配置快照、均值和区间齐全         |
| H14  | 可选真实系统适配层，保持现有仿真接口                                                                                    | 单任务检查点传输与恢复                            | 测得实际开销并与仿真模型对照               |

**本轮交付范围**：已完成附件公式核对和完整技术路线；没有修改文件、安装依赖或运行实验，因此没有虚构测试结果。下一步可直接从 H1–H3 实现单机最小仿真。
