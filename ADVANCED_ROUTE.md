# 鲁棒求解、代理 PGD 与统计实验补充

本文补上 MVP 文档中列出的研究边界和后续实现入口。它不声称连续参数鲁棒 MDP 已被全局精确求解，也不声称已连接真实集群。

## 1. 可精确求解的有限鲁棒 MDP

`ftm.robust_mdp_exact.TvRobustMDP` 对一个明确给定的有限 MDP、折扣因子 `gamma < 1`、以及每个状态动作各自的矩形总变差球，执行鲁棒值迭代。内层通过把概率质量从高价值后继状态转移到低价值后继状态，精确求得 `min_q q·V`。在这个有限模型及其 TV 不确定集内，Bellman 更新收敛到全局鲁棒最优策略。

```powershell
python -c "from ftm.robust_mdp_exact import two_state_tv_example; m=two_state_tv_example(); print(m.value_iteration(gamma=0.9))"
```

这个精确结论不自动推广到连续参数椭球、非矩形不确定集或原始高维仿真器。若要将仿真转移概率映射为 TV 半径，需要离散化后用留出数据校准半径，并报告校准误差。原来的 `ftm.robust_mdp` 仍保留“给定有限候选转移行”的精确枚举求解。

## 2. 与仿真相连的代理模型 PGD

硬 CTMC 抽样、任务状态枚举和迁移事务包含不可微分支，因此不能将普通 autograd 直接穿过这些仿真事件。新增路径使用学习到的一步条件期望代理：

1. `ftm.surrogate_dataset` 在硬事件环境里采样 `(s, a, 参数, s', r, 回报)` 记录。
2. `ftm.surrogate` 拟合一步状态增量/奖励模型及价值模型，对 Bellman 目标关于无约束扰动参数求梯度，并由梯度和经验尺度得到椭球轴长。
3. `ftm.surrogate_robust_ppo.SurrogatePGDRobustPPO` 在每个动作后通过 PGD 最小化代理 Bellman 目标，将扰动坐标映射回合法的故障倍率、拥塞概率、公共冲击概率和负载突发参数，再让原硬事件仿真器执行一步。PPO 的回报仍来自硬事件仿真，不来自代理预测。

这是一个可训练、端到端闭环的**近似**：PGD 只优化学得的单步条件期望，椭球对角轴忽略参数相关性，物理环境随后会重新采样离散事件。应在独立参数分布上检查代理的 next-state/reward 误差，并比较 PGD 选出的扰动与直接仿真评估；若误差偏大，应先扩充数据、加入验证集早停或缩小 `alpha`，而不是将代理结果解释成理论全局最坏转移。

采集数据（需 PyTorch 外不强制第三方依赖）：

```powershell
python -m ftm.surrogate_dataset --episodes-per-scenario 10 --scenario all --policy all --seed 2026
```

训练代理与策略需要已安装项目的可选 `learning` 依赖（PyTorch、NumPy、Gymnasium）：

```powershell
python -m ftm.train_surrogate_robust --dataset runs/surrogate_transitions.jsonl --episodes 100 --alpha 1.0 --pgd-steps 8 --device cpu
```

命令会输出 PPO/代理模型包、训练 CSV 与 PGD 摘要 JSON。它是真正执行 PPO 更新的训练入口；`ftm.main` 的回合命令只评估基线策略。

## 3. 多种子比较与区间

```powershell
python -m ftm.experiment_suite --episodes 30 --scenario all --policy all --seed 2026 --output runs/research
```

输出每回合原始记录、各策略指标的均值/百分位 bootstrap 区间，以及相同随机种子下相对 `no_migration` 的配对差值区间。配对结果同时给出双侧 sign-flip 随机化检验 p 值，并在同一扰动场景内对策略与指标检验做 Holm 校正。该检验依赖配对差值在零假设下可交换正负号；区间和 p 值只描述已运行的样本，不能证明分布外泛化。论文报告应冻结代码/参数，预先定义训练与测试场景，另留出未参与代理拟合和策略选择的测试种子。

标准固定场景 PPO 和代理 PGD PPO 可显式纳入同一评估器，且与 `no_migration` 使用配对种子：

```powershell
python -m ftm.train_rl --scenario mixed --episodes 100 --output models/ppo_mixed.pt
python -m ftm.experiment_suite --episodes 30 --scenario all --policy trained_ppo --checkpoint models/ppo_mixed.pt --output runs/ppo_eval
python -m ftm.experiment_suite --episodes 30 --scenario all --policy trained_surrogate_ppo --checkpoint models/surrogate_robust_ppo.pt --output runs/surrogate_ppo_eval
```

## 4. 真实执行器边界

`ftm.backend_api` 给出检查点存储、执行器、迁移传输与带 fencing epoch 的 lease registry 接口，以及 reserve→copy→restore→ready→CAS 切换→释放源端的两阶段迁移编排。当前没有 Kubernetes、SSH、对象存储或真实节点驱动。部署前需由具体平台实现这些接口，并验证：校验和/版本一致、lease CAS 原子性、旧执行器 fencing、失败清理可重试、检查点耐久性和应用侧幂等。原子租约只能避免两个受控执行器同时持有有效所有权；它不能保证任意外部副作用 exactly-once。

## 5. 分层结论

| 结论 | 当前实现范围 |
|---|---|
| 全局鲁棒最优 | 仅 `TVRobustMDP` 中有限状态、矩形 TV 不确定集的精确值迭代 |
| 连续状态策略 | PPO + 每步代理模型 PGD；策略/代理训练和硬事件 rollout 均为近似流程 |
| 统计比较 | 多场景、多策略、配对种子及 bootstrap 区间；实际区间必须运行后才能报告 |
| 真实集群 | 仅接口契约和两阶段控制流程；尚无真实平台接入 |
| 训练结果 | 代码提供训练入口；未运行并不等于已训练，模型文件需显式生成 |
