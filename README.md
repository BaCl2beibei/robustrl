# 面向分布式节点容错的任务迁移仿真

这是一个单机 CPU 可运行的科研原型，围绕“节点故障—网络波动—检查点迁移—鲁棒决策—韧性评估”组织代码。核心仿真仅依赖 Python 标准库；PyTorch、NumPy 和 Gymnasium 是后续深度强化学习扩展的可选依赖。

## 当前实现内容

- 两状态连续时间马尔可夫节点模型，含固定步长精确转移概率、瞬态可用率和异构 (k)-of-(N) 可用度。
- 正常/拥塞两状态链路，以及正值截断的双峰高斯混合时延采样。
- 故障率/修复率与两成分网络时延混合参数的基础估计函数。
- 节点公共冲击、突发背景负载和持续拥塞状态。
- Node、Task、Cluster、Network、MigrationManager 和固定步长环境。
- 带检查点版本、目标资源预留、传输、恢复、确认和失败回滚的迁移状态机。
- 不迁移、随机、贪心、负载优先和保守最坏情形规则五种策略。
- 可配置的离散有限状态鲁棒值迭代，内层在传入的有限候选转移行上精确求最小值。
- 敏感度椭球、投影和 PGD 工具；可选 PyTorch 重参数化与 autograd 辅助函数。
- 可选 PyTorch 掩码 PPO 训练器；训练回合按近期折扣回报选择有限场景中的最差场景。
- 可选的一步转移代理、Bellman 敏感度椭球校准，以及每个决策周期上的代理 PGD 对抗 PPO 训练闭环。
- 有限 MDP 上对矩形总变差球精确求解的鲁棒值迭代，以及多场景/配对种子 bootstrap 评估。
- 供真实系统适配的两阶段迁移接口契约；当前仍是接口，不含真实集群驱动。
- 续算成功率、(k)-of-(N) 集群可用性、任务完成时间、SLA 违约率、迁移次数/时间、CPU/内存利用率，以及 CSV 事件和指标导出。

## 运行方式

在本目录中运行：

```powershell
python -m ftm.main --mode robust-demo
python -m ftm.main --mode episodes --scenario mixed --policy robust_heuristic --episodes 5 --seed 2026
python -m ftm.train_rl --episodes 100 --scenario all --seed 2026 --device cpu
python -m ftm.train_rl --episodes 100 --scenario mixed --seed 2026 --device cpu
python -m ftm.experiment_suite --episodes 30 --scenario all --policy all --output runs/research
python -m ftm.surrogate_dataset --episodes-per-scenario 10 --scenario all --policy all
python -m ftm.train_surrogate_robust --dataset runs/surrogate_transitions.jsonl --episodes 100 --device cpu
```

`--scenario all` 用有限场景对抗者训练；指定单一场景则训练普通固定分布 PPO。将训练好的标准 PPO 或代理 PGD PPO 纳入配对测试时，分别使用 `--policy trained_ppo` 或 `--policy trained_surrogate_ppo` 并传入 `--checkpoint <模型路径>`。`--policy all` 指五个无学习策略基线，不会自动读取模型。

可选策略：`no_migration`、`random`、`greedy`、`load_based`、`robust_heuristic`。可选场景：`normal`、`failure`、`network`、`load`、`mixed`。默认结果写到 `runs/`，每行对应一个 episode，保留随机种子以便复现。

安装为可编辑包时可运行 `python -m pip install -e .`。强化学习依赖安装方式为 `python -m pip install -e ".[learning]"`；当前版本不自动安装任何依赖。

## 关键接口

```python
from ftm.config import SimulationConfig
from ftm.environment import MigrationEnvironment, scenario_by_name
from ftm.policies import RobustHeuristicPolicy

config = SimulationConfig(n_nodes=4, n_tasks=6, k_required=3)
env = MigrationEnvironment(config, scenario_by_name("mixed", config))
observation, info = env.reset(seed=7)
policy = RobustHeuristicPolicy()
action = policy.act(env)  # 0..N-1 表示目标节点，N 表示等待
observation, reward, terminated, truncated, info = env.step(action)
```

`MigrationEnvironment` 提供 `reset/step` 风格接口，但不继承 Gymnasium，避免 MVP 强依赖第三方包。动作目标筛选由 `MigrationManager.valid_targets` 和 `valid_action_mask` 完成；迁移中的焦点任务不再被重复迁移。

## 模块结构

```text
ftm/
├── config.py          # 场景和仿真参数
├── entities.py        # 节点、任务、检查点状态、事件
├── reliability.py     # CTMC、k-of-N 可用度
├── disturbances.py    # 节点/网络/负载联合扰动
├── network.py         # 时延、带宽、传输估算
├── cluster.py         # 资源分配、执行、故障响应
├── migration.py       # 检查点迁移事务状态机
├── environment.py     # reset/step、奖励、观测
├── policies.py        # 基线与保守规则
├── model_estimation.py # CTMC 参数和双峰高斯混合估计
├── robust_mdp.py      # 离散鲁棒值迭代
├── uncertainty.py     # 敏感度椭球与纯 Python PGD
├── torch_ops.py       # 可选的 PyTorch 重参数化/autograd
├── robust_ppo.py      # 可选 PPO 和有限场景对抗
├── train_rl.py        # PPO 命令行训练入口
├── metrics.py         # episode 指标
└── main.py            # CLI 入口
```

## 数学实现说明

1. **节点 CTMC**：正常到故障概率为 \(\frac{\lambda}{\lambda+\mu}(1-e^{-(\lambda+\mu)\Delta})\)，故障到修复概率为 \(\frac{\mu}{\lambda+\mu}(1-e^{-(\lambda+\mu)\Delta})\)。故障前未提交的进度会丢失，恢复到最新检查点。
2. **集群可用度**：`k_of_n_availability` 计算异构伯努利节点的泊松二项尾概率。公共冲击场景不满足独立假设，经验可用度由仿真状态直接统计。
3. **BGMM 时延**：按链路正常/拥塞状态抽取对应高斯分量，并拒绝非正样本。链路状态具备持续性；拥塞还降低有效带宽。平稳正常分量权重为 `congestion_leave / (congestion_enter + congestion_leave)`，可从场景对象的 `bgmm_normal_weight` 读取。
4. **迁移事务**：目标先预留资源，检查点版本固定，随后传输和恢复。确认时目标预留提升为任务占用并释放原资源；失败时释放目标预留，任务回到源节点或保留为中断状态。
5. **不确定集**：`sensitivity_axes` 保证每维轴长有正下界；`Ellipsoid` 在无约束坐标中构造集合。`projected_gradient_minimize` 返回局部 PGD 候选，不保证全局最坏解。
6. **离散鲁棒 MDP**：`FiniteRobustMDP` 对每个状态动作提供的有限候选概率行取最小值。`two_state_example` 用于核对标称和鲁棒策略的差异。它不等于对连续椭球的精确优化。
7. **PyTorch 梯度**：`torch_ops.gaussian_reparameterize` 使用 `mu + L epsilon`，其中 `L` 是 Cholesky 因子。离散事件需要枚举并按概率加权；不能期待硬采样类别自动获得路径梯度。

## 近似范围与适用边界

`train_rl.py` 实现 PPO Actor-Critic 和有限场景对抗。`surrogate_dataset.py` 与 `train_surrogate_robust.py` 是另一个训练入口：先用硬事件仿真数据拟合可微一步代理，在代理 Bellman 目标上做 PGD，再将合法的物理扰动参数用于后续硬事件 rollout。代理并非原离散仿真路径，故这一连续扰动优化仍是近似，必须评估代理误差和独立测试场景。

`robust_heuristic` 只是保守基线，不能视为式（2-25）的最优策略。`FiniteRobustMDP` 对输入的有限候选转移行精确枚举内层最小值；`TVRobustMDP` 对有限状态、矩形 TV 球精确求内层并执行鲁棒值迭代，但这不等于连续参数椭球上的全局 min-max。多种子脚本输出 bootstrap 描述区间，不单独证明统计显著性或分布外泛化。任务模型假设检查点可序列化、检查点存储可从目标节点访问；真实接口契约见 `ftm.backend_api`，当前未接入真实集群，外部副作用 exactly-once 需由应用另行实现。详见 [ADVANCED_ROUTE.md](ADVANCED_ROUTE.md)。
