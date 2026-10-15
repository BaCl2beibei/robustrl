# 修改前源码备份

`source_snapshot.zip` 是加入 PPO 候选节点六项特征前的源码快照，包含当时的 `ftm/`、`scripts/`、`tests/`、`.vscode/`、`.gitignore` 和 `实验.md`，也包含当时尚未提交的源码改动。它不包含 `.git/`、`models/` 和 `runs/`；这些目录没有被本次代码修改触及。

`git_status.txt` 记录备份时的工作区状态；`working_changes.patch` 记录备份时已跟踪文件的未提交改动。恢复单个文件时，从 ZIP 中取出对应路径并复制回项目即可。恢复整个源码快照前，先保存备份之后的新改动，再解压 ZIP 覆盖对应文件。
