# Q 表输出说明

本目录保留给手动保存或复用的 tabular Q 表。

默认评估脚本会把训练出的 Q 表写入对应 run 目录：

`experiments/runs/<run_id>/tables/tabular_q_policy.json`

如需复用某个 Q 表，可将其路径传给：

```powershell
.\.venv\Scripts\python.exe comparison_algorithms\scripts\run_comparison.py `
  --q_table_path experiments\runs\<run_id>\tables\tabular_q_policy.json
```

