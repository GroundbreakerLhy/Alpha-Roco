# Alpha Roco

conda 环境 `roco`。

```bash
python server.py
python client.py --team team0
python client.py --team data/teams/team4.json [--auto best|teacher]
```

模型入口在 `alpha_roco/`：`train.py`、`play.py`、`evaluate.py`。

## 训练

默认用 `watch` 启动：自动挑空闲显卡、按显存选 update chunk、中断后从最新 checkpoint 续跑。

```bash
tmux new -s train "python -m alpha_roco.watch --output <logs_location> --batches <iter_num> -- --warm-start <model> --learning-rate 2e-4 --entropy-coefficient 0.01"
```

`--` 之后的参数原样传给 `train.py`；`--batches` 是总批数，`--interval` 是检查显卡的间隔秒数（默认 1800）。启动时自动挑空闲的卡并按显存选 update chunk；运行中若另一张卡能放下更大的 chunk 且多出 15 GB 以上空闲显存，会中断当前批、从最新 checkpoint 切过去。
