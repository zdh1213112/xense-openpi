# Dobot Nova5 DH 单臂推理

该示例连接单臂 `dobot_nova5_dh`，硬件和字段与
`/home/zdh/loreal/loreal_lerobot/src/lerobot/robots/dobot_nova5_dh` 的采集代码一致：

- 机械臂：右臂 `192.168.111.102`
- 相机：`head`、`wrist`
- 状态和动作（10D）：`tcp.x/y/z`、`tcp.r1-r6`、`gripper.pos`
- 控制模式：`ControlMode.CARTESIAN_MOTION`

推理所用 checkpoint 必须由相同的 10D 字段顺序和 `head`/`wrist` 相机键训练。旧的双臂
checkpoint 是 20D，并且使用 `head`/`left_wrist`/`right_wrist`，不能直接用于此客户端。

## 依赖

活动环境中的 `lerobot` 需要包含本地的 `dobot_nova5_dh` 实现。如果尚未 editable install，可在命令前加入：

```bash
export PYTHONPATH=/home/zdh/loreal/loreal_lerobot/src:$PYTHONPATH
```

## 启动策略服务

在策略机器上运行，并替换训练配置和 checkpoint 路径：

```bash
cd /home/zdh/xense-openpi
mamba run -n lerobot-xense-v4 python scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=<single_arm_train_config> \
    --policy.dir=<checkpoint_step_path>
```

## Dry run

先连接真实机械臂和相机、检查观测与策略输出，但不下发策略动作：

```bash
cd /home/zdh/xense-openpi
mamba run -n lerobot-xense-v4 python -m examples.dobot_nova5_dh.main \
    --args.host <policy_server_ip> \
    --args.port 8000 \
    --args.runtime-hz 30 \
    --args.rtc-enabled \
    --args.dry-run
```

注意：运行时 `reset()` 仍会按 `--args.reset-target` 移动机械臂，默认目标为 `home`。

## 正式推理

确认 dry run 的 state shape 为 `(10,)`、图像键为 `head` 和 `wrist`、策略 action shape 为
`(10,)` 后，去掉 `--args.dry-run`：

```bash
cd /home/zdh/xense-openpi
mamba run -n lerobot-xense-v4 python -m examples.dobot_nova5_dh.main \
    --args.host <policy_server_ip> \
    --args.port 8000 \
    --args.runtime-hz 30 \
    --args.rtc-enabled
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--args.robot-ip` | `192.168.111.102` | 单臂 Dobot IP |
| `--args.reset-target` | `home` | 每个 episode 开始时移动到 `home` 或 `start` |
| `--args.go-to-start` | `False` | 连接阶段是否先去 start；runtime reset 仍会执行 |
| `--args.max-cartesian-step-m` | `0.05` | 每次笛卡尔动作最大平移距离 |
| `--args.no-enable-clip` | - | 关闭采集代码中的工作空间裁剪，不建议 |
| `--args.no-use-gripper` | - | 不连接/控制 DH 夹爪 |
| `--args.no-go-to-home-on-disconnect` | - | 退出时不自动回 home |
| `--args.action-hz` | `0` | 大于 0 时启用观测/动作解耦运行时 |
