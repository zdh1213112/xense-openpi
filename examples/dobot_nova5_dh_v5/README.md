# Dobot Nova5 DH 单臂推理 V5

此入口是原 `examples.dobot_nova5_dh.main` 的独立副本。默认沿用原单臂
运行方式；需要与 101 V5 联动时，显式增加 `--args.turntable-interlock`。

该示例连接单臂 `dobot_nova5_dh`，硬件和字段与
`/home/zdh/loreal/loreal_lerobot/src/lerobot/robots/dobot_nova5_dh` 的采集代码一致：

- 机械臂：右臂 `192.168.111.102`
- 相机：`head`、`wrist`
- 状态和动作（10D）：`tcp.x/y/z`、`tcp.r1-r6`、`gripper.pos`
- 控制模式：`ControlMode.CARTESIAN_MOTION`

本项目中的 `LeRobotDobotNova5DataConfig` 与该 10D 字段及相机契约一致。旧的双臂
`LeRobotBiDobotNova5DHDataConfig` 是 20D，并且使用
`head`/`left_wrist`/`right_wrist`，不能直接用于此客户端。

## 依赖

活动环境中的 `lerobot` 需要包含本地的 `dobot_nova5_dh` 实现。如果尚未 editable install，可在命令前加入：

```bash
export PYTHONPATH=/home/zdh/loreal/loreal_lerobot/src:$PYTHONPATH
```

## 启动策略服务

在策略机器上运行，并替换 checkpoint 路径：

```bash
cd /home/zdh/xense-openpi
mamba run -n lerobot-xense-v4 python scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi05_base_dobot_nova5_loreal_returns_sorting_0831 \
    --policy.dir=<checkpoint_step_path>
```


```bash
cd /home/zdh/xense-openpi
mamba run -n lerobot-xense-v4 python scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi05_base_dobot_nova5_loreal_returns_sorting_0929 \
    --policy.dir=<checkpoint_step_path>
```

## Dry run

先连接真实机械臂和相机、检查观测与策略输出，但不下发策略动作：

```bash
cd /home/zdh/xense-openpi
mamba run -n lerobot-xense-v4 python -m examples.dobot_nova5_dh_v5.main \
    --args.host <policy_server_ip> \
    --args.port 8000 \
    --args.runtime-hz 30 \
    --args.rtc-enabled \
    --args.dry-run
```

双臂转盘联动时，在 V5 命令末尾显式添加 `--args.turntable-interlock`。
开启后，102 的放料动作目标 `tcp.y > 0.400 m` 前必须从 101 V5 ROS 节点
取得 `/dual_arm_102_turntable_lease`；102 退到 `tcp.y < 0.400 m` 并稳定后释放。

101 在转盘抓取和底面条码翻转期间通过
`/dual_arm_102_guard_xyz` 发布 Y/X/Z 危险区边界及阶段序号；102 排空异步动作后
回复 `/dual_arm_102_guard_ack`，101 收到对应确认后才进入抓取区。默认
102 User 0 / Tool 1 TCP 同时满足 `Y>325 mm、X<535 mm、Z>75 mm` 时
为危险区。102 保守检查从实测和上一目标到新目标的整段 Cartesian 动作，
命令边界留 5 mm；若可能穿过危险区，先保持 X/Z 并退到 Y<=320 mm，
再移动 X/Z。101 也持续检查实测 102 TCP，进入危险区时停止自身动作。
放料仍需原有 400 mm 租约。
`/dual_arm_102_post_place_fetch` 标识 101 已观测到本次放料后的 Y 撤离，
便于区分后续抓下一件与再次放料；该标识单独不会解除保护。抓下一件
经过 Y>400 mm 时，实测、上一个目标和新目标均须在 `X>=545 mm` 的
安全侧，才可免除放料租约；下一次危险 X 侧放料仍需许可。实测边界是否
覆盖夹爪、物料和两臂连杆的净空须由实机确认。
102 等待许可或受到区域限位时，保持最后一次已下发的夹爪目标，暂停消费
VLA 动作序列。许可恢复后丢弃旧动作块，使用最新观测重新推理；普通分块、
RTC 和观测/动作解耦模式都采用这一处理。101 抬离转盘后恢复正常动作。
101 在初始位等待首件时，或确认上一件物料及 TCP 已离开转盘、结束抓取/翻底保护后授予许可；后一种情况下 102 可以与 101 的固定点放置并行。
101 回初始位前会等待 102 退出放料区并释放许可。底面翻转仍在转盘侧时
不授予许可。VLA 每 100 ms
发布 `/dual_arm_102_interlock_heartbeat`。请先启动 ROS 101 V5 节点，再启动
本 VLA；旧 VLA 进程必须退出后再运行新代码。若 101/102 不在同一 ROS 域，
许可无法取得，102 将保持等待。双臂运行时必须使用
`--args.turntable-interlock`。本软件互锁不替代机器人安全区域、急停和实机
轨迹验证。

注意：运行时 `reset()` 仍会按 `--args.reset-target` 移动机械臂，默认目标为 `home`。
开启双臂互锁后按一次 Ctrl+C，V5 会先停止策略动作，再等待 101 自动授予转盘许可，
随后把 102 移到 `--args.reset-target`，确认实测 TCP 已退出放料区后释放许可并断开连接。
等待许可最多 60 秒；若 101 无法授予许可，102 会保持当前位置并记录错误，不会执行无许可的回位关节运动。

## 正式推理

确认 dry run 的 state shape 为 `(10,)`、图像键为 `head` 和 `wrist`、策略 action shape 为
`(10,)` 后，去掉 `--args.dry-run`：

```bash
cd /home/zdh/xense-openpi
mamba run -n lerobot-xense-v4 python -m examples.dobot_nova5_dh_v5.main \
    --args.host <policy_server_ip> \
    --args.port 8000 \
    --args.runtime-hz 30 \
    --args.rtc-enabled \
    --args.turntable-interlock
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--args.robot-ip` | `192.168.111.102` | 单臂 Dobot IP |
| `--args.turntable-interlock` | `False` | 显式开启与 101 V5 的双臂避障和放料许可 |
| `--args.reset-target` | `home` | 每个 episode 开始时移动到 `home` 或 `start` |
| `--args.go-to-start` | `False` | 连接阶段是否先去 start；runtime reset 仍会执行 |
| `--args.max-cartesian-step-m` | `0.05` | 每次笛卡尔动作最大平移距离 |
| `--args.no-enable-clip` | - | 关闭采集代码中的工作空间裁剪，不建议 |
| `--args.no-use-gripper` | - | 不连接/控制 DH 夹爪 |
| `--args.no-go-to-home-on-disconnect` | - | 退出时不自动回 home |
| `--args.action-hz` | `0` | 大于 0 时启用观测/动作解耦运行时 |
