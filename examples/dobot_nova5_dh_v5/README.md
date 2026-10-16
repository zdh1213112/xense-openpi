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
102 User 0 / Tool 1 TCP 同时满足 `Y>300 mm、X<535 mm、Z>75 mm` 时
为危险区。102 保守检查从实测和上一目标到新目标的整段 Cartesian 动作，
Y 命令边界默认提前 10 mm；若可能穿过危险区，先保持 X/Z 并退到 Y<=290 mm，
再移动 X/Z。101 也持续检查实测 102 TCP，进入危险区时停止自身动作。
放料仍需原有 400 mm 租约。
提前量由 102 V5 启动参数 `--args.turntable-guard-y-margin-m 0.010` 配置，
单位为米；限幅、Y 路径检查和保护期间未使用许可的归还判断共用该值。
X/Z 的路径检查余量仍为 5 mm。101 的 300 mm 越界停止线保持不变。
`/dual_arm_102_post_place_fetch` 标识 101 已观测到本次放料后的 Y 撤离，
便于区分后续抓下一件与再次放料；该标识单独不会解除保护。抓下一件
经过 Y>400 mm 时，实测、上一个目标和新目标均须在 `X>=545 mm` 的
安全侧，才可免除放料租约；下一次危险 X 侧放料仍需许可。实测边界是否
覆盖夹爪、物料和两臂连杆的净空须由实机确认。
102 等待许可或受到区域限位时，保持最后一次已下发的夹爪目标，暂停消费
VLA 动作序列。许可恢复后丢弃旧动作块，使用最新观测重新推理；普通分块、
RTC 和观测/动作解耦模式都采用这一处理。101 抬离转盘后恢复正常动作。

### 暂停后的恢复推理

101 确认物料和 TCP 离开转盘、进入放置阶段时，立即通过
`/dual_arm_102_placement_ready` 发布可领取放料许可的状态，并每 100 ms 刷新。
102 在后台每 20 ms 异步检查并领取，取下一件物料时即可持有许可，不再等
动作目标越过 400 mm 才申请。该通知只是窗口信息，最终进入许可仍由原有
租约服务确认；通知或保护反馈超过 500 ms 未更新时，不发起提前领取。
提前领取的租约不会被 `X>=545 mm` 安全侧取料经过 400 mm 误释放；只有
实测进入放料侧，随后按原条件稳定撤离，才会正常释放。

偶尔回复未及时到达时，102 在实测 Y<=390 mm、抓取保护未生效且动作未被
改写的情况下，可以保留同一条动作等待最多 150 ms。等待期间不发送该动作
及其夹爪指令，运行时每个控制周期用最新状态重试；不清空 VLA 队列，也不
额外推理。同步与解耦运行时都保持动作顺序。收到拒绝、保护介入或等待超过
150 ms 时，仍进入保持和重新规划，迟到的许可不会直接执行过期动作。

因抓取保护限制而暂停、且目标不属于安全 X 侧取料通道时，102 在保护解除后
先异步取得放料租约，再用最新观测重新规划。即使被暂停的旧目标仍低于
400 mm，也会先确认下一次放料许可，避免“保护解除后规划一次、首动作进入
放料区时再次等待租约并规划一次”。安全 X 侧取料保留原有的整段检查和免租约条件。

RTC 的两阶段 warmup 仅用于 episode 启动。暂停恢复使用新的 broker 丢弃旧
队列和过期推理结果，发送固定形状的实测保持姿态前缀，且 `inference_delay=0`；
前缀不冻结任何动作，一次推理完成后从新轨迹第 0 步执行，不再把未执行的
预测轨迹中段作为恢复起点。发送恢复首动作前重新读取实际反馈，并再次检查
保护区与租约。普通分块模式也直接从新观测推理；解耦模式继续清理预取动作队列。

提前取得放料许可时打印 `102 placement lease acquired ahead of approach`。
短暂等待结束时打印 `continuing the retained action without replanning`。
真正保护暂停后的恢复会打印：

```text
102 protection cleared and resume permission ready; lease_held=True; ...
RTC: Resume inference from held state; inference_delay=0, prev_chunk shape=(30, 10)
RTC: Resume plan ready after one inference; starting at action 0
102 VLA replanned once from the latest observation after resume permission
```

前缀行列数随队列阈值和动作维度变化，安全 X 侧取料的恢复许可可以不持有
放料租约。租约申请进行中显示 `grant request pending`，不再沿用上一轮的
`lease held` 状态。一次恢复仍需等待一次策略推理；具体耗时以新日志为准。

### 101 联动与退出

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


## V5 反馈与动作线程诊断

V5 使用本目录的 `feedback_robot.py` 适配 LeRobot 单臂驱动。30004 反馈按
1440 字节帧重组，保留分片、跳过失步数据，并使用同次接收中的最新完整帧；
读取失败会记录原因并重连，不再静默退出。相同控制器时间戳不会刷新反馈年龄。
超过 100 ms 没有新反馈时，V5 拒绝动作、开爪和回位命令，并报告错误。

102 已经进入联锁暂停、且当前姿态保持命令成功下发后，观测遇到短暂反馈超时
会保持动作阻断，排空动作线程，额外等待最多 2 秒的新鲜反馈，避免仅因一次
106 ms 延迟直接退出。等待期间保留 ROS 心跳，不返回缓存姿态、不发送新的 VLA
运动或夹爪指令；重复控制器时间戳不算恢复。恢复后仍需抓取保护解除并取得必要的
放料许可，使用新观测重新规划。运动中、尚未成功建立保持、用户正在停止或
持续断流时，仍按故障处理；100 ms 的命令新鲜度限制保持不变。

动作线程采用暂停计数，重叠的 ROS/策略/复位暂停退出后可恢复；取出的动作在
发送限频等待期间也计入正在执行，保护状态确认前会等它发送完成。
心跳由独立线程发布，避免保护切换回调阻塞心跳。每 2 秒的
`102 execution status` 显示策略阻塞、许可、保护状态、反馈年龄、反馈线程、
动作线程暂停/忙碌状态，以及实际发送/错误计数。Runtime 耗时只表示调用耗时，
不能作为机械臂动作成功的证据。

V5 断开夹爪通信时保留最后下发的位置，清理错误不会额外开爪；显式回初始位后的夹爪初始化仍按复位流程执行。
断连前先停止反馈读取及重连；非联锁模式的可选回位在停止读取之前完成。
