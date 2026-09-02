# BiDobot Nova5 DH Inference

This example runs OpenPI/Xense websocket inference on the `lerobot` robot type you used for collection:

```bash
--robot.type=bi_dobot_nova5_dh \
--teleop.type=bi_pico4
```

The inference runtime does not use `--teleop.type`; it connects directly to the trained policy server and sends 20D Cartesian actions to `BiDobotNova5DH`.

## Assumptions

- `lerobot==0.4.1` is installed from `/home/zdh/loreal/loreal_lerobot` in the active environment.
- The trained policy expects the same feature order as the BiPico4 collection path:
  `[left_tcp(9), right_tcp(9), left_gripper.pos, right_gripper.pos]`.
- Cameras are the default `bi_dobot_nova5_dh` cameras: `head`, `left_wrist`, `right_wrist`.
- Default robot IPs are `192.168.111.101` and `192.168.111.102`.

If `python` cannot import `lerobot`, either install it editable or prefix the client commands with:

```bash
PYTHONPATH=/home/zdh/loreal/loreal_lerobot/src:$PYTHONPATH
```

## 1. Start Policy Server

Run this on the GPU/policy machine. Replace the config and checkpoint path with your trained Dobot policy:

```bash
cd /home/zdh/xense-openpi
mamba run -n lerobot-xense python scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=<your_dobot_train_config> \
    --policy.dir=<path/to/checkpoint/step>
```

The server listens on port `8000` by default.

## 2. Dry-Run Robot Client

Run this on the robot machine first. It connects robot/cameras and prints policy actions without sending policy actions to the arms. Reset/home motions still run because the runtime connects real hardware:

```bash
cd /home/zdh/xense-openpi
mamba run -n lerobot-xense python -m examples.bi_dobot_nova5_dh.main \
    --args.host <policy_server_ip> \
    --args.port 8000 \
    --args.control-frequency 100 \
    --args.runtime-hz 30 \
    --args.rtc-enabled \
    --args.dry-run
```

## 3. Real Robot Inference

Remove `--args.dry-run` after verifying observations, cameras, and action values:

```bash
cd /home/zdh/xense-openpi
mamba run -n lerobot-xense python -m examples.bi_dobot_nova5_dh.main \
    --args.host <policy_server_ip> \
    --args.port 8000 \
    --args.control-frequency 100 \
    --args.runtime-hz 30 \
    --args.rtc-enabled
```

## Useful Options

| Flag | Default | Description |
|---|---:|---|
| `--args.left-robot-ip` | `192.168.111.101` | Override left arm IP only if your robot network differs. |
| `--args.right-robot-ip` | `192.168.111.102` | Override right arm IP only if your robot network differs. |
| `--args.go-to-start` | `False` | Move to configured start pose during connect. |
| `--args.enable-clip` | `True` | Use lerobot workspace clipping before ServoP. |
| `--args.max-cartesian-step-m` | `0.05` | Max per-action Cartesian step in meters. |
| `--args.async-action-worker` / `--args.no-async-action-worker` | `True` | Queue newest action and send via Dobot worker thread. |
| `--args.async-action-worker-frequency` | `30` | Worker send frequency. |
| `--args.use-left-gripper` / `--args.no-use-left-gripper` | `True` | Enable left DH gripper. |
| `--args.use-right-gripper` / `--args.no-use-right-gripper` | `True` | Enable right DH gripper. |
| `--args.action-hz` | `0` | Enable decoupled obs/action runtime when `>0`. |

If only the right DH gripper is mounted, add:

```bash
--args.no-use-left-gripper
```
