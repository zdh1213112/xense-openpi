# BiDobot client for shilin-vla

This client keeps the existing `BiDobotNova5DH` robot/environment stack but
talks to `shilin-vla/deploy/lingbot_vla_v2_policy.py`.

At the websocket boundary it converts:

- robot state: 20D `xyz + first two rotation-matrix columns + grippers`
  to LingBot's 16D `xyz + quaternion_xyzw + grippers`;
- LingBot action chunks: 16D back to the robot's 20D representation;
- OpenPI observation keys to the raw LeRobot keys expected by
  `configs/robot_configs/own_robot.yaml`.

## 1. Start the LingBot server

Run on the GPU machine:

```bash
cd ~/AI_Model/shilin-vla
python deploy/lingbot_vla_v2_policy.py \
    --model_path /path/to/checkpoint/hf_ckpt \
    --port 8006 \
    --chunk_ret true \
    --use_length 50 \
    --use_compile true
```

The checkpoint's `lingbotvla_cli.yaml` must point
`data.norm_stats_file` at the normalization statistics used to train this
robot.

## 2. Dry-run the robot client

Run on the robot machine:

```bash
cd ~/AI_Model/xense-openpi
python -m examples.bi_dobot_shilin.main \
    --args.host <gpu_machine_ip> \
    --args.port 8006 \
    --args.robot-name own_robot \
    --args.task "your task instruction" \
    --args.action-horizon 50 \
    --args.dry-run
```

Dry-run still connects and resets the physical robot, but converted policy
actions are only logged.

## 3. Run real inference

After checking image/state/action values, remove `--args.dry-run`:

```bash
python -m examples.bi_dobot_shilin.main \
    --args.host <gpu_machine_ip> \
    --args.port 8006 \
    --args.robot-name own_robot \
    --args.task "Pick up the returned cosmetic product, scan its barcode, and place it into the return cart." \
    --args.action-horizon 50
```

`--args.action-horizon` must match the chunk length returned by the server.
This client intentionally uses `ActionChunkBroker`; the LingBot server does
not implement OpenPI RTC prefix conditioning.

## Reset behavior

The stock Xense `WebsocketClientPolicy.reset()` is a no-op. The adapter in
`client_policy.py` overrides reset semantics and sends:

```python
{"reset": True, "robo_name": "own_robot"}
```

The Xense Runtime calls this at the beginning of every episode through
`ActionChunkBroker.reset()`. `LingbotVLAv2Server.infer()` receives the control
message on the GPU machine and calls its server-side `reset()`, where
`FeatureTransform` is actually created and action-chunk state is cleared.

## Wire data

Each regular inference request sent to LingBot contains:

```python
{
    "observation.state": state_16d,
    "observation.images.head": head_hwc,
    "observation.images.left_wrist": left_wrist_hwc,
    "observation.images.right_wrist": right_wrist_hwc,
    "task": task_instruction,
}
```

All three images are required and must be RGB. Position units and gripper
scales must be the same as the data used to compute the training
normalization statistics.
