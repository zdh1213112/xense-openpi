"""OpenPI Environment wrapper for BiDobot Nova5 DH dual-arm robot."""

import einops
from lerobot.utils.robot_utils import get_logger
import numpy as np
from typing_extensions import override
from xense_client import image_tools
from xense_client.runtime import environment as _environment

import examples.bi_dobot_nova5_dh.real_env as _real_env

logger = get_logger("BiDobotNova5DHEnv")

_ACTION_LABELS = [
    "L.x",
    "L.y",
    "L.z",
    "L.r1",
    "L.r2",
    "L.r3",
    "L.r4",
    "L.r5",
    "L.r6",
    "R.x",
    "R.y",
    "R.z",
    "R.r1",
    "R.r2",
    "R.r3",
    "R.r4",
    "R.r5",
    "R.r6",
    "L.grip",
    "R.grip",
]


class BiDobotNova5DHEnvironment(_environment.Environment):
    """OpenPI environment for BiDobot Nova5 DH Cartesian inference."""

    def __init__(
        self,
        left_robot_ip: str = "192.168.111.101",
        right_robot_ip: str = "192.168.111.102",
        control_frequency: float = 100.0,
        go_to_start: bool = False,
        async_action_worker: bool = True,
        async_action_worker_frequency: float = 30.0,
        enable_cartesian_ik_guard: bool = True,
        cartesian_ik_servoj: bool = True,
        max_cartesian_step_m: float = 0.05,
        enable_clip: bool = True,
        enable_tactile_sensors: bool = False,
        use_left_gripper: bool = True,
        use_right_gripper: bool = True,
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,
    ) -> None:
        self._env = _real_env.BiDobotNova5DHRealEnv(
            left_robot_ip=left_robot_ip,
            right_robot_ip=right_robot_ip,
            control_frequency=control_frequency,
            go_to_start=go_to_start,
            async_action_worker=async_action_worker,
            async_action_worker_frequency=async_action_worker_frequency,
            enable_cartesian_ik_guard=enable_cartesian_ik_guard,
            cartesian_ik_servoj=cartesian_ik_servoj,
            max_cartesian_step_m=max_cartesian_step_m,
            enable_clip=enable_clip,
            enable_tactile_sensors=enable_tactile_sensors,
            use_left_gripper=use_left_gripper,
            use_right_gripper=use_right_gripper,
            setup_robot=setup_robot,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._step_count = 0

    @override
    def reset(self) -> None:
        self._env.reset()
        self._step_count = 0

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        raw_obs = self._env.get_observation()

        processed_images = {}
        for cam_name, img in raw_obs["images"].items():
            if "_depth" in cam_name or "tactile" in cam_name:
                continue
            batch = np.expand_dims(img, axis=0)
            resized = image_tools.resize_with_pad(batch, self._render_height, self._render_width)[0]
            processed_images[cam_name] = einops.rearrange(resized, "h w c -> c h w")

        raw_images = {
            cam: img for cam, img in raw_obs["images"].items() if "_depth" not in cam and "tactile" not in cam
        }

        return {
            "state": raw_obs["qpos"],
            "images": processed_images,
            "images_raw": raw_images,
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._step_count += 1
        actions = action.get("actions")
        if actions is not None:
            parts = " | ".join(f"{lbl}={v:+.4f}" for lbl, v in zip(_ACTION_LABELS, actions))
            logger.debug(f"Step {self._step_count}: {parts}")
        self._env.send_action(action["actions"])

    def disconnect(self) -> None:
        self._env.disconnect()
