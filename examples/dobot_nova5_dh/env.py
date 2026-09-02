"""OpenPI environment wrapper for the single-arm Dobot Nova5 DH."""

import einops
from lerobot.utils.robot_utils import get_logger
import numpy as np
from typing_extensions import override
from xense_client import image_tools
from xense_client.runtime import environment as _environment

import examples.dobot_nova5_dh.real_env as _real_env

logger = get_logger("DobotNova5DHEnv")

_ACTION_LABELS = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "grip"]


class DobotNova5DHEnvironment(_environment.Environment):
    """OpenPI environment for 10D single-arm Cartesian inference."""

    def __init__(
        self,
        robot_ip: str = "192.168.111.102",
        control_frequency: float = 100.0,
        go_to_start: bool = False,
        reset_target: str = "home",
        go_to_home_on_disconnect: bool = True,
        async_action_worker: bool = True,
        async_action_worker_frequency: float = 30.0,
        enable_cartesian_ik_guard: bool = True,
        cartesian_ik_servoj: bool = True,
        max_cartesian_step_m: float = 0.05,
        enable_clip: bool = True,
        enable_tactile_sensors: bool = False,
        use_gripper: bool = True,
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,
    ) -> None:
        self._env = _real_env.DobotNova5DHRealEnv(
            robot_ip=robot_ip,
            control_frequency=control_frequency,
            go_to_start=go_to_start,
            reset_target=reset_target,
            go_to_home_on_disconnect=go_to_home_on_disconnect,
            async_action_worker=async_action_worker,
            async_action_worker_frequency=async_action_worker_frequency,
            enable_cartesian_ik_guard=enable_cartesian_ik_guard,
            cartesian_ik_servoj=cartesian_ik_servoj,
            max_cartesian_step_m=max_cartesian_step_m,
            enable_clip=enable_clip,
            enable_tactile_sensors=enable_tactile_sensors,
            use_gripper=use_gripper,
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
        raw_images = {}
        for camera_name, image in raw_obs["images"].items():
            if "_depth" in camera_name or "tactile" in camera_name:
                continue
            batch = np.expand_dims(image, axis=0)
            resized = image_tools.resize_with_pad(batch, self._render_height, self._render_width)[0]
            processed_images[camera_name] = einops.rearrange(resized, "h w c -> c h w")
            raw_images[camera_name] = image

        return {
            "state": raw_obs["qpos"],
            "images": processed_images,
            "images_raw": raw_images,
        }

    @override
    def apply_action(self, action: dict) -> None:
        actions = action.get("actions")
        if actions is None:
            raise KeyError("Policy result does not contain an 'actions' field")

        self._step_count += 1
        flat_action = np.asarray(actions).reshape(-1)
        parts = " | ".join(f"{label}={value:+.4f}" for label, value in zip(_ACTION_LABELS, flat_action))
        logger.debug(f"Step {self._step_count}: {parts}")
        self._env.send_action(flat_action)

    def disconnect(self) -> None:
        self._env.disconnect()
