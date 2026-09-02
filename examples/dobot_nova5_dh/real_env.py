"""Real environment for single-arm Dobot Nova5 DH inference.

This wraps the ``lerobot`` robot used by the single-arm collection path.

State/action format (10D):
    [tcp.x, tcp.y, tcp.z, tcp.r1-r6, gripper.pos]

Camera keys:
    ``head`` and ``wrist``
"""

import collections
import time
from typing import Any

import dm_env
from lerobot.robots.dobot_nova5_dh.config_dobot_nova5_dh import ControlMode
from lerobot.robots.dobot_nova5_dh.config_dobot_nova5_dh import DobotNova5DHConfig
from lerobot.robots.dobot_nova5_dh.config_dobot_nova5_dh import ResetTarget
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import get_logger
import numpy as np

logger = get_logger("DobotNova5DHRealEnv")

_POLICY_CAMERAS = ("head", "wrist")
_ACTION_DIM = 10


class DobotNova5DHRealEnv:
    """Single right-arm Dobot Nova5 environment using Cartesian control."""

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
        setup_robot: bool = True,
    ) -> None:
        self.config = DobotNova5DHConfig(
            id="dobot_nova5_dh_inference",
            robot_ip=robot_ip,
            control_mode=ControlMode.CARTESIAN_MOTION,
            control_frequency=control_frequency,
            go_to_start=go_to_start,
            reset_target=ResetTarget(reset_target),
            go_to_home_on_disconnect=go_to_home_on_disconnect,
            async_action_worker=async_action_worker,
            async_action_worker_frequency=async_action_worker_frequency,
            enable_cartesian_ik_guard=enable_cartesian_ik_guard,
            cartesian_ik_servoj=cartesian_ik_servoj,
            max_cartesian_step_m=max_cartesian_step_m,
            enable_clip=enable_clip,
            enable_tactile_sensors=enable_tactile_sensors,
            use_gripper=use_gripper,
        )
        self.robot = make_robot_from_config(self.config)

        if setup_robot:
            self.setup_robot()

    def setup_robot(self) -> None:
        """Connect the arm, integrated DH gripper, and both cameras."""
        logger.info(f"Connecting to Dobot Nova5 DH at {self.config.robot_ip}...")
        try:
            self.robot.connect(calibrate=False, go_to_start=self.config.go_to_start)
            logger.info("Dobot Nova5 DH connected and ready")
        except Exception as exception:
            logger.error(f"Failed to connect Dobot Nova5 DH: {exception}")
            raise

    @staticmethod
    def get_qpos(obs: dict[str, Any]) -> np.ndarray:
        """Build the collection-compatible 10D Cartesian state vector."""
        tcp = [obs["tcp.x"], obs["tcp.y"], obs["tcp.z"]]
        tcp.extend(obs[f"tcp.r{index}"] for index in range(1, 7))
        return np.asarray([*tcp, obs["gripper.pos"]], dtype=np.float32)

    @staticmethod
    def get_images(obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Extract the two RGB images using their dataset camera names."""
        images = {}
        for camera_name in _POLICY_CAMERAS:
            if camera_name in obs:
                images[camera_name] = obs[camera_name]
            else:
                logger.warning(f"Camera {camera_name!r} is missing from the robot observation")
        return images

    def get_observation(self) -> dict[str, Any]:
        """Return an observation in the OpenPI/Xense runtime format."""
        raw_obs = self.robot.get_observation()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(raw_obs)
        obs["images"] = self.get_images(raw_obs)
        return obs

    @staticmethod
    def get_reward() -> float:
        return 0.0

    def reset(self, *, fake: bool = False) -> dm_env.TimeStep:
        """Move to the configured reset target and return the first observation."""
        if not fake:
            logger.info(f"Resetting Dobot Nova5 DH to {self.config.reset_target.value}...")
            try:
                self.robot.reset_to_initial_position()
                logger.info("Dobot Nova5 DH reset completed")
            except Exception as exception:
                logger.error(f"Failed to reset Dobot Nova5 DH: {exception}")
                raise

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Execute one 10D action and return the resulting observation."""
        self.send_action(action)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def send_action(self, action: np.ndarray) -> None:
        """Send a collection-compatible 10D Cartesian action to the robot."""
        action_dict = self._build_action_dict(action)
        start = time.perf_counter()
        try:
            self.robot.send_action(action_dict)
        except Exception as exception:
            logger.error(f"Failed to send Dobot Nova5 DH action: {exception}")
            raise
        logger.debug(f"send_action: {(time.perf_counter() - start) * 1e3:.2f}ms")

    @staticmethod
    def _build_action_dict(action: np.ndarray) -> dict[str, float]:
        """Convert a 10D policy action to ``DobotNova5DH.send_action`` keys."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (_ACTION_DIM,):
            raise ValueError(f"Expected a {_ACTION_DIM}D action, got shape {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError(f"Policy action contains NaN or Inf: {action}")

        action_dict = {
            "tcp.x": float(action[0]),
            "tcp.y": float(action[1]),
            "tcp.z": float(action[2]),
        }
        for index in range(6):
            action_dict[f"tcp.r{index + 1}"] = float(action[3 + index])
        action_dict["gripper.pos"] = float(np.clip(action[9], 0.0, 1.0))
        return action_dict

    def disconnect(self) -> None:
        """Disconnect the arm, gripper, and cameras."""
        logger.info("Disconnecting Dobot Nova5 DH...")
        try:
            # DobotNova5DH.disconnect() is idempotent. Calling it unconditionally
            # also cleans up the arm if a camera dropped and is_connected became false.
            self.robot.disconnect()
            logger.info("Dobot Nova5 DH disconnected")
        except Exception as exception:
            logger.warning(f"Error during Dobot Nova5 DH disconnect: {exception}")
