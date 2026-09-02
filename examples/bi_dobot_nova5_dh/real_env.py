"""Real environment for BiDobot Nova5 DH dual-arm robot.

Wraps the lerobot BiDobotNova5DH robot for OpenPI/Xense websocket inference.

State/action format (20D):
    [left_tcp.x/y/z/r1-r6 (0-8), right_tcp.x/y/z/r1-r6 (9-17),
     left_gripper.pos (18), right_gripper.pos (19)]
"""

import collections
import time
from typing import Any

import dm_env
from lerobot.robots.bi_dobot_nova5_dh.config_bi_dobot_nova5_dh import BiDobotNova5DHConfig
from lerobot.robots.bi_dobot_nova5_dh.config_bi_dobot_nova5_dh import ControlMode
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import get_logger
import numpy as np

logger = get_logger("BiDobotNova5DHRealEnv")

_POLICY_CAMERAS = ("head", "left_wrist", "right_wrist")


class BiDobotNova5DHRealEnv:
    """Environment for BiDobot Nova5 DH dual-arm Cartesian inference.

    The underlying lerobot robot exposes the same 20D Cartesian state/action
    layout used by the BiPico4 collection path:
        left_tcp.{x,y,z,r1-r6} + right_tcp.{x,y,z,r1-r6}
        + left_gripper.pos + right_gripper.pos
    """

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
        setup_robot: bool = True,
    ):
        self.config = BiDobotNova5DHConfig(
            id="bi_dobot_nova5_dh_inference",
            left_robot_ip=left_robot_ip,
            right_robot_ip=right_robot_ip,
            control_mode=ControlMode.CARTESIAN_MOTION,
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
        )
        self.robot = make_robot_from_config(self.config)

        if setup_robot:
            self.setup_robot()

    def setup_robot(self) -> None:
        """Connect and initialize both Dobot arms, grippers, and cameras."""
        logger.info("Connecting to BiDobot Nova5 DH robot...")
        try:
            self.robot.connect(calibrate=False, go_to_start=self.config.go_to_start)
            logger.info("BiDobot Nova5 DH connected and ready")
        except Exception as e:
            logger.error(f"Failed to connect BiDobot Nova5 DH: {e}")
            raise

    def get_qpos(self, obs: dict[str, Any]) -> np.ndarray:
        """Build 20D state vector from observation dict.

        Ordering matches the dataset/action feature order:
            [left_tcp(0-8), right_tcp(9-17), left_gripper(18), right_gripper(19)]
        """
        left_tcp = [obs["left_tcp.x"], obs["left_tcp.y"], obs["left_tcp.z"]]
        left_tcp += [obs[f"left_tcp.r{i}"] for i in range(1, 7)]
        right_tcp = [obs["right_tcp.x"], obs["right_tcp.y"], obs["right_tcp.z"]]
        right_tcp += [obs[f"right_tcp.r{i}"] for i in range(1, 7)]
        return np.array(left_tcp + right_tcp + [obs["left_gripper.pos"], obs["right_gripper.pos"]], dtype=np.float32)

    def get_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Extract policy camera images from lerobot observation dict."""
        images = {}
        for cam_name in _POLICY_CAMERAS:
            if cam_name in obs:
                images[cam_name] = obs[cam_name]
            else:
                logger.debug(f"Camera {cam_name} not found in observation")
        return images

    def get_observation(self) -> dict[str, Any]:
        """Get observation compatible with OpenPI/Xense runtime format."""
        raw_obs = self.robot.get_observation()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(raw_obs)
        obs["images"] = self.get_images(raw_obs)
        return obs

    def get_reward(self) -> float:
        return 0.0

    def reset(self, *, fake: bool = False) -> dm_env.TimeStep:
        """Move both arms to the configured reset target and return the first observation."""
        if not fake:
            logger.info("Resetting BiDobot Nova5 DH...")
            try:
                self.robot.reset_to_initial_position()
                logger.info("BiDobot Nova5 DH reset completed")
            except Exception as e:
                logger.error(f"Failed to reset BiDobot Nova5 DH: {e}")
                raise

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Execute one 20D action and return the resulting observation."""
        self.send_action(action)
        obs = self.get_observation()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

    def send_action(self, action: np.ndarray) -> None:
        """Send a 20D Cartesian action to the robot without reading observations."""
        action_dict = self._build_action_dict(action)
        t0 = time.time()
        try:
            self.robot.send_action(action_dict)
        except Exception as e:
            logger.error(f"Failed to send BiDobot Nova5 DH action: {e}")
            raise
        logger.debug(f"send_action: {(time.time() - t0) * 1000:.2f}ms")

    def _build_action_dict(self, action: np.ndarray) -> dict[str, float]:
        """Build the per-key action dict expected by BiDobotNova5DH.send_action."""
        action_dict = {}
        action_dict["left_tcp.x"] = float(action[0])
        action_dict["left_tcp.y"] = float(action[1])
        action_dict["left_tcp.z"] = float(action[2])
        for i in range(6):
            action_dict[f"left_tcp.r{i + 1}"] = float(action[3 + i])

        action_dict["right_tcp.x"] = float(action[9])
        action_dict["right_tcp.y"] = float(action[10])
        action_dict["right_tcp.z"] = float(action[11])
        for i in range(6):
            action_dict[f"right_tcp.r{i + 1}"] = float(action[12 + i])

        action_dict["left_gripper.pos"] = float(np.clip(action[18], 0.0, 1.0))
        action_dict["right_gripper.pos"] = float(np.clip(action[19], 0.0, 1.0))
        return action_dict

    def disconnect(self) -> None:
        """Disconnect both arms, grippers, and cameras."""
        if self.robot.is_connected:
            logger.info("Disconnecting BiDobot Nova5 DH...")
            try:
                self.robot.disconnect()
                time.sleep(1)
                logger.info("BiDobot Nova5 DH disconnected")
            except Exception as e:
                logger.warning(f"Error during BiDobot Nova5 DH disconnect: {e}")
