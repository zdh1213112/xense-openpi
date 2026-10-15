"""Real environment for single-arm Dobot Nova5 DH inference.

This wraps the ``lerobot`` robot used by the single-arm collection path.

State/action format (10D):
    [tcp.x, tcp.y, tcp.z, tcp.r1-r6, gripper.pos]

Camera keys:
    ``head`` and ``wrist``
"""

import collections
import threading
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
        turntable_interlock: bool = False,
        turntable_safe_y_m: float = 0.400,
        setup_robot: bool = True,
    ) -> None:
        self._guard_action_lock = threading.RLock()
        # The V5 interlock is opt-in. Without it, keep the original VLA
        # runtime independent of ROS and its shared libraries.
        if turntable_interlock:
            from examples.dobot_nova5_dh_v5.turntable_lease import TurntableLease

            self.turntable_lease = TurntableLease(safe_y_m=turntable_safe_y_m)
        else:
            self.turntable_lease = None
        self._pick_guard_log_state = "clear"
        self._latest_tcp_y = None
        self._latest_raw_obs = None
        self._latest_raw_obs_at = 0.0
        self._last_target_tcp_y = float("inf")
        self._last_target_tcp_x = float("-inf")
        self._last_target_tcp_z = float("-inf")
        self._last_lease_wait_log_at = 0.0
        self._last_gripper_target = None
        self._policy_blocked = False
        self._blocked_action = None
        self._policy_generation = 0
        self.config = DobotNova5DHConfig(
            id="dobot_nova5_dh_inference",
            robot_ip=robot_ip,
            control_mode=ControlMode.CARTESIAN_MOTION,
            control_frequency=control_frequency,
            # connect() can issue a joint move before reset() obtains a
            # turntable lease. Keep all reset motion behind the lease.
            go_to_start=(go_to_start and not turntable_interlock),
            reset_target=ResetTarget(reset_target),
            # An unannounced disconnect-time MovJ could enter 101's zone.
            go_to_home_on_disconnect=(
                go_to_home_on_disconnect and not turntable_interlock
            ),
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
        if self.turntable_lease is not None:
            self.turntable_lease.set_guard_transition_handler(
                self._install_guard_after_worker_drain
            )

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

    def _install_guard_after_worker_drain(self, install_guard) -> None:
        with self._guard_action_lock:
            with self.robot._pause_action_worker():
                install_guard()
                if self.turntable_lease.pick_guard_active and self.robot.is_connected:
                    # A Servo command already sent by the worker can keep
                    # moving after its queue is drained. Replace that target
                    # with the measured current pose before acknowledging 101.
                    current = self.robot.get_observation()
                    hold = {
                        key: float(current[key])
                        for key in (
                            "tcp.x", "tcp.y", "tcp.z",
                            *(f"tcp.r{i}" for i in range(1, 7)),
                        )
                    }
                    hold["gripper.pos"] = self._gripper_hold_target(current)
                    self.robot._send_action_sync(hold)
                    self._latest_raw_obs = current
                    self._latest_raw_obs_at = time.monotonic()
                    self._latest_tcp_y = float(current["tcp.y"])
                    self._last_target_tcp_x = float(current["tcp.x"])
                    self._last_target_tcp_y = float(current["tcp.y"])
                    self._last_target_tcp_z = float(current["tcp.z"])

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
                logger.warn(f"Camera {camera_name!r} is missing from the robot observation")
        return images

    def get_observation(self) -> dict[str, Any]:
        """Return an observation in the OpenPI/Xense runtime format."""
        raw_obs = self.robot.get_observation()
        self._latest_raw_obs = raw_obs
        self._latest_raw_obs_at = time.monotonic()
        self._latest_tcp_y = float(raw_obs["tcp.y"])
        if self.turntable_lease is not None:
            self.turntable_lease.note_position(
                self._latest_tcp_y, self._last_target_tcp_y
            )
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(raw_obs)
        obs["images"] = self.get_images(raw_obs)
        return obs

    def _gripper_hold_target(self, current: dict[str, Any]) -> float:
        # Measured finger opening is not a safe replacement for a commanded
        # grasp, especially while the gripper worker is still closing.
        target = getattr(self, "_last_gripper_target", None)
        if target is None:
            target = current.get("gripper.pos", 0.0)
        # A queued open may have been dropped by the async worker. If either
        # source still says closed, preserve the grasp rather than opening.
        observed = float(current.get("gripper.pos", target))
        return 1.0 if float(target) > 0.5 and observed > 0.5 else 0.0

    @property
    def policy_generation(self) -> int:
        return getattr(self, "_policy_generation", 0)

    @property
    def policy_blocked(self) -> bool:
        return getattr(self, "_policy_blocked", False)

    def _pause_policy_for_action(self, action: dict[str, float]) -> None:
        if not getattr(self, "_policy_blocked", False):
            self._policy_generation = getattr(self, "_policy_generation", 0) + 1
            self._policy_blocked = True
            # Discard any prefetched servo/gripper command before acknowledging
            # the pause. A queued open must not execute after the arm is held.
            if hasattr(self.robot, "_pause_action_worker"):
                try:
                    with self.robot._pause_action_worker():
                        current = self.robot.get_observation()
                        hold = {
                            key: float(current[key]) for key in (
                                "tcp.x", "tcp.y", "tcp.z",
                                *(f"tcp.r{i}" for i in range(1, 7)),
                            )
                        }
                        hold["gripper.pos"] = self._gripper_hold_target(current)
                        self.robot._send_action_sync(hold)
                        self._latest_raw_obs = current
                        self._latest_raw_obs_at = time.monotonic()
                        self._latest_tcp_y = float(current["tcp.y"])
                        self._last_target_tcp_x = float(current["tcp.x"])
                        self._last_target_tcp_y = float(current["tcp.y"])
                        self._last_target_tcp_z = float(current["tcp.z"])
                except Exception as exception:
                    logger.error(f"102 could not drain the pending action worker: {exception}")
            logger.warn(
                "102 VLA action stream paused by turntable protection; "
                "holding the last gripper command until a fresh plan is ready"
            )
        self._blocked_action = dict(action)

    def poll_policy_resume(self) -> bool:
        """Poll the lease without consuming another VLA action."""
        with self._guard_action_lock:
            if not self._policy_blocked:
                return True
            if self.turntable_lease.pick_guard_active:
                return False
            current = self._latest_raw_obs
            fresh = time.monotonic() - self._latest_raw_obs_at <= 0.10
            if current is None or not fresh:
                current = self.robot.get_observation()
                self._latest_raw_obs = current
                self._latest_raw_obs_at = time.monotonic()
                self._latest_tcp_y = float(current["tcp.y"])
            target = self._blocked_action
            if target is None:
                return False
            allowed = self.turntable_lease.require_for_action(
                float(current["tcp.y"]), target["tcp.y"],
                actual_x_m=current.get("tcp.x"),
                actual_z_m=current.get("tcp.z"),
                target_x_m=target["tcp.x"],
                target_z_m=target["tcp.z"],
                previous_target_x_m=self._last_target_tcp_x,
                previous_target_z_m=self._last_target_tcp_z,
                observation_fresh=True,
            )
            if not allowed:
                return False
            self._policy_blocked = False
            self._blocked_action = None
            self._policy_generation += 1
            logger.info("102 VLA action stream resumed; discarding the old action chunk")
            return True

    @staticmethod
    def get_reward() -> float:
        return 0.0

    def reset(self, *, fake: bool = False) -> dm_env.TimeStep:
        """Move to the configured reset target and return the first observation."""
        if not fake:
            # Runtime calls reset() after a graceful Ctrl+C as part of episode
            # cleanup. Do not reacquire the lease or command a home move then.
            if self.turntable_lease is not None and self.turntable_lease.stopping:
                return dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=self.get_reward(),
                    discount=None,
                    observation=self.get_observation(),
                )
            if self.turntable_lease is not None:
                self.turntable_lease.acquire()
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

    def send_action(self, action: np.ndarray, *, expected_generation: int | None = None) -> None:
        """Send a collection-compatible 10D Cartesian action to the robot."""
        guard_lock = getattr(self, "_guard_action_lock", None)
        if guard_lock is None:
            if expected_generation is None or expected_generation == self.policy_generation:
                self._send_action_guarded(action)
        else:
            with guard_lock:
                if expected_generation is None or expected_generation == self.policy_generation:
                    self._send_action_guarded(action)

    def _send_action_guarded(self, action: np.ndarray) -> None:
        action_dict = self._build_action_dict(action)
        requested_action = dict(action_dict)
        if self.turntable_lease is not None:
            if self._latest_tcp_y is None:
                self._latest_raw_obs = self.robot.get_observation()
                self._latest_raw_obs_at = time.monotonic()
                self._latest_tcp_y = float(self._latest_raw_obs["tcp.y"])
            current = self._latest_raw_obs
            if current is None:
                current = self.robot.get_observation()
                self._latest_raw_obs = current
                self._latest_raw_obs_at = time.monotonic()
            current_y = float(current["tcp.y"])

            observation_fresh = (
                time.monotonic() - getattr(self, "_latest_raw_obs_at", 0.0)
                <= 0.10
            )
            if not observation_fresh and self.turntable_lease.pick_guard_active:
                current = self.robot.get_observation()
                self._latest_raw_obs = current
                self._latest_raw_obs_at = time.monotonic()
                self._latest_tcp_y = current_y = float(current["tcp.y"])
                observation_fresh = True

            if self.turntable_lease.pick_guard_active:
                guard_y = self.turntable_lease.guard_max_y_m
                guard_target_y = guard_y - 0.005
                current_xyz = tuple(float(current.get(f"tcp.{axis}", float("nan")))
                                    for axis in "xyz")
                target_xyz = tuple(float(action_dict[f"tcp.{axis}"]) for axis in "xyz")
                prior_xyz = (
                    float(getattr(self, "_last_target_tcp_x", current_xyz[0])),
                    float(getattr(self, "_last_target_tcp_y", current_xyz[1])),
                    float(getattr(self, "_last_target_tcp_z", current_xyz[2])),
                )
                segment_clear = (
                    observation_fresh
                    and self.turntable_lease.collision_segment_clear(current_xyz, target_xyz)
                    and self.turntable_lease.collision_segment_clear(prior_xyz, target_xyz)
                )
                if current_y > guard_y and not segment_clear:
                    action_dict = {
                        key: float(current[key])
                        for key in (
                            "tcp.x", "tcp.y", "tcp.z",
                            *(f"tcp.r{i}" for i in range(1, 7)),
                        )
                    }
                    action_dict["tcp.y"] = max(
                        guard_target_y,
                        current_y - 0.010,
                    )
                    action_dict["gripper.pos"] = self._gripper_hold_target(current)
                    guard_state = "retreat"
                else:
                    if not segment_clear:
                        action_dict["tcp.y"] = min(
                            float(action_dict["tcp.y"]), guard_target_y
                        )
                        guard_state = "y_clamp"
                        staged_xyz = tuple(
                            float(action_dict[f"tcp.{axis}"]) for axis in "xyz"
                        )
                        if not self.turntable_lease.collision_segment_clear(
                            prior_xyz, staged_xyz
                        ):
                            action_dict = {
                                key: float(current[key])
                                for key in (
                                    "tcp.x", "tcp.y", "tcp.z",
                                    *(f"tcp.r{i}" for i in range(1, 7)),
                                )
                            }
                            action_dict["tcp.y"] = min(current_y, guard_target_y)
                            action_dict["gripper.pos"] = self._gripper_hold_target(current)
                            guard_state = "staged_hold"
                    else:
                        guard_state = "volume_clear"
                if guard_state != getattr(self, "_pick_guard_log_state", "clear"):
                    logger.info(
                        f"101 pick/flip X/Y/Z guard: {guard_state}; "
                        f"102 phase={'post-place fetch' if self.turntable_lease.post_place_fetch else 'place/unknown'}; "
                        f"collision volume={self.turntable_lease.collision_volume_description}"
                    )
                    self._pick_guard_log_state = guard_state
            elif getattr(self, "_pick_guard_log_state", "clear") != "clear":
                logger.info(
                    "101 pick/flip guard released; normal VLA action execution resumed"
                )
                self._pick_guard_log_state = "clear"

            # Stage X onto the safe side while Y remains outside the pick
            # volume. This permits post-place fetching across Y=400 without
            # confusing that transit with a new deposit.
            if (
                self.turntable_lease.post_place_fetch
                and current_y <= self.turntable_lease.safe_y_m
                and action_dict["tcp.y"] > self.turntable_lease.safe_y_m
                and self.turntable_lease.fetch_x_point_clear(action_dict["tcp.x"])
            ):
                current_x = float(current.get("tcp.x", float("nan")))
                prior_x = float(getattr(self, "_last_target_tcp_x", float("-inf")))
                if not (
                    observation_fresh
                    and self.turntable_lease.fetch_x_point_clear(current_x)
                    and self.turntable_lease.fetch_x_point_clear(prior_x)
                ):
                    stage_y = (
                        self.turntable_lease.guard_max_y_m
                        if self.turntable_lease.pick_guard_active
                        else self.turntable_lease.safe_y_m
                    )
                    guard_target_y = stage_y - 0.005
                    if current_y > stage_y:
                        action_dict = {
                            key: float(current[key])
                            for key in (
                                "tcp.x", "tcp.y", "tcp.z",
                                *(f"tcp.r{i}" for i in range(1, 7)),
                            )
                        }
                        action_dict["tcp.y"] = max(
                            guard_target_y, current_y - 0.010
                        )
                        action_dict["gripper.pos"] = self._gripper_hold_target(current)
                    else:
                        # Change X outside the currently protected Y side.
                        action_dict["tcp.y"] = min(
                            action_dict["tcp.y"], guard_target_y
                        )

            allowed = self.turntable_lease.require_for_action(
                self._latest_tcp_y,
                action_dict["tcp.y"],
                actual_x_m=current.get("tcp.x"),
                actual_z_m=current.get("tcp.z"),
                target_x_m=action_dict["tcp.x"],
                target_z_m=action_dict["tcp.z"],
                previous_target_x_m=getattr(self, "_last_target_tcp_x", None),
                previous_target_z_m=getattr(self, "_last_target_tcp_z", None),
                observation_fresh=(
                    time.monotonic() - getattr(self, "_latest_raw_obs_at", 0.0)
                    <= 0.10
                ),
            )
            if not allowed:
                if current_y <= self.turntable_lease.safe_y_m - 0.010:
                    self._pause_policy_for_action(requested_action)
                    now = time.monotonic()
                    if now - getattr(self, "_last_lease_wait_log_at", 0.0) >= 1.0:
                        self._last_lease_wait_log_at = now
                        logger.warn(
                            "102 action held: waiting for 101 turntable lease grant; "
                            f"lease_status={self.turntable_lease.last_acquire_status}, "
                            f"post_place_fetch={self.turntable_lease.post_place_fetch}, "
                            f"guard_active={self.turntable_lease.pick_guard_active}, "
                            f"guard_age_ms={(now - self.turntable_lease._guard_last_at) * 1000.0:.0f}, "
                            f"actual XYZ=({float(current.get('tcp.x', float('nan'))):.3f},"
                            f"{current_y:.3f},{float(current.get('tcp.z', float('nan'))):.3f})m, "
                            f"target XYZ=({action_dict['tcp.x']:.3f},"
                            f"{action_dict['tcp.y']:.3f},{action_dict['tcp.z']:.3f})m, "
                            f"previous X={getattr(self, '_last_target_tcp_x', float('nan')):.3f}m"
                        )
                    return
                action_dict = {
                    key: float(current[key])
                    for key in ("tcp.x", "tcp.y", "tcp.z", *(f"tcp.r{i}" for i in range(1, 7)))
                }
                action_dict["tcp.y"] = max(
                    self.turntable_lease.safe_y_m - 0.015,
                    current_y - 0.010,
                )
                action_dict["gripper.pos"] = self._gripper_hold_target(current)
                logger.warn(
                    "101 turntable protection: 102 retreating 10 mm in User -Y "
                    "with current X/Z/attitude held"
                )
            arm_keys = ("tcp.x", "tcp.y", "tcp.z", *(f"tcp.r{i}" for i in range(1, 7)))
            intervened = any(action_dict[key] != requested_action[key] for key in arm_keys)
            if intervened or not allowed:
                action_dict["gripper.pos"] = self._gripper_hold_target(current)
                self._pause_policy_for_action(requested_action)
            self._last_target_tcp_y = action_dict["tcp.y"]
            self._last_target_tcp_x = action_dict["tcp.x"]
            self._last_target_tcp_z = action_dict["tcp.z"]
        start = time.perf_counter()
        try:
            self.robot.send_action(action_dict)
            if self.turntable_lease is None or not getattr(self, "_policy_blocked", False):
                self._last_gripper_target = 1.0 if action_dict["gripper.pos"] >= 0.5 else 0.0
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
            if (
                self.turntable_lease is not None
                and self.turntable_lease.held
                and self.robot.is_connected
            ):
                try:
                    actual_y_m = float(self.robot.get_observation()["tcp.y"])
                    self.turntable_lease.release_after_shutdown(actual_y_m)
                    logger.info("102 turntable lease released before disconnect")
                except Exception as exception:
                    logger.error(f"102 turntable lease could not be released: {exception}")
            # DobotNova5DH.disconnect() is idempotent. Calling it unconditionally
            # also cleans up the arm if a camera dropped and is_connected became false.
            self.robot.disconnect()
            logger.info("Dobot Nova5 DH disconnected")
        except Exception as exception:
            logger.warn(f"Error during Dobot Nova5 DH disconnect: {exception}")
        finally:
            if self.turntable_lease is not None:
                self.turntable_lease.close()

    def request_stop(self) -> None:
        if self.turntable_lease is not None:
            self.turntable_lease.request_stop()

    def return_home_after_shutdown(self) -> None:
        """Restore the V4 Ctrl+C home move under 101's V5 turntable lease."""
        if self.turntable_lease is None or not self.robot.is_connected:
            return
        logger.info("102 shutdown: stopping policy actions before requesting home")
        self.robot._stop_action_worker()
        current = self.robot.get_observation()
        hold = {
            key: float(current[key])
            for key in (
                "tcp.x", "tcp.y", "tcp.z",
                *(f"tcp.r{i}" for i in range(1, 7)),
            )
        }
        hold["gripper.pos"] = self._gripper_hold_target(current)
        self.robot._send_action_sync(hold)
        # With the policy worker stopped, a guard update needs no action drain.
        # Keep ROS callbacks and the heartbeat responsive during the lease wait
        # and the blocking joint move.
        self.turntable_lease.set_guard_transition_handler(None)
        logger.info("102 shutdown: waiting for 101 turntable lease to return home")
        self.turntable_lease.acquire_for_shutdown()
        logger.info(f"102 shutdown: returning to {self.config.reset_target.value}")
        self.robot.reset_to_initial_position()
        logger.info("102 shutdown home move completed")
