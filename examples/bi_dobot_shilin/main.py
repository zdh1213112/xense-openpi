#!/usr/bin/env python
"""Run BiDobot Nova5 DH against a shilin-vla LingBot VLA server."""

from __future__ import annotations

from dataclasses import dataclass
import os
import signal
import threading
from typing import Optional

from lerobot.utils.robot_utils import get_logger
from typing_extensions import override
import tyro

from xense_client import action_chunk_broker
from xense_client import paced_broker as _paced_broker
from xense_client.runtime import decoupled_runtime as _decoupled_runtime
from xense_client.runtime import environment as _environment
from xense_client.runtime import runtime as _runtime
from xense_client.runtime.agents import policy_agent as _policy_agent

import examples.bi_dobot_nova5_dh.env as _env
from examples.bi_dobot_shilin.client_policy import ShilinVLAClientPolicy


logger = get_logger("BiDobotShilinMain")


class DryRunEnvironmentWrapper(_environment.Environment):
    """Log converted 20D commands without sending them to the robot."""

    def __init__(self, wrapped_env: _env.BiDobotNova5DHEnvironment):
        self._wrapped_env = wrapped_env
        self._step_count = 0

    @override
    def reset(self) -> None:
        self._wrapped_env.reset()
        self._step_count = 0

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        return self._wrapped_env.get_observation()

    @override
    def apply_action(self, action: dict) -> None:
        self._step_count += 1
        actions = action.get("actions")
        if actions is not None:
            logger.info(
                "DRY RUN: converted action not sent to robot; "
                f"step={self._step_count}, shape={actions.shape}, action={actions}"
            )

    def disconnect(self) -> None:
        self._wrapped_env.disconnect()


@dataclass
class Args:
    """Arguments for the shilin-vla BiDobot inference client."""

    # LingBot websocket server.
    host: str = "localhost"
    port: int = 8006
    api_key: Optional[str] = None
    robot_name: str = "own_robot"
    task: str = "do something"

    # Robot network/configuration.
    left_robot_ip: str = "192.168.111.101"
    right_robot_ip: str = "192.168.111.102"
    control_frequency: float = 100.0
    go_to_start: bool = False
    async_action_worker: bool = True
    async_action_worker_frequency: float = 30.0
    enable_cartesian_ik_guard: bool = True
    cartesian_ik_servoj: bool = True
    max_cartesian_step_m: float = 0.05
    enable_clip: bool = True
    enable_tactile_sensors: bool = False
    use_left_gripper: bool = True
    use_right_gripper: bool = True

    # Local rendering is retained for compatibility; the adapter sends
    # images_raw and lets the LingBot server perform its own resize.
    render_height: int = 224
    render_width: int = 224

    # Runtime/chunk settings.
    runtime_hz: float = 30.0
    num_episodes: int = 1
    max_episode_steps: int = 1_000_000
    dry_run: bool = False
    action_horizon: int = 50

    # Optional decoupled action execution.
    action_hz: float = 0.0
    paced_queue_size: int = 50


def main(args: Args) -> None:
    if args.action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    decoupled_mode = args.action_hz > 0

    shilin_policy = ShilinVLAClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        task=args.task,
        robot_name=args.robot_name,
    )
    logger.info(f"Server metadata: {shilin_policy.get_server_metadata()}")

    base_environment = _env.BiDobotNova5DHEnvironment(
        left_robot_ip=args.left_robot_ip,
        right_robot_ip=args.right_robot_ip,
        control_frequency=args.control_frequency,
        go_to_start=args.go_to_start,
        async_action_worker=args.async_action_worker,
        async_action_worker_frequency=args.async_action_worker_frequency,
        enable_cartesian_ik_guard=args.enable_cartesian_ik_guard,
        cartesian_ik_servoj=args.cartesian_ik_servoj,
        max_cartesian_step_m=args.max_cartesian_step_m,
        enable_clip=args.enable_clip,
        enable_tactile_sensors=args.enable_tactile_sensors,
        use_left_gripper=args.use_left_gripper,
        use_right_gripper=args.use_right_gripper,
        render_height=args.render_height,
        render_width=args.render_width,
        setup_robot=True,
    )

    if args.dry_run:
        logger.info("DRY RUN mode: actions will be converted and logged only")
        environment: _environment.Environment = DryRunEnvironmentWrapper(
            base_environment
        )
    else:
        environment = base_environment

    policy = action_chunk_broker.ActionChunkBroker(
        policy=shilin_policy,
        action_horizon=args.action_horizon,
    )
    if decoupled_mode:
        policy = _paced_broker.PacedBroker(
            inner=policy,
            queue_size=args.paced_queue_size,
            target_hz=args.action_hz,
        )

    agent = _policy_agent.PolicyAgent(policy=policy)
    if decoupled_mode:
        runtime = _decoupled_runtime.DecoupledRuntime(
            environment=environment,
            broker=policy,
            subscribers=[],
            obs_hz=args.runtime_hz,
            action_hz=args.action_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
    else:
        runtime = _runtime.Runtime(
            environment=environment,
            agent=agent,
            subscribers=[],
            max_hz=args.runtime_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )

    def safe_disconnect() -> None:
        try:
            actual_env = environment
            if isinstance(actual_env, DryRunEnvironmentWrapper):
                actual_env = actual_env._wrapped_env
            actual_env.disconnect()
        except Exception as exc:
            logger.warning(f"Error disconnecting BiDobot Nova5 DH: {exc}")

    shutdown_in_progress = threading.Event()

    def signal_handler(sig, frame):
        del sig, frame
        if shutdown_in_progress.is_set():
            logger.warning(
                "Second Ctrl+C — forcing exit. Arms may not return home cleanly."
            )
            os._exit(1)
        shutdown_in_progress.set()
        logger.info(
            "Ctrl+C — stopping runtime gracefully "
            "(press Ctrl+C again to force exit)"
        )
        runtime.request_stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        runtime.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    except Exception as exc:
        logger.error(f"Runtime error: {exc}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        safe_disconnect()


if __name__ == "__main__":
    tyro.cli(main)
