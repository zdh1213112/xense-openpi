"""Xense/OpenPI policy adapter for the shilin-vla websocket server."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np
from typing_extensions import override

from xense_client import base_policy as _base_policy
from xense_client import websocket_client_policy as _websocket_client_policy
from xense_client.logger import get_logger

from examples.bi_dobot_shilin.pose_adapter import (
    LINGBOT_POSE_DIM,
    ROBOT_POSE_DIM,
    lingbot_actions_to_robot_actions,
    robot_state_to_lingbot_state,
)


logger = get_logger("ShilinVLAClientPolicy")

_CAMERA_KEYS = {
    "head": "observation.images.head",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}


class ShilinVLAClientPolicy(_base_policy.BasePolicy):
    """Adapt the Dobot runtime schema to ``LingbotVLAv2Server``.

    Xense's stock ``WebsocketClientPolicy.reset`` is intentionally a no-op.
    This wrapper instead sends the reset control message understood by
    ``LingbotVLAv2Server.infer``. The actual ``FeatureTransform`` is therefore
    created on the GPU server, not in this robot-side process.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        task: str,
        robot_name: str = "own_robot",
        api_key: Optional[str] = None,
    ) -> None:
        if not task.strip():
            raise ValueError("task must be a non-empty instruction")
        self._task = task
        self._robot_name = robot_name
        self._policy = _websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
            api_key=api_key,
        )
        self._server_initialized = False
        # One websocket connection must not be used concurrently.
        self._rpc_lock = threading.Lock()

    def get_server_metadata(self) -> Dict:
        return self._policy.get_server_metadata()

    def _reset_server_locked(self) -> None:
        response = self._policy.infer(
            {
                "reset": True,
                "robo_name": self._robot_name,
            }
        )
        if response.get("action", None) is not None:
            raise RuntimeError(f"Unexpected LingBot reset response: {response.keys()}")
        self._server_initialized = True
        logger.info(
            f"LingBot server reset complete; robot config={self._robot_name!r}"
        )

    @override
    def reset(self) -> None:
        """Reset server-side chunk state and create its ``FeatureTransform``."""

        with self._rpc_lock:
            self._reset_server_locked()

    @staticmethod
    def _to_hwc_rgb(image: Any, *, camera_name: str) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(
                f"Camera {camera_name!r} must have 3 dimensions, got {image.shape}"
            )
        if image.shape[-1] == 3:
            hwc = image
        elif image.shape[0] == 3:
            hwc = np.transpose(image, (1, 2, 0))
        else:
            raise ValueError(
                f"Camera {camera_name!r} must be HWC or CHW RGB, got {image.shape}"
            )
        if not np.all(np.isfinite(hwc)) and np.issubdtype(hwc.dtype, np.floating):
            raise ValueError(f"Camera {camera_name!r} contains NaN or infinite values")
        return np.ascontiguousarray(hwc)

    def _prepare_server_observation(self, observation: Dict) -> Dict:
        if "state" not in observation:
            raise KeyError("Robot observation is missing 'state'")
        robot_state = np.asarray(observation["state"])
        if robot_state.shape != (ROBOT_POSE_DIM,):
            raise ValueError(
                f"Robot state must have shape ({ROBOT_POSE_DIM},), got {robot_state.shape}"
            )

        # Prefer unresized HWC frames. Fall back to the CHW images kept for
        # regular OpenPI policies when an environment does not expose images_raw.
        images = observation.get("images_raw")
        if images is None:
            images = observation.get("images")
        if not isinstance(images, dict):
            raise KeyError("Robot observation must contain an image dictionary")

        task = observation.get("task", observation.get("prompt", self._task))
        if isinstance(task, (list, tuple)) and len(task) == 1:
            task = task[0]
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Task must be a non-empty string, got {task!r}")

        server_observation: Dict[str, Any] = {
            "observation.state": robot_state_to_lingbot_state(robot_state),
            "task": task,
        }
        for source_key, destination_key in _CAMERA_KEYS.items():
            if source_key not in images:
                raise KeyError(
                    f"Required camera {source_key!r} is missing; available={tuple(images)}"
                )
            server_observation[destination_key] = self._to_hwc_rgb(
                images[source_key],
                camera_name=source_key,
            )
        return server_observation

    @override
    def infer(self, observation: Dict, **kwargs) -> Dict:
        """Send one observation and return a 20D absolute Cartesian action chunk."""

        if kwargs:
            # LingBot VLA does not implement OpenPI RTC prefix conditioning.
            # Ignoring these keeps the wire payload valid, but callers should
            # use ActionChunkBroker rather than RTCActionChunkBroker.
            logger.debug(f"Ignoring unsupported RTC kwargs: {tuple(kwargs)}")

        server_observation = self._prepare_server_observation(observation)
        with self._rpc_lock:
            # Direct users may call infer without the Runtime episode reset.
            if not self._server_initialized:
                self._reset_server_locked()
            response = self._policy.infer(server_observation)

        if "action" not in response:
            raise KeyError(
                f"LingBot response is missing 'action'; keys={tuple(response)}"
            )
        lingbot_actions = np.asarray(response["action"])
        if lingbot_actions.ndim != 2 or lingbot_actions.shape[-1] != LINGBOT_POSE_DIM:
            raise ValueError(
                "LingBot server must run with --chunk_ret true and return an "
                f"(N, {LINGBOT_POSE_DIM}) action chunk; got {lingbot_actions.shape}"
            )

        result: Dict[str, Any] = {
            "actions": lingbot_actions_to_robot_actions(lingbot_actions),
        }
        if "server_timing" in response:
            result["server_timing"] = response["server_timing"]
        return result
