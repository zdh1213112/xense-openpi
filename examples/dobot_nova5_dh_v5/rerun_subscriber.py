"""Optional, best-effort V5 telemetry to a Rerun 0.23 viewer on another computer.

Only sampled camera images are copied on the runtime thread. Resizing, JPEG
encoding, SDK imports, network probes and bounded flushes run on a daemon thread.
The pending queue holds one snapshot; a slow viewer loses frames, not VLA time.
"""

from collections import deque
import contextlib
from dataclasses import dataclass
from itertools import pairwise
import math
import queue
import socket
import threading
import time
from typing import Literal

import numpy as np
from xense_client.logger import get_logger
from xense_client.runtime.subscriber import Subscriber

logger = get_logger("V5RemoteRerun")

_LABELS = ("tcp/x", "tcp/y", "tcp/z", *(f"rotation/r{i}" for i in range(1, 7)), "gripper")


@dataclass
class _Snapshot:
    episode: int
    step: int
    elapsed: float
    state: np.ndarray
    images: dict[str, np.ndarray]
    action: np.ndarray | None
    holding: bool


class RemoteRerunSubscriber(Subscriber):
    """Observe the existing runtime without opening cameras or a local viewer."""

    def __init__(
        self,
        host: str = "192.168.111.183",
        port: int = 9876,
        hz: float = 5.0,
        jpeg_quality: int = 90,
        dry_run: bool = False,
        image_source: Literal["raw", "policy"] = "raw",
        image_max_width: int = 1280,
        trail_seconds: float = 5.0,
    ) -> None:
        if not host.strip() or "://" in host or "/" in host:
            raise ValueError("rerun_host must be an IP address or hostname")
        if not 1 <= port <= 65535:
            raise ValueError("rerun_port must be in [1, 65535]")
        if not math.isfinite(hz) or not 0 < hz <= 30:
            raise ValueError("rerun_hz must be in (0, 30]")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("rerun_jpeg_quality must be in [1, 100]")
        if image_source not in ("raw", "policy"):
            raise ValueError("rerun_image_source must be raw or policy")
        if not 64 <= image_max_width <= 4096:
            raise ValueError("rerun_image_max_width must be in [64, 4096]")
        if not math.isfinite(trail_seconds) or not 0 <= trail_seconds <= 60:
            raise ValueError("rerun_trail_seconds must be in [0, 60]")
        self.host = host
        self.port = port
        self.url = f"rerun+http://{host}:{port}/proxy"
        self._period = 1.0 / hz
        self._jpeg_quality = jpeg_quality
        self._dry_run = dry_run
        self._image_source = image_source
        self._image_max_width = image_max_width
        self._trail_seconds = trail_seconds
        trail_capacity = max(2, math.ceil(hz * trail_seconds) + 2)
        self._tcp_trail: deque[tuple[float, np.ndarray]] = deque(maxlen=trail_capacity)
        self._action_trail: deque[tuple[float, np.ndarray]] = deque(maxlen=trail_capacity)
        self._trail_episode: int | None = None
        self._trail_last_at: float | None = None
        self._pending: queue.Queue[_Snapshot] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._recording = None
        self._started_at = time.monotonic()
        self._last_sample_at = -math.inf
        self._last_warning_at = -math.inf
        self._next_probe_at = 0.0
        self._episode = 0
        self._step = 0

    def on_episode_start(self) -> None:
        self._episode += 1
        self._last_sample_at = -math.inf
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(target=self._worker, name="v5-rerun", daemon=True)
            self._thread.start()

    def on_step(self, observation: dict, action: dict) -> None:
        self._step += 1
        now = time.monotonic()
        if self._stop.is_set() or now - self._last_sample_at < self._period:
            return
        self._last_sample_at = now
        try:
            # Copy existing buffers only at the display rate. Raw camera frames
            # improve visualization without changing the model's 224x224 input.
            policy_images = observation.get("images", {})
            raw_images = observation.get("images_raw", {}) if self._image_source == "raw" else {}
            images = {}
            for name in ("head", "wrist"):
                image = raw_images.get(name)
                if image is None:
                    image = policy_images.get(name)
                if image is not None:
                    images[name] = np.asarray(image).copy()
            holding = bool(action.get("_interlock_hold", False))
            values = action.get("actions")
            snapshot = _Snapshot(
                episode=self._episode,
                step=self._step,
                elapsed=now - self._started_at,
                state=np.asarray(observation["state"]).reshape(-1).copy(),
                images=images,
                action=None if holding or values is None else np.asarray(values).reshape(-1).copy(),
                holding=holding,
            )
            if snapshot.state.size != 10 or (snapshot.action is not None and snapshot.action.size != 10):
                raise ValueError("V5 Rerun expects 10D state and action vectors")
            with contextlib.suppress(queue.Empty):
                self._pending.get_nowait()
            self._pending.put_nowait(snapshot)
        except Exception as exc:
            self._warn(exc)

    def on_episode_end(self) -> None:
        # Keep one recording across episodes. main() closes the worker even
        # when runtime.run() fails before on_episode_end can be called.
        pass

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _warn(self, error: Exception) -> None:
        now = time.monotonic()
        if now - self._last_warning_at >= 5.0:
            self._last_warning_at = now
            logger.warn(f"Remote Rerun frame skipped; VLA continues: {error}")

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    snapshot = self._pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._publish(snapshot)
                except Exception as exc:
                    self._warn(exc)
                    self._disconnect()
        finally:
            self._disconnect()

    def _disconnect(self) -> None:
        recording, self._recording = self._recording, None
        self._reset_trails()
        if recording is not None:
            try:
                recording.disconnect()
            except Exception as exc:
                self._warn(exc)

    def _open_recording(self):
        import rerun as rr  # noqa: PLC0415 -- SDK imports belong on the sender thread.
        import rerun.blueprint as rrb  # noqa: PLC0415

        if rr.__version__.split(".")[:2] != ["0", "23"]:
            raise RuntimeError("V5 remote viewer requires rerun-sdk==0.23.1 on both computers")
        recording = rr.RecordingStream("dobot_v5_vla")
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="cameras/head", name=f"Head / {self._image_source} image"),
                    rrb.Spatial2DView(origin="cameras/wrist", name=f"Wrist / {self._image_source} image"),
                ),
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="world", name="TCP / VLA target (m)"),
                    rrb.TimeSeriesView(
                        origin="/", contents=["state/tcp/**", "action/tcp/**"], name="TCP XYZ (m)"
                    ),
                ),
                rrb.TimeSeriesView(
                    origin="/", contents=["state/gripper", "action/gripper", "status/holding"],
                    name="Gripper (0 closed / 1 open) and hold",
                ),
                row_shares=[3, 2, 1],
            ),
            collapse_panels=True,
        )
        try:
            # A bounded flush on EVERY sampled frame also bounds the SDK's
            # queue when a viewer stalls between our connection probes.
            recording.connect_grpc(self.url, flush_timeout_sec=0.2, default_blueprint=blueprint)
            # Refresh image-source labels even when the remote viewer retains
            # the previous client's active blueprint.
            recording.send_blueprint(blueprint)
            recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            recording.log(
                "world/axes",
                rr.Arrows3D(
                    origins=[[0, 0, 0]] * 3,
                    vectors=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
                    colors=[[230, 70, 70], [70, 210, 100], [70, 140, 240]],
                    labels=["X (m)", "Y (m)", "Z (m)"],
                ),
                static=True,
            )
            recording.log(
                "info",
                rr.TextDocument(
                    "V5 VLA monitoring. XYZ: metres, robot User 0 / Tool 1. "
                    "Action values are policy targets before driver clipping/interlock; "
                    "they are not confirmation of robot motion. "
                    "Gripper state uses the existing observation (command cache by default). "
                    f"Dry run: {self._dry_run}. No force measurement is available."
                ),
                static=True,
            )
        except Exception:
            recording.disconnect()
            raise
        logger.info(f"Remote Rerun connected to {self.url}; local viewer is not started")
        return recording

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        height, width = image.shape[:2]
        if width > self._image_max_width:
            import cv2  # noqa: PLC0415 -- Resizing only runs on the sender thread.

            height = max(1, round(height * self._image_max_width / width))
            image = cv2.resize(image, (self._image_max_width, height), interpolation=cv2.INTER_AREA)
        return image

    def _reset_trails(self) -> None:
        self._tcp_trail.clear()
        self._action_trail.clear()
        self._trail_episode = None
        self._trail_last_at = None

    def _log_trails(self, recording, snapshot: _Snapshot) -> None:
        if self._trail_seconds == 0:
            return
        import rerun as rr  # noqa: PLC0415 -- SDK calls only run on the sender thread.

        # Never draw an invented connecting segment across episodes or long
        # telemetry gaps. A held/absent policy target breaks its own trail.
        if (
            snapshot.episode != self._trail_episode
            or (self._trail_last_at is not None
                and snapshot.elapsed - self._trail_last_at > max(1.0, 3 * self._period))
        ):
            self._reset_trails()
        self._trail_episode = snapshot.episode
        self._trail_last_at = snapshot.elapsed
        cutoff = snapshot.elapsed - self._trail_seconds
        for path, trail, position, color in (
            ("world/tcp_trail", self._tcp_trail, snapshot.state[:3], [80, 200, 255]),
            ("world/vla_target_trail", self._action_trail,
             None if snapshot.action is None else snapshot.action[:3], [255, 170, 60]),
        ):
            if position is None or not np.all(np.isfinite(position)):
                trail.clear()
            else:
                while trail and trail[0][0] < cutoff:
                    trail.popleft()
                trail.append((snapshot.elapsed, position.copy()))
            if len(trail) < 2:
                recording.log(path, rr.Clear(recursive=True))
                continue
            segments, colors = [], []
            for previous, current in pairwise(trail):
                segments.append([previous[1], current[1]])
                opacity = round(40 + 180 * (current[0] - cutoff) / self._trail_seconds)
                colors.append([*color, opacity])
            recording.log(path, rr.LineStrips3D(segments, colors=colors, radii=0.0015))

    def _publish(self, snapshot: _Snapshot) -> None:
        now = time.monotonic()
        if now >= self._next_probe_at:
            self._next_probe_at = now + 1.0
            # Do not create/log into an SDK stream while the receiver is
            # unreachable: reconnect attempts cannot build an offline backlog.
            with socket.create_connection((self.host, self.port), timeout=0.15):
                pass
        elif self._recording is None:
            return
        if self._recording is None:
            self._recording = self._open_recording()

        import rerun as rr  # noqa: PLC0415 -- SDK imports belong on the sender thread.

        recording = self._recording
        # A single custom timeline makes the viewer follow elapsed VLA time.
        # An episode timeline is constant within a run and hides scalar curves.
        recording.set_time("vla_time", duration=snapshot.elapsed)
        recording.log("status/step", rr.Scalars(snapshot.step))
        recording.log("status/episode", rr.Scalars(snapshot.episode))
        for name, value in snapshot.images.items():
            value = self._prepare_image(value)
            recording.log(f"cameras/{name}", rr.Image(value).compress(jpeg_quality=self._jpeg_quality))
        for label, value in zip(_LABELS, snapshot.state, strict=True):
            recording.log(f"state/{label}", rr.Scalars(float(value)))
        recording.log("world/tcp", rr.Points3D([snapshot.state[:3]], colors=[80, 200, 255], radii=0.008))
        recording.log("status/holding", rr.Scalars(int(snapshot.holding)))
        if snapshot.action is not None:
            for label, value in zip(_LABELS, snapshot.action, strict=True):
                recording.log(f"action/{label}", rr.Scalars(float(value)))
            recording.log(
                "world/vla_target", rr.Points3D([snapshot.action[:3]], colors=[255, 170, 60], radii=0.008)
            )
        else:
            recording.log("world/vla_target", rr.Clear(recursive=True))
        self._log_trails(recording, snapshot)
        # Only the worker waits, at most the configured SDK flush timeout if
        # the network makes no progress. The runtime keeps replacing its slot.
        recording.flush(blocking=True)
