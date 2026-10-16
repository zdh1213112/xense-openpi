"""V5 robot adapter with framed feedback and explicit feedback freshness."""

import contextlib
from collections import deque
import socket
import threading
import time

import numpy as np

from lerobot.robots.dobot_nova5.TCP_IP_Python_V4.dobot_api import MyType
from lerobot.robots.dobot_nova5_dh.dobot_nova5_dh import DobotNova5DH
from lerobot.robots.dobot_nova5_dh.dh_gripper_integrated import DHGripperIntegrated


class StaleFeedbackError(RuntimeError):
    """No robot state is fresh enough to authorize an observation or command."""


class HoldingDHGripperIntegrated(DHGripperIntegrated):
    """Closing communication must not release a box during fault cleanup."""

    def disconnect(self):
        if not self._is_connected:
            return
        self._worker_running = False
        with self._state_lock:
            self._target_dirty = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
        with self._io_lock:
            if self._modbus is not None:
                self._modbus.close()
                self._modbus = None
        self._is_connected = False
        self._logger.info("V5 DH gripper disconnected with its last commanded position held")


class FeedbackFrameBuffer:
    """Reassemble TCP fragments, resynchronize, and return the newest frame."""

    def __init__(self):
        self.buffer = bytearray()
        self.frame_size = MyType.itemsize
        self.magic_offset = MyType.fields["TestValue"][1]
        self.magic = (0x123456789ABCDEF).to_bytes(8, "little")
        self.header = self.frame_size.to_bytes(2, "little")

    def feed(self, chunk):
        self.buffer.extend(chunk)
        latest = None
        while len(self.buffer) >= self.frame_size:
            valid_header = self.buffer[:2] == self.header
            valid_magic = (
                self.buffer[self.magic_offset:self.magic_offset + 8] == self.magic
            )
            if not (valid_header and valid_magic):
                next_header = self.buffer.find(self.header, 1)
                if next_header < 0:
                    # Retain a possible fragmented header at the tail.
                    del self.buffer[:-1]
                    break
                del self.buffer[:next_header]
                continue
            frame = bytes(self.buffer[:self.frame_size])
            del self.buffer[:self.frame_size]
            candidate = np.frombuffer(frame, dtype=MyType)[0]
            if np.all(np.isfinite(candidate["ToolVectorActual"])) and np.all(
                np.isfinite(candidate["QActual"])
            ):
                latest = candidate
        return latest


class FeedbackSafeDobotNova5DH(DobotNova5DH):
    """Use fresh 102 measurements instead of silently reusing a dead reader."""

    feedback_max_age_s = 0.10

    def __init__(self, config):
        super().__init__(config)
        if config.use_gripper:
            self._gripper = HoldingDHGripperIntegrated(config.dh_gripper, name="right")
        self._v5_feedback_stop = threading.Event()
        self._v5_feedback_at = 0.0
        self._v5_feedback_timestamp = None
        self._v5_feedback_error = "waiting for first packet"
        self._v5_pause_depth = 0
        self._v5_feedback_fault = threading.Event()
        self._v5_feedback_sequence = 0
        self._v5_feedback_samples = deque(maxlen=8)
        self._v5_rx_at = 0.0
        self._v5_rx_count = 0
        self._v5_duplicate_frames = 0
        self._v5_max_frame_gap_s = 0.0
        self._v5_reconnect_count = 0

    def connect(self, *args, **kwargs):
        if not self._is_connected:
            self._v5_feedback_stop.clear()
            self._v5_feedback_at = 0.0
            self._v5_feedback_timestamp = None
            self._v5_feedback_fault.clear()
            self._v5_feedback_sequence = 0
            self._v5_feedback_samples.clear()
        return super().connect(*args, **kwargs)

    def _publish_feedback(self, packet):
        timestamp = int(packet["TimeStamp"])
        with self._feed_lock:
            if timestamp == self._v5_feedback_timestamp:
                self._v5_duplicate_frames += 1
                return False
            state = self._feed_data
            state.MessageSize = int(packet["len"])
            state.RobotMode = int(packet["RobotMode"])
            state.DigitalInputs = int(packet["DigitalInputs"])
            state.DigitalOutputs = int(packet["DigitalOutputs"])
            state.User = int(packet["User"])
            state.Tool = int(packet["Tool"])
            state.robotCurrentCommandID = int(packet["CurrentCommandId"])
            state.tcpPose = packet["ToolVectorActual"].copy()
            state.qActual = packet["QActual"].copy()
            state.PauseCmdFlag = int(packet["PauseCmdFlag"])
            self._v5_feedback_timestamp = timestamp
            now = time.monotonic()
            if self._v5_feedback_at:
                self._v5_max_frame_gap_s = max(
                    self._v5_max_frame_gap_s, now - self._v5_feedback_at
                )
            self._v5_feedback_at = now
            self._v5_feedback_sequence += 1
            self._v5_feedback_samples.append(
                (self._v5_feedback_sequence, now, state.RobotMode, state.PauseCmdFlag)
            )
            self._v5_feedback_error = "none"
        return True

    def _reconnect_feedback(self):
        if self._v5_feedback_stop.is_set() or not self._is_connected:
            return False
        feed = self._feed
        if feed is None:
            return False
        with contextlib.suppress(Exception):
            feed.socket_dobot.close()
        replacement = socket.create_connection(
            (self.config.robot_ip, self.config.feedPortFour), timeout=1.0
        )
        replacement.settimeout(0.05)
        if self._v5_feedback_stop.is_set() or not self._is_connected:
            replacement.close()
            return False
        feed.socket_dobot = replacement
        self._v5_reconnect_count += 1
        self.logger.info("102 feedback socket reconnected; waiting for a fresh frame")
        return True

    def _feed_loop(self):
        if self._v5_feedback_stop.is_set() or not self._is_connected:
            return
        parser = FeedbackFrameBuffer()
        last_new_frame = time.monotonic()
        last_error_log = 0.0
        try:
            self._feed.socket_dobot.settimeout(0.05)
            while self._is_connected and not self._v5_feedback_stop.is_set():
                try:
                    try:
                        chunk = self._feed.socket_dobot.recv(144000)
                    except socket.timeout:
                        chunk = None
                    if chunk == b"":
                        raise ConnectionError("102 feedback socket closed by peer")
                    if chunk:
                        with self._feed_lock:
                            self._v5_rx_at = time.monotonic()
                            self._v5_rx_count += 1
                    packet = parser.feed(chunk) if chunk else None
                    if packet is not None and self._publish_feedback(packet):
                        last_new_frame = time.monotonic()
                    if time.monotonic() - last_new_frame > 0.50:
                        raise TimeoutError("no new complete 102 feedback frame for 500 ms")
                except Exception as exc:
                    self._v5_feedback_error = str(exc)
                    if not self._is_connected or self._v5_feedback_stop.is_set():
                        return
                    if time.monotonic() - last_error_log >= 1.0:
                        self.logger.error(f"102 feedback reader failed: {exc}")
                        last_error_log = time.monotonic()
                    try:
                        if not self._reconnect_feedback():
                            return
                        parser = FeedbackFrameBuffer()
                        last_new_frame = time.monotonic()
                    except Exception as reconnect_error:
                        self._v5_feedback_error = str(reconnect_error)
                        self._v5_feedback_stop.wait(0.10)
        finally:
            if self._is_connected and not self._v5_feedback_stop.is_set():
                self.logger.error("102 feedback reader exited unexpectedly")

    def feedback_diagnostics(self):
        with self._feed_lock:
            age = time.monotonic() - self._v5_feedback_at
            state = self._feed_data
            return (
                f"feedback_age_ms={age * 1000.0:.0f}, "
                f"reader_alive={self._feed_thread is not None and self._feed_thread.is_alive()}, "
                f"raw_timestamp={self._v5_feedback_timestamp}, "
                f"RobotMode={state.RobotMode}, "
                f"CurrentCommandId={state.robotCurrentCommandID}, "
                f"PauseCmdFlag={getattr(state, 'PauseCmdFlag', -1)}, "
                f"worker_alive={self._action_worker_thread is not None and self._action_worker_thread.is_alive()}, "
                f"worker_paused={self._action_worker_paused}, "
                f"worker_busy={self._action_worker_busy}, "
                f"sent={self._async_action_send_count}, "
                f"errors={self._async_action_error_count}, "
                f"reader_error={self._v5_feedback_error}"
                f", rx_age_ms={(time.monotonic() - self._v5_rx_at) * 1000:.0f}"
                f", rx_chunks={self._v5_rx_count}, fresh_frames={self._v5_feedback_sequence}"
                f", duplicate_frames={self._v5_duplicate_frames}"
                f", max_frame_gap_ms={max(age, self._v5_max_frame_gap_s) * 1000:.0f}"
                f", reconnects={self._v5_reconnect_count}"
            )

    @property
    def feedback_recovery_pending(self):
        return self._v5_feedback_fault.is_set()

    def latch_feedback_fault(self):
        # This latch survives a fresh packet. Only verified recovery clears it,
        # so queued actions cannot restart themselves after a brief gap.
        self._v5_feedback_fault.set()

    def _require_fresh_feedback(self, stage):
        with self._feed_lock:
            fresh = time.monotonic() - self._v5_feedback_at <= self.feedback_max_age_s
        if not fresh:
            self.latch_feedback_fault()
            raise StaleFeedbackError(
                f"102 stale feedback at {stage}; "
                f"motion/gripper commands blocked: {self.feedback_diagnostics()}"
            )

    def _require_feedback_command_ready(self, stage):
        self._require_fresh_feedback(stage)
        if self.feedback_recovery_pending:
            raise StaleFeedbackError(
                f"102 feedback recovery pending at {stage}; commands blocked"
            )

    def stop_for_feedback_recovery(self, timeout_s=0.5):
        """Send only Stop(), with a bounded reply, after the worker is drained.

        The SDK's Stop() retries send/reconnect indefinitely. Use the same
        Dashboard socket and lock without that retry loop; an uncertain Stop
        must never authorize automatic motion recovery.
        """
        self.latch_feedback_fault()
        dashboard = self._robot
        lock = dashboard._DobotApi__globalLock
        deadline = time.monotonic() + timeout_s
        if not lock.acquire(timeout=max(0.0, timeout_s)):
            raise TimeoutError("102 Dashboard busy; feedback recovery Stop unconfirmed")
        try:
            channel = dashboard.socket_dobot
            previous_timeout = channel.gettimeout()
            try:
                channel.settimeout(max(0.001, deadline - time.monotonic()))
                channel.sendall(b"Stop()")
                reply = bytearray()
                while b";" not in reply:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("102 feedback recovery Stop reply timed out")
                    channel.settimeout(remaining)
                    chunk = channel.recv(1024)
                    if not chunk:
                        raise ConnectionError("102 Dashboard closed during Stop")
                    reply.extend(chunk)
                    if len(reply) > 4096:
                        raise RuntimeError("102 feedback recovery Stop reply invalid")
                response = reply.decode("utf-8")
                code, _ = self._parse_dobot_response(response)
                if code != 0 or not response.strip().endswith("Stop();"):
                    raise RuntimeError(f"102 feedback recovery Stop rejected: {response!r}")
            finally:
                channel.settimeout(previous_timeout)
        finally:
            lock.release()
        with self._feed_lock:
            # Require independent idle packets published after the Stop reply.
            return self._v5_feedback_sequence

    def _feedback_ready_after_stop(self, after_sequence):
        samples = list(self._v5_feedback_samples)[-3:]
        return (
            len(samples) == 3
            and samples[0][0] > after_sequence
            and all(mode == 5 and paused == 0 for _, _, mode, paused in samples)
            and time.monotonic() - samples[0][1] <= self.feedback_max_age_s
        )

    def wait_for_stopped_feedback(self, timeout_s, *, after_sequence, cancel_requested):
        deadline = time.monotonic() + timeout_s
        while not (self._v5_feedback_stop.is_set() or cancel_requested()):
            if not self._is_connected:
                return False
            with self._feed_lock:
                if self._feedback_ready_after_stop(after_sequence):
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._v5_feedback_stop.wait(min(0.01, remaining))
        return False

    def finish_feedback_recovery(self, after_sequence):
        with self._feed_lock:
            if not self._feedback_ready_after_stop(after_sequence):
                raise StaleFeedbackError("102 stopped feedback is not yet stable/fresh")
            self._v5_feedback_fault.clear()

    def wait_for_fresh_feedback(self, timeout_s, *, cancel_requested):
        """Wait for the reader; never refresh timestamps or issue commands here.

        The caller must already hold the policy and drain its action worker.
        All ordinary observation/action/home checks retain the 100 ms limit.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            if (
                self._v5_feedback_stop.is_set()
                or not self._is_connected
                or cancel_requested()
            ):
                return False
            with self._feed_lock:
                if time.monotonic() - self._v5_feedback_at <= self.feedback_max_age_s:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            # Wait outside the feedback lock so the reader can publish, and
            # outside ROS callbacks so the interlock heartbeat stays alive.
            self._v5_feedback_stop.wait(min(0.01, remaining))

    def get_observation(self):
        self._require_fresh_feedback("observation")
        return super().get_observation()

    def _cartesian_feed_state(self):
        self._require_feedback_command_ready("Cartesian command")
        return super()._cartesian_feed_state()

    def _send_action_sync(self, action):
        self._require_feedback_command_ready("action execution")
        return super()._send_action_sync(action)

    def send_action(self, action):
        self._require_feedback_command_ready("action submission")
        return super().send_action(action)

    def _move_joint_movj(self, joint_degrees, description):
        self._require_feedback_command_ready(description)
        return super()._move_joint_movj(joint_degrees, description)

    def _wait_for_command_id(self, command_id, description, timeout_s=60.0):
        deadline = time.monotonic() + timeout_s
        next_log = 0.0
        while time.monotonic() < deadline:
            self._require_fresh_feedback(f"waiting for {description}")
            with self._feed_lock:
                mode = int(self._feed_data.RobotMode)
                current = int(self._feed_data.robotCurrentCommandID)
            if mode == 9:
                raise RuntimeError(
                    f"102 {description} failed: {self._dobot_error_detail()}"
                )
            if mode == 5 and current == command_id:
                return
            if time.monotonic() >= next_log:
                self.logger.info(
                    f"Waiting for right {description}: target={command_id}; "
                    f"{self.feedback_diagnostics()}"
                )
                next_log = time.monotonic() + 2.0
            time.sleep(0.05)
        raise TimeoutError(
            f"102 {description} did not complete command {command_id}: "
            f"{self.feedback_diagnostics()}"
        )

    def _action_worker_loop(self) -> None:
        minimum_period = 1.0 / max(
            float(self.config.async_action_worker_frequency), 1e-6
        )
        next_send = time.perf_counter()
        while True:
            with self._action_worker_cv:
                while self._action_worker_running and (
                    self._action_worker_paused
                    or self._pending_async_action is None
                ):
                    self._action_worker_cv.wait(timeout=0.1)
                if not self._action_worker_running:
                    break
                action = self._pending_async_action
                self._pending_async_action = None
                self._action_worker_busy = action is not None

            if action is None:
                continue
            now = time.perf_counter()
            if now < next_send:
                time.sleep(next_send - now)
            send_start = time.perf_counter()
            try:
                with self._action_worker_cv:
                    self._action_worker_busy = True
                sent_action = self._send_action_sync(action)
                elapsed_ms = (time.perf_counter() - send_start) * 1e3
                with self._action_worker_lock:
                    self._latest_sent_async_action = sent_action
                    self._async_action_send_count += 1
                    self._last_async_action_timing = {
                        "worker_send_ms": elapsed_ms,
                        "queue_drop_count": float(self._async_action_drop_count),
                        "send_count": float(self._async_action_send_count),
                        "error_count": float(self._async_action_error_count),
                    }
            except Exception as exception:
                with self._action_worker_lock:
                    self._async_action_error_count += 1
                    self._last_async_action_timing = {
                        "worker_error_count": float(self._async_action_error_count),
                        "queue_drop_count": float(self._async_action_drop_count),
                    }
                self.logger.warn(
                    f"Async action worker skipped a failed action: {exception}"
                )
            finally:
                with self._action_worker_cv:
                    self._action_worker_busy = False
                    self._action_worker_cv.notify_all()
            next_send = max(next_send + minimum_period, time.perf_counter())

    @contextlib.contextmanager
    def _pause_action_worker(self, *, timeout_s=None):
        # Overlapping reset/ROS/policy pauses must not leave previous_paused=True
        # installed forever when their context managers exit out of order.
        if not self.config.async_action_worker:
            yield
            return
        with self._action_worker_cv:
            self._v5_pause_depth += 1
            self._action_worker_paused = True
            self._pending_async_action = None
            self._action_worker_cv.notify_all()
        try:
            deadline = None if timeout_s is None else time.monotonic() + timeout_s
            with self._action_worker_cv:
                while self._action_worker_busy:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("102 action worker did not drain before feedback Stop")
                    self._action_worker_cv.wait(timeout=0.01)
            yield
        finally:
            with self._action_worker_cv:
                self._v5_pause_depth -= 1
                self._action_worker_paused = self._v5_pause_depth > 0
                self._action_worker_cv.notify_all()

    def disconnect(self):
        if not self._is_connected:
            return
        self._stop_action_worker()
        if self.config.go_to_home_on_disconnect and not self.feedback_recovery_pending:
            try:
                self._go_to_home()
            except Exception as exc:
                self.logger.warn(f"Failed to move right arm home before disconnect: {exc}")
        self._v5_feedback_stop.set()  # never reconnect during teardown
        try:
            try:
                with self._pause_action_worker(timeout_s=0.5):
                    self.stop_for_feedback_recovery()
            except Exception as exc:
                self.logger.warn(f"102 shutdown Stop unconfirmed: {exc}")
            gripper = getattr(self, "_gripper", None)
            if gripper is not None and self.config.use_gripper:
                with contextlib.suppress(Exception):
                    gripper.disconnect()
            for name in ("_robot", "_gripper_robot", "_feed"):
                api = getattr(self, name, None)
                if api is None:
                    continue
                channel = api.socket_dobot
                # Suppress SDK __del__ attempting a second socket shutdown.
                api.socket_dobot = 0
                if channel:
                    with contextlib.suppress(Exception):
                        channel.shutdown(socket.SHUT_RDWR)
                    with contextlib.suppress(Exception):
                        channel.close()
                setattr(self, name, None)
            for camera in self.cameras.values():
                if camera.is_connected:
                    with contextlib.suppress(Exception):
                        camera.disconnect()
        finally:
            self._is_connected = False
            self._gripper_connected = False
            if self._feed_thread is not None:
                self._feed_thread.join(timeout=1.0)
                self._feed_thread = None
            self.logger.info("Dobot Nova5 right arm disconnected.")
