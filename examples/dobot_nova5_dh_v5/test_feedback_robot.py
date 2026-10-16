from collections import deque
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from examples.dobot_nova5_dh_v5.feedback_robot import FeedbackFrameBuffer, FeedbackSafeDobotNova5DH, HoldingDHGripperIntegrated, MyType
from examples.dobot_nova5_dh_v5.feedback_robot import StaleFeedbackError


def packet(timestamp=1, command=42, xyz=(100.0, 200.0, 300.0)):
    data = np.zeros(1, dtype=MyType)
    data["len"] = MyType.itemsize
    data["TestValue"] = 0x123456789ABCDEF
    data["TimeStamp"] = timestamp
    data["RobotMode"] = 5
    data["CurrentCommandId"] = command
    data["ToolVectorActual"][0, :3] = xyz
    return data.tobytes()


def robot_without_hardware():
    robot = FeedbackSafeDobotNova5DH.__new__(FeedbackSafeDobotNova5DH)
    robot._feed_lock = threading.Lock()
    robot._feed_data = SimpleNamespace(RobotMode=5, robotCurrentCommandID=42)
    robot._feed_thread = None
    robot._v5_feedback_at = time.monotonic()
    robot._v5_feedback_timestamp = None
    robot._v5_feedback_error = "none"
    robot._v5_feedback_stop = threading.Event()
    robot._action_worker_thread = None
    robot._action_worker_paused = False
    robot._action_worker_busy = False
    robot._action_worker_running = True
    robot._action_worker_cv = threading.Condition()
    robot._pending_async_action = None
    robot._async_action_send_count = 0
    robot._async_action_error_count = 0
    robot._v5_pause_depth = 0
    robot._v5_feedback_fault = threading.Event()
    robot._v5_feedback_sequence = 0
    robot._v5_feedback_samples = deque(maxlen=8)
    robot._v5_rx_at = time.monotonic()
    robot._v5_rx_count = 0
    robot._v5_duplicate_frames = 0
    robot._v5_max_frame_gap_s = 0.0
    robot._v5_reconnect_count = 0
    robot.config = SimpleNamespace(async_action_worker=True)
    robot.logger = MagicMock()
    return robot


def test_feedback_fragmented_at_every_boundary_is_reassembled():
    raw = packet()
    for boundary in range(1, len(raw)):
        parser = FeedbackFrameBuffer()
        assert parser.feed(raw[:boundary]) is None
        decoded = parser.feed(raw[boundary:])
        assert int(decoded["CurrentCommandId"]) == 42


def test_merged_frames_use_latest_and_preserve_partial_next_frame():
    parser = FeedbackFrameBuffer()
    raw = packet(3, 103)
    decoded = parser.feed(packet(1, 101) + packet(2, 102) + raw[:613])
    assert int(decoded["CurrentCommandId"]) == 102
    decoded = parser.feed(raw[613:])
    assert int(decoded["CurrentCommandId"]) == 103


def test_garbage_and_bad_magic_resynchronize_without_losing_next_frame():
    parser = FeedbackFrameBuffer()
    invalid = bytearray(packet())
    invalid[MyType.fields["TestValue"][1]] ^= 1
    decoded = parser.feed(bytes(invalid) + b"garbage" + packet(2, 200))
    assert int(decoded["CurrentCommandId"]) == 200


def test_nonfinite_pose_is_not_published_as_fresh_feedback():
    parser = FeedbackFrameBuffer()
    assert parser.feed(packet(xyz=(np.nan, 0.0, 0.0))) is None


def test_duplicate_controller_timestamp_does_not_refresh_pose_age():
    robot = robot_without_hardware()
    decoded = FeedbackFrameBuffer().feed(packet())
    assert robot._publish_feedback(decoded)
    robot._v5_feedback_at = 123.0
    assert not robot._publish_feedback(decoded)
    assert robot._v5_feedback_at == 123.0


def test_stale_feedback_blocks_observation_action_and_home_before_any_send():
    robot = robot_without_hardware()
    robot._v5_feedback_at = time.monotonic() - 1.0
    for callback in (
        robot.get_observation,
        lambda: robot._send_action_sync({"gripper.pos": 1.0}),
        lambda: robot._move_joint_movj([0.0] * 6, "home"),
    ):
        with pytest.raises(RuntimeError, match="102 stale feedback"):
            callback()


def test_home_wait_reports_stale_command_id_instead_of_waiting_60_seconds():
    robot = robot_without_hardware()
    robot._v5_feedback_at = time.monotonic() - 1.0
    with pytest.raises(RuntimeError, match="CurrentCommandId=42"):
        robot._wait_for_command_id(4702, "home")


def test_overlapping_pause_contexts_do_not_leave_worker_permanently_paused():
    robot = robot_without_hardware()
    first = robot._pause_action_worker()
    second = robot._pause_action_worker()
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    assert robot._action_worker_paused
    second.__exit__(None, None, None)
    assert not robot._action_worker_paused


def test_reader_reconnects_after_socket_error_and_publishes_new_feedback():
    robot = robot_without_hardware()
    robot._is_connected = True
    robot._feed = SimpleNamespace(socket_dobot=MagicMock())
    robot._feed.socket_dobot.recv.side_effect = ConnectionResetError("reset")
    replacement = MagicMock()

    def recv(_size):
        robot._is_connected = False
        return packet(100, 400)

    replacement.recv.side_effect = recv

    def reconnect():
        robot._feed.socket_dobot = replacement
        return True

    robot._reconnect_feedback = MagicMock(side_effect=reconnect)
    robot._feed_loop()
    robot._reconnect_feedback.assert_called_once_with()
    assert robot._feed_data.robotCurrentCommandID == 400
    assert robot._v5_feedback_error == "none"
    robot.logger.error.assert_called()


def test_guard_pause_waits_for_action_already_taken_during_send_pacing(monkeypatch):
    robot = robot_without_hardware()
    robot.config.async_action_worker_frequency = 30.0
    robot._action_worker_lock = robot._action_worker_cv
    robot._async_action_drop_count = 0
    robot._pending_async_action = {"number": 1}
    pacing = threading.Event()
    release = threading.Event()
    pause_entered = threading.Event()

    def send(action):
        with robot._action_worker_cv:
            if action["number"] == 1:
                robot._pending_async_action = {"number": 2}
            else:
                robot._action_worker_running = False
        return action

    def sleep(_delay):
        assert robot._action_worker_busy
        pacing.set()
        assert release.wait(2.0)

    def pause():
        with robot._pause_action_worker():
            pause_entered.set()

    robot._send_action_sync = send
    monkeypatch.setattr("examples.dobot_nova5_dh_v5.feedback_robot.time.sleep", sleep)
    worker = threading.Thread(target=robot._action_worker_loop)
    pauser = threading.Thread(target=pause)
    try:
        worker.start()
        assert pacing.wait(2.0)
        pauser.start()
        assert not pause_entered.wait(0.05)
        release.set()
        assert pause_entered.wait(2.0)
    finally:
        release.set()
        worker.join(timeout=2.0)
        if pauser.ident is not None:
            pauser.join(timeout=2.0)
    assert not worker.is_alive()
    assert not pauser.is_alive()
    assert not robot._action_worker_paused


def test_fault_disconnect_closes_modbus_without_opening_gripper():
    gripper = HoldingDHGripperIntegrated.__new__(HoldingDHGripperIntegrated)
    gripper._is_connected = True
    gripper._worker_running = True
    gripper._worker_thread = None
    gripper._target_dirty = True
    gripper._state_lock = threading.Lock()
    gripper._io_lock = threading.Lock()
    modbus = MagicMock()
    gripper._modbus = modbus
    gripper._logger = MagicMock()
    gripper.disconnect()
    modbus.close.assert_called_once_with()
    modbus.write_register.assert_not_called()
    assert not gripper._target_dirty
    assert not gripper._is_connected


def test_held_feedback_wait_recovers_only_when_reader_publishes_new_packet():
    robot = robot_without_hardware()
    robot._is_connected = True
    robot._v5_feedback_at = time.monotonic() - 0.106
    waiting = threading.Event()
    failures = []

    def reader():
        try:
            assert waiting.wait(1.0)
            # Recovery waiting never weakens the command freshness check.
            with pytest.raises(StaleFeedbackError):
                robot._send_action_sync({"gripper.pos": 1.0})
            robot._publish_feedback(FeedbackFrameBuffer().feed(packet(200, 500)))
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        waiting.set()
        assert robot.wait_for_fresh_feedback(1.0, cancel_requested=lambda: False)
    finally:
        thread.join(1.0)
    assert not thread.is_alive()
    assert not failures
    assert robot._feed_data.robotCurrentCommandID == 500


def test_held_feedback_wait_expires_without_reusing_duplicate_packets(monkeypatch):
    robot = robot_without_hardware()
    robot._is_connected = True
    decoded = FeedbackFrameBuffer().feed(packet(100))
    robot._publish_feedback(decoded)
    old = robot._v5_feedback_at = time.monotonic() - 1.0
    wait = robot._v5_feedback_stop.wait

    def duplicate_during_wait(delay):
        assert not robot._publish_feedback(decoded)
        return wait(delay)

    monkeypatch.setattr(robot._v5_feedback_stop, "wait", duplicate_during_wait)
    assert not robot.wait_for_fresh_feedback(0.03, cancel_requested=lambda: False)
    assert robot._v5_feedback_at == old
    with pytest.raises(StaleFeedbackError):
        robot.get_observation()


@pytest.mark.parametrize("reason", ["stop", "disconnected", "cancel"])
def test_held_feedback_wait_cancels_before_accepting_even_a_fresh_pose(reason):
    robot = robot_without_hardware()
    robot._is_connected = reason != "disconnected"
    if reason == "stop":
        robot._v5_feedback_stop.set()
    assert not robot.wait_for_fresh_feedback(1.0, cancel_requested=lambda: reason == "cancel")


@pytest.mark.parametrize("go_home", [False, True])
def test_disconnect_stops_callbacks_and_reader_before_socket_cleanup_without_sdk_retries(go_home):
    robot = robot_without_hardware()
    robot._is_connected = True
    robot.config.go_to_home_on_disconnect = go_home
    robot.config.use_gripper = True
    robot.cameras = {}
    robot._stop_action_worker = MagicMock()
    robot._go_to_home = MagicMock(
        side_effect=lambda: pytest.fail("reader stopped before home")
        if robot._v5_feedback_stop.is_set() else None
    )
    robot.stop_for_feedback_recovery = MagicMock(return_value=1)
    channels = [MagicMock() for _ in range(3)]
    apis = [SimpleNamespace(socket_dobot=ch) for ch in channels]
    robot._robot, robot._gripper_robot, robot._feed = apis
    robot._gripper = MagicMock()
    robot._gripper.disconnect.side_effect = lambda: (
        pytest.fail("reader not stopped before gripper cleanup")
        if not robot._v5_feedback_stop.is_set() else None
    )

    robot.disconnect()

    assert robot._go_to_home.call_count == int(go_home)
    robot.stop_for_feedback_recovery.assert_called_once()
    for api, channel in zip(apis, channels):
        channel.close.assert_called_once()
        assert api.socket_dobot == 0
    assert not robot._is_connected
    assert robot._robot is None
    assert robot.config.go_to_home_on_disconnect is go_home


def test_feedback_reader_does_not_reconnect_or_dereference_feed_during_shutdown(monkeypatch):
    robot = robot_without_hardware()
    robot._is_connected = True
    robot._feed = None
    robot._v5_feedback_stop.set()
    create = MagicMock()
    monkeypatch.setattr("examples.dobot_nova5_dh_v5.feedback_robot.socket.create_connection", create)
    robot._feed_loop()
    assert not robot._reconnect_feedback()
    create.assert_not_called()
    robot.logger.error.assert_not_called()
