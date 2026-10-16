"""Offline fault injection: normal-motion feedback gaps must not exit/replay."""

import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from examples.dobot_nova5_dh_v5.feedback_robot import FeedbackFrameBuffer, MyType, StaleFeedbackError
from examples.dobot_nova5_dh_v5.interlocked_policy import InterlockedPolicy
from examples.dobot_nova5_dh_v5.test_feedback_robot import packet, robot_without_hardware
from examples.dobot_nova5_dh_v5.test_turntable_lease import lease_without_ros, paused_environment


def recovery_environment(*, lease=True):
    environment = paused_environment(lease_without_ros() if lease else None, y=0.564)
    current = dict(environment._latest_raw_obs)
    robot = environment.robot
    robot.feedback_recovery_pending = False
    robot.is_connected = True
    robot.latch_feedback_fault = lambda: setattr(robot, "feedback_recovery_pending", True)
    robot.stop_for_feedback_recovery = MagicMock(return_value=10)
    robot.wait_for_stopped_feedback = MagicMock(return_value=True)
    robot.finish_feedback_recovery = MagicMock(
        side_effect=lambda _seq: setattr(robot, "feedback_recovery_pending", False)
    )
    robot.feedback_diagnostics = lambda: "feedback_age_ms=193"
    robot._send_action_sync = MagicMock()
    robot._pending_action = {"gripper.pos": 1.0, "tcp.y": 0.7}
    robot.events = []

    @contextmanager
    def drain(**kwargs):
        robot.events.append("drain")
        robot._pending_action = None
        yield

    robot._pause_action_worker = drain
    robot.get_observation.side_effect = [StaleFeedbackError("193ms"), current]
    return environment, robot, current


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("already_paused", [False, True])
def test_normal_motion_and_prior_protection_hold_recover_without_old_actions(held, already_paused):
    env, robot, current = recovery_environment()
    lease = env.turntable_lease
    lease._held = held
    lease._safe_since = time.monotonic() - 1
    env._policy_blocked = already_paused
    env._policy_hold_established = already_paused
    old_generation = env.policy_generation

    def wait(_timeout, *, after_sequence, cancel_requested):
        assert _timeout == 2.0 and after_sequence == 10
        assert not cancel_requested()
        assert env.policy_blocked and not env.poll_policy_resume()
        assert env._latest_raw_obs_at == 0.0
        assert lease._safe_since is None
        assert robot._pending_action is None
        # A stale action, including gripper-open, is discarded even if a new
        # packet arrived while an action thread was paused on the lock.
        env.send_action(np.zeros(10), expected_generation=old_generation)
        robot.send_action.assert_not_called()
        assert lease.held is held
        return True

    robot.wait_for_stopped_feedback.side_effect = wait
    obs = env.get_observation()

    assert obs["qpos"][1] == pytest.approx(current["tcp.y"])
    assert env.policy_blocked and env._policy_hold_established
    assert env.policy_generation == old_generation + 1
    assert env._blocked_action["gripper.pos"] == 0.0
    assert not env._feedback_recovery_hold
    robot.stop_for_feedback_recovery.assert_called_once()
    robot._send_action_sync.assert_not_called()  # Stop only; no stale-pose Servo
    assert env._last_gripper_target == 0.0


def test_action_refresh_gap_discards_the_failing_action_then_recovers_next_observation():
    env, robot, current = recovery_environment()
    env._latest_raw_obs_at = 0.0  # force refresh inside apply_action path
    old_generation = env.policy_generation

    assert env.send_action(np.ones(10), expected_generation=old_generation) is None
    assert env.policy_blocked
    robot.send_action.assert_not_called()
    robot.stop_for_feedback_recovery.assert_not_called()
    env.get_observation()
    env.send_action(np.ones(10), expected_generation=old_generation)
    robot.send_action.assert_not_called()
    robot.stop_for_feedback_recovery.assert_called_once()


def test_worker_only_gap_is_detected_even_when_observations_are_fresh_again():
    env, robot, current = recovery_environment()
    robot.feedback_recovery_pending = True
    robot.get_observation.side_effect = [current]
    assert env.policy_blocked
    env.get_observation()
    robot.stop_for_feedback_recovery.assert_called_once()
    assert env.policy_blocked


@pytest.mark.parametrize("stop_failed", [False, True])
def test_persistent_fault_remains_paused_until_cancel_and_never_returns_cached_state(stop_failed):
    env, robot, _ = recovery_environment()
    waits = []
    if stop_failed:
        robot.stop_for_feedback_recovery.side_effect = socket.timeout("Stop ACK missing")

        def cancel_wait(_timeout):
            assert robot.feedback_recovery_pending and env.policy_blocked
            waits.append(True)
            env.request_stop()

        env._stop_requested = SimpleNamespace(is_set=lambda: bool(waits), set=lambda: waits.append(True), wait=cancel_wait)
    else:
        def no_frames(_timeout, **kwargs):
            waits.append(True)
            if len(waits) == 2:
                env.request_stop()
            return False

        robot.wait_for_stopped_feedback.side_effect = no_frames
    with pytest.raises(KeyboardInterrupt, match="feedback recovery cancelled"):
        env.get_observation()
    assert env._latest_raw_obs_at == 0.0
    assert env._feedback_recovery_hold and env.policy_blocked
    robot.send_action.assert_not_called()
    robot.finish_feedback_recovery.assert_not_called()
    if stop_failed:
        robot.stop_for_feedback_recovery.assert_called_once()
        robot.wait_for_stopped_feedback.assert_not_called()
    else:
        assert robot.wait_for_stopped_feedback.call_count == 2


def test_recovered_feedback_does_not_bypass_active_101_guard():
    env, robot, _ = recovery_environment()
    env.turntable_lease._held = True
    env.turntable_lease._guard_max_y_m = 0.300
    env.get_observation()
    generation = env.policy_generation
    assert not env.poll_policy_resume()
    assert env.policy_generation == generation
    env.turntable_lease._guard_max_y_m = 0.400
    assert env.poll_policy_resume()
    assert env.policy_generation == generation + 1


def test_feedback_recovery_without_ros_also_requires_fresh_policy_plan():
    env, robot, _ = recovery_environment(lease=False)
    env.get_observation()
    latest = {"state": np.zeros(10)}
    state = SimpleNamespace(
        policy_blocked=True,
        poll_policy_resume=env.poll_policy_resume,
        get_observation=lambda: latest,
        policy_generation=env.policy_generation,
    )
    brokers = []

    def factory():
        broker = SimpleNamespace(stop=MagicMock(), replan=MagicMock(), infer=MagicMock(return_value={"actions": np.zeros(10)}))
        brokers.append(broker)
        return broker

    policy = InterlockedPolicy(factory, state)
    policy.infer({"state": np.ones(10)})
    brokers[0].stop.assert_called_once()
    brokers[0].infer.assert_not_called()
    brokers[1].replan.assert_called_once()
    brokers[1].infer.assert_called_once_with(latest)


def test_guard_ack_stays_pending_during_feedback_fault_without_losing_held_lease():
    lease = lease_without_ros()
    lease._guard_sequence = 10
    lease._held = True
    ready = [False]
    lease.set_guard_transition_handler(lambda install: install(), ready_check=lambda: ready[0])
    msg = SimpleNamespace(data=[11.0, 0.300, 0.535, 0.075, 2.0])
    lease._guard_limit_callback(msg)
    lease._guard_limit_callback(msg)
    assert lease.pick_guard_active and lease.held
    lease._guard_ack_publisher.publish.assert_not_called()
    ready[0] = True
    lease._guard_limit_callback(msg)
    lease._guard_ack_publisher.publish.assert_called_once()
    assert lease.held


def test_guard_installed_before_stale_observation_must_retry_before_ack():
    lease = lease_without_ros()
    calls = []

    def transition(install):
        install()
        calls.append(True)
        return len(calls) > 1

    lease.set_guard_transition_handler(transition)
    msg = SimpleNamespace(data=[1.0, 0.300, 0.535, 0.075, 2.0])
    lease._guard_limit_callback(msg)
    lease._guard_ack_publisher.publish.assert_not_called()
    lease._guard_limit_callback(msg)
    lease._guard_ack_publisher.publish.assert_called_once()
    assert len(calls) == 2


def test_fault_latch_blocks_queued_arm_and_gripper_even_after_feedback_returns():
    robot = robot_without_hardware()
    robot._v5_feedback_at = time.monotonic() - 0.193
    with pytest.raises(StaleFeedbackError):
        robot._send_action_sync({"gripper.pos": 1.0})
    robot._publish_feedback(FeedbackFrameBuffer().feed(packet(200)))
    assert robot.feedback_recovery_pending
    with pytest.raises(StaleFeedbackError, match="recovery pending"):
        robot.send_action({"gripper.pos": 1.0})
    assert robot._pending_async_action is None


def test_stop_uses_bounded_raw_dashboard_request_and_never_sdk_reconnect_loop():
    robot = robot_without_hardware()
    channel = MagicMock()
    channel.gettimeout.return_value = None
    channel.recv.side_effect = [b"0,{", b"},Stop();"]
    robot._robot = SimpleNamespace(socket_dobot=channel, _DobotApi__globalLock=threading.Lock())
    assert robot.stop_for_feedback_recovery() == 0
    assert robot.feedback_recovery_pending
    channel.sendall.assert_called_once_with(b"Stop()")
    assert channel.settimeout.call_args.args == (None,)


@pytest.mark.parametrize("reply", [b"-1,{},Stop();", b"0,{},ServoP();", b""])
def test_unconfirmed_stop_cannot_clear_feedback_fault(reply):
    robot = robot_without_hardware()
    channel = MagicMock()
    channel.recv.return_value = reply
    robot._robot = SimpleNamespace(socket_dobot=channel, _DobotApi__globalLock=threading.Lock())
    with pytest.raises((RuntimeError, ConnectionError)):
        robot.stop_for_feedback_recovery()
    assert robot.feedback_recovery_pending


def test_only_three_new_idle_frames_after_stop_can_release_fault_latch():
    robot = robot_without_hardware()
    robot.latch_feedback_fault()
    seq = robot._v5_feedback_sequence
    for stamp in (1, 2):
        robot._publish_feedback(FeedbackFrameBuffer().feed(packet(stamp)))
    with pytest.raises(StaleFeedbackError):
        robot.finish_feedback_recovery(seq)
    moving = np.frombuffer(packet(3), dtype=MyType).copy()
    moving["RobotMode"] = 7
    robot._publish_feedback(moving[0])
    with pytest.raises(StaleFeedbackError):
        robot.finish_feedback_recovery(seq)
    for stamp in (4, 5, 6):
        robot._publish_feedback(FeedbackFrameBuffer().feed(packet(stamp)))
    robot.finish_feedback_recovery(seq)
    assert not robot.feedback_recovery_pending


def test_worker_drain_timeout_is_bounded_and_keeps_fault_gate_closed():
    robot = robot_without_hardware()
    robot._action_worker_busy = True
    robot.latch_feedback_fault()
    with pytest.raises(TimeoutError, match="did not drain"):
        with robot._pause_action_worker(timeout_s=0.01):
            pytest.fail("entered before worker drained")
    assert robot._v5_pause_depth == 0
    assert robot.feedback_recovery_pending


def test_guard_callback_cannot_send_after_shutdown_begins():
    env, robot, _ = recovery_environment()
    env._closing = True
    installed = MagicMock()
    assert env._install_guard_after_worker_drain(installed) is False
    installed.assert_not_called()
    robot._send_action_sync.assert_not_called()


def test_runtime_discards_retained_action_from_before_fault_and_replans_once():
    from examples.dobot_nova5_dh_v5.env import DobotNova5DHEnvironment
    from xense_client.runtime.agents.policy_agent import PolicyAgent
    from xense_client.runtime.runtime import Runtime

    env, robot, current = recovery_environment()
    env.turntable_lease._held = True
    reads = []

    def read():
        reads.append(True)
        if len(reads) == 1:
            raise StaleFeedbackError("193ms")
        return current

    robot.get_observation.side_effect = read
    wrapper = DobotNova5DHEnvironment.__new__(DobotNova5DHEnvironment)
    wrapper._env = env
    wrapper._step_count = 0
    wrapper._render_height = wrapper._render_width = 224
    brokers = []
    fresh_action = np.array([0.506, 0.560, 0.210, *([0.] * 6), 0.])

    def factory():
        broker = SimpleNamespace(
            stop=MagicMock(), replan=MagicMock(),
            infer=MagicMock(return_value={"actions": fresh_action}),
        )
        brokers.append(broker)
        return broker

    policy = InterlockedPolicy(factory, wrapper)
    runtime = Runtime(wrapper, PolicyAgent(policy), [])
    # This reproduces an action retained while awaiting an asynchronous lease.
    runtime._pending_action = {"actions": np.ones(10), "_policy_generation": 0}
    runtime._step()
    assert runtime._pending_action is None
    robot.send_action.assert_not_called()
    runtime._step()
    assert len(brokers) == 2
    brokers[0].stop.assert_called_once()
    brokers[0].infer.assert_not_called()
    brokers[1].replan.assert_called_once()
    robot.send_action.assert_called_once()
    assert robot.send_action.call_args.args[0]["tcp.y"] == pytest.approx(0.560)
    assert robot.send_action.call_args.args[0]["gripper.pos"] == 0.0


def test_real_worker_cannot_restart_old_commands_when_new_feedback_arrives(monkeypatch):
    robot = robot_without_hardware()
    robot._is_connected = True
    robot.config.async_action_worker_frequency = 1000
    robot._action_worker_lock = robot._action_worker_cv
    robot._async_action_drop_count = 0
    robot._v5_feedback_at = time.monotonic() - 0.193
    robot._pending_async_action = {"gripper.pos": 1.0}
    skipped = threading.Event()
    robot.logger.warn.side_effect = lambda *_: skipped.set()
    send = MagicMock()
    monkeypatch.setattr(
        "examples.dobot_nova5_dh_v5.feedback_robot.DobotNova5DH._send_action_sync", send,
    )
    worker = threading.Thread(target=robot._action_worker_loop)
    worker.start()
    try:
        assert skipped.wait(1)
        assert robot.feedback_recovery_pending
        robot._publish_feedback(FeedbackFrameBuffer().feed(packet(200)))
        skipped.clear()
        with robot._action_worker_cv:
            robot._pending_async_action = {"gripper.pos": 1.0}
            robot._action_worker_cv.notify_all()
        assert skipped.wait(1)
        send.assert_not_called()
    finally:
        with robot._action_worker_cv:
            robot._action_worker_running = False
            robot._action_worker_cv.notify_all()
        worker.join(timeout=1)
    assert not worker.is_alive()


def test_idle_frames_must_all_be_fresh_and_later_than_stop_barrier():
    robot = robot_without_hardware()
    robot.latch_feedback_fault()
    for stamp in (1, 2, 3):
        robot._publish_feedback(FeedbackFrameBuffer().feed(packet(stamp)))
    with pytest.raises(StaleFeedbackError):
        robot.finish_feedback_recovery(1)  # first sample predates Stop
    samples = robot._v5_feedback_samples
    first = samples[0]
    samples[0] = (first[0], time.monotonic() - 0.101, *first[2:])
    with pytest.raises(StaleFeedbackError):
        robot.finish_feedback_recovery(0)
    assert robot.feedback_recovery_pending


def test_guard_callback_stale_observation_defers_ack_without_killing_ros_executor():
    env, robot, _ = recovery_environment()
    env.turntable_lease._guard_max_y_m = 0.300
    install = MagicMock()
    assert env._install_guard_after_worker_drain(install) is False
    assert env.policy_blocked and robot.feedback_recovery_pending
    robot.stop_for_feedback_recovery.assert_not_called()  # never wait in ROS callback


def test_feedback_fault_suppresses_unused_placement_prefetch():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._placement_ready_callback(SimpleNamespace(data=True))
    lease.set_guard_transition_handler(None, ready_check=lambda: False)
    assert not lease.poll_ready_placement()
    lease._client.call_async.assert_not_called()


def test_ros_close_failure_still_disconnects_robot():
    env, robot, _ = recovery_environment()
    robot._stop_action_worker = MagicMock()
    robot.disconnect = MagicMock()
    env.turntable_lease.close = MagicMock(side_effect=RuntimeError("ROS already stopped"))
    env.disconnect()
    robot.disconnect.assert_called_once()
    robot.send_action.assert_not_called()


def test_operator_pause_feedback_never_authorizes_automatic_resume():
    robot = robot_without_hardware()
    robot.latch_feedback_fault()
    for stamp in (1, 2, 3):
        data = np.frombuffer(packet(stamp), dtype=MyType).copy()
        data["PauseCmdFlag"] = 1
        robot._publish_feedback(data[0])
    with pytest.raises(StaleFeedbackError):
        robot.finish_feedback_recovery(0)
    assert robot.feedback_recovery_pending
