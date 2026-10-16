import threading
import time
from contextlib import nullcontext
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from examples.dobot_nova5_dh_v5.real_env import DobotNova5DHRealEnv
from examples.dobot_nova5_dh_v5.feedback_robot import StaleFeedbackError
from examples.dobot_nova5_dh_v5.interlocked_policy import InterlockedPolicy
from examples.dobot_nova5_dh_v5.turntable_lease import TurntableLease


def lease_without_ros():
    lease = TurntableLease.__new__(TurntableLease)
    lease.safe_y_m = 0.400
    lease._guard_max_y_m = 0.400
    lease._collision_x_lt_m = 0.535
    lease._collision_z_gt_m = 0.075
    lease._post_place_fetch = False
    lease._guard_sequence = -1
    lease._guard_last_at = time.monotonic()
    lease._guard_transition_handler = None
    lease._lease_lock = threading.RLock()
    lease._placement_ready = False
    lease._placement_ready_at = 0.0
    lease._node = MagicMock()
    lease._guard_ack_publisher = MagicMock()
    lease._held = False
    lease._release_armed = False
    lease._stopping = threading.Event()
    lease._safe_since = None
    lease._acquire_future = None
    lease._release_future = None
    lease._last_async_acquire_request = 0.0
    lease._client = MagicMock()
    lease._client.service_is_ready.return_value = True
    lease._heartbeat = MagicMock()
    lease._call = MagicMock(return_value=True)
    return lease


def test_shutdown_home_can_acquire_after_policy_stop_and_release_from_safe_pose():
    lease = lease_without_ros()
    lease.request_stop()

    lease.acquire_for_shutdown(timeout_s=1.0)
    assert lease.held
    lease._call.assert_called_once_with(True, ignore_stopping=True)

    lease.release_after_shutdown(0.390, timeout_s=1.0)
    assert not lease.held
    assert lease._call.call_args_list[-1].args == (False,)
    assert lease._call.call_args_list[-1].kwargs == {"ignore_stopping": True}


def test_shutdown_lease_stays_held_if_home_tcp_has_not_retreated():
    lease = lease_without_ros()
    lease._held = True

    with pytest.raises(RuntimeError, match="inside the turntable place side"):
        lease.release_after_shutdown(0.400)

    assert lease.held
    lease._call.assert_not_called()


def test_shutdown_home_stops_policy_worker_then_waits_for_lease():
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    order = []
    environment.robot = SimpleNamespace(
        is_connected=True,
        _stop_action_worker=lambda: order.append("worker_stop"),
        get_observation=lambda: {
            **{"tcp.x": 0.5, "tcp.y": 0.3, "tcp.z": 0.2, "gripper.pos": 0.5},
            **{f"tcp.r{i}": 0.0 for i in range(1, 7)},
        },
        _send_action_sync=lambda action: order.append(("hold", action["tcp.y"])),
        reset_to_initial_position=lambda: order.append("home"),
    )
    environment.turntable_lease = SimpleNamespace(
        set_guard_transition_handler=lambda handler: order.append("guard_unhook"),
        acquire_for_shutdown=lambda: order.append("lease_grant"),
    )
    environment.config = SimpleNamespace(reset_target=SimpleNamespace(value="home"))

    environment.return_home_after_shutdown()

    assert order == [
        "worker_stop", ("hold", 0.3), "guard_unhook", "lease_grant", "home",
    ]


def test_disconnect_releases_held_lease_from_measured_safe_home():
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    order = []
    environment.robot = SimpleNamespace(
        is_connected=True,
        get_observation=lambda: {"tcp.y": 0.300},
        disconnect=lambda: order.append("robot_disconnect"),
    )
    environment.turntable_lease = SimpleNamespace(
        held=True,
        release_after_shutdown=lambda y: order.append(("lease_release", y)),
        close=lambda: order.append("lease_close"),
    )

    environment.disconnect()

    assert order == [
        ("lease_release", 0.300), "robot_disconnect", "lease_close",
    ]


def test_entering_turntable_requires_101_grant():
    lease = lease_without_ros()
    future = Future()
    lease._client.call_async.return_value = future
    lease.require_for_action(0.397, 0.399)
    lease._client.call_async.assert_not_called()

    assert not lease.require_for_action(0.399, 0.401)
    lease._client.call_async.assert_called_once()
    future.set_result(SimpleNamespace(success=True))
    assert lease.require_for_action(0.399, 0.401)
    assert lease.held
    assert not lease._release_armed

    lease.note_position(0.401, 0.401)
    assert lease._release_armed


def test_target_only_crossing_does_not_consume_101_grant(monkeypatch):
    lease = lease_without_ros()
    lease._held = True

    # This reproduces the observed sequence: the policy briefly predicts a
    # target beyond 400 mm, but measured TCP feedback never enters that zone.
    lease.require_for_action(0.399, 0.411)
    assert not lease._release_armed

    monkeypatch.setattr("time.monotonic", lambda: 10.0)
    lease.note_position(0.399, 0.390)
    monkeypatch.setattr("time.monotonic", lambda: 10.31)
    lease.note_position(0.399, 0.390)

    assert lease.held
    assert not lease._release_armed
    assert all(call.args != (False,) for call in lease._call.call_args_list)


def test_missing_lease_blocks_policy_action_until_async_grant():
    lease = lease_without_ros()
    lease._client.service_is_ready.return_value = False
    assert not lease.require_for_action(0.410, 0.420)
    assert lease.last_acquire_status == "101 lease service unavailable"
    assert not lease.require_for_action(0.410, 0.350)


def test_denied_lease_reports_101_reason_without_blocking_runtime():
    lease = lease_without_ros()
    future = Future()
    lease._client.call_async.return_value = future
    assert not lease.require_for_action(0.390, 0.420)
    future.set_result(SimpleNamespace(success=False, message="101 shared zone unavailable"))
    assert not lease.require_for_action(0.390, 0.420)
    assert lease.last_acquire_status == "101 shared zone unavailable"


def test_placement_advertisement_acquires_before_the_400mm_action():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    grant = Future()
    lease._client.call_async.return_value = grant
    lease._placement_ready_callback(SimpleNamespace(data=True))
    assert not lease.poll_ready_placement()
    lease._client.call_async.assert_called_once()
    assert not lease.held  # the advertisement itself is never an entry grant

    grant.set_result(SimpleNamespace(success=True))
    # 101 stops advertising once it has granted this request. Still consume
    # the pending response and retain the lease for the next placement.
    lease._placement_ready_callback(SimpleNamespace(data=False))
    assert lease.poll_ready_placement()
    assert lease.held
    assert not lease._release_armed
    assert lease.require_for_action(0.299, 0.406)
    lease._client.call_async.assert_called_once()


@pytest.mark.parametrize("condition", ["closed", "stale_ready", "stale_guard", "pick_guard", "stopping"])
def test_placement_prefetch_requires_a_live_cleared_window(condition):
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._placement_ready_callback(SimpleNamespace(data=True))
    if condition == "closed":
        lease._placement_ready = False
    elif condition == "stale_ready":
        lease._placement_ready_at = time.monotonic() - 1.0
    elif condition == "stale_guard":
        lease._guard_last_at = time.monotonic() - 1.0
    elif condition == "pick_guard":
        lease._guard_max_y_m = 0.325
    else:
        lease.request_stop()
    assert not lease.poll_ready_placement()
    lease._client.call_async.assert_not_called()


def test_early_placement_lease_survives_safe_x_fetch_crossing(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._held = True
    lease._post_place_fetch = True
    lease.note_position(0.420, 0.430, actual_x_m=0.600)
    assert not lease._release_armed
    lease.note_position(0.390, 0.390, actual_x_m=0.600)
    clock[0] += 0.31
    lease.note_position(0.390, 0.390, actual_x_m=0.600)
    assert lease.held
    lease._client.call_async.assert_not_called()

    # The subsequent measured deposit on the placement side arms normal
    # release, preserving the one-grant-per-material handoff.
    lease.note_position(0.420, 0.430, actual_x_m=0.510)
    assert lease._release_armed
    lease.note_position(0.390, 0.390, actual_x_m=0.510)
    clock[0] += 0.31
    lease.note_position(0.390, 0.390, actual_x_m=0.510)
    assert lease._client.call_async.call_args.args[0].data is False


def test_stale_101_guard_falls_back_to_325mm_y_only():
    lease = lease_without_ros()
    lease._guard_max_y_m = 0.400
    lease._guard_last_at = time.monotonic() - 1.0

    assert lease.guard_max_y_m == pytest.approx(0.325)
    assert lease.pick_guard_active
    assert not lease.collision_segment_clear((0.600, 0.300, 0.100), (0.600, 0.340, 0.100))


def test_post_place_phase_alone_does_not_bypass_pick_guard_or_lease():
    lease = lease_without_ros()
    lease._guard_max_y_m = 0.325
    lease._fetch_phase_callback(SimpleNamespace(data=True))

    assert lease.post_place_fetch
    assert lease.pick_guard_active
    lease._client.service_is_ready.return_value = False
    assert not lease.require_for_action(0.399, 0.401)

    lease._fetch_phase_callback(SimpleNamespace(data=False))
    assert not lease.post_place_fetch


def test_only_fresh_whole_segment_on_safe_x_side_skips_lease():
    lease = lease_without_ros()
    lease._guard_sequence = 2
    lease._post_place_fetch = True
    lease._client.service_is_ready.return_value = False
    safe = dict(
        actual_x_m=0.550, actual_z_m=0.320,
        target_x_m=0.560, target_z_m=0.330,
        previous_target_x_m=0.550, previous_target_z_m=0.320,
        observation_fresh=True,
    )

    assert lease.require_for_action(0.390, 0.420, **safe)
    lease._client.call_async.assert_not_called()
    for changed in (
        {"actual_x_m": 0.540},
        {"target_x_m": 0.530},
        {"previous_target_x_m": 0.500},
        {"observation_fresh": False},
    ):
        assert not lease.require_for_action(0.390, 0.420, **(safe | changed))
    lease._post_place_fetch = False
    assert not lease.require_for_action(0.390, 0.420, **safe)


def test_collision_segment_check_is_conservative_for_diagonal_crossing():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._guard_max_y_m = 0.325
    assert lease.collision_segment_clear((0.560, 0.300, 0.100), (0.560, 0.390, 0.120))
    assert lease.collision_segment_clear((0.500, 0.300, 0.065), (0.500, 0.390, 0.065))
    assert lease.collision_segment_clear((0.500, 0.300, 0.100), (0.500, 0.315, 0.120))
    assert not lease.collision_segment_clear((0.550, 0.310, 0.100), (0.500, 0.390, 0.100))
    assert not lease.collision_segment_clear((0.500, 0.310, 0.060), (0.500, 0.390, 0.100))


def test_stable_retreat_releases_lease(monkeypatch):
    lease = lease_without_ros()
    lease._held = True
    lease._release_armed = True
    future = Future()
    lease._client.call_async.return_value = future
    monkeypatch.setattr("time.monotonic", lambda: 10.0)
    lease.note_position(0.390, 0.390)
    assert lease.held
    monkeypatch.setattr("time.monotonic", lambda: 10.31)
    lease.note_position(0.390, 0.390)
    assert lease.held
    lease._client.call_async.assert_called_once()
    future.set_result(SimpleNamespace(success=True))
    lease.note_position(0.390, 0.390)
    assert not lease.held


def test_initial_safe_prefix_does_not_release_reset_lease(monkeypatch):
    lease = lease_without_ros()
    lease._held = True
    monkeypatch.setattr("time.monotonic", lambda: 10.0)
    lease.note_position(0.398, 0.398)
    monkeypatch.setattr("time.monotonic", lambda: 10.31)
    lease.note_position(0.398, 0.398)
    assert lease.held
    assert not lease._release_armed
    lease._call.assert_not_called()


def test_revoked_policy_action_becomes_y_only_retreat():
    lease = lease_without_ros()
    lease._client.service_is_ready.return_value = False
    current = {"tcp.x": 0.4, "tcp.y": 0.4, "tcp.z": 0.3, "gripper.pos": 0.5}
    current.update({f"tcp.r{i}": float(i) for i in range(1, 7)})
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment.turntable_lease = lease
    environment._latest_tcp_y = 0.4
    environment._latest_raw_obs = current
    environment._last_target_tcp_y = 0.4
    environment.robot = SimpleNamespace(
        get_observation=MagicMock(return_value=current),
        send_action=MagicMock(),
    )
    policy_action = np.asarray([0.9, 0.5, 0.9, *([0.0] * 6), 1.0], dtype=np.float32)

    environment.send_action(policy_action)

    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.y"] == pytest.approx(0.39)
    assert all(sent[key] == value for key, value in current.items()
               if key not in ("tcp.y", "gripper.pos"))
    assert sent["gripper.pos"] == 0.0


def test_cached_observation_avoids_a_second_camera_read():
    lease = lease_without_ros()
    current = {"tcp.y": 0.398}
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment.turntable_lease = lease
    environment._latest_tcp_y = 0.398
    environment._latest_raw_obs = current
    environment._latest_raw_obs_at = time.monotonic()
    environment._last_target_tcp_y = 0.398
    environment.robot = SimpleNamespace(
        get_observation=MagicMock(),
        send_action=MagicMock(),
    )
    policy_action = np.asarray(
        [0.431, 0.398, 0.313, *([0.0] * 6), 1.0], dtype=np.float32
    )

    environment.send_action(policy_action)

    environment.robot.get_observation.assert_not_called()
    environment.robot.send_action.assert_called_once()


def test_101_pick_guard_retreats_y_only_without_stopping_policy_execution():
    lease = lease_without_ros()
    lease._guard_max_y_m = 0.325
    current = {"tcp.x": 0.4, "tcp.y": 0.39, "tcp.z": 0.3, "gripper.pos": 0.5}
    current.update({f"tcp.r{i}": float(i) for i in range(1, 7)})
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment.turntable_lease = lease
    environment._pick_guard_log_state = "clear"
    environment._latest_tcp_y = 0.39
    environment._latest_raw_obs = current
    environment._last_target_tcp_y = 0.39
    environment.robot = SimpleNamespace(
        get_observation=MagicMock(return_value=current),
        send_action=MagicMock(),
    )

    environment.send_action(
        np.asarray([0.9, 0.395, 0.9, *([0.0] * 6), 1.0], dtype=np.float32)
    )

    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.y"] == pytest.approx(0.38)
    assert all(sent[key] == value for key, value in current.items()
               if key not in ("tcp.y", "gripper.pos"))
    assert sent["gripper.pos"] == 0.0


def test_101_pick_guard_clamps_y_and_keeps_last_closed_gripper_target():
    lease = lease_without_ros()
    lease._guard_max_y_m = 0.325
    current = {"tcp.x": 0.400, "tcp.y": 0.280,
               "tcp.z": 0.100, "gripper.pos": 0.5}
    current.update({f"tcp.r{i}": float(i) for i in range(1, 7)})
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment.turntable_lease = lease
    environment._pick_guard_log_state = "inside"
    environment._latest_tcp_y = 0.280
    environment._latest_raw_obs = current
    environment._last_target_tcp_y = 0.280
    environment._last_gripper_target = 0.0
    environment.robot = SimpleNamespace(
        get_observation=MagicMock(return_value=current),
        send_action=MagicMock(),
    )
    policy_action = np.asarray(
        [0.431, 0.390, 0.313, *([0.1] * 6), 0.7], dtype=np.float32
    )

    environment.send_action(policy_action)

    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.y"] == pytest.approx(0.320)
    assert sent["tcp.x"] == pytest.approx(float(policy_action[0]))
    assert sent["tcp.z"] == pytest.approx(float(policy_action[2]))
    assert sent["gripper.pos"] == 0.0
    assert environment.policy_blocked


def test_lease_denial_pauses_policy_and_rejects_prefetched_actions():
    lease = lease_without_ros()
    lease._client.service_is_ready.return_value = False
    current = {"tcp.x": 0.488, "tcp.y": 0.364,
               "tcp.z": 0.244, "gripper.pos": 0.0}
    current.update({f"tcp.r{i}": 0.0 for i in range(1, 7)})
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment._guard_action_lock = threading.RLock()
    environment.turntable_lease = lease
    environment._latest_tcp_y = current["tcp.y"]
    environment._latest_raw_obs = current
    environment._latest_raw_obs_at = time.monotonic()
    environment._last_target_tcp_x = current["tcp.x"]
    environment._last_target_tcp_y = current["tcp.y"]
    environment._last_target_tcp_z = current["tcp.z"]
    environment._last_lease_wait_log_at = time.monotonic()
    environment._last_gripper_target = 0.0
    environment._policy_generation = 0
    environment._policy_blocked = False
    environment.robot = SimpleNamespace(send_action=MagicMock())

    blocked = np.asarray([0.493, 0.402, 0.237, *([0.0] * 6), 1.0])
    environment.send_action(blocked, expected_generation=0)
    environment.robot.send_action.assert_not_called()
    assert environment.policy_blocked
    assert environment.policy_generation == 1
    assert environment._last_gripper_target == 0.0

    # Actions prefetched before the hold cannot slip through while paused.
    environment.send_action(
        np.asarray([0.490, 0.360, 0.240, *([0.0] * 6), 1.0]),
        expected_generation=0,
    )
    environment.robot.send_action.assert_not_called()

    lease._held = True
    assert environment.poll_policy_resume()
    assert not environment.policy_blocked
    assert environment.policy_generation == 2
    environment.send_action(
        np.asarray([0.490, 0.360, 0.240, *([0.0] * 6), 1.0]),
        expected_generation=1,
    )
    environment.robot.send_action.assert_not_called()


def test_hold_never_reopens_when_queued_and_observed_gripper_disagree():
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment._last_gripper_target = 1.0
    assert environment._gripper_hold_target({"gripper.pos": 0.0}) == 0.0
    environment._last_gripper_target = 0.0
    assert environment._gripper_hold_target({"gripper.pos": 1.0}) == 0.0


def test_first_policy_hold_drains_worker_and_resends_closed_gripper():
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment._last_gripper_target = 0.0
    current = {"tcp.x": 0.488, "tcp.y": 0.364, "tcp.z": 0.244,
               "gripper.pos": 1.0}
    current.update({f"tcp.r{i}": 0.0 for i in range(1, 7)})
    environment.robot = SimpleNamespace(
        _pause_action_worker=lambda: nullcontext(),
        get_observation=MagicMock(return_value=current),
        _send_action_sync=MagicMock(),
    )

    environment._pause_policy_for_action({"tcp.y": 0.402})

    held = environment.robot._send_action_sync.call_args.args[0]
    assert held["tcp.y"] == pytest.approx(0.364)
    assert held["gripper.pos"] == 0.0
    assert environment.policy_blocked


def test_interlocked_policy_replans_from_fresh_observation_after_grant():
    class FakeBroker:
        def __init__(self):
            self.infer = MagicMock(return_value={"actions": np.zeros(10)})
            self.warmup = MagicMock()
            self.replan = MagicMock()
            self.stop = MagicMock()
            self.reset = MagicMock()

    state = SimpleNamespace(policy_blocked=True, policy_generation=1)
    state.poll_policy_resume = MagicMock(return_value=False)
    brokers = []

    def factory():
        broker = FakeBroker()
        brokers.append(broker)
        return broker

    policy = InterlockedPolicy(factory, state)
    first = {"state": "old"}
    assert policy.infer(first) == {"_interlock_hold": True}
    brokers[0].infer.assert_not_called()

    def grant():
        state.policy_blocked = False
        state.policy_generation = 2
        return True

    state.poll_policy_resume.side_effect = grant
    latest = {"state": np.zeros(10), "frame": "fresh"}
    state.get_observation = MagicMock(return_value=latest)
    result = policy.infer({"state": "before permission"})
    assert result["_policy_generation"] == 2
    assert len(brokers) == 2
    brokers[0].stop.assert_called_once()
    brokers[1].warmup.assert_not_called()
    brokers[1].replan.assert_called_once_with(latest, hold_action=latest["state"])
    brokers[1].infer.assert_called_once_with(latest)
    state.get_observation.assert_called_once()


def paused_environment(lease, *, x=0.506, y=0.312, z=0.210):
    current = {"tcp.x": x, "tcp.y": y, "tcp.z": z, "gripper.pos": 0.0}
    current.update({f"tcp.r{i}": 0.0 for i in range(1, 7)})
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment._guard_action_lock = threading.RLock()
    environment.turntable_lease = lease
    environment._latest_raw_obs = current
    environment._latest_raw_obs_at = time.monotonic()
    environment._latest_tcp_y = y
    environment._last_target_tcp_x = x
    environment._last_target_tcp_y = y
    environment._last_target_tcp_z = z
    environment._last_gripper_target = 0.0
    environment._policy_generation = 0
    environment._policy_blocked = False
    environment.robot = SimpleNamespace(
        get_observation=MagicMock(return_value=current),
        send_action=MagicMock(),
    )
    return environment


def test_stale_feedback_during_established_hold_waits_without_commands_or_losing_guard():
    lease = lease_without_ros()
    lease._guard_max_y_m = 0.325
    env = paused_environment(lease)
    env.robot._pause_action_worker = lambda: nullcontext()
    env.robot._send_action_sync = MagicMock()
    env.robot.feedback_diagnostics = lambda: "feedback_age_ms=106"
    env._pause_policy_for_action({"tcp.x": 0.5, "tcp.y": 0.42, "tcp.z": 0.2}, reason="pick_guard")
    assert env._policy_hold_established
    held_generation = env.policy_generation
    current = dict(env._latest_raw_obs)
    current["tcp.y"] = 0.310
    env.robot.get_observation.side_effect = [StaleFeedbackError("106ms"), current]
    holds_before = env.robot._send_action_sync.call_count

    def recover(timeout, *, cancel_requested):
        assert timeout == 2.0
        assert not cancel_requested()
        assert not env.poll_policy_resume()
        # Even if 101 clears its guard during the wait, an old cached pose
        # cannot authorize motion or opening the gripper.
        lease._guard_max_y_m = 0.400
        lease._held = True
        assert not env.poll_policy_resume()
        env.send_action(np.asarray([0.5, 0.42, 0.2, *([0.0] * 6), 1.0]),
                        expected_generation=held_generation)
        return True

    env.robot.wait_for_fresh_feedback = MagicMock(side_effect=recover)
    result = env.get_observation()
    assert result["qpos"][1] == pytest.approx(0.310)
    assert env.policy_blocked
    assert env.policy_generation == held_generation
    assert env._blocked_reason == "pick_guard"
    assert env._last_gripper_target == 0.0
    assert not env._feedback_recovery_hold
    env.robot.send_action.assert_not_called()
    assert env.robot._send_action_sync.call_count == holds_before
    assert env.poll_policy_resume()  # fresh pose AND permission are now available
    assert env.policy_generation == held_generation + 1


def test_persistent_feedback_fault_latches_hold_and_never_returns_cached_observation():
    lease = lease_without_ros()
    env = paused_environment(lease)
    env._policy_blocked = True
    env._policy_hold_established = True
    env.robot.get_observation.side_effect = StaleFeedbackError("106ms")
    env.robot.feedback_diagnostics = lambda: "feedback_age_ms=2106"
    env.robot._pause_action_worker = lambda: nullcontext()
    env.robot.wait_for_fresh_feedback = MagicMock(return_value=False)
    with pytest.raises(StaleFeedbackError, match="did not recover during protection hold"):
        env.get_observation()
    assert env._feedback_recovery_hold
    assert env._latest_raw_obs_at == 0.0
    lease._held = True
    assert not env.poll_policy_resume()
    env.send_action(np.zeros(10))
    env.robot.send_action.assert_not_called()


@pytest.mark.parametrize("state", ["moving", "hold_not_established", "stopping", "no_interlock"])
def test_feedback_wait_does_not_hide_faults_outside_established_interlock_hold(state):
    lease = lease_without_ros()
    env = paused_environment(lease)
    env._policy_blocked = state != "moving"
    env._policy_hold_established = state != "hold_not_established"
    if state == "stopping":
        lease.request_stop()
    if state == "no_interlock":
        env.turntable_lease = None
    env.robot.get_observation.side_effect = StaleFeedbackError("106ms")
    env.robot.wait_for_fresh_feedback = MagicMock()
    with pytest.raises(StaleFeedbackError, match="106ms"):
        env.get_observation()
    env.robot.wait_for_fresh_feedback.assert_not_called()


def test_held_feedback_wait_does_not_swallow_other_observation_failures():
    env = paused_environment(lease_without_ros())
    env._policy_blocked = True
    env._policy_hold_established = True
    env.robot.get_observation.side_effect = RuntimeError("camera failed")
    with pytest.raises(RuntimeError, match="camera failed"):
        env.get_observation()


def test_normal_pending_lease_retries_same_action_without_a_replan_hold(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    lease = lease_without_ros()
    lease._guard_sequence = 1
    grant = Future()
    lease._client.call_async.return_value = grant
    environment = paused_environment(lease, x=0.520, y=0.299, z=0.218)
    action = np.asarray([0.515, 0.406, 0.233, *([0.0] * 6), 1.0])

    assert environment.send_action(action, expected_generation=0) is False
    assert not environment.policy_blocked
    assert environment.policy_generation == 0
    environment.robot.send_action.assert_not_called()

    clock[0] += 0.032
    grant.set_result(SimpleNamespace(success=True))
    assert environment.send_action(action, expected_generation=0) is None
    assert not environment.policy_blocked
    assert environment.policy_generation == 0
    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.y"] == pytest.approx(0.406)
    assert sent["gripper.pos"] == 1.0
    assert environment._lease_wait_started_at is None


def test_runtime_retains_rtc_queue_during_a_normal_lease_round_trip():
    from examples.dobot_nova5_dh_v5.env import DobotNova5DHEnvironment
    from xense_client.rtc_action_chunk_broker import RTCActionChunkBroker
    from xense_client.runtime.runtime import Runtime
    from xense_client.runtime.agents.policy_agent import PolicyAgent

    lease = lease_without_ros()
    lease._guard_sequence = 1
    grant = Future()
    lease._client.call_async.return_value = grant
    environment = paused_environment(lease, x=0.520, y=0.299, z=0.218)
    wrapper = DobotNova5DHEnvironment.__new__(DobotNova5DHEnvironment)
    wrapper._env = environment
    wrapper._step_count = 0
    actions = np.repeat(
        np.asarray([[0.515, 0.406, 0.233, *([0.0] * 6), 0.0]], dtype=np.float32), 50, axis=0,
    )
    actions[:, 1] += np.arange(50, dtype=np.float32) * 0.001
    server = MagicMock()
    server.infer.return_value = {"actions": actions}
    policy = InterlockedPolicy(
        lambda: RTCActionChunkBroker(server, action_queue_size_to_get_new_actions=30), wrapper,
    )
    try:
        observation = wrapper.get_observation()
        policy._inner.replan(observation, hold_action=observation["state"])
        original_broker = policy._inner
        runtime = Runtime(wrapper, PolicyAgent(policy), [])
        runtime._step()
        assert runtime._pending_action is not None
        assert not environment.policy_blocked
        environment.robot.send_action.assert_not_called()

        grant.set_result(SimpleNamespace(success=True))
        runtime._step()
        runtime._step()
        assert runtime._pending_action is None
        assert policy._inner is original_broker
        assert environment.policy_generation == 0
        assert not environment.policy_blocked
        sent = environment.robot.send_action.call_args_list
        assert len(sent) == 2
        assert sent[0].args[0]["tcp.y"] == pytest.approx(actions[0, 1])
        assert sent[1].args[0]["tcp.y"] == pytest.approx(actions[1, 1])
        server.infer.assert_called_once()
    finally:
        policy._inner.stop()


@pytest.mark.parametrize("late_grant", [False, True])
def test_lease_wait_over_150ms_requires_fresh_plan(monkeypatch, late_grant):
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    lease = lease_without_ros()
    lease._guard_sequence = 1
    grant = Future()
    lease._client.call_async.return_value = grant
    environment = paused_environment(lease, y=0.299)
    action = np.asarray([0.515, 0.406, 0.233, *([0.0] * 6), 1.0])
    assert environment.send_action(action) is False
    clock[0] += 0.16
    if late_grant:
        grant.set_result(SimpleNamespace(success=True))
    assert environment.send_action(action) is None
    assert environment.policy_blocked
    assert environment.policy_generation == 1
    environment.robot.send_action.assert_not_called()


def test_explicit_lease_denial_does_not_use_the_short_wait_path():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    grant = Future()
    lease._client.call_async.return_value = grant
    environment = paused_environment(lease, y=0.299)
    action = np.asarray([0.515, 0.406, 0.233, *([0.0] * 6), 1.0])
    assert environment.send_action(action) is False
    grant.set_result(SimpleNamespace(success=False, message="101 shared zone unavailable"))
    environment.send_action(action)
    assert environment.policy_blocked
    assert environment.policy_generation == 1
    environment.robot.send_action.assert_not_called()


def test_background_retry_cannot_hide_a_denial_during_the_short_wait(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._placement_ready_callback(SimpleNamespace(data=True))
    first, retry = Future(), Future()
    lease._client.call_async.side_effect = [first, retry]
    environment = paused_environment(lease, y=0.299)
    action = np.asarray([0.515, 0.406, 0.233, *([0.0] * 6), 1.0])
    assert environment.send_action(action) is False
    first.set_result(SimpleNamespace(success=False, message="101 shared zone unavailable"))
    clock[0] += 0.11
    assert not lease.poll_ready_placement()
    assert lease.last_acquire_status == "grant request pending"
    retry.set_result(SimpleNamespace(success=True))
    assert environment.send_action(action) is None
    assert environment.policy_blocked
    assert environment._blocked_reason == "lease_denied"
    environment.robot.send_action.assert_not_called()


def test_pending_action_is_rechecked_when_pick_guard_reactivates():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._post_place_fetch = True
    lease._client.call_async.return_value = Future()
    environment = paused_environment(lease, y=0.299)
    action = np.asarray([0.515, 0.406, 0.233, *([0.0] * 6), 1.0])
    assert environment.send_action(action) is False
    lease._guard_max_y_m = 0.325
    environment.send_action(action)
    assert environment.policy_blocked
    assert environment._blocked_reason == "pick_guard"
    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.y"] <= 0.320
    assert sent["gripper.pos"] == 0.0


def test_pick_guard_resume_reserves_placement_even_when_old_target_is_below_400():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._guard_max_y_m = 0.325
    lease._post_place_fetch = True
    future = Future()
    lease._client.call_async.return_value = future
    environment = paused_environment(lease)

    # Reproduce the log: 323 mm is clipped to the 320 mm command boundary,
    # but would not by itself require a 400 mm placement lease.
    environment.send_action(np.asarray([0.506, 0.323, 0.210, *([0.0] * 6), 0.0]))
    assert environment.policy_blocked
    assert environment._blocked_reason == "pick_guard"
    assert environment.policy_generation == 1
    assert not environment.poll_policy_resume()
    lease._client.call_async.assert_not_called()

    lease._guard_max_y_m = 0.400
    assert not environment.poll_policy_resume()
    assert environment.policy_blocked
    assert not lease.held
    assert lease.last_acquire_status == "grant request pending"
    lease._client.call_async.assert_called_once()

    future.set_result(SimpleNamespace(success=True))
    assert environment.poll_policy_resume()
    assert lease.held
    assert not lease._release_armed
    assert environment.policy_generation == 2
    assert environment._refresh_after_resume
    assert not environment.policy_blocked

    # The newly generated first placement target no longer pauses for a grant.
    environment.send_action(np.asarray([0.500, 0.498, 0.216, *([0.0] * 6), 0.0]))
    assert not environment.policy_blocked
    assert environment.robot.send_action.call_args.args[0]["tcp.y"] == pytest.approx(0.498)
    assert not lease._release_armed  # predicted entry is not measured entry


def test_pick_guard_resume_stays_blocked_after_lease_denial():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._post_place_fetch = True
    future = Future()
    lease._client.call_async.return_value = future
    environment = paused_environment(lease)
    environment._pause_policy_for_action(
        {"tcp.x": 0.506, "tcp.y": 0.323, "tcp.z": 0.210}, reason="pick_guard",
    )
    assert not environment.poll_policy_resume()
    future.set_result(SimpleNamespace(success=False, message="101 shared zone unavailable"))
    assert not environment.poll_policy_resume()
    assert environment.policy_blocked
    assert not lease.held
    environment.robot.send_action.assert_not_called()


def test_safe_x_fetch_resume_does_not_reserve_placement_lease():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._post_place_fetch = True
    lease._client.service_is_ready.return_value = False
    environment = paused_environment(lease, x=0.550, y=0.315)
    environment._pause_policy_for_action(
        {"tcp.x": 0.560, "tcp.y": 0.420, "tcp.z": 0.210}, reason="pick_guard",
    )
    assert environment.poll_policy_resume()
    assert not lease.held
    lease._client.call_async.assert_not_called()


def test_resume_refreshes_feedback_and_rechecks_a_reactivated_guard():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._guard_max_y_m = 0.325
    environment = paused_environment(lease, x=0.600, y=0.300)
    environment._refresh_after_resume = True
    fresh = dict(environment._latest_raw_obs, **{"tcp.x": 0.500, "tcp.y": 0.310})
    environment.robot.get_observation.return_value = fresh

    environment.send_action(np.asarray([0.600, 0.390, 0.210, *([0.0] * 6), 1.0]))

    environment.robot.get_observation.assert_called_once()
    assert environment._latest_raw_obs is fresh
    assert environment.robot.send_action.call_args.args[0]["tcp.y"] <= 0.320
    assert environment.robot.send_action.call_args.args[0]["gripper.pos"] == 0.0
    assert environment.policy_blocked


def test_expired_feedback_is_refreshed_even_after_guard_release():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._held = True
    environment = paused_environment(lease)
    environment._latest_raw_obs_at = time.monotonic() - 0.5
    fresh = dict(environment._latest_raw_obs, **{"tcp.y": 0.330})
    environment.robot.get_observation.return_value = fresh

    environment.send_action(np.asarray([0.500, 0.498, 0.216, *([0.0] * 6), 0.0]))

    environment.robot.get_observation.assert_called_once()
    assert environment._latest_tcp_y == pytest.approx(0.330)
    assert not environment.policy_blocked


def test_resume_keeps_generation_from_before_inference_if_interlock_changes():
    state = SimpleNamespace(policy_blocked=True, policy_generation=1)
    state.get_observation = MagicMock(return_value={"state": np.zeros(10)})

    def grant():
        state.policy_blocked = False
        state.policy_generation = 2
        return True

    state.poll_policy_resume = grant

    def changed_during_replan(*args, **kwargs):
        state.policy_generation = 3
        state.policy_blocked = True

    brokers = []

    def factory():
        broker = SimpleNamespace(
            stop=MagicMock(), reset=MagicMock(),
            replan=MagicMock(side_effect=changed_during_replan),
            infer=MagicMock(return_value={"actions": np.zeros(10)}),
        )
        brokers.append(broker)
        return broker

    policy = InterlockedPolicy(factory, state)
    result = policy.infer({"state": np.ones(10)})
    assert result["_policy_generation"] == 2
    assert state.policy_generation == 3
    assert state.policy_blocked
    brokers[0].stop.assert_called_once()


def test_reset_after_a_hold_replaces_the_stopped_broker():
    state = SimpleNamespace(policy_blocked=True, poll_policy_resume=lambda: False)
    brokers = []

    def factory():
        broker = SimpleNamespace(stop=MagicMock(), reset=MagicMock())
        brokers.append(broker)
        return broker

    policy = InterlockedPolicy(factory, state)
    assert policy.infer({}) == {"_interlock_hold": True}
    assert policy.infer({}) == {"_interlock_hold": True}
    brokers[0].stop.assert_called_once()
    policy.reset()
    assert len(brokers) == 2
    brokers[1].reset.assert_called_once()


def test_non_rtc_resume_uses_one_new_inference_without_warmup():
    from xense_client.action_chunk_broker import ActionChunkBroker

    latest = {"state": np.zeros(10), "frame": "after grant"}
    state = SimpleNamespace(policy_blocked=True, policy_generation=1)
    state.get_observation = MagicMock(return_value=latest)

    def grant():
        state.policy_blocked = False
        state.policy_generation = 2
        return True

    state.poll_policy_resume = grant
    server = MagicMock()
    server.infer.return_value = {"actions": np.ones((50, 10))}
    policy = InterlockedPolicy(lambda: ActionChunkBroker(server, action_horizon=20), state)
    result = policy.infer({"state": np.ones(10), "frame": "before grant"})
    server.infer.assert_called_once_with(latest)
    assert result["_policy_generation"] == 2
    np.testing.assert_array_equal(result["actions"], np.ones(10))


def test_rtc_placement_resume_does_not_replan_twice_after_323mm_hold():
    from examples.dobot_nova5_dh_v5.env import DobotNova5DHEnvironment
    from xense_client.rtc_action_chunk_broker import RTCActionChunkBroker

    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._guard_max_y_m = 0.325
    lease._post_place_fetch = True
    grant = Future()
    lease._client.call_async.return_value = grant
    environment = paused_environment(lease)
    held = np.asarray([0.506, 0.323, 0.210, *([0.0] * 6), 0.0], dtype=np.float32)
    environment.send_action(held)
    assert environment.policy_blocked

    wrapper = DobotNova5DHEnvironment.__new__(DobotNova5DHEnvironment)
    wrapper._env = environment
    wrapper._step_count = 0
    actions = np.repeat(held[None, :], 50, axis=0)
    actions[:, 1] += np.arange(50, dtype=np.float32) * 0.005
    server = MagicMock()
    server.infer.return_value = {"actions": actions}
    policy = InterlockedPolicy(
        lambda: RTCActionChunkBroker(server, action_queue_size_to_get_new_actions=30), wrapper,
    )
    try:
        lease._guard_max_y_m = 0.400
        assert policy.infer({}) == {"_interlock_hold": True}
        server.infer.assert_not_called()
        grant.set_result(SimpleNamespace(success=True))

        result = policy.infer({"state": np.ones(10)})
        wrapper.apply_action(result)

        assert lease.held
        assert not environment.policy_blocked
        assert environment.policy_generation == 2
        server.infer.assert_called_once()
        assert server.infer.call_args.kwargs["inference_delay"] == 0
        assert environment.robot.send_action.call_args.args[0]["tcp.y"] == pytest.approx(actions[0, 1])
        assert actions[0, 1] != actions[20, 1]
    finally:
        policy._inner.stop()


def test_guard_holds_before_changing_x_when_previous_servo_target_was_high_y():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._guard_max_y_m = 0.325
    current = {"tcp.x": 0.550, "tcp.y": 0.315,
               "tcp.z": 0.100, "gripper.pos": 0.5}
    current.update({f"tcp.r{i}": float(i) for i in range(1, 7)})
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment.turntable_lease = lease
    environment._latest_tcp_y = 0.315
    environment._latest_raw_obs = current
    environment._latest_raw_obs_at = time.monotonic()
    environment._last_target_tcp_x = 0.550
    environment._last_target_tcp_y = 0.390
    environment._last_target_tcp_z = 0.100
    environment.robot = SimpleNamespace(send_action=MagicMock())

    environment.send_action(np.asarray(
        [0.500, 0.390, 0.100, *([0.0] * 6), 0.5], dtype=np.float32,
    ))

    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.x"] == pytest.approx(0.550)
    assert sent["tcp.y"] == pytest.approx(0.315)
    assert sent["tcp.z"] == pytest.approx(0.100)


def test_release_in_flight_cannot_reenter_turntable():
    lease = lease_without_ros()
    lease._held = True
    lease._release_armed = True
    lease._safe_since = time.monotonic() - 1.0
    lease._client.call_async.return_value = Future()

    lease.note_position(0.390, 0.390)

    assert lease._release_future is not None
    assert not lease.require_for_action(0.390, 0.500)


def test_xyz_guard_ack_invalidates_old_lease_after_phase_gap():
    lease = lease_without_ros()
    lease._held = True
    lease._guard_sequence = 10
    transition = MagicMock(side_effect=lambda apply: apply())
    lease.set_guard_transition_handler(transition)

    lease._guard_limit_callback(
        SimpleNamespace(data=[17.0, 0.325, 0.535, 0.075, 2.0])
    )

    assert not lease.held
    assert lease.guard_max_y_m == pytest.approx(0.325)
    assert lease._collision_x_lt_m == pytest.approx(0.535)
    assert lease._collision_z_gt_m == pytest.approx(0.075)
    transition.assert_called_once()
    assert lease._guard_ack_publisher.publish.call_args.args[0].data == 17


def test_xyz_guard_retreats_before_changing_x_through_danger_volume():
    lease = lease_without_ros()
    lease._guard_max_y_m = 0.325
    lease._post_place_fetch = True
    lease._guard_sequence = 1
    current = {"tcp.x": 0.50, "tcp.y": 0.36, "tcp.z": 0.10, "gripper.pos": 0.5}
    current.update({f"tcp.r{i}": float(i) for i in range(1, 7)})
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment.turntable_lease = lease
    environment._latest_tcp_y = 0.36
    environment._latest_raw_obs = current
    environment._latest_raw_obs_at = time.monotonic()
    environment._last_target_tcp_y = 0.36
    environment._last_target_tcp_x = 0.50
    environment._last_target_tcp_z = 0.10
    environment.robot = SimpleNamespace(send_action=MagicMock())

    environment.send_action(
        np.asarray([0.56, 0.39, 0.10, *([0.0] * 6), 1.0], dtype=np.float32)
    )
    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.y"] == pytest.approx(0.35)
    assert sent["tcp.z"] == pytest.approx(0.10)
    assert sent["tcp.x"] == pytest.approx(0.50)

    current["tcp.y"] = 0.32
    environment._latest_tcp_y = 0.32
    environment.send_action(
        np.asarray([0.56, 0.39, 0.10, *([0.0] * 6), 1.0], dtype=np.float32)
    )
    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.y"] == pytest.approx(0.320)
    assert sent["tcp.x"] == pytest.approx(0.50)

    environment.send_action(
        np.asarray([0.56, 0.39, 0.10, *([0.0] * 6), 1.0], dtype=np.float32)
    )
    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.y"] == pytest.approx(0.320)
    assert sent["tcp.x"] == pytest.approx(0.56)

    current["tcp.x"] = 0.56
    environment._last_target_tcp_x = 0.56
    environment.send_action(
        np.asarray([0.56, 0.39, 0.10, *([0.0] * 6), 1.0], dtype=np.float32)
    )
    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.y"] == pytest.approx(0.39)


def test_post_place_fetch_crosses_400_only_on_safe_x_side():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._post_place_fetch = True
    lease._client.service_is_ready.return_value = False
    current = {"tcp.x": 0.550, "tcp.y": 0.390, "tcp.z": 0.320, "gripper.pos": 0.5}
    current.update({f"tcp.r{i}": float(i) for i in range(1, 7)})
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment.turntable_lease = lease
    environment._latest_tcp_y = 0.390
    environment._latest_raw_obs = current
    environment._latest_raw_obs_at = time.monotonic()
    environment._last_target_tcp_y = 0.390
    environment._last_target_tcp_x = 0.550
    environment._last_target_tcp_z = 0.320
    environment.robot = SimpleNamespace(send_action=MagicMock())

    environment.send_action(np.asarray(
        [0.560, 0.420, 0.330, *([0.0] * 6), 0.5], dtype=np.float32,
    ))
    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.y"] == pytest.approx(0.420)
    assert sent["tcp.z"] == pytest.approx(0.330)

    environment.robot.send_action.reset_mock()
    environment.send_action(np.asarray(
        [0.500, 0.420, 0.330, *([0.0] * 6), 0.5], dtype=np.float32,
    ))
    environment.robot.send_action.assert_not_called()


def test_post_place_fetch_stages_x_near_400_when_pick_guard_is_inactive():
    lease = lease_without_ros()
    lease._guard_sequence = 1
    lease._post_place_fetch = True
    current = {"tcp.x": 0.500, "tcp.y": 0.390,
               "tcp.z": 0.100, "gripper.pos": 0.5}
    current.update({f"tcp.r{i}": float(i) for i in range(1, 7)})
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment.turntable_lease = lease
    environment._latest_tcp_y = 0.390
    environment._latest_raw_obs = current
    environment._latest_raw_obs_at = time.monotonic()
    environment._last_target_tcp_x = 0.500
    environment._last_target_tcp_y = 0.390
    environment._last_target_tcp_z = 0.100
    environment.robot = SimpleNamespace(send_action=MagicMock())

    environment.send_action(np.asarray(
        [0.550, 0.420, 0.100, *([0.0] * 6), 0.5], dtype=np.float32,
    ))

    sent = environment.robot.send_action.call_args.args[0]
    assert sent["tcp.x"] == pytest.approx(0.550)
    assert sent["tcp.y"] == pytest.approx(0.395)


def test_guard_ack_transition_replaces_previous_async_target_with_hold():
    lease = lease_without_ros()
    lease._guard_max_y_m = 0.325
    current = {"tcp.x": 0.4, "tcp.y": 0.28, "tcp.z": 0.25, "gripper.pos": 0.5}
    current.update({f"tcp.r{i}": float(i) for i in range(1, 7)})
    environment = DobotNova5DHRealEnv.__new__(DobotNova5DHRealEnv)
    environment._guard_action_lock = threading.RLock()
    environment.turntable_lease = lease
    environment.robot = SimpleNamespace(
        is_connected=True,
        _pause_action_worker=lambda: nullcontext(),
        get_observation=MagicMock(return_value=current),
        _send_action_sync=MagicMock(),
    )
    installed = MagicMock()

    environment._install_guard_after_worker_drain(installed)

    installed.assert_called_once()
    sent = environment.robot._send_action_sync.call_args.args[0]
    assert sent["tcp.y"] == pytest.approx(0.28)
    assert sent["tcp.z"] == pytest.approx(0.25)


def test_heartbeat_continues_during_blocked_guard_transition_and_policy_stop():
    lease = lease_without_ros()
    lease._heartbeat_stop = threading.Event()
    lease._node = MagicMock()
    entered = threading.Event()
    release = threading.Event()
    beat = threading.Event()
    lease._heartbeat = beat.set

    def delayed_transition(install):
        entered.set()
        assert release.wait(2.0)
        install()

    lease.set_guard_transition_handler(delayed_transition)
    guard_thread = threading.Thread(
        target=lease._guard_limit_callback,
        args=(SimpleNamespace(data=[1.0, 0.325, 0.535, 0.075, 2.0]),),
    )
    heartbeat_thread = threading.Thread(target=lease._heartbeat_loop)
    try:
        guard_thread.start()
        assert entered.wait(2.0)
        lease.request_stop()
        heartbeat_thread.start()
        assert beat.wait(0.5)
        assert guard_thread.is_alive()
    finally:
        release.set()
        lease._heartbeat_stop.set()
        guard_thread.join(timeout=2.0)
        if heartbeat_thread.ident is not None:
            heartbeat_thread.join(timeout=2.0)
    assert not guard_thread.is_alive()
    assert not heartbeat_thread.is_alive()
