import threading
import time
from contextlib import nullcontext
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from examples.dobot_nova5_dh_v5.real_env import DobotNova5DHRealEnv
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
    latest = {"state": "fresh"}
    result = policy.infer(latest)
    assert result["_policy_generation"] == 2
    assert len(brokers) == 2
    brokers[0].stop.assert_called_once()
    brokers[1].warmup.assert_called_once_with(latest)
    brokers[1].infer.assert_called_once_with(latest)


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
