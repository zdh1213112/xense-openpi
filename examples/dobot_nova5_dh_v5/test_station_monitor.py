"""Offline monitoring tests: no real robot, camera, ROS node or policy server."""

import importlib.util
import os
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

source = Path(os.environ.get("VLA102_MONITOR_SOURCE", str(Path(__file__).with_name("station_monitor.py"))))
spec = importlib.util.spec_from_file_location("vla102_monitor_tested", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VlaMonitor = module.VlaMonitor


def pose(x=0.5, grip=0.0):
    return (x, 0.1, 0.2, 1, 0, 0, 0, 1, 0, grip)


def ready_monitor(**kwargs):
    now = [0.0]
    monitor = VlaMonitor(clock=lambda: now[0], **kwargs)
    epoch = monitor.begin_plan()
    monitor.mark_ready()
    return monitor, now, epoch


def captures(now):
    return {name: (1, 100, now) for name in ("head", "wrist")}


def test_warmup_and_legitimate_interlock_wait_are_not_failures():
    monitor = VlaMonitor()
    assert monitor.snapshot({"head": None})[0] == "initializing"
    monitor.mark_ready()
    monitor.context(blocked=True, reason="pick_guard")
    assert monitor.snapshot({"head": None})[0] == "normal"
    monitor.context(blocked=True, reason="feedback", feedback_fault=True)
    assert monitor.snapshot({})[0] == "fault"
    monitor.context(blocked=True, reason="feedback")
    assert monitor.snapshot({})[0] == "initializing"


def test_actual_queue_wait_not_a_low_queue_or_initial_jit_warns_and_recovers():
    monitor, now, epoch = ready_monitor()
    monitor.queue_wait(True)
    now[0] = 0.49
    assert monitor.snapshot(captures(now[0]))[0] == "normal"
    now[0] = 0.5
    assert monitor.snapshot(captures(now[0]))[1] == "vla102_action_starvation"
    monitor.queue_wait(False)
    assert monitor.snapshot(captures(now[0]))[0] == "normal"
    monitor.begin_plan()
    monitor.queue_wait(True)
    now[0] = 100
    assert monitor.snapshot(captures(now[0]))[0] == "initializing"


def test_consecutive_inference_failures_require_two_good_replies_to_clear():
    monitor, now, epoch = ready_monitor()
    for _ in range(2):
        monitor.response(epoch, success=False)
        assert monitor.snapshot(captures(now[0]))[0] == "normal"
    monitor.response(epoch, success=False)
    assert monitor.snapshot(captures(now[0]))[1] == "vla102_inference_retry"
    monitor.response(epoch, success=True)
    assert monitor.snapshot(captures(now[0]))[0] == "warning"
    monitor.response(epoch, success=True)
    assert monitor.snapshot(captures(now[0]))[0] == "normal"
    new_epoch = monitor.begin_plan()
    monitor.mark_ready()
    for _ in range(5):
        monitor.response(epoch, success=False)
    assert monitor.epoch == new_epoch
    assert monitor.snapshot(captures(now[0]))[0] == "normal"


def test_frame_age_uses_capture_not_observation_return_time_and_no_pixel_comparison():
    monitor, now, epoch = ready_monitor()
    for _ in range(10):
        monitor.request(epoch, captures(0))  # Fast calls can share one fresh frame.
    assert monitor.snapshot(captures(now[0]))[0] == "normal"
    now[0] = 2
    monitor.request(epoch, captures(0))
    assert monitor.snapshot(captures(now[0]))[1] == "vla102_stale_vision"
    monitor.request(epoch, captures(now[0]))
    assert monitor.snapshot(captures(now[0]))[0] == "normal"
    assert monitor.snapshot(captures(0))[1] == "vla102_stale_vision"


def test_real_capture_probe_does_not_refresh_time_for_duplicate_hardware_frame():
    now = [1.0]
    frame_id = [1]
    frame = SimpleNamespace(get_frame_number=lambda: frame_id[0], get_timestamp=lambda: frame_id[0] * 100.0)
    pipeline = SimpleNamespace(try_wait_for_frames=lambda **kwargs: (True, SimpleNamespace(get_color_frame=lambda: frame)))
    camera = SimpleNamespace(rs_pipeline=pipeline)

    def read(**kwargs):
        camera.rs_pipeline.try_wait_for_frames(**kwargs)
        return object()

    camera.read = read
    probe = module.CameraStampProbe(camera, clock=lambda: now[0])
    image = camera.read()
    assert probe.snapshot(image) == (1, 100, 1)
    now[0] = 10
    duplicate = camera.read()
    assert probe.snapshot(duplicate) == (1, 100, 1)
    frame_id[0] = 2
    newer = camera.read()
    assert probe.snapshot(newer)[2] == pytest.approx(1.1)  # Delayed delivery is not a new capture.
    assert probe.snapshot(image) == (1, 100, 1)


def test_broker_reset_returns_to_warmup_instead_of_reporting_a_stale_old_camera():
    monitor, now, epoch = ready_monitor()
    calls = []
    inner = SimpleNamespace(reset=lambda: calls.append("reset"))
    module.ObservedBroker(inner, monitor).reset()
    assert calls == ["reset"]
    assert monitor.snapshot({"head": None})[0] == "initializing"


def test_reporter_reuses_node_and_preserves_fault_on_shutdown(monkeypatch):
    import json
    import sys
    import threading
    from unittest.mock import Mock

    monkeypatch.setitem(sys.modules, "rclpy.qos", SimpleNamespace(
        QoSProfile=lambda **kw: SimpleNamespace(**kw),
        DurabilityPolicy=SimpleNamespace(TRANSIENT_LOCAL=1),
        ReliabilityPolicy=SimpleNamespace(BEST_EFFORT=1)))
    monkeypatch.setitem(sys.modules, "std_msgs.msg", SimpleNamespace(String=lambda: SimpleNamespace(data="")))
    messages = []
    logger = Mock()
    node = SimpleNamespace(create_publisher=lambda *a: SimpleNamespace(publish=lambda msg: messages.append(msg.data)),
        create_subscription=lambda *a: object(), create_timer=lambda *a: SimpleNamespace(cancel=Mock()),
        destroy_timer=Mock(), destroy_subscription=Mock(), destroy_publisher=Mock(), get_logger=lambda: logger)
    env = SimpleNamespace(turntable_lease=SimpleNamespace(_node=node), policy_blocked=False,
        _feedback_recovery_hold=False, robot=SimpleNamespace(feedback_recovery_pending=False),
        _stop_requested=threading.Event(), _closing=False, _blocked_reason=None)
    monitor, now, epoch = ready_monitor()
    reporter = module.VlaStatusReporter(env, monitor,
        {"head": SimpleNamespace(snapshot=lambda: (1, 100, 0))})
    assert messages and '"normal"' in messages[-1]
    run_irregular_small_motion(monitor, now, epoch)
    reporter.publish()
    warning = json.loads(messages[-1])
    assert warning["state"] == "warning" and warning["code"] == "vla102_repeated_motion"
    assert warning["repeat"]["gate"] == "low_motion"
    assert warning["repeat"]["xyz_span_mm"][1] > 20
    logger.warning.assert_called_once()
    monitor.terminal_error("existing runtime error")
    reporter.close()
    assert '"fault"' in messages[-1]
    node.destroy_timer.assert_called_once()
    count = len(messages)
    reporter.publish()  # A queued timer callback after close cannot overwrite red.
    reporter.close()    # Idempotent shutdown does not double-destroy entities.
    assert len(messages) == count
    node.destroy_timer.assert_called_once()


def test_v5_entry_default_off_and_common_rtc_camera_files_are_not_hooked():
    main_path = Path(__file__).with_name("main.py")
    if not main_path.exists():
        pytest.skip("staging directory has no V5 main")
    tree = ast.parse(main_path.read_text())
    args = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "Args")
    flag = next(item for item in args.body if isinstance(item, ast.AnnAssign)
                and item.target.id == "alarm_light_enabled")
    assert flag.value.value is False
    main = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "main")
    branch = next(item for item in main.body if isinstance(item, ast.If)
                  and ast.unparse(item.test) == "args.alarm_light_enabled")
    assert any(isinstance(item, ast.ImportFrom) and item.module.endswith("station_monitor")
               for item in branch.body)


def run_motion(monitor, now, epoch, *, repeating=True, new_chunks=True):
    for index in range(76):
        now[0] = index * 0.2
        if new_chunks and index % 5 == 0:
            monitor.response(epoch, success=True)
        x = 0.5 + (0.08 * (index // 3 % 2) if repeating else index * 0.005)
        monitor.target(pose(x))
        monitor.observe(pose(x))


def test_cross_chunk_repeated_waypoints_warn_but_real_handoff_progress_resets():
    monitor, now, epoch = ready_monitor()
    run_motion(monitor, now, epoch)
    assert monitor.snapshot(captures(now[0]))[1] == "vla102_repeated_motion"
    monitor.progress()
    assert monitor.snapshot(captures(now[0]))[0] == "normal"


@pytest.mark.parametrize("repeating,new_chunks", [(False, True), (True, False)])
def test_moving_forward_and_repetitions_inside_one_chunk_do_not_warn(repeating, new_chunks):
    monitor, now, epoch = ready_monitor()
    run_motion(monitor, now, epoch, repeating=repeating, new_chunks=new_chunks)
    assert monitor.snapshot(captures(now[0]))[0] == "normal"


def test_active_stationary_policy_across_new_chunks_warns_after_the_window():
    monitor, now, epoch = ready_monitor()
    for index in range(81):
        now[0] = index * 0.25
        if index % 3 == 0:
            monitor.response(epoch, success=True)
        monitor.target(pose())
        monitor.observe(pose())
        if now[0] < 14.5:
            assert monitor.snapshot(captures(now[0]))[0] == "normal"
    assert monitor.snapshot(captures(now[0]))[1] == "vla102_repeated_motion"
    assert monitor.repeat_diagnostics()["gate"] == "low_motion"


def run_irregular_small_motion(monitor, now, epoch):
    # A 28 mm Y envelope like the recorded 20:24 run; no exact waypoint period.
    import random
    rng = random.Random(102)
    for index in range(81):
        now[0] = index * 0.25
        if index % 3 == 0:
            monitor.response(epoch, success=True)
        state = list(pose(0.655 + rng.uniform(-0.004, 0.004), grip=1))
        state[1] = 0.23 + rng.uniform(-0.014, 0.014)
        state[2] = -0.065 + rng.uniform(-0.003, 0.003)
        state[3] = -0.19 + rng.uniform(-0.035, 0.035)
        monitor.target(state)
        monitor.observe(state)


def test_irregular_28mm_jitter_warns_and_logs_the_numeric_evidence():
    monitor, now, epoch = ready_monitor()
    run_irregular_small_motion(monitor, now, epoch)
    evidence = monitor.repeat_diagnostics()
    assert 20 < evidence["xyz_span_mm"][1] < 28
    assert evidence["position_span_limit_mm"] == 35
    assert evidence["gate"] == "low_motion"
    assert evidence["new_chunks"] >= 3 and evidence["applied_actions"] >= 10
    assert monitor.snapshot(captures(now[0]))[1] == "vla102_repeated_motion"
    now[0] += .25
    monitor.target(pose(.75))
    monitor.observe(pose(.75))
    assert monitor.snapshot(captures(now[0]))[0] == "normal"
    assert monitor.repeat_diagnostics()["gate"] == "progressing"


def test_small_motion_span_is_configurable_and_rejects_invalid_thresholds():
    monitor, now, epoch = ready_monitor(position_span_m=.02)
    run_irregular_small_motion(monitor, now, epoch)
    assert monitor.snapshot(captures(now[0]))[0] == "normal"
    for invalid in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            VlaMonitor(position_span_m=invalid)


@pytest.mark.parametrize("phase", ["warmup", "interlock_wait", "paused", "no_recent_actions", "need_new_chunks"])
def test_long_static_observations_without_eligible_execution_do_not_warn(phase):
    monitor, now, epoch = ready_monitor()
    if phase == "warmup":
        epoch = monitor.begin_plan()
    if phase == "interlock_wait":
        monitor.context(blocked=True, reason="pick_guard")
    if phase == "paused":
        monitor.context(stopping=True)
    monitor.target(pose())  # One old target alone cannot prove continued execution.
    for index in range(81):
        now[0] = index * .25
        if phase != "need_new_chunks" and index % 3 == 0:
            monitor.response(epoch, success=True)
        if phase != "no_recent_actions":
            monitor.target(pose())
        monitor.observe(pose())
    assert monitor.snapshot(captures(now[0]))[0] != "warning"
    assert monitor.repeat_diagnostics()["gate"] == phase


def test_metadata_is_not_sent_to_policy_server_and_old_replies_do_not_rearm():
    monitor, now, epoch = ready_monitor()
    seen = []
    client = SimpleNamespace(infer=lambda obs, **kw: seen.append(obs) or {"actions": [pose()]})
    observed = module.ObservedPolicy(client, monitor, epoch)
    result = observed.infer({"state": pose(), "images": {}, module.META_KEY: captures(0)})
    assert module.META_KEY not in seen[0]
    assert result == {"actions": [pose()]}
    monitor.begin_plan()
    observed.infer({"state": pose(), module.META_KEY: None})
    assert not monitor.warnings


def test_monitor_never_swallows_original_execution_error():
    monitor, now, epoch = ready_monitor()

    def broken(_obs):
        raise RuntimeError("existing queue timeout")

    broker = module.ObservedBroker(SimpleNamespace(infer=broken), monitor)
    with pytest.raises(RuntimeError, match="existing queue timeout"):
        broker.infer({})
    assert monitor.snapshot(captures(0))[0] == "fault"

    monitor.context(stopping=True)
    assert monitor.snapshot(captures(0))[0] == "fault"


def test_reporting_shutdown_error_cannot_prevent_real_environment_disconnect():
    calls = []

    def broken_reporter():
        raise RuntimeError("ROS context already closed")

    env = module.MonitoredEnvironment.__new__(module.MonitoredEnvironment)
    env.reporter = SimpleNamespace(close=broken_reporter)
    env.inner = SimpleNamespace(disconnect=lambda: calls.append("disconnect"))
    env.disconnect()
    assert calls == ["disconnect"]


def test_broker_reset_updates_only_its_monitoring_epoch_not_an_action():
    monitor, now, epoch = ready_monitor()
    client = module.ObservedPolicy(SimpleNamespace(infer=lambda *_: {"actions": [pose()]}), monitor, epoch)
    inner = SimpleNamespace(_policy=client, reset=lambda: None)
    module.ObservedBroker(inner, monitor).reset()
    assert client.epoch == monitor.epoch != epoch
    monitor.mark_ready()
    for _ in range(3):
        monitor.response(epoch, success=False)
    assert monitor.snapshot(captures(0))[0] == "normal"
    monitor.context(stopping=True)
    assert monitor.snapshot(captures(0))[0] == "waiting"


def test_inflight_reply_keeps_the_epoch_it_started_with_during_reset():
    monitor, now, epoch = ready_monitor()
    holder = {}

    def infer(_obs):
        holder["broker"].reset()
        monitor.mark_ready()
        return {"actions": [pose()]}

    client = module.ObservedPolicy(SimpleNamespace(infer=infer), monitor, epoch)
    inner = SimpleNamespace(_policy=client, reset=lambda: None)
    holder["broker"] = module.ObservedBroker(inner, monitor)
    result = client.infer({"state": pose()})
    assert result == {"actions": [pose()]}
    assert client.epoch == monitor.epoch != epoch
    assert monitor.chunks == monitor.good_chunks == 0


def test_new_plan_clears_previous_yellow_causes_but_never_a_terminal_fault():
    monitor, now, epoch = ready_monitor()
    monitor.warnings = dict(repeated_motion="loop", inference_retry="retry", stale_vision="old")
    monitor.last_target = pose()
    monitor.failures = 3
    monitor.begin_plan()
    monitor.mark_ready()
    assert monitor.snapshot(captures(0))[0] == "normal"
    assert monitor.last_target is None
    assert monitor.failures == 0
    monitor.terminal_error("existing safety stop")
    monitor.begin_plan()
    monitor.mark_ready()
    assert monitor.snapshot(captures(0))[0] == "fault"


def test_three_yellow_causes_use_requested_priority_and_remain_below_red():
    monitor, now, epoch = ready_monitor()
    monitor.warnings = dict(repeated_motion="loop", inference_retry="retry", stale_vision="old")
    assert monitor.snapshot(captures(-10))[1] == "vla102_repeated_motion"
    monitor.progress()
    assert monitor.snapshot(captures(-10))[1] == "vla102_inference_retry"
    monitor.warnings.pop("inference_retry")
    assert monitor.snapshot(captures(-10))[1] == "vla102_stale_vision"
    monitor.context(feedback_fault=True)
    assert monitor.snapshot(captures(-10))[0] == "fault"


def test_paced_actions_still_delivered_do_not_report_inner_queue_starvation():
    monitor, now, epoch = ready_monitor()
    monitor.queue_wait(True)
    for index in range(1, 11):
        now[0] = index * .2
        monitor.target(pose())  # Called after successful apply_action.
        assert monitor.snapshot(captures(now[0]))[0] == "normal"
    now[0] += .51
    assert monitor.snapshot(captures(now[0]))[1] == "vla102_action_starvation"
    monitor.queue_wait(False)
    assert monitor.snapshot(captures(now[0]))[0] == "normal"


@pytest.mark.parametrize("kind", ["rtc", "synchronous"])
def test_actual_broker_wait_is_observed_and_recovers_without_altering_result(kind):
    monitor, now, epoch = ready_monitor()
    expected = {"actions": pose()}

    def infer(_obs):
        now[0] = .6
        assert monitor.snapshot(captures(now[0]))[1] == "vla102_action_starvation"
        return expected

    inner = SimpleNamespace(infer=infer)
    if kind == "rtc":
        inner._action_queue = SimpleNamespace(qsize=lambda: 0)
    else:
        inner._last_results = None
    broker = module.ObservedBroker(inner, monitor)
    assert broker.infer({}) is expected
    assert monitor.snapshot(captures(now[0]))[0] == "normal"


def test_reporting_failures_never_replace_a_valid_policy_result_or_its_error():
    from unittest.mock import Mock
    monitor, now, epoch = ready_monitor()
    monitor.request = Mock(side_effect=RuntimeError("monitor request failed"))
    monitor.response = Mock(side_effect=RuntimeError("monitor response failed"))
    expected = {"actions": [pose()]}
    original = SimpleNamespace(infer=lambda obs: expected)
    client = module.ObservedPolicy(original, monitor, epoch)
    assert client.infer({"state": pose(), module.META_KEY: captures(0)}) is expected

    def failed(_obs):
        raise ValueError("original policy failure")

    original.infer = failed
    with pytest.raises(ValueError, match="original policy failure"):
        client.infer({})


def test_monitor_failure_does_not_prevent_original_broker_actions_reset_or_stop():
    from unittest.mock import Mock
    monitor, now, epoch = ready_monitor()
    for name in ("queue_wait", "mark_ready", "reset_phase", "terminal_error"):
        setattr(monitor, name, Mock(side_effect=RuntimeError("monitor failed")))
    expected = {"actions": pose()}
    inner = SimpleNamespace(infer=Mock(return_value=expected), reset=Mock(), stop=Mock(),
                            warmup=Mock(), _warmup_done=True)
    broker = module.ObservedBroker(inner, monitor)
    assert broker.infer({}) is expected
    broker.warmup({})
    broker.reset()
    broker.stop()
    inner.reset.assert_called_once_with()
    inner.stop.assert_called_once_with()
    inner.warmup.assert_called_once_with({})

    def failed(_obs):
        raise ValueError("original broker failure")

    inner.infer = failed
    with pytest.raises(ValueError, match="original broker failure"):
        broker.infer({})


def test_monitor_failure_never_prevents_environment_stop_or_changes_motion_failure():
    from unittest.mock import Mock
    monitor, now, epoch = ready_monitor()
    monitor.context = Mock(side_effect=RuntimeError("monitor failed"))
    monitor.terminal_error = Mock(side_effect=RuntimeError("monitor failed"))
    env = module.MonitoredEnvironment.__new__(module.MonitoredEnvironment)
    env.monitor = monitor
    env.inner = SimpleNamespace(request_stop=Mock(), apply_action=Mock(side_effect=ValueError("original motion failed")))
    env.request_stop()
    env.inner.request_stop.assert_called_once_with()
    with pytest.raises(ValueError, match="original motion failed"):
        env.apply_action({"actions": pose()})


def test_reporter_initialization_failure_keeps_existing_environment_available(monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr(module, "VlaStatusReporter", Mock(side_effect=RuntimeError("ROS unavailable")))
    inner = SimpleNamespace(_env=SimpleNamespace(robot=SimpleNamespace(cameras={})), disconnect=Mock())
    monitor, now, epoch = ready_monitor()
    env = module.MonitoredEnvironment(inner, monitor)
    assert env.reporter is None
    env.disconnect()
    inner.disconnect.assert_called_once_with()
