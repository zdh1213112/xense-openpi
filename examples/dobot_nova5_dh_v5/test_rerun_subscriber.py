"""Offline tests: telemetry must never hold up V5 or mutate policy inputs."""

import io
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from examples.dobot_nova5_dh_v5 import rerun_subscriber as telemetry


def sample(value=0.0):
    return {
        "state": np.full(10, value, dtype=np.float32),
        "images": {name: np.full((3, 32, 32), value, dtype=np.uint8) for name in ("head", "wrist")},
        "images_raw": {"unused": object()},
    }, {"actions": np.full(10, value, dtype=np.float32)}


def test_throttle_and_snapshot_ownership(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: clock[0])
    subscriber = telemetry.RemoteRerunSubscriber(hz=5)
    observation, action = sample(1)
    subscriber.on_step(observation, action)
    snapshot = subscriber._pending.get_nowait()
    observation["images"]["head"][:] = 7
    observation["state"][:] = 7
    action["actions"][:] = 7
    assert np.all(snapshot.images["head"] == 1)
    assert np.all(snapshot.state == 1)
    assert np.all(snapshot.action == 1)
    assert set(snapshot.images) == {"head", "wrist"}
    clock[0] += 0.1
    subscriber.on_step(observation, action)
    assert subscriber._pending.empty()
    clock[0] += 0.2
    subscriber.on_step(observation, action)
    assert subscriber._pending.qsize() == 1


def test_raw_camera_images_are_preferred_copied_and_fall_back_per_camera():
    subscriber = telemetry.RemoteRerunSubscriber()
    observation, action = sample(1)
    head = np.full((720, 1280, 3), 12, dtype=np.uint8)
    observation["images_raw"]["head"] = head
    subscriber.on_step(observation, action)
    snapshot = subscriber._pending.get_nowait()
    head[:] = 77
    assert snapshot.images["head"].shape == (720, 1280, 3)
    assert np.all(snapshot.images["head"] == 12)
    assert np.all(snapshot.images["wrist"] == 1)
    assert np.all(observation["images"]["head"] == 1)


def test_policy_image_mode_can_keep_original_low_bandwidth_behavior():
    subscriber = telemetry.RemoteRerunSubscriber(image_source="policy")
    observation, action = sample(1)
    observation["images_raw"]["head"] = np.zeros((720, 1280, 3), dtype=np.uint8)
    subscriber.on_step(observation, action)
    snapshot = subscriber._pending.get_nowait()
    assert snapshot.images["head"].shape == (3, 32, 32)
    assert np.all(snapshot.images["head"] == 1)


def test_image_width_limit_preserves_aspect_ratio_without_upscaling():
    subscriber = telemetry.RemoteRerunSubscriber(image_max_width=640)
    native = np.full((720, 1280, 3), 25, dtype=np.uint8)
    image = subscriber._prepare_image(native)
    assert image.shape == (360, 640, 3)
    assert native.shape == (720, 1280, 3)
    assert np.all(image == 25)
    assert subscriber._prepare_image(np.zeros((3, 224, 224), dtype=np.uint8)).shape == (224, 224, 3)


def trajectory_snapshot(elapsed, *, episode=1, holding=False):
    observation, action = sample()
    observation["state"][:3] = (elapsed, elapsed / 2, 0.1)
    action["actions"][:3] = observation["state"][:3] + 0.02
    return telemetry._Snapshot(
        episode=episode, step=round(elapsed * 10), elapsed=elapsed,
        state=observation["state"], images={}, action=None if holding else action["actions"], holding=holding,
    )


def latest_entity(recording, path):
    return next(call.args[1] for call in reversed(recording.log.call_args_list) if call.args[0] == path)


def test_trails_show_recent_positions_with_older_segments_fading():
    pytest.importorskip("rerun")
    subscriber = telemetry.RemoteRerunSubscriber(hz=10, trail_seconds=1)
    recording = MagicMock()
    for elapsed in np.arange(8) * 0.25:
        subscriber._log_trails(recording, trajectory_snapshot(elapsed))
    for path in ("world/tcp_trail", "world/vla_target_trail"):
        entity = latest_entity(recording, path)
        segments = entity.strips.as_arrow_array().to_pylist()
        alphas = [rgba & 0xFF for rgba in entity.colors.as_arrow_array().to_pylist()]
        offset = 0.02 if "target" in path else 0
        assert len(segments) == 4
        assert segments[0][0][0] == pytest.approx(0.75 + offset)
        assert segments[-1][-1][0] == pytest.approx(1.75 + offset)
        assert alphas[0] < alphas[-1]
        assert len(subscriber._tcp_trail) <= subscriber._tcp_trail.maxlen


@pytest.mark.parametrize(("episode", "elapsed"), [(2, 0.4), (1, 4.0)])
def test_trails_do_not_bridge_episodes_or_long_data_gaps(episode, elapsed):
    rr = pytest.importorskip("rerun")
    subscriber = telemetry.RemoteRerunSubscriber()
    recording = MagicMock()
    subscriber._log_trails(recording, trajectory_snapshot(0))
    subscriber._log_trails(recording, trajectory_snapshot(0.2))
    subscriber._log_trails(recording, trajectory_snapshot(elapsed, episode=episode))
    assert isinstance(latest_entity(recording, "world/tcp_trail"), rr.Clear)
    assert isinstance(latest_entity(recording, "world/vla_target_trail"), rr.Clear)
    assert len(subscriber._tcp_trail) == 1


def test_hold_breaks_target_trail_while_tcp_trail_continues():
    rr = pytest.importorskip("rerun")
    subscriber = telemetry.RemoteRerunSubscriber()
    recording = MagicMock()
    for elapsed in (0, 0.2):
        subscriber._log_trails(recording, trajectory_snapshot(elapsed))
    subscriber._log_trails(recording, trajectory_snapshot(0.4, holding=True))
    assert isinstance(latest_entity(recording, "world/tcp_trail"), rr.LineStrips3D)
    assert isinstance(latest_entity(recording, "world/vla_target_trail"), rr.Clear)
    subscriber._log_trails(recording, trajectory_snapshot(0.6))
    assert isinstance(latest_entity(recording, "world/vla_target_trail"), rr.Clear)
    assert len(subscriber._action_trail) == 1


def test_zero_seconds_disables_trails():
    subscriber = telemetry.RemoteRerunSubscriber(trail_seconds=0)
    recording = MagicMock()
    subscriber._log_trails(recording, trajectory_snapshot(0))
    recording.log.assert_not_called()


def test_slow_sender_keeps_only_latest_snapshot_without_blocking_runtime():
    subscriber = telemetry.RemoteRerunSubscriber(hz=30)
    sending, release, latest = threading.Event(), threading.Event(), threading.Event()
    received = []

    def publish(snapshot):
        received.append(snapshot.state[0])
        if len(received) == 1:
            sending.set()
            assert release.wait(3)
        else:
            latest.set()

    subscriber._publish = publish
    subscriber.on_episode_start()
    try:
        subscriber.on_step(*sample(0))
        assert sending.wait(1)
        # A deliberately stuck sender must not prevent 20 runtime callbacks
        # from completing; they overwrite one pending frame instead.
        for value in range(1, 21):
            subscriber._last_sample_at = -float("inf")
            subscriber.on_step(*sample(value))
        assert subscriber._pending.qsize() == 1
        assert received == [0]
        release.set()
        assert latest.wait(1)
        assert received == [0, 20]
    finally:
        release.set()
        subscriber.close()
    assert not subscriber._thread.is_alive()


def test_unreachable_viewer_never_creates_sdk_backlog_and_can_reconnect(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: clock[0])
    subscriber = telemetry.RemoteRerunSubscriber()
    subscriber.on_step(*sample())
    snapshot = subscriber._pending.get_nowait()
    probe = MagicMock(side_effect=OSError("viewer offline"))
    monkeypatch.setattr(telemetry.socket, "create_connection", probe)
    subscriber._open_recording = MagicMock()
    with pytest.raises(OSError, match="offline"):
        subscriber._publish(snapshot)
    for _ in range(100):
        subscriber._publish(snapshot)
    probe.assert_called_once()
    subscriber._open_recording.assert_not_called()
    # On the next probe interval the receiver can return without restarting VLA.
    clock[0] += 2
    probe.side_effect = None
    subscriber._publish(snapshot)
    subscriber._open_recording.assert_called_once()
    subscriber._recording.flush.assert_called_once_with(blocking=True)
    subscriber._disconnect()


def test_sender_exception_does_not_kill_worker_or_inference():
    subscriber = telemetry.RemoteRerunSubscriber()
    failed, recovered = threading.Event(), threading.Event()

    def publish(snapshot):
        if not failed.is_set():
            failed.set()
            raise OSError("receiver closed")
        recovered.set()

    subscriber._publish = publish
    subscriber.on_episode_start()
    try:
        subscriber.on_step(*sample())
        assert failed.wait(1)
        subscriber._last_sample_at = -float("inf")
        subscriber.on_step(*sample())
        assert recovered.wait(1)
    finally:
        subscriber.close()


def test_hold_does_not_display_unexecuted_action_and_bad_input_is_ignored():
    subscriber = telemetry.RemoteRerunSubscriber()
    observation, action = sample()
    action["_interlock_hold"] = True
    subscriber.on_step(observation, action)
    snapshot = subscriber._pending.get_nowait()
    assert snapshot.action is None
    assert snapshot.holding
    subscriber._last_sample_at = -float("inf")
    observation["state"] = np.zeros(20)
    subscriber.on_step(observation, action)
    assert subscriber._pending.empty()


def test_real_sdk_can_encode_layout_images_and_scalars_without_viewer(tmp_path, monkeypatch):
    rr = pytest.importorskip("rerun")
    if rr.__version__.split(".")[:2] != ["0", "23"]:
        pytest.skip("Integration requires the deployed Rerun 0.23 SDK")
    subscriber = telemetry.RemoteRerunSubscriber()
    recording = rr.RecordingStream("v5_offline_test")
    recording.save(str(tmp_path / "telemetry.rrd"))
    recording.log = MagicMock(wraps=recording.log)
    recording.set_time = MagicMock(wraps=recording.set_time)
    recording.send_blueprint = MagicMock(wraps=recording.send_blueprint)
    # Exercise the real layout and archetypes, redirecting all output to disk.
    recording.connect_grpc = MagicMock()
    monkeypatch.setattr(rr, "RecordingStream", lambda *args, **kwargs: recording)
    subscriber._recording = subscriber._open_recording()
    assert recording.connect_grpc.call_args.kwargs["default_blueprint"] is not None
    assert recording.connect_grpc.call_args.kwargs["flush_timeout_sec"] == 0.2
    recording.send_blueprint.assert_called_once()
    subscriber._next_probe_at = float("inf")
    observation, action = sample()
    observation["images_raw"]["head"] = np.full((720, 1280, 3), 80, dtype=np.uint8)
    subscriber.on_step(observation, action)
    snapshot = subscriber._pending.get_nowait()
    subscriber._publish(snapshot)
    recording.set_time.assert_called_once_with("vla_time", duration=snapshot.elapsed)
    jpeg = bytes(latest_entity(recording, "cameras/head").blob.as_arrow_array().to_pylist()[0])
    image_module = pytest.importorskip("PIL.Image")
    with image_module.open(io.BytesIO(jpeg)) as image:
        assert image.size == (1280, 720)
    for elapsed in (0.25, 0.5, 0.75):
        subscriber._publish(trajectory_snapshot(elapsed))
    assert isinstance(latest_entity(recording, "world/tcp_trail"), rr.LineStrips3D)
    assert isinstance(latest_entity(recording, "world/vla_target_trail"), rr.LineStrips3D)
    subscriber._disconnect()
    assert (tmp_path / "telemetry.rrd").stat().st_size > 100


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("action_hz", [0.0, 30.0])
def test_main_connects_optional_subscriber_to_both_runtimes_and_closes_it(monkeypatch, enabled, action_hz):
    from examples.dobot_nova5_dh_v5 import main as entry  # noqa: PLC0415

    subscriber = MagicMock()
    factory = MagicMock(return_value=subscriber)
    monkeypatch.setattr(telemetry, "RemoteRerunSubscriber", factory)
    monkeypatch.setattr(entry._websocket_client_policy, "WebsocketClientPolicy", MagicMock())
    monkeypatch.setattr(entry._env, "DobotNova5DHEnvironment", MagicMock())
    monkeypatch.setattr(entry, "InterlockedPolicy", MagicMock())
    monkeypatch.setattr(entry._paced_broker, "PacedBroker", MagicMock())
    sync_runtime, decoupled_runtime = MagicMock(), MagicMock()
    monkeypatch.setattr(entry._runtime, "Runtime", sync_runtime)
    monkeypatch.setattr(entry._decoupled_runtime, "DecoupledRuntime", decoupled_runtime)
    monkeypatch.setattr(entry.signal, "signal", MagicMock())

    entry.main(entry.Args(rerun_enabled=enabled, action_hz=action_hz))

    active_runtime = decoupled_runtime if action_hz else sync_runtime
    assert active_runtime.call_args.kwargs["subscribers"] == ([subscriber] if enabled else [])
    if enabled:
        assert factory.call_args.kwargs["host"] == "192.168.111.183"
        assert factory.call_args.kwargs["image_source"] == "raw"
        assert factory.call_args.kwargs["image_max_width"] == 1280
        assert factory.call_args.kwargs["jpeg_quality"] == 90
        assert factory.call_args.kwargs["trail_seconds"] == 5.0
        subscriber.close.assert_called_once()
    else:
        factory.assert_not_called()


@pytest.mark.parametrize("kwargs", [
    {"hz": 0}, {"hz": float("nan")}, {"port": 0}, {"jpeg_quality": 101},
    {"image_source": "invalid"}, {"image_max_width": 0},
    {"trail_seconds": -1}, {"trail_seconds": float("inf")},
])
def test_bad_configuration_is_rejected_before_hardware_is_connected(kwargs):
    with pytest.raises(ValueError, match="rerun_"):
        telemetry.RemoteRerunSubscriber(**kwargs)
