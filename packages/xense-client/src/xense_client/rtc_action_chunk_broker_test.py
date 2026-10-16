import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from xense_client.rtc_action_chunk_broker import RTCActionChunkBroker


def test_resume_uses_one_hot_shape_inference_and_executes_action_zero():
    held = np.asarray([0.506, 0.321, 0.211, *([0.0] * 6), 0.0], dtype=np.float32)
    actions = np.repeat(held[None, :], 50, axis=0)
    actions[:, 1] += np.arange(50, dtype=np.float32) * 0.005
    policy = MagicMock()
    policy.infer.return_value = {"actions": actions}
    broker = RTCActionChunkBroker(policy, frequency_hz=30, action_queue_size_to_get_new_actions=30)
    observation = {"state": held, "images": {}}
    try:
        broker.replan(observation, hold_action=held)
        first = broker.infer(observation)["actions"]
        broker.stop()
        policy.infer.assert_called_once()
        call = policy.infer.call_args
        assert call.args[0] is observation
        assert call.kwargs["inference_delay"] == 0
        np.testing.assert_array_equal(
            call.kwargs["prev_chunk_left_over"], np.repeat(held[None, :], 30, axis=0),
        )
        np.testing.assert_array_equal(first, actions[0])
        assert first[1] != actions[20, 1]
        assert broker._warmup_done
    finally:
        broker.stop()


def test_initial_episode_still_performs_both_warmup_calls():
    policy = MagicMock()
    policy.infer.return_value = {"actions": np.ones((50, 10), dtype=np.float32)}
    broker = RTCActionChunkBroker(policy, action_queue_size_to_get_new_actions=30)
    try:
        broker.warmup({"state": np.zeros(10)})
        broker.stop()
        assert policy.infer.call_count == 2
        assert policy.infer.call_args_list[0].kwargs["prev_chunk_left_over"] is None
        assert policy.infer.call_args_list[1].kwargs["prev_chunk_left_over"].shape == (30, 10)
        assert broker._warmup_done
    finally:
        broker.stop()


def test_retired_broker_discards_a_late_inference_response():
    started = threading.Event()
    release = threading.Event()

    def infer(*args, **kwargs):
        started.set()
        assert release.wait(2.0)
        return {"actions": np.ones((50, 10), dtype=np.float32)}

    policy = MagicMock()
    policy.infer.side_effect = infer
    broker = RTCActionChunkBroker(policy)
    broker._warmup_done = True
    broker._latest_obs = {"state": np.zeros(10)}
    stopper = threading.Thread(target=broker.stop)
    try:
        broker._start_thread_if_needed()
        assert started.wait(1.0)
        stopper.start()
        assert broker._stop_event.wait(1.0)
        release.set()
        stopper.join(timeout=2.0)
        assert not stopper.is_alive()
        assert broker._action_queue.empty()
        assert not broker._first_inference_done.is_set()
    finally:
        release.set()
        if stopper.ident is not None:
            stopper.join(timeout=2.0)
        broker.stop()


@pytest.mark.parametrize("held", [np.zeros((1, 10)), np.zeros(0), np.full(10, np.nan)])
def test_resume_rejects_invalid_held_actions_before_starting_inference(held):
    policy = MagicMock()
    broker = RTCActionChunkBroker(policy)
    with pytest.raises(ValueError, match="finite 1-D"):
        broker.replan({}, hold_action=held)
    assert not broker._thread_started
    policy.infer.assert_not_called()
