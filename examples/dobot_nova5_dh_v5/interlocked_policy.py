"""Keep VLA action time aligned with the robot during turntable holds."""

from typing import Callable

from xense_client import base_policy
from lerobot.utils.robot_utils import get_logger


logger = get_logger("DobotNova5DHInterlockedPolicy")


class InterlockedPolicy(base_policy.BasePolicy):
    """Stop consuming chunks while 102 is blocked, then infer from fresh state."""

    def __init__(self, factory: Callable[[], base_policy.BasePolicy], environment):
        self._factory = factory
        self._inner = factory()
        self._environment = environment
        self._on_resume = None

    def set_on_resume(self, callback) -> None:
        self._on_resume = callback

    def infer(self, observation: dict) -> dict:
        if self._environment.policy_blocked:
            if not self._environment.poll_policy_resume():
                return {"_interlock_hold": True}
            stop = getattr(self._inner, "stop", None)
            if stop is not None:
                stop()
            self._inner = self._factory()
            if self._on_resume is not None:
                self._on_resume()
            # RTC needs two fresh warmup chunks; ordinary chunking is a no-op.
            self._inner.warmup(observation)
            logger.info("102 VLA replanned from the latest observation after lease grant")

        generation = self._environment.policy_generation
        result = dict(self._inner.infer(observation))
        result["_policy_generation"] = generation
        return result

    def reset(self) -> None:
        self._inner.reset()

    def warmup(self, observation: dict) -> None:
        self._inner.warmup(observation)
