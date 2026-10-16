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
        self._stopped_for_hold = False

    def set_on_resume(self, callback) -> None:
        self._on_resume = callback

    def infer(self, observation: dict) -> dict:
        resumed = False
        if self._environment.policy_blocked:
            if not self._stopped_for_hold:
                # Retire any in-flight chunk while the arm is already held,
                # rather than waiting for it after 101 clears the turntable.
                stop = getattr(self._inner, "stop", None)
                if stop is not None:
                    stop()
                self._stopped_for_hold = True
            if not self._environment.poll_policy_resume():
                return {"_interlock_hold": True}
            self._inner = self._factory()
            self._stopped_for_hold = False
            if self._on_resume is not None:
                self._on_resume()
            generation = self._environment.policy_generation
            observation = self._environment.get_observation()
            # The server was warmed at episode start. An RTC hold has consumed
            # no actions, so one delay=0 inference replaces both warmup calls.
            replan = getattr(self._inner, "replan", None)
            if replan is not None:
                replan(observation, hold_action=observation["state"])
            resumed = True
        else:
            generation = self._environment.policy_generation
        result = dict(self._inner.infer(observation))
        result["_policy_generation"] = generation
        if resumed:
            logger.info("102 VLA replanned once from the latest observation after resume permission")
        return result

    def reset(self) -> None:
        if self._stopped_for_hold:
            self._inner = self._factory()
        self._inner.reset()
        self._stopped_for_hold = False

    def stop(self) -> None:
        stop = getattr(self._inner, "stop", None)
        if stop is not None:
            stop()

    def warmup(self, observation: dict) -> None:
        self._inner.warmup(observation)
