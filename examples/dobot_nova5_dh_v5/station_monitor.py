"""Opt-in V5 indication only: no robot commands, image copies or new threads.

Uses the existing camera capture/RTC/ROS threads. All findings are suspicions
unless the original runtime has already entered a fault. Never authorizes,
retries, stops or alters an action.
"""

from collections import OrderedDict, deque
import json
import logging
import math
import threading
import time
import uuid


STATE_TOPIC = "/station_v5/vla102_state"
META_KEY = "_vla102_monitor"


def report_only(callback, *args, **kwargs):
    """An indication failure must not change an action, error, or shutdown."""
    try:
        return callback(*args, **kwargs)
    except Exception:
        return None


class CameraStampProbe:
    """Attach hardware frame IDs to returned images without touching pixels."""

    def __init__(self, camera, clock=time.monotonic):
        self.camera, self.clock = camera, clock
        self.lock = threading.Lock()
        self.latest = None
        self.device_offset = None
        self.images = OrderedDict()
        original_pipeline = camera.rs_pipeline
        original_read = camera.read
        probe = self

        class PipelineProxy:
            def __getattr__(self, name):
                return getattr(original_pipeline, name)

            def try_wait_for_frames(self, *args, **kwargs):
                result = original_pipeline.try_wait_for_frames(*args, **kwargs)
                try:
                    if result[0] and result[1] is not None:
                        frame = result[1].get_color_frame()
                        key = (int(frame.get_frame_number()), float(frame.get_timestamp()))
                        if not math.isfinite(key[1]):
                            return result
                        with probe.lock:
                            if probe.latest is None or probe.latest[:2] != key:
                                now = probe.clock()
                                offset = now - key[1] / 1000.0
                                if (probe.latest is None or key[0] < probe.latest[0]
                                        or key[1] < probe.latest[1]):
                                    probe.device_offset = offset
                                else:
                                    probe.device_offset = min(probe.device_offset, offset)
                                probe.latest = (*key, min(now, key[1] / 1000.0 + probe.device_offset))
                except Exception:
                    pass  # Monitoring must never break capture.
                return result

        def read_with_stamp(*args, **kwargs):
            image = original_read(*args, **kwargs)
            try:
                with probe.lock:
                    probe.images[id(image)] = probe.latest
                    probe.images.move_to_end(id(image))
                    while len(probe.images) > 8:
                        probe.images.popitem(last=False)
            except Exception:
                pass
            return image

        camera.rs_pipeline = PipelineProxy()
        camera.read = read_with_stamp

    def snapshot(self, image=None):
        with self.lock:
            return self.latest if image is None else self.images.get(id(image))


class VlaMonitor:
    def __init__(self, *, window_s=15.0, position_span_m=0.035, frame_age_s=1.5, failure_count=3,
                 starvation_s=0.5, clock=time.monotonic):
        for value in (window_s, position_span_m, frame_age_s, starvation_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("VLA monitor times must be finite and positive")
        if failure_count < 1:
            raise ValueError("failure_count must be positive")
        self.window_s, self.frame_age_s = window_s, frame_age_s
        self.position_span_m = position_span_m
        self.failure_count, self.starvation_s = failure_count, starvation_s
        self.clock, self.lock = clock, threading.RLock()
        self.ready = self.stopping = False
        self.blocked = False
        self.fault = ""
        self.epoch = self.chunks = self.failures = self.good_chunks = 0
        self.queue_wait_at = None
        self.warnings = {}
        self.samples = deque(maxlen=256)
        self.last_sample_at = float("-inf")
        self.last_target = None
        self.last_target_at = None
        self.applied_actions = 0
        self.repeat_gate = "collecting_window"
        self.last_input_keys = None
        self.reused_inputs = 0

    def begin_plan(self):
        return self.reset_phase()

    def mark_ready(self):
        with self.lock:
            if not self.ready:
                self.samples.clear()
                self.chunks = 0
            self.ready = True

    def reset_phase(self):
        with self.lock:
            self.epoch += 1
            self.ready = False
            self.samples.clear()
            self.queue_wait_at = None
            self.failures = self.good_chunks = 0
            self.chunks = 0
            self.last_target = None
            self.last_target_at = None
            self.applied_actions = 0
            self.repeat_gate = "collecting_window"
            self.last_input_keys = None
            self.reused_inputs = 0
            self.last_sample_at = float("-inf")
            self.warnings.clear()
            return self.epoch

    def context(self, *, blocked=False, feedback_fault=False, stopping=False, reason="lease"):
        with self.lock:
            if feedback_fault:
                self.fault = "102 已进入反馈故障保护，运动/夹爪指令被禁止"
            elif not self.stopping:
                self.fault = "" if self.fault.startswith("102 已进入反馈故障保护") else self.fault
            if blocked != self.blocked or blocked or stopping:
                self.samples.clear()
                self.warnings.pop("repeated_motion", None)
                self.repeat_gate = "collecting_window"
                self.queue_wait_at = None
            self.blocked = blocked
            self.blocked_reason = reason
            self.stopping = stopping

    def progress(self):
        with self.lock:
            self.samples.clear()
            self.warnings.pop("repeated_motion", None)
            self.repeat_gate = "collecting_window"

    def observe(self, state):
        with self.lock:
            now = self.clock()
            if not self.ready or self.blocked or self.stopping or now - self.last_sample_at < 0.2:
                return
            pose = tuple(float(x) for x in state)
            if len(pose) != 10 or not all(math.isfinite(x) for x in pose):
                return
            self.last_sample_at = now
            self.samples.append((now, pose, self.last_target, self.chunks, self.applied_actions))
            while self.samples and now - self.samples[0][0] > self.window_s:
                self.samples.popleft()
            if len(self.samples) < 10 or now - self.samples[0][0] < max(0.2, self.window_s - 0.5):
                self.repeat_gate = "collecting_window"
                return
            if self.chunks - self.samples[0][3] < 3:
                self.repeat_gate = "need_new_chunks"
                return  # RTC frozen prefixes inside one chunk are not loops.
            if (self.applied_actions - self.samples[0][4] < 10 or self.last_target_at is None
                    or now - self.last_target_at > 1.0):
                self.repeat_gate = "no_recent_actions"
                self.warnings.pop("repeated_motion", None)
                return  # A quiet observation alone is not evidence of executing a loop.
            poses = [sample[1] for sample in self.samples]
            cells = []
            for pose in poses:
                cell = (*[round(x / 0.015) for x in pose[:3]],
                        *[round(x / 0.25) for x in pose[3:9]], int(pose[9] >= 0.5))
                if not cells or cell != cells[-1]:
                    cells.append(cell)
            periodic = any(len(cells) >= 3 * period and
                           cells[-period:] == cells[-2*period:-period] == cells[-3*period:-2*period]
                           for period in range(2, 13))
            spans = [max(p[i] for p in poses) - min(p[i] for p in poses) for i in range(9)]
            low_motion = max(spans[:3]) <= self.position_span_m and max(spans[3:]) < 0.15
            self.repeat_gate = "repeated_waypoints" if periodic else "low_motion" if low_motion else "progressing"
            if periodic or low_motion:
                self.warnings["repeated_motion"] = (
                    "102 疑似重复动作且任务无进展，请检查物料与取放动作" if periodic else
                    "102 持续执行但长期停留在小范围内，疑似任务无进展，请检查物料与取放动作")
            else:
                self.warnings.pop("repeated_motion", None)

    def target(self, action):
        with self.lock:
            flat = action.reshape(-1) if hasattr(action, "reshape") else action
            target = tuple(float(x) for x in flat)
            if len(target) != 10 or not all(math.isfinite(x) for x in target):
                return
            self.last_target = target
            self.last_target_at = self.clock()
            self.applied_actions += 1
            # In decoupled mode the outer paced buffer may still deliver
            # actions while its producer waits on the inner RTC queue.
            if self.queue_wait_at is not None:
                self.queue_wait_at = self.clock()

    def repeat_diagnostics(self):
        """Small numeric evidence for the existing status timer, never an action."""
        with self.lock:
            poses = [sample[1] for sample in self.samples]
            spans = ([max(p[i] for p in poses) - min(p[i] for p in poses) for i in range(9)]
                     if poses else None)
            gate = ("fault" if self.fault else "paused" if self.stopping else "interlock_wait"
                    if self.blocked else "warmup" if not self.ready else self.repeat_gate)
            return dict(gate=gate, window_s=self.window_s, position_span_limit_mm=round(self.position_span_m * 1000, 1),
                observed_s=round(self.samples[-1][0] - self.samples[0][0], 2) if poses else 0.0,
                new_chunks=self.chunks - self.samples[0][3] if poses else 0,
                applied_actions=self.applied_actions - self.samples[0][4] if poses else 0,
                xyz_span_mm=[round(x * 1000, 1) for x in spans[:3]] if spans else None,
                orientation_component_span=round(max(spans[3:]), 3) if spans else None)

    def request(self, epoch, metadata):
        with self.lock:
            if epoch != self.epoch or not self.ready or self.blocked or self.stopping:
                return
            stamps = metadata if isinstance(metadata, dict) else {}
            keys = tuple((name, tuple(stamps[name][:2])) for name in sorted(stamps) if stamps[name])
            bad = not keys or any(not stamp or self.clock() - stamp[2] > self.frame_age_s
                                  for stamp in stamps.values())
            self.reused_inputs = self.reused_inputs + 1 if keys and keys == self.last_input_keys else 0
            self.last_input_keys = keys
            # Several fast RPC retries may legitimately share one fresh frame.
            # Reuse alone, or a visually static scene, is never staleness proof.
            if bad:
                self.warnings["stale_vision"] = "102 视觉输入过期或跨多次推理重复使用旧帧，请检查 head/wrist 相机"
            else:
                self.warnings.pop("stale_vision", None)

    def response(self, epoch, *, success):
        with self.lock:
            if epoch != self.epoch or self.stopping:
                return  # Retired brokers' late replies cannot change indication.
            if success:
                self.chunks += 1
                self.good_chunks += 1
                self.failures = 0
                if self.good_chunks >= 2:
                    self.warnings.pop("inference_retry", None)
            else:
                self.good_chunks = 0
                self.failures += 1
                if self.ready and self.failures >= self.failure_count:
                    self.warnings["inference_retry"] = "102 推理连续失败，正在等待有效动作恢复"

    def queue_wait(self, waiting):
        with self.lock:
            if waiting and self.ready and not self.blocked:
                if self.queue_wait_at is None:
                    self.queue_wait_at = self.clock()
            else:
                self.queue_wait_at = None

    def terminal_error(self, error):
        with self.lock:
            self.fault = "102 原运行流程已报错停止：" + str(error)[:256]

    def snapshot(self, captures):
        with self.lock:
            now = self.clock()
            if self.fault:
                return "fault", "vla102_fault", self.fault
            if self.stopping:
                return "waiting", "vla102_stopped", "102 VLA 已停止或人工暂停"
            if self.blocked:
                if getattr(self, "blocked_reason", "") == "feedback":
                    return "initializing", "vla102_replanning", "102 反馈已验证恢复，等待联锁放行和新计划"
                return "normal", "vla102_interlock_wait", "102 正常等待 101 许可或保护避让"
            if not self.ready:
                return "initializing", "vla102_initializing", "102 VLA 正在预热或等待恢复后的新计划"
            if "repeated_motion" in self.warnings:
                return "warning", "vla102_repeated_motion", self.warnings["repeated_motion"]
            if self.queue_wait_at is not None and now - self.queue_wait_at >= self.starvation_s:
                return "warning", "vla102_action_starvation", "102 动作队列实际断供，正在等待推理动作恢复"
            if "inference_retry" in self.warnings:
                return "warning", "vla102_inference_retry", self.warnings["inference_retry"]
            dead_camera = any(not stamp or now - stamp[2] > self.frame_age_s for stamp in captures.values())
            if dead_camera:
                return "warning", "vla102_stale_vision", "102 相机采集帧停止更新或无法确认新鲜帧，请检查 head/wrist 相机"
            for code in ("stale_vision",):
                if code in self.warnings:
                    return "warning", "vla102_" + code, self.warnings[code]
            return "normal", "vla102_running", "102 VLA 正常运行"


class ObservedPolicy:
    def __init__(self, policy, monitor, epoch):
        self.inner, self.monitor, self.epoch = policy, monitor, epoch

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def infer(self, observation, **kwargs):
        epoch = self.epoch
        report_only(self.monitor.request, epoch, observation.get(META_KEY))
        clean = {key: value for key, value in observation.items() if key != META_KEY}
        try:
            result = self.inner.infer(clean, **kwargs)
        except Exception:
            report_only(self.monitor.response, epoch, success=False)
            raise
        report_only(self.monitor.response, epoch, success=isinstance(result, dict) and result.get("actions") is not None)
        return result


class ObservedBroker:
    def __init__(self, broker, monitor):
        self.inner, self.monitor = broker, monitor

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def infer(self, observation):
        try:
            queue = getattr(self.inner, "_action_queue", None)
            waiting = (queue.qsize() == 0 if queue is not None else
                       hasattr(self.inner, "_last_results") and self.inner._last_results is None)
            report_only(self.monitor.queue_wait, waiting)
        except Exception:
            pass
        try:
            result = self.inner.infer(observation)
        except Exception as error:
            report_only(self.monitor.terminal_error, error)
            raise
        finally:
            report_only(self.monitor.queue_wait, False)
        report_only(self.monitor.mark_ready)
        return result

    def warmup(self, observation):
        try:
            self.inner.warmup(observation)
        except Exception as error:
            report_only(self.monitor.terminal_error, error)
            raise
        if getattr(self.inner, "_warmup_done", False):
            report_only(self.monitor.mark_ready)

    def reset(self):
        epoch = report_only(self.monitor.reset_phase)
        try:
            client = getattr(self.inner, "_policy", None)
            if isinstance(client, ObservedPolicy) and epoch is not None:
                client.epoch = epoch
        except Exception:
            pass
        self.inner.reset()


class MonitoredEnvironment:
    def __init__(self, environment, monitor):
        self.inner, self.monitor = environment, monitor
        robot = environment._env.robot
        self.probes = {}
        for name in ("head", "wrist"):
            try:
                self.probes[name] = CameraStampProbe(robot.cameras[name])
            except Exception:
                self.probes[name] = None
        # Reuse the existing interlock node/executor; no new ROS thread.
        try:
            self.reporter = VlaStatusReporter(environment._env, monitor, self.probes)
        except Exception as error:
            self.reporter = None
            logging.getLogger(__name__).warning("102 alarm reporting unavailable: %s", error)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def get_observation(self):
        observation = self.inner.get_observation()
        try:
            observation[META_KEY] = {name: probe.snapshot(observation["images_raw"][name]) if probe else None
                                     for name, probe in self.probes.items()}
            self.monitor.observe(observation["state"])
        except Exception:
            pass
        return observation

    def apply_action(self, action):
        try:
            result = self.inner.apply_action(action)
        except Exception as error:
            report_only(self.monitor.terminal_error, error)
            raise
        if result is not False and not action.get("_interlock_hold"):
            try:
                self.monitor.target(action["actions"])
            except Exception:
                pass
        return result

    def request_stop(self):
        report_only(self.monitor.context, stopping=True)
        self.inner.request_stop()

    def disconnect(self):
        try:
            if self.reporter is not None:
                self.reporter.close()
        except Exception:
            pass
        self.inner.disconnect()


class VlaStatusReporter:
    def __init__(self, environment, monitor, probes):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        self.env, self.monitor, self.probes, self.message_type = environment, monitor, probes, String
        self.node = environment.turntable_lease._node
        self.logger = (self.node.get_logger() if hasattr(self.node, "get_logger")
                       else logging.getLogger(__name__))
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.node.create_publisher(String, STATE_TOPIC, qos)
        self.progress_subscription = self.node.create_subscription(
            String, "/cosmetic_pick_cycle_status", self._progress,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.session = uuid.uuid4().hex
        self.publish_lock = threading.RLock()
        self.closed = False
        self.final_sent = False
        self.sequence = 0
        self.last, self.last_at = None, float("-inf")
        self.last_diagnostic_at = float("-inf")
        self.timer = self.node.create_timer(0.5, self.publish)
        self.publish()

    def _progress(self, message):
        if message.data.startswith("101 turntable grasp/flip phase armed;"):
            self.monitor.progress()  # Fresh target accepted: progress evidence, not grasp proof.

    def publish(self):
        with self.publish_lock:
            if self.closed and self.final_sent:
                return
            self._publish_current()

    def _publish_current(self):
        try:
            self.monitor.context(blocked=self.env.policy_blocked,
                feedback_fault=(self.env._feedback_recovery_hold or self.env.robot.feedback_recovery_pending),
                stopping=(self.closed or self.env._stop_requested.is_set() or self.env._closing),
                reason=self.env._blocked_reason or "lease")
            state = self.monitor.snapshot({name: probe.snapshot() if probe else None for name, probe in self.probes.items()})
            now = time.monotonic()
            if state == self.last and now - self.last_at < 2.0:
                return
            evidence = report_only(self.monitor.repeat_diagnostics) or {}
            changed = state != self.last
            self.sequence += 1
            message = self.message_type()
            message.data = json.dumps(dict(source="vla102", session=self.session, seq=self.sequence,
                stamp_ns=time.time_ns(), state=state[0], code=state[1], reason=state[2],
                repeat=evidence), ensure_ascii=False)
            self.publisher.publish(message)
            self.last, self.last_at = state, now
            if changed or now - self.last_diagnostic_at >= 10.0:
                detail = (f"102 VLA monitor: state={state[0]}, code={state[1]}; {state[2]}; "
                          f"repeat={json.dumps(evidence, ensure_ascii=False)}")
                if state[0] == "fault":
                    self.logger.error(detail)
                elif state[0] == "warning":
                    self.logger.warning(detail)
                else:
                    self.logger.info(detail)
                self.last_diagnostic_at = now
        except Exception:
            pass  # No reporting failure is allowed to affect control.

    def close(self):
        with self.publish_lock:
            if self.closed:
                return
            self.closed = True
            report_only(self.monitor.context, stopping=True)
            self.last = None
            self._publish_current()
            self.final_sent = True
        report_only(self.timer.cancel)
        report_only(self.node.destroy_timer, self.timer)
        report_only(self.node.destroy_subscription, self.progress_subscription)
        report_only(self.node.destroy_publisher, self.publisher)
