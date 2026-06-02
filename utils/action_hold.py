"""动作保持工具：统一训练、数据采集和评估中的 Hys/TTT 生效时长。"""
import math
from typing import Callable, Optional


def compute_action_hold_steps(
    action: int,
    action_space,
    delta_t_s: float,
    fixed_hold_steps: int = 0,
    min_steps: int = 6,
    max_steps: int = 20,
) -> int:
    """
    根据动作对应的 TTT 计算动作保持步数。

    Hys/TTT 是一段时间内生效的切换参数，而不是瞬时控制量。保持时长至少覆盖
    TTT 所需步数并留出少量余量，避免参数频繁变化导致 TTT 计时器反复清零。
    """
    if fixed_hold_steps and fixed_hold_steps > 0:
        return max(1, int(fixed_hold_steps))

    _, ttt_ms = action_space.action_to_hys_ttt(int(action))
    dt_ms = max(float(delta_t_s) * 1000.0, 1e-6)
    ttt_steps = int(math.ceil(float(ttt_ms) / dt_ms))
    return int(min(max(int(min_steps), ttt_steps + 2), int(max_steps)))


class ActionHoldController:
    """在 hold 耗尽时重新决策，否则复用上一组 Hys/TTT 动作。"""

    def __init__(
        self,
        action_space,
        delta_t_s: float,
        fixed_hold_steps: int = 0,
        min_steps: int = 6,
        max_steps: int = 20,
    ):
        self.action_space = action_space
        self.delta_t_s = float(delta_t_s)
        self.fixed_hold_steps = int(fixed_hold_steps or 0)
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self.current_action: Optional[int] = None
        self.remaining_steps = 0

    def reset(self):
        self.current_action = None
        self.remaining_steps = 0

    def select(self, select_action_fn: Callable[[], int]) -> int:
        """返回本步应执行的动作；必要时调用 select_action_fn 产生新动作。"""
        if self.remaining_steps <= 0 or self.current_action is None:
            self.current_action = int(select_action_fn())
            self.remaining_steps = compute_action_hold_steps(
                self.current_action,
                self.action_space,
                self.delta_t_s,
                fixed_hold_steps=self.fixed_hold_steps,
                min_steps=self.min_steps,
                max_steps=self.max_steps,
            )

        action = int(self.current_action)
        self.remaining_steps -= 1
        return action

