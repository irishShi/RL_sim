"""工具模块"""
from .replay_buffer import ReplayBuffer, NStepBuffer
from .c51_projection import project_distribution
from .action_hold import ActionHoldController, compute_action_hold_steps
from .scenario_profiles import ScenarioProfileSampler

# 可选导入（避免循环依赖）
try:
    from .dataset_loader import OfflineDataset
    __all__ = [
        'ReplayBuffer', 'NStepBuffer', 'project_distribution', 'OfflineDataset',
        'ActionHoldController', 'compute_action_hold_steps', 'ScenarioProfileSampler'
    ]
except ImportError:
    __all__ = [
        'ReplayBuffer', 'NStepBuffer', 'project_distribution',
        'ActionHoldController', 'compute_action_hold_steps', 'ScenarioProfileSampler'
    ]
