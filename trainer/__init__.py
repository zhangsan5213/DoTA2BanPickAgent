"""BP Agent Trainer Module"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from .bp_agent_trainer import BPAgentTrainer

__all__ = ["BPAgentTrainer"]
