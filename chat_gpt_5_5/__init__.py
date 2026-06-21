"""Source-guided ARC-AGI-3 public-set solver."""
"""Game-agnostic ARC-AGI-3 agent.

The policy consumes only observations and a deliberately narrow optional
black-box simulator protocol.  It contains no public game identifiers, source
readers, recorded action traces, or environment-specific policy branches.
"""

from .agent import AgentConfig, GeneralistAgent
from .core import Action, Observation

__all__ = ["Action", "Observation", "AgentConfig", "GeneralistAgent"]
