"""Response cache package for ModelMesh.

Provides prompt-answer caching (exact and, optionally, semantic) in front of
the routing + provider call so repeated or near-identical prompts are served
from storage at zero API cost.
"""

__all__: list[str] = []
