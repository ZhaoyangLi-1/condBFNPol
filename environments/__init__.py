from environments.base import BaseEnv
from environments.maze import MazeEnv

__all__ = ["BaseEnv", "MazeEnv"]

# Optional AntMaze (requires gymnasium[robotics])
try:  # pragma: no cover - optional
    from environments.antmaze import AntMazeEnv

    __all__.append("AntMazeEnv")
except ImportError:
    AntMazeEnv = None

# Optional PushT (requires gymnasium and gym-pusht)
try:  # pragma: no cover - optional
    from environments.pusht import PushTEnv

    __all__.append("PushTEnv")
except ImportError:
    PushTEnv = None
