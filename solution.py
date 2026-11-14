"""Your team's solution. At the end of the hackathon you pass this class to evaluate(Solution)
from evaluate.py, which creates one Solution and replays every 2017 booking request through it.

RULES -- Downtown Hotel has 124 rooms, Seaside Hotel 103. A guest who shows up earns
nightly_rate x nights. When a hotel is full at check-in, the cheapest arriving guests are
walked: they earn nothing and cost 2.5 x their stay value. A seated guest keeps the room.
Your result is the net revenue: revenue of seated guests - walk penalties.

Reading any 2017 data yourself is cheating.
"""
import sys
from pathlib import Path

# The working solution lives in henri_logicstical_regression/. Its modules import
# each other as `src.*`, so that folder has to be on the import path.
_PROJECT = Path(__file__).resolve().parent / "henri_logicstical_regression"
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.solution import Solution  # noqa: E402

__all__ = ["Solution"]
