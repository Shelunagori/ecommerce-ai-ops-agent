"""Loop bounds. Set only from trusted application configuration - never from model input."""

from dataclasses import dataclass

from app.core.config import Settings, get_settings

HARD_MAX_MODEL_ROUNDS = 10
HARD_MAX_TOOL_CALLS = 20
HARD_MAX_TOOL_CALLS_PER_TURN = 8


@dataclass(frozen=True)
class AssistantLimits:
    max_model_rounds: int = 5
    max_tool_calls: int = 8
    max_tool_calls_per_turn: int = 4

    def __post_init__(self) -> None:
        for name, value, cap in (
            ("max_model_rounds", self.max_model_rounds, HARD_MAX_MODEL_ROUNDS),
            ("max_tool_calls", self.max_tool_calls, HARD_MAX_TOOL_CALLS),
            ("max_tool_calls_per_turn", self.max_tool_calls_per_turn, HARD_MAX_TOOL_CALLS_PER_TURN),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= cap:
                raise ValueError(f"{name} must be an integer between 1 and {cap}")

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "AssistantLimits":
        s = settings or get_settings()
        return cls(
            max_model_rounds=s.assistant_max_model_rounds,
            max_tool_calls=s.assistant_max_tool_calls,
            max_tool_calls_per_turn=s.assistant_max_tool_calls_per_turn,
        )
