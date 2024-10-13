from dataclasses import dataclass, field
from typing import List


@dataclass
class LoraTragets:
    targets: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])