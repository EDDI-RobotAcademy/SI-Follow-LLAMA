from dataclasses import dataclass


@dataclass
class LoraArgs:
    r: int = 4
    alpha: int = 8
    dropout: float = 0.01
