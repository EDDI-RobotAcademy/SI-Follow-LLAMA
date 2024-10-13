from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelCofig:
    model_id: str = None
    torch_dtype: torch.dtype = torch.bfloat16
    max_length: int = 8192
