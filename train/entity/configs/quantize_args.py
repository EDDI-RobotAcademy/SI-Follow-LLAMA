from dataclasses import dataclass

import torch


@dataclass
class QuantizeArgs:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
