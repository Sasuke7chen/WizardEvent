from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LoraArguments:
    max_seq_length: int = field(metadata={"help": "Max length of the input sequence"})
    train_file: str = field(metadata={"help": "Path to the training data"})
    model_name_or_path: str = field(metadata={"help": "Path to the model checkpoint"})
    eval_file: Optional[str] = field(default="", metadata={"help": "Path to the evaluate data"})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=32, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
    
@dataclass
class SFTArguments:
    max_seq_length: int = field(metadata={"help": "Max length of the input sequence"})
    train_file: str = field(metadata={"help": "Path to the training data"})
    model_name_or_path: str = field(metadata={"help": "Path to the model checkpoint"})
    