from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class DataGeneratorConfig:
    amount: int = 3
    raw_data_dir: Path = Path("./datasets") 
    image_resize: bool = True
    variation: int = 14
    task_names = ["pick_up_cup"]
    image_size = [224, 224]
    