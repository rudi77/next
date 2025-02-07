from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class SwiftTrainingConfig(BaseModel):
    model_type: str
    model_id_or_path: str
    num_train_epochs: int = 1
    sft_type: str = "lora"
    dataset: str
    val_dataset: Optional[str] = None
    batch_size: int = 4
    eval_steps: int = 150
    max_new_tokens: int = 128
    gradient_accumulation_steps: int = 4
    gpu_indices: List[int] = []
    
    # Environment variables
    size_factor: int = 8
    max_pixels: int = 602112
    
    def get_command(self) -> str:
        """Generate the swift command for training"""
        gpu_str = ','.join(map(str, self.gpu_indices))
        nproc = len(self.gpu_indices)
        
        cmd = [
            f"SIZE_FACTOR={self.size_factor}",
            f"MAX_PIXELS={self.max_pixels}",
            f"CUDA_VISIBLE_DEVICES={gpu_str}",
            f"NPROC_PER_NODE={nproc}",
            "swift", "sft",
            f"--model_type {self.model_type}",
            f"--model_id_or_path {self.model_id_or_path}",
            f"--num_train_epochs {self.num_train_epochs}",
            f"--sft_type {self.sft_type}",
            f"--dataset {self.dataset}",
            f"--batch_size {self.batch_size}",
            f"--eval_steps {self.eval_steps}",
            f"--max-new-tokens {self.max_new_tokens}",
            f"--gradient_accumulation_steps {self.gradient_accumulation_steps}"
        ]
        
        if self.val_dataset:
            cmd.append(f"--val_dataset {self.val_dataset}")
            
        return " ".join(cmd) 