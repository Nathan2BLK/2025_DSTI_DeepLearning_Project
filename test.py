import torch, os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

use_bf16 = (DEVICE == "cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = (DEVICE == "cuda") and (not use_bf16)

torch.backends.cuda.matmul.allow_tf32 = True   # speedup on Ampere+
torch.set_float32_matmul_precision("high")     # PyTorch 2.x matmul fast path
print("Device:", DEVICE, "| bf16:", use_bf16, "| fp16:", use_fp16)


from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

torch_dtype = (
    torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
)

config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")
model  = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch_dtype
)
# Trainer will move the model to CUDA automatically.
