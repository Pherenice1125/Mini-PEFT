import torch
import mini_peft

mini_peft.setup_logging("INFO")
base_model = "TinyLlama/TinyLlama_v1.1"

model = mini_peft.LLMModel.from_pretrained(
    base_model,
    device=mini_peft.executor.default_device_name(),
    load_dtype=torch.bfloat16,
    )
tokenizer = mini_peft.Tokenizer(base_model)

LoRA_config = mini_peft.LoraConfig(
    adapter_name="lora_0",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    )

# model.
