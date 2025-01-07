import sys
import torch
import os
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import make_prompt
def setup():
    BASE_MODEL = "VietAI/gpt-neo-1.3B-vietnamese-news"
    PEFT_WEIGHTS = "/chatbot-rl"
    load_in_8bit = False

    if torch.cuda.is_available():
        device = "cuda"
        device_map = {'': 0}
        if load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)
            model = PeftModel.from_pretrained(model, PEFT_WEIGHTS, torch_dtype=torch.float16, device_map=device_map)
        else:
            model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map=device_map)
            model = PeftModel.from_pretrained(model, PEFT_WEIGHTS, device_map=device_map)
    else:
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(model, PEFT_WEIGHTS)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model.eval()
    if torch.__version__ >= "2": # tăng tốc
        model = torch.compile(model)
    return model, tokenizer, device