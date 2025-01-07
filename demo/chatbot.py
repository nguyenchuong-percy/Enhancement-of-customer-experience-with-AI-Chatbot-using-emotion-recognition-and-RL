#!/usr/bin/python3
import sys
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import make_prompt
from setup import setup

model, tokenizer, device = setup()

def get_answer(q, max_new_tokens=196, skip_tl=False):
    input_ids = tokenizer(make_prompt(q), return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        gen_tokens = model.generate(
            input_ids=input_ids,
            max_length=len(input_ids) + max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_k=20,
            repetition_penalty=1.2,
            eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.eos_token_id,
        )
    origin_output = tokenizer.batch_decode(gen_tokens)[0]
    output = origin_output.split("###")[2]
    try:
        k = output.index(":")
        if k < 10: output = output[k+1:]
    except:
        output = output
    print(f"\n- - -{origin_output}- - -\n")
    return output.strip()

def main():
    print("\n")
    while True:
        query = input("\nBạn：")
        print(f"Bot: {get_answer(query)}")

if __name__ == "__main__": main()