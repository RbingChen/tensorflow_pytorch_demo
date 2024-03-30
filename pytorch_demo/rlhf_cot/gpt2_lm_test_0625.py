# coding:utf-8

import torch

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging

logging.set_verbosity_warning()


from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

input_txt = "Transformers are the"
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
iterations = []
n_steps = 8 # 进行8步解码
choices_per_step = 5 # 每一步候选数量

with torch.no_grad():# eval模式
    for _ in range(n_steps):# 每步解码
        iteration = dict()
        iteration["Input"] = tokenizer.decode(input_ids[0]) # 提示文本
        output = model(input_ids=input_ids) # 将提示文本输入到模型进行解码
        # Select logits of the first batch and the last token and apply softmax
        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
        # Store tokens with highest probabilities
        for choice_idx in range(choices_per_step): # 概率最大的五个token
            token_id = sorted_ids[choice_idx]
            token_prob = next_token_probs[token_id].cpu().numpy()
            token_choice = (
                f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)" # 取百分号两位数
            )
            iteration[f"Choice {choice_idx+1}"] = token_choice
        # Append predicted next token to input
        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1) # 将概率最大的字符拼接到提示文本
        iterations.append(iteration)

input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
output = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)
print(tokenizer.decode(output[0]))