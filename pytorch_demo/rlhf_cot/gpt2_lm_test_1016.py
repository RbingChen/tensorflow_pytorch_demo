# coding:utf-8

import torch

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging
import torch.nn.functional as F

logging.set_verbosity_warning()

from transformers import AutoTokenizer, AutoModelForCausalLM


def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(0))
    print(logprobs_labels.shape)
    return logprobs_labels.squeeze(-1)


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "/Users/bing/Desktop/IDEA/model_weight/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

max_length = 128
input_txt = """In a shocking finding, scientist discovered"""
input_ids = tokenizer(input_txt, return_tensors="pt")
# print(input_ids["input_ids"])
print('len(input_ids["input_ids"]==> {}'.format(input_ids["input_ids"].shape))
ref_logits = model(
    **input_ids
).logits

labels = torch.tensor([11444]).unsqueeze(0)

print(ref_logits.shape)
# print("ref_logits.shape:",ref_logits.shape) #torch.Size([1, 49, 50257]) 和输入有关
print(ref_logits[:, -1:, :].shape)

tt = logprobs_of_labels(ref_logits[:, -1:, :], labels=labels)

print(tt.detach().numpy())
print(tt.shape)

input_ids = tokenizer(input_txt, return_tensors="pt")
output_greedy = model.generate(**input_ids, max_length=max_length,
                               do_sample=False)

print(output_greedy.shape)
#print(tokenizer.decode(output_greedy[0]))
