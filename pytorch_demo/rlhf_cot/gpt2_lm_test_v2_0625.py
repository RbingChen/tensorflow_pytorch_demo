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
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    print(logprobs_labels.shape)
    return logprobs_labels.squeeze(-1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "/Users/bing/Desktop/IDEA/model_weight/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

max_length = 128
input_txt = """In a shocking finding, scientist discovered \
a herd of unicorns living in a remote, previously unexplored \
valley, in the Andes Mountains. Even more surprising to the a\
researchers was the fact that the unicorns spoke perfect English.
"""
input_ids = tokenizer(input_txt, return_tensors="pt")
output_greedy = model.generate(**input_ids, max_length=max_length,
                               do_sample=False)
#print(input_ids)
#print(output_greedy.shape)
#print("input tokens shape:",input_ids["input_ids"].shape)
result = tokenizer.decode(output_greedy[0])
#print(len(result), len(input_txt))
print(result)


ref_logits = model(
    **input_ids,
    return_dict=True
).logits


print('ref_logits.shape ===> ', ref_logits.shape)
#print("ref_logits.shape:",ref_logits.shape) #torch.Size([1, 49, 50257]) 和输入有关
print('input_ids["input_ids"].shape ===> ', input_ids["input_ids"].shape)
tt = logprobs_of_labels(ref_logits[:, :-1, :], input_ids["input_ids"][:, 1:])

#print(tt)
print('tt.shape ===> ', tt.shape)
