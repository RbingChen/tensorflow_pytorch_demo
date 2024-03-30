# coding:utf-8

import torch

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging

logging.set_verbosity_warning()
# ls ~/.cache/huggingface/hub 缓存目录
#classifier = pipeline('sentiment-analysis', model="gpt2")
#result = classifier("I hate you")[0]
#print(result)


model = AutoModelForSequenceClassification.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

input = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(input)

pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)


output = model(**pt_batch)

print(output)


