# coding:utf-8

from transformers import pipeline

#classifier = pipeline("sentiment-analysis")
#classifier("I've been waiting for a HuggingFace course my whole life.")

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")