#!/usr/bin/env python
# coding: utf-8

# ## Install and import the needed libraries
import transformers
import nltk
import pandas as pd
import numpy as np
import torch
import evaluate
import os
import re
from googletrans import Translator
from lingua import Language, LanguageDetectorBuilder
from nltk.tokenize import sent_tokenize, word_tokenize
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from nltk.tokenize import RegexpTokenizer
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


nltk.download('punkt')


# ## Import and split dataset into train, test, and validation

file_loc = filepath
dataset = load_dataset("csv", data_files=file_loc)
print("---------- Dataset ----------")
print(dataset)

datasets_train_test = dataset["train"].train_test_split(test_size=0.1, seed=0)
datasets_train_validation = datasets_train_test["train"].train_test_split(test_size=0.1, seed=0)

dataset["train"] = datasets_train_validation["train"]
dataset["test"] = datasets_train_test["test"]
dataset["validation"] = datasets_train_validation["test"]

print("---------- Updatad dataset ----------")
print(dataset)
print("---------- Example output ----------")
print(dataset["train"][4])


# ### Language Identification
# Identify the language of each token in the tweet using the lingua library.


def identify(tweet):
  languages = [Language.ENGLISH, Language.YORUBA]
  detector = LanguageDetectorBuilder.from_languages(*languages).build()
  lang_list = []
  for word in tweet:
    lang = detector.detect_language_of(word)
    if (lang == None):
      lang_list.append(None)
    else:
      lang_list.append(lang.name)
  return lang_list

dataset = dataset.map(lambda x: {"Language": identify(x['Tweets'])})
print("---------- Updatad dataset ----------")
print(dataset)
print("---------- Example output ----------")
print(dataset["train"][4])


# ### Step 2: Code-switch detection
# Detect the language switch in the tweet using regular expression

def detect(tweet):
  return re.findall(r'\b\w+\b', tweet)
  
dataset = dataset.map(lambda x: {"Code_switches": detect(x['Tweets'])})
print("---------- Updatad dataset ----------")
print(dataset)
print("---------- Example output ----------")
print(dataset["train"][4])


# ### Step 3: Translation
# Translate each tweet using google translate

def translate_tweet(tweet):
  return translator.translate(tweet, src='yo', dest='en').text

translator = Translator()
dataset = dataset.map(lambda x: {"Translated_tweet": translate_tweet(x['Tweets'])})
print("---------- Updatad dataset ----------")
print(dataset)
print("---------- Example output ----------")
print(dataset["train"][4])


# #### Evaluate the performance of the translator using BLEU (Bilingual Evaluation Understudy) metric

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_bleu_score(predictions, references):
    smoothing = SmoothingFunction()
    # Tokenize the predictions and references
    predictions = [prediction.split() for prediction in predictions]
    references = [[reference.split()] for reference in references]

    # Compute the BLEU score
    bleu_score = corpus_bleu(references, predictions, smoothing_function=smoothing.method2)

    return bleu_score

# def compute_bleu_score(predictions, references):
#     # Tokenize the predictions and references
#     # predictions = [prediction.split() for prediction in predictions]
#     # references = [[reference.split()] for reference in references]

#     predictions = [predictions]
#     references = [[references]]

#     bleu = evaluate.load("bleu")
#     bleu_score = bleu.compute(predictions=predictions, references=references)

#     return bleu_score


# **We calculated the bleu score for each tweet and compute the average.**

# dataset = dataset.map(lambda x: {"Bleu_score": compute_bleu_score([x['Translated_tweet']], [x['Eng_source']])})
dataset = dataset.map(lambda x: {"Bleu_score": compute_bleu_score([x['Translated_tweet']], [x['Eng_source']])})
print("---------- Updatad dataset ----------")
print(dataset)
print("---------- Example output ----------")
print(dataset["train"][4])

from statistics import mean

# bleu = mean(dataset["train"]["Bleu_score"])
bleu = mean(dataset["train"]["Bleu_score"])
print(f"Bleu score: {bleu:.4f}")


# ### Step 4: Summarization
# Fine-tune the BART model for summarization and evaluate its performance using the ROUGE metrics.

# define the variables
max_input = 512
max_target = 128
batch_size = 3
model_checkpoints = "facebook/bart-base"

# toenize the data
tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)

# preprocess the data
def preprocess_data(data_to_process):
  # get all the translated tweets
  inputs = [tweet for tweet in data_to_process['Translated_tweet']]
  # tokenize the translated tweets
  model_inputs = tokenizer(inputs,  max_length=max_input, padding='max_length', truncation=True)
  # tokenize the summaries
  with tokenizer.as_target_tokenizer():
    targets = tokenizer(data_to_process['Summary'], max_length=max_target, padding='max_length', truncation=True)
    
  #set labels
  model_inputs['labels'] = targets['input_ids']

  #return the tokenized data
  return model_inputs

tokenize_data = dataset.map(preprocess_data, batched = True)

# load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints)

collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    return_tensors="pt")

rouge_metric = evaluate.load("rouge")

# define function for custom tokenization
def tokenize_sentence(arg):
    encoded_arg = tokenizer(arg)
    return tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

# define function for computing the rouge score
def compute_rouge(eval_arg):
    preds, labels = eval_arg
    
      # Replace -100
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
      # Convert id tokens to text
    text_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    text_preds = [(p if p.endswith(("!", "！", "?", "？", ".")) else p + ".") for p in text_preds]
    text_labels = [(l if l.endswith(("!", "！", "?", "？", ".")) else l + ".") for l in text_labels]
    sent_tokenizer_c = RegexpTokenizer(u'[^!！?？.]*[!！?？.]')
    text_preds = ["\n".join(np.char.strip(sent_tokenizer_c.tokenize(p))) for p in text_preds]
    text_labels = ["\n".join(np.char.strip(sent_tokenizer_c.tokenize(l))) for l in text_labels]
    
      # compute ROUGE score with custom tokenization
    return rouge_metric.compute(
        predictions=text_preds,
        references=text_labels,
        tokenizer=tokenize_sentence
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# set parameters for training the model
args = Seq2SeqTrainingArguments(
    'code-switch-summ', 
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size= 2,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    eval_accumulation_steps=3,
    # fp16=True ,
    seed = 42
    )

trainer = Seq2SeqTrainer(
    model, 
    args,
    train_dataset=tokenize_data['train'],
    eval_dataset=tokenize_data['validation'],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_rouge
)

# train the model
trainer.train()


# ### Test fine-tuned model on Test data

#tokenize the conversation
model_inputs = tokenizer(dataset["test"]["Translated_tweet"][6],  max_length=max_input, padding='max_length', truncation=True)
#make prediction
raw_pred, _, _ = trainer.predict([model_inputs])
#decode the output
output = tokenizer.decode(raw_pred[0])
print("Original tweet: ", dataset["test"]["Tweets"][6])
print("Translated tweet: ", dataset["test"]["Translated_tweet"][6])
print("Generated summary: ", output)

