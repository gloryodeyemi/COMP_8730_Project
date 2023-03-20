# Install and import the needed libraries
import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
from utils.prepare import *
from utils.translation import *
from utils.summarizer import Summarize


# Import and split dataset into train, test, and validation
file_loc = "Data/twitter_data.csv"
dataset = load_dataset("csv", data_files=file_loc)
print("---------- Dataset ----------")
print(dataset)


dataset = split_data(dataset)
print("---------- Updated dataset ----------")
print(dataset)
print("---------- Example output ----------")
print(dataset["train"][31])


# ### Language Identification
# Identify the language of each token in the tweet using the lingua library.
dataset = identified(dataset)
print("---------- Updated dataset ----------")
print(dataset)
print("---------- Example output ----------")
print(dataset["train"][31])


# ### Step 2: Code-switch detection
# Detect the language switch in the tweet using regular expression
dataset = detect(dataset)
print("---------- Updated dataset ----------")
print(dataset)
print("---------- Example output ----------")
print(dataset["train"][31])


# ### Step 3: Translation
# Translate each tweet using google translate
dataset = translate_tweet(dataset)
print("---------- Updated dataset ----------")
print(dataset)
print("---------- Example output ----------")
print(dataset["train"][31])


# #### Evaluate the performance of the translator using BLEU (Bilingual Evaluation Understudy) metric
# **We calculated the bleu score for each tweet and compute the average.**
dataset = computed_bleu_score(dataset)
print("---------- Updated dataset ----------")
print(dataset)
print("---------- Example output ----------")
print(dataset["train"][31])

bleu_avg(dataset)


# ### Step 4: Summarization
# create an instance of the Summarize class
summarize = Summarize()

# tokenize the data
tokenize_data = summarize.tokenize_data(dataset)

# load the model
model = summarize.load_model()

# data collator
collator = summarize.collate_data(model)

# define the model trainer
trainer = summarize.model_trainer(model, tokenize_data, collator)

# train the model
trainer.train()

# evaluate the model
trainer.evaluate()

# test the model
index = 16
output = summarize.test_model(dataset, index, trainer)
print("Original tweet: ", dataset["test"]["Tweets"][index])
print("Translated tweet: ", dataset["test"]["Translated_tweet"][index])
print("Generated summary: ", output)