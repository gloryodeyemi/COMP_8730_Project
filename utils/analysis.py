get_ipython().system('pip install lingua-language-detector==1.1.0')

# import all the needed libraries
import nltk
import pandas as pd
import string
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from lingua import Language, LanguageDetectorBuilder

# download the needed modules
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# function to tokenize the tweets into sentences
def tokenize_sent(text):
    sent_tokens = sent_tokenize(text.lower())
    return sent_tokens

# function to count the number of sentences in the text
def count_sents(text):
    sent_count = len(tokenize_sent(text))
    return sent_count

# function to count the number of characters in the text
def count_chars(text):
    char_count = len(text.replace(" ",""))
    return char_count

# function to count the number of punctuation in the text
def count_punc(text):
    punc_count = len([a for a in text if a in string.punctuation])
    return punc_count

# function to remove punctuations in the text
def remove_punctuation(text):
    without_punc = "".join([i for i in text if i not in string.punctuation])
    return without_punc

# function for tokenizing the text into words
def tokenize_text(text):
    tokens = word_tokenize(text.lower())
    return tokens

# function to count the number of words in the text
def count_words(text):
    word_count = len(tokenize_text(text))
    return word_count

# function to calculate the word density in the text
def word_density(data, colname1, colname2):
    w_density = data[colname1] / (data[colname2] + 1)
    return w_density

# function to calculate the sentence density in the text
def sent_density(data, colname1, colname2):
    s_density = data[colname1] / (data[colname2] + 1)
    return s_density

# function to display sum of values in all numeric columns
def data_sum(data):
    print("Sum of values in numerical columns")
    print("----------------------------------")
    print(data.sum(axis=0, numeric_only = True))

# function to display average of values in all numeric columns
def data_average(data):
    print("Average of values in numerical columns")
    print("--------------------------------------")
    print(data.mean(axis=0, numeric_only = True))

"""
    function to get number of english and yoruba words from the code-switch tweets 
    using the lingua-language-detector library
"""
def lang_count(data, col):
    languages = [Language.ENGLISH, Language.YORUBA]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    count_eng = 0
    count_yor = 0

    for row in data[col]:
        for word in row:
            lang = detector.detect_language_of(word)
            if (lang == Language.YORUBA):
                count_yor += 1
            else:
                count_eng += 1

    print("Number of Yoruba words: ", count_yor)
    print("Number of English words: ", count_eng)
    print(f"Percentage of Yoruba words: {((count_yor/(count_yor + count_eng)) * 100):.2f}%")
    print(f"Percentage of English words: {((count_eng/(count_yor + count_eng)) * 100):.2f}%")
    
# function to get number of english and yoruba words by comparing words in the tweet to words in the english text
def lang_count_comp(data, col1, col2):
    count_eng = 0
    count_yor = 0

    for ind in data.index:
        list_1 = data[col1][ind]
        list_2 = data[col2][ind]
        for tok in list_1:
            if (tok in list_2):
                count_eng += 1
            else:
                count_yor += 1

    print("Number of Yoruba words: ", count_yor)
    print("Number of Eng words: ", count_eng)
    print(f"Percentage of Yoruba words: {((count_yor/(count_yor + count_eng)) * 100):.2f}%")
    print(f"Percentage of English words: {((count_eng/(count_yor + count_eng)) * 100):.2f}%")

# function to get number of unique words
def get_unique_words(data, col, alias):
    results = set()
    data[col].apply(results.update)
    print(f"Number of unique words in {alias} = {len(results)}")

# function to get number of unique sentences
def get_unique_sents(data, col, alias):
    results = set()
    data[col].apply(results.update)
    print(f"Number of unique sentences in {alias} = {len(results)}")

