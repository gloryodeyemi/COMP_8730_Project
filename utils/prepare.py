import re
from lingua import Language, LanguageDetectorBuilder

def split_data(dataset):
    datasets_train_test = dataset["train"].train_test_split(test_size=0.2, seed=0)
    datasets_train_validation = datasets_train_test["train"].train_test_split(test_size=0.2, seed=0)

    dataset["train"] = datasets_train_validation["train"]
    dataset["test"] = datasets_train_test["test"]
    dataset["validation"] = datasets_train_validation["test"]

    return dataset

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

def identified(dataset):
    dataset = dataset.map(lambda x: {"Language": identify(x['Tweets'])})
    return dataset

def detect_code_switch(tweet):
    # Define a regular expression pattern to match English and Yoruba words
    en_pattern = r'[A-Za-z]+'
    yo_pattern = r'[aàáèéiìíoòóuùúeẹẹ́ọọ́gbṣsAEẸIOỌU]+'

    # Split the tweet into words
    words = tweet.split()

    # Initialize a list to store the detected code-switches
    code_switches = []

    # Loop through each pair of adjacent words
    for i in range(len(words) - 1):
        # Check if the current word and the next word belong to different languages
        if re.match(en_pattern, words[i]) and re.match(yo_pattern, words[i+1]):
            code_switches.append((words[i], words[i+1]))
        elif re.match(yo_pattern, words[i]) and re.match(en_pattern, words[i+1]):
            code_switches.append((words[i], words[i+1]))

    # Return the list of code-switches
    return code_switches

def detect(dataset):
    dataset = dataset.map(lambda x: {"Code_switches": detect_code_switch(x['Tweets'])})
    return dataset

# def detect(dataset):
#     dataset = dataset.map(lambda x: {"Code_switches": re.findall(r'\b\w+\b', x['Tweets'])})
#     return dataset