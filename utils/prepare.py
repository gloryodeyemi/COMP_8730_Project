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

def detect(dataset):
    dataset = dataset.map(lambda x: {"Code_switches": re.findall(r'\b\w+\b', x['Tweets'])})
    return dataset 