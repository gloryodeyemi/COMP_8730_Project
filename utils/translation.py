from googletrans import Translator
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from statistics import mean

def translate_tweet(dataset):
    translator = Translator()
    dataset = dataset.map(lambda x: {"Translated_tweet": translator.translate(x['Tweets'], src='yo', dest='en').text})
    return dataset

def compute_bleu_score(predictions, references):
    smoothing = SmoothingFunction()
    # Tokenize the predictions and references
    predictions = [prediction.split() for prediction in predictions]
    references = [[reference.split()] for reference in references]

    # Compute the BLEU score
    bleu_score = corpus_bleu(references, predictions, smoothing_function=smoothing.method2)

    return bleu_score

def computed_bleu_score(dataset):
    dataset = dataset.map(lambda x: {"Bleu_score": compute_bleu_score([x['Translated_tweet']], [x['Eng_source']])})
    return dataset

def bleu_avg(dataset):
    bleu = mean(dataset["train"]["Bleu_score"])
    print(f"Bleu score: {bleu:.4f}")

