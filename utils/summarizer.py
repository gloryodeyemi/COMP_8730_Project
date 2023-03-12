import transformers
import nltk
import torch
import evaluate
import numpy as np
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

class Summarize:
    # define the variables
    max_input = 512
    max_target = 128
    batch_size = 3
    model_checkpoints = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)
    rouge_metric = evaluate.load("rouge")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # preprocess the data
    def preprocess_data(self, data_to_process):
        # get all the translated tweets
        inputs = [tweet for tweet in data_to_process['Translated_tweet']]
        # tokenize the translated tweets
        model_inputs = self.tokenizer(inputs,  max_length=self.max_input, padding='max_length', truncation=True)
        # tokenize the summaries
        with self.tokenizer.as_target_tokenizer():
          targets = self.tokenizer(data_to_process['Summary'], max_length=self.max_target, padding='max_length', truncation=True)
          
        #set labels
        model_inputs['labels'] = targets['input_ids']

        #return the tokenized data
        return model_inputs

    def tokenize_data(self, dataset):
        tokenize_data = dataset.map(self.preprocess_data, batched = True)
        return tokenize_data

    def load_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoints)
        return model

    def collate_data(self, model):
        collator = DataCollatorForSeq2Seq(
        self.tokenizer,
        model=model,
        return_tensors="pt")
        return collator

    # define function for custom tokenization
    def tokenize_sentence(self, arg):
        encoded_arg = self.tokenizer(arg)
        return self.tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

    # define function for computing the rouge score
    def compute_rouge(self, eval_arg):
        preds, labels = eval_arg
        
          # Replace -100
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
          # Convert id tokens to text
        text_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        text_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        text_preds = [(p if p.endswith(("!", "！", "?", "？", ".")) else p + ".") for p in text_preds]
        text_labels = [(l if l.endswith(("!", "！", "?", "？", ".")) else l + ".") for l in text_labels]
        sent_tokenizer_c = RegexpTokenizer(u'[^!！?？.]*[!！?？.]')
        text_preds = ["\n".join(np.char.strip(sent_tokenizer_c.tokenize(p))) for p in text_preds]
        text_labels = ["\n".join(np.char.strip(sent_tokenizer_c.tokenize(l))) for l in text_labels]
        
          # compute ROUGE score with custom tokenization
        return self.rouge_metric.compute(
            predictions=text_preds,
            references=text_labels,
            tokenizer=self.tokenize_sentence
        )

    # set parameters for training the model
    def training_args(self, model):
        model.to(self.device)
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
        return args

    def model_trainer(self, model, tokenize_data, collator):
        trainer = Seq2SeqTrainer(
        model, 
        self.training_args(model),
        train_dataset=tokenize_data['train'],
        eval_dataset=tokenize_data['validation'],
        data_collator=collator,
        tokenizer=self.tokenizer,
        compute_metrics=self.compute_rouge
        )
        return trainer

    def test_model(self, dataset, index, trainer):
        #tokenize the conversation
        model_inputs = self.tokenizer(dataset["test"]["Translated_tweet"][index],  max_length=self.max_input, padding='max_length', truncation=True)
        #make prediction
        raw_pred, _, _ = trainer.predict([model_inputs])
        #decode the output
        output = self.tokenizer.decode(raw_pred[0])
        return output

