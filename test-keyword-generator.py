import sys

import torch
from datasets import load_metric
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, \
    EarlyStoppingCallback

import constants
import processing

KEYWORDMODEL = "./keyword-generator-t5-small-30-3-4"
# KEYWORDMODEL = "t5-small"


def get_keywords_from_model(text, tokenizer, model):
    tokenized = tokenizer(text, padding="max_length", return_tensors='pt')
    logits = model.generate(tokenized['input_ids'])[0]
    return tokenizer.decode(logits, skip_special_tokens=True)


if __name__ == '__main__':
    print("Hello, World")

    print("load tokenizers")
    keyword_tokenizer = AutoTokenizer.from_pretrained(KEYWORDMODEL)
    keyword_tokenizer.max_length = constants.KEYWORDTOKENIZER_SOURCE_LENGTH

    print("load models")
    keyword_model = AutoModelForSeq2SeqLM.from_pretrained(KEYWORDMODEL)

    dataset = torch.load("scripts/keyword-generator-4.pickle")
    size = int(len(dataset) * 0.5)
    split_at = int(size * 0.8)
    print("split at ", split_at, "/", size)
    train_data = dataset[:split_at]
    eval_data = dataset[split_at:size]

    results = []

    for sid, pid, sentences, next_keywords in train_data[:1000]:
        text = processing.process_sentences_for_keyword_generation(sentences)
        keywords = get_keywords_from_model(text, keyword_tokenizer, keyword_model).split()
        results.append((sid, pid, sentences, next_keywords, keywords))
        print("%s - %s" % (keywords, next_keywords))
        if pid == 4:
            print()

    torch.save(results, "scripts/keywords-result-train-4.pickle")

    results = []

    for sid, pid, sentences, next_keywords in eval_data[:1000]:
        text = processing.process_sentences_for_keyword_generation(sentences)
        keywords = get_keywords_from_model(text, keyword_tokenizer, keyword_model).split()
        results.append((sid, pid, sentences, next_keywords, keywords))
        print("%s - %s" % (keywords, next_keywords))
        if pid == 4:
            print()

    torch.save(results, "scripts/keywords-result-eval-4.pickle")
