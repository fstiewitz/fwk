import sys

import torch
from datasets import load_metric
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, \
    EarlyStoppingCallback

import constants
import processing


def process_sentence_for_text(sentences, keywords):
    return "summarize: %s %s" % (sentences, " ".join(keywords))


def process_sentence_for_keywords(sentences):
    return "summarize: " + sentences


KEYWORDMODEL = "./keyword-generator-t5-small-30-3-4"
TEXTMODEL = "./text-generator-t5-small-30-3-4"


def get_keywords_from_model(text, tokenizer, model):
    tokenized = tokenizer(text, padding="max_length", return_tensors='pt')
    logits = model.generate(tokenized['input_ids'])[0]
    return tokenizer.decode(logits, skip_special_tokens=True).split()


def get_text_from_model(sentences, keywords, max_length, tokenizer, model):
    text = processing.process_sentences_for_text_generation(sentences, keywords)
    tokenized = tokenizer(text, padding="max_length", max_length=max_length, return_tensors='pt')
    logits = model.generate(tokenized['input_ids'])[0]
    return tokenizer.decode(logits, skip_special_tokens=True)


if __name__ == '__main__':
    print("load tokenizers")
    keyword_tokenizer = AutoTokenizer.from_pretrained(KEYWORDMODEL)
    text_tokenizer = AutoTokenizer.from_pretrained(TEXTMODEL)
    keyword_tokenizer.max_length = constants.KEYWORDTOKENIZER_SOURCE_LENGTH
    text_tokenizer.max_length = constants.TEXTTOKENIZER_SOURCE_LENGTH

    print("load models")
    keyword_model = AutoModelForSeq2SeqLM.from_pretrained(KEYWORDMODEL)
    text_model = AutoModelForSeq2SeqLM.from_pretrained(TEXTMODEL)

    while True:
        print("Enter first sentence, empty for quit:")
        text = sys.stdin.readline().strip()
        if len(text) == 0:
            break
        if text[-1] != '.':
            text += '.'
        for i in range(5):
            keywords = get_keywords_from_model(text, keyword_tokenizer, keyword_model)
            next_sentence = get_text_from_model(text, keywords, constants.TEXTTOKENIZER_SOURCE_LENGTH, text_tokenizer, text_model)
            text += next_sentence + ('.' if next_sentence[-1] != '.' else '')
            print(next_sentence)
