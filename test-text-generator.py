import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import constants
import processing

TEXTMODEL = "./text-generator-t5-small-30-3-4"


def get_text_from_model(sentences, keywords, max_length, tokenizer, model):
    text = processing.process_sentences_for_text_generation(sentences, keywords)
    tokenized = tokenizer(text, padding="max_length", max_length=max_length, return_tensors='pt')
    logits = model.generate(tokenized['input_ids'])[0]
    return tokenizer.decode(logits, skip_special_tokens=True)


if __name__ == '__main__':
    print("Hello, World")

    print("load tokenizers")
    text_tokenizer = AutoTokenizer.from_pretrained(TEXTMODEL)

    text_tokenizer.max_length = constants.TEXTTOKENIZER_SOURCE_LENGTH

    print("load models")
    text_model = AutoModelForSeq2SeqLM.from_pretrained(TEXTMODEL)

    dataset = torch.load("scripts/text-generator-4.pickle")

    size = int(len(dataset) * 0.5)
    split_at = int(size * 0.8)
    print("split at ", split_at, "/", size)
    train_data = dataset[:split_at]
    eval_data = dataset[split_at:size]

    results = []

    for sid, pid, sentences, next_keywords, next_sentence in train_data[:1000]:
        next_s = get_text_from_model(sentences, next_keywords, constants.TEXTTOKENIZER_SOURCE_LENGTH, text_tokenizer,
                                     text_model)
        next_t = text_tokenizer.tokenize(next_sentence)
        results.append((sid, pid, sentences, next_keywords, next_s, next_t, next_sentence))
        print("%s\n%s\n" % (next_s, next_sentence))
        if pid == 4:
            print()

    torch.save(results, "scripts/text-result-gt-train-4.pickle")

    results = []

    for sid, pid, sentences, next_keywords, next_sentence in eval_data[:1000]:
        next_s = get_text_from_model(sentences, next_keywords, constants.TEXTTOKENIZER_SOURCE_LENGTH, text_tokenizer,
                                     text_model)
        next_t = text_tokenizer.tokenize(next_sentence)
        results.append((sid, pid, sentences, next_keywords, next_s, next_t, next_sentence))
        print("%s\n%s\n" % (next_s, next_sentence))
        if pid == 4:
            print()

    torch.save(results, "scripts/text-result-gt-eval-4.pickle")
