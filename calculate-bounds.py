import torch
from transformers import AutoTokenizer

import processing

KEYMODEL = "keyword-generator-t5-small-30-3"
TEXTMODEL = "text-generator-t5-small-30-3"

# This script calculates the smallest possible input size to fit all inputs and all outputs.
# These values can be put into constants.py
if __name__ == '__main__':
    keytokenizer = AutoTokenizer.from_pretrained(KEYMODEL)
    texttokenizer = AutoTokenizer.from_pretrained(TEXTMODEL)

    print("loaded tokenizer")

    dataset = torch.load("scripts/keyword-generator-4.pickle")
    size = int(len(dataset) * 0.3)
    # size = int(len(dataset) * 0.01)
    split_at = int(size * 0.8)
    print("split at ", split_at, "/", size)
    train_data = dataset[:split_at]
    eval_data = dataset[split_at:size]

    keytokenizer.max_length = 0
    max_target_length = 0
    for tr in train_data:
        s = processing.process_sentences_for_keyword_generation(tr[2])
        keytokenizer.max_length = max(keytokenizer.max_length, len(keytokenizer.tokenize(s)))
        max_target_length = max(max_target_length,
                                len(keytokenizer.tokenize(" ".join(tr[3]), padding=False, truncation=True)))

    print("keyword tokenizer max length: ", keytokenizer.max_length)
    print("keyword tokenizer max target length: ", max_target_length)

    dataset = torch.load("scripts/text-generator-4.pickle")
    size = int(len(dataset) * 0.3)
    # size = int(len(dataset) * 0.01)
    split_at = int(size * 0.8)
    print("split at ", split_at, "/", size)
    train_data = dataset[:split_at]
    eval_data = dataset[split_at:size]

    keytokenizer.max_length = 0
    max_target_length = 0
    for tr in train_data:
        s = processing.process_sentences_for_text_generation(tr[2], tr[3])
        keytokenizer.max_length = max(keytokenizer.max_length, len(keytokenizer.tokenize(s)))
        max_target_length = max(max_target_length,
                                len(keytokenizer.tokenize(tr[4], padding=False, truncation=True)))

    print("text tokenizer max length: ", keytokenizer.max_length)
    print("text tokenizer max target length: ", max_target_length)
