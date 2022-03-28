import os

import torch
from datasets import load_metric
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, \
    EarlyStoppingCallback, DataCollatorForSeq2Seq

import constants
import processing
from performant_trainer import PerformantTrainer
from textgeneratordataset import TextGeneratorDataset

MODEL = "./text-generator-t5-small-30-3-4"


def process_model(modelname, dataset):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModelForSeq2SeqLM.from_pretrained(modelname)

    tokenizer.max_length = constants.KEYWORDTOKENIZER_SOURCE_LENGTH
    max_target_length = constants.KEYWORDTOKENIZER_TARGET_LENGTH

    full_eval_dataset = TextGeneratorDataset(tokenizer, dataset, tokenizer.max_length, max_target_length)

    training_args = Seq2SeqTrainingArguments("eval-keywords")
    training_args.per_device_train_batch_size = 16
    training_args.per_device_eval_batch_size = 4
    # training_args.eval_accumulation_steps = 1
    # training_args.num_train_epochs = 1
    training_args.eval_accumulation_steps = 100
    training_args.num_train_epochs = 4
    training_args.load_best_model_at_end = True
    # training_args.learning_rate = 2e-4
    training_args.learning_rate = 0.001
    training_args.metric_for_best_model = 'f1'
    training_args.eval_steps = 10
    # training_args.eval_steps = 10
    training_args.save_steps = 4000

    trainer = PerformantTrainer(
        model=model,
        args=training_args,
        eval_dataset=full_eval_dataset,
        compute_metrics=processing.compute_bertscore(tokenizer),
        tokenizer=tokenizer
    )
    print("evaluating", modelname)
    return trainer.evaluate()


def numkey(x):
    s = x.split("-")[-1]
    if s.isnumeric():
        return int(s)
    return 0


if __name__ == '__main__':
    keyword_data = torch.load("scripts/keywords-result-eval-4.pickle")
    dataset_base = torch.load("scripts/text-generator-4.pickle")
    dataset_map = dict([((x[0], x[1]), x) for x in dataset_base])
    dataset_out = {}
    for kw in keyword_data:
        (sid, pid, text, _kw, next_sentence) = dataset_map[(kw[0], kw[1])]
        dataset_out[(sid, pid)] = (sid, pid, text, kw[4], next_sentence)
    dataset = list(dataset_out.values())
    results = []
    testing = ["%s/%s" % (MODEL, x) for x in sorted(os.listdir(MODEL), key=numkey) if x.startswith("checkpoint-")]
    testing.append(MODEL)
    print(testing)
    #for p in testing:
    #    results.append((p, process_model(p, eval_data)))
    results.append((MODEL, process_model(MODEL, dataset)))

    for i, (p, f) in enumerate(results):
        print(i, p, f)
