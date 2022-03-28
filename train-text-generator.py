import torch
from datasets import load_metric
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, \
    EarlyStoppingCallback

import constants
import processing
from performant_trainer import PerformantTrainer
from textgeneratordataset import TextGeneratorDataset

MODEL = "t5-small"
OUTPUT = "text-generator-t5-small-30-3-4"

if __name__ == '__main__':
    print("Hello, World")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    print("loaded tokenizer")

    dataset = torch.load("scripts/text-generator-4.pickle")
    size = int(len(dataset) * 0.3)
    split_at = int(size * 0.8)
    print("split at ", split_at, "/", size)
    train_data = dataset[:split_at]
    eval_data = dataset[split_at:size]

    tokenizer.max_length = constants.TEXTTOKENIZER_SOURCE_LENGTH
    max_target_length = constants.TEXTTOKENIZER_TARGET_LENGTH
    print("tokenizer max source length: ", tokenizer.max_length)
    print("tokenizer max target length: ", max_target_length)

    full_train_dataset = TextGeneratorDataset(tokenizer, train_data, tokenizer.max_length, max_target_length)
    full_eval_dataset = TextGeneratorDataset(tokenizer, eval_data, tokenizer.max_length, max_target_length)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    print("initializing trainer")

    training_args = Seq2SeqTrainingArguments(OUTPUT)
    training_args.per_device_train_batch_size = 8
    training_args.per_device_eval_batch_size = 1
    training_args.eval_accumulation_steps = 100
    training_args.num_train_epochs = 30
    training_args.metric_for_best_model = 'bleu'
    training_args.learning_rate = 2e-4
    # training_args.learning_rate = 0.001
    training_args.eval_steps = 100
    training_args.save_steps = 4000

    trainer = PerformantTrainer(
        model=model,
        args=training_args,
        train_dataset=full_train_dataset,
        eval_dataset=full_eval_dataset,
        compute_metrics=processing.compute_bleu(tokenizer),
        tokenizer=tokenizer
    )

    print("training")
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()
