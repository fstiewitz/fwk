from datasets import load_metric


def process_sentences_for_keyword_generation(sentences):
    return "summarize: %s" % sentences


def process_sentences_for_text_generation(sentences, keywords):
    return "summarize: %s keywords: %s. next: " % (sentences, " ".join(keywords))


bleu = load_metric("bleu")
bertscore = load_metric("bertscore")
# metric = load_metric("bleu", cache_dir="/torch/metrics")

accuracy = load_metric("accuracy")


# metric = load_metric("accuracy", cache_dir="/torch/metrics")

def compute_bleu(tokenizer):
    def _compute(eval_text):
        logits, labels = eval_text
        predictions = logits
        labels[labels == -100] = tokenizer.pad_token_id
        predictions = [tokenizer.convert_ids_to_tokens(p, skip_special_tokens=True) for p in predictions]
        labels = [[tokenizer.convert_ids_to_tokens(l, skip_special_tokens=True)] for l in labels]
        return bleu.compute(predictions=predictions, references=labels)

    return _compute


def compute_bertscore(tokenizer):
    def _compute(eval_text):
        logits, labels = eval_text
        predictions = logits
        labels[labels == -100] = tokenizer.pad_token_id
        predictions = [tokenizer.decode(p, skip_special_tokens=True).split() for p in predictions]
        labels = [[tokenizer.decode(l, skip_special_tokens=True).split()] for l in labels]
        return bertscore.compute(predictions=predictions, references=labels, lang="en", rescale_with_baseline=True)

    return _compute


def compute_accuracy(tokenizer):
    def _compute(eval_text):
        logits, labels = eval_text
        predictions = logits
        labels[labels == -100] = tokenizer.pad_token_id
        return accuracy.compute(predictions=predictions.reshape(-1), references=labels.reshape(-1))

    return _compute
