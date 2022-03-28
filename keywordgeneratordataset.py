from torch.utils.data import Dataset

import processing


# Prepares a dataset for training the keyword generator
class KeywordGeneratorDataset(Dataset):
    def __init__(self, tokenizer, data, max_source_length, max_target_length):
        self.tokenized = []
        for tr in data:
            s = processing.process_sentences_for_keyword_generation(tr[2])
            tokenized_input = tokenizer(
                s,
                padding="max_length",
                truncation=True,
                max_length=max_source_length,
                return_tensors='pt')
            labels = tokenizer(
                ' '.join(tr[3]),
                padding="max_length",
                truncation=True,
                max_length=max_target_length,
                return_tensors='pt').input_ids
            labels[labels == tokenizer.pad_token_id] = -100
            self.tokenized.append({
                'input_ids': tokenized_input.input_ids[0],
                'attention_mask': tokenized_input.attention_mask[0],
                'labels': labels[0]
            })
