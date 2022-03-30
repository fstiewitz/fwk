import torch
from datasets import load_dataset

TRAINSIZE = 15000

# This script loads ROCStories and prepares the dataset for text generation
if __name__ == '__main__':
    keyword_generator = torch.load("keyword-generator-4.pickle")
    roc_stories = load_dataset("adamlin/roc_story")

    testdata = []

    for (textId, sentenceId, preText, keywords) in keyword_generator:
        nextSentence = roc_stories['train'][textId]['sentence' + str(sentenceId + 1)]
        testdata.append((textId, sentenceId, preText, keywords, nextSentence))

    torch.save(testdata, "text-generator-4.pickle")
