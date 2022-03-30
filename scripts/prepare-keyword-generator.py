import torch
from datasets import load_dataset
from rake_nltk import Rake

TRAINSIZE = 15000
MAXKEYWORDS = 4

# This script loads ROCStories and prepares the dataset for keyword generation
if __name__ == '__main__':
    rake = Rake()
    roc_stories = load_dataset("adamlin/roc_story")

    testdata = []

    for i in range(TRAINSIZE):
        story = roc_stories['train'][i]

        text = ""
        for j in range(1, 5):
            text += story['sentence%s' % j]
            rake.extract_keywords_from_text(story['sentence%s' % (j + 1)])
            keywords = rake.get_ranked_phrases()[:MAXKEYWORDS]
            testdata.append((i, j, text, keywords))

        if i % 500 == 0:
            print(i)

    torch.save(testdata, "keyword-generator-4.pickle")
