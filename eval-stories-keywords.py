import torch
from transformers import AutoTokenizer

import processing

KEYWORDMODEL = "./keyword-generator-t5-small-30-3-4"


def bertscore(pred, label):
    return processing.bertscore.compute(predictions=[" ".join(pred)], references=[" ".join(label)], lang="en",
                                        rescale_with_baseline=True)["f1"][0]


def compute(dataset):
    results = {}
    for (story, part, pretext, next_kw, generated) in dataset:
        if story not in results:
            results[story] = {
                "f1": 0,
                "c": 0,
                "f1-by-part": {},
                "min": (-1, 9999),
                "max": (-1, -9999)
            }
        results[story]["c"] += 1
        bert = bertscore(generated, next_kw)
        results[story]["f1"] += bert
        results[story]["f1-by-part"][part] = bert
        if results[story]["min"][1] > bert:
            results[story]["min"] = (part, bert)
        if results[story]["max"][1] < bert:
            results[story]["max"] = (part, bert)
        if results[story]["c"] == 4:
            print("%s:" % story)
            for i in range(4):
                print("\t%s: %s" % (i + 1, results[story]["f1-by-part"][i + 1]))
            print("\tavg: %s" % (results[story]["f1"] / 4.0))
            print("\tmin: %s" % (str(results[story]["min"])))
            print("\tmax: %s" % (str(results[story]["max"])))

    return results


if __name__ == '__main__':
    kw_train = torch.load("scripts/keywords-result-train-4.pickle")
    kw_eval = torch.load("scripts/keywords-result-eval-4.pickle")
    torch.save(compute(kw_train), "scripts/keywords-results-f1-train-4.pickle")
    torch.save(compute(kw_eval), "scripts/keywords-results-f1-eval-4.pickle")
