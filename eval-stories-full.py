import torch

import processing

def bleuscore(pred, label):
    return processing.bleu.compute(predictions=[pred.split()], references=[[label.split()]], smooth=True)["bleu"]


def bertscore(pred, label):
    return processing.bertscore.compute(predictions=[pred], references=[label], lang="en",
                                        rescale_with_baseline=True)["f1"][0]


def compute(dataset):
    results = {}
    for (story, part, pretext, next_kw, generated_kw, generated_text, expected_tok, expected_str) in dataset:
        if story not in results:
            results[story] = {
                "f1": 0,
                "c": 0,
                "f1-by-part": {},
                "min": (-1, 9999),
                "max": (-1, -9999)
            }
        results[story]["c"] += 1
        bert = bertscore(generated_text, expected_str)
        results[story]["f1"] += bert
        results[story]["f1-by-part"][part] = bert
        if results[story]["min"][1] > bert:
            results[story]["min"] = (part, bert)
        if results[story]["max"][1] < bert:
            results[story]["max"] = (part, bert)
        if results[story]["c"] == 4:
            print("%s:" % story)
            for i in range(4):
                b = results[story]["f1-by-part"][i + 1]
                print("\t%s: %s" % (i + 1, b))
            print("\tavg: %s" % (results[story]["f1"] / 4.0))
            print("\tmin: %s" % (str(results[story]["min"])))
            print("\tmax: %s" % (str(results[story]["max"])))

    return results


if __name__ == '__main__':
    kw_train = torch.load("scripts/text-result-gt-full-train-4.pickle")
    kw_eval = torch.load("scripts/text-result-gt-full-eval-4.pickle")
    torch.save(compute(kw_train), "scripts/text-results-bert-full-train-4.pickle")
    torch.save(compute(kw_eval), "scripts/text-results-bert-full-eval-4.pickle")
