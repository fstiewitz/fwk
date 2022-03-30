from math import floor

import numpy as np
import torch
from matplotlib import pyplot as plt

keys = [2, 8, 104, 114, 198, 210, 249]
eval_values = []


def make_indiviual_plot(ax0):
    global eval_values
    eval_values = list(
        sorted([(x['f1'] / 4.0, x['min'], x['max'], k) for k, x in eval_results.items()], key=lambda k: k[0]))
    x = np.arange(len(eval_values))
    ax0.errorbar(x, list(map(lambda x: x[0], eval_values)), yerr=[
        list(map(lambda x: x[0] - x[1][1], eval_values)),
        list(map(lambda x: x[2][1] - x[0], eval_values)),
    ])
    ax0.set_ylabel("BERTScore")
    ax0.set_title("BERTScores of eval samples")
    # for x in keys:
    #    ax0.axvline(x=x, color='red', alpha=0.5)
    ax0.legend()


if __name__ == '__main__':
    train_results = torch.load("scripts/text-results-bert-train-4.pickle")
    eval_results = torch.load("scripts/text-results-bert-eval-4.pickle")

    group_count = 16

    groups = [1.0 * x / group_count for x in range(group_count + 1)]
    group_labels = [str(x) for x in groups]

    train_grouped = [0 for x in groups]
    for x in train_results.values():
        i = floor(group_count * x['f1'] / 4.0)
        train_grouped[max(0, i)] += 1
    eval_grouped = [0 for x in groups]
    for x in eval_results.values():
        i = floor(group_count * x['f1'] / 4.0)
        eval_grouped[max(0, i)] += 1

    x = np.arange(len(group_labels))
    width = 0.4
    fig, ax = plt.subplots(2, 1, figsize=(group_count, 8))
    train_rects = ax[0].bar(x - width / 2, train_grouped, width, label="Train")
    eval_rects = ax[0].bar(x + width / 2, eval_grouped, width, label="Eval")
    ax[0].set_ylabel("Count")
    ax[0].set_xlabel("BERTScore")
    ax[0].set_title("BERTScore distribution of training/eval samples")
    ax[0].set_xticks(x, group_labels)
    ax[0].legend()

    make_indiviual_plot(ax[1])

    # ax.bar_label(train_rects, padding=3)
    # ax.bar_label(eval_rects, padding=3)
    fig.tight_layout()
    plt.show()

    inputs = torch.load("scripts/text-result-gt-eval-4.pickle")
    input_map = dict([((x[0], x[1]), x) for x in inputs])
    for i in keys:
        print(i)
        print(eval_values[i])
        for j in range(4):
            x = input_map[(eval_values[i][3], j + 1)]
            s = x[2].split(".")[j]
            print("\t", s)
            print("\t\texpected:  ", x[6])
            print("\t\tgenerated: ", x[4])
        print()
