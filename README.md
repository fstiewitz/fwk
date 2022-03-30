### FunWithKeywords

<small>Course Assignment Submission</small>

    The goal of this software project is to build a rudimentary story
    generator using two neural networks. The first neural network gener-
    ates a set of keywords given a story input; the second generates the
    next sentence using the story input and the generated keywords. Given
    performance constraints the project employs small transformer models
    for implementing a proof-of-concept, but can be adjusted to work with
    larger networks on more powerful machines.

#### Usage

Most scripts save their results in `scripts` for use by other scripts. All data is serialized using `torch.save`
/ `torch.load`.

1. Load dataset from HuggingFace and prepare for our tasks.

```sh
cd scripts
./prepare-keyword-generator.py
./prepare-text-generator.py
```

2. Calculate maximum lengths of inputs and outputs for faster training. Then insert the results into `constants.py`.
   Default values are already set.

```sh
./calculate-bounds.py
vim constants.py
```

3. Train generators

```sh
./train-keyword-generator.py
./train-text-generator.py
```

4. Test generator on dataset

```sh
./test-keyword-generator.py
./test-text-generator.py
./test-generator.py
```

or use your own stories with `inference.py`

or evaluate entire dataset with `eval-(keyword|text|full)-generator.py`

5. Evaluate test data

```sh
./eval-stories-keywords.py
./eval-stories-text.py
./eval-stories-full.py
```

6. Generate graphs

```sh
./generate-keyword-graph.py
./generate-text-graph.py
./generate-full-graph.py
```