# TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
This repo contains code for the paper [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension][triviaqa-arxiv]. The data can be downloaded from the [TriviaQA website][triviaqa-website]. Please contact [Mandar Joshi][mandar-home] (<first-name>90@cs.washington.edu) for suggestions and comments.

## 1. Requirements
#### General
Python 3 (You should be able to run the evaluation scripts using Python 2.7 if you take care of unicode issues (see [utils.py][utils-link]). BiDAF requires Python 3 -- check the original [repository][bidaf-orig-github] for more details.)

#### Python Packages
- tensorflow (deep learning library, verified on r0.11)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)

## Evaluation
The ```dataset file``` parameter refers to files in the ```qa``` directory of the data (e.g., ```wikipedia-dev.json```). For file format, check out the ```sample``` directory in the repo.
```
python3 -m evaluation.triviaqa_evaluation --dataset_file samples/triviaqa_sample.json --prediction_file samples/sample_predictions.json
```

## BiDAF evaluation
```
python3 -m evaluation.evaluate_bidaf --dataset_file <triviaqa-file> --bidaf_file <bidaf-prediction-file>
```

## BiDAF code
The original code is available at [here][bidaf-orig-github]. A slightly modified version to run on TriviaQA is coming soon!

[bidaf-orig-github]: https://github.com/allenai/bi-att-flow/
[triviaqa-arxiv]: https://arxiv.org/abs/1705.03551
[mandar-home]: http://homes.cs.washington.edu/~mandar90/
[triviaqa-website]: http://nlp.cs.washington.edu/triviaqa/
