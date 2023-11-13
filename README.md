# TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
- This repo contains code for the paper
Mandar Joshi, Eunsol Choi, Daniel Weld, Luke Zettlemoyer.

[TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension][triviaqa-arxiv] 
In Association for Computational Linguistics (ACL) 2017, Vancouver, Canada.

- The data can be downloaded from the [TriviaQA website][triviaqa-website]. The Apache 2.0 License applies to both the code and the data.
- Please contact [Mandar Joshi][mandar-home] (\<first-name\>90@cs.washington.edu) for suggestions and comments.

## Requirements
#### General
- Python 3. You should be able to run the evaluation scripts using Python 2.7 if you take care of unicode in ```utils.utils.py```.
- BiDAF requires Python 3 -- check the [original repository][bidaf-orig-github] for more details.

#### Python Packages
- tensorflow (only if you want to run BiDAF, verified on r0.11)
- nltk
- tqdm

## Evaluation
The ```dataset file``` parameter refers to files in the ```qa``` directory of the data (e.g., ```wikipedia-dev.json```). For file format, check out the ```sample``` directory in the repo.
```
python3 -m evaluation.triviaqa_evaluation --dataset_file samples/triviaqa_sample.json --prediction_file samples/sample_predictions.json
```
## Miscellaneous
- If you have a SQuAD model and want to run on TriviaQA, please refer to ```utils.convert_to_squad_format.py```



[bidaf-orig-github]: https://github.com/allenai/bi-att-flow/
[triviaqa-arxiv]: https://arxiv.org/abs/1705.03551
[mandar-home]: http://homes.cs.washington.edu/~mandar90/
[triviaqa-website]: http://nlp.cs.washington.edu/triviaqa/
