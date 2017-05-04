# Triviaqa
## Evaluation
```
python3 -m evaluation.triviaqa_evaluation --dataset_file samples/triviaqa_sample.json --prediction_file samples/sample_predictions.json
```

## BiDAF evaluation
```
python3 -m evaluation.evaluate_bidaf --dataset_file <triviaqa-file> --bidaf_file <bidaf-prediction-file>
```

## BiDAF code
The original code is available at https://github.com/allenai/bi-att-flow/. A slightly modified version to run on TriviaQA is coming soon!