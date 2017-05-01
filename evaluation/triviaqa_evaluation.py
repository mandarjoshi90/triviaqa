# -*- coding: utf-8 -*-
""" Official evaluation script for v1.0 of the TriviaQA dataset. Extended from the evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set((string.punctuation) + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text )

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def is_exact_match(answer_object, prediction):
    ground_truths = get_ground_truths(answer_object)
    for ground_truth in ground_truths:
        if exact_match_score(prediction, ground_truth):
            return True
    return False


def has_exact_match(ground_truths, candidates):
    for ground_truth in ground_truths:
        if ground_truth in candidates:
            return True
    return False


def get_ground_truths(answer):
    return answer['NormalizedAliases'] + [normalize_answer(ans) for ans in answer.get('HumanAnswers', [])]

def get_oracle_score(ground_truth, predictions, qid_list=None, mute=False):
    exact_match = common = 0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predictions:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = predictions[qid]
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = has_exact_match(ground_truths, prediction)
        exact_match += int(em_for_this_question)

    exact_match = 100.0 * exact_match / len(qid_list)

    return {'oracle_exact_match': exact_match,  'common': common, 'denominator': len(qid_list), 'pred_len': len(predictions), 'gold_len': len(ground_truth) }

def evaluate_triviaqa(ground_truth, predictions, qid_list=None, mute=False):
    f1 = exact_match = common = 0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predictions:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        if qid not in ground_truth:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = predictions[qid]
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        if em_for_this_question == 0 and not mute:
            print("em=0:", prediction, ground_truths)
        exact_match += em_for_this_question
        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1 += f1_for_this_question

    exact_match = 100.0 * exact_match / len(qid_list)
    f1 = 100.0 * f1 / len(qid_list)

    return {'exact_match': exact_match, 'f1': f1, 'common': common, 'denominator': len(qid_list),
            'pred_len': len(predictions), 'gold_len': len(ground_truth)}

def evaluate_triviaqa_debug(ground_truth, predictions, qid_list=None, mute=False):
    f1 = exact_match = common = 0
    if qid_list is None:
        qid_list = ground_truth.keys()
    correct = set()
    for qid in qid_list:
        if qid not in predictions:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = predictions[qid]
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        if em_for_this_question == 1.0:
            correct.add(qid)
        if em_for_this_question == 0 and not mute:
            print("em=0:", prediction, ground_truths)
        exact_match += em_for_this_question
        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1 += f1_for_this_question

    exact_match = 100.0 * exact_match / len(qid_list)
    f1 = 100.0 * f1 / len(qid_list)

    return {'exact_match': exact_match, 'f1': f1, 'common': common, 'denominator': len(qid_list),
            'pred_len': len(predictions), 'gold_len': len(ground_truth)}, correct