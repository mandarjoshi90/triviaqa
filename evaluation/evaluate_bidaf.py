# -*- coding: utf-8 -*-
import argparse
import utils.utils
import evaluation.triviaqa_evaluation
import utils.dataset_utils
from collections import defaultdict


def create_answer_dict(answer_json, ques_level):
    key_to_answer_scores = {}
    key_to_pred_answer = {}
    key_to_pred_score = {}
    qid_to_confidence = answer_json['scores']
    for qid_filename in answer_json:
        if not (qid_filename == 'scores' or qid_filename == 'all_scores'):
            confidence = qid_to_confidence[qid_filename]
            qid_filename_tup = tuple(qid_filename.split('--'))

            key = qid_filename_tup[0] if ques_level else qid_filename
            # key = '_'.join(qid_filename.split('_')[:2])

            answer = answer_json[qid_filename]
            answer = evaluation.triviaqa_evaluation.normalize_answer(answer)
            key_to_answer_scores[key] = key_to_answer_scores.get(key, defaultdict(float))
            key_to_answer_scores[key][answer] += confidence
    for key in key_to_answer_scores:
        if len(key_to_answer_scores[key]) > 0:
            sorted_ans = sorted(key_to_answer_scores[key].items(), key=lambda x: float(x[1]), reverse=True) # confidence
            key_to_pred_answer[key] = sorted_ans[0][0]
            key_to_pred_score[key] = sorted_ans[0][1]

    return key_to_pred_answer, key_to_pred_score


def evaluate(bidaf_op_file, questions_file, limited=False):
    bidaf_json = utils.utils.read_json(bidaf_op_file)
    triviaqa_data = utils.dataset_utils.read_triviaqa_data(questions_file)
    key_to_pred, key_to_pred_score = create_answer_dict(bidaf_json, triviaqa_data['Domain'] == 'Wikipedia')
    key_to_ground_truth = utils.dataset_utils.get_key_to_ground_truth(triviaqa_data)
    qids = key_to_pred.keys() if limited else None
    print (evaluation.triviaqa_evaluation.evaluate_triviaqa(key_to_ground_truth, key_to_pred, qid_list=qids, mute=True))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', help='Triviaqa file')
    parser.add_argument('--bidaf_file', help='BiDAF output file')

    parser.add_argument('--limited', default=False, type=bool, help='Evaluate only qids appearing in predictions')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    evaluate(args.bidaf_file, args.dataset_file, args.limited)

