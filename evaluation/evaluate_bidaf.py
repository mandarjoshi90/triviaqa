# -*- coding: utf-8 -*-
import os
import utils.file_utils
import sys
import config.properties as properties
import json
import triviaqa_evaluation
import evaluation_utils

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
            key = qid_filename_tup[0] if ques_level else qid_filename_tup
            # key = '_'.join(qid_filename.split('_')[:2])
            answer = answer_json[qid_filename]
            answer = triviaqa_evaluation.normalize_answer(answer)
            key_to_answer_scores[key] = key_to_answer_scores.get(key, defaultdict(float))
            key_to_answer_scores[key][answer] += confidence
    for key in key_to_answer_scores:
        if len(key_to_answer_scores[key]) > 0:
            sorted_ans = sorted(key_to_answer_scores[key].items(), key=lambda x: float(x[1]), reverse=True) # confidence
            key_to_pred_answer[key] = sorted_ans[0][0]
            key_to_pred_score[key] = sorted_ans[0][1]

    return key_to_pred_answer, key_to_pred_score

def evaluate(bidaf_op_file, questions_file, is_wiki, limited=False, qids=None, is_clean=False):
    bidaf_json = json.loads(utils.file_utils.get_file_content_from_local(bidaf_op_file, encoding='utf-8'))
    triviaqa_data = evaluation_utils.read_triviaqa_data(questions_file, is_wiki, is_clean)
    key_to_pred, key_to_pred_score  = create_answer_dict(bidaf_json, is_wiki)
    key_to_ground_truth = evaluation_utils.get_key_to_ground_truth(triviaqa_data, is_wiki)
    if qids is None:
        qids = key_to_pred.keys() if limited else None
    print triviaqa_evaluation.evaluate_triviaqa(key_to_ground_truth, key_to_pred, qid_list=qids, mute=True)
    return qids


def test():
    split = 'dev'
    task = 'ms'
    bidaf_home = '/home/mandar90/workspace/restore-bidaf-for-triviaqa/'
    # bidaf_home = '/home/mandar90/workspace/bidaf-for-triviaqa/'
    # bidaf_home = '/home/mandar90/workspace/restore-bidaf-for-triviaqa/'
    data_file = bidaf_home + 'out/basic/51/answer/dev-011000.json'
    # data_file = bidaf_home + 'out/basic/triviaqa-ss-first-span-800-bv/answer/dev-020000.json'
    # data_file = bidaf_home + 'output_files/v-1.2/triviaqa-ss-first-span-800-bv_' + split + '.json'
    # data_file = bidaf_home + 'output_files/v-1.2/triviaqa-ms-first-span-doc-bv_' + split + '.json'
    # data_file = bidaf_home + 'output_files/v-1.2/triviaqa-ms-single-top50_' + split + '.json'
    # data_file = bidaf_home + 'output_files/v-1.2/triviaqa-wiki-sent-select-first3-top10_' + split + '.json'
    # data_file = bidaf_home + 'output_files/v-1.2/triviaqa-wiki-sent-selection-ntoldm-old-d' +  '.json'
    # data_file = bidaf_home + 'answer_files_2/' + task + '-' + split + '.json'
    # data_file = bidaf_home + 'answer_files/ms-' + split + '.json'
    qa_json = os.path.join(properties.dataset_home, 'qa/v-1.2/camera-ready/' + task + '-' + split + '.json')
    # data_file = bidaf_home + 'out/basic/20/answer/dev-017000.json'
    # qids = evaluate(data_file, qa_json, task=='ss', limited=False, is_clean=True)
    # data_file = bidaf_home + 'out/basic/backup_10/answer/dev-017000.json'
    # qids = evaluate(data_file, qa_json, task=='ss', limited=True, is_clean=False)

    for step in range(9, 25):
        data_file = bidaf_home + 'out/basic/51/answer/dev-0' + str(step).zfill(2) + '000.json'
        print step,
        evaluate(data_file, qa_json, task == 'ss', limited=True)

        # for task in ['ss', 'ms']:
        #     for split in ['dev', 'test']:
        #         for type in ['', 'cleaned-']:
        #             print task, split, type
        #             data_file = bidaf_home + 'answer_files/' + task + '-' + split + '.json'
        #             qa_json = os.path.join(properties.dataset_home, 'qa/v-1.2/camera-ready/' + type + task + '-' + split + '.json')
        #             # data_file = bidaf_home + 'out/basic/20/answer/dev-017000.json'
        #             qids = evaluate(data_file, qa_json, task == 'ss', limited=False, is_clean=type=='cleaned-')


def main(argv=None):
    if argv is None:
        argv = sys.argv
    test()

if __name__ == '__main__':
    main(sys.argv)
