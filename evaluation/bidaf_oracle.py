import sys
import utils.file_utils
import evaluation_utils
import json
import config.properties as properties
import os
from collections import defaultdict

def calculate_oracle(qajson, squad_format_file, is_wiki, is_clean):
    triviaqa_data = evaluation_utils.read_triviaqa_data(qajson, is_wiki, is_clean)
    key_to_ground_truth = evaluation_utils.get_key_to_ground_truth(triviaqa_data, is_wiki)

    squad_format_data = json.loads(utils.file_utils.get_file_content_from_local(squad_format_file, encoding='utf-8'))
    keys_to_answer = {tuple(datum['paragraphs'][0]['qas'][0]['id'].split('--')) : datum['paragraphs'][0]['qas'][0]['answers'] for datum in squad_format_data['data']}
    keys_to_answer_len = defaultdict(int)
    for k, v in keys_to_answer.iteritems():
        nk = k if not is_wiki else k[0]
        keys_to_answer_len[nk] += len(v)

    oracle = 0.0
    for key in key_to_ground_truth:
        if key in keys_to_answer_len and keys_to_answer_len[key]:
            oracle += 1
    common = len(set(key_to_ground_truth.keys()).intersection(set(keys_to_answer_len)))
    print 'Common: {} preds: {} ground: {}'.format(common, len(keys_to_answer_len), len(key_to_ground_truth))
    print 'Oracle', oracle * 100.0 / len(key_to_ground_truth)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    task = 'ms'
    split = 'dev'
    type = ''

    squad_file = '/home/mandar90/data/cam-ready/' + task + '/' + split + '-v1.2.json'
    qa_json = os.path.join(properties.dataset_home, 'qa/v-1.2/camera-ready/' + type + task + '-' + split + '.json')
    # calculate_oracle(qa_json, squad_file, task=='ss', False)

    for task in ['ss', 'ms']:
        for split in ['dev', 'test']:
            for type in ['', 'cleaned-']:
                print task, split, type
                squad_file = '/home/mandar90/data/cam-ready/' + task + '/' + split + '-v1.2.json'
                qa_json = os.path.join(properties.dataset_home,
                                       'qa/v-1.2/camera-ready/' + type + task + '-' + split + '.json')
                calculate_oracle(qa_json, squad_file, task == 'ss', is_clean=(type=='cleaned-'))




if __name__ == '__main__':
    main(sys.argv)
