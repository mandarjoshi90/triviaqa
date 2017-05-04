import utils.utils
import utils.dataset_utils
from collections import defaultdict
import argparse

def calculate_oracle(qajson, squad_format_file):
    triviaqa_data = utils.dataset_utils.read_triviaqa_data(qajson)
    is_wiki = triviaqa_data['Domain'] == 'Wikipedia'
    key_to_ground_truth = utils.dataset_utils.get_key_to_ground_truth(triviaqa_data)

    squad_format_data = utils.utils.read_json(squad_format_file)

    keys_to_answer = {tuple(datum['paragraphs'][0]['qas'][0]['id'].split('--')) : datum['paragraphs'][0]['qas'][0]['answers'] for datum in squad_format_data['data']}
    keys_to_answer_len = defaultdict(int)
    for k, v in keys_to_answer.items():
        nk = utils.dataset_utils.get_question_doc_string(k[0], k[1]) if not is_wiki else k[0]
        keys_to_answer_len[nk] += len(v)

    oracle = 0.0
    for key in key_to_ground_truth:
        if key in keys_to_answer_len and keys_to_answer_len[key]:
            oracle += 1
    common = len(set(key_to_ground_truth.keys()).intersection(set(keys_to_answer_len)))
    print ('Common: {} preds: {} ground: {}'.format(common, len(keys_to_answer_len), len(key_to_ground_truth)))
    print ('Oracle', oracle * 100.0 / len(key_to_ground_truth))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--triviaqa_file', help='Triviaqa file')
    parser.add_argument('--squad_format_file', help='Squad (BiDAF input) format file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    calculate_oracle(args.triviaqa_file, args.squad_format_file)
