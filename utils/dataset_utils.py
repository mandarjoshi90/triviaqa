# -*- coding: utf-8 -*-
import utils.utils
import utils


# Key for wikipedia eval is question-id. Key for web eval is the (question_id, filename) tuple
def get_key_to_ground_truth(data):
    if data['Domain'] == 'Wikipedia':
        return {datum['QuestionId']: datum['Answer'] for datum in data['Data']}
    else:
        return get_qd_to_answer(data)


def get_question_doc_string(qid, doc_name):
    return '{}--{}'.format(qid, doc_name)

def get_qd_to_answer(data):
    key_to_answer = {}
    for datum in data['Data']:
        for page in datum.get('EntityPages', []) + datum.get('SearchResults', []):
            qd_tuple = get_question_doc_string(datum['QuestionId'], page['Filename'])
            key_to_answer[qd_tuple] = datum['Answer']
    return key_to_answer


def read_clean_part(datum):
    for key in ['EntityPages', 'SearchResults']:
        new_page_list = []
        for page in datum.get(key, []):
            if page['DocPartOfVerifiedEval']:
                new_page_list.append(page)
        datum[key] = new_page_list
    assert len(datum['EntityPages']) + len(datum['SearchResults']) > 0
    return datum


def read_triviaqa_data(qajson):
    data = utils.utils.read_json(qajson)
    # read only documents and questions that are a part of clean data set
    if data['VerifiedEval']:
        clean_data = []
        for datum in data['Data']:
            if datum['QuestionPartOfVerifiedEval']:
                if data['Domain'] == 'Web':
                    datum = read_clean_part(datum)
                clean_data.append(datum)
        data['Data'] = clean_data
    return data


def answer_index_in_document(answer, document):
    answer_list = answer['NormalizedAliases']
    for answer_string_in_doc in answer_list:
        index = document.lower().find(answer_string_in_doc)
        if index != -1:
            return answer_string_in_doc, index
    return answer['NormalizedValue'], -1
