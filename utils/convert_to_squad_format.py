import json
import utils.file_utils
import utils.dataset_utils
import utils.nltk_utils
import os
import config.properties as props
from tqdm import tqdm
import random
seed = 10
sample_size = 80000

wiki_dir = os.path.join(props.dataset_home, 'evidence/wikiExtractions2')
# web_dir = '/home/mandar90/homes/cycle_home/triviaqa_spider/article_top10_txt_pages_dec_15'
web_dir = os.path.join(props.dataset_home, 'evidence/article_top10_txt_pages_apr_2')

def get_text(qad):
    if 'Rank' in qad:
        local_file = os.path.join(web_dir, qad['Filename'])
    else:
        local_file = os.path.join(wiki_dir, qad['Filename'])
    return utils.file_utils.get_file_content_from_local(local_file, encoding='utf-8')

def select_relevant_portion(text):
    paras = text.split('\n')
    selected = []
    done  = False
    for para in paras:
        sents = utils.nltk_utils.get_sentences(para)

        for sent in sents:
            words = utils.nltk_utils.word_tokenize(sent)
            for word in words:
                selected.append(word)
                if len(selected) >= 800:
                    done = True
                    break
            if done:
                break
        if done:
            break
        selected.append('\n')
    # print ' '.join(selected)
    st = ' '.join(selected).strip()
    return st


def add_triple_data(datum, page):
    qad = {'Source': 'Wikipedia'}
    for property in ['QuestionId', 'Question', 'Answer']:
        qad[property] = datum[property]
    for property in page:
        qad[property] = page[property]
    return qad

def get_qad_triples(data):
    qad_triples = []
    for datum in data:
        for page in datum.get('EntityPages', []) + datum.get('SearchResults', []):
            qad = add_triple_data(datum, page)
            qad_triples.append(qad)
    return qad_triples

def convert_to_squad_format(qa_json_file, squad_file, split, task):
    qa_json = json.loads(utils.file_utils.get_file_content_from_local(qa_json_file, encoding='utf-8'))
    # qid_to_data = {datum['QuestionId']: datum for datum in qa_json}
    qad_triples = get_qad_triples(qa_json)
    
    random.seed(seed)
    random.shuffle(qad_triples)

    squad = {'data': []}
    squad['version'] = '1.2'
    data = squad['data']
    text_is_none = 0
    answer_not_found = 0
    for qad in tqdm(qad_triples):
        # cols = line.split('\t')
        qid = qad['QuestionId']

        text = get_text(qad)
        selected_text = select_relevant_portion(text)

        question = qad['Question']
        para = {'context': selected_text, 'qas': [{'question': question, 'answers': []}]}
        # paras.append(para)
        data.append({'paragraphs': [para]})
        qa = para['qas'][0]
        qa['id'] = qid + '--' + qad['Filename']
        qa['qid'] = qid

        ans_string, index = utils.dataset_utils.answer_index_in_document(qad['Answer'], selected_text)
        if index == -1:
            answer_not_found += 1
            if split == 'train':
                continue
        else:
            qa['answers'].append({'text': ans_string, 'answer_start': index})

        if split == 'train' and len(data) >= sample_size and task == 'ms':
            break
    utils.file_utils.write_to_json(squad, squad_file)
    print 'Added', len(data), text_is_none, answer_not_found

if __name__ == '__main__':
    qa_json_dir = os.path.join(props.dataset_home, 'qa/v-1.2/camera-ready/')
    # url_to_local = utils.dataset_utils.read_url_file(props.url_to_local_file)
    task = 'ms'
    for split in ['train', 'dev', 'test']:
        qa_json_file = os.path.join(qa_json_dir, task + '-' + split + '.json')
        squad_file = '/home/mandar90/data/cam-ready/' + task + '/expt2/' + split + '-v1.2.json'
        convert_to_squad_format(qa_json_file, squad_file, split, task)
