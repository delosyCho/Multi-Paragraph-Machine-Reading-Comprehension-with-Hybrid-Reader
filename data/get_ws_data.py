from HTML_Utils import *
import json
import os
import re

import numpy as np

import Chuncker

import Table_Holder
from Table_Holder import detect_num_word, detect_simple_num_word, get_space_of_num, get_space_num_lists

from transformers import AutoTokenizer


def RepresentsInt(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


pattern = '<[^>]*>'
table_holder = Table_Holder.Holder()
chuncker = Chuncker.Chuncker()

max_length = 512

path_dir = '/data/korquad_data/korquad2_train/'

sequence_has_ans = np.zeros(shape=[80000 * 2, 3, max_length], dtype=np.int32)
segments_has_ans = np.zeros(shape=[80000 * 2, 3, max_length], dtype=np.int32)
positions_has_ans = np.zeros(shape=[80000 * 2, 3, max_length], dtype=np.int32)
ranks_has_ans = np.zeros(shape=[80000 * 2, 3, max_length], dtype=np.int32)
names_has_ans = np.zeros(shape=[80000 * 2, 3, max_length], dtype=np.int32)
cols_has_ans = np.zeros(shape=[80000 * 2, 3, max_length], dtype=np.int32)
rows_has_ans = np.zeros(shape=[80000 * 2, 3, max_length], dtype=np.int32)
mask_has_ans = np.zeros(shape=[80000 * 2, 3, max_length], dtype=np.int32)
numeric_space = np.zeros(shape=[80000 * 2, 3, 10, max_length], dtype=np.int32)
numeric_mask = np.zeros(shape=[80000 * 2, 10, max_length], dtype=np.int32)
answer_span = np.zeros(shape=[80000 * 2, 2], dtype=np.int32)

file_list = os.listdir(path_dir)
file_list.sort()
file_list.pop(-1)
file_list.pop(-1)

questions = []
for file_name in file_list[0:1]:
    print(file_name, '...')
    in_path = path_dir + '/' + file_name
    data = json.load(open(in_path, 'r', encoding='utf-8'))

    for article in data['data']:
        for qas in article['qas']:
            question = qas['question']
            questions.append(str(question))

q_idx = np.array(range(len(questions)), dtype=np.int32)
np.random.shuffle(q_idx)

print(file_list)
data_num = 0

tokenizer_ = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
tokenizer_.add_tokens('[STA]')
tokenizer_.add_tokens('[END]')

count = 0
false_count = 0
false_count2 = 0

cor = 0
wrong_case = 0

for file_name in file_list:
    print(file_name, 'processing....', data_num)

    in_path = path_dir + '/' + file_name
    data = json.load(open(in_path, 'r', encoding='utf-8'))

    for article in data['data']:
        doc = str(article['context'])
        doc = doc.replace('\t', ' ')
        doc = doc.replace('\a', ' ')

        print(count, false_count, false_count2, file_name)

        for qas in article['qas']:
            error_code = -1

            answer = qas['answer']
            answer_start = answer['answer_start']
            answer_text = answer['text']
            question = qas['question']

            chuncker.get_feautre(query=question)

            if len(answer_text) > 40:
                continue

            query_tokens = []
            query_tokens.append('[CLS]')
            q_tokens = tokenizer_.tokenize(question.lower())
            for tk in q_tokens:
                query_tokens.append(tk)
            query_tokens.append('[SEP]')
            ######
            # 정답에 ans 토큰을 임베딩하기 위한 코드
            ######

            ans1 = ''
            ans2 = ''
            if doc[answer_start - 1] == ' ':
                ans1 = ' [STA] '
            else:
                ans1 = ' [STA]'

            if doc[answer_start + len(answer_text)] == ' ':
                ans2 = ' [END] '
            else:
                ans2 = ' [END]'

            doc_ = doc[0: answer_start] + ans1 + answer_text + ans2 + doc[answer_start + len(answer_text): -1]
            doc_ = str(doc_)
            #
            #####

            paragraphs = doc_.split('<h2>')
            sequences = []

            tables = []
            for paragraph in paragraphs:
                paragraph_, table_list = pre_process_document(paragraph, answer_setting=False,
                                                              a_token1='',
                                                              a_token2='')

                for table_text in table_list:
                    tables.append(table_text)
            print(len(tables))
            chuncker.get_feautre(query=question)

            ch_scores = []
            selected = -1

            check_table_case = False

            for i, table_text in enumerate(tables):
                if table_text.find('[STA]') != -1 and table_text.find('table') != -1:
                    check_table_case = True
                    selected = i
                    ch_scores.append(-9999)
                else:
                    ch_scores.append(chuncker.get_chunk_score(table_text))
            ch_scores = np.array(ch_scores, dtype=np.float32)


            #print('num of tables:', len(tables))

            if check_table_case is False:
                continue
            if len(tables) == 0:
                continue

            table_text = tables[selected]
            table_text = table_text.replace('<th', '<td')
            table_text = table_text.replace('</th', '</td')
            table_text = table_text.replace(' <td>', '<td>')
            table_text = table_text.replace(' <td>', '<td>')
            table_text = table_text.replace('\n<td>', '<td>')
            table_text = table_text.replace('</td> ', '</td>')
            table_text = table_text.replace('</td> ', '</td>')
            table_text = table_text.replace('\n<td>', '<td>')
            table_text = table_text.replace('[STA]<td>', '<td>[STA] ')
            table_text = table_text.replace('</td>[END]', ' [END]</td>')
            table_text = table_text.replace('</td>', '  </td>')
            table_text = table_text.replace('<td>', '<td> ')
            table_text = table_text.replace('[STA]', '[STA] ')
            table_text = table_text.replace('[END]', ' [END]')

            table_text, child_texts = overlap_table_process(table_text=table_text)
            table_text = head_process(table_text=table_text)

            table_holder.get_table_text(table_text=table_text)
            table_data = table_holder.table_data
            lengths = []

            for data in table_data:
                lengths.append(len(data))
            if len(lengths) <= 0:
                break

            length = max(lengths)

            idx = 0
            tokens_ = []
            rows_ = []
            cols_ = []
            spaces_ = []
            positions_ = []

            for i in range(len(table_data)):
                for j in range(len(table_data[i])):
                    if table_data[i][j] is not None:
                        tokens = tokenizer_.tokenize(table_data[i][j])
                        #name_tag = name_tagger.get_name_tag(table_data[i][j])

                        is_num, number_value = detect_num_word(table_data[i][j])

                        for k, tk in enumerate(tokens):
                            tokens_.append(tk)
                            rows_.append(i + 1)
                            cols_.append(j)
                            positions_.append(k)

                            if is_num is True:
                                if detect_simple_num_word(tk) is True:
                                    space_lists = get_space_num_lists(number_value)
                                    spaces_.append(space_lists)
                                else:
                                    spaces_.append(-1)
                            else:
                                spaces_.append(-1)

                            if k >= 40:
                                break

                        if len(tokens) > 40 and str(table_data[i][j]).find('[END]') != -1:
                            tokens_.append('[END]')
                            rows_.append(i + 1)
                            cols_.append(j)
                            positions_.append(0)
                            spaces_.append(-1)

            start_idx = -1
            end_idx = -1

            tokens = []
            rows = []
            cols = []
            #ranks = []
            segments = []
            #name_tags = []
            positions = []
            spaces = []

            for j, tk in enumerate(query_tokens):
                tokens.append(tk)
                rows.append(0)
                cols.append(0)
                #ranks.append(0)
                segments.append(0)
                #name_tags.append(0)
                positions.append(j)
                spaces.append(-1)

            for j, tk in enumerate(tokens_):
                if tk == '[STA]':
                    start_idx = len(tokens)
                elif tk == '[END]':
                    end_idx = len(tokens) - 1
                else:
                    tokens.append(tk)
                    rows.append(rows_[j] + 1)
                    cols.append(cols_[j] + 1)
                    segments.append(1)
                    #ranks.append(ranks_[j])
                    #name_tags.append(name_tags_[j])
                    positions.append(positions_[j])
                    spaces.append(spaces_[j])

            ids = tokenizer_.convert_tokens_to_ids(tokens=tokens)

            if end_idx > max_length or start_idx > max_length:
                false_count += 1
                continue

            if start_idx == -1 or end_idx == -1:
                false_count += 1
                continue

            length = len(ids)
            if length > max_length:
                length = max_length

            for j in range(length):
                sequence_has_ans[count, 0, j] = ids[j]
                segments_has_ans[count, 0, j] = segments[j]
                cols_has_ans[count, 0, j] = cols[j]
                rows_has_ans[count, 0, j] = rows[j]
                mask_has_ans[count, 0, j] = 1
            answer_span[count, 0] = start_idx
            answer_span[count, 1] = end_idx
            ##########

            for z in range(2):
                if len(tables) == 1:
                    false_count2 += 1
                    break

                zero_table_text = tables[ch_scores.argmax()]
                ch_scores[ch_scores.argmax()] = -99

                table_text = zero_table_text
                table_text = table_text.replace('<th', '<td')
                table_text = table_text.replace('</th', '</td')

                table_text = table_text.replace(' <td>', '<td>')
                table_text = table_text.replace(' <td>', '<td>')
                table_text = table_text.replace('\n<td>', '<td>')
                table_text = table_text.replace('</td> ', '</td>')
                table_text = table_text.replace('</td> ', '</td>')
                table_text = table_text.replace('\n<td>', '<td>')
                table_text = table_text.replace('[STA]<td>', '<td>[STA] ')
                table_text = table_text.replace('</td>[END]', ' [END]</td>')
                table_text = table_text.replace('</td>', '  </td>')
                table_text = table_text.replace('<td>', '<td> ')
                table_text = table_text.replace('[STA]', '[STA] ')
                table_text = table_text.replace('[END]', ' [END]')

                #print(table_text.replace('\n', ''))
                #print(question)
                #input()

                table_text, child_texts = overlap_table_process(table_text=table_text)
                table_text = head_process(table_text=table_text)

                table_holder.get_table_text(table_text=table_text)
                table_data = table_holder.table_data
                lengths = []

                for data in table_data:
                    lengths.append(len(data))
                if len(lengths) <= 0:
                    break

                length = max(lengths)

                idx = 0
                tokens_ = []
                rows_ = []
                cols_ = []
                spaces_ = []
                positions_ = []

                for i in range(len(table_data)):
                    for j in range(len(table_data[i])):
                        if table_data[i][j] is not None:
                            tokens = tokenizer_.tokenize(table_data[i][j])
                            # name_tag = name_tagger.get_name_tag(table_data[i][j])

                            is_num, number_value = detect_num_word(table_data[i][j])

                            for k, tk in enumerate(tokens):
                                tokens_.append(tk)
                                rows_.append(i + 1)
                                cols_.append(j)
                                positions_.append(k)

                                if is_num is True:
                                    if detect_simple_num_word(tk) is True:
                                        space_lists = get_space_num_lists(number_value)
                                        spaces_.append(space_lists)
                                    else:
                                        spaces_.append(-1)
                                else:
                                    spaces_.append(-1)

                                if k >= 40:
                                    break

                            if len(tokens) > 40 and str(table_data[i][j]).find('[END]') != -1:
                                tokens_.append('[END]')
                                rows_.append(i + 1)
                                cols_.append(j)
                                positions_.append(0)
                                spaces_.append(-1)

                tokens = []
                rows = []
                cols = []
                segments = []

                for j, tk in enumerate(query_tokens):
                    tokens.append(tk)
                    rows.append(0)
                    cols.append(0)
                    segments.append(0)

                for j, tk in enumerate(tokens_):
                    tokens.append(tk)
                    rows.append(rows_[j] + 1)
                    cols.append(cols_[j] + 1)
                    segments.append(1)

                ids = tokenizer_.convert_tokens_to_ids(tokens=tokens)

                length = len(ids)
                if length > max_length:
                    length = max_length

                for j in range(length):
                    sequence_has_ans[count, 1 + z, j] = ids[j]
                    segments_has_ans[count, 1 + z, j] = segments[j]
                    cols_has_ans[count, 1 + z, j] = cols[j]
                    rows_has_ans[count, 1 + z, j] = rows[j]
                    mask_has_ans[count, 1 + z, j] = 1
            count += 1
            checked = True

sequence_has_ans_ = np.zeros(shape=[count, 3, max_length], dtype=np.int32)
segments_has_ans_ = np.zeros(shape=[count, 3, max_length], dtype=np.int32)
mask_has_ans_ = np.zeros(shape=[count, 3, max_length], dtype=np.int32)
cols_has_ans_ = np.zeros(shape=[count, 3, max_length], dtype=np.int32)
rows_has_ans_ = np.zeros(shape=[count, 3, max_length], dtype=np.int32)
answer_span_ = np.zeros(shape=[count, 2], dtype=np.int32)

for i in range(count):
    sequence_has_ans_[i] = sequence_has_ans[i]
    segments_has_ans_[i] = segments_has_ans[i]
    #positions_has_ans_[i] = positions_has_ans[i]
    mask_has_ans_[i] = mask_has_ans[i]
    rows_has_ans_[i] = rows_has_ans[i]
    cols_has_ans_[i] = cols_has_ans[i]
    #ranks_has_ans_[i] = ranks_has_ans[i]
    #names_has_ans_[i] = names_has_ans[i]
    #numeric_space_[i] = numeric_space[i]
    #numeric_mask_[i] = numeric_mask[i]
    answer_span_[i] = answer_span[i]

np.save('sequence_table', sequence_has_ans_)
np.save('segments_table', segments_has_ans_)
np.save('mask_table', mask_has_ans_)
np.save('rows_table', rows_has_ans_)
np.save('cols_table', cols_has_ans_)
np.save('answer_span_table', answer_span_)
