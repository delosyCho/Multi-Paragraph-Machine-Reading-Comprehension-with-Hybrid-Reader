from utils.tokenization import BasicTokenizer, whitespace_tokenize

import utils.Chuncker as Chuncker
from utils.HTML_Utils import *
import json
import os

import numpy as np
from utils.HTML_Processor import process_document
from transformers import AutoTokenizer

from konlpy.tag import Mecab

import collections

import re

pattern = '<[^>]*>'

def read_squad_example(orig_answer_text, answer_offset, paragraph_text):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    start_position = None
    end_position = None

    answer_length = len(orig_answer_text)
    start_position = char_to_word_offset[answer_offset]
    end_position = char_to_word_offset[answer_offset + answer_length - 1]

    # Only add answers where the text can be exactly recovered from the
    # document. If this CAN'T happen it's likely due to weird Unicode
    # stuff so we will just skip the example.
    #
    # Note that this means for training mode, every example is NOT
    # guaranteed to be preserved.
    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
    cleaned_answer_text = " ".join(
        whitespace_tokenize(orig_answer_text))
    if actual_text.find(cleaned_answer_text) == -1:
        print("Could not find answer: '%s' vs. '%s'",
              actual_text, cleaned_answer_text)
        return -1, -1, -1

    return doc_tokens, start_position, end_position


def convert_example_to_tokens(question_text,
                              start_position, end_position,
                              doc_tokens, orig_answer_text, doc_stride=128):
    max_seq_length = 512

    query_tokens = tokenizer.tokenize(question_text)

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None

    if True:
        tok_start_position = orig_to_tok_index[start_position]
        if end_position < len(doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    arr_input_ids = []
    arr_segment_ids = []
    arr_start_position = []
    arr_end_position = []

    doc_texts = []
    doc_tokens = []

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None
        if True:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
                # print('answer:', answer_text, tokens[start_position: end_position + 1])
                start_position, end_position = _improve_answer_span(tokens, start_position, end_position, tokenizer,
                                                                    answer_text)
                # print('answer:', answer_text, tokens[start_position: end_position + 1])
                # print('-------------------')
        # print('len:', len(input_ids))

        arr_input_ids.append(input_ids)
        arr_segment_ids.append(segment_ids)
        arr_start_position.append(start_position)
        arr_end_position.append(end_position)

        doc_text = ''
        opened = False
        for doc_token in tokens:
            if opened is True:
                doc_text += doc_token + ' '

            if doc_token == '[SEP]':
                opened = True
        doc_text = doc_text.replace(' ##', '')
        doc_texts.append(doc_text)
        doc_tokens.append(tokens)

    return arr_input_ids, arr_segment_ids, arr_start_position, arr_end_position, doc_texts, doc_tokens


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


chuncker = Chuncker.Chuncker()

max_length = 512
# tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

path_dir = '/data/korquad_data/korquad2_train/'

N_BATCH = 3

sequence_has_ans = np.zeros(shape=[250000, N_BATCH, max_length], dtype=np.int32)
segments_has_ans = np.zeros(shape=[250000, N_BATCH, max_length], dtype=np.int32)
answer_span = np.zeros(shape=[250000, 3], dtype=np.int32)

file_list = os.listdir(path_dir)
file_list.sort()

file_list.pop(-1)

print(file_list)
data_num = 0

count = 0
false_count = 0
false_count2 = 0

zero_case = 0

cor = 0
wrong_case = 0

tagger = Mecab()

pattern = '<[^>]*>'

for file_name in file_list:

    print(file_name, 'processing....', data_num)

    in_path = path_dir + '' + file_name
    data = json.load(open(in_path, 'r', encoding='utf-8'))

    for article in data['data']:
        # print(count)

        doc = str(article['context'])
        doc = doc.replace('\t', ' ')
        doc = doc.replace('\a', ' ')

        print(count, false_count, false_count2, zero_case, file_name)

        for qas in article['qas']:
            error_code = -1

            answer = qas['answer']
            answer_start = answer['answer_start']
            answer_text = answer['text']
            answer_text = re.sub(pattern=pattern, repl='', string=answer_text)
            question = qas['question']

            chuncker.get_feautre(query=question)

            long_short_label = 0
            if len(answer_text) > 50:
                continue

            query_tokens = []
            query_tokens.append('[CLS]')
            q_tokens = tokenizer.tokenize(question.lower())
            for tk in q_tokens:
                query_tokens.append(tk)
            query_tokens.append('[SEP]')

            ######
            # 정답에 ans 토큰을 임베딩하기 위한 코드
            ######
            pattern = '<[^>]*>'

            ans1 = '[STA]'

            doc_ = doc[0: answer_start] + ans1 + answer_text + doc[answer_start + len(answer_text): -1]
            doc_ = str(doc_)

            # """
            paragraphs = doc_.split('<h2>')
            tables = []
            for paragraph in paragraphs:
                tk = paragraph.split('</h2>')
                title = re.sub(pattern=pattern, repl='', string=tk[0]).replace('[편집]', '')

                sub_paragraphs = paragraph.split('<h3>')
                for sub_paragraph in sub_paragraphs:
                    if sub_paragraph.find('</h3>') != -1:
                        tk = sub_paragraph.split('</h3>')
                        title += ' ' + re.sub(pattern=pattern, repl='', string=tk[0]).replace('[편집]', '')

                    paragraph_, table_list = pre_process_document(sub_paragraph, answer_setting=False, a_token1='',
                                                                  a_token2='')

                    for table_text in table_list:
                        tables.append(table_text)

            # """

            texts = []
            new_doc = ''
            seq = ''

            splits = ['</ul>', '</p>', '</table>']

            for i in range(len(doc)):
                seq += doc_[i]

                if i > 5:
                    for spliter in splits:
                        if doc_[i - len(spliter): i] == spliter:
                            texts.append(seq)
                            seq = ''
            texts.append(seq)

            table_case = False

            for text in texts:
                if text.find('<table>') != -1 and text.find('[STA]') != -1:
                    text = text.replace('[STA]', '').replace('[END]', '')
                    text = text.replace('<table>', '[STA] <table> [END]')

                    table_case = True

                new_doc += text + ' '

            doc_ = doc_.replace('<table>', ' <table> ')
            doc_ = doc_.replace('</table>', ' </table> ')
            doc_ = doc_.replace('<td>', ' , ')
            doc_ = doc_.replace('</td>', ' ')
            doc_ = doc_.replace('<th>', ' , ')
            doc_ = doc_.replace('</th>', ' ')
            doc_ = doc_.replace('<ul>', ' <ul> ')
            doc_ = doc_.replace('</ul>', ' </ul> ')
            doc_ = doc_.replace('<li>', ' , ')
            doc_ = doc_.replace('</li>', ' ')
            doc_ = doc_.replace('<p>', ' ')
            doc_ = doc_.replace('</p>', ' ')
            #
            #####

            doc_ = process_document(doc_)
            # doc_ = doc_.replace('[table]', '[h2][table]')
            # doc_ = doc_.replace('[/table]', '[table][h2]')

            paragraphs = doc_.split('[h2]')

            sequences = []

            for paragraph in paragraphs:
                tokens = tokenizer.tokenize(paragraph)

                if len(tokens) < 32:
                    continue

                try:
                    title = paragraph.split('[/h2]')[0]
                    paragraph = paragraph.split('[/h2]')[1]
                except:
                    title = ''

                sub_paragraphs = paragraph.split('[h3]')

                sequence = ''
                total_length = 0
                temp_queue = []

                for sub_paragraph in sub_paragraphs:
                    tokens = tokenizer.tokenize(sub_paragraph)
                    if len(tokens) + len(query_tokens) > max_length:
                        sub_sentences = sub_paragraph.replace('. ', '.\n').split('\n')

                        for sentence in sub_sentences:
                            sentence += ' '
                            sequence += sentence
                            temp_queue.append(sentence)

                            tokens = tokenizer.tokenize(sentence)

                            if total_length + len(tokens) + len(query_tokens) + 30 >= max_length:
                                sequences.append(title + ' ' + sequence)

                                sequence = ' ' + sentence + ' '
                                total_length = 0

                                try:
                                    while True:
                                        temp_sequence = temp_queue.pop(-1)
                                        sequence = temp_sequence + ' ' + sequence
                                        total_length += len(tokenizer.tokenize(temp_sequence))

                                        if total_length > 196:
                                            break
                                except:
                                    None

                            total_length += len(tokens)
                        sequences.append(title + ' ' + sequence)
                    else:
                        sequences.append(title + ' ' + sub_paragraph)

            total_score = 0
            for sequence in sequences:
                score = chuncker.get_chunk_score(paragraph=sequence)
                total_score += score
            avg_score = total_score / len(sequences)

            sequences_ = sequences
            sequences = []

            for sequence in sequences_:
                score = chuncker.get_chunk_score(paragraph=sequence)

                # if score < avg_score:
                #    continue

                if sequence.find('위키') != -1 and sequence.find('목차') != -1 and sequence.find('[list]') != -1:
                    continue

                if len(sequence) > 80:
                    sequences.append(sequence)

            selected = 0
            scores = []
            for i, sequence in enumerate(sequences):
                score = chuncker.get_chunk_score(paragraph=sequence)
                if sequence.find('[STA') != -1:
                    selected += 1
                    scores.append(-9999)
                else:
                    scores.append(score)

            if selected == 0:
                false_count += 1
                continue

            scores = np.array(scores, dtype=np.float32)

            for k, sequence in enumerate(sequences):
                if sequence.find('[STA]') == -1:
                    continue

                if sequence.find('table') != -1:
                    continue

                answer_start = sequence.find('[STA]')
                sequence = sequence.replace('[STA]', '')

                try:
                    tokens, start_position, end_position = read_squad_example(orig_answer_text=answer_text,
                                                                              paragraph_text=sequence,
                                                                              answer_offset=answer_start)
                except:
                    false_count2 += 1
                    continue

                if tokens == -1:
                    false_count2 += 1
                    continue
                input_ids_arrays, input_segments_arrays, start_positions, end_positions, doc_texts, doc_tokens \
                    = convert_example_to_tokens(
                    question, start_position, end_position,
                    tokens, answer_text
                )
                chuncker.get_feautre(query=question)
                for s in range(len(input_ids_arrays)):
                    if start_positions[s] == 0:
                        continue
                    sequence_has_ans[count, 0] = input_ids_arrays[s]
                    segments_has_ans[count, 0] = input_segments_arrays[s]
                    answer_span[count, 0] = start_positions[s]
                    answer_span[count, 1] = end_positions[s]
                    tokens = doc_tokens[s]

                    # zero cases
                    for n in range(N_BATCH - 1):
                        zero_idx = scores.argmax()
                        scores[zero_idx] -= 99

                        seq_tokens = tokenizer.tokenize(sequences[zero_idx])
                        tokens = []
                        segments = [0] * len(query_tokens)
                        segments.extend([1] * len(seq_tokens))

                        tokens.extend(query_tokens)
                        tokens.extend(seq_tokens)

                        ids = tokenizer.convert_tokens_to_ids(tokens=tokens)

                        length = len(ids)
                        if length > max_length:
                            length = max_length
                        for j in range(length):
                            sequence_has_ans[count, n + 1, j] = ids[j]
                            segments_has_ans[count, n + 1, j] = segments[j]
                    count += 1

np.save('sequence_crs2.npy', sequence_has_ans[0:count])
np.save('segments_crs2.npy', segments_has_ans[0:count])
np.save('answer_span_crs2.npy', answer_span[0:count])
