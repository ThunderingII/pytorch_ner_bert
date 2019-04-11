"""
This file contains all the methon used to generate data for
model train and test.
"""

__author__ = "Zhang Lin"

import json
import argparse
import collections
from pathlib import Path

import pickle
import numpy as np

from util import base_util as bu

INPUT_DIR = '../data/origin/news/'
DATA_DIR = '../data'


def load_bert_embedding(file, embeddings, tags, tokens, mode='sum'):
    """Load bert embedding form json file.

    :param file: bert embedding json file
    :param mode: 'average', 'sum, 'concat'...
    :return: dict of {linex: embedding}, shape=seqence_length*[768]
    """
    # embeddings = collections.OrderedDict()
    # tags = collections.OrderedDict

    index = 0
    with open(file, 'rb') as f:
        out = pickle.load(f)
        for tmp in out:
            if (index + 1) % 1000 == 0:
                print(f'load index: {index}')
            if LIMITED > 0 and len(embeddings) >= LIMITED:
                break
            index += 1
            line_index = tmp['linex_index']

            seq_embed = []
            seq_tokens = []
            for i, token_embedding in enumerate(tmp['features']):
                # remove [CLS] and [SEP]
                if i == 0 or i == len(tmp['features']) - 1:
                    continue
                t1 = token_embedding['layers'][0]['values']
                t2 = token_embedding['layers'][1]['values']
                t3 = token_embedding['layers'][2]['values']
                t4 = token_embedding['layers'][3]['values']
                t = []
                if mode == 'average':
                    t = np.average([t1, t2, t3, t4], axis=0)
                elif mode == 'sum':
                    t = np.sum([t1, t2, t3, t4], axis=0)
                seq_embed.append(t)
                seq_tokens.append(token_embedding['token'])
            embeddings[line_index] = seq_embed
            tags[line_index] = (tmp['orgs'], tmp['pers'])
            tokens[line_index] = seq_tokens

    print(f'load {index} lines')


def generator_fn(embeddings, tags, tokens):
    for k in embeddings.keys():
        line_embedding = embeddings[k]
        line_tags = tags[k]
        line_tokens = tokens[k]
        embedding_e = []
        tag_e = []
        token_e = []

        assert len(line_embedding) == len(
            line_tokens), f"embeddings({len(line_embedding)}) and tokens({len(line_tokens)}) lengths don't match"

        remove_size = 1
        append_to = False
        for i in range(len(line_tokens)):
            if i == 0 or i == len(line_tokens) - 1:
                continue
            current_word_emb = line_embedding[i]
            token = line_tokens[i]

            if append_to:
                embedding_e[-1] = np.sum(
                    (embedding_e[-1], current_word_emb), axis=0)
                token_e[-1] = token_e[-1][:-2] + token
                remove_size += 1
            else:
                embedding_e.append(current_word_emb)
                token_e.append(token)
                for s, e in line_tags[0]:
                    if s - remove_size == i:
                        tag_e.append('B-ORG')
                    elif s - remove_size < i and i < e - remove_size:
                        tag_e.append('I-ORG')
                for s, e in line_tags[1]:
                    if s - remove_size == i:
                        tag_e.append('B-PER')
                    elif s - remove_size < i and i < e - remove_size:
                        tag_e.append('I-PER')
                if len(tag_e) != len(token_e):
                    tag_e.append('O')
            append_to = token.endswith('##')

        try:
            assert len(embedding_e) == len(tag_e) == len(
                token_e), "Words and tags lengths don't match"
        except Exception as e:
            print('-' * 50)
            print(e)
            print(len(line_embedding))
            print(line_tokens)
            print(line_tags)
            print(len(embedding_e))
            print(token_e)
            print(tag_e)
            print('*' * 50)
            continue
        yield k, embedding_e, token_e, tag_e, len(token_e)


def main(limit):
    """
    This method is used to generate processed data for train and test.

    :return: No return, only write files
             (1) {mode}_{LIMITED}_wfd.pkl
                write index, words, tags, len_w
             (2) {mode}_{LIMITED}_wbd.pkl
                write bert embedding in line

    """
    # change LIMITED
    global LIMITED
    LIMITED = limit

    modes = ['train', 'valid', 'test']
    mode_map = {'train': [0, 1, 2, 3, 4], 'valid': [5], 'test': [6]}
    mod_num = 7

    import os
    for mode in modes:

        if mode == 'valid':
            LIMITED = LIMITED // 5

        word_flag_data = []
        word_bert_emb_data = []
        origin_bert_emb = collections.OrderedDict()
        tags = collections.OrderedDict()
        tokens = collections.OrderedDict()
        for f in os.listdir(INPUT_DIR):
            if 'json' not in f and bu.get_str_index(f, mod_num) in mode_map[
                mode]:
                data_file = INPUT_DIR + f
                if LIMITED > 0 and len(origin_bert_emb) < LIMITED:
                    with bu.timer(f'load {data_file} bert emb'):
                        load_bert_embedding(data_file, origin_bert_emb, tags,
                                            tokens)

        index = 0

        for i, words_emb_bert, words, tags, len_w in generator_fn(
                origin_bert_emb, tags, tokens):
            # used to generate small dataset if set LIMITED's value

            if LIMITED > 0 and index >= LIMITED:
                break

            if words_emb_bert is None:
                continue
            if i % 1000 == 0:
                print(f'{mode} index:{i} finished!')
            assert len(words_emb_bert) == len(
                words) == len_w, f'length not match in {i},' \
                                 f'{len(words_emb_bert)}-{len(words)}-{len_w}'
            word_bert_emb_data.append(words_emb_bert)
            word_flag_data.append((index, words, tags, len_w))
            index += 1

        with open(DATA_DIR + f'/processed/{mode}_{LIMITED}_wfd.pkl',
                  'wb') as wfd, open(
            DATA_DIR + f'/processed/{mode}_{LIMITED}_wbd.pkl', 'wb') as wbd:
            with bu.timer(f'write {mode} to file'):
                # each line is (index, words, tags, len_w)
                pickle.dump(word_flag_data, wfd)
                # each line is word's bert context embedding
                pickle.dump(word_bert_emb_data, wbd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', help="max num select from origin data",
                        type=int, default=30000)
    args = parser.parse_args()
    main(args.limit)
