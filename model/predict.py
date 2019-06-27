import re
import shutil
import math
import argparse
import functools
import pickle
import visdom
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
import numpy as np

import util.base_util as bu
import model.data_provider as data_provider
import util.metrics as metrics
from model.ner_bert_glove_crf_pytorch import Bert_CRF

DATA_DIR = '../data'
RESULT_DIR = '../result'

START_TAG = '<START>'
STOP_TAG = '<STOP>'

UNK = '✌'

torch.manual_seed(2019)


def main():
    params = {
        'output_dir': str(Path(RESULT_DIR, 'res_torch')),
        'checkpoint': str(Path(RESULT_DIR, 'res_torch/model')),
        'glove_dim': 300,
        'vocab_tags': str(Path(DATA_DIR, 'processed/vocab.tags.txt')),
        'glove': str(Path(DATA_DIR, 'embedding/glove.npz')),
        'words': str(Path(DATA_DIR, 'processed/vocab.words.txt')),
        'tags': str(Path(DATA_DIR, 'processed/vocab.tags.txt')),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="input dir or file",
                        type=str, required=True)
    parser.add_argument('--output', help="output file dir for writing result",
                        type=str, default=params['output_dir'])
    parser.add_argument('--limit', help="if use data limit",
                        type=int, default=0)
    parser.add_argument('--gpu_index', help="gpu index must>-1,if use gpu",
                        type=int, default=0)
    parser.add_argument('--model_name', help="file name of model file",
                        type=str, default='ner_model_crf')
    args = parser.parse_args()

    model_time_str = args.model_name + '_' + bu.get_time_str()

    log = bu.get_logger(model_time_str)

    log.info('begin predict')
    fn_model = params['checkpoint'] + f'/{args.model_name}_torch.pkl'
    fn_config = params['checkpoint'] + f'/{args.model_name}_config.pkl'
    with Path(fn_model).open('rb') as mp:
        if args.gpu_index < 0:
            ml = 'cpu'
        else:
            ml = None
        best_state_dict = torch.load(mp, map_location=ml)
    with Path(fn_config).open('rb') as mp:
        params, tag_to_ix = pickle.load(mp)
    print(tag_to_ix)
    idx_to_tag = {tag_to_ix[key]: key for key in tag_to_ix}
    if args.gpu_index > -1:
        device = torch.device(f'cuda:{args.gpu_index}')
    else:
        device = torch.device('cpu')
    model = Bert_CRF(tag_to_ix, params, device)
    model.to(device)
    model.load_state_dict(best_state_dict, strict=False)

    with bu.timer('load data'):
        dataset = data_provider.BBNDatasetCombine(args.input, args.limit)
    # change batch_size to 1
    args.batch_size = 1

    # model, bert_dim, tag_to_ix, word_to_ix, rw, batch
    collate_fn = functools.partial(data_provider.collect_fn, model,
                                   params['bert_dim'], tag_to_ix, None, True)
    log.warn(f"{'-'*25}test_valid{'-'*25}")
    evaluate(collate_fn, model, args, tag_to_ix, idx_to_tag, True, False,
             f"{args.output}/{args.model_name}.txt", dataset_in=dataset)


def get_entity(entity, text):
    try:
        m = re.search(entity.replace('[UNK]', '.'), text)
    except:
        print(entity, text)
        m = None
    if m:
        entity = m.group()
    return entity


dataset_map = {}


def evaluate(collate_fn, model, args, tag_to_ix=None, idx_to_tag=None,
             fpr=True, get_loss=False, output_file=None,
             dataset_in=None, sampler=None):
    with torch.no_grad():

        if dataset_in:
            dataset_ = dataset_in
        else:
            input_dir = args.valid_input
            limit = args.limit
            data_id = f'{input_dir}_{limit}'
            if data_id in dataset_map:
                dataset_ = dataset_map[data_id]
            else:
                # input_dir, limit=0, from_one_file=True
                dataset_ = data_provider.BBNDatasetCombine(input_dir, limit)
                dataset_map[data_id] = dataset_

        data_loader = tud.DataLoader(dataset_, args.batch_size,
                                     shuffle=False, collate_fn=collate_fn,
                                     drop_last=True, sampler=sampler)
        ss = []
        ss_error = []
        data_analysis_map = {}

        finish_size = 0
        # i, w, wi, l, t, _
        for ots, w, wi, l, t, sts in data_loader:
            if (finish_size + 1) % 1000 == 0:
                print(f'finish {finish_size + 1}')

            finish_size += args.batch_size
            s, p = model(w, wi, l)
            for i, ot in enumerate(ots):
                set_map = {}
                if fpr:
                    oc = metrics.get_chunks([tag_to_ix[tid] for tid in ot],
                                            tag_to_ix, idx_to_tag)
                    pc = metrics.get_chunks(p[i], tag_to_ix, idx_to_tag)
                    # ground truth
                    for c in oc:
                        if c[0] not in set_map:
                            # 0:ground truth, 1:predict
                            set_map[c[0]] = [set(), set()]
                        set_map[c[0]][0].add(c)
                    # predict
                    for c in pc:
                        if c[0] not in set_map:
                            set_map[c[0]] = [set(), set()]
                        set_map[c[0]][1].add(c)

                    for k in set_map:
                        if k not in data_analysis_map:
                            # true and positive, true, positive
                            data_analysis_map[k] = [0, 0, 0]
                        data_analysis_map[k][0] += len(
                            set_map[k][0] & set_map[k][1])
                        data_analysis_map[k][1] += len(set_map[k][0])
                        data_analysis_map[k][2] += len(set_map[k][1])

                if output_file:
                    text = sts[i]
                    ss.append(text)
                    for k in set_map:
                        en_li = [k + ': ']
                        for c in set_map[k][1]:
                            entity = get_entity(''.join(w[i][c[1]:c[2]]), text)
                            en_li.append((len(en_li) + 1, c, entity))
                        ss.append(en_li)
                    if fpr:
                        text_not_in = True
                        for k in set_map:
                            if set_map[k][0] != set_map[k][1]:
                                if text_not_in:
                                    ss_error.append(text)
                                    text_not_in = False
                                ts = []
                                ps = []
                                for c in set_map[k][1] - set_map[k][0]:
                                    entity = get_entity(
                                        ''.join(w[i][c[1]:c[2]]), text)
                                    ps.append(entity)
                                for c in set_map[k][0] - set_map[k][1]:
                                    entity = get_entity(
                                        ''.join(w[i][c[1]:c[2]]), text)
                                    ts.append(entity)
                                ss_error.append(
                                    f'{k}: tc:{set_map[k][0]}, pc:{set_map[k][1]}')
                                ss_error.append(f'{k}: true:{ts}')
                                ss_error.append(f'{k}: pre:{ps}')
                    pre_tags = [idx_to_tag[tid] for tid in p[i]]
                    if get_loss:
                        for a, b, c in zip(w[i][:len(ot)], ot,
                                           [idx_to_tag[tid] for tid in p[i]]):
                            if b != 'O':
                                ss.append((a, b, c))
                        ss.append(f'origin tag:{ot}')
                    ss.append(f'score:{s[i]}    pre tag:{pre_tags}')
                    ss.append('\n')
        # print f1 precision and recall
        for k in data_analysis_map:
            print(k, ': ', metrics.get_fpr(*data_analysis_map[k]))

        if output_file:
            with open(output_file, 'w') as out:
                for s in ss_error:
                    out.write(f'{s}\n')
                out.write('*' * 20 + 'begin result' + '*' * 20 + '\n')
            with open(output_file, 'a') as out:
                for s in ss:
                    out.write(f'{s}\n')


if __name__ == '__main__':
    main()
