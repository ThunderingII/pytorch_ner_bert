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
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--undo_train_valid', help="undo train data as valid",
                        action='store_true', default=False)

    parser.add_argument('--input', help="input dir or file",
                        type=str, required=True)
    parser.add_argument('--output', help="output file dir for writing result",
                        type=str, default=params['output_dir'])
    parser.add_argument('--limit', help="if use data limit",
                        type=int, default=0)
    parser.add_argument('--gpu_index', help="gpu index must>-1,if use gpu",
                        type=int, default=0)
    parser.add_argument('--dropout',
                        help="dropout rate in embed and liner layer",
                        type=float, default=0.2)
    parser.add_argument('--epochs', help="epochs of train",
                        type=int, default=100)
    parser.add_argument('--train_ratio',
                        help="the value of train size/test size",
                        type=int, default=5)
    parser.add_argument('--monitor', help="monitor f1,acc,precision or recall",
                        type=str, default='f1')
    parser.add_argument('--use_glove', help="denote whether use use_glove",
                        type=bool, default=False)
    parser.add_argument('--model_name', help="file name of model file",
                        type=str, default='ner_model_crf')

    parser.add_argument('--pre_model_path', help="the pre model path",
                        type=str, default='')
    args = parser.parse_args()

    model_time_str = args.model_name + '_' + bu.get_time_str()

    log = bu.get_logger(model_time_str)

    word_to_ix = {'<pad>': 0}
    if params['use_glove']:
        with open(params['words']) as wvf:
            for word in wvf:
                word = word.strip()
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

    log.info('begin predict')
    fn = params['checkpoint'] + f'/{args.model_name}.pkl'
    with Path(fn).open('rb') as mp:
        best_state_dict, params, tag_to_ix = pickle.load(mp)
    print(tag_to_ix)
    idx_to_tag = {tag_to_ix[key]: key for key in tag_to_ix}

    if args.gpu_index > -1:
        device = torch.device(f'cuda:{args.gpu_index}')
    else:
        device = torch.device('cpu')
    model = Bert_CRF(tag_to_ix, params, device)
    model.to(device)
    model.load_state_dict(best_state_dict, strict=False)

    train_index = [i + 1 for i in range(args.train_ratio)]
    with bu.timer('load train data'):
        dataset = data_provider.BBNDatasetCombine(args.input,
                                                  train_index,
                                                  args.train_ratio + 1,
                                                  args.limit)
    # change batch_size to 1
    args.batch_size = 1

    # model, bert_dim, tag_to_ix, word_to_ix, rw, batch
    collate_fn = functools.partial(data_provider.collect_fn, model,
                                   params['bert_dim'],
                                   tag_to_ix, None, True)
    log.warn(f"{'-'*25}test_valid{'-'*25}")
    evaluate(collate_fn, model, args, 'test',
             tag_to_ix, idx_to_tag, True, False,
             f"{args.output}/{args.model_name}.txt",
             dataset_in=dataset
             )


dataset_map = {}


def evaluate(collate_fn, model, args, valid_status, tag_to_ix=None,
             idx_to_tag=None, fpr=True, get_loss=False, output_file=None,
             dataset_in=None, sampler=None):
    with torch.no_grad():

        if dataset_in:
            dataset_ = dataset_in
        else:
            input_dir = args.input
            limit = args.limit
            ratio = args.train_ratio
            if valid_status != 'train' and ratio > 0:
                limit //= ratio
                mod_index = [0]
            else:
                mod_index = [i + 1 for i in range(ratio)]
            data_id = f'{input_dir}_{ratio}_{limit}_{mod_index}'
            if data_id in dataset_map:
                dataset_ = dataset_map[data_id]
            else:
                # input_dir, mode='train', get_indexs=None, num=1, limit=0
                dataset_ = data_provider.BBNDatasetCombine(input_dir,
                                                           mod_index,
                                                           ratio + 1, limit)
                dataset_map[data_id] = dataset_

        data_loader = tud.DataLoader(dataset_, args.batch_size,
                                     shuffle=False, collate_fn=collate_fn,
                                     drop_last=True, sampler=sampler)
        ss = []
        ss_error = []
        correct_preds_org = 0
        total_preds_org = 0
        total_correct_org = 0

        correct_preds_per = 0
        total_preds_per = 0
        total_correct_per = 0

        finish_size = 0
        for index_batch, ots, w, wi, l, t in data_loader:
            if (finish_size + 1) % 1000 == 0:
                print(f'finish {finish_size + 1}')

            finish_size += args.batch_size
            s, p = model(w, wi, l)
            for i, ot in enumerate(ots):

                if fpr:
                    orgs_t_set = set()
                    orgs_p_set = set()
                    pers_t_set = set()
                    pers_p_set = set()

                    oc = metrics.get_chunks([tag_to_ix[tid] for tid in ot],
                                            tag_to_ix, idx_to_tag)
                    pc = metrics.get_chunks(p[i], tag_to_ix, idx_to_tag)

                    for c in oc:
                        if c[0] == 'ORG':
                            orgs_t_set.add(c)
                        else:
                            pers_t_set.add(c)

                    for c in pc:
                        if c[0] == 'ORG':
                            orgs_p_set.add(c)
                        else:
                            pers_p_set.add(c)

                    correct_preds_org += len(orgs_t_set & orgs_p_set)
                    total_preds_org += len(orgs_p_set)
                    total_correct_org += len(orgs_t_set)

                    correct_preds_per += len(pers_t_set & pers_p_set)
                    total_preds_per += len(pers_p_set)
                    total_correct_per += len(pers_t_set)

                if output_file:
                    text = ''.join(w[i][:len(ot)])
                    ss.append(text)
                    if fpr and orgs_t_set != orgs_p_set:
                        ss_error.append(text)
                        ts = []
                        ps = []
                        for c in orgs_p_set - orgs_t_set:
                            ps.append(''.join(w[i][c[1]:c[2]]))
                        for c in orgs_t_set - orgs_p_set:
                            ts.append(''.join(w[i][c[1]:c[2]]))
                        ss_error.append(f'tc:{orgs_t_set}, pc:{orgs_p_set}')
                        ss_error.append(f'true:{ts}')
                        ss_error.append(f'pre:{ps}')
                        ss_error.append('\n')

                    pre_tags = [idx_to_tag[tid] for tid in p[i]]
                    if get_loss:
                        for a, b, c in zip(w[i][:len(ot)], ot,
                                           [idx_to_tag[tid] for tid in p[i]]):
                            if b != 'O':
                                ss.append((a, b, c))
                        ss.append(f'origin tag:{ot}')
                    ss.append(f'score:{s[i]}    pre tag:{pre_tags}')
                    ss.append('\n')
        if output_file:
            with open(output_file, 'w') as out:
                for s in ss_error:
                    out.write(f'{s}\n')

                out.write('*' * 20 + 'begin result' + '*' * 20 + '\n')

            with open(output_file, 'a') as out:
                for s in ss:
                    out.write(f'{s}\n')
    return None


if __name__ == '__main__':
    main()
