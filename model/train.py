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
    parser.add_argument('--undo_train_valid', help="undo train data as valid",
                        action='store_true', default=False)
    parser.add_argument('--input', help="input dir or file",
                        type=str, required=True)
    parser.add_argument('--valid_input', help="valid data input dir or file",
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
    parser.add_argument('--batch_size', help="batch size od data",
                        type=int, default=32)
    parser.add_argument('--hidden_size', help="set the hidden size",
                        type=int, default=128)
    parser.add_argument('--epochs', help="epochs of train",
                        type=int, default=100)

    parser.add_argument('--monitor',
                        help="monitor f1,acc,precision or recall, "
                             "value like ORG:f1 or PER:acc or LOC:recall",
                        type=str, default='ORG:f1')
    parser.add_argument('--use_glove', help="denote whether use use_glove",
                        type=bool, default=False)
    parser.add_argument('--model_name', help="file name of model file",
                        type=str, default='ner_model_crf')
    parser.add_argument('--mode_type',
                        help="choose transformer(t) or biLstm(b) or only crf(c)",
                        choices=['b', 't', 'c', 'bt', 'cnn'],
                        type=str, default='b')
    parser.add_argument('--bert_dim', help="bert dim",
                        type=int, default=768)
    parser.add_argument('--te_dropout', help="te dropout",
                        type=float, default=0.1)
    parser.add_argument('--lr', help="learning rate",
                        type=float, default=3e-4)
    parser.add_argument('--lr_times', help="learning rate decay times",
                        type=int, default=0)
    parser.add_argument('--wd', help="weight decay",
                        type=float, default=1e-3)
    parser.add_argument('--head_num', help="set the head num",
                        type=int, default=8)
    parser.add_argument('--vip', help="the ip or domain of visdom server",
                        type=str, default='')
    parser.add_argument('--env', help="the name of env of visdom",
                        type=str, default='ner')

    parser.add_argument('--pre_model_path', help="the pre model path",
                        type=str, default='')
    parser.add_argument('--use_cross_entropy', help="use cross entropy loss",
                        action='store_true', default=False)
    args = parser.parse_args()

    params['dropout'] = args.dropout
    params['use_glove'] = args.use_glove
    params['bert_dim'] = args.bert_dim
    params['mode_type'] = args.mode_type
    params['hidden_size'] = args.hidden_size
    # just for transformer
    params['te_dropout'] = args.te_dropout
    params['head_num'] = args.head_num
    params['use_cross_entropy'] = args.use_cross_entropy

    model_time_str = args.model_name + '_' + bu.get_time_str()

    log = bu.get_logger(model_time_str)

    if args.vip:
        vis = visdom.Visdom(args.vip, env=args.env)
    else:
        vis = None

    word_to_ix = {'<pad>': 0}
    if params['use_glove']:
        with open(params['words']) as wvf:
            for word in wvf:
                word = word.strip()
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {'O': 0}
    with open(params['tags']) as wvf:
        for tag in wvf:
            tag = tag.strip()
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    idx_to_tag = {tag_to_ix[key]: key for key in tag_to_ix}

    if args.gpu_index > -1:
        device = torch.device(f'cuda:{args.gpu_index}')
    else:
        device = torch.device('cpu')

    model = Bert_CRF(tag_to_ix, params, device)
    model.to(device)

    if args.pre_model_path:
        with Path(args.pre_model_path).open('rb') as mp:
            if args.gpu_index < 0:
                ml = 'cpu'
            else:
                ml = None
            best_state_dict = torch.load(mp, map_location=ml)
            model.load_state_dict(best_state_dict, False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.wd)

    # begin to train model
    step_index = 0

    # model, bert_dim, tag_to_ix, word_to_ix, rw, batch
    collate_fn = functools.partial(data_provider.collect_fn, model,
                                   params['bert_dim'], tag_to_ix, None,
                                   False)
    with bu.timer('load train data'):
        dataset = data_provider.BBNDatasetCombine(args.input,
                                                  args.limit)
    data_loader = tud.DataLoader(dataset, args.batch_size,
                                 shuffle=True, collate_fn=collate_fn,
                                 drop_last=True)

    if not args.undo_train_valid:
        sampler = tud.RandomSampler(data_source=dataset,
                                    replacement=True,
                                    num_samples=5000)
    else:
        sampler = None

    log.info('begin to train')
    Path(params['checkpoint']).mkdir(parents=True, exist_ok=True)
    monitor_best = 0
    wait = 0
    loss_train_epoch = []
    loss_valid_epoch = []
    loss_train_t = []
    loss_train_valid = []
    criterion_key = ['f1', 'precision', 'recall']
    criterion_map = {}

    lr_times = args.lr_times
    lr = args.lr
    for epoch in range(args.epochs):
        loss_train = []

        # index_batch, words_batch, words_ids_batch, len_w_batch, tags_batch
        # sentence_batch
        for i, w, wi, l, t, _ in data_loader:
            # Step 1. Remember that Pytorch accumulates gradients.
            model.zero_grad()
            # Step 2. Run our forward pass.
            # words, words_ids, len_w, tags
            loss = model.neg_log_likelihood(w, wi, l, t)
            # Step 3. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            ls = loss.mean()
            ls.backward()
            optimizer.step()
            step_index += 1
            step_loss = ls.item()
            log.info(
                f'global step:{step_index} epoch:{epoch} loss:{step_loss}')
            loss_train.append(step_loss)
            loss_train_t.append(step_loss)
            plot(vis, loss_train_t, args.model_name, ['train_loss'])

        if sampler:
            # collate_fn, model, args, tag_to_ix = None, idx_to_tag = None,
            # fpr = True, get_loss = False, input_dir = None, dataset_in = None,
            # sampler = None
            criterion, loss_valid_ = evaluate(collate_fn, model, args,
                                              tag_to_ix, idx_to_tag,
                                              True, True,
                                              dataset_in=dataset,
                                              sampler=sampler)
            for k in criterion:
                # ['f1', 'precision', 'recall']
                for ck in criterion_key:
                    key = f'train_{k}_{ck}'
                    if key not in criterion_map:
                        criterion_map[key] = []
                    criterion_map[key].append(criterion[k][ck])
            loss_train_valid.append(np.mean(loss_valid_))

        criterion, loss_valid = evaluate(collate_fn, model, args,
                                         tag_to_ix, idx_to_tag, True, True,
                                         input_dir=args.valid_input)
        loss_train_epoch.append(np.mean(loss_train))
        loss_valid_epoch.append(np.mean(loss_valid))

        for k in criterion:
            # ['f1', 'precision', 'recall']
            for ck in criterion_key:
                key = f'valid_{k}_{ck}'
                if key not in criterion_map:
                    criterion_map[key] = []
                criterion_map[key].append(criterion[k][ck])
        plot_data = []
        keys = list(criterion_map.keys())
        for k in criterion_map:
            plot_data.append(criterion_map[k])
        if sampler:
            legend = ['train_loss', 'valid_loss',
                      'train_loss_t'] + keys
            x_in = zip(loss_train_epoch, loss_valid_epoch,
                       loss_train_valid, *plot_data)
        else:
            legend = ['train_loss', 'valid_loss'] + keys
            x_in = zip(loss_train_epoch, loss_valid_epoch, *plot_data)
        plot(vis, x_in, args.model_name, legend)

        log.info(f'valid:{criterion}')
        tag_type, monitor_type = args.monitor.split(':')
        if (criterion[tag_type][monitor_type] > monitor_best
                or monitor_best == 0):
            monitor_best = criterion[tag_type][monitor_type]
            wait = 0
            best_state_dict = model.state_dict()
            if monitor_best:
                save_mode(best_state_dict, params, tag_to_ix, args.model_name)
        else:
            wait += 1
        if (epoch + 1) % 5 == 0:
            temp_name = f't_{args.model_name}_{epoch+1}'
            save_mode(model.state_dict(), params, tag_to_ix, temp_name)
        if wait > 8:
            if lr_times:
                lr_times -= 1
                wait = 3
                lr /= 3
                optimizer = optim.Adam(model.parameters(), lr=lr,
                                       weight_decay=args.wd)
            else:
                log.warn(f'meat early stopping! best score is {monitor_best}')
                break
        log.info('finish train')


def save_mode(best_state_dict, params, tag_to_ix, model_name):
    fn_config = params['checkpoint'] + f'/{model_name}_config.pkl'
    fn_model = params['checkpoint'] + f'/{model_name}_torch.pkl'
    with open(fn_model, 'wb') as f:
        torch.save(best_state_dict, f)
    with open(fn_config, 'wb') as f:
        pickle.dump((params, tag_to_ix), f)


def plot(vis, x, model_name, legend=None):
    if vis:
        x = [item for item in x]
        vis.line(
            X=np.arange(start=0, stop=len(x)),
            Y=x,
            opts={
                'legend': legend,
                'title': f'{model_name} loss line{len(legend)}',
                'ylabel': 'loss',
                'xlabel': 'step index',

            },
            win=f'loss_{"".join(legend)}_{model_name}',
        )


dataset_map = {}


def evaluate(collate_fn, model, args, tag_to_ix=None, idx_to_tag=None,
             fpr=True, get_loss=False, input_dir=None, dataset_in=None,
             sampler=None):
    with torch.no_grad():
        batch_size = args.batch_size
        limit = args.limit
        if dataset_in:
            dataset_ = dataset_in
        else:
            data_id = f'{input_dir}_{limit}'
            if data_id in dataset_map:
                dataset_ = dataset_map[data_id]
            else:
                # input_dir, limit=0, from_one_file=True
                dataset_ = data_provider.BBNDatasetCombine(input_dir, limit)
                dataset_map[data_id] = dataset_

        data_loader = tud.DataLoader(dataset_, batch_size,
                                     shuffle=False, collate_fn=collate_fn,
                                     drop_last=True, sampler=sampler)
        loss = []
        data_analysis_map = {}
        finish_size = 0
        for ots, w, wi, l, t, sentences in data_loader:
            if (finish_size + 1) % 1000 == 0:
                print(f'finish {finish_size + 1}')
            finish_size += batch_size
            s, p = model(w, wi, l)
            if get_loss:
                loss.append(
                    model.neg_log_likelihood(w, wi, l, t).mean().item())
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

    criterion = {}
    if fpr:
        # print f1 precision and recall
        for k in data_analysis_map:
            criterion[k] = metrics.get_fpr(*data_analysis_map[k])
    return criterion, loss


if __name__ == '__main__':
    main()
