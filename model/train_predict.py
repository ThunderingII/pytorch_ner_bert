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
    group.add_argument('--do_train', help="do_train",
                       action='store_true', default=False)
    group.add_argument('--do_predict', help="do_predict",
                       action='store_true', default=False)
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
    parser.add_argument('--batch_size', help="batch size od data",
                        type=int, default=32)
    parser.add_argument('--hidden_size', help="set the hidden size",
                        type=int, default=128)
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
    parser.add_argument('--mode_type',
                        help="choose transformer(t) or biLstm(b) or only crf(c)",
                        choices=['b', 't', 'c', 'bt'],
                        type=str, default='b')
    parser.add_argument('--bert_dim', help="bert dim",
                        type=int, default=768)
    parser.add_argument('--te_dropout', help="te dropout",
                        type=float, default=0.1)
    parser.add_argument('--lr', help="learning rate",
                        type=float, default=3e-4)
    parser.add_argument('--wd', help="weight decay",
                        type=float, default=5e-4)
    parser.add_argument('--head_num', help="set the head num",
                        type=int, default=8)
    parser.add_argument('--vip', help="the ip or domain of visdom server",
                        type=str, default='10.61.1.245')
    parser.add_argument('--env', help="the name of env of visdom",
                        type=str, default='ner')
    args = parser.parse_args()

    params['dropout'] = args.dropout
    params['use_glove'] = args.use_glove
    params['bert_dim'] = args.bert_dim
    params['mode_type'] = args.mode_type
    params['hidden_size'] = args.hidden_size
    # just for transformer
    params['te_dropout'] = args.te_dropout
    params['head_num'] = args.head_num

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

    tag_to_ix = {'O': 0, START_TAG: 1, STOP_TAG: 2}
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.wd)

    # begin to train model
    step_index = 0
    if args.do_train:
        # model, bert_dim, tag_to_ix, word_to_ix, rw, batch
        collate_fn = functools.partial(data_provider.collect_fn, model,
                                       params['bert_dim'],
                                       tag_to_ix, None, False)
        train_index = [i + 1 for i in range(args.train_ratio)]
        with bu.timer('load train data'):
            dataset = data_provider.BBNDatasetCombine(args.input,
                                                      train_index,
                                                      args.train_ratio + 1,
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
        best_state_dict = None
        loss_train_epoch = []
        loss_valid_epoch = []
        loss_train_t = []
        loss_train_valid = []

        f1_dev_org = []
        f1_dev_per = []
        f1_train_org = []
        f1_train_per = []
        for epoch in range(args.epochs):
            loss_train = []

            # index_batch, words_batch, words_ids_batch, len_w_batch, tags_batch
            for i, w, wi, l, t in data_loader:
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
                meo_, met_, loss_valid_ = evaluate(collate_fn, model, args,
                                                   'test', tag_to_ix,
                                                   idx_to_tag, True, True,
                                                   dataset_in=dataset)
                f1_train_org.append(meo_[args.monitor])
                f1_train_per.append(met_[args.monitor])
                loss_train_valid.append(np.mean(loss_valid_))

            meo, met, loss_valid = evaluate(collate_fn, model, args, 'test',
                                            tag_to_ix, idx_to_tag, True, True)

            loss_train_epoch.append(np.mean(loss_train))
            loss_valid_epoch.append(np.mean(loss_valid))
            f1_dev_org.append(meo[args.monitor])
            f1_dev_per.append(met[args.monitor])

            if sampler:
                legend = ['train_loss', 'valid_loss', 'f1_org', 'f1_per',
                          'f1_org_t', 'f1_per_t', 'train_loss_t']
                x_in = zip(loss_train_epoch, loss_valid_epoch, f1_dev_org,
                           f1_dev_per, f1_train_org, f1_train_per,
                           loss_train_valid)

            else:
                legend = ['train_loss', 'valid_loss', 'f1_org', 'f1_per']
                x_in = zip(loss_train_epoch, loss_valid_epoch, f1_dev_org,
                           f1_dev_per)
            plot(vis, x_in, args.model_name, legend)

            log.info(f'valid:{meo}{met}')
            if meo[args.monitor] > monitor_best or monitor_best == 0:
                monitor_best = meo[args.monitor]
                wait = 0
                best_state_dict = model.state_dict()
                if monitor_best:
                    fn = params['checkpoint'] + f'/{args.model_name}.pkl'
                    with Path(fn).open('wb') as mp:
                        pickle.dump((best_state_dict, params, tag_to_ix), mp)
            else:
                wait += 1

            if (epoch + 1) % 5 == 0:
                fn = params['checkpoint'] + f'/si_{epoch+1}.pkl'
                with Path(fn).open('wb') as mp:
                    pickle.dump((best_state_dict, params, tag_to_ix), mp)
                ratio = args.train_ratio
                if ratio > 0:
                    meo, met = evaluate(collate_fn, model, args, 'test',
                                        tag_to_ix, idx_to_tag, True, False)
                    log.info(f"train(f1,p,r):{meo}{met}")
            if wait > 4:
                log.warn(f'meat early stopping! best score is {monitor_best}')
                break
        log.info('finish train')
    if args.do_predict:
        log.info('begin predict')
        fn = params['checkpoint'] + f'/{args.model_name}.pkl'
        with Path(fn).open('rb') as mp:
            best_state_dict, params, tag_to_ix = pickle.load(mp)

        # change batch_size to 1
        args.batch_size = 1

        if args.gpu_index > -1:
            device = torch.device(f'cuda:{args.gpu_index}')
        else:
            device = torch.device('cpu')
        model = Bert_CRF(tag_to_ix, params, device)
        model.to(device)
        model.load_state_dict(best_state_dict, strict=False)

        # model, bert_dim, tag_to_ix, word_to_ix, rw, batch
        collate_fn = functools.partial(data_provider.collect_fn, model,
                                       params['bert_dim'],
                                       tag_to_ix, None, True)
        log.warn(f"{'-'*25}test_valid{'-'*25}")
        meo, met = evaluate(collate_fn, model, args, 'test',
                            tag_to_ix, idx_to_tag, True, False,
                            f"{args.output}/{args.model_name}.txt",
                            )
        log.info(f'org:{meo} per:{met}')


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
        loss = []
        correct_preds_org = 0
        total_preds_org = 0
        total_correct_org = 0

        correct_preds_per = 0
        total_preds_per = 0
        total_correct_per = 0

        finish_size = 0
        for ots, w, wi, l, t in data_loader:
            if (finish_size + 1) % 1000 == 0:
                print(f'finish {finish_size + 1}')

            finish_size += args.batch_size

            s, p = model(w, wi, l)
            if get_loss:
                loss.append(
                    model.neg_log_likelihood(w, wi, l, t).mean().item())
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
    r = []
    if fpr:
        r.append(metrics.get_fpr(correct_preds_org, total_correct_org,
                                 total_preds_org))
        r.append(metrics.get_fpr(correct_preds_per, total_correct_per,
                                 total_preds_per))
    if get_loss:
        r.append(loss)
    return r


if __name__ == '__main__':
    main()
