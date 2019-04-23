import os
import torch
import numpy as np
import pickle
import torch.utils.data as tud

DATA_DIR = '../../data'
BERT_IDR = '../../bert'

LIMITED = 0
import util.base_util as bu


class BBNDataset(tud.Dataset):
    def __init__(self, mode, limit=0):
        with open(
                DATA_DIR + f'/processed/{mode}_{limit}_wfd.pkl',
                'rb') as wfd:
            self.word_flag_data = pickle.load(wfd)
        with open(
                DATA_DIR + f'/processed/{mode}_{limit}_wbd.pkl',
                'rb') as wbd:
            self.word_bert_emb_data = pickle.load(wbd)

    def __getitem__(self, index):
        # index, words, words_emb_bert, tags, len_w
        i, words, tags, len_w = self.word_flag_data[index]
        words_emb_bert = self.word_bert_emb_data[index]
        return (i, words, words_emb_bert, tags, len_w)

    def __len__(self):
        return len(self.word_flag_data)


class BBNDatasetCombine(tud.Dataset):
    def __init__(self, input_dir, get_indexs=None, num=1, limit=0, kw=True,
                 from_one_file=True):
        self.data = []
        inputs = []
        log = bu.get_logger()

        if os.path.isfile(input_dir):
            inputs.append(input_dir)
        else:
            for input_file in os.listdir(input_dir):
                if kw:
                    str_for_mod = input_file.split('.pkl')[0]
                else:
                    str_for_mod = input_file
                if num <= 1 or bu.get_str_index(str_for_mod,
                                                num) in get_indexs:
                    inputs.append(input_dir + '/' + input_file)

        one_file_limit = limit // len(inputs)
        for input_file in inputs:
            if one_file_limit > 0 and len(self.data) >= limit:
                break

            with open(input_file, 'rb') as wfd:
                log.info(input_file)
                if one_file_limit > 0:
                    self.data.extend(pickle.load(wfd)[:one_file_limit])
                else:
                    self.data.extend(pickle.load(wfd))
                if limit > 0 and len(self.data) > limit:
                    self.data = self.data[:limit]

    def __getitem__(self, index):
        # index, words, words_emb_bert, tags, len_w
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collect_fn(model, bert_dim, tag_to_ix, word_to_ix, rw, batch):
    """
    Used to process data when generate a batch
    :param model: model to load embedding
    :param bert_dim: bert dim
    :param word_to_ix: map, word to id
    :param tag_to_ix: map, tag to id
    :param batch: origin data from dataset
    :return: batched data
    """
    batch_size = len(batch)
    len_w_batch = np.zeros((batch_size,), dtype=np.int64)
    size = 1
    len_summed = 0
    max_len = 0
    for i in range(batch_size):
        len_w = batch[i][4]
        len_summed += len_w
        max_len = max(len_w, max_len)
        len_w_batch[i] = len_w

    # sort the array by length, sort the len_w by length
    # ([10,2,5],[0,1,2]) after sort ([10,5,2],[0,2,1])
    rz = zip(len_w_batch, [i for i in range(len(len_w_batch))])
    r = sorted(rz, key=lambda item: item[0], reverse=True)
    _, index_list = zip(*r)

    bert_embedding = np.zeros((len_summed + 1, bert_dim))

    origin_tags = [None for _ in range(batch_size)]

    if word_to_ix:
        words_batch = np.zeros((batch_size, max_len), dtype=np.int64)
    else:
        words_batch = [None for _ in range(batch_size)]
    words_ids_batch = np.zeros((batch_size, max_len), dtype=np.int64)
    tags_batch = np.zeros((batch_size, max_len), dtype=np.int64)

    for ni, bi in enumerate(index_list):
        _, words, words_emb_bert, tags, len_w = batch[bi]
        for i in range(len_w):
            bert_embedding[size + i] = words_emb_bert[i]
            words_ids_batch[ni][i] = size + i
            if word_to_ix:
                words_batch[ni][i] = word_to_ix[words[i]]
            if tags[i] in tag_to_ix:
                tags_batch[ni][i] = tag_to_ix[tags[i]]
            else:
                tags[i] = 'O'
        size += len_w
        if rw:
            words_batch[ni] = words
        origin_tags[ni] = tags
        len_w_batch[ni] = len_w

    ots, w, wi, l, t = (origin_tags, words_batch, words_ids_batch,
                        len_w_batch, tags_batch)

    # load bert embedding
    model.load_bert_embeds(bert_embedding)

    if word_to_ix:
        w_tensor = torch.from_numpy(w).to(model.device)
    elif rw:
        w_tensor = w
    else:
        w_tensor = torch.Tensor([0])

    # get gpu tensor
    return ots, w_tensor, torch.from_numpy(wi).to(
        model.device), torch.from_numpy(l).to(model.device), torch.from_numpy(
        t).to(model.device)
