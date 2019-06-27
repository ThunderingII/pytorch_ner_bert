from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
import re
import multiprocessing
import json

import pickle
import numpy as np
import torch

import pathlib

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

import util.base_util as bu

logger = bu.get_logger(__name__)
CUDA_ID_PATTERN = re.compile('(\d+,)*\d+')

MODEL_PATH = '../config/bert/bert-base-chinese.tar.gz'
VOCAB_PATH = '../config/bert/bert-base-chinese-vocab.txt'
DATA_SET = set()

LIMITED = 0


class InputExample(object):

    def __init__(self, unique_id, text, entity_map):
        self.unique_id = unique_id
        self.text = text
        self.entity_map = entity_map


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask,
                 input_type_ids, entity_map, sentence):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.entity_map = entity_map
        self.sentence = sentence


def convert_examples_to_features(examples, seq_length, tokenizer, lower_case):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        origin_tokens = tokenizer.tokenize(example.text)
        # Account for [CLS] and [SEP] with "- 2"
        if len(origin_tokens) > seq_length - 2:
            origin_tokens = origin_tokens[0:(seq_length - 2)]
        entity_map = _get_entity(example.text, example.entity_map, lower_case,
                                 tokenizer)
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in origin_tokens:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info(
                "input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join(
                    [str(x) for x in input_type_ids]))
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                entity_map=entity_map,
                sentence=example.text))
    return features


def _get_entity(text, entity_index_map, lower_case, tokenizer):
    entity_map = {}
    entity2tokens = {}
    if lower_case:
        text = text.lower()
    #     tokenizer.tokenize(example.text)
    # k is tag type
    for k in entity_index_map:
        entity_map[k] = []
        for s, e in entity_index_map[k]:
            entity = text[s:e]
            if entity not in entity2tokens:
                entity2tokens[entity] = tokenizer.tokenize(text[s:e])
                entity_map[k].append(entity2tokens[entity])
    return entity_map


def _get_pair_list(s_list):
    rs = []
    for s in s_list:
        s, e = s.split(', ')
        rs.append((int(s), int(e)))
    return rs


def read_examples(input_file, entity_types, mode):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    limit = LIMITED
    mode_is_train = mode == 'train'
    with open(input_file, "r", encoding='utf-8') as reader:
        while limit < 0 or limit > 0:
            limit -= 1
            line = reader.readline().strip()
            lmd5 = bu.md5(line)
            if not line:
                break
            tag_str = None
            if mode_is_train:
                tag_str = reader.readline().strip()
            if lmd5 in DATA_SET:
                continue
            else:
                DATA_SET.add(lmd5)
            entity_map = {}
            if mode_is_train:
                entity_map = json.loads(tag_str, encoding='UTF-8')
                keys_to_del = [key for key in entity_map if
                               key not in entity_types]
                for k in keys_to_del:
                    del entity_map[k]
            if entity_map:
                examples.append(
                    InputExample(unique_id=unique_id, text=line,
                                 entity_map=entity_map))
                unique_id += 1
    return examples


def build_feature(mpi, input_q, layer_indexes, features, output_file, mode):
    index = 0
    out_list = []
    logger.info(f'begin!{mpi}_{index}_{id(input_q)}')
    while True:
        example_indices, all_encoder_layers = input_q.get(True)
        if len(out_list) >= 2000 or example_indices is None:
            logger.info(f'begin to write!{mpi}_{index}')
            with open(output_file + f'_{mpi}_{index}', "wb") as writer:
                index += 1
                pickle.dump(out_list, writer)
                out_list = []
        if example_indices is None:
            break

        all_encoder_layers = [layer.numpy() for layer in
                              all_encoder_layers]

        for i, embedding_e, token_e, tag_e, seq_len, st in process_embedding(
                all_encoder_layers, example_indices, features,
                layer_indexes, mode):
            # index, words, words_emb_bert, tags, len_w
            out_list.append((i, token_e, embedding_e, tag_e, seq_len, st))


def process_embedding(all_encoder_layers, example_indices, features,
                      layer_indexes, mode):
    for b, example_index in enumerate(example_indices):
        feature = features[example_index]
        unique_id = int(feature.unique_id)

        seq_embed = []
        seq_tokens = []
        for (i, token) in enumerate(feature.tokens):
            all_layers = []
            for (j, layer_index) in enumerate(layer_indexes):
                all_layers.append(all_encoder_layers[layer_index][b][i])
            if mode == 'average':
                t = np.average(all_layers, axis=0)
            else:
                t = np.sum(all_layers, axis=0)
            seq_embed.append(t)
            seq_tokens.append(token)
        rs = process_token(seq_embed, feature.entity_map, seq_tokens)
        if rs:
            embedding_e, token_e, tag_e, len_w = rs
            yield unique_id, embedding_e, token_e, tag_e, len_w, feature.sentence
        else:
            continue


def process_token(line_embedding, entity_map, line_tokens):
    embedding_e = []
    tag_e = []
    token_e = []
    START_INDEX = 1
    END_INDEX = len(line_tokens) - 2
    for i in range(len(line_tokens)):
        if i == 0 or i == len(line_tokens) - 1:
            # remove [CLS] and [SEP]
            continue
        current_word_emb = line_embedding[i]
        token = line_tokens[i]

        if token.startswith('##'):
            embedding_e[-1] = np.sum(
                (embedding_e[-1], current_word_emb), axis=0)
            token_e[-1] = token_e[-1] + token[2:]
        else:
            embedding_e.append(current_word_emb)
            token_e.append(token)
            has_append = False
            for k in entity_map:
                if has_append:
                    break
                for e_li in entity_map[k]:
                    if has_append:
                        break
                    for j in range(len(e_li)):
                        if has_append:
                            break
                        if token == e_li[j]:
                            ei = i + len(e_li) - j - 1
                            si = i - j
                            if si >= START_INDEX and ei <= END_INDEX:
                                if line_tokens[si] == e_li[0] and line_tokens[
                                    ei] == e_li[-1]:
                                    if j == 0:
                                        tag_e.append(f'B-{k}')
                                    else:
                                        tag_e.append(f'I-{k}')
                                    has_append = True
                                    break
            if len(tag_e) < len(token_e):
                tag_e.append('O')

    try:
        assert len(embedding_e) == len(tag_e) == len(
            token_e), f"embed len{len(embedding_e)} Words{len(token_e)} " \
                      f"and tags lengths{len(tag_e)} don't match"

        return embedding_e, token_e, tag_e, len(token_e)
    except Exception as e:
        print('-' * 50)
        print(e)
        print('line_embedding', len(line_embedding[1:-1]))
        print(line_tokens[1:-1])
        print(entity_map)
        print('token:', len(token_e), token_e)
        print('tag_e:', len(tag_e), tag_e)
        print('em', len(embedding_e))
        print('*' * 50)
        return None


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", type=str,
                        help='input file or input dir')
    parser.add_argument("--output_file", type=str,
                        help='output file or output dir')
    parser.add_argument("--entity_types", type=str, default='ORG',
                        help='the types need processed, like "ORG,PER", use '
                             '"," to connect str')

    ## Other parameters
    parser.add_argument("--not_do_lower_case", action='store_false',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--cuda_id",
                        type=str,
                        default='',
                        help="input cuda id, example:'0,1' or '2' or '1,2,3'")
    parser.add_argument("--tn",
                        type=int, default=4,
                        help="thread num")
    parser.add_argument("--limit",
                        type=int, default=-1,
                        help="limit the num of data generate")
    parser.add_argument("--layer_mode",
                        type=str, default='sum',
                        help="how to process layers' output")

    parser.add_argument("--mode",
                        type=str, default='train',
                        help="train or test")
    args = parser.parse_args()

    global LIMITED
    LIMITED = args.limit
    no_cuda = False
    cuda_ids = []
    if args.cuda_id and CUDA_ID_PATTERN.match(args.cuda_id):
        cuda_ids = args.cuda_id.split(',')
        cuda_ids = [int(n) for n in cuda_ids]
    else:
        print('no use cuda')
        no_cuda = True

    if args.local_rank == -1 or no_cuda:
        n_gpu = len(cuda_ids)
        if no_cuda or not torch.cuda.is_available():
            device = torch.device('cpu')
        elif n_gpu == 1:
            device = torch.device(f'cuda:{cuda_ids[0]}')
        else:
            device = torch.device('cuda')
    else:
        print('init_process_group')
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        f"device: {device} n_gpu: {n_gpu} distributed "
        f"training: {args.local_rank != -1} "
        f"cuda ids {cuda_ids}")

    layer_indexes = [int(x) for x in args.layers.split(",")]

    tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH,
                                              do_lower_case=args.not_do_lower_case)
    model = BertModel.from_pretrained(MODEL_PATH)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[
                                                              args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=cuda_ids)

    model.eval()

    inputs = []
    outputs = []

    if os.path.isdir(args.input_file):
        pathlib.Path(args.output_file).mkdir(parents=True, exist_ok=True)
        for f in os.listdir(args.input_file):
            if os.path.isfile(args.input_file + '/' + f):
                inputs.append(args.input_file + '/' + f)
                outputs.append(args.output_file + '/' + f + '.pkl')

    else:
        inputs.append(args.input_file)
        outputs.append(args.output_file)

    if len(inputs) > 0:
        LIMITED //= len(inputs)
    ti = 0
    entity_types = set(args.entity_types.split(','))
    for input_file, output_file in zip(inputs, outputs):
        examples = read_examples(input_file, entity_types, args.mode)
        features = convert_examples_to_features(
            examples=examples, seq_length=args.max_seq_length,
            tokenizer=tokenizer, lower_case=args.not_do_lower_case)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features],
                                      dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0),
                                         dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask,
                                  all_example_index)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                     batch_size=args.batch_size)
        input_q = multiprocessing.Queue(maxsize=200)
        ts = []
        for i in range(args.tn):
            p = multiprocessing.Process(target=build_feature, args=(
                i, input_q, layer_indexes, features, output_file, args.mode))
            p.start()
            ts.append(p)
        index = 0
        for input_ids, input_mask, example_indices in eval_dataloader:
            index += 1
            ti += 1
            if ti % 100 == 0:
                logger.info(f'index:{ti}')
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            all_encoder_layers, _ = model(input_ids, token_type_ids=None,
                                          attention_mask=input_mask)
            all_encoder_layers = [layer.detach().cpu() for layer in
                                  all_encoder_layers]
            input_q.put((example_indices, all_encoder_layers))

            if LIMITED > 0 and index * args.batch_size >= LIMITED:
                break

        for i in range(args.tn * 2):
            input_q.put((None, None))

        for i in range(args.tn):
            ts[i].join()


if __name__ == "__main__":
    main()
