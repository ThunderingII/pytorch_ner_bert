# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
import logging
import pickle
import re
import json
import multiprocessing

import torch

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

import util.base_util as bu

logger = bu.get_logger(__name__)
PA_PATTERN = re.compile('\d+,\s\d+')
CUDA_ID_PATTERN = re.compile('(\d+,)*\d+')

MODEL_PATH = '../config/bert/bert-base-chinese.tar.gz'
VOCAB_PATH = '../config/bert/bert-base-chinese-vocab.txt'
DATA_SET = set()


class InputExample(object):

    def __init__(self, unique_id, text_a, orgs, pers, text_b=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.orgs = orgs
        self.pers = pers


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask,
                 input_type_ids, orgs, pers):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.orgs = orgs
        self.pers = pers


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        orgs = _get_tag_position(example.text_a, example.orgs, tokens_a)
        pers = _get_tag_position(example.text_a, example.pers, tokens_a)

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
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
                orgs=orgs,
                pers=pers))
    return features


def _get_tag_position(text, tag_position_list, tokens):
    rl = []
    for s, e in tag_position_list:
        e = e - 1
        for ts in range(0, s + 1)[::-1]:
            te = ts + e - s
            try:
                if te < len(tokens) and tokens[ts] == text[s] and \
                        tokens[te] == text[e]:
                    rl.append((ts, te + 1))
                    break
            except:
                pass

    return rl


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _get_pair_list(s_list):
    rs = []
    for s in s_list:
        s, e = s.split(', ')
        rs.append((int(s), int(e)))
    return rs


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    limit = -1
    with open(input_file, "r", encoding='utf-8') as reader:
        while limit < 0 or limit > 0:
            limit -= 1
            line = reader.readline()
            lmd5 = bu.md5(line)
            if not line:
                break
            tagline = reader.readline()
            if lmd5 in DATA_SET:
                continue
            else:
                DATA_SET.add(lmd5)
            org_str, per_str = tagline.split('][')
            orgs = _get_pair_list(PA_PATTERN.findall(org_str))

            pers = _get_pair_list(PA_PATTERN.findall(per_str))

            examples.append(
                InputExample(unique_id=unique_id, text_a=line, orgs=orgs,
                             pers=pers))
            unique_id += 1
    return examples


def build_feature(mpi, input_q, layer_indexes, features, output_file):
    index = 0
    out_list = []
    logger.info(f'begin!{mpi}_{index}_{id(input_q)}')
    while True:
        example_indices, all_encoder_layers = input_q.get(True)
        if len(out_list) >= 5000 or example_indices is None:
            logger.info(f'begin to write!{mpi}_{index}')
            with open(output_file + f'_{mpi}_{index}', "wb") as writer:
                index += 1
                pickle.dump(out_list, writer)
                out_list = []
        if example_indices is None:
            break

        all_encoder_layers = [layer.numpy() for layer in
                              all_encoder_layers]

        for b, example_index in enumerate(example_indices):
            feature = features[example_index]
            unique_id = int(feature.unique_id)
            output_dict = collections.OrderedDict()
            output_dict["linex_index"] = unique_id
            output_dict["orgs"] = feature.orgs
            output_dict["pers"] = feature.pers
            all_out_features = []

            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = all_encoder_layers[layer_index][b][
                        i]
                    all_layers.append(layers)
                out_features = collections.OrderedDict()
                out_features["token"] = token
                out_features["layers"] = all_layers
                all_out_features.append(out_features)
            output_dict["features"] = all_out_features
            out_list.append(output_dict)
            if index == 0 and len(out_list) == 3:
                with open(output_file + f'_{mpi}.json', "w") as writer:
                    writer.write(str(out_list))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", type=str,
                        help='input file or input dir')
    parser.add_argument("--output_file", type=str,
                        help='output file or output dir')

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true',
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
                        type=int, default=2,
                        help="thread num")

    args = parser.parse_args()
    no_cuda = False
    cuda_ids = []
    if args.cuda_id and CUDA_ID_PATTERN.match(args.cuda_id):
        cuda_ids = args.cuda_id.split(',')
        cuda_ids = [int(n) for n in cuda_ids]
    else:
        print('no use cuda')
        no_cuda = True

    if args.local_rank == -1 or no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = len(cuda_ids)
    else:
        print('init_process_group')
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        f"device: {device} n_gpu: {n_gpu} distributed training: {args.local_rank != -1}")

    layer_indexes = [int(x) for x in args.layers.split(",")]

    tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH,
                                              do_lower_case=args.do_lower_case)

    inputs = []
    outputs = []
    if os.path.isdir(args.input_file):
        for f in os.listdir(args.input_file):
            if os.path.isfile(args.input_file + '/' + f):
                inputs.append(args.input_file + '/' + f)
                outputs.append(args.output_file + '/' + f + '.pkl')

    else:
        inputs.append(args.input_file)
        outputs.append(args.output_file)

    for input_file, output_file in zip(inputs, outputs):
        examples = read_examples(input_file)

        features = convert_examples_to_features(
            examples=examples, seq_length=args.max_seq_length,
            tokenizer=tokenizer)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        model = BertModel.from_pretrained(MODEL_PATH)
        model.to(device)

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[
                                                                  args.local_rank],
                                                              output_device=args.local_rank)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=cuda_ids)

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

        model.eval()
        index = 1
        logger.info(id(input_q))

        ts = []
        for i in range(args.tn):
            p = multiprocessing.Process(target=build_feature, args=(
                i, input_q, layer_indexes, features, output_file))
            p.start()
            ts.append(p)

        for input_ids, input_mask, example_indices in eval_dataloader:
            if index % 100 == 0:
                logger.info(f'index:{index}')
            index += 1
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            all_encoder_layers, _ = model(input_ids, token_type_ids=None,
                                          attention_mask=input_mask)
            all_encoder_layers = [layer.detach().cpu() for layer in
                                  all_encoder_layers]
            input_q.put((example_indices, all_encoder_layers))
        for i in range(args.tn * 2):
            input_q.put((None, None))

        for i in range(args.tn):
            ts[i].join()


if __name__ == "__main__":
    main()
