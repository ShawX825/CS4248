import json
import math
import os
import pickle
import random
import re
from collections import defaultdict
from itertools import permutations
import logging
from models.BERT import tokenization

import dgl
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, speaker_ids, mention_ids, mentioned_h, mentioned_t):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.speaker_ids = speaker_ids
        self.mention_ids = mention_ids
        #add
        self.mentioned_h = mentioned_h
        self.mentioned_t = mentioned_t

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class bertsProcessor(DataProcessor): #bert_s
    def __init__(self, src_file, n_class):
        def is_speaker(a):
            a = a.split()
            return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()
        
        def rename(d, x, y):
            # they replace argument pair x,y by unused1, unused2 
            # and replace speaker_i by unused_j if speaker_i == x or y
            unused = ["[unused1]", "[unused2]"]
            a = []
            if is_speaker(x):
                a += [x]
            else:
                a += [None]
            if x != y and is_speaker(y):
                a += [y]
            else:
                a += [None]
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", unused[i] + " :")
                if x == a[i]:
                    x = unused[i]
                if y == a[i]:
                    y = unused[i]
            return d, x, y
            
        random.seed(42)
        self.D = [[], [], []]
        for sid in range(3):
            with open(src_file+["/train.json", "/dev.json", "/test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])): # argument pairs
                    rid = []
                    for k in range(n_class): # generate one-hot label vector
                        if k+1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    # replace speaker argument pair x,y by unused1, unused2 
                    d, h, t = rename('[TURNEND]\n'.join(data[i][0]).lower(), data[i][1][j]["x"].lower(), data[i][1][j]["y"].lower())
                    d = [d,
                         h,
                         t,
                         rid]
                    self.D[sid] += [d]
        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(2)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i][0]
            text_b = data[i][1]
            text_c = data[i][2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=data[i][3], text_c=text_c))
            
        return examples

class bertsf1cProcessor(DataProcessor): #bert_s (conversational f1)
    def __init__(self, src_file, n_class):
        def is_speaker(a):
            a = a.split()
            return (len(a) == 2 and a[0] == "speaker" and a[1].isdigit())
        
        def rename(d, x, y):
            unused = ["[unused1]", "[unused2]"]
            a = []
            if is_speaker(x):
                a += [x]
            else:
                a += [None]
            if x != y and is_speaker(y):
                a += [y]
            else:
                a += [None]
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", unused[i] + " :")
                if x == a[i]:
                    x = unused[i]
                if y == a[i]:
                    y = unused[i]
            return d, x, y
            
        random.seed(42)
        self.D = [[], [], []]
        for sid in range(1, 3):
            with open(src_file+["/dev.json", "/test.json"][sid-1], "r", encoding="utf8") as f:
                data = json.load(f)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    for k in range(n_class):
                        if k+1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    for l in range(1, len(data[i][0])+1):
                        d, h, t = rename('[TURNEND]\n'.join(data[i][0][:l]).lower(), data[i][1][j]["x"].lower(), data[i][1][j]["y"].lower())
                        d = [d,
                             h,
                             t,
                             rid]
                        self.D[sid] += [d]
        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(2)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i][0]
            text_b = data[i][1]
            text_c = data[i][2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=data[i][3], text_c=text_c))
            
        return examples

# original
# def tokenize(text, tokenizer, start_mention_id):
#     speaker2id = {'[unused1]' : 11, '[unused2]' : 12, 'speaker 1' : 1, 'speaker 2' : 2, 'speaker 3' : 3, 'speaker 4' : 4, 'speaker 5' : 5, 'speaker 6' : 6, 'speaker 7' : 7, 'speaker 8' : 8, 'speaker 9' : 9}
#     D = ['[unused1]', '[unused2]', 'speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']
#     text_tokens = []
#     textraw = [text]
#     # split the text by speaker n
#     for delimiter in D:
#         ntextraw = []
#         for i in range(len(textraw)):
#             t = textraw[i].split(delimiter)
#             for j in range(len(t)):
#                 ntextraw += [t[j]]
#                 if j != len(t)-1:
#                     ntextraw += [delimiter]
#         textraw = ntextraw
#     text = []
#     # speaker_ids, mention_ids may relate to speaker embedding in Fig 2 
#     # mention_id is the order of apperance for a specific speaker_id 
#     # both speaker_id and mention_id are assigned to each token in current turn
#     speaker_ids = [] # same length as tokens
#     mention_ids = [] # same length as tokens
#     mention_id = start_mention_id
#     speaker_id = 0 # number of mentioned speakers?
#     for t in textraw:
#         if t in ['speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']:
#             speaker_id = speaker2id[t]
#             mention_id += 1
#             tokens = tokenizer.tokenize(t+" ")
#             for tok in tokens:
#                 text += [tok]
#                 speaker_ids.append(speaker_id)
#                 mention_ids.append(mention_id)
#         elif t in ['[unused1]', '[unused2]']:
#             speaker_id = speaker2id[t]
#             mention_id += 1
#             text += [t]
#             speaker_ids.append(speaker_id)
#             mention_ids.append(mention_id)
#         else:
#             tokens = tokenizer.tokenize(t)
#             for tok in tokens:
#                 text += [tok]
#                 speaker_ids.append(speaker_id)
#                 mention_ids.append(mention_id)

#     return text, speaker_ids, mention_ids

def tokenize(text, tokenizer, start_mention_id, head, tail):
    speaker2id = {'[unused1]' : 11, '[unused2]' : 12, 'speaker 1' : 1, 'speaker 2' : 2, 'speaker 3' : 3, 'speaker 4' : 4, 'speaker 5' : 5, 'speaker 6' : 6, 'speaker 7' : 7, 'speaker 8' : 8, 'speaker 9' : 9}
    D = ['[unused1]', '[unused2]', 'speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']
    text_tokens = []
    # textraw = [text]
    textraw = text.split('[turnend]\n')
    # split the text by speaker n
    # for delimiter in D:
    #     ntextraw = []
    #     for i in range(len(textraw)):
    #         t = textraw[i].split(delimiter)
    #         # cannot think up an easier way to aviod 
    #         # spliting by speakers in the middle of the sentence
    #         for j in range(len(t)):
    #             ntextraw += [t[j]]
    #             if j != len(t)-1:
    #                 ntextraw += [delimiter]
    #     textraw = ntextraw

    ntextraw = []
    for i, turn in enumerate(textraw):
        first_colon = turn.find(':')
        speakers = turn[:first_colon]
        dialog = turn[first_colon:]  
        speakers = [speakers]
        for delimiter in D:
            tmp_text = []
            for k in range(len(speakers)):
                tt = speakers[k].split(delimiter)
                for j,t in enumerate(tt):
                    tmp_text.append(t)
                    if j != len(tt)-1:
                        tmp_text.append(delimiter)
            speakers = tmp_text
        ntextraw.extend(speakers)
        ntextraw.append(dialog)
    textraw = ntextraw  

    text = []
    # speaker_ids, mention_ids may relate to speaker embedding in Fig 2 
    # mention_id is the order of apperance for a specific speaker_id 
    # both speaker_id and mention_id are assigned to each token in current turn
    speaker_ids = [] # same length as tokens
    mention_ids = [] # same length as tokens
    mention_id = start_mention_id
    speaker_id = 0 # number of mentioned speakers?
    mentioned_h = set()
    mentioned_t = set()

    for t in textraw:
        
        if t in ['speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']:
            speaker_id = speaker2id[t]
            mention_id += 1
            tokens = tokenizer.tokenize(t+" ")
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)
                mention_ids.append(mention_id)

        elif t in ['[unused1]', '[unused2]']:
            speaker_id = speaker2id[t]
            mention_id += 1
            text += [t]
            speaker_ids.append(speaker_id)
            mention_ids.append(mention_id)
        else:
            tokens = tokenizer.tokenize(t)

            for tok in tokens:

                text += [tok]
                speaker_ids.append(speaker_id)
                mention_ids.append(mention_id)
        if head in t:
            mentioned_h.add(mention_id)
        if tail in t:
            mentioned_t.add(mention_id)

    return text, speaker_ids, mention_ids, mentioned_h, mentioned_t


def tokenize2(text, tokenizer):
    speaker2id = {'[unused1]' : 11, '[unused2]' : 12, 'speaker 1' : 1, 'speaker 2' : 2, 'speaker 3' : 3, 'speaker 4' : 4, 'speaker 5' : 5, 'speaker 6' : 6, 'speaker 7' : 7, 'speaker 8' : 8, 'speaker 9' : 9}
    D = ['[unused1]', '[unused2]', 'speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']
    text_tokens = []
    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t)-1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    speaker_ids = []
    speaker_id = 0

    # add 
    mentioned_h = set()
    mentioned_t = set()

    for t in textraw:
        if t in ['speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']:
            speaker_id = speaker2id[t]
            tokens = tokenizer.tokenize(t+" ")
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)
        elif t in ['[unused1]', '[unused2]']:
            speaker_id = speaker2id[t]
            text += [t]
            speaker_ids.append(speaker_id)
        else:
            tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)

    return text, speaker_ids

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    features = [[]]
    for (ex_index, example) in enumerate(examples):
        # add h,t
        h, t = example.text_b, example.text_c

        tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids, mentioned_h, mentioned_t = tokenize(example.text_a, tokenizer, 0, h, t)
        
        tokens_b, tokens_b_speaker_ids = tokenize2(example.text_b, tokenizer)
        tokens_c, tokens_c_speaker_ids = tokenize2(example.text_c, tokenizer)

        popped_mention_id = _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4, tokens_a_speaker_ids, tokens_b_speaker_ids, tokens_c_speaker_ids, tokens_a_mention_ids)
        mentioned_h = [x for x in mentioned_h if x not in popped_mention_id]
        mentioned_t = [x for x in mentioned_t if x not in popped_mention_id]

        tokens_b_mention_ids = [max(tokens_a_mention_ids) + 1 for _ in range(len(tokens_b))]
        tokens_c_mention_ids = [max(tokens_a_mention_ids) + 2 for _ in range(len(tokens_c))]

        tokens_b = tokens_b + ["[SEP]"] + tokens_c
        tokens_b_speaker_ids = tokens_b_speaker_ids + [0] + tokens_c_speaker_ids
        tokens_b_mention_ids = tokens_b_mention_ids + [0] + tokens_c_mention_ids

        tokens = []
        segment_ids = []
        speaker_ids = []
        mention_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        speaker_ids = speaker_ids + tokens_a_speaker_ids
        mention_ids = mention_ids + tokens_a_mention_ids
        tokens.append("[SEP]")
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        speaker_ids = speaker_ids + tokens_b_speaker_ids
        mention_ids = mention_ids + tokens_b_mention_ids
        tokens.append("[SEP]")
        segment_ids.append(1)
        speaker_ids.append(0)
        mention_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            speaker_ids.append(0)
            mention_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(speaker_ids) == max_seq_length
        assert len(mention_ids) == max_seq_length

        label_id = example.label 
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [x for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                    "speaker_ids: %s" % " ".join([str(x) for x in speaker_ids]))
            logger.info(
                    "mention_ids: %s" % " ".join([str(x) for x in mention_ids]))


        features[-1].append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        speaker_ids=speaker_ids,
                        mention_ids=mention_ids,                       
                        mentioned_h=mentioned_h,
                        mentioned_t=mentioned_t
                        )) # add h,t
        if len(features[-1]) == 1:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features

def convert_examples_to_features_roberta(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    features = [[]]
    for (ex_index, example) in enumerate(examples):
        
        # add h,t 
        h, t = example.text_b, example.text_c
        tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids, mentioned_h, mentioned_t = tokenize(example.text_a, tokenizer, 0, h, t)

        tokens_b, tokens_b_speaker_ids = tokenize2(example.text_b, tokenizer)
        tokens_c, tokens_c_speaker_ids = tokenize2(example.text_c, tokenizer)

        popped_mention_id = _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 6, tokens_a_speaker_ids, tokens_b_speaker_ids, tokens_c_speaker_ids, tokens_a_mention_ids)
        mentioned_h = set([x for x in mentioned_h if x not in popped_mention_id])
        mentioned_t = set([x for x in mentioned_t if x not in popped_mention_id])
        tokens_b_mention_ids = [max(tokens_a_mention_ids) + 1 for _ in range(len(tokens_b))]
        tokens_c_mention_ids = [max(tokens_a_mention_ids) + 2 for _ in range(len(tokens_c))]

        tokens_b = tokens_b + ['</s>', '</s>'] + tokens_c
        tokens_b_speaker_ids = tokens_b_speaker_ids + [0, 0] + tokens_c_speaker_ids
        tokens_b_mention_ids = tokens_b_mention_ids + [0, 0] + tokens_c_mention_ids

        tokens = []
        segment_ids = []
        speaker_ids = []
        mention_ids = []
        tokens.append('<s>')
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        speaker_ids = speaker_ids + tokens_a_speaker_ids
        mention_ids = mention_ids + tokens_a_mention_ids
        tokens.append('</s>')
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)
        tokens.append('</s>')
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        speaker_ids = speaker_ids + tokens_b_speaker_ids
        mention_ids = mention_ids + tokens_b_mention_ids
        tokens.append('</s>')
        segment_ids.append(1)
        speaker_ids.append(0)
        mention_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        assert len(input_ids) == len(input_mask)
        assert len(input_mask) == len(segment_ids)
        assert len(segment_ids) == len(speaker_ids)
        assert len(speaker_ids) == len(mention_ids)
        assert len(mention_ids) == len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(1)
            input_mask.append(0)
            segment_ids.append(0)
            speaker_ids.append(0)
            mention_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(speaker_ids) == max_seq_length
        assert len(mention_ids) == max_seq_length

        label_id = example.label 
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [x for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                    "speaker_ids: %s" % " ".join([str(x) for x in speaker_ids]))
            logger.info(
                    "mention_ids: %s" % " ".join([str(x) for x in mention_ids]))


        features[-1].append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        speaker_ids=speaker_ids,
                        mention_ids=mention_ids,
                        mentioned_h=mentioned_h,
                        mentioned_t=mentioned_t
                        )) # add h,t
        if len(features[-1]) == 1:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features

def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length, tokens_a_speaker_ids, tokens_b_speaker_ids, tokens_c_speaker_ids, tokens_a_mention_ids):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    popped_mention_id = set()
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
            tokens_a_speaker_ids.pop()
            popped_mention_id.add(tokens_a_mention_ids.pop())
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
            tokens_b_speaker_ids.pop()
        else:
            tokens_c.pop()
            tokens_c_speaker_ids.pop()
    return popped_mention_id
def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def mention2mask(mention_id):
    slen = len(mention_id)
    mask = []
    turn_mention_ids = [i for i in range(1, np.max(mention_id) - 1)]
    for j in range(slen):
        tmp = None
        if mention_id[j] not in turn_mention_ids:
            tmp = np.zeros(slen, dtype=bool)
            tmp[j] = 1
        else:
            start = mention_id[j]
            end = mention_id[j]
            if mention_id[j] - 1 in turn_mention_ids:
                start = mention_id[j] - 1

            if mention_id[j] + 1 in turn_mention_ids:
                end = mention_id[j] + 1
            tmp = (mention_id >= start) & (mention_id <= end)
        mask.append(tmp)
    mask = np.stack(mask)
    return mask

class TUCOREGCNDataset(IterableDataset):

    def __init__(self, src_file, save_file, max_seq_length, tokenizer, n_class, encoder_type):

        super(TUCOREGCNDataset, self).__init__()

        self.data = None
        self.input_max_length = max_seq_length

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            self.data = []

            bertsProcessor_class = bertsProcessor(src_file, n_class)
            # just set it as itself for convenience 
            bertsProcessor_class_resolved_pronoun = bertsProcessor(src_file, n_class) 
            # # resolve pronoun
            # bertsProcessor_class_resolved_pronoun = bertsProcessor('../CS4248/data/naive_pronoun_resolution', n_class)
            
            if "train" in save_file:
                examples = bertsProcessor_class.get_train_examples(save_file)
                examples_resolved_pronoun = bertsProcessor_class_resolved_pronoun.get_train_examples(None) 
            elif "dev" in save_file:
                examples = bertsProcessor_class.get_dev_examples(save_file)
                examples_resolved_pronoun = bertsProcessor_class_resolved_pronoun.get_dev_examples(None) 
            elif "test" in save_file:
                examples = bertsProcessor_class.get_test_examples(save_file)
                examples_resolved_pronoun = bertsProcessor_class_resolved_pronoun.get_test_examples(None) 

            else:
                print(error)
            if encoder_type == "BERT":
                features = convert_examples_to_features(examples, max_seq_length, tokenizer)
            else:
                features = convert_examples_to_features_roberta(examples, max_seq_length, tokenizer)
                # resolve pronoun
                ref_features =  convert_examples_to_features_roberta(examples_resolved_pronoun, max_seq_length, tokenizer)

            # cnt = 0
            i = 0
            for f, ref_f in zip(features,ref_features):
                
                if max(f[0].mention_ids) - 2 == max(ref_f[0].mention_ids) - 2 and\
                    (f[0].mentioned_h != ref_f[0].mentioned_h or f[0].mentioned_t != ref_f[0].mentioned_t):
                    entity_edges_infor = {'h':ref_f[0].mentioned_h, 't':ref_f[0].mentioned_t}
                else:
                    entity_edges_infor = {'h':f[0].mentioned_h, 't':f[0].mentioned_t}

                speaker_infor = self.make_speaker_infor(f[0].speaker_ids, f[0].mention_ids)
                turn_node_num = max(f[0].mention_ids) - 2
                head_mention_id = max(f[0].mention_ids) - 1
                tail_mention_id = max(f[0].mention_ids)
                entity_edges_infor['h'].add(head_mention_id)
                entity_edges_infor['t'].add(tail_mention_id)

                graph, used_mention = self.create_graph(speaker_infor, turn_node_num, entity_edges_infor, head_mention_id, tail_mention_id)

                assert len(used_mention) == (max(f[0].mention_ids) + 1)

                self.data.append({
                    'input_ids': np.array(f[0].input_ids),
                    'segment_ids': np.array(f[0].segment_ids),
                    'input_mask': np.array(f[0].input_mask),
                    'speaker_ids': np.array(f[0].speaker_ids),
                    'label_ids': np.array(f[0].label_id),
                    'mention_id': np.array(f[0].mention_ids),
                    'turn_mask': mention2mask(np.array(f[0].mention_ids)),
                    'graph': graph
                    })
                i += 1
            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data}, fw)
            # print('mention not the same',cnt)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)
    
    def turn2speaker(self, turn):
        return turn.split()[1]
    
    def make_speaker_infor(self, speaker_id, mention_id):
        tmp = defaultdict(set)
        for i in range(1, len(speaker_id)):
            if speaker_id[i] == 0:
                break
            tmp[speaker_id[i]].add(mention_id[i])
        
        speaker_infor = dict()
        for k, va in tmp.items():
            speaker_infor[k] = list(va)
        return speaker_infor
    
    def make_entity_edges_infor(self, input_ids, mention_id, ref_input_ids):
        # recall: '[unused1]' : 11, '[unused2]' : 12
        # here head_mention_id and tail_mention_id are argument pairs
        entity_edges_infor = {'h':[], 't':[]}
        entity_edges_infor_tmp = {'h':[], 't':[]} 
        head_mention_id = max(mention_id) - 1
        tail_mention_id = max(mention_id)
        head = list()
        tail = list()

        # get vocab id for argument
        for i in range(len(mention_id)):
            if mention_id[i] == head_mention_id: 
                head.append(input_ids[i])
        
        for i in range(len(mention_id)):
            if mention_id[i] == tail_mention_id:
                tail.append(input_ids[i])
        
        # scan token(s) and establish an edge if equal to argument
        for i in range(len(input_ids) - len(head)):            
            if input_ids[i:i+len(head)] == head:
                entity_edges_infor_tmp['h'].append(mention_id[i])
        
        for i in range(len(input_ids) - len(tail)): 

            if input_ids[i:i+len(tail)] == tail:
                entity_edges_infor_tmp['t'].append(mention_id[i])
        

        # resolve pronoun
        for i in range(len(ref_input_ids) - len(head)):
            if ref_input_ids[i:i+len(head)] == head:
                entity_edges_infor['h'].append(mention_id[i])
        
        for i in range(len(ref_input_ids) - len(tail)):
            if ref_input_ids[i:i+len(tail)] == tail:
                entity_edges_infor['t'].append(mention_id[i])  

        # print('head:',head)
        # print('tail:',tail)
        if np.all(np.array(entity_edges_infor_tmp['t']) ==  np.array(entity_edges_infor['t'])) and\
        np.all(np.array(entity_edges_infor_tmp['h']) ==  np.array(entity_edges_infor['h'])) :
            return entity_edges_infor, entity_edges_infor, True
        else:

            return entity_edges_infor, entity_edges_infor_tmp, False

    def create_graph(self, speaker_infor, turn_node_num, entity_edges_infor, head_mention_id, tail_mention_id):

        # used_mention is used for sanity check
        d = defaultdict(list)
        used_mention = set()
        used_mention.add(0)
        # the reason they do not specify node type, 
        # e.g. Dialogue Node, Turn Node, Arg Node, is that
        # it can be inferred by the type of edges 
        # e.g. nodes in (Node_1, Speaker Edge, Node_2) must be turn nodes

        # add speaker edges
        # turn nodes of the same speaker should be connected pair-wisely
        # thus here they use permutation(·)
        for _, mentions in speaker_infor.items():
            for h, t in permutations(mentions, 2):
                d[('node', 'speaker', 'node')].append((h, t))
                used_mention.add(h)
                used_mention.add(t)
                
        if d[('node', 'speaker', 'node')] == []:
            d[('node', 'speaker', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)

        # add dialog edges for HAN v3 v5
        for i in range(1, turn_node_num+1):
            d[('node', 'dialog', 'node')].append((i, 0))
            d[('node', 'dialog', 'node')].append((0, i))
            used_mention.add(i)
            used_mention.add(0)
        if d[('node', 'dialog', 'node')] == []:
            d[('node', 'dialog', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)

        # add entity edges
        for mention in entity_edges_infor['h']:
            # HAN v1-3
            d[('node', 'entity', 'node')].append((head_mention_id, mention))
            d[('node', 'entity', 'node')].append((mention, head_mention_id))
            used_mention.add(head_mention_id)
            used_mention.add(mention)
            # # for HAN v4, v5
            # used_mention.add(mention)
            # for offset in [-1,0,1]:
            #     mention = min(max(mention + offset, 1), turn_node_num)
            #     d[('node', 'entity', 'node')].append((head_mention_id, mention))
            #     d[('node', 'entity', 'node')].append((mention, head_mention_id))
            #     used_mention.add(head_mention_id)
                
        
        for mention in entity_edges_infor['t']:
            d[('node', 'entity', 'node')].append((tail_mention_id, mention))
            d[('node', 'entity', 'node')].append((mention, tail_mention_id))
            used_mention.add(tail_mention_id)
            used_mention.add(mention)
            # # for HAN v4, v5
            # used_mention.add(mention)
            # for offset in [-1,0,1]:
            #     mention = min(max(mention + offset, 1), turn_node_num)
            #     d[('node', 'entity', 'node')].append((tail_mention_id, mention))
            #     d[('node', 'entity', 'node')].append((mention, tail_mention_id))
            #     used_mention.add(tail_mention_id)
                
        
        # if an entity is not mentioned in any turns
        # then connect it to all turns
        if entity_edges_infor['h'] == []:
            for i in range(1, turn_node_num+1):
                d[('node', 'entity', 'node')].append((head_mention_id, i))
                d[('node', 'entity', 'node')].append((i, head_mention_id))
                used_mention.add(head_mention_id)
                # used_mention.add(i)
        
        if entity_edges_infor['t'] == []:
            for i in range(1, turn_node_num+1):
                d[('node', 'entity', 'node')].append((tail_mention_id, i))
                d[('node', 'entity', 'node')].append((i, tail_mention_id))
                used_mention.add(tail_mention_id)
                # used_mention.add(i)

        # add sequence edges 
        # if there is only one turn in a dialogue, sequence edge won't exist
        # do nothing in this case 
        turns = [i for i in range(1, turn_node_num+1)]
        if len(turns) >= 2:
            for h, t in permutations(turns, 2):
                d[('node', 'sequence', 'node')].append((h, t))
                used_mention.add(h)                
                used_mention.add(t)
        
        if d[('node', 'sequence', 'node')] == []:
            d[('node', 'sequence', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)
        

        graph = dgl.heterograph(d)

        return graph, used_mention

class TUCOREGCNDataset4f1c(IterableDataset):

    def __init__(self, src_file, save_file, max_seq_length, tokenizer, n_class, encoder_type):

        super(TUCOREGCNDataset4f1c, self).__init__()

        self.data = None
        self.input_max_length = max_seq_length

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            self.data = []

            bertsProcessor_class = bertsf1cProcessor(src_file, n_class)
            # just set it as itself for convenience 
            bertsProcessor_class_resolved_pronoun = bertsProcessor(src_file, n_class) 
            # resolve pronoun
            # bertsProcessor_class_resolved_pronoun = bertsf1cProcessor('../CS4248/data/naive_pronoun_resolution', n_class)
            if "dev" in save_file:
                examples = bertsProcessor_class.get_dev_examples(save_file)
                # resolve pronoun
                examples_resolved_pronoun = bertsProcessor_class_resolved_pronoun.get_dev_examples(None)
            elif "test" in save_file:
                examples = bertsProcessor_class.get_test_examples(save_file)
                # resolve pronoun
                examples_resolved_pronoun = bertsProcessor_class_resolved_pronoun.get_test_examples(None)

            else:
                print(error)
            if encoder_type == "BERT":
                features = convert_examples_to_features(examples, max_seq_length, tokenizer)
            else:
                features = convert_examples_to_features_roberta(examples, max_seq_length, tokenizer)
                # resolve pronoun
                ref_features =  convert_examples_to_features_roberta(examples_resolved_pronoun, max_seq_length, tokenizer)

            # cnt = []
            for f, ref_f in zip(features,ref_features):
                # cnt.append(1 if np.all(f[0].mention_ids == ref_f[0].mention_ids) else 0)
                if max(f[0].mention_ids) - 2 == max(ref_f[0].mention_ids) - 2 and\
                    (f[0].mentioned_h != ref_f[0].mentioned_h or f[0].mentioned_t != ref_f[0].mentioned_t):
                    entity_edges_infor = {'h':ref_f[0].mentioned_h, 't':ref_f[0].mentioned_t}
                else:
                    entity_edges_infor = {'h':f[0].mentioned_h, 't':f[0].mentioned_t}

                speaker_infor = self.make_speaker_infor(f[0].speaker_ids, f[0].mention_ids)
                turn_node_num = max(f[0].mention_ids) - 2
                head_mention_id = max(f[0].mention_ids) - 1
                tail_mention_id = max(f[0].mention_ids)
                entity_edges_infor['h'].add(head_mention_id)
                entity_edges_infor['t'].add(tail_mention_id)
                # entity_edges_infor = self.make_entity_edges_infor(f[0].input_ids, f[0].mention_ids, ref_f[0].input_ids)
                graph, used_mention = self.create_graph(speaker_infor, turn_node_num, entity_edges_infor, head_mention_id, tail_mention_id)
                assert len(used_mention) == (max(f[0].mention_ids) + 1)

                self.data.append({
                    'input_ids': np.array(f[0].input_ids),
                    'segment_ids': np.array(f[0].segment_ids),
                    'input_mask': np.array(f[0].input_mask),
                    'speaker_ids': np.array(f[0].speaker_ids),
                    'label_ids': np.array(f[0].label_id),
                    'mention_id': np.array(f[0].mention_ids),
                    'turn_mask': mention2mask(np.array(f[0].mention_ids)),
                    'graph': graph
                    })
            # save data
            # print(len(cnt),sum(cnt))
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)
    
    def turn2speaker(self, turn):
        return turn.split()[1]
    
    def make_speaker_infor(self, speaker_id, mention_id):
        tmp = defaultdict(set)
        for i in range(1, len(speaker_id)):
            if speaker_id[i] == 0:
                break
            tmp[speaker_id[i]].add(mention_id[i])
        
        speaker_infor = dict()
        for k, va in tmp.items():
            speaker_infor[k] = list(va)
        return speaker_infor
    
    def make_entity_edges_infor(self, input_ids, mention_id, ref_input_ids):
        entity_edges_infor = {'h':[], 't':[]}
        head_mention_id = max(mention_id) - 1
        tail_mention_id = max(mention_id)
        head = list()
        tail = list()
        for i in range(len(mention_id)):
            if mention_id[i] == head_mention_id:
                head.append(input_ids[i])

        for i in range(len(mention_id)):
            if mention_id[i] == tail_mention_id:
                tail.append(input_ids[i])

        # original
        # for i in range(len(input_ids) - len(head)):
        #     if input_ids[i:i+len(head)] == head:
        #         entity_edges_infor['h'].append(mention_id[i])
        
        # for i in range(len(input_ids) - len(tail)):
        #     if input_ids[i:i+len(tail)] == tail:
        #         entity_edges_infor['t'].append(mention_id[i])

        # resolve pronoun
        for i in range(len(ref_input_ids) - len(head)):
            if ref_input_ids[i:i+len(head)] == head:
                entity_edges_infor['h'].append(mention_id[i])
        
        for i in range(len(ref_input_ids) - len(tail)):
            if ref_input_ids[i:i+len(tail)] == tail:
                entity_edges_infor['t'].append(mention_id[i])        
        return entity_edges_infor

    def create_graph(self, speaker_infor, turn_node_num, entity_edges_infor, head_mention_id, tail_mention_id):

        # used_mention is used for sanity check
        d = defaultdict(list)
        used_mention = set()
        used_mention.add(0)
        # the reason they do not specify node type, 
        # e.g. Dialogue Node, Turn Node, Arg Node, is that
        # it can be inferred by the type of edges 
        # e.g. nodes in (Node_1, Speaker Edge, Node_2) must be turn nodes

        # add speaker edges
        # turn nodes of the same speaker should be connected pair-wisely
        # thus here they use permutation(·)
        for _, mentions in speaker_infor.items():
            for h, t in permutations(mentions, 2):
                d[('node', 'speaker', 'node')].append((h, t))
                used_mention.add(h)
                used_mention.add(t)

        if d[('node', 'speaker', 'node')] == []:
            d[('node', 'speaker', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)
        
        # add dialog edges for HAN v3, v5
        for i in range(1, turn_node_num+1):
            d[('node', 'dialog', 'node')].append((i, 0))
            d[('node', 'dialog', 'node')].append((0, i))
            used_mention.add(i)
            used_mention.add(0)
        if d[('node', 'dialog', 'node')] == []:
            d[('node', 'dialog', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)

        # add entity edges
        for mention in entity_edges_infor['h']:
            # HAN v1-3
            d[('node', 'entity', 'node')].append((head_mention_id, mention))
            d[('node', 'entity', 'node')].append((mention, head_mention_id))
            used_mention.add(head_mention_id)
            used_mention.add(mention)
            # # for HAN v4, v5
            # used_mention.add(mention)
            # for offset in [-1,0,1]:
            #     mention = min(max(mention + offset, 1), turn_node_num)
            #     d[('node', 'entity', 'node')].append((head_mention_id, mention))
            #     d[('node', 'entity', 'node')].append((mention, head_mention_id))
            #     used_mention.add(head_mention_id)
                
        
        for mention in entity_edges_infor['t']:
            d[('node', 'entity', 'node')].append((tail_mention_id, mention))
            d[('node', 'entity', 'node')].append((mention, tail_mention_id))
            used_mention.add(tail_mention_id)
            used_mention.add(mention)
            # # for HAN v4, v5
            # used_mention.add(mention)
            # for offset in [-1,0,1]:
            #     mention = min(max(mention + offset, 1), turn_node_num)
            #     d[('node', 'entity', 'node')].append((tail_mention_id, mention))
            #     d[('node', 'entity', 'node')].append((mention, tail_mention_id))
            #     used_mention.add(tail_mention_id)
                
        
        # if an entity is not mentioned in any turns
        # then connect it to all turns
        if entity_edges_infor['h'] == []:
            for i in range(1, turn_node_num+1):
                d[('node', 'entity', 'node')].append((head_mention_id, i))
                d[('node', 'entity', 'node')].append((i, head_mention_id))
                used_mention.add(head_mention_id)
                used_mention.add(i)
        
        if entity_edges_infor['t'] == []:
            for i in range(1, turn_node_num+1):
                d[('node', 'entity', 'node')].append((tail_mention_id, i))
                d[('node', 'entity', 'node')].append((i, tail_mention_id))
                used_mention.add(tail_mention_id)
                used_mention.add(i)

        # add sequence edges 
        # if there is only one turn in a dialogue, sequence edge won't exist 
        turns = [i for i in range(1, turn_node_num+1)]
        if len(turns) >= 2:
            for h, t in permutations(turns, 2):
                d[('node', 'sequence', 'node')].append((h, t))
                used_mention.add(h)                
                used_mention.add(t)

        if d[('node', 'sequence', 'node')] == []:
            d[('node', 'sequence', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)


        graph = dgl.heterograph(d)

        return graph, used_mention

class TUCOREGCNDataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, relation_num=36, max_length=512):
        super(TUCOREGCNDataloader, self).__init__(dataset, batch_size=batch_size, worker_init_fn=np.random.seed(42))
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length
        
        self.relation_num = relation_num

        self.order = list(range(self.length))

    def __iter__(self):
        # shuffle
        if self.shuffle:
            random.shuffle(self.order)
            self.data = [self.dataset[idx] for idx in self.order]
        else:
            self.data = self.dataset
        batch_num = math.ceil(self.length / self.batch_size)
        self.batches = [self.data[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                        for idx in range(0, batch_num)]
        self.batches_order = [self.order[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                              for idx in range(0, batch_num)]

        # begin
        input_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        segment_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        input_mask = torch.LongTensor(self.batch_size, self.max_length).cpu()
        mention_id = torch.LongTensor(self.batch_size, self.max_length).cpu()
        speaker_id = torch.LongTensor(self.batch_size, self.max_length).cpu()
        turn_masks = torch.LongTensor(self.batch_size, self.max_length, self.max_length).cpu()
        label_ids = torch.Tensor(self.batch_size, self.relation_num).cpu()

        for idx, minibatch in enumerate(self.batches):
            cur_bsz = len(minibatch)

            for mapping in [input_ids, segment_ids, input_mask, mention_id, label_ids, turn_masks, speaker_id]:
                if mapping is not None:
                    mapping.zero_()

            graph_list = []

            for i, example in enumerate(minibatch):
                mini_input_ids, mini_segment_ids, mini_input_mask, mini_label_ids, mini_mention_id, mini_speaker_id, turn_mask, graph = \
                    example['input_ids'], example['segment_ids'], example['input_mask'], example['label_ids'], \
                    example['mention_id'], example['speaker_ids'], example['turn_mask'], example['graph']
                graph_list.append(graph.to(torch.device('cuda:0')))

                word_num = mini_input_ids.shape[0]
                relation_num = mini_label_ids.shape[0]

                input_ids[i, :word_num].copy_(torch.from_numpy(mini_input_ids))
                segment_ids[i, :word_num].copy_(torch.from_numpy(mini_segment_ids))
                input_mask[i, :word_num].copy_(torch.from_numpy(mini_input_mask))
                mention_id[i, :word_num].copy_(torch.from_numpy(mini_mention_id))
                speaker_id[i, :word_num].copy_(torch.from_numpy(mini_speaker_id))
                turn_masks[i, :word_num, :word_num].copy_(torch.from_numpy(turn_mask))

                label_ids[i, :relation_num].copy_(torch.from_numpy(mini_label_ids))

            context_word_mask = input_ids > 0
            context_word_length = context_word_mask.sum(1)
            batch_max_length = context_word_length.max()

            yield {'input_ids': get_cuda(input_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'segment_ids': get_cuda(segment_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'input_masks': get_cuda(input_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'mention_ids': get_cuda(mention_id[:cur_bsz, :batch_max_length].contiguous()),
                   'speaker_ids': get_cuda(speaker_id[:cur_bsz, :batch_max_length].contiguous()),
                   'label_ids': get_cuda(label_ids[:cur_bsz, :self.relation_num].contiguous()),
                   'turn_masks': get_cuda(turn_masks[:cur_bsz, :batch_max_length, :batch_max_length].contiguous()),
                   'graphs': graph_list
                   }
