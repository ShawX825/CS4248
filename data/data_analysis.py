import json
import spacy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from spacy.lang.en import English


# Data Analysis
def get_dialogue_tokens(data):
    dialogue_length = {}
    tokenizer = English()
    for i in range(len(data)):
        dialogue = data[i][0]
        label = data[i][1]
        cur_dialogue_length = 0
        for j in range(len(dialogue)):
            sentence = dialogue[j]
            tokens = tokenizer(sentence)
            cur_dialogue_length += len(tokens)
        dialogue_length[cur_dialogue_length] = 1 + dialogue_length.get(cur_dialogue_length,0)
    return dialogue_length

def get_dialogue_sentences(data):
    dialogue_sentences = {}
    for i in range(len(data)):
        dialogue = data[i][0]
        label = data[i][1]
        cur_dialogue_sentences = len(dialogue)
        dialogue_sentences[cur_dialogue_sentences] = 1 + dialogue_sentences.get(cur_dialogue_sentences,0)
    return dialogue_sentences

def get_data_distribution(data):
    label_distribution = {}
    for i in range(len(data)):
        dialogue = data[i][0]
        labels = data[i][1]
        for j in range(len(labels)):
            cur_label_ids = labels[j]["rid"]
            for label_id in cur_label_ids:
                label_distribution[label_id] = label_distribution.get(label_id,0) + 1
    return label_distribution

def get_pronoun_number(data):
    total_number = 0
    pronoun_number = 0
    pronoun = ["me","i","he","him","her","she","we","us","them","they","it","you","mine","yours","theirs"]
    for i in range(len(data)):
        dialogue = data[i][0]
        labels = data[i][1]
        for j in range(len(dialogue)):
            cur_sentence = dialogue[j].split(" ")
            for token in cur_sentence:
                if token.lower() in pronoun:
                    pronoun_number += 1
                total_number += 1
    return pronoun_number, total_number
