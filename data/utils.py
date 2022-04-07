from audioop import reverse
from dataclasses import replace

from matplotlib.pyplot import text
import spacy
import neuralcoref
import json
from tqdm import tqdm
import re
from unidecode import unidecode  # $ pip install unidecode


def concatenate_sentences(text):
    text_concated = ""
    for sentence in text:
        text_concated += sentence + " "
    return text_concated


def get_mention2main_dict(doc, black_list=["me", "us", "we", "i"]):
    # Add neural coref to SpaCy's pipe
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp, greedyness=0.5)

    # Get the naive coref_clusters for the doc
    doc = nlp(doc)

    mention2main = {}  # A dict with key:mention instance(span) and value:main instance(span)
    coref_clusters = doc._.coref_clusters

    for cluster in coref_clusters:
        main = cluster.main
        mentions = cluster.mentions
        for mention in mentions:
            if mention.text.lower() not in black_list:  # Remove the mention in blacklist
                assert mention not in mention2main
                mention2main[mention] = main

    return mention2main


def replace_pronouns(text, mention2main, debug=False):
    # print("before replacing: ", text)

    for mention, main in sorted(mention2main.items(), key=lambda x: x[0].start, reverse=True):
        start = mention.start_char
        end = mention.end_char
        main_text = main.text
        if debug:
            print("text to be replaced: ", text[start:end])
            print("mention: ", mention)

        text = text[:start] + main_text + text[end:]

    # print("after replacing: ",text)
    return text


def split_doc(doc):
    sentences = re.sub("(Speaker \d):", r"\n\1:", doc).split("\n")
    sentences = sentences[1:]
    for i in range(len(sentences)):
        sentences[i] = sentences[i][:-1]
    return sentences


def replace_pronoun_with_speaker(sentences):
    sentences_new = []
    for sentence in sentences:
        new_sentence = ""
        naive_tokens = sentence.split(" ")
        speaker = str(naive_tokens[0]) + " " + str(naive_tokens[1])[:-1]
        new_sentence += speaker + ':' + " "
        i = 2
        # Some rules to replace the pronoun and "be" verb
        while i < len(naive_tokens):
            token = naive_tokens[i]
            token = re.sub("^(me|Me|I)(?=\W+$)", speaker, token)
            if token.lower() in ["me", "i"]:
                new_sentence += speaker + " "
            elif token.lower() == "my":
                new_sentence += speaker + "'s" + " "
            elif token.lower() == "i'm":
                new_sentence += speaker + " " + "is" + " "
            elif token.lower() == "i'll":
                new_sentence += speaker + " " + "will" + " "
            elif token.lower() == "i'd":
                new_sentence += speaker + " " + "would" + " "
            elif token.lower() == "am":
                new_sentence += "is" + " "
            else:
                new_sentence += token + " "
            i += 1
        new_sentence = new_sentence[:-1]  # Drop the last spaces

        # new_sentence = new_sentence.replace(u"\u2018", "'").replace(u"\u2019", "'") # Address the issue of symbol encoding
        sentences_new.append(new_sentence)
        # print(new_sentence)
    return sentences_new


def pipeline(sentences, black_list, replace_pronoun=False):
    if replace_pronoun:
        sentences = replace_pronoun_with_speaker(sentences)
    doc = concatenate_sentences(sentences)
    mention2main = get_mention2main_dict(doc, black_list=black_list)
    doc_processed = replace_pronouns(doc, mention2main, debug=False)
    sentences = split_doc(doc_processed)

    return sentences


# avodi utf-8 encoding issue
def encode_label(label):
    for i in range(len(label)):
        for key in label[i]:
            if isinstance(label[i][key], str):
                label[i][key] = unidecode(label[i][key])
            elif isinstance(label[i][key], int):
                continue
            else:
                for j in range(len(label[i][key])):
                    if isinstance(label[i][key][j], str):
                        label[i][key][j] = unidecode(label[i][key][j])
                    elif isinstance(label[i][key][j], int):
                        continue
    return label


def process_data(data, black_list=["me", "us", "we", "i"], replace_pronoun=False):
    processed_data = []

    for sample in tqdm(data):
        sample_processed = []
        sample_text = [unidecode(sentence) for sentence in sample[0]]
        # sample_text = unidecode(sample[0])
        sample_label = encode_label(sample[1])
        sample_text_processed = pipeline(sample_text, black_list, replace_pronoun)
        sample_processed.append(sample_text_processed)
        sample_processed.append(sample_label)
        processed_data.append(sample_processed)

    return processed_data


def dump_data(data, filename):
    out_file = open(filename, "w")
    json.dump(data, out_file, indent=2)
    out_file.close()


