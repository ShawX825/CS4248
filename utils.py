from audioop import reverse
from dataclasses import replace
import spacy
import neuralcoref
import json
from tqdm import tqdm


def concatenate_sentences(text):
   text_concated = ""
   for sentence in text:
      text_concated += sentence + "\n"
   return text_concated


def get_mention2main_dict(doc, black_list=["me","us","we","i"]):
   # Add neural coref to SpaCy's pipe
   nlp = spacy.load('en')
   neuralcoref.add_to_pipe(nlp,greedyness=0.5)

   # Get the naive coref_clusters for the doc
   doc = nlp(doc)

   mention2main = {}   # A dict with key:mention instance(span) and value:main instance(span)
   coref_clusters = doc._.coref_clusters
   
   for cluster in coref_clusters:
      main = cluster.main
      mentions = cluster.mentions
      for mention in mentions:
         if mention.text.lower() not in black_list: # Remove the mention in blacklist
            assert mention not in mention2main
            mention2main[mention] = main
   
   return mention2main


def replace_pronouns(text, mention2main, debug=False):
   # print("before replacing: ", text)
   
   for mention, main in sorted(mention2main.items(), key=lambda x:x[0].start, reverse=True):
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
   sentences = doc.split("\n")
   sentences = sentences[:-1] # Remove the space
   return sentences


def replace_pronoun_with_speaker(sentences):
   sentences_new = []
   for sentence in sentences:
      new_sentence = ""
      naive_tokens = sentence.split(" ")
      speaker = str(naive_tokens[0]) +" "+ str(naive_tokens[1])[:-1]
      new_sentence += speaker+':'+" "
      i = 2
      # Some rules to replace the pronoun and "be" verb
      while i < len(naive_tokens):
         token = naive_tokens[i]
         if token.lower() in ["me","i"]:
            new_sentence += speaker + " " 
         elif token.lower() == "my":
            new_sentence += speaker+"'s" + " "
         elif token.lower() == "i'm":
            new_sentence += speaker+" "+"is" + " " 
         elif token.lower() == "am":
            new_sentence += "is" + " "
         else:
            new_sentence += naive_tokens[i] + " "
         i += 1
      sentences_new.append(new_sentence)
   return sentences_new
      

def pipeline(sentences, black_list, replace_pronoun=False):
   if replace_pronoun:
      sentences = replace_pronoun_with_speaker(sentences)
   doc = concatenate_sentences(sentences)
   mention2main = get_mention2main_dict(doc, black_list=black_list)
   doc_processed = replace_pronouns(doc, mention2main, debug=False)
   sentences = split_doc(doc_processed)
   return sentences


def process_data(data, black_list=["me","us","we","i"],replace_pronoun=False):
   processed_data = []

   for sample in tqdm(data):
      sample_processed = []
      sample_text = sample[0]
      sample_label = sample[1]
      sample_text_processed = pipeline(sample_text,black_list,replace_pronoun)
      sample_processed.append(sample_text_processed)
      sample_processed.append(sample_label)
      processed_data.append(sample_processed)
   
   return processed_data


def dump_data(data, filename):
   out_file = open(filename, "w")
   json.dump(data, out_file, indent=2)
   out_file.close()


