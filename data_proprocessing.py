import json
from utils import *

import os
if not os.path.exists('data/original'):
    os.makedirs('data/original')
if not os.path.exists('data/rewrite_first_pronoun'):
    os.makedirs('data/rewrite_first_pronoun')

# load training and testing data
training_data = json.load(open("data/original/train.json"))
testing_data = json.load(open("data/original/test.json"))
dev_data = json.load(open("data/original/dev.json"))

# process data 
training_data_processed = process_data(training_data,replace_pronoun=True)
testing_data_processed = process_data(testing_data,replace_pronoun=True)
dev_data_processed = process_data(dev_data,replace_pronoun=True)

# dump data
dump_data(training_data_processed, filename="data/rewrite_first_pronoun/train_processed.json")
dump_data(testing_data_processed, filename="data/rewrite_first_pronoun/test_processed.json")
dump_data(dev_data_processed, filename="data/rewrite_first_pronoun/dev_processed.json")


