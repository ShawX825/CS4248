import json
from utils import *

# load training and testing data
training_data = json.load(open("data/train.json"))
testing_data = json.load(open("data/test.json"))

# process data 
training_data_processed = process_data(training_data,replace_pronoun=True)
testing_data_processed = process_data(testing_data,replace_pronoun=True)

# dump data
dump_data(training_data_processed, filename="data/train_processed.json")
dump_data(testing_data_processed, filename="data/test_processed.json")

