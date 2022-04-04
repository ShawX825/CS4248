# Data proprocessing

run `data_preprocessing.ipynb`. 

Remember to change the input and output directory.

Input directory is at line 56, variable `path_input_docs`.

The source data used `CS4248/data/original/[train, dev, test].json` from this repo.
Output directory is at line 127, `dump_data` params. Our result was saved to `CS4248/topic_seg/topic_data/[train, dev, test].json` in this repo. 

# Set up baseline, with Bert Tiny 

The baseline can be set up in Google drive using `CS4248/baseline/TUCORE_GCN.ipynb`, in which it will download the original TUCORE_GCN model, the dataset and the Bert tiny model. 

# Run experiment using origianl model 

We need to add three files. 

Firstly go to `CS4248/topic_seg/original_codes`. Assume the TUCORE_GCN model on Google Drive have root folder `TOCURE_GCN`.

+ Drag `data_original.py` into `TUCORE_GCN` so that it is in the same directory as `data.py`.

+ Drag `run_classifier_original.py` into `TUCORE_GCN` so that it is in the same directory as `run_classifier.py`.

+ Drag `models\BERT\TUCOREGCN_BERT_original.py` into `TUCORE_GCN\models\BERT` so that it is in the same directory as `TUCOREGCN_BERT.py`.

We can run the training and evaluation using the following commands.

```
# Train
!pip3 install dgl-cu101 dglgo -f https://data.dgl.ai/wheels/repo.html
!CUDA_LAUNCH_BLOCKING=1
!pip3 install transformers
!pwd
%env BERT_BASE_DIR=/content/gdrive/MyDrive/CS4248project/TUCORE-GCN/pre-trained_model/BERT/uncased_L-2_H-128_A-2
%env TUCORE_GCN_BASE_DIR = /content/gdrive/MyDrive/CS4248project/TUCORE-GCN
!python3 $TUCORE_GCN_BASE_DIR/run_classifier_original.py --do_train --do_eval  --encoder_type BERT  --data_dir $TUCORE_GCN_BASE_DIR/datasets/DialogRE/topic --data_name DialogRE   --vocab_file $BERT_BASE_DIR/vocab.txt   --config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 4.0   --output_dir $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_original  --gradient_accumulation_steps 2
```

The evaluation code is as follows: 

```
# Evaluate
!python $TUCORE_GCN_BASE_DIR/evaluate.py --dev $TUCORE_GCN_BASE_DIR/datasets/DialogRE/topic/dev.json --test $TUCORE_GCN_BASE_DIR/datasets/DialogRE/topic/test.json --f1dev $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_original/logits_dev.txt --f1test $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_original/logits_test.txt --f1cdev $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_original/logits_devc.txt --f1ctest $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_original/logits_testc.txt --result_path $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_original/result.txt
```

A sample of our result was saved to `CS4248/original_results/results.txt`. The content says:

```
best T2: 0.23
dev (P R F1): 0.3803071364046974 0.2636192861615529 0.3113905325443787
test (P R F1): 0.3627450980392157 0.26601307189542484 0.30693815987933637
```

# Run experiment using topic embedding model 

We need to add three files. 

Firstly go to `CS4248/topic_seg/topic_codes`. Assume the TUCORE_GCN model on Google Drive have root folder `TOCURE_GCN`.

+ Drag `data_embedding.py` into `TUCORE_GCN` so that it is in the same directory as `data.py`.

+ Drag `run_classifier_embedding.py` into `TUCORE_GCN` so that it is in the same directory as `run_classifier.py`.

+ Drag `models\BERT\TUCOREGCN_BERT_embedding.py` into `TUCORE_GCN\models\BERT` so that it is in the same directory as `TUCOREGCN_BERT.py`.

We can run the training and evaluation using the following commands.

```
# Train
!pip3 install dgl-cu101 dglgo -f https://data.dgl.ai/wheels/repo.html
!CUDA_LAUNCH_BLOCKING=1
!pip3 install transformers
!pwd
%env BERT_BASE_DIR=/content/gdrive/MyDrive/CS4248project/TUCORE-GCN/pre-trained_model/BERT/uncased_L-2_H-128_A-2
%env TUCORE_GCN_BASE_DIR = /content/gdrive/MyDrive/CS4248project/TUCORE-GCN
!python3 $TUCORE_GCN_BASE_DIR/run_classifier_embedding.py --do_train --do_eval  --encoder_type BERT  --data_dir $TUCORE_GCN_BASE_DIR/datasets/DialogRE/topic --data_name DialogRE   --vocab_file $BERT_BASE_DIR/vocab.txt   --config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 4.0   --output_dir $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_topic  --gradient_accumulation_steps 2
```

The evaluation code is as follows: 

```
# Evaluate
!python $TUCORE_GCN_BASE_DIR/evaluate.py --dev $TUCORE_GCN_BASE_DIR/datasets/DialogRE/topic/dev.json --test $TUCORE_GCN_BASE_DIR/datasets/DialogRE/topic/test.json --f1dev $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_topic/logits_dev.txt --f1test $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_topic/logits_test.txt --f1cdev $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_topic/logits_devc.txt --f1ctest $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_topic/logits_testc.txt --result_path $TUCORE_GCN_BASE_DIR/TUCOREGCN_BERT_DialogRE_topic/result.txt
```

A sample of our result was saved to `CS4248/original_results/results.txt`. The content says:

```
best T2: 0.15
dev (P R F1): 0.28671737858396723 0.3068252974326863 0.2964307320024198
test (P R F1): 0.28064903846153844 0.30522875816993467 0.29242329367564185
```