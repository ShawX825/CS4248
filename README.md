# Dialogue Relation Extraction
### Group 19
### Member: Liu Xiao, Zhang Heng, Zhang Jian
### Mentor: Xue Fuzhao

This project aims to identify relation types between two entities given a dialogue text as evidence.

We used the TUCORE-GCN model as baseline, and identified its shortcomings in terms of encoding techniques and model structure. 
In terms of encoding techniques, we proposed that entities can benefit from three types of dialogue turns: turns that contains 
this entity, turns uttered by the same speaker, and turns discussing about the same topic. 
In terms of model structure, we augment information flow efficiency by changing many-to-one communication to one-to-one communication. 
As a result, our model achieved similar performance against the existing relation graph convolution module.

For the usage and introduction of each component, please refer to the introduction in each sub-folders.

### Reproducibility
To reproduce our training process in main experiments, 
- download [RoBERTa](https://github.com/pytorch/fairseq/tree/main/examples/roberta) and unzip it to ```HAN/pre-trained_model/RoBERTa/```.
- download ```merges.txt``` and ```vocab.json``` from [here](https://huggingface.co/roberta-large/tree/main). 
- run following command under ```HAN```
  - ```export RoBERTa_LARGE_DIR=/path/to/HAN/pre-trained_model/RoBERTa```
  - ```nohup python run_classifier.py --do_train --do_eval --encoder_type RoBERTa  --data_dir ../data/original --data_name DialogRE  --vocab_file $RoBERTa_LARGE_DIR/vocab.json --merges_file $RoBERTa_LARGE_DIR/merges.txt  --config_file $RoBERTa_LARGE_DIR/config.json   --init_checkpoint $RoBERTa_LARGE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 10   --learning_rate 7.5e-6   --num_train_epochs 30   --output_dir HAN --gradient_accumulation_steps 2 > train.log 2>&1 &```

To evaluate our results, run following command under ```HAN```
- three main results without coreference resolution, which also can be found in ```HAN/results/HAN/main_result/```
  - ```nohup python evaluate.py --dev ../data/original/dev.json --test ../data/original/test.json --f1dev results/HAN/main_result/0/logits_dev.txt --f1test results/HAN/main_result/0/logits_test.txt --f1cdev results/HAN/main_result/0/logits_devc.txt --f1ctest results/HAN/main_result/0/logits_testc.txt --result_path results/HAN/main_result/0/result.txt > eval.log 2>&1 &```
- three main results with coreference resolution or other ablation study results, which also can be found in ```HAN/results/HAN/```. Use the same command as above, substitute ```results/HAN/main_result/``` with coresponding paths. 


