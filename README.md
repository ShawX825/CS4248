# Dialogue Relation Extraction
### Group 19
### Member: Liu Xiao, Zhang Heng, Zhang Jian
### Mentor: Xue Fuzhao

This project aims to identify relation types between two entities given a dialogue text as evidence.

We used the TUCORE-GCN model as baseline, and identified its shortcomings in terms of encoding techniques and model structure. 
In terms of encoding techniques, we proposed that entities can benefit from three types of dialogue turns: turns that contains 
this entity, turns uttered by the same speaker, and turns discussing about the same topic. 
In terms of model structure, we augment information flow efficiency by changing many-to-one communication to one-to-one communication. 
As a result, our model achieved similar performance against the existing relation graph convolution module, 
but the number of parameters reduced by 78.2\%.

For the usage and introduction of each component, please refer to the introduction in each sub-folders.
