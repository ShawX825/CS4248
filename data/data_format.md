Data directory:
+ `original`: raw data
+ `naive_pronoun_resolution`: The datasets processed by neuralcoref without any heuristic rules.
+ `data_processed_without_rewrite`: The datasets processed by neuralcoref with Black List
+ `rewrite_first_pronoun`: The datasets processed by neuralcoref with Pronoun Rewrite.



## 1. naive_pronoun_resolution

To study the effect of our heuristic rules proposed, we also build up a dataset which is purely processed by neuralcoref

## 2. data_processed_without_rewrite

For the first person pronouns, we do not process them with Coreference Resolution.

## 3. rewrite_first_pronoun

Instead of studying the relationship between the pronouns in the turn of different speakers, we directly replace the first person pronouns and possessive determiners with the corresponding speaker. (e.g. "My father" in the turn of Speaker 1 would be replaced with "Speaker 1's father")

