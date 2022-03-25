Data directory:
+ `original`: raw data
+ `rewrite_first_pronoun`: The datasets processed by neuralcoref with customized heuristic rules.
+ `remove_stopwords`: The datasets processed by neuralcoref with customized heuristic rules as well as removed stopwords.
+ `naive_pronoun_resolution`: The datasets processed by neuralcoref without any heuristic rules.

## 1. rewrite_first_pronoun

refer to `utils.py`.

We created a blacklist for pronoun replacement to avoid replacing the first person pronouns - ["we","us","I","me"].

Apart from the black list implemented, we also modify the pronouns "I" and "me" as the speaker himself/herself.

For example:
Speaker 1: "Yes, Ross is me". -> Speaker 1: "Yes, Ross is speaker 1."

## 2. remove_stopwords

On top of the rewrite_first_pronoun dataset, we removed the stopwords in the dataset to study the affect of stopwords regarding model performance.

## 3. naive_pronoun_resolution

To study the effect of our heuristic rules proposed, we also build up a dataset which is purely processed by neuralcoref