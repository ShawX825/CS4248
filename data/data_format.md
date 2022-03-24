Data directory:
+ `original`: raw data
+ `rewrite_first_pronoun`: replaced "I, we, us, me" with "speaker x".
+ `remove_stopwords`: remove stopwords.

## 1. rewrite first pronoun

refer to `utils.py`.

We created a blacklist for pronoun replacement in
`
def get_mention2main_dict(doc, black_list=["me","us","we","i"])
`.


The detailed rules are:
```
if token.lower() in ["me","i"]:
            new_sentence += speaker + " " 
         elif token.lower() == "my":
            new_sentence += speaker+"'s" + " "
         elif token.lower() == "i'm":
            new_sentence += speaker+" "+"is" + " " 
         elif token.lower() == "am":
            new_sentence += "is" + " "
```
## 2. remove stopwords
