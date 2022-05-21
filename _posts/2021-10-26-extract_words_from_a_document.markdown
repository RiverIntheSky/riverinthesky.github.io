---
layout: post
title: How to extract lemmatized words from a document
date:   2021-10-26 21:30:00 +0200
tags: programming python NLP
---
A friend introduced to me [this GitHub repo](https://github.com/RFG1024/GlossaryGenerator), which can extract lemmas from a document; it then compares the lemmas with the entries in a dictionary and outputs the less common lemmas that appear in the dictionary. The idea looks very nice, but I soon discovered that the [stemmer](https://www.geeksforgeeks.org/introduction-to-stemming/) does not parse the words correctly. Examples:
```python
>>> porter_stemmer.stem('evening')
'even'
>>> porter_stemmer.stem('simplified')
'simplifi'
>>> porter_stemmer.stem('troubles')
'troubl'
```
This results in a lot of words that are just wrong and thus not present in the dictionary. The [Snowball stemmer](https://snowballstem.org/) is doing a better job but still not satisfying. A better strategy is to use a [lemmatizer](https://www.geeksforgeeks.org/python-lemmatization-with-nltk/):
```python
>>> from nltk.stem import WordNetLemmatizer
>>> lemmatizer = WordNetLemmatizer()
>>> lemmatizer.lemmatize('troubles')
'trouble'
```
This is already much better. However, it leaves many words unchanged since the lemmatizer doesn't know their [part of speech](https://en.wikipedia.org/wiki/Part_of_speech):
```python
>>> lemmatizer.lemmatize('simplified')
'simplified'
>>> lemmatizer.lemmatize('simplified', 'v') # 'v' stands for verb
'simplify'
```
The lemmatizer only finds the correct lemma when we explicitly say that we are processing a verb. The part-of-speech tags in a sentence can be detected automatically by some [Natural Language Processing](https://www.ibm.com/cloud/learn/natural-language-processing) engines such as [spaCy](https://spacy.io/):
```python
>>> import spacy
>>> nlp = spacy.load('en_core_web_sm')
>>> sentence = "I had some simplified troubles this evening."
>>> doc = nlp(sentence)
>>> ' '.join([token.lemma_ for token in doc])
'I have some simplify trouble this evening .'
```
`have -> had`, `simplified -> simplify`, `troubles -> trouble`, all the lemmas are detected correctly. The process would be slower but more accurate than the original implementation.

I hereby provide the complete script. The dependencies are [textract](https://textract.readthedocs.io/en/stable/), [NLTK](https://www.nltk.org/) and [spaCy](https://spacy.io/). I installed the packages with [Miniconda](https://docs.conda.io/en/latest/miniconda.html) using `conda install -c conda-forge spacy nltk textract`. spaCy models are downloaded with `spacy download en_core_web_sm`. This script also includes other minor changes, such as how the punctuations are handled.
```python
filename = 'pride_and_prejudice.txt'
discard = 8000 # the first words in the dictionary to be discarded
dictionary = 'common30k.txt'

import string
import textract
from collections import Counter
from string import punctuation
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm

def remove_punctuation_and_digits(from_text):
    # map '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789' to ''
    table = str.maketrans('', '', string.punctuation + string.digits)
    stripped = [w.translate(table) for w in from_text]
    return stripped

print('Initializing...')
# Initialize spacy model
nlp = spacy.load('en_core_web_sm')

print('Reading', filename, '...')
# read file into a list of sentences
byte = textract.process(filename)
text = byte.decode("utf-8")
print('Done')

print('Extracting tokens...')
tokenized_text = sent_tokenize(text)
tokens = [[word for word in line.split()] for line in tokenized_text]
print('Done')

print('Extracting lemmas...')
lemmas = []
for s in tqdm(tokens):
    doc = nlp(' '.join(s))
    lemmas.append(remove_punctuation_and_digits(
        [token.lemma_ for token in doc]))

print('Done')

print('Sorting words...')
# remove stopwords such as 'the', 'i'
sw = stopwords.words('english')
sw.append('nt')
words = [token for sentence in lemmas for token in sentence
         if (token.lower() not in sw and token.isalnum())]
word_count = Counter(words)
# favors words with higher frequency
sorted_words = sorted(word_count, key=word_count.get, reverse=True)

glossary = filename.split('.')[0] + '_glossary.txt'
print('and discarding the first', discard, 'words in', dictionary, '...')
with open(dictionary) as dict:
    dict_words = [word for line in dict.readlines() for word in line.split()]
    less_common_dict_words = dict_words[discard:]
    new_words = [word for word in sorted_words
                 if word in less_common_dict_words]

print("Done. There are", len(new_words), "potential new words")

with open(glossary, 'w') as output:
    output.write('\n'.join(new_words))
    print("Wrote to", glossary)
```
spaCy can also process text in [other languages](https://spacy.io/usage/models), for example German:
```python
>>> nlp = spacy.load('de_core_news_sm')
>>> doc = nlp("Brauner Bursche führt zum Tanze sein blauäugig schönes Kind; schlägt die Sporen keck zusammen, Csardas-Melodie beginnt; küßt und herzt sein süßes Täubchen, dreht sie, führt sie, jauchzt und springt; wirft drei blanke Silbergulden auf das Cimbal, daß es klingt.")
>>> ' '.join([token.lemma_ for token in doc])
'Brauner Bursche führen zum Tanz mein blauäugig schön Kind ; schlagen der Spore keck zusammen , Csardas-Melodie beginnen ; küssen und herzen mein süß Täubchen , drehen ich , führen ich , jauchzen und springen ; werfen drei blank Silbergulden auf der Cimbal , daß ich klingen .'
```
Reference links:

[Lemmatization Approaches with Examples in Python](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/)

[How to find the lemmas and frequency count of each word in list of sentences in a list?](https://stackoverflow.com/questions/52860350/how-to-find-the-lemmas-and-frequency-count-of-each-word-in-list-of-sentences-in)