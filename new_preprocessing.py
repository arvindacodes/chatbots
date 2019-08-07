"""
Author: Arvind subramaniam Ramesh
Reg no : R00171371:
Msc Artificial intelligence:

"""


import re
import nltk

from string import punctuation
from nltk.corpus import stopwords
import os



#creating a empty list to load the dataset:
text = []
#getting the input file:
# with open('chat_test.txt','r',encoding="utf8") as file: ## sample file
with open('chat_test.txt', 'r', encoding="utf8") as file: ## original twitter dataset
    for line in file:
        text.append(line.rstrip("\n"))

# am creating a new function to do the complete preprocessing
def preprocessing(text):
    """removing emojis from text as the first part """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u200d"
                               u"\u2640-\u2642"
                               "]", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    """  Removing URL in the 2nd step"""
    url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")
    text = re.sub(url_regex, '', text)

    """removing characters [\], ['] and ["] from text. this also removes "'" in words like "didn't" """
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    """ removing spl characters from text"""
    text = re.sub('[^\w\s]', '' , text)
    text = re.sub('_','', text)


    # replacing punctuation characters with spaces
    filter = '!"\'โ#$%&()*+,-./:;<=>?@[\\]^_`{|}~คธ\t\n'
    translate_dict = dict((c, " ") for c in filter)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    #removing all numbers by replacing them with ' '

    text = re.sub(r'[0-9]+', '', text)

    # converting text to lowercase
    text = text.strip().lower()
    #print("the new text is :",text)
    return text
#putting the preprocessed data back to a new list  and tokenising .
corpus = []
for line in text:
    corpus.append(preprocessing(line))
    tokenized_word = nltk.word_tokenize(preprocessing(line))
    # corpus.append(tokenized_word)
    # print(tokenized_word)
print(corpus)
""" Here am following a question and answer approach. That is all the odd lines will be my question and even lines will
be my answer"""
questions = []
answers = []

for j in range(1, len(corpus)):
    if j%2 == 0:
        answers.append(corpus[j-1])
    elif j%2 != 0:
        questions.append(corpus[j-1])


""" here am creating the train and test encoding and decoding files. And this will be passed in to the data.py file"""
PROCESSED_PATH = '/Users/aravi/PycharmProjects/chatbot/processed'

filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
files = []
for filename in filenames:
    files.append(open(os.path.join(PROCESSED_PATH, filename), 'w',encoding="utf8"))
    """ am  segregating 75% of my data for training and remaining for my test file"""
for i in range(int(len(questions) * (3 / 4))):
    files[0].write(questions[i] + "\n")

for j in range(int(len(questions) * (3 / 4)), len(questions)):
    files[2].write(questions[j]+ "\n")

for k in range(int(len(answers) * (3 / 4))):
    files[1].write(answers[k]+ "\n")

for l in range(int(len(answers) * (3 / 4)), len(answers)):
    files[3].write(answers[l]+ "\n")

for file in files:
    file.close()

""" the below step is written to write the output of preprocessing to out.txt for easy evaluation"""
with open('out_test.txt', 'w',encoding="utf8") as f:
   for item in corpus:
       print(item, file=f)