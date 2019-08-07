""" A neural chatbot using sequence to sequence model with
attentional decoder. 
This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
Sequence to sequence model by Cho et al.(2014)
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
This file contains the hyperparameters for the model.
See README.md for instruction on how to run the starter code.
"""
"""
Author: Arvind subramaniam Ramesh
Reg no : R00171371:
Msc Artificial intelligence:

"""

# parameters for processing the dataset
# DATA_PATH = 'data/cornell movie-dialogs corpus'
DATA_PATH = '/Users/aravi/PycharmProjects/chatbot/data/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus'
# C:\Users\aravi\PycharmProjects\chatbot\data\cornell_movie_dialogs_corpus\cornell movie-dialogs corpus
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
# LINE_FILE = 'chat.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000
""" Here am checking for the total length of each sentance in dataset and according creating bucket sizes
1. the max of words in a sentance was 45. Hence am splitting my bucket sizes as below to accomadate all the words"""
BUCKETS = [(5, 5), (10, 10), (15, 15), (25, 25), (35, 35), (50, 50)]


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "), 
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0
# NUM_SAMPLES = 512
NUM_SAMPLES = 600
ENC_VOCAB = 45621
DEC_VOCAB = 39822

ENC_VOCAB = 24499
DEC_VOCAB = 24639
ENC_VOCAB = 24496
DEC_VOCAB = 24712
ENC_VOCAB = 24445
DEC_VOCAB = 24725
ENC_VOCAB = 24445
DEC_VOCAB = 24725
ENC_VOCAB = 24445
DEC_VOCAB = 24725
