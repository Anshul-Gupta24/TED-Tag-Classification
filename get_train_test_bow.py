'''
	get training and test data
'''

import pickle
import get_tags
import pandas as pd
import nltk
import random




# change fraction of test data
frac_test = 0.1


df = pd.read_csv('transcripts.csv',encoding='utf-8')

transcripts = df['transcript'].values
transcripts = [s.lower() for s in transcripts]
transcripts = [s.replace('-',' ') for s in transcripts]



transcripts = [nltk.word_tokenize(t) for t in transcripts]


# remove stopwords

from nltk.corpus import stopwords

stwords = set(stopwords.words('english'))

for i,t in enumerate(transcripts):
        for j,w in enumerate(t):
                if w in stwords:
                        transcripts[i][j] = 'UNK'


# stem words

from nltk.stem.porter import *

stemmer = PorterStemmer()

for i,t in enumerate(transcripts):
        for j,w in enumerate(t):
                transcripts[i][j] = stemmer.stem(w)



with open('talk_tags', 'rb') as fp:
	talk_tags = pickle.load(fp)

print len(talk_tags)

with open('tags_50', 'rb') as fp:
	tags_50 = pickle.load(fp)


data_train_bow = []
data_test_bow = []
tags_train = []
tags_test = []
used_indices = []


def shuffle(X, ind):

        T = X[:]
        for i,ii in enumerate(ind):
                T[i] = X[ii]

        return T

shuffled_ind = range(len(transcripts))
random.shuffle(shuffled_ind)

transcripts = shuffle(transcripts, shuffled_ind)
talk_tags = shuffle(talk_tags, shuffled_ind)


for t in tags_50:

	talks = get_tags.get_talk_with_tag(talk_tags, t)
	talks2 = talks[:]
	
	for tk in talks2:
		if(tk in used_indices):
			talks.remove(tk)

	used_indices.extend(talks)

	num_talk_test = int(frac_test * len(talks))
	test_indices = talks[:num_talk_test]
	train_indices = talks[num_talk_test:]

	data_train_bow.extend([transcripts[i] for i in train_indices])
	data_test_bow.extend([transcripts[i] for i in test_indices])
	tags_train.extend([talk_tags[i] for i in train_indices])
	tags_test.extend([talk_tags[i] for i in test_indices])

print len(used_indices)
print len(tags_train)

with open('tags_train', 'wb') as fp:
	pickle.dump(tags_train, fp)

with open('tags_test', 'wb') as fp:
	pickle.dump(tags_test, fp)

with open('data_train_bow', 'wb') as fp:
	pickle.dump(data_train_bow, fp)

with open('data_test_bow', 'wb') as fp:
	pickle.dump(data_test_bow, fp)

