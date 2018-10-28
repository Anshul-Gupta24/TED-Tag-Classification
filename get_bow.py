from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import pandas as pd
import random
import get_tags
import pickle
import numpy as np

data=[]
data_t=[]
vocab=[]
talk_tags=[]

with open('data_train_bow', 'rb') as fp:
	data_train = pickle.load(fp)

for t in data_train:
	data.append(' '.join(t))

with open('tags_train', 'rb') as fp:
	talk_tags = pickle.load(fp)

with open('tags_50', 'rb') as fp:
	tags_50 = pickle.load(fp)

with open('vocab','rb') as fp:
	vocab = pickle.load(fp)


# get tf-idf scores for a transcipt

def get_score(transcripts):

	countvec = CountVectorizer(vocabulary=vocab)


	X = countvec.fit_transform(transcripts)
        tf_transformer = TfidfTransformer(use_idf=True).fit(X)
        X_train_tf = tf_transformer.transform(X)
        #print X_train_tf

        return X_train_tf



# get bag of words for a tag

def get_bow(tag):

	talks = get_tags.get_talk_with_tag(talk_tags,tag)
	
	score = get_score([data[i] for i in talks])
		
	#print score
	score = np.array(score.todense())
	#print score.shape
	sum_score = np.sum(score, axis=0)
	#print sum_score
	np.squeeze(sum_score)
	#print sum_score.shape


	# get top 100 words with highest tf-idf scores

	indexes = sum_score.argsort()[-100:]

	bow = [vocab[i] for i in indexes]

	return bow



if __name__ == '__main__':

	for tag in tags_50:
		bow = get_bow(tag)

		with open('./tag_bow100/tag:'+tag, 'wb') as fp:
			pickle.dump(bow, fp)

	print get_bow('science')
