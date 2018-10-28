'''
	get tf-idf scores of words in bow for a tag
	train model using logistic regression
	repeat for every tag
'''
import pickle
import numpy as np
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 2 is simply randomized version of first
# hope is that more accurate train, test distribution

with open('tags_train2', 'rb') as fp:
	tags_train = pickle.load(fp)

with open('tags_test2', 'rb') as fp:
	tags_test = pickle.load(fp)


with open('data_train_bow_text2', 'rb') as fp:
	data_train = pickle.load(fp)


with open('data_test_bow_text2', 'rb') as fp:
	data_test = pickle.load(fp)


def get_talks_with_tag(talk_tags, tag):

        talks = []

        for i,tt in enumerate(talk_tags):
                if tag in tt:
                        talks.append(i)


	print 'total', len(talks)
        return talks



def shuffle(X, ind):
	
	T = X[:]
	for i,ii in enumerate(ind):
		T[i] = X[ii]

	return T


def get_data(data, tag, bow, tags_train):

	talks = get_talks_with_tag(tags_train, tag)
	X_p = [data[i] for i in talks]
	labels_p = [1]*len(X_p)

	talks_n = range(len(data))
	for i in talks:
		talks_n.remove(i)
	
	talks_n = random.sample(talks_n, len(X_p))
	X_n = [data[i] for i in talks_n]
	labels_n = [0]*len(X_n)
	
	X = X_p[:]
	Y = labels_p[:]
	X.extend(X_n)
	Y.extend(labels_n)
	
	shuffled_ind = range(len(X))
	random.shuffle(shuffled_ind)
	X  = shuffle(X, shuffled_ind)
	Y  = shuffle(Y, shuffled_ind)

	return get_features(X, bow), Y


def get_features(transcripts, bow):

	countvec = CountVectorizer(vocabulary=bow)
	X = countvec.fit_transform(transcripts)
        tf_transformer = TfidfTransformer(use_idf=True).fit(X)
        X_train_tf = tf_transformer.transform(X)

	print X_train_tf.shape

	return X_train_tf.toarray()



def chunks(l, k):

    a = [0]*k
    n = int(len(l)/k)
	
    it = range(0, len(l), n)
    if((float(len(l)) % float(k)) != 0):
    	it.pop()
    for j,i in enumerate(it):
        a[j] =  l[i:i + n]

    return a



def get_CV_data(X, Y, K, i):

	X_chunks = chunks(X,K)
	Y_chunks = chunks(Y,K)

	X_cv = X_chunks[i]
	Y_cv = Y_chunks[i]

	del X_chunks[i]
	del Y_chunks[i]

	X_t = np.concatenate(X_chunks, 0)
	Y_t = np.concatenate(Y_chunks, 0)

	return X_t, Y_t, X_cv, Y_cv



def get_true_positive(model, X, Y):

	pred_p = 0
	t_p = 0

	for x,y in zip(X, Y):
		
		y_pred = model.predict([x])
		#print y_pred
		#print y
		if(y_pred[0]==1 and y==1):
			pred_p += 1
			t_p += 1
		elif y==1:
			t_p += 1

	
	#print 'pred_p', pred_p
	#print 't_p', t_p
	true_pos = float(pred_p) / float(t_p)
	
	return true_pos


	
def get_false_positive(model, X, Y):

	pred_p = 0
	t_p = 0

	for x,y in zip(X, Y):
		
		y_pred = model.predict([x])
		#print y_pred
		#print y
		if(y_pred[0]==1 and y==0):
			pred_p += 1
			t_p += 1
		elif y==0:
			t_p += 1

	
	#print 'pred_p', pred_p
	#print 't_p', t_p
	true_pos = float(pred_p) / float(t_p)
	
	return true_pos
			



def classify(X, Y):

	# perform 5 fold cross validation

	K = 5

	C_sum = 0

	for x in range(K):


		X_t, Y_t, X_cv, Y_cv = get_CV_data(X, Y, K, x)

		#inc = np.arange(0.0001, 0.001, 0.0001)
		inc = np.arange(0.01, 1.01, 0.01)

		
		max_score=0		# for maximizing
		#max_score=1		# for minimizing
		max_C = 0

		for i in inc:
			#LogReg =  LogisticRegression(C=i, max_iter=10000)
			#model = LogReg.fit(X_t, Y_t)

			#clf = MLPClassifier(hidden_layer_sizes=(100, 10), solver = 'lbfgs', alpha=i)
			#model = clf.fit(X_t, Y_t)

			MultiNB =  MultinomialNB(alpha=i)
			model = MultiNB.fit(X_t, Y_t)

			#GaussNB =  GaussianNB()
			#model = GaussNB.fit(X_t, Y_t)

			score = model.score(X_cv, Y_cv)
			#score = get_false_positive(model, X_cv, Y_cv)
			#score = get_true_positive(model, X_cv, Y_cv)
	
			if(score>max_score):		# for maximizing
			#if(score < max_score):		# for minimizing
				max_C = i
				max_score=score

		C_sum += max_C

	C_f = C_sum/K
	print "C: ", C_f
	MultiNB =  MultinomialNB(alpha=C_f)
	model = MultiNB.fit(X, Y)

	#GaussNB =  GaussianNB()
	#model = GaussNB.fit(X, Y)

	#clf = MLPClassifier(hidden_layer_sizes=(100, 10), solver = 'lbfgs', alpha=C_f)
	#model = clf.fit(X, Y)

	return model



if __name__=='__main__':
	
	tag = 'technology'
	with open('./tag_bow100/tag:'+tag, 'rb') as fp:
		bow = pickle.load(fp)

	X, Y = get_data(data_train, tag, bow, tags_train)
	model = classify(X, Y)
	print model
	
	X_t, Y_t = get_data(data_test, tag, bow, tags_test)
	print 'test data:', len(X_t)
	print model.score(X_t, Y_t)
	tp = get_true_positive(model, X_t, Y_t)
	print "recall:", get_true_positive(model, X_t, Y_t)
	fp = get_false_positive(model, X_t, Y_t)
	precision = tp / (tp + fp)
	print "precision:", precision
	print "false +ve:", fp
	print "F score:", (2 * (precision * tp) / (precision + tp))
