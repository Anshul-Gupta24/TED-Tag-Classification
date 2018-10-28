import pickle
import nltk
from collections import defaultdict

data=[]
with open('data', 'rb') as fp:
	data = pickle.load(fp)
	fp.close()

words = []
for t in data:
        #print t
        for s in t:
                words.append(s)


vocab_train = list(set(words))
#print vocab_train
print len(vocab_train)

vtrain = nltk.FreqDist(words)
v20000 = [word for (word, _) in vtrain.most_common(20000)]

#print len(v20000)


with open('vocab', 'wb') as fp:
	pickle.dump(v20000, fp)

