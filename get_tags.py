import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('ted_main.csv')
df_tr = pd.read_csv('transcripts.csv')

# align transcripts and tags

url1 = df['url'].values
url2 = df_tr['url'].values

indexes = []

for i, u in enumerate(url1):
	if u not in url2:
		indexes.append(i)

df.drop(df.index[indexes], inplace=True)
url1 = df['url'].values
		
print 'total len:', len(df.values)
print 'total len:', len(df_tr.values)



unique, counts = np.unique(url2, return_counts=True)
d = dict(zip(unique, counts))
for i in d:
	if d[i]>1:
		print i



tags = df['tags']

num_rows = df.shape[0]
all_tags=[]
talk_tags = []

for i in range(num_rows):
	
	l=tags.values[i].replace('\'',',')
	l=l.replace('[',',')
	l=l.replace(']',',')
	l = l.split(',')

	l = filter(lambda x: x!='', l)
	l = filter(lambda x: x!=' ', l)

	all_tags.extend(l)
	talk_tags.append(l)

set_tags = set(all_tags)

freq = [all_tags.count(w) for w in set_tags]
tf = {a:i for a,i in zip(set_tags,freq)}

#print tf['culture']

tf = sorted(tf.iteritems(), key=lambda (k,v): (v,k))

print tf


# change number of classes here


tf2=[]
tags_50=[]
for a,i in tf[-50:]:
	tf2.append((a,i))
	tags_50.append(a)

with open('tags_50', 'wb') as fp:
	pickle.dump(tags_50, fp)
#print tf2
#print len(tf2)


# get number of talks with top 50 tags


def intersection(lst1, lst2):
 
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3	

c=0

for t in talk_tags:
	if(intersection(t, tags_50) != []):
		c+=1

print 'total talks:', len(talk_tags)
print 'count of talks within 50 tags:', c


# get new tag list with only those in top 50

for i,tt in enumerate(talk_tags):
	for j,t in enumerate(tt):
		if t not in tags_50:
			talk_tags[i][j] = 'UNK'

#for tt in talk_tags:
#	tt.append('<eos>')
	
print talk_tags[0]
print len(talk_tags)


with open('talk_tags','wb') as fp:
	pickle.dump(talk_tags,fp)


def get_talk_with_tag(talk_tags, tag):
	
	talks = []

	for i,tt in enumerate(talk_tags):
		if tag in tt:
			talks.append(i)

	
	return talks
	
