import pickle

def get_text(transcripts):

	for i, t in enumerate(transcripts):
		transcripts[i] = ' '.join(t)

	return transcripts


with open('data_train_bow2', 'rb') as fp:
	data_train_bow2 = pickle.load(fp)

with open('data_test_bow2', 'rb') as fp:
	data_test_bow2 = pickle.load(fp)


data_train_bow_text2 = get_text(data_train_bow2)
data_test_bow_text2 = get_text(data_test_bow2)
		

with open('data_test_bow_text2', 'wb') as fp:
	pickle.dump(data_test_bow_text2, fp)

with open('data_train_bow_text2', 'wb') as fp:
	pickle.dump(data_train_bow_text2, fp)
