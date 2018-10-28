import pickle

def get_text(transcripts):

	for i, t in enumerate(transcripts):
		transcripts[i] = ' '.join(t)

	return transcripts


with open('data_train_bow', 'rb') as fp:
	data_train_bow = pickle.load(fp)

with open('data_test_bow', 'rb') as fp:
	data_test_bow = pickle.load(fp)


data_train_bow_text = get_text(data_train_bow)
data_test_bow_text = get_text(data_test_bow)
		

with open('data_test_bow_text', 'wb') as fp:
	pickle.dump(data_test_bow_text, fp)

with open('data_train_bow_text', 'wb') as fp:
	pickle.dump(data_train_bow_text, fp)
