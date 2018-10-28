## Classification of TED Talks into Tags

#### Project to classify TED talks into different tags using their transcripts. This is a multiclass classification problem due to the large number of possible tags for each video. We use only the top 50 tags with the highest frequency of occurence. The problem is solved as a one-vs-all problem for each of the tags. </br>

#### Our feature vector consists of the tf-idf scores for a bag of words of size 100 for each of the tags. This bag of words is selected by taking the top 100 words with the highest tf-idf scores for each tag. We then apply a number of algorithms such as Naive Bayes, Logistic Regression and Neural Networks to classify the TED talk into a tag. We also perform 5 fold cross validation to choose the appropriate parameter values.
#### </br>

### Requirements
* Python 2.7
* NLTK
* Scikit-learn
* Pandas
* Numpy
#### </br>

### Dataset
#### Download the files ''ted_main.csv' and 'transcripts.csv' from https://www.kaggle.com/rounakbanik/ted-talks.
#### </br>

### Running the Code
#### To perform the necessary precomputations, run:
#### ```>>bash run.sh```
#### To perform one-vs-all classification for a particular tag, run:
#### ```>>python bow_classify.py```
#### To modify the tag for which you want to see the classification, change the tag in the main function in 'bow_classify.py'.
