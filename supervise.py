#!/bin/python
def feature_engineering(sentiment):
	"""Perform feature engineering on the data

	:return: features for logistic regression
	"""
	from sklearn import preprocessing
	sentiment.le = preprocessing.LabelEncoder()
	sentiment.le.fit(sentiment.train_labels)
	sentiment.target_labels = sentiment.le.classes_
	sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
	sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
	from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
	from nltk.corpus import stopwords
	stopWords = set(stopwords.words('english'))
	stopWords.remove('not')
	stopWords.remove('but')
	stopWords.add('chicago')
	stopWords.add('michigan')
	sentiment.count_vect = CountVectorizer(stop_words=list(stopWords), ngram_range=(1,3),min_df=2)
	sentiment.tfidf_vect = TfidfVectorizer(stop_words=list(stopWords), ngram_range=(1,3), min_df=2, sublinear_tf=True)
	sentiment.trainX = sentiment.tfidf_vect.fit_transform(sentiment.train_data)
	sentiment.devX = sentiment.tfidf_vect.transform(sentiment.dev_data)

def process_unlabeled(unlabeled, sentiment):
	"""Process the unlabeled data to have the same feature engineering procedure

	:return: processed unlabeled data
	"""
	unlabeled.X = sentiment.tfidf_vect.transform(unlabeled.data)
	return unlabeled
