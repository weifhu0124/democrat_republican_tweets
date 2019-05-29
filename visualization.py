import numpy as np
import pickle
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt

def get_terms(idx):
	with open('model/logistic_regression.pkl', 'rb') as fp:
		cls = pickle.load(fp)
	with open('model/sentiment_tfidf.pkl', 'rb') as fp:
		sentiment = pickle.load(fp)
	print(cls.classes_)

	idx = np.argsort(np.asarray(cls.coef_[0]))
	terms = sentiment.vocabulary_
	terms = {v: k for k, v in terms.items()}
	positive_words = []
	negative_words = []
	for i in idx[:15]:
		negative_words.append(terms[i])
	for i in idx[-15:]:
		positive_words.append(terms[i])
	positive_words.reverse()
	return positive_words, negative_words

def get_lime():
	with open('model/logistic_regression.pkl', 'rb') as fp:
		cls = pickle.load(fp)
	with open('model/sentiment_tfidf.pkl', 'rb') as fp:
		sentiment_tfidf = pickle.load(fp)
	with open('model/sentiment_dev.pkl', 'rb') as fp:
		test_data = pickle.load(fp)
	c = make_pipeline(sentiment_tfidf, cls)
	explainer = LimeTextExplainer(class_names=['democrat', 'republican'])
	idx = 43
	exp = explainer.explain_instance(test_data[idx], c.predict_proba)
	print('Document id: %d' % idx)
	print('Probability(republican) =', c.predict_proba([test_data[idx]])[0, 1])
	exp.save_to_file('html/result.html')
	print(exp.as_list())
	#print(exp.map_exp_ids(exp.as_map()[1]))



def get_extreme_example():
	with open('model/logistic_regression.pkl', 'rb') as fp:
		cls = pickle.load(fp)
	with open('model/sentiment_tfidf.pkl', 'rb') as fp:
		sentiment_tfidf = pickle.load(fp)
	with open('model/sentiment_train.pkl', 'rb') as fp:
		train_data = pickle.load(fp)
	c = make_pipeline(sentiment_tfidf, cls)
	curmax = -1
	curmin = -1
	max_str = ''
	min_str = ''
	for i in range(len(train_data)):
		prob = c.predict_proba([train_data[i]])[0, 1]
		if prob > curmax:
			curmax = prob
			max_str = train_data[i]
		if prob < curmin or curmin == -1:
			curmin = prob
			min_str = train_data[i]
	print('Most republican:', max_str)
	print('Most democrat:', min_str)
	return max_str, min_str



if __name__ == '__main__':
	# get_extreme_example()
	get_lime()
	# pos, neg = get_terms(15)
	# print('republican words', pos)
	# print('democrat words', neg)