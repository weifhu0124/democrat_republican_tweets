import numpy as np
import pickle
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

def get_terms():
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
	explainer = LimeTextExplainer(class_names=['deceptive', 'truthful'])
	idx = 6
	exp = explainer.explain_instance(test_data[idx], c.predict_proba)
	print('Document id: %d' % idx)
	print('Probability(truthful) =', c.predict_proba([test_data[idx]])[0, 1])
	print(c.predict_proba([test_data[idx]]))
	exp.save_to_file('html/result.html')


if __name__ == '__main__':
	get_lime()
	# pos, neg = get_terms()
	# print('truthful words', pos)
	# print('deceptive words', neg)