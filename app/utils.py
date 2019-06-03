import pickle

with open('task1/model/cls.pkl', 'rb') as fp:
  cls_m1 = pickle.load(fp)
with open('task1/model/sentiment_tfidf.pkl', 'rb') as fp:
  tfidf_m1 = pickle.load(fp)
with open('task2/model/cls.pkl', 'rb') as fp:
  cls_m2 = pickle.load(fp)
with open('task2/model/sentiment_tfidf.pkl', 'rb') as fp:
  tfidf_m2 = pickle.load(fp)

# with open('data_m1.pkl', 'wb') as f:
#     result = []
#     vocab = tfidf_m1.vocabulary_
#     coef = cls_m1.coef_[0]
#     for token in vocab:
#         idx = vocab[token]
#         weight = coef[idx]*tfidf_m1.idf_[idx]
#         tmp = [token, coef[idx], tfidf_m1.idf_[idx], weight]
#         result.append(tmp)
#     pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('view/data_m2.pkl', 'wb') as f:
    result = []
    vocab = tfidf_m2.vocabulary_
    coef = cls_m2.coef_[0]
    for token in vocab:
        idx = vocab[token]
        weight = coef[idx]*tfidf_m2.idf_[idx]
        if abs(weight) > 10:
            tmp = [token, coef[idx], tfidf_m2.idf_[idx], weight]
            result.append(tmp)
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)