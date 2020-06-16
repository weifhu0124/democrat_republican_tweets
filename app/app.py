from flask import Flask, request, jsonify
from flask import render_template
import random, sys
import json, pickle, re, string
from nltk import everygrams
#import pdb

sys.path.append("..")
app = Flask(__name__)
app.debug = True

'''
preload all necessary components
model1: PA2 model
model2: Twitter Democrat/Republican model
'''
with open('task1/model/cls.pkl', 'rb') as fp:
  cls_m1 = pickle.load(fp)
with open('task1/model/sentiment_tfidf.pkl', 'rb') as fp:
  tfidf_m1 = pickle.load(fp)
with open('task2/model/cls.pkl', 'rb') as fp:
  cls_m2 = pickle.load(fp)
with open('task2/model/sentiment_tfidf.pkl', 'rb') as fp:
  tfidf_m2 = pickle.load(fp)

def get_class_proba(lower_nopunc, cls, tfidf):
    X = tfidf.transform([lower_nopunc])
    return cls.predict_proba(X).tolist()[0]

def get_feature_weights(raw, lower_nopunc, raw_nopunc, cls, tfidf, gram_length):
  X = tfidf.transform([lower_nopunc])
  coef = cls.coef_[0]
  lower_grams = list(everygrams(lower_nopunc.split(), gram_length, gram_length))
  original_grams = list(everygrams(raw_nopunc.split(), gram_length, gram_length))
#   print(lower_grams)
  # print(len(original_grams), len(lower_grams))
  vocab = tfidf.vocabulary_
  weights = []
  for i in range(len(lower_grams)):
    lower_token, original_token = " ".join(lower_grams[i]), " ".join(original_grams[i])
    if lower_token in vocab and original_token in raw:
      idx = vocab[lower_token]
      weights.append([original_token, coef[idx]*tfidf.idf_[idx]])
    else:
      pass
      # print(lower_token)
  filtered = []
  if gram_length > 1:
    filtered = remove_overlap(weights, lower_nopunc, raw, raw_nopunc)
  else:
    weights = remove_duplicate(weights)

  weights.sort(key = lambda x:abs(x[1]), reverse=True)
  return weights, filtered

def remove_duplicate(weights):
  added = set()
  result = []
  for weight in weights:
    if weight[0] not in added:
      added.add(weight[0])
      result.append(weight)
  return result

def remove_overlap(weights, lower_nopunc, raw, raw_nopunc):
  trigrams = set(everygrams(raw_nopunc.split(), 3, 3))
  tokens = [weight[0] for weight in weights]
  # print(trigrams)
  result = []
  added = set()
  for weight in weights:
    curr = weight[0]
    currList = curr.split(' ')
    res = [weight[0], weight[1]]
    for w in weights:
      t = w[0]
      tList = t.split(' ')
      if currList[1] == tList[0] and (currList[0], currList[1], tList[1]) in trigrams:
        if abs(weight[1]) > abs(w[1]):
          if weight[0] in raw:
            if w[0] not in added:
              res = [weight[0], weight[1]]
          else:
            if weight[0] not in added:
              res = [w[0], w[1]]
        else:
          if w[0] in raw:
            if weight[0] not in added:
              res = [w[0], w[1]]
          else:
            if w[0] not in added:
              res = [weight[0], weight[1]]
  
    if res[0] not in added and res[0] in raw:
      added.add(res[0])
      result.append(res)
      for r in result:
        if res[0].split(' ')[0] == r[0].split(' ')[1]:
          result.pop()
  return result

def get_diagram_view(cls, tfidf, weights):
  result = []
  vocab = tfidf.vocabulary_
  coef = cls.coef_[0]
  for weight in weights:
    token = weight[0].lower()
    idx = vocab[token]
    tmp = [token, coef[idx], tfidf.idf_[idx], weight[1]]
    result.append(tmp)
  return result

def get_feature_view(raw, weights):
  view = []
  added = set()
  for weight in weights:
    token, weight = weight[0], weight[1]
    indices = [s.start() for s in re.finditer(token, raw)] # find all starting char indices of token
    for idx in indices:
      if token not in added:
        added.add(token)
        view.append([token, idx, weight])
  return view

@app.route('/')
def run():
  import pickle
  f = open('view/data_m1.pkl', 'rb')
  data1 = pickle.load(f)
  f = open('view/data_m2.pkl', 'rb')
  data2 = pickle.load(f)
  return render_template('index.html', data1=data1, data2=data2)

@app.route('/upload',methods=['post'])
def upload():
  if request.method == 'POST':
    text = request.form.get('text')
    model_type = request.form.get('model_type')

  if model_type != 'model1' and model_type != 'model2':
    return render_template('index.html')

  components = {
    'model1': {
      'cls': cls_m1, 
      'tfidf': tfidf_m1
    },
    'model2': {
      'cls': cls_m2, 
      'tfidf': tfidf_m2
    },
  }
  raw = text
  lower_nopunc = text.lower().translate(str.maketrans('', '', string.punctuation))
  raw_nopunc = text.translate(str.maketrans('', '', string.punctuation))

  cls, tfidf = components[model_type]['cls'], components[model_type]['tfidf']
  res = {}
  if model_type == 'model1':
    res['class_name'] = ["negative", "positive"]
  else:
    res['class_name'] = ["republican", "democrat"]
  res['class_proba'] = get_class_proba(lower_nopunc, cls, tfidf)
  unigram_weights, _ = get_feature_weights(raw, lower_nopunc, raw_nopunc, cls, tfidf, 1)
  bigram_weights, filtered = get_feature_weights(raw, lower_nopunc, raw_nopunc, cls, tfidf, 2)
  # res['feature_weights'] = unigram_weights
  # res['feature_view'] = get_feature_view(raw, unigram_weights)
  # res['feature_weights'] = bigram_weights
  # res['feature_view'] = get_feature_view(raw, filtered)
  res['raw_str'] = raw
  res['feature_weights'] = unigram_weights + bigram_weights
  res['feature_view'] = {
    'unigram': get_feature_view(raw, unigram_weights),
    'bigram': get_feature_view(raw, filtered)
  }
  digram_view = get_diagram_view(cls, tfidf, unigram_weights+filtered)
  # digram_bigram = get_diagram_view(cls, tfidf, filtered)
  res['diagram_view'] = digram_view
  
  # [token, coeff, idf , weight], ...]
  # res['feature_view'] = get_feature_view(text)
  # [[word, char starting index, weight]]
  # res['class_proba'] = [0.9, 0.1]
  # res['feature_weights'] = [["TaxReform", -0.020770347480948175], ["FoxBusiness", -0.017348559557315277], ["Chairman", -0.016276587467941726], ["FoxNews", -0.014993825621698636], ["RepKevinBrady", -0.00954154749073416], ["https", 0.007872493594009546], ["highlights", -0.004874949990182821], ["benefits", -0.004290721079843876], ["the", 0.001588871746664454], ["and", -0.0015407369063189303]]
  # res['feature_view'] = [["#Tax Reform", 81, -0.020770347480948175], ["FoxBusiness", 41, -0.017348559557315277], ["Chairman", 0, -0.016276587467941726], ["FoxNews", 28, -0.014993825621698636], ["RepKevinBrady", 10, -0.00954154749073416], ["https", 111, 0.007872493594009546], ["highlights", 97, -0.004874949990182821], ["benefits", 68, -0.004290721079843876], ["the", 64, 0.001588871746664454], ["and", 36, -0.0015407369063189303]]
  # res['raw_str']="Chairman @RepKevinBrady on @FoxNews and @FoxBusiness to discuss the benefits of #TaxReform. Some highlights \u2b07\ufe0f https://t.co/urDcaXeWjF"
  # for v in res['feature_weights']:
  #   print(v)
  # print(res)
  return render_template("mylime.html", data = res, text = raw)

if __name__ == '__main__':
  app.run(host='localhost',port=8000,debug = True)
