from flask import Flask, request, jsonify
from flask import render_template
import random, sys
import json, pickle, re, string
from nltk import everygrams

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

def get_feature_weights(lower_nopunc, raw_nopunc, cls, tfidf, gram_length):
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
    if lower_token in vocab:
      idx = vocab[lower_token]
      weights.append([original_token, coef[idx]*tfidf.idf_[idx]])
    else:
      pass
      # print(lower_token)
  weights.sort(key = lambda x:abs(x[1]), reverse=True)
  return weights

def get_feature_weights_total(lower_nopunc, raw_nopunc, cls, tfidf):
  pass

def get_feature_view(raw, weights):
  view = []
  for weight in weights:
    token, weight = weight[0], weight[1]
    indices = [s.start() for s in re.finditer(token, raw)] # find all starting char indices of token
    for idx in indices:
      view.append([token, idx, weight])
  return view

@app.route('/')
def run():
  return render_template('index.html')

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
  unigram_weights = get_feature_weights(lower_nopunc, raw_nopunc, cls, tfidf, 1)
  bigram_weights = get_feature_weights(lower_nopunc, raw_nopunc, cls, tfidf, 2)
  res['feature_weights'] = unigram_weights
#  res['feature_weights'] = bigram_weights
  res['feature_view'] = get_feature_view(raw, unigram_weights)
#  res['feature_view'] = get_feature_view(raw, bigram_weights)
  res['raw_str'] = raw
#   res['feature_weights'] = {
#       'unigram': unigram_weights,
#       'bigram': bigram_weights
#   }

  # res['feature_view'] = get_feature_view(text)
  # [[word, char starting index, weight]]
  # res['class_proba'] = [0.9, 0.1]
  # res['feature_weights'] = [["TaxReform", -0.020770347480948175], ["FoxBusiness", -0.017348559557315277], ["Chairman", -0.016276587467941726], ["FoxNews", -0.014993825621698636], ["RepKevinBrady", -0.00954154749073416], ["https", 0.007872493594009546], ["highlights", -0.004874949990182821], ["benefits", -0.004290721079843876], ["the", 0.001588871746664454], ["and", -0.0015407369063189303]]
  # res['feature_view'] = [["#Tax Reform", 81, -0.020770347480948175], ["FoxBusiness", 41, -0.017348559557315277], ["Chairman", 0, -0.016276587467941726], ["FoxNews", 28, -0.014993825621698636], ["RepKevinBrady", 10, -0.00954154749073416], ["https", 111, 0.007872493594009546], ["highlights", 97, -0.004874949990182821], ["benefits", 68, -0.004290721079843876], ["the", 64, 0.001588871746664454], ["and", 36, -0.0015407369063189303]]
  # res['raw_str']="Chairman @RepKevinBrady on @FoxNews and @FoxBusiness to discuss the benefits of #TaxReform. Some highlights \u2b07\ufe0f https://t.co/urDcaXeWjF"
  print(res)
  for v in res['feature_view']:
    print(v)
  return render_template("mylime.html", data = res, text = raw)

if __name__ == '__main__':
  app.run(host='localhost',port=8000,debug = True)
