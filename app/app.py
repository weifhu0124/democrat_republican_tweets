from flask import Flask, request, jsonify
from flask import render_template
import random, sys
import json, pickle, re
from nltk import everygrams

sys.path.append("..")
app = Flask(__name__)
app.debug = True


'''
preload all necessary components
model1: PA2 model
model2: Twitter Democrat/Republican model
'''
cls_m1 = None
tfidf_m1 = None
with open('model/cls.pkl', 'rb') as fp:
  cls_m2 = pickle.load(fp)
with open('model/sentiment_tfidf.pkl', 'rb') as fp:
  tfidf_m2 = pickle.load(fp)

def get_class_proba(text, cls, tfidf):
    X = tfidf.transform([text])
    return cls.predict_proba(X).tolist()[0]

def get_feature_weights(text, cls, tfidf):
  X = tfidf.transform([text])
  coef = cls.coef_[0]
  grams = list(everygrams(text.split(), 1, 3))
  vocab = tfidf.vocabulary_
  # f = open('vocab.txt','w')
  # f.write(str(vocab))
  weights = []
  for gram in grams:
    token = " ".join(gram)
    if token in vocab:
      idx = vocab[token]
      weights.append([token, coef[idx]*tfidf.idf_[idx]])
    else:
      print(token)
  weights.sort(key = lambda x:abs(x[1]), reverse=True)
  return weights

def get_feature_view(text, weights):
  view = []
  for weight in weights:
    token, weight = weight[0], weight[1]
    indices = [s.start() for s in re.finditer(token, text)] # find all starting char indices of token
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
  text = text.lower()
  cls, tfidf = components[model_type]['cls'], components[model_type]['tfidf']
  res = {}
  res['class_name'] = ["republican", "democrat"]
  res['class_proba'] = get_class_proba(text, cls, tfidf)
  res['feature_weights'] = get_feature_weights(text, cls, tfidf)
  res['feature_view'] = get_feature_view(text, res['feature_weights'])
  res['raw_str'] = text

  # res['feature_view'] = get_feature_view(text)
  # [[word, char starting index, weight]]
  # res['class_proba'] = [0.9, 0.1]
  # res['feature_weights'] = [["TaxReform", -0.020770347480948175], ["FoxBusiness", -0.017348559557315277], ["Chairman", -0.016276587467941726], ["FoxNews", -0.014993825621698636], ["RepKevinBrady", -0.00954154749073416], ["https", 0.007872493594009546], ["highlights", -0.004874949990182821], ["benefits", -0.004290721079843876], ["the", 0.001588871746664454], ["and", -0.0015407369063189303]]
  # res['feature_view'] = [["#Tax Reform", 81, -0.020770347480948175], ["FoxBusiness", 41, -0.017348559557315277], ["Chairman", 0, -0.016276587467941726], ["FoxNews", 28, -0.014993825621698636], ["RepKevinBrady", 10, -0.00954154749073416], ["https", 111, 0.007872493594009546], ["highlights", 97, -0.004874949990182821], ["benefits", 68, -0.004290721079843876], ["the", 64, 0.001588871746664454], ["and", 36, -0.0015407369063189303]]
  # res['raw_str']="Chairman @RepKevinBrady on @FoxNews and @FoxBusiness to discuss the benefits of #TaxReform. Some highlights \u2b07\ufe0f https://t.co/urDcaXeWjF"
  print(res)
  return render_template("result.html", data = res, text = text)



if __name__ == '__main__':
  app.run(host='localhost',port=8000,debug = True)
