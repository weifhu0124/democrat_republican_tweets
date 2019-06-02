from flask import Flask, request, jsonify
from flask import render_template
import random
import json

app = Flask(__name__)
app.debug = True

@app.route('/')
def run():
    import pickle
    f = open('fake_data.pkl','rb')
    data = pickle.load(f)
    return render_template('diagram.html', data = data)


if __name__ == '__main__':
  app.run(host='0.0.0.0',port=8888,debug = True)
