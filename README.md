# Website Deployment
https://political-tweets-sentiment.herokuapp.com/  
(It might take a while to load)

## How to use this website
When you open the link above, you will be directed to homepage
![Alt text](./images/homepage.png?raw=true "Homepage")

### Left Page
You can select your model and then input texts. Once you click submit, a new tab will be opened with the model prediction results as well as reasoning output from LIME
![Alt text](./images/select_model.png?raw=true "Model Selection")
![Alt text](./images/input_text.png?raw=true "Text Input")

### Right Page
One the right side of the homepage, it contains information of the training set, our model performance and the most representative words in the training set for each label. 
![Alt text](./images/home_side.png?raw=true "Model and Data")

### Reasoning Visualization
Once you click submit, you will be directed to a page that contains the predictions and reasoning of your input sentence. We will use the following example sentence: **BREAKING: House Passes the Tax Cuts and Jobs Act #taxreform** 
![Alt text](./images/predict_prob.png?raw=true "Prediction Probability")

We can see that the model predicts with 99.99% certainty that this tweet is from a republican, based on the useage of taxreform, Jobs Act and Tax Cuts. Moreover, you can visualize how representative those words are using the graph below.
![Alt text](./images/word_vis.png?raw=true "Word Visualization")

Finally, you can also toggle between unigram and bigram to see which words or two words contribute to the prediction.
![Alt text](./images/unibigram.png?raw=true "Unigram and Bigram")

# Disclaimer
All the results are coming from the training data and a linear model. There is no intent for political bias and we do not authorize nor take responsibilities in the malicious applications of this work.
