from operator import mod
from pyexpat import model
from statistics import mode
from numpy import vectorize
import sklearn
import nltk
import fitz
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from model import input_process
from textblob import TextBlob


def load_model_and_vectorizer():
    '''
    Load model and vectorizer.
    '''
    model = pickle.load(open('classifier.model','rb'))
    vectorizer = pickle.load(open('vectorizer.pickle','rb'))
    return model, vectorizer

if __name__ =='__main__':
    model,vectorizer = load_model_and_vectorizer()
    content = input('Enter Review you want to Classify:')
    content = input_process(content)

    content_vec = vectorizer.transform([content])
    pred = model.predict(content_vec)


    if pred[0] == 1:
        print('Good Review!')
    else:
        print('Bad Review!')

'''
TextBlob is used to determine the attitude or the emotion of the writer
subjectivity [0:1] is the amount of personal opinion and factual info contained in the text
higher value of subjectivity means more personal opinion rather than factual info.
polarity [-1:1] defines how positive a statement is. 1 is positive and -1 is negative
'''

print(content)
Subjectivity = TextBlob(content).sentiment.subjectivity
Polarity = TextBlob(content).sentiment.polarity
print(Subjectivity,Polarity)


if float(pred[0] == 1) and Polarity >= 0 or float(pred[0] == 0) and Polarity < 0:
  print('Correct Prediction')
else:
  print('Incorrect Prediction')