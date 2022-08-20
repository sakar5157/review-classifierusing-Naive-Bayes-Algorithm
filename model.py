from pyexpat import model
import pandas as pd
import nltk
from nltk.corpus import stopwords
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import string

nltk.download('stopwords')
vectorizer = CountVectorizer()

def pre_process_df():
    '''
    Import CSV file and store it in a DataFrame to use.
    '''
    f_df = pd.read_csv('Total_Reviews.csv')
    df = pd.DataFrame(columns=['Text','Label'])

    df['Text'] = f_df['Text']
    df['Label'] = f_df['Label']
    return df

def input_process(text):
    '''
    Remove Stop Words.
    '''
    translator = str.maketrans('','',string.punctuation)
    nopunc = text.translate(translator)
    words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(words)

def remove_stop_words(input):
    final_input = []
    for line in input:
        line = input_process(line)
        final_input.append(line)
    return final_input

def train_model(df):
    '''
    Define a model and train its transfromed form from the vectorizer.
    '''
    input = df['Text']
    output = df['Label']
    input = remove_stop_words(input)
    df['Text'] = input
    input = vectorizer.fit_transform(input)
    nb = MultinomialNB()
    nb.fit(input,output)
    return nb



if __name__ == '__main__':
    df = pre_process_df()
    model = train_model(df)
    pickle.dump(model,open('classifier.model','wb'))
    pickle.dump(vectorizer,open('vectorizer.pickle','wb'))