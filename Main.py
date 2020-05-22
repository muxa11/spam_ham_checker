import os
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline

def create_dataset():
    path = 'emails/'
    files = os.listdir(path)
    emails = [path + email for email in files]
    labels = []
    mail_text = []
    words = []
    for email in emails:
        if "ham" in email:
            labels.append(1)
        elif "spam" in email:
            labels.append(0)
        with open(email, 'r', encoding="ascii", errors="ignore") as f:
            data = f.read()
        mail_text.append(data)
        f.close()
    print(len(labels))
    return emails, mail_text, labels


def process_words(mail):
    words = [char for char in mail if char not in string.punctuation]
    words = ''.join(words)
    words = [c for c in words.split() if c not in stopwords.words('english')]
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words]
    # Get a List of all words, punctuations and numbers in all mails
    # print(words)
    # print('\n')
    # remove punctuations from the words
    # Stemming Words
    # Remove Duplicate words
    # words = list(dict.fromkeys(words))
    # remove common words (stopwords) which dont help in distinguishing
    # print(words)
    return words


if __name__ == '__main__':
    paths, mail_text, labels = create_dataset()
    d = {"Path": paths, "Text": mail_text, "Label": labels}
    mails_with_labels = pd.DataFrame(d)
    # print(df)
    """
        Do some Visualzation
    """
    mails_with_labels['length'] = mails_with_labels['Text'].apply(len)
    bar_plt = sns.barplot(x = 'Label', y = 'length', data=mails_with_labels)
    plt.show()
    sns.countplot(x='Label', data=mails_with_labels)
    plt.show()
   # mails_with_labels.hist(column='length', by='label', bins=50, figsize=(12, 4))
    # words = process_words(mail_text)
    # dictionary = Counter(words)
    # print(dictionary)
"""
    bag_of_words_transformer = CountVectorizer(analyzer=process_words)
    bag_of_words_transformer.fit(mails_with_labels['Text'])
    mails_bag_of_words = bag_of_words_transformer.transform(mails_with_labels['Text'])
    print(mails_bag_of_words.shape)
    print(mails_bag_of_words.nnz)

    tfidf_transformer = TfidfTransformer().fit(mails_bag_of_words)
    mails_tfidf = tfidf_transformer.transform(mails_bag_of_words)
    X = mails_tfidf
    y = mails_with_labels['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))"""
