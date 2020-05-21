import os
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk


def create_dataset():
    path = 'emails/'
    files = os.listdir(path)
    emails = [path + email for email in files]
    labels = []
    mail_text = []
    for email in emails:
        if "ham" in email:
            labels.append(1)
        elif "spam" in email:
            labels.append(0)
        with open(email, 'r', encoding="ascii", errors="ignore") as f:
            data = f.read()
        mail_text.append(data)
        f.close()
    return emails, mail_text, labels


if __name__ == '__main__':
    paths, mail_text, labels = create_dataset()
    d = {"Path": paths, "Text": mail_text, "Label": labels}
    df = pd.DataFrame(d)
    print(df)
    """
        Do some Visualzation
    """