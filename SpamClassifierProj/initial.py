
import csv
import os
import numpy as np
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, framenet15
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import cProfile

def preprocess_email(text):
    """Clean and preprocess email text."""
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize words
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)  # Return processed text as a string

def load_and_preprocess_emails(directory):
    emails, labels = [], []
    for label, folder in enumerate(["ham", "spam"]):  # 0 for ham, 1 for spam
        folder_path = os.path.join(directory, folder)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                raw_text = file.read()
                processed_text = preprocess_email(raw_text)
                emails.append(processed_text)
                labels.append(label)
    return emails, labels
def train_multinomial_nb(X_train, y_train, vocabulary):
        num_docs = len(y_train) # Total emails

        num_words = len(vocabulary) # vocabulary size = # features

        num_spam = sum(y_train)
        num_ham = num_docs - num_spam

        prior_spam = num_spam / num_docs
        prior_ham = num_ham / num_docs

        word_counts_spam = np.ones(num_words) # array of size # features where all counts start with 1
        # T_ct = T_spam_for feature t

        word_counts_ham = np.ones(num_words)  # both for Laplace smoothing


        total_words_spam = num_words # Start with vocab size due to Laplace smoothing
        total_words_ham = num_words

        for j in range(num_docs):
            if y_train[j] ==  1 :# Spam email?
                word_counts_spam += X_train[j] # Add all the counts of the features for this email to the total word counts (+1)
                total_words_spam += sum(X_train[j]) #Add total words in the email to the total spam word count
            else: # HAM
                word_counts_ham += X_train[j]
                total_words_ham += sum(X_train[j])

        cond_prob_spam = np.log(word_counts_spam / total_words_spam)
        cond_prob_ham = np.log(word_counts_ham / total_words_ham)

        return prior_spam, prior_ham, cond_prob_spam, cond_prob_ham

def pred_multinomial_nb(X_test, prior_spam, prior_ham, cond_prob_spam, cond_prob_ham):

    predictions = []
    for email in X_test:
        log_prob_spam = math.log(prior_spam) + sum(email * cond_prob_spam)
        log_prob_ham = math.log(prior_ham) + sum(email * cond_prob_ham)


        #compare logP(email|spam) + logP(spam) and logP(email | ham) + logP(ham)
        if log_prob_spam > log_prob_ham:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions

def train_discrete_nb(X_train, y_train, vocabulary):

        num_docs = len(y_train)  # Total emails
        num_words = len(vocabulary)  # vocabulary size = # features
        priors = [0, 0]
        conditional_probability = np.zeros((2, num_words))

        y_train = np.array(y_train)

        for c in range(2):
            emails_with_class = X_train[y_train == c] # filter X_train to include emails where the label is equal to 1
            N_emails_with_class = len(emails_with_class)

            priors[c] = N_emails_with_class / num_docs
            # P(c) = P(ham) = P(spam)




            #Count how many documents have each word
            Word_counts_for_class = np.sum(emails_with_class, axis=0) #  is # ((Xj = 1, Y = c))
            #summing column wise

            conditional_probability = ( Word_counts_for_class + 1) / (N_emails_with_class + 2)
            # Get 2-D array of conditional probabilities
            # conditional_prob[1] = [ #(X1 = 1, Y = 1], #(X2 = 1, Y = 1), . . .]
            # conditional_prob[0] = [ #(X1 = 1, Y = 0], #(X2 = 1, Y = 0), . . .]


        return priors, conditional_probability

def predict_discrete_nb(X_test, prior, cond_prob):
    num_docs = X_test.shape[0]
    predictions = []

    for i in range(num_docs):
        email = X_test[i]
        log_probs = [0, 0]

        for c in range(2):
            log_probs[c] = np.log(prior[c])

            log_probs[c] += sum(email * np.log(cond_prob[c]) + (1 - email) * np.log(1 - cond_prob[c]))
                #sum(
        if log_probs[0] > log_probs[1]:
            predictions.append(0)
        else:
            predictions.append(1)

    return predictions


def pred_multinomial_nb(X_test, prior_spam, prior_ham, cond_prob_spam, cond_prob_ham):
    predictions = []
    for email in X_test:
        log_prob_spam = math.log(prior_spam) + sum(email * cond_prob_spam)
        log_prob_ham = math.log(prior_ham) + sum(email * cond_prob_ham)

        # compare logP(email|spam) + logP(spam) and logP(email | ham) + logP(ham)
        if log_prob_spam > log_prob_ham:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions

def sigmoid(x):
    x = np.clip(x, -500, 500)  # clip x to a range that won't cause overflow
    return 1 / (1 + np.exp(-x))

def train_logistic_regression(X, y, learning_rate=.01, max_iterations=5, lambda_reg=.1):
    d, n = X.shape # d = number of samples, n = number of features
    X = np.hstack((np.ones((d, 1)), X))
    w = np.random.randn(n + 1)
    for t in range(0, max_iterations):
        y_pred = np.zeros(d)
        z =  np.zeros(d) # stores w0 + sum(n, j -1) {(w_j * x_j^i)}
        for i in range(0,d):
            z[i] = 0
            for j in range(n):
                z[i] = z[i] + w[j] * X[i][j] # w^Tx^(i)
            y_pred[i] = sigmoid(z[i]) # P(Y=1 | x^(i); w) = sigmoid(w^tx(i))
        #g = np.zeros(n+1) # gradient vector
        g = X.T.dot(y-y_pred) - lambda_reg * np.r_[0, w[1:]]
        #for j in range(n+1):
          #  g[j] = 0
           # for i in range(d):
               # g[j] = g[j] + X[i][j] * (y[i] - y_pred[i]) #jth component of gradient vector

       # g[1:] -= lambda_reg * w[1:]
        w += learning_rate * g
    return w

def predict(X, w):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
    y_pred = sigmoid(X.dot(w))  # Compute probability
    return (y_pred >= 0.5).astype(int)

def findBestLamda(X_train, y_train, X_val, y_val):
    best_lambda = None
    best_f1 = 0
    for lambda_reg in [0.001, 0.01, 0.1, 1]:
        w = train_logistic_regression(X_train, y_train, lambda_reg=lambda_reg)
        y_val_pred = predict(X_val, w)
        f1 = calcf1_score(y_val, y_val_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_lambda = lambda_reg
    return best_lambda

def printstats(test_labels, predictions):
    print('accuracy = ' + str(calc_accuracy(test_labels, predictions)))
    print('precision = ' + str(calcprecision(test_labels, predictions)))
    print('recall = ' + str(calcrecall(test_labels, predictions)))
    print('f1_score = ' + str(calcf1_score(test_labels, predictions)))

def main():
    datasets = ["enron1", "enron2", "enron4"]  # Add more as needed
    data_root = input("What is the full path, using \\\\ to specify path, of project1_datasets? ")
    file_location_root = input("What is the absolute path of where the csv files should be located? ")
    #max_features = int(input("Enter max number of features, w: "))
    #data_root = "C:\\project1_datasets"

    train_emails = ""
    test_emails = ""
    for dataset in datasets:
        if dataset == "enron2":
            train_dir = os.path.join(data_root, f"{dataset}_train", "train")
            test_dir = os.path.join(data_root, f"{dataset}_test", "test")
        else:
            train_dir = os.path.join(data_root, f"{dataset}_train", f"{dataset}", "train")
            test_dir = os.path.join(data_root, f"{dataset}_test", f"{dataset}", "test")


        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            print(f"Skipping {dataset}, directories not found")
            continue

        train_emails, train_labels = load_and_preprocess_emails(train_dir)
        test_emails, test_labels = load_and_preprocess_emails(test_dir)

        print(f"Processed {len(train_emails)} training and {len(test_emails)} test emails from {dataset}")



        all_words = [word for email in train_emails for word in email.split()]
        word_features = list(FreqDist(all_words).keys())[:3000] # get features

        vectorizer_bow = CountVectorizer(vocabulary=word_features)
        X_train_bow = vectorizer_bow.fit_transform(train_emails)
        X_test_bow = vectorizer_bow.transform(test_emails)

        vectorizer_ber = CountVectorizer(vocabulary=word_features, binary=True)
        X_train_ber = vectorizer_ber.fit_transform(train_emails)
        X_test_ber = vectorizer_ber.transform(test_emails)

        X_train_ber_dense = X_train_ber.toarray()
        X_train_bow_dense = X_train_bow.toarray()
        X_test_bow_dense = X_test_bow.toarray()
        X_test_ber_dense = X_test_ber.toarray()

        # C:\Users\alech\PycharmProjects\NBandLR\
        with open(os.path.join(file_location_root, f'{dataset}_bow_train.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list(vectorizer_bow.get_feature_names_out()) + ['label']) # Write header
            for i in range(len(train_emails)):
                writer.writerow(list(X_train_bow_dense[i]) + [train_labels[i]])

        #C:\Users\alech\PycharmProjects\NBandLR\
        with open(os.path.join(file_location_root, f'{dataset}_bow_test.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list(vectorizer_bow.get_feature_names_out()) + ['label']) # Write header
            for i in range(len(test_emails)):
                writer.writerow(list(X_test_bow_dense[i]) + [test_labels[i]])

        # C:\Users\alech\PycharmProjects\NBandLR\
        with open(os.path.join(file_location_root, f'{dataset}_ber_train.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list(vectorizer_ber.get_feature_names_out())+ ['label'])  # Write header
            for i in range(len(train_emails)):
                writer.writerow(list(X_train_ber_dense[i]) + [train_labels[i]])

        #C:\Users\alech\PycharmProjects\NBandLR\
        with open(os.path.join(file_location_root, f'{dataset}_ber_test.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list(vectorizer_ber.get_feature_names_out()) + ['label']) # Write header
            for i in range(len(test_emails)):
                writer.writerow(list(X_test_ber_dense[i]) + [test_labels[i]])

        prior_spam, prior_ham, cond_prob_spam, cond_prob_ham = train_multinomial_nb(X_train_bow_dense, train_labels, word_features)

        predictions = pred_multinomial_nb(X_test_bow_dense, prior_spam, prior_ham, cond_prob_spam, cond_prob_ham)

        print("Multinomial NB")
        printstats(test_labels, predictions)
        priors, cond_probs  = train_discrete_nb(X_train_ber_dense, train_labels, word_features)

        predictions = predict_discrete_nb(X_test_ber_dense, priors, cond_probs)

        print("Discrete NB")
        printstats(test_labels, predictions)

        X_train, X_valid_bow, y_train , y_val = train_test_split(X_train_ber_dense, train_labels, test_size=.3, random_state=42)
        best_lambda_bow = findBestLamda(X_train, y_train, X_valid_bow, y_val)
        final_w_bow = train_logistic_regression(X_train_bow_dense, train_labels, lambda_reg=best_lambda_bow)
        y_test_pred_bow = predict(X_test_bow_dense, final_w_bow)
        print("Logistic Regression Performance for Bag of Words:")
        printstats(test_labels, y_test_pred_bow)
        X_train, X_valid_ber, y_train, y_val = train_test_split(X_train_ber_dense, train_labels, test_size=0.3, random_state=42)
        best_lambda_ber= findBestLamda(X_train, y_train, X_valid_ber, y_val)
        final_w_bow = train_logistic_regression(X_train_bow_dense, train_labels, lambda_reg=best_lambda_ber)
        y_test_pred_ber = predict(X_test_ber_dense, final_w_bow)
        print("Logistic Regression Performance for Bernoulli:")
        printstats(test_labels, y_test_pred_ber)

def calc_accuracy(test_labels, predictions):
    correct = 0
    for i in range(len(predictions)):
        if test_labels[i] ==  predictions[i]:
            correct+=1

    return correct / (len(predictions))

def calcprecision(test_labels, predictions):
    true_pos = 0
    false_pos = 0
    for i in range(len(predictions)):
        if test_labels[i] == 1 and predictions[i] == 1:
            true_pos+=1
        elif test_labels[i] == 0 and predictions[i] == 1:
            false_pos+=1
    if true_pos+false_pos==0:
        return 0.0
    return true_pos / (true_pos+false_pos)

def calcrecall(test_labels, predictions):
    true_pos = 0
    false_neg = 0
    for i in range(len(predictions)):
        if test_labels[i] == 1 and predictions[i] == 1:
            true_pos+=1
        elif test_labels[i] == 1 and predictions[i] == 0:
            false_neg+=1
    if true_pos + false_neg == 0:
        return 0.0
    return true_pos / (true_pos+false_neg)

def calcf1_score(test_labels, predictions):
    precision = calcprecision(test_labels, predictions)
    recall = calcrecall(test_labels, predictions)
    if precision + recall == 0:
        return 0.0
    return 2 *  ( (precision * recall) / (precision + recall) )


main()









