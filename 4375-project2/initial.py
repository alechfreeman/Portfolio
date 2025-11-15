
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
from itertools import product
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np


def load_dataset(clauses, examples, data_dir):
    #"C:\\Users\\alech\\PycharmProjects\\decisionTreesAndEnsemble\\project2_data\\all_data"
    train_file = f"train_c{clauses}_d{examples}.csv"
    valid_file = f"valid_c{clauses}_d{examples}.csv"
    test_file = f"test_c{clauses}_d{examples}.csv"

    train_df = pd.read_csv(os.path.join(data_dir, train_file), header=None)
    valid_df = pd.read_csv(os.path.join(data_dir, valid_file), header=None)
    test_df = pd.read_csv(os.path.join(data_dir, test_file), header=None)

    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1] # separate data and labels.
    X_valid, y_valid = valid_df.iloc[:, :-1], valid_df.iloc[:, -1]
    X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def evaluate_model(best_model, X_test, y_test):
    y = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y)
    f1 = f1_score(y_test, y)
    return accuracy, f1

def train_bagging_classifier(X_train, y_train, X_valid, y_valid, multiclass=False):

    parameter_grid = {
        "n_estimators": [10, 25],
        "max_samples": [1.0, 0.75],
        "max_features": [1.0]
    }

    if not multiclass:
        parameter_grid = {
            "n_estimators": [10, 25, 50],
            "max_samples": [0.5, 0.75, 1.0],
            "max_features": [0.5, 0.75, 1.0]
        }

    best_model = None
    best_params = None
    best_f1 = 0

    for n_estimators, max_samples, max_features in product(parameter_grid["n_estimators"],
                                                           parameter_grid["max_samples"],
                                                           parameter_grid["max_features"]):
        model = BaggingClassifier(estimator=DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None),n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, random_state=69, n_jobs=-1)

    # use gini because slightly faster, best because bagging already adds randomness compared to using random, use max_depth=None because
        #bagging reduces variance, overfitting and can capture more complex patterns

        model.fit(X_train, y_train) # Train using only training data
        y_pred = model.predict(X_valid) # Evaluate on the validation data by getting predicted value

        if multiclass:
            f1 = f1_score(y_valid, y_pred, average='weighted')  # Use F1-score as selection criteria
        else:
            f1 = f1_score(y_valid, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = {
                "n_estimators": n_estimators,
                "max_samples": max_samples,
                "max_features": max_features
            }
    return best_model, best_params



def train_random_forest(X_train, y_train, X_valid, y_valid, multiclass=False):

    parameter_grid = {
        "n_estimators": [10, 25],
        "max_depth": [10, 25],
        "max_features": [1.0],
    }

    if not multiclass:
        parameter_grid = {
            "n_estimators": [10, 25, 50],
            "max_depth": [None, 10, 25],
            "max_features": [0.5, 0.75, 1.0]
        }

    # "n_estimators": [10, 25, 50],
    # "max_depth": [None, 10, 25],
    # "max_features": [0.5, 0.75, 1.0],

    best_model = None
    best_params = None
    best_f1 = 0

    for n_estimators, max_depth, max_features in product(parameter_grid["n_estimators"],
                                                           parameter_grid["max_depth"],
                                                           parameter_grid["max_features"],
                                                         ):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=69, n_jobs=-1)

        model.fit(X_train, y_train) # Train using only training data
        y_pred = model.predict(X_valid) # Evaluate on the validation data by getting predicted value

        if multiclass:
            f1 = f1_score(y_valid, y_pred, average='weighted')  # Use F1-score as selection criteria
        else:
            f1 = f1_score(y_valid, y_pred)


        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "max_features": max_features,

            }

    return best_model, best_params

def train_boosting_classifier(X_train, y_train, X_valid, y_valid, multiclass=False):

    parameter_grid = {
        "n_estimators": [100],
        "learning_rate": [0.1],
        "max_depth": [3]
    }

    if not multiclass:
        parameter_grid = {
            "n_estimators": [100, 250],
            "learning_rate": [0.1, 0.5],
            "max_depth": [3, 5]
        }


    best_model = None
    best_params = None
    best_f1 = 0

    for n_estimators, learning_rate, max_depth, in product(parameter_grid["n_estimators"],
                                                           parameter_grid["learning_rate"],
                                                           parameter_grid["max_depth"],
                                                         ):
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=69)

        model.fit(X_train, y_train) # Train using only training data
        y_pred = model.predict(X_valid) # Evaluate on the validation data by getting predicted value

        if multiclass:
            f1 = f1_score(y_valid, y_pred, average='weighted')  # Use F1-score as selection criteria
        else:
            f1 = f1_score(y_valid, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,

            }

    return best_model, best_params


def train_decision_tree(X_train, y_train, X_valid, y_valid, multiclass=False):

    parameter_grid = {
        "criterion" : ["entropy"],
        "splitter": ["best"],
        "max_depth": [10, 25],
    }

    if not multiclass:
        parameter_grid = {
        "criterion": ["entropy", "gini"],
        "splitter": ["best", "random"],
        "max_depth": [None, 10, 25, 50],
                                  }


    best_model = None
    best_params = None
    best_f1 = 0
    for criterion, splitter, max_depth in product(parameter_grid["criterion"],
                                                  parameter_grid["splitter"],
                                                  parameter_grid["max_depth"]):
        model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, random_state=69)
        model.fit(X_train, y_train)  # Train only on training set
        y_pred = model.predict(X_valid)  # Evaluate on validation set

        if multiclass:
            f1 = f1_score(y_valid, y_pred, average='weighted')  # Use F1-score as selection criteria
        else:
            f1 = f1_score(y_valid, y_pred)


        if f1 > best_f1:  # If better, update the best model
            best_f1 = f1
            best_model = model
            best_params = {"criterion": criterion, "splitter": splitter, "max_depth": max_depth}

    return best_model, best_params

def evaluate_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def mnist_train_boosting_classifier(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    parameter_grid = {
        "max_iter": [100],
        "learning_rate": [0.1],
        "max_depth": [3,5]
    }

    #speed up
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(n_iter_no_change=5, random_state=69)

    grid_search = GridSearchCV(model, parameter_grid, scoring="f1_weighted", cv=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params


def mnist():
    #Load MNIST dataset
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X / 255.0


    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]


    #Classifier was trained on numpy array that does not have feature names, unlike pd
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    classifiers = {
        "Decision Tree": train_decision_tree,
        "Bagging": train_bagging_classifier,
        "Random Forest": train_random_forest,
        "Boosting": train_boosting_classifier # not really
    }

    for name, train_func in classifiers.items():
        print(f"\n{name.upper()}")

        # Train using training and validation split

        if name == 'Boosting': # Have to speed up by using cross validation technique
            best_model, best_params = mnist_train_boosting_classifier(X_train, y_train) #cross validation

        else:
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=.30, random_state=45)
            best_model, best_params = train_func(X_train, y_train, X_valid, y_valid, multiclass=True)

            # Retrain on full training + validation data
            X_train_valid = np.concatenate([X_train, X_valid])
            y_train_valid = np.concatenate([y_train, y_valid])
            best_model.fit(X_train_valid, y_train_valid)

        # Evaluate on test set
        accuracy = evaluate_accuracy(best_model, X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")



def main():
    clause_counts = [300, 500, 1000, 1500, 1800]
    example_sizes = [100, 1000, 5000]

    path = input("Enter the pathname for the files using \\\\ instead of :")


    classifiers = {
            "Decision Tree": train_decision_tree,
            "Bagging": train_bagging_classifier,
            "Random Forest": train_random_forest,
            "Boosting": train_boosting_classifier
    }
    for name, train_func in classifiers.items():
         for c in clause_counts:
            for d in example_sizes:
                X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(c, d, path)
                best_model, best_params = train_func(X_train, y_train, X_valid, y_valid)
                X_train_valid = pd.concat([X_train, X_valid])
                y_train_valid = pd.concat([y_train, y_valid])
                best_model.fit(X_train_valid, y_train_valid) # Train model again with training and validation data combined

                accuracy, f1 = evaluate_model(best_model, X_test, y_test)

                print(f"\n{name.upper()}")
                print(f"Clauses: {c}, Examples: {d}")
                print(f"Best Parameters: {best_params}")
                print(f"Test Accuracy {accuracy: .4f}, Test F1-Score: {f1:.4f}")

# CODE FOR INDIVIDUAL TESTING:

    # clause_counts = [300, 500, 1000, 1500, 1800]
    # example_sizes = [100, 1000, 5000]
    #
    # for c in clause_counts:
    #     for d in example_sizes:
    #         #DECISION TREE
    #         X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(c, d)
    #         best_model, best_params = train_decision_tree(X_train, y_train, X_valid, y_valid)
    #
    #         X_train_valid = pd.concat([X_train, X_valid])
    #         y_train_valid = pd.concat([y_train, y_valid])
    #         best_model.fit(X_train_valid, y_train_valid) # Train model again with training and validation data combined
    #
    #         accuracy, f1 = evaluate_model(best_model, X_test, y_test)
    #
    #         print("DECISION TREE")
    #         print(f"Clauses: {c}, Examples: {d}")
    #         print(f"Best Parameters: {best_params}")
    #         print(f"Test Accuracy {accuracy: .4f}, Test F1-Score: {f1:.4f}")
    #
    #        BAGGING
            # best_model, best_params = train_bagging_classifier(X_train, y_train, X_valid, y_valid)
            #
            # X_train_valid = pd.concat([X_train, X_valid])
            # y_train_valid = pd.concat([y_train, y_valid])
            # best_model.fit(X_train_valid, y_train_valid) # Train model again with training and validation data combined
            #
            # accuracy, f1 = evaluate_model(best_model, X_test, y_test)
            #
            # print("BAGGING")
            # print(f"Clauses: {c}, Examples: {d}")
            # print(f"Best Parameters: {best_params}")
            # print(f"Test Accuracy {accuracy: .4f}, Test F1-Score: {f1:.4f}")


            #RANDOM FORREST
            #
            # best_model, best_params = train_random_forest(X_train, y_train, X_valid, y_valid)
            #
            # X_train_valid = pd.concat([X_train, X_valid])
            # y_train_valid = pd.concat([y_train, y_valid])
            # best_model.fit(X_train_valid, y_train_valid) # Train model again with training and validation data combined
            #
            # accuracy, f1 = evaluate_model(best_model, X_test, y_test)
            #
            # print("RANDOM FOREST")
            # print(f"Clauses: {c}, Examples: {d}")
            # print(f"Best Parameters: {best_params}")
            # print(f"Test Accuracy {accuracy: .4f}, Test F1-Score: {f1:.4f}")
            #
            #BOOSTING
            # best_model, best_params = train_boosting_classifier(X_train, y_train, X_valid, y_valid)
            #
            # X_train_valid = pd.concat([X_train, X_valid])
            # y_train_valid = pd.concat([y_train, y_valid])
            # best_model.fit(X_train_valid, y_train_valid) # Train model again with training and validation data combined
            #
            # accuracy, f1 = evaluate_model(best_model, X_test, y_test)
            #
            # print("BOOSTING")
            # print(f"Clauses: {c}, Examples: {d}")
            # print(f"Best Parameters: {best_params}")
            # print(f"Test Accuracy {accuracy: .4f}, Test F1-Score: {f1:.4f}")
            #
main()
mnist()



