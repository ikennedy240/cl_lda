"""
This module includes functions for supporting ML tasks and model comparision
"""

import numpy as np
# Produces a score report with various output
def score_report(model, X, y, print_it=True, mode='classifier'):
    #calculate predictions
    from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
    try:
        probs = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, probs)
    except:
        auc = None
    predictions = model.predict(X)
    confusion = confusion_matrix(y, predictions)
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    #if print is true, print the results
    if print_it:
        print("Accuracy: ", accuracy)
        print("F1 Score: ", f1)
        if auc:
            print("ROC AUC: ", auc)
        print("\t    P_no P_yes\nActual No:", confusion[0], "\nActual Yes", confusion[1])
    #otherwise return them as a dictionary
    else:
        if auc:
            return({"Accuracy": accuracy, "F1" : f1, "ROC AUC" : auc, "Confusion Matrix" : confusion })
        else:
            return({"Accuracy": accuracy, "F1" : f1, "Confusion Matrix" : confusion })



    # Returns the most important features
def top_features(model, vect, n=20, print_it = True):
    feature_names = np.array(vect.get_feature_names())
    try:
        sorted_import_index = model.feature_importances_.argsort()
        if print_it:
            print('Smallest Import:\n{}\n'.format(feature_names[sorted_import_index[:n]]))
            print('Largest Import:\n{}\n'.format(feature_names[sorted_import_index[-n:]]))
    except:
        # Sort the coefficients from the model
        sorted_coef_index = model.coef_[0].argsort()
        opp_coef_index = (-model.coef_[0]).argsort()
        if print_it:
            # Find the 10 smallest and 10 largest coefficients
            # The 10 largest coefficients are being indexed using [:-11:-1]
            # so the list returned is in order of largest to smallest
            print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:n]]))
            print('Largest Coefs: \n{}'.format(feature_names[opp_coef_index[:n]]))


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/cl_lda4_15.csv')
    df.columns
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df[[str(x) for x in list(range(50))]], df['high_white'], random_state=0)
    """Random Forest Classifier"""
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators = 10, n_jobs = 3).fit(X_train_vectorized,y_train)
    if rf.jon:
        print("exists")
    else:
        print('nope')
    rf.model_coef
    type(rf)
    predictions = rf.predict(X_test)
    confusion = confusion_matrix(y_test, predictions)
    print("\t    P_no P_yes\nActual No:", confusion[0], "\nActual Yes", confusion[1])
    predictions = rf.predict_proba(X_test)[:,1]
    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_test, predictions)
    sorted_import_index = rf.feature_importances_.argsort()
    rf.score(X_test, y_test)

    from sklearn.feature_extraction.text import CountVectorizer
    # Fit the CountVectorizer to the training data
    with open("resources/stopwords.txt", 'r') as f:
        stop_words = f.readlines()

    X_train, X_test, y_train, y_test = train_test_split(df['body_text'], df['high_white'], random_state=0)

    vect = CountVectorizer(stop_words=stop_words).fit(X_train)
    len(vect.get_feature_names())
    X_train_vectorized =  vect.transform(X_train)
    """LogisticRegression"""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty = 'l1',C=.1).fit(X_train_vectorized, y_train)
    score_report(model, vect.transform(X_test),y_test)
    sorted_import_index = model.feature_importances_.argsort()
    top_features(rf, vect)
