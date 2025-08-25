# hw3_challenge.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc)
from sklearn.metrics import precision_recall_curve






import hw3_main
from helper import *

def generate_feature_vector_challenge(df):
    return hw3_main.generate_feature_vector(df)

def impute_missing_values_challenge(X):
    return hw3_main.impute_missing_values(X)

def normalize_feature_matrix_challenge(X):
    return hw3_main.normalize_feature_matrix(X)


def run_challenge(X_challenge, y_challenge, X_heldout, feature_names):
    """
    Part 3: Train a classifier on (X_challenge, y_challenge) with labels in {-1,1},
    do threshold optimization for F1, produce predictions in {-1,1} for the heldout set,
    then generate 'challenge.csv'. Also print the confusion matrix and metrics on the training set.
    """

    print("================= Part 3 ===================")
    print("Part 3: Challenge (keeping labels in {-1,1})")

    param_distributions = {
        'C': np.logspace(-3, 3, 10),
        'penalty': ['l1', 'l2'],
    }
    clf_base = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced')
    random_search = RandomizedSearchCV(
        estimator=clf_base,
        param_distributions=param_distributions,
        scoring='roc_auc',  
        cv=5,
        n_iter=10,
        random_state=42
    )
    random_search.fit(X_challenge, y_challenge)
    best_params = random_search.best_params_
    print(f"Best hyperparams: {best_params}")

    clf = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        **best_params
    )
    clf.fit(X_challenge, y_challenge)

    
    y_scores_train = clf.predict_proba(X_challenge)[:,1]

    y_challenge_bin = np.where(y_challenge == 1, 1, 0)
    precisions, recalls, thresholds = precision_recall_curve(y_challenge_bin, y_scores_train)

    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_thresh_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else 0.5
    print(f"Optimal threshold for F1: {best_thresh:.4f}  (F1={f1_scores[best_thresh_idx]:.4f})")

    y_pred_train = np.where(y_scores_train >= best_thresh, 1, -1)

    cm = confusion_matrix(y_challenge, y_pred_train, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if (tp+fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp+fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall)>0 else 0.0
    auroc = roc_auc_score(y_challenge_bin, y_scores_train)  

    print("\nTraining Confusion Matrix (labels=[-1,1]):")
    print(cm)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}  (Sensitivity)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"AUROC:       {auroc:.4f}")

    y_scores_heldout = clf.predict_proba(X_heldout)[:,1]
    y_pred_heldout   = np.where(y_scores_heldout >= best_thresh, 1, -1)

    # 7) Output the challenge file
    # The assignment presumably wants "label" in {-1,1} or some mention of 0. 
    # We'll do -1/1 plus the raw probability for "risk_score".
    print("\nSaving challenge output...")
    make_challenge_submission(y_pred_heldout, y_scores_heldout)

    # 8) Check the file format
    test_challenge_output()


if __name__ == '__main__':
    X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()

    run_challenge(X_challenge, y_challenge, X_heldout, feature_names)