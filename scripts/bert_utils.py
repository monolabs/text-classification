import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import transformers
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt

# specify GPU
device = torch.device("cuda")

def get_evaluation_df(y, y_hat, label):
    
    accuracy = accuracy_score(y, y_hat)
    precision = precision_score(y, y_hat)
    recall = recall_score(y, y_hat)
    f1 = f1_score(y, y_hat)
    roc_auc = roc_auc_score(y, y_hat)
    
    cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metrics = [[accuracy, precision, recall, f1, roc_auc]]
    df = pd.DataFrame(metrics, index=[label], columns=cols)
    
    return df