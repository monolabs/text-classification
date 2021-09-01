import pandas as pd
import numpy as np
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import contractions

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score




def convert_to_wordnet_tag(tag):
    '''
    Get wordnet tag based on the first letter of POS tag
    '''
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag.upper()[0], wordnet.NOUN)


def preprocess_document(doc):
    
    # 0. fix contractions
    doc = contractions.fix(doc)
    
    # 1. split documents into sentences
    splitter = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = splitter.tokenize(doc)
    
    # 2. word tokenization
    tokens_all = [nltk.word_tokenize(sentence) for sentence in sentences]
    
    
    # 3. WordNet tagging
    
    tags_all = []
    for tokens in tokens_all:
        sentence_tags = nltk.pos_tag([token.lower() for token in tokens])
        tags_all.append(sentence_tags)
    
    wordnet_tags_all = []
    for tags in tags_all:
        wordnet_tags = [(tag[0], convert_to_wordnet_tag(tag[1])) for tag in tags]
        wordnet_tags_all.append(wordnet_tags)
        
        
    # 4. lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentences = []
    for tags in wordnet_tags_all:
        lemmatized_sentence = ' '.join([lemmatizer.lemmatize(tag[0], pos=tag[1]) for tag in tags])
        lemmatized_sentences.append(lemmatized_sentence)
        
    # 5. sentences to doc
    lemmatized_doc = ' '.join(lemmatized_sentences)
    
    return lemmatized_doc



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



def get_wordcount(doc):
    return len(nltk.word_tokenize(doc))