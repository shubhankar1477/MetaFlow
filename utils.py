import pandas as pd

from nltk.stem.snowball import EnglishStemmer
from copy import deepcopy
snowball = EnglishStemmer()
from bs4 import BeautifulSoup
import unicodedata
from nltk.tokenize import sent_tokenize
import re
# from sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')




def remove_whitespace_and_special_chars(text):
        """general text cleaning"""
        """Includes removal of extra whitespace, special characters"""
        
        tab_newline_pattern = '[\t\n]'
        multi_space = ' {2,}'
        special_chars = '[^a-zA-Z0-9 ]'        
        formatted_text = re.sub(tab_newline_pattern, ' ', text)
        formatted_text = re.sub(multi_space, ' ', formatted_text)
        formatted_text = re.sub(special_chars, ' ', formatted_text)
        formatted_text = re.sub(multi_space, ' ', formatted_text)
        return formatted_text
def removeStopWord(s):
        """ Removing English stopwords from the text """
        
        stop = set(stopwords.words('english'))
        return " ".join([words for words in word_tokenize(s) if words not in stop])
    
def remove_non_ascii(text):
        """Remove non-ASCII characters from list of tokenized words"""
        text = re.sub(r'\x85', '', text) # replace ellipses
        text = re.sub(r'\x91', '', text)  # replace left single quote
        text = re.sub(r'\x92', '', text)  # replace right single quote
        text = re.sub(u'\x93', '', text)  # replace left double quote
        text = re.sub(u'\x94', '', text)  # replace right double quote
        text = re.sub(r'\x95', '', text)   # replace bullet
        text = re.sub(r'\x96', '', text)   # replace bullet
        text = re.sub(r'\x99', '', text)   #replace TM
        text = re.sub(r'\xae', '', text)    # replace (R)
        text = re.sub(r'\xb0', '', text)    # replace degree symbol
        text = re.sub(r'\xba', '', text)    # replace degree symbol
        new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return new_text    
def snowball_stemmer(token):
            return snowball.stem(token)
def stemmer(text):
        return " ".join([snowball_stemmer(w) for w in text.split()]).strip()
    
def transform(x):
        text = remove_non_ascii(x.lower())
        text = remove_whitespace_and_special_chars(text)
        text = removeStopWord(text)
        # text = stemmer(text)
        return text