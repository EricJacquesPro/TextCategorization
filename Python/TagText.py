class TagText:
    import nltk
    import numpy as np
    import os
    import pandas as pd
    import re                      # Regular expressions
    import sys
    
    from bs4 import BeautifulSoup
    from collections import defaultdict
    from nltk import sent_tokenize, word_tokenize
    from nltk.collocations import *
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    try:
        stopwords = set(stopwords.words('english'))
    except LookupError:
        import nltk
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        stopwords = set(stopwords.words('english'))
    #stopwords
    
    
    urlDirectory = "Data/"
    fileName = 'QuestionVsTags.csv'
    
    def test(self):
        '''
        Function to try this class
        '''
        return 'Ok'

    def read_source(self):
        '''
        Function to read csv source to manage tag
        '''
        df = self.pd.DataFrame()
        if self.os.path.exists(self.urlDirectory):
            df = self.pd.read_csv(
                self.urlDirectory+self.fileName,
                header=0,
                sep=',',
                encoding='utf-8'
            )
        return df
    
    def preprocessing(self, text):
        '''
        Function to prepare data before 
        '''
        clean_data = text
        clean_data = clean_data.lower()
        clean_data = self.BeautifulSoup(clean_data, 'html.parser').get_text()
        #clean_data = self.re.sub(r'[^\w\s]', '', clean_data) # remove punctuation
        return clean_data

    def manage_corpora(self, df_text):
        corpora = self.defaultdict(list)
        
        tokenizer = self.nltk.RegexpTokenizer(r'\w+')

        # create corpus
        for id, row in df_text.head().iteritems():
            corpora[id] += tokenizer.tokenize(row)
        
        stats, freq = dict(), dict()
        print(corpora)
        for k, v in corpora.iteritems():
            freq[k] = fq = self.nltk.FreqDist(v)
            stats[k] = {'total': len(v)} 
        return (freq, stats, corpora)
    
    def freq_stats_corpora(self, data_question):

        # count
        freq, stats, corpora = self.manage_corpora(data_question)
        df = self.pd.DataFrame.from_dict(stats, orient='index')
        return df
