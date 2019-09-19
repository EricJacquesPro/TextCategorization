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
    #from nltk.collocations import *
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from string import punctuation as ponctuation
    
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
    
    def __init__(self):
        self.urlDirectory = "Data/"
        self.fileName = 'QuestionVsTags.csv'
        
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
    
    def cleanIngred(self, s):
        # remove leading and trailing whitespace
        s = s.strip()
        # remove unwanted words
        s = ' '.join(word for word in s.split() if not (word in self.stopwords))
        #s = ' '.join(word for word in s.split() if not ((word in self.badwords) or (word in self.stopwords)))

        punctuation = self.re.compile('[{}]+'.format(self.re.escape(self.ponctuation)))

        s = punctuation.sub('', s)

        return s

    def preprocessing(self, text):
        '''
        Function to prepare data before 
        '''
        clean_data = text
        clean_data = clean_data.lower()
        clean_data = self.BeautifulSoup(clean_data, 'html.parser').get_text()
        #clean_data = self.re.sub(r'[^\w\s]', '', clean_data) # remove punctuation
        clean_data = self.cleanIngred(clean_data)
        return clean_data


    def manage_corpora(self, df_text):
        corpora = self.defaultdict(list)
        
        tokenizer = self.nltk.RegexpTokenizer(r'\w+')

        # create corpus
        for id, row in df_text.head().iteritems():
            corpora[id] += tokenizer.tokenize(row)
        
        stats, freq = dict(), dict()
        for k, v in corpora.iteritems():
            freq[k] = self.nltk.FreqDist(v)
            stats[k] = {'total': len(v)} 
        return (freq, stats, corpora)
    
    def freq_stats_corpora(self, data_question):

        # count
        freq, stats, corpora = self.manage_corpora(data_question)
        df = self.pd.DataFrame.from_dict(stats, orient='index')
        return df

    def count_word_occurencies(self, df):
        corpora = self.defaultdict(list)                
        tokenizer = self.nltk.RegexpTokenizer(r'\w+')
        for id, row in df.iteritems():
            for word in tokenizer.tokenize(row):
                if not word in corpora :
                    corpora[word] = 0
                #print(word)
                corpora[word] += 1
        corpora = dict((k, v) for k, v in corpora.items() if v > 1)        
        return (sorted(corpora.items(), reverse=True,  key=lambda x: x[1]))

    def unsupervised_tag(self, dict_word_key, new_question, number_max_tag):
        tags = [word for word in new_question.split() if (word in dict_word_key)]
        return tags[0:number_max_tag]
