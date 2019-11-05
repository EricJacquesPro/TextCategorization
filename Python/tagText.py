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
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

    def display_topics(self, model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic :", topic_idx, ":" )
            print (" ".join([feature_names[i]
                             for i in topic.argsort()[:-no_top_words - 1:-1]])
                )

    def lda_prepare_tag(self, data_preprocessed):
        no_topics = 20
        no_top_words = 5
        
        documents = data_preprocessed[0:9].unique()

        tf_vectorizer = self.CountVectorizer(
            max_df=0.95,
            min_df=2, 
            max_features=1000,
            stop_words='english'
        )
        tf = tf_vectorizer.fit_transform(documents)

        # Run LDA
        lda = self.LatentDirichletAllocation(n_topics=no_topics, 
                                        max_iter=5, 
                                        learning_method='online', 
                                        learning_offset=50.,
                                        random_state=0
                ).fit(tf)

        self.display_topics(lda, tf_vectorizer.get_feature_names(), no_top_words)
        
        topicnames = ["Topic" + str(i) for i in range(lda.n_topics)]

        # Topic-Keyword Matrix
        df_topic_keywords = self.pd.DataFrame(lda.components_)

        # Assign Column and Index
        df_topic_keywords.columns = tf_vectorizer.get_feature_names()
        df_topic_keywords.index = topicnames
        return lda, df_topic_keywords, tf_vectorizer
    
    def lda_predict(self, text, lda, df_topic_keywords, tf_vectorizer, no_top_words):#, nlp=nlp):
        global sent_to_words
        mytext = tf_vectorizer.transform(text)
        topic_probability_scores = lda.transform(mytext)
        topic = df_topic_keywords.iloc[self.np.argmax(topic_probability_scores), :].values.tolist()
        
        topic_array = self.np.array(topic)
        feature_names_lda = tf_vectorizer.get_feature_names()
        #topic_dict = dict(topic)
        return (" ".join([feature_names_lda[i]
                        for i in topic_array.argsort()[:-no_top_words - 1:-1]]))
#        return topic, topic_probability_scores
        
    def nmf_prepare_tag(self, data_preprocessed):
        no_top_words = 5
        no_features=0
        X = data_preprocessed[0:9].unique()

        # NMF is able to use tf-idf
        tfidf_vectorizer = self.TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=None,
            stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(X)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()

        no_topics = 5

        # Run NMF
        nmf = self.NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

        no_top_words = 10
        self.display_topics(nmf, tfidf_feature_names, no_top_words)
        
        topicnames = ["Topic" + str(i) for i in range(nmf.n_topics)]
        
        # Topic-Keyword Matrix
        df_topic_keywords = self.pd.DataFrame(nmf.components_)

        # Assign Column and Index
        df_topic_keywords.columns = tfidf_vectorizer.get_feature_names()
        df_topic_keywords.index = topicnames
        return lda, df_topic_keywords, tfidf_vectorizer

    def nmf_predict(self, text, nmf, df_topic_keywords, tfidf_vectorizer, no_top_words):#, nlp=nlp):
        global sent_to_words
        mytext = tfidf_vectorizer.transform(text)
        topic_probability_scores = nmf.transform(mytext)
        topic = df_topic_keywords.iloc[self.np.argmax(topic_probability_scores), :].values.tolist()
        return topic, topic_probability_scores