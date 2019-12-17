class TagText:

    import nltk
    import numpy as np
    import os
    import pandas as pd
    import re                      # Regular expressions
    import sys

    from bs4 import BeautifulSoup
    from collections import defaultdict
    from sklearn.externals import joblib
    from nltk import sent_tokenize, word_tokenize
    
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from string import punctuation as ponctuation

    try:
        stopwords = set(stopwords.words('english'))
    except LookupError:
        import nltk
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        stopwords = set(stopwords.words('english'))
    

    urlDirectory = "Data/"
    fileName = 'QuestionVsTags.csv'


    urlDirectoryLoad = 'Data/'
    lda_filename = 'lda.joblib'
    lda_df_filename = 'lda_df_topic_keywords.joblib'
    lda_tf_filename = 'lda_tf_vectorizer.joblib'
    nmf_filename = 'nmf.joblib'
    nmf_df_filename = 'nmf_df_topic_keywords.joblib'
    nmf_tf_filename = 'nmf_tf_vectorizer.joblib'
    supervised_classifier_filename = 'supervised_classifier.joblib'
    supervised_classes_filename = 'supervised_classes.joblib'
    precision = 4000
    probabilite_minimun = 0.050

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
                encoding='utf-8',
                nrows=self.precision
            )
        return df

    def cleanIngred(self, s):
        '''
        Function to clean data
        '''
        # remove leading and trailing whitespace
        s = s.strip()
        # remove unwanted words
        s = ' '.join(
            word for word in s.split() if not (
                word in self.stopwords
            )
        )
        punctuation = self.re.compile('[{}]+'.format(
            self.re.escape(self.ponctuation))
        )

        s = punctuation.sub('', s)

        return s

    def preprocessing(self, text):
        '''
        Function to prepare data
        '''
        clean_data = text
        clean_data = clean_data.lower()
        clean_data = self.BeautifulSoup(clean_data, 'html.parser').get_text()
#        remove punctuation
#        clean_data = self.re.sub(r'[^\w\s]', '', clean_data)
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
        freq, stats, corpora = self.manage_corpora(data_question)
        df = self.pd.DataFrame.from_dict(stats, orient='index')
        return df

    def count_word_occurencies(self, df):
        '''
        make a dictionnary of words from dataframe and sort them by the number of occurences
        '''
        corpora = self.defaultdict(list)
        tokenizer = self.nltk.RegexpTokenizer(r'\w+')
        for id, row in df.iteritems():
            for word in tokenizer.tokenize(row):
                if word not in corpora:
                    corpora[word] = 0
                corpora[word] += 1
        corpora = dict((k, v) for k, v in corpora.items() if v > 1)
        return (sorted(corpora.items(), reverse=True,  key=lambda x: x[1]))

    def unsupervised_tag(self, dict_word_key, new_question, number_max_tag):
        '''
        find tag of message from dictionnary. The best words in dictionnary and in message 
        '''
        tags = [
            word
            for word in new_question.split()
            if (word in dict_word_key)
        ]
        return tags[0:number_max_tag]

    def display_topics(self, model, feature_names, no_top_words):
        '''
        display topic
        '''
        for topic_idx, topic in enumerate(model.components_):
            print("Topic :", topic_idx, ":")
            print (
                    " ".join([
                            feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]
                            ])
                )

    def lda_load(self):
        '''
        Load lda from file
        '''
        return self.urlDirectoryLoad+self.lda_filename
        lda = self.joblib.load(
            self.urlDirectoryLoad +
            self.lda_filename
        )
        return lda

    def lda_df_load(self):
        '''
        Load dataframe lda from file
        '''
        lda_df_topic_keywords = self.joblib.load(
            self.urlDirectoryLoad +
            self.lda_df_filename
        )
        return lda_df_topic_keywords

    def lda_tf_load(self):
        '''
        Load tf vectorize from file
        '''
        lda_tf_vectorizer = self.joblib.load(
            self.urlDirectoryLoad +
            self.lda_tf_filename
        )
        return lda_tf_vectorizer

    def lda_prepare_tag_load(self):
        '''
        load lda, topic ad tf vectorizer from file
        '''
        lda = self.joblib.load(
            self.urlDirectoryLoad +
            self.lda_filename
        )
        lda_df_topic_keywords = self.joblib.load(
            self.urlDirectoryLoad +
            self.lda_df_filename
        )
        lda_tf_vectorizer = self.joblib.load(
            self.urlDirectoryLoad +
            self.lda_tf_filename
        )
        return lda, lda_df_topic_keywords, lda_tf_vectorizer

    def lda_prepare_tag(self, data_preprocessed):
        '''
        prepare lda, topic ad tf vectorizer from data preprocessed
        '''
        no_components = 20

        documents = data_preprocessed[0:self.precision].unique()

        lda_tf_vectorizer = self.CountVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=50000,
            stop_words='english'
        )
        lda_tf = lda_tf_vectorizer.fit_transform(documents)

        # Run LDA
        lda = self.LatentDirichletAllocation(
            n_topics=no_components, #n_components=no_components,
            max_iter=5,
            learning_method='online',
            learning_offset=50.,
            random_state=0
        ).fit(lda_tf)
        
        lda_topicnames = ["Topic" + str(i) for i in range(lda.n_topics)]#lda.n_components)]
        # Topic-Keyword Matrix
        lda_df_topic_keywords = self.pd.DataFrame(lda.components_)
        
        # Assign Column and Index
        lda_df_topic_keywords.columns = lda_tf_vectorizer.get_feature_names()
        lda_df_topic_keywords.index = lda_topicnames
        return lda, lda_df_topic_keywords, lda_tf_vectorizer

    def lda_prepare_tag_and_save(self, data_preprocessed):
        '''
        prepare lda, topic ad tf vectorizer from data preprocessed
        and save them in file for future loading
        '''
        lda, lda_df_topic_keywords, lda_tf_vectorizer = self.lda_prepare_tag(
            data_preprocessed
        )
        self.lda_save(lda, lda_df_topic_keywords, lda_tf_vectorizer)
        return lda, lda_df_topic_keywords, lda_tf_vectorizer

    def lda_save(self, lda, lda_df_topic_keywords, lda_tf_vectorizer):
        '''
        save lda, topic ad tf vectorizer in file for future loading
        '''
        self.joblib.dump(
            lda,
            (
                self.urlDirectoryLoad +
                self.lda_filename
            ),
            0
        )
        self.joblib.dump(
            lda_df_topic_keywords,
            (
                self.urlDirectoryLoad +
                self.lda_df_filename
            )
        )
        self.joblib.dump(
            lda_tf_vectorizer,
            (
                self.urlDirectoryLoad +
                self.lda_tf_filename
            ),
            0
        )
        return

    def lda_predict(
        self,
        text,
        lda,
        lda_df_topic_keywords,
        lda_tf_vectorizer,
        no_top_words
    ):
        '''
        predict tag form text in function of lda, topic ad tf vectorizer
        '''
        text = [text]
        mytext = lda_tf_vectorizer.transform(text)
        lda_topic_probability_scores = lda.transform(mytext)
        lda_topic = lda_df_topic_keywords.iloc[
            self.np.argmax(lda_topic_probability_scores),
            :
        ].values.tolist()
        topic_array = self.np.array(lda_topic)
        lda_feature_names = lda_tf_vectorizer.get_feature_names()
        return (
            " ".join(
                        [
                            lda_feature_names[i]
                            for i in topic_array.argsort()
                            [
                                :-no_top_words - 1:-1
                            ]
                        ]
                )
        )

    def nmf_prepare_tag_load(self):
        '''
        load nmf, topic ad tf vectorizer from file
        '''
        nmf = self.joblib.load(
            self.urlDirectoryLoad +
            self.nmf_filename
            )
        nmf_df_topic_keywords = self.joblib.load(
            self.urlDirectoryLoad +
            self.nmf_df_filename
            )
        nmf_tf_vectorizer = self.joblib.load(
            self.urlDirectoryLoad +
            self.nmf_tf_filename
            )
        return nmf, nmf_df_topic_keywords, nmf_tf_vectorizer

    def nmf_prepare_tag(self, data_preprocessed):
        '''
        prepare nmf, topic ad tf vectorizer from data preprocessed
        '''
        X = data_preprocessed[0:self.precision].unique()
        # NMF is able to use tf-idf
        nmf_tfidf_vectorizer = self.TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=50000,
            stop_words='english'
        )
        nmf_tfidf = nmf_tfidf_vectorizer.fit_transform(X)
        no_components = 5
        # Run NMF
        nmf = self.NMF(
            n_components=no_components,
            random_state=1,
            alpha=.1,
            l1_ratio=.5,
            init='nndsvd'
        ).fit(nmf_tfidf)

        nmf_topicnames = ["Topic" + str(i) for i in range(nmf.n_components)]

        # Topic-Keyword Matrix
        nmf_df_topic_keyword = self.pd.DataFrame(nmf.components_)

        # Assign Column and Index
        nmf_df_topic_keyword.columns = nmf_tfidf_vectorizer.get_feature_names()
        nmf_df_topic_keyword.index = nmf_topicnames
        return nmf, nmf_df_topic_keyword, nmf_tfidf_vectorizer

    def nmf_prepare_tag_and_save(self, data_preprocessed):
        '''
        prepare nmf, topic ad tf vectorizer from data preprocessed
        and save them in file for future loading
        '''
        nmf, nmf_df_topic_keyword, nmf_tfidf_vectorizer = self.nmf_prepare_tag(
            data_preprocessed
        )
        self.nmf_save(
            nmf,
            nmf_df_topic_keyword,
            nmf_tfidf_vectorizer
        )
        return nmf, nmf_df_topic_keyword, nmf_tfidf_vectorizer

    def nmf_save(
        self,
        nmf,
        nmf_df_topic_keywords,
        nmf_tfidf_vectorizer
    ):
        '''
        save lda, topic ad tf vectorizer 
        in file for future loading
        '''
        self.joblib.dump(
                nmf,
                (
                    self.urlDirectoryLoad +
                    self.nmf_filename
                ),
                0
            )
        self.joblib.dump(
                nmf_df_topic_keywords,
                (
                    self.urlDirectoryLoad +
                    self.nmf_df_filename
                ),
                0
            )
        self.joblib.dump(
                nmf_tfidf_vectorizer, (
                    self.urlDirectoryLoad +
                    self.nmf_tf_filename
                ),
                0
            )
        return

    def nmf_predict(
        self,
        text,
        nmf,
        nmf_df_topic_keywords,
        nmf_tfidf_vectorizer,
        no_top_words
    ):
        '''
        save nmf, topic ad tf vectorizer in file for future loading
        '''
        text = [text]
        mytext = nmf_tfidf_vectorizer.transform(text)
        nmf_topic_probability_scores = nmf.transform(mytext)
        nmf_topic = nmf_df_topic_keywords.iloc[
            self.np.argmax(nmf_topic_probability_scores),
            :
        ].values.tolist()
        nmf_topic_array = self.np.array(nmf_topic)
        nmf_feature_names = nmf_tfidf_vectorizer.get_feature_names()
        return (" ".join([
                        nmf_feature_names[i]
                        for i in nmf_topic_array.argsort()
                        [
                            :-no_top_words - 1:-1
                        ]
                    ])
                )
    
    def test_supervised_rg(self, data_tag, data_preprocessed):
        X = [str(item) for item in data_preprocessed]
        y_train_tag = [
            item[:-1].split(',')
            for item in data_tag
        ]
#        -1 car il y a un ',' en fin de ligne
        print(y_train_tag)

        lb = self.MultiLabelBinarizer()
        Y = lb.fit_transform(y_train_tag)
        print(Y)

        classifier = self.Pipeline([
            ('vectorizer', self.CountVectorizer()),
            ('tfidf', self.TfidfTransformer()),
            (
                'clf',
                self.RandomForestClassifier(
                    n_estimators=100,
                    max_depth=2,
                    random_state=0
                )
            )
        ])

        classifier.fit(X, Y)
        predicted = classifier.predict_proba(["Git ist good"])

        tempTag = [item[0][1] for item in predicted]
        print(lb.classes_)
        list_id = [i for i, x in enumerate(tempTag) if x > 0.050]
        print([lb.classes_[id] for id in list_id])
        return str([lb.classes_[id] for id in list_id])
    
    def supervised_prepare_tag_load(self):
        '''
        load classifier and classe from file for supervised model
        '''
        classifier = self.joblib.load(
            self.urlDirectoryLoad +
            self.supervised_classifier_filename
        )
        classes = self.joblib.load(
            self.urlDirectoryLoad +
            self.supervised_classes_filename
        )
        return classifier, classes

    def supervised_prepare_tag(self, data_preprocessed, data_tag):
        '''
        prepare classifier and classe from file for supervised model
        '''
        X = [str(item) for item in data_preprocessed]
        y_train_tag = [
            item[:-1].split(',')
            for item in data_tag
        ]
#       -1 car il y a un ',' à la fin de la ligne
        print(y_train_tag)
        lb = self.MultiLabelBinarizer()
        Y = lb.fit_transform(y_train_tag)
        print(Y)

        classifier = self.Pipeline([
            ('vectorizer', self.CountVectorizer()),
            ('tfidf', self.TfidfTransformer()),
            (
                'clf',
                self.RandomForestClassifier(
                    n_estimators=100,
                    max_depth=2,
                    random_state=0
                )
            )
        ])  
        classifier.fit(X, Y)
        return classifier, lb.classes_

    def supervised_prepare_tag_and_save(self, data_preprocessed, data_tag):
        '''
        prepare and save classifier and classe from file for supervised model
        '''
        classifier, classes = self.supervised_prepare_tag(
            data_preprocessed,
            data_tag
        )
        self.supervised_save(classifier, classes)
        return classifier, classes

    def supervised_save(self, classifier, classes):
        '''
        save classifier and classe from file for supervised model
        '''
        self.joblib.dump(
            classifier,
            (
                self.urlDirectoryLoad +
                self.supervised_classifier_filename
            ),
            0
        )
        self.joblib.dump(
            classes,
            (
                self.urlDirectoryLoad +
                self.supervised_classes_filename
            ),
            0
        )
        return

    def supervised_predict(self, text, classifier, classes):
        '''
        predict tag form text in function of supervised model
        '''
        predicted = classifier.predict_proba([text])
        tempTag = [item[0][1] for item in predicted]
        print(classes)
        list_id = [i for i, x in enumerate(tempTag) if x > self.probabilite_minimun] #0.050]
        print([classes[id] for id in list_id])
        return str([classes[id] for id in list_id])

