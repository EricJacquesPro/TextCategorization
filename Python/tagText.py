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
    #from nltk.collocations import *
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
    from string import punctuation as ponctuation

    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

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


    urlDirectoryLoad = '/home/ejacques/mysite/data/tag_text/V3k/'
    lda_filename = 'lda.joblib'
    lda_df_filename = 'lda_df_topic_keywords.joblib'
    lda_tf_filename = 'lda_tf_vectorizer.joblib'
    nmf_filename = 'nmf.joblib'
    nmf_df_filename = 'nmf_df_topic_keywords.joblib'
    nmf_tf_filename = 'nmf_tf_vectorizer.joblib'
    supervised_classifier_filename = 'supervised_classifier.joblib'
    supervised_classes_filename = 'supervised_classes.joblib'
    precision = 4000

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

    def lda_load(self):
        return self.urlDirectoryLoad+self.lda_filename
        lda = self.joblib.load(self.urlDirectoryLoad+self.lda_filename)
        return lda

    def lda_df_load(self):
        lda_df_topic_keywords = self.joblib.load(self.urlDirectoryLoad+self.lda_df_filename)
        return lda_df_topic_keywords

    def lda_tf_load(self):
        lda_tf_vectorizer = self.joblib.load(self.urlDirectoryLoad+self.lda_tf_filename)
        return lda_tf_vectorizer

    def lda_prepare_tag_load(self):
        lda = self.joblib.load(self.urlDirectoryLoad+self.lda_filename)
        lda_df_topic_keywords = self.joblib.load(self.urlDirectoryLoad+self.lda_df_filename)
        lda_tf_vectorizer = self.joblib.load(self.urlDirectoryLoad+self.lda_tf_filename)
        return lda, lda_df_topic_keywords, lda_tf_vectorizer

    def lda_prepare_tag(self, data_preprocessed):
        #no_topics = 20
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
        lda = self.LatentDirichletAllocation(n_components =no_components, #n_topics=no_topics,
                                        max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0
                ).fit(lda_tf)
        '''
        no_top_words = 5
        self.display_topics(lda, lda_tf_vectorizer.get_feature_names(), no_top_words)
        '''

        lda_topicnames = ["Topic" + str(i) for i in range(lda.n_components )]#.n_topics)]

        # Topic-Keyword Matrix
        lda_df_topic_keywords = self.pd.DataFrame(lda.components_)

        # Assign Column and Index
        lda_df_topic_keywords.columns = lda_tf_vectorizer.get_feature_names()
        lda_df_topic_keywords.index = lda_topicnames
        return lda, lda_df_topic_keywords, lda_tf_vectorizer

    def lda_prepare_tag_and_save(self, data_preprocessed):
        lda, lda_df_topic_keywords, lda_tf_vectorizer = self.lda_prepare_tag(data_preprocessed)
        self.lda_save(lda, lda_df_topic_keywords, lda_tf_vectorizer)
        return lda, lda_df_topic_keywords, lda_tf_vectorizer

    def lda_save(self, lda, lda_df_topic_keywords, lda_tf_vectorizer):
        self.joblib.dump(lda, self.urlDirectoryLoad+self.lda_filename, 0)
        self.joblib.dump(lda_df_topic_keywords, self.urlDirectoryLoad+self.lda_df_filename)
        self.joblib.dump(lda_tf_vectorizer, self.urlDirectoryLoad+self.lda_tf_filename, 0)
        return

    def lda_predict(self, text, lda, lda_df_topic_keywords, lda_tf_vectorizer, no_top_words):#, nlp=nlp):
        text=[text]
        mytext = lda_tf_vectorizer.transform(text)
        lda_topic_probability_scores = lda.transform(mytext)
        lda_topic = lda_df_topic_keywords.iloc[self.np.argmax(lda_topic_probability_scores), :].values.tolist()

        topic_array = self.np.array(lda_topic)
        lda_feature_names = lda_tf_vectorizer.get_feature_names()
        #topic_dict = dict(topic)
        return (" ".join([lda_feature_names[i]
                        for i in topic_array.argsort()[:-no_top_words - 1:-1]]))

    def nmf_prepare_tag_load(self):
        nmf = self.joblib.load(self.urlDirectoryLoad+self.nmf_filename)
        nmf_df_topic_keywords = self.joblib.load(self.urlDirectoryLoad+self.nmf_df_filename)
        nmf_tf_vectorizer = self.joblib.load(self.urlDirectoryLoad+self.nmf_tf_filename)
        return nmf, nmf_df_topic_keywords, nmf_tf_vectorizer

    def nmf_prepare_tag(self, data_preprocessed):
        #no_top_words = 5
        X = data_preprocessed[0:self.precision].unique()

        # NMF is able to use tf-idf
        nmf_tfidf_vectorizer = self.TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=50000,
            stop_words='english')
        nmf_tfidf = nmf_tfidf_vectorizer.fit_transform(X)
        #nmf_tfidf_feature_names = nmf_tfidf_vectorizer.get_feature_names()

        no_components = 5

        # Run NMF
        nmf = self.NMF(n_components=no_components, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(nmf_tfidf)

        '''
        no_top_words = 10
        self.display_topics(nmf, nmf_tfidf_feature_names, no_top_words)
        '''

        nmf_topicnames = ["Topic" + str(i) for i in range(nmf.n_components)]

        # Topic-Keyword Matrix
        nmf_df_topic_keywords = self.pd.DataFrame(nmf.components_)

        # Assign Column and Index
        nmf_df_topic_keywords.columns = nmf_tfidf_vectorizer.get_feature_names()
        nmf_df_topic_keywords.index = nmf_topicnames
        return nmf, nmf_df_topic_keywords, nmf_tfidf_vectorizer

    def nmf_prepare_tag_and_save(self, data_preprocessed):
        nmf, nmf_df_topic_keywords, nmf_tfidf_vectorizer = self.nmf_prepare_tag(data_preprocessed)
        self.nmf_save(nmf, nmf_df_topic_keywords, nmf_tfidf_vectorizer)
        return nmf, nmf_df_topic_keywords, nmf_tfidf_vectorizer

    def nmf_save(self, nmf, nmf_df_topic_keywords, nmf_tfidf_vectorizer):
        self.joblib.dump(nmf, self.urlDirectoryLoad+self.nmf_filename, 0)
        self.joblib.dump(nmf_df_topic_keywords, self.urlDirectoryLoad+self.nmf_df_filename, 0)
        self.joblib.dump(nmf_tfidf_vectorizer, self.urlDirectoryLoad+self.nmf_tf_filename, 0)
        return

    def nmf_predict(self, text, nmf, nmf_df_topic_keywords, nmf_tfidf_vectorizer, no_top_words):#, nlp=nlp):
        text = [text]
        mytext = nmf_tfidf_vectorizer.transform(text)
        nmf_topic_probability_scores = nmf.transform(mytext)
        nmf_topic = nmf_df_topic_keywords.iloc[self.np.argmax(nmf_topic_probability_scores), :].values.tolist()
        nmf_topic_array = self.np.array(nmf_topic)
        nmf_feature_names = nmf_tfidf_vectorizer.get_feature_names()
        #topic_dict = dict(topic)
        return (" ".join([nmf_feature_names[i]
                        for i in nmf_topic_array.argsort()[:-no_top_words - 1:-1]]))


    def supervised_define_words(self, data_preprocessed):
        list_word = data_preprocessed[0:9].unique()
        return list_word

    def supervised_define_tags(self, data_preprocessed):
        list_tag = data_preprocessed[0:9].unique()
        return list_tag

    def testRg(self):
        X_train = self.np.array(["new york is a hell of a town",
                        "new york was originally dutch",
                        "the big apple is great",
                        "new york is also called the big apple",
                        "nyc is nice",
                        "people abbreviate new york city as nyc",
                        "the capital of great britain is london",
                        "london is in the uk",
                        "london is in england",
                        "london is in great britain",
                        "it rains a lot in london",
                        "london hosts the british museum",
                        "new york is great and so is london",
                        "i like london better than new york",
                        "france"])

        '''
        y_train_text = [["new york"],["new york"],["new york"],["new york"],
                        ["new york"],["new york"],["london"],["london"],
                        ["london"],["london"],["london"],["london"],
                        ["new york","England"],["new york"],["France"]]
        '''
        y_train = ["new york","new york","new york","new york",
                    "new york","new york","london","london",
                    "london","london","london","london",
                   "new york,England","new york","France"]
        y_train_text = [item.split(',') for item in y_train]

        #lb = self.MultiLabelBinarizer(classes=("new york","london","England","France"))
        lb = self.MultiLabelBinarizer()
        Y = lb.fit_transform(y_train_text)
        print(Y)

        forest = self.RandomForestClassifier(n_estimators=10, random_state=1)

        classifier = self.Pipeline([
        ('vectorizer', self.CountVectorizer()),
        ('tfidf', self.TfidfTransformer()),
        #('clf',  DecisionTreeClassifier(random_state=0))
        ('clf', self.RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0))
        #('clf', OneVsRestClassifier(LinearSVC()))])
        ])

        classifier.fit(X_train, Y)
        predicted = classifier.predict_proba(["new york is great and so is london"])
        '''
        print(predicted)
        print([item[0][1] for item in predicted] )
        '''
        tempTag = [item[0][1] for item in predicted]
        print(lb.classes_)
        '''
        print([i for i, x in enumerate(tempTag) if x > 0.050])
        '''
        list_id = [i for i, x in enumerate(tempTag) if x > 0.050]
        '''
        print([id for id in list_id])
        '''
        print([lb.classes_[id] for id in list_id])
        #print("Accuracy Score: ",accuracy_score(Y_test, predicted))
        return str([lb.classes_[id] for id in list_id])

    def test_supervised_rg(self, data_tag, data_preprocessed):


        X = [str(item) for item in data_preprocessed]
        y_train_tag = [item[:-1].split(',') for item in data_tag]#-1 car il y a un , à la fin de la ligne
        print(y_train_tag)

        lb = self.MultiLabelBinarizer()
        Y = lb.fit_transform(y_train_tag)#y_train_text
        print(Y)

        classifier = self.Pipeline([
            ('vectorizer', self.CountVectorizer()),
            ('tfidf', self.TfidfTransformer()),
            ('clf', self.RandomForestClassifier(
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

    def test2(self):
        X_train = self.np.array(["new york is a hell of a town",
                        "new york was originally dutch",
                        "the big apple is great",
                        "new york is also called the big apple",
                        "nyc is nice",
                        "people abbreviate new york city as nyc",
                        "the capital of great britain is london",
                        "london is in the uk",
                        "london is in england",
                        "london is in great britain",
                        "it rains a lot in london",
                        "london hosts the british museum",
                        "new york is great and so is london",
                        "i like london better than new york",
                        "france"])

        '''
        y_train_text = [["new york"],["new york"],["new york"],["new york"],
                        ["new york"],["new york"],["london"],["london"],
                        ["london"],["london"],["london"],["london"],
                        ["new york","England"],["new york"],["France"]]
        '''
        y_train = ["new york","new york","new york","new york",
                    "new york","new york","london","london",
                    "london","london","london","london",
                   "new york,England","new york","France"]
        y_train_text = [item.split(',') for item in y_train]

        #lb = self.MultiLabelBinarizer(classes=("new york","london","England","France"))
        lb = self.MultiLabelBinarizer()
        Y = lb.fit_transform(y_train_text)
        print(Y)

        forest = self.RandomForestClassifier(n_estimators=10, random_state=1)

        classifier = self.Pipeline([
        ('vectorizer', self.CountVectorizer()),
        ('tfidf', self.TfidfTransformer()),
        #('clf',  DecisionTreeClassifier(random_state=0))
        ('clf', self.RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0))
        #('clf', OneVsRestClassifier(LinearSVC()))])
        ])

        classifier.fit(X_train, Y)
        predicted = classifier.predict_proba(["new york is great and so is london"])
        '''
        print(predicted)
        print([item[0][1] for item in predicted] )
        '''
        tempTag = [item[0][1] for item in predicted]
        print(lb.classes_)
        '''
        print([i for i, x in enumerate(tempTag) if x > 0.050])
        '''
        list_id = [i for i, x in enumerate(tempTag) if x > 0.050]
        '''
        print([id for id in list_id])
        '''
        print([lb.classes_[id] for id in list_id])
        #print("Accuracy Score: ",accuracy_score(Y_test, predicted))
        return 'finie'

    def supervised_prepare_tag_load(self):
        classifier = self.joblib.load(self.urlDirectoryLoad+self.supervised_classifier_filename)
        classes = self.joblib.load(self.urlDirectoryLoad+self.supervised_classes_filename)
        return classifier, classes

    def supervised_prepare_tag(self, data_preprocessed, data_tag):
        X = [str(item) for item in data_preprocessed]
        y_train_tag = [item[:-1].split(',') for item in data_tag]#-1 car il y a un , à la fin de la ligne
        print(y_train_tag)

        lb = self.MultiLabelBinarizer()
        Y = lb.fit_transform(y_train_tag)#y_train_text
        print(Y)

        classifier = self.Pipeline([
            ('vectorizer', self.CountVectorizer()),
            ('tfidf', self.TfidfTransformer()),
            ('clf', self.RandomForestClassifier(
                    n_estimators=100,
                    max_depth=2,
                    random_state=0
                )
            )
        ])

        classifier.fit(X, Y)

        return classifier, lb.classes_

    def supervised_prepare_tag_and_save(self, data_preprocessed, data_tag):
        classifier, classes = self.supervised_prepare_tag(data_preprocessed, data_tag)
        self.supervised_save(classifier, classes)
        return classifier, classes

    def supervised_save(self, classifier, classes):
        self.joblib.dump(classifier, self.urlDirectoryLoad+self.supervised_classifier_filename, 0)
        self.joblib.dump(classes, self.urlDirectoryLoad+self.supervised_classes_filename, 0)
        return

    def supervised_predict(self, text, classifier, classes):
        predicted = classifier.predict_proba([text])

        tempTag = [item[0][1] for item in predicted]
        print(classes)
        list_id = [i for i, x in enumerate(tempTag) if x > 0.050]
        print([classes[id] for id in list_id])
        return str([classes[id] for id in list_id])

