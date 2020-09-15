class TagText:

    import nltk
    import numpy as np
    import os
    import pandas as pd
    import re                      # Regular expressions
    import sys
    
    import matplotlib.pyplot as plt

    from bs4 import BeautifulSoup
    from collections import defaultdict
    from gensim.models.coherencemodel import CoherenceModel
    from sklearn.externals import joblib
    
    from nltk import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn import metrics
    from sklearn import svm, datasets
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
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
    lda_clf_filename = 'lda_clf.joblib'
    nmf_filename = 'nmf.joblib'
    nmf_df_filename = 'nmf_df_topic_keywords.joblib'
    nmf_tf_filename = 'nmf_tf_vectorizer.joblib'
    supervised_classifier_filename = 'supervised_classifier.joblib'
    supervised_classes_filename = 'supervised_classes.joblib'
    nombre_post_entree = 50000
    precision = 50000
    probabilite_minimun = 0.050
    n_topic = 10

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
                nrows=self.nombre_post_entree
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
    
    def train_test_split(self, X, y, test_size=0.33):
        '''
        split data into train and test data
        '''
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size)

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

    def lda_prepare_tag_V2_load(self):
        '''
        load lda, topic, tf vectorizer and classifier from file
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
        lda_clf = self.joblib.load(
            self.urlDirectoryLoad +
            self.lda_clf_filename
        )
        return lda, lda_df_topic_keywords, lda_tf_vectorizer, lda_clf
    
    def lda_train(self, lda_tf, no_tropics):
        '''
        train lda model
        '''
        from sklearn.pipeline import Pipeline
        
        lda = self.LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                    evaluate_every=-1, learning_decay=0.7,
                    learning_method='online', learning_offset=10.0,
                    max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001,
                    n_topics=no_tropics, #n_components=no_components,
                    n_jobs=-1, perp_tol=0.1,
                    random_state=100, topic_word_prior=None,
                    total_samples=1000000.0, verbose=0
                ).fit(lda_tf)
        # Log Likelyhood: Higher the better
        score =  lda.score(lda_tf)

        # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
        perplexity = lda.perplexity(lda_tf)

        # See model parameters
        print(lda.get_params())
        
        return lda, score, perplexity
    
    def lda_init(self, documents):
        '''
        init vector of document
        '''
        lda_tf_vectorizer = self.CountVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=50000,
            stop_words='english',
            lowercase=True,
        )
        lda_tf = lda_tf_vectorizer.fit_transform(documents)
        return lda_tf, lda_tf_vectorizer

    def lda_find_topic_number(
        self, 
        documents_train,
        documents_test,
        topic_number_min, 
        topic_number_max, 
        topic_number_step
    ):
        '''
        find the better number of topic for the LDA representation of document
        '''
        #documents = data_preprocessed[0:self.precision].unique()
        import numpy as np
        from sklearn import metrics
        from sklearn.preprocessing import LabelEncoder
        #from sklearn.model_selection import train_test_split
        
        #Y_all = np.zeros(len(documents))
        
        #documents_train, documents_test, y_train, y_test = self.train_test_split(X=list(documents), y=Y_all, test_size=0.33)
        
        lda_tf, lda_tf_vectorizer = self.lda_init(documents_train)
        lda_tf_test, lda_tf_vectorizer_test = self.lda_init(documents_train)
        
        #performance_indicateurs=[]
        performance_score_indicateurs=[]
        performance_score_validation_indicateurs=[]
        performance_perplexity_indicateurs=[]
        performance_perplexity_validation_indicateurs=[]
        for no_tropics in range(topic_number_min, topic_number_max, topic_number_step):
            lda, score, perplexity = self.lda_train(lda_tf, no_tropics)
            score_validation = lda.score(lda_tf_test)
            perplexity_validation = lda.score(lda_tf_test)
            performance_score_indicateurs.append(score)
            performance_perplexity_indicateurs.append(perplexity)
            performance_score_validation_indicateurs.append(score_validation)
            performance_perplexity_validation_indicateurs.append(perplexity_validation)
            del lda, score, perplexity, score_validation
        return performance_score_indicateurs, performance_perplexity_indicateurs, performance_score_validation_indicateurs, performance_perplexity_validation_indicateurs
    
    def lda_prepare_tag_V2(self, data, tag, no_tropics=32):
		'''
        prepare lda, topic, tf vectorizer and classifier from data preprocessed
        '''
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.decomposition import LatentDirichletAllocation

        # Sampling dataset
        vectorizer_X = TfidfVectorizer(
                    max_df=0.95,
                    min_df=2,
                    max_features=50000,
                    stop_words='english'
        )
        #Y
        y_all = [
            item[:-1].split(',')#-1 car il y a un ',' à la fin de la ligne
            for item in tag
        ]

        #print(y_train_tag)
        lb = self.MultiLabelBinarizer()
        Y_all = lb.fit_transform(y_all)

        # 80/20 split
        X_lda_train, X_lda_test, y_lda_train, y_lda_test = train_test_split(
            data, y_all, test_size=0.05,train_size=0.95, random_state=0)
        y_lda_train = lb.transform(y_lda_train)
        y_lda_test = lb.transform(y_lda_test)

        # TF-IDF matrices
        X_tfidf_train = vectorizer_X.fit_transform(X_lda_train)
        X_tfidf_test = vectorizer_X.transform(X_lda_test)

        lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 
        lda_top=lda_model.fit_transform(X_tfidf_train)

        clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=2,
                random_state=0
        )
        #lda_top=lda_model.fit_transform(X_tfidf_train)
        clf.fit(lda_top, y_lda_train)
        return lda_model, lb.classes_, vectorizer_X, clf
    
    def lda_prepare_tag(self, data_preprocessed, no_tropics=32):
        '''
		obsolete
        prepare lda, topic ad tf vectorizer from data preprocessed
        '''
        self.n_topic = no_tropics
        documents = data_preprocessed[0:self.precision].unique()
        lda_tf, lda_tf_vectorizer = self.lda_init(documents)
        
        lda, score, perplexity = self.lda_train(lda_tf, no_tropics)
        
        print("Log Likelihood: ", score)
        print("Perplexity: ", perplexity)

        # See model parameters
        print(lda.get_params())

        lda_topicnames = ["Topic" + str(i) for i in range(lda.n_topics)]
        # Topic-Keyword Matrix
        lda_df_topic_keywords = self.pd.DataFrame(lda.components_)
        
        # Assign Column and Index
        lda_df_topic_keywords.columns = lda_tf_vectorizer.get_feature_names()
        lda_df_topic_keywords.index = lda_topicnames
        return lda, lda_df_topic_keywords, lda_tf_vectorizer

    def lda_prepare_tag_and_save(self, data_preprocessed):
        '''
		obsolete
        prepare lda, topic ad tf vectorizer from data preprocessed
        and save them in file for future loading
        '''
        lda, lda_df_topic_keywords, lda_tf_vectorizer = self.lda_prepare_tag(
            data_preprocessed
        )
        self.lda_save(lda, lda_df_topic_keywords, lda_tf_vectorizer)
        return lda, lda_df_topic_keywords, lda_tf_vectorizer

    def lda_prepare_tag_and_save_V2(self, data_preprocessed, tag, no_tropics=32):
        '''
        prepare lda, topic, tf vectorizer and classifier from data preprocessed
        and save them in file for future loading
        '''
        lda, lda_df_topic_keywords, lda_tf_vectorizer, lda_clf = self.lda_prepare_tag_V2(
            data=data_preprocessed,
            tag=tag,
            no_tropics=no_tropics
        )
        self.lda_save_V2(lda, lda_df_topic_keywords, lda_tf_vectorizer, lda_clf)
        return lda, lda_df_topic_keywords, lda_tf_vectorizer, lda_clf

    def lda_save(self, lda, lda_df_topic_keywords, lda_tf_vectorizer):
        '''
		obsolete
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
    
    def lda_save_V2(self, lda, lda_df_topic_keywords, lda_tf_vectorizer, lda_clf):
        '''
        save lda, topic, tf vectorizer and classifier in file for future loading
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
        self.joblib.dump(
            lda_clf,
            (
                self.urlDirectoryLoad +
                self.lda_clf_filename
            ),
            0
        )    
        return
    
    
    def lda_predict_V2(
        self,
        text,
        lda_tf_vectorizer,
        lda,
        classifier,
        classes,
        no_top_words=5
    ):
        '''
        predict tag form text in function of lda, topic, tf vectorizer and classifier
        '''
        text = self.preprocessing(text)
        text = [text]
        mytext = lda_tf_vectorizer.transform(text)

        text_projection = lda.transform(mytext)
        predicted = classifier.predict_proba(text_projection)
        tempTag = [(1-item[0][0]) for item in predicted]
        list_id = [[i, x] for i, x in enumerate(tempTag) if x > 0.0050]

        list_id_sorted = sorted(list_id, reverse=True, key=lambda x: x[1])

        list_id_sorted_suggested = [x[0] for i, x in enumerate(list_id_sorted[:-no_top_words - 1:-1])]

        return str([classes[id] for id in list_id_sorted_suggested])


    def lda_predict(
        self,
        text,
        lda,
        lda_df_topic_keywords,
        lda_tf_vectorizer,
        no_top_words
    ):
        '''
		Obsolete
        predict tag form text in function of lda, topic ad tf vectorizer
        '''
        threshold = 0.010
        list_scores = []
        list_words = []
        used = set()

        text = self.preprocessing(text)
        text = [text]
        mytext = lda_tf_vectorizer.transform(text)
        text_projection = lda.transform(mytext)
        lda_feature_names = lda_tf_vectorizer.get_feature_names()
        lda_components = lda.components_ / lda.components_.sum(axis=1)[:, self.np.newaxis] # normalization

        for topic in range(self.n_topic):
            topic_score = text_projection[0][topic]

            for (word_idx, word_score) in zip(lda_components[topic].argsort()[:-5:-1], sorted(lda_components[topic])[:-5:-1]):
                score = topic_score*word_score

                if score >= threshold:
                    list_scores.append(score)
                    list_words.append(lda_feature_names[word_idx])
                    used.add(lda_feature_names[word_idx])

        results = [tag for (y,tag) in sorted(zip(list_scores,list_words), key=lambda pair: pair[0], reverse=True)]
        unique_results = [x for x in results if x not in used] # get only unique tags
        tags = " ".join(results[:5])

        return tags

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
    
    def nmf_train(self, nmf_tfidf, no_tropics):
        '''
        train nmf model
        '''
        # Run NMF
        nmf = self.NMF(
            n_components=no_tropics,
            random_state=1,
            alpha=.1,
            l1_ratio=.5,
            init='nndsvd'
        ).fit(nmf_tfidf)
        return nmf

    def nmf_init(self, documents):
        '''
        init vector of document
        '''
        # NMF is able to use tf-idf
        nmf_tfidf_vectorizer = self.TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=50000,
            stop_words='english'
        )
        
        nmf_tfidf = nmf_tfidf_vectorizer.fit_transform(documents)
        return nmf_tfidf, nmf_tfidf_vectorizer
    
    def nmf_find_topic_number(
        self, 
        data_preprocessed, 
        topic_number_min, 
        topic_number_max, 
        topic_number_step
    ):
        '''
        find the best number of cluster
        '''
        documents = data_preprocessed.unique()[0:self.precision]
        
        nmf_tfidf, nmf_tfidf_vectorizer = self.nmf_init(documents)
        
        performance_indicateurs=[]
        for no_topics in range(topic_number_min, topic_number_max, topic_number_step):
            nmf = self.nmf_train(nmf_tfidf, no_topics)
            performance_indicateurs.append(self.get_score(nmf, data_preprocessed))
            del nmf
        return performance_indicateurs;
    def get_score(self, model, data):
        '''
        Estimate performance of the model on the data 
        '''
        scorer=self.metrics.explained_variance_score
        prediction = model.inverse_transform(model.transform(data))
        return scorer(data, prediction)

    def nmf_prepare_tag(self, data_preprocessed, no_topics=32):
        '''
        prepare nmf, topic ad tf vectorizer from data preprocessed
        '''
        from sklearn.decomposition.nmf import _beta_divergence
        documents = data_preprocessed.unique()[0:self.precision]
        
        nmf_tfidf, nmf_tfidf_vectorizer = self.nmf_init(documents)
        nmf = self.nmf_train(nmf_tfidf, no_topics)
        
        print('original reconstruction error automatically calculated -> TRAIN: ', nmf.reconstruction_err_)

        """ Manual reconstruction_err_ calculation
            -> use transform to get W
            -> ask fitted NMF to get H
            -> use available _beta_divergence-function to calculate desired metric
        """
        W_train = nmf.transform(nmf_tfidf)
        rec_error = _beta_divergence(nmf_tfidf, W_train, nmf.components_, 'frobenius', square_root=True)
        print('Manually calculated rec-error train: ', rec_error)

        nmf_topicnames = ["Topic" + str(i) for i in range(nmf.n_components)]

        # Topic-Keyword Matrix
        nmf_df_topic_keyword = self.pd.DataFrame(nmf.components_)

        # Assign Column and Index
        nmf_df_topic_keyword.columns = nmf_tfidf_vectorizer.get_feature_names()
        nmf_df_topic_keyword.index = nmf_topicnames
        return nmf, nmf_df_topic_keyword, nmf_tfidf_vectorizer

    def nmf_prepare_tag_and_save(self, data_preprocessed, no_topics=32):
        '''
        prepare nmf, topic ad tf vectorizer from data preprocessed
        and save them in file for future loading
        '''
        nmf, nmf_df_topic_keyword, nmf_tfidf_vectorizer = self.nmf_prepare_tag(
            data_preprocessed, no_topics
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
        save nmf model, topic ad tf vectorizer 
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
        Predict tag of new post
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

    def supervised_prepare_tagV2(self, data_preprocessed, data_tag):
        '''
        prepare classifier and class from file for supervised model
        '''
        from sklearn.pipeline import Pipeline
        from sklearn import metrics
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer

        y_all = [
            item[:-1].split(',')#-1 car il y a un ',' à la fin de la ligne
            for item in data_tag
        ]

        #print(y_train_tag)
        lb = self.MultiLabelBinarizer()
        Y_all = lb.fit_transform(y_all)


        # 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            data_preprocessed, y_all, test_size=0.05,train_size=0.95, random_state=0)
        y_train = lb.transform(y_train)
        y_test = lb.transform(y_test)

        classifier = Pipeline([
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
        classifier.fit(X_train, y_train)
		
        nb1, score1, nb5, score5 = self.scoring(X_test, y_test, classifier, lb, False, None)

        print("score 1 occurence {}".format(score1))
        print("score 5 occurences {}".format(score5))

        return classifier, lb.classes_

    
    def supervised_prepare_tag(self, data_preprocessed, data_tag):
        '''
        prepare classifier and class from file for supervised model
        '''
        from sklearn.pipeline import Pipeline
        from sklearn import metrics
        y_all = [
            item[:-1].split(',')#-1 car il y a un ',' à la fin de la ligne
            for item in data_tag
        ]

        #print(y_train_tag)
        lb = self.MultiLabelBinarizer()
        Y_all = lb.fit_transform(y_all)
        
        X_train, X_test, y_train, y_test = self.train_test_split(X=data_preprocessed, y=Y_all, test_size=0.33)
        
        classifier = Pipeline([
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
        classifier.fit(X_train, y_train)
        
        predicted_train = classifier.predict(X_train)
        
        predicted_test = classifier.predict(X_test)
        return classifier, lb.classes_
        
    def supervised_prepare_tag_and_save(self, data_preprocessed, data_tag):
        '''
        prepare and save classifier and classe from file for supervised model
        '''
        classifier, classes = self.supervised_prepare_tagV2(
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

    def supervised_predict(self, text, classifier, classes, no_top_words=5):
        '''
        predict tag form text in function of supervised model
        '''
        predicted = classifier.predict_proba([text])
        tempTag = [(1-item[0][0]) for item in predicted]
        list_id = [[i, x] for i, x in enumerate(tempTag) if x > 0.0050]
        
        list_id_sorted = sorted(list_id, reverse=True, key=lambda x: x[1])
        
        list_id_sorted_suggested = [x[0] for i, x in enumerate(list_id_sorted[:-no_top_words - 1:-1])]
        
        return str([classes[id] for id in list_id_sorted_suggested])
                        
        
    def visualize_data(self, data_preprocessed):
        '''
        visualize data
        '''
        
        import matplotlib.pyplot as plt
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.manifold import TSNE
        documents = data_preprocessed[0:self.precision].unique()
        
        lda_tf_vectorizer = self.TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=50000,
            stop_words='english',
            lowercase=True,
        )
        lda_tf = lda_tf_vectorizer.fit_transform(documents)
        
        tsne_num_components=20
        
        # t-SNE plot
        embeddings = TSNE(n_components=tsne_num_components)
        Y = embeddings.fit_transform(lda_tf)
        plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
        plt.show()
    
    def scoring(self, x_test, y_true, clf, lb, mode_supervise_with_lda = False, lda_model = None):
		'''
		score the predict model
		'''
        import time
        debut_scoring = time.time()
        nb_tag_1 = 0.0
        nb_tag_5 = 0.0
        classes = lb.classes_
        no_top_words = 5
        for i in range(x_test.shape[0]):

            debut_boucle = time.time()
            text_projection = x_test
            if(mode_supervise_with_lda):
                text_projection = lda_model.transform(x_test[i])
            else:
                text_projection = x_test.values[i]
                text_projection = [text_projection]
            
            debut_prediction = time.time()
            predicted = clf.predict_proba(text_projection)
            fin_prediction = time.time()
            del debut_prediction, fin_prediction

            debut_generation_tag = time.time()
            tempTag = [(1-item[0][0]) for item in predicted]
            list_id = [[i, x] for i, x in enumerate(tempTag) if x > 0.0050]
            list_id_sorted = sorted(list_id, reverse=True, key=lambda x: x[1])
            list_id_sorted_suggested = [x[0] for i, x in enumerate(list_id_sorted[:-no_top_words - 1:-1])]
            
            prediction = [classes[id] for id in list_id_sorted_suggested]
            fin_generation_tag = time.time()
            
            del debut_generation_tag, fin_generation_tag

            debut_y = time.time()
            l_y = [[i, x] for i, x in enumerate(y_true[i]) if x > 0]
            l_y_tagged = [x[0] for i, x in enumerate(l_y[:-no_top_words - 1:-1])]
            l_y_tags = [classes[id] for id in l_y_tagged]
            fin_y = time.time()
            del debut_y, fin_y

            debut_ch1 = time.time()
            check_1 = False
            check_1 = any(item in prediction for item in l_y_tags)

            if check_1 is True:
                nb_tag_1 = nb_tag_1 + 1
            fin_ch1 = time.time()
            del debut_ch1, fin_ch1

            debut_ch5 = time.time()
            check_5 = False
            check_5 = all(item in prediction for item in l_y_tags)
            if check_5 is True:
                nb_tag_5 = nb_tag_5 + 1

            fin_ch5 = time.time()
            del debut_ch5, fin_ch5
            fin_boucle = time.time()
            del fin_boucle, debut_boucle
        fin_scoring = time.time()
        del fin_scoring, debut_scoring
        return nb_tag_1, (100.0 * nb_tag_1 / float(x_test.shape[0])), nb_tag_5, (100.0 * nb_tag_5 / float(x_test.shape[0]))