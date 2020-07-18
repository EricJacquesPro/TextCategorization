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
    nmf_filename = 'nmf.joblib'
    nmf_df_filename = 'nmf_df_topic_keywords.joblib'
    nmf_tf_filename = 'nmf_tf_vectorizer.joblib'
    supervised_classifier_filename = 'supervised_classifier.joblib'
    supervised_classes_filename = 'supervised_classes.joblib'
    nombre_post_entree = 50000
    precision = 50000
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
        data_preprocessed, 
        topic_number_min, 
        topic_number_max, 
        topic_number_step
    ):
        documents = data_preprocessed[0:self.precision].unique()
        import numpy as np
        from sklearn import metrics
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        
        Y_all = np.zeros(len(documents))
        
        documents_train, documents_test, y_train, y_test = self.train_test_split(X=list(documents), y=Y_all, test_size=0.33)
        
        lda_tf, lda_tf_vectorizer = self.lda_init(documents_train)
        lda_tf_test, lda_tf_vectorizer_test = self.lda_init(documents_train)
        
        performance_indicateurs=[]
        performance_score_indicateurs=[]
        performance_score_validation_indicateurs=[]
        for no_tropics in range(topic_number_min, topic_number_max, topic_number_step):
            lda, score, perplexity = self.lda_train(lda_tf, no_tropics)
            score_validation = lda.score(lda_tf_test)
            performance_score_indicateurs.append(score)
            performance_score_validation_indicateurs.append(score_validation)
            del lda, score, perplexity, score_validation
        return performance_score_indicateurs, performance_score_validation_indicateurs;
    '''
    def lda_find_best_topic_number(self, data_preprocessed):
        
        documents = data_preprocessed[0:self.precision].unique()
        lda_tf, lda_tf_vectorizer = self.lda_init(documents)
        
        # Define Search Param
        search_params = {'n_components': [2,4,6,8,10,12,14, 16,18,20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40], 'learning_decay': [.5, .7, .9]}

        # Init the Model
        lda = self.LatentDirichletAllocation()

        # Init Grid Search Class
        model = self.GridSearchCV(lda, param_grid=search_params)

        # Do the Grid Search
        model.fit(lda_tf)
        
        # Best Model
        best_lda_model = model.best_estimator_

        # Model Parameters
        print("Best Model's Params: ", model.best_params_)

        # Log Likelihood Score
        print("Best Log Likelihood Score: ", model.best_score_)

        # Perplexity
        print("Model Perplexity: ", best_lda_model.perplexity(lda_tf))
        
        # Get Log Likelyhoods from Grid Search Output
        n_topics = [2,4,6,8,10,12,14, 16,18,20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
        log_likelyhoods_5 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.5]
        log_likelyhoods_7 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.7]
        log_likelyhoods_9 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.9]

        # Show graph
        self.plt.figure(figsize=(12, 8))
        self.plt.plot(n_topics, log_likelyhoods_5, label='0.5')
        self.plt.plot(n_topics, log_likelyhoods_7, label='0.7')
        self.plt.plot(n_topics, log_likelyhoods_9, label='0.9')
        self.plt.title("Choosing Optimal LDA Model")
        self.plt.xlabel("Num Topics")
        self.plt.ylabel("Log Likelyhood Scores")
        self.plt.legend(title='Learning decay', loc='best')
        self.plt.show()
    '''
    
    def lda_prepare_tag(self, data_preprocessed, no_tropics=32):
        '''
        prepare lda, topic ad tf vectorizer from data preprocessed
        '''
        documents = data_preprocessed[0:self.precision].unique()
        lda_tf, lda_tf_vectorizer = self.lda_init(documents)
        
        lda, score, perplexity = self.lda_train(lda_tf, no_tropics)
        
        print("Log Likelihood: ", score)
        print("Perplexity: ", perplexity)

        # See model parameters
        print(lda.get_params())

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
    '''
    def nmf_find_best_topic_number(self, data_preprocessed, topic_number_min=10, topic_number_max=40, topic_number_step=5, learning_decay_min=0.5, learning_decay_max=1, learning_decay_step=0.2):
        n_topics = range(topic_number_min, topic_number_max, topic_number_step)
        documents = data_preprocessed[0:self.precision].unique()
        
        nmf_tfidf, nmf_tfidf_vectorizer = self.nmf_init(documents)
        
        # Define Search Param
        search_params = {'n_components': n_topics}#[10, 15, 20, 25, 30]}

        # Init the Model
        nmf = self.NMF(
            random_state=1,
            alpha=.1,
            l1_ratio=.5,
            init='nndsvd'
        )
        lda = self.LatentDirichletAllocation()

        # Init Grid Search Class
        model = self.GridSearchCV(nmf, param_grid=search_params)

        # Do the Grid Search
        model.fit(nmf_tfidf)
        
        # Best Model
        best_nmf_model = model.best_estimator_

        # Model Parameters
        print("Best Model's Params: ", model.best_params_)

        # Log Likelihood Score
        print("Best Log Likelihood Score: ", model.best_score_)

        # Perplexity
        print("Model Perplexity: ", best_lda_model.perplexity(nmf_tfidf))
        
        # Get Log Likelyhoods from Grid Search Output
        #n_topics = [10, 15, 20, 25, 30]
        self.plt.figure(figsize=(12, 8))

        for learning_decay in range(learning_decay_min, learning_decay_max, learning_decay_step):
            log_likelyhoods = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==learning_decay]
            self.plt.plot(n_topics, log_likelyhoods, label=str(learning_decay))
        
        self.plt.title("Choosing Optimal LDA Model")
        self.plt.xlabel("Num Topics")
        self.plt.ylabel("Log Likelyhood Scores")
        self.plt.legend(title='Learning decay', loc='best')
        self.plt.show()
    '''    
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
    '''
    def evaluation_analysis(self, true_label, predicted): 
        from sklearn import metrics
        print ("accuracy: {}".format(metrics.accuracy_score(true_label, predicted)))
        print ("f1 score macro: {}".format(metrics.f1_score(true_label, predicted, average='macro')  ) )  
        print ("f1 score micro: {}".format(metrics.f1_score(true_label, predicted, average='micro') ))
        print ("precision score: {}".format(metrics.precision_score(true_label, predicted, average='macro'))) 
        print ("recall score: {}".format(metrics.recall_score(true_label, predicted, average='macro')) )
        print ("hamming_loss: {}".format(metrics.hamming_loss(true_label, predicted)))
        print ("classification_report: {}".format(metrics.classification_report(true_label, predicted)))
        print ("jaccard_similarity_score: {}".format( metrics.jaccard_similarity_score(true_label, predicted)))
        print ("log_loss: {}".format( metrics.log_loss(true_label, predicted)))
        print ("zero_one_loss: {}".format(metrics.zero_one_loss(true_label, predicted)))
        print ("AUC&ROC: {}".format(metrics.roc_auc_score(true_label, predicted)))
        print ("matthews_corrcoef: {}".format( metrics.matthews_corrcoef(true_label, predicted) ))
    '''    
        
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
        tempTag = [(1-item[0][0]) for item in predicted]
        #print(classes)
        #print([classes[id] for id in list_id])
        return str([classes[id] for id in list_id])
    
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
        
    
