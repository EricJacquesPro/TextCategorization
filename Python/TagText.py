class TagText:
	import os
	import nltk
	import numpy as np
	import pandas as pd
	import re                      # Regular expressions
	import sys

	from nltk import sent_tokenize, word_tokenize
	from nltk.tokenize import word_tokenize
	from nltk.corpus import stopwords
	from nltk.collocations import *
	from nltk.stem.snowball import SnowballStemmer
	from nltk.stem.wordnet import WordNetLemmatizer

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
