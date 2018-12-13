
''' This script is used to create word2vec corpus model from a folder containing extracted text from a given set of URLs.
We can create 2 different word2vec model using normal Python and Gensim. Create Gensim model by uncommenting the code '''

import os
#from gensim.models import Word2Vec
import word2vec
import codecs
import numpy as np
from nltk.corpus import stopwords


def mergeAllContents():            
	all_files = os.listdir("otherstotext/")
	big_f = open("all200Files.txt", "w")
	for i in all_files:
		f=open("otherstotext/"+str(i), "r")
		big_f.write(f.read())
	


def read_lines(file_lines):
	stop_words = set(stopwords.words('english'))
	print(stopwords)
	with open(file_lines) as f:
		content = f.readlines()
    	sentences = []
    	for line in content:
	        tokens = line.split()
	        for r in tokens:
	        	if not r in stop_words:
	        		sentences.append(tokens)
	return np.asarray(sentences)

mergeAllContents()

# # Building a model Using Gensim
# # define training data
# sentences = read_lines("all200Files.txt")
# # train model
# model = Word2Vec(sentences, min_count=100)
# # summarize the loaded model
# print(model)
# # summarize vocabulary
# words = list(model.wv.vocab)
# print(words)
# access vector for one word
# print(model['protection'])
# # save model
# model.save('ocean_gensim.bin')
# Loading Gensim Model
# new_model = Word2Vec.load('ocean_gensim.bin')

word2vec.word2phrase('all200Files.txt', 'ocean-full-phrases', verbose=True)
word2vec.word2vec('ocean-full-phrases', 'ocean.bin', size=500, verbose=True, min_count=5)
model = word2vec.load('ocean.bin',kind='bin', encoding = "ISO-8859-1")
word='ocean'
print(model[word])
print(model.vectors.shape)



