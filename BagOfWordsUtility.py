from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

class BagOfWordsUtility(object):
    '''a utility class for processing raw html text into segments for further reading'''
    
    @staticmethod
    def review_to_words(review, stopwords):
        #Function to convert a review into a sequence of words
        review_text = BeautifulSoup(review).get_text()
        review_text = re.sub("[^a-zA-Z]",
                             ' ',
                             review_text)
        words = review_text.lower().split()
        words = [w for w in words if w not in stopwords]
        return " ".join(words)
    
#     @staticmethod
#     def review_to_sentences(review, tokenizer, stopwords):
#         raw_sentences = tokenizer.tokenize(review.decode('utf-8').strip())
