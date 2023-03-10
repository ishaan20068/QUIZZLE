import string
from flashtext import KeywordProcessor
from sense2vec import Sense2Vec
from collections import OrderedDict
from nltk.tokenize import sent_tokenize
from nltk import FreqDist
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from nltk.corpus import stopwords
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import pke.unsupervised
import spacy


def is_possible(element,sense_object):
    '''
    Takes in a word nd tells the sense of the word. Example- boy-NOUN, John-PROPN(proper noun)
    '''
    element = element.replace(" ", "_")
    value = sense_object.get_best_sense(element)
    return (not (value==None))


def one_apart(token):
    '''
    takes in a string and returns all the possible strings with an edit distance of one 
    '''
    alphabets_and_punctuations = string.ascii_lowercase + " " + string.punctuation
    
    final_tokens = []
    for i in range(len(token)+1):    # splits token into all possible pairs of left and right substrings
        left = token[:i]
        right = token[i:]
        
        if len(right) != 0:     # list of words by deleting each non-empty right substring of a given input word
            fin_token = left + right[1:]
            final_tokens.append(fin_token)

        if len(right) > 1:      # list of words by swapping the adjacent character in every non-empty right substring of a given word
            fin_token = left + right[1] + right[0] + right[2:]
            final_tokens.append(fin_token)
        
        if len(right) != 0:     # list of words by replacing each character in every non-empty right substring of a given word
            for alpha_punct in alphabets_and_punctuations:
                fin_token = left + alpha_punct + right[1:]
                final_tokens.append(fin_token)
        
        for alphabet in alphabets_and_punctuations:     # list of words by inserting each character in every possible position of every non-empty right substring of a given word
            fin_token = left + alphabet + right
            final_tokens.append(fin_token)
    
    return set(final_tokens)

def words_similar(token, sense_object):
    '''
    Takes a word and returns all the words in the database that are similar to the given word
    with the same sense. In case of multiple words it returns at most 15 words ordered in 
    descending order by their closeness. The best match word comes before.
    '''
    result = []
    processed_tokens = token.translate(token.maketrans("","", string.punctuation)).lower()

    token_edits = one_apart(processed_tokens)
    token = token.replace(" ", "_")
    sense = sense_object.get_best_sense(token)

    if sense == None:
        return None, "None"
    
    similar_tokens = sense_object.most_similar(sense, n = 15)
    cmp_lst = []
    cmp_lst.append(processed_tokens)

    for each_token in similar_tokens:
        temp = each_token[0].split("|")[0].replace("_", " ")
        temp = temp.strip().lower()
        temp = temp.translate(temp.maketrans("","", string.punctuation))
        if temp not in cmp_lst:
            if processed_tokens not in temp:
                if temp not in token_edits:
                    result.append(temp.title())
                    cmp_lst.append(temp)
    
    out = list(OrderedDict.fromkeys(result))

    return out, "sense2vec"


def para_to_sentences(text):
    ''' 
    function takes text as a input string and performs string tokenization on it
    and returns the list of sentences
    '''
    sent = sent_tokenize(text)
    fin_sent = []
    for i in sent:
        if len(i) > 20:
            fin_sent.append(i)
    
    return fin_sent


def distance(elements,this_item,max_val):
    '''
    Calculates the normalized edit distance between the currentword and elements of words_list.
    if all the words have a distance greater than a threshhold then returns true else false
    '''
    min_val=1000000000
    for x in elements:
        min_val=min(min_val,NormalizedLevenshtein().distance(x.lower(),this_item.lower()))
    return min_val>=max_val


def generate_sentences(k, s):
    kp = KeywordProcessor()
    ans = {}
    #initializing dictionary
    for i in k:
        i = i.strip()
        kp.add_keyword(i)
        ans[i] = []
    #extracting
    for i in s:
        ext = kp.extract_keywords(i)
        for j in ext:
            ans[j].append(i)
    #sorting and storing
    for i in ans.keys():
        ans[i] = sorted(ans[i], key=len, reverse=True)
    #deleting unnecessary keys
    dele=[k for k in ans.keys() if len(ans[k])==0]
    for i in dele:
      del ans[i]

    return ans
