print("Importing libraries...")
print("This will take time. Please wait...")

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
import sys
import random

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

def generate_choices(token, sense_object):
    '''
    Takes the token and sense_object as input and returns the list of possible choices for the token.
    Uses sense2vec to generate the choices, after that it uses the one_apart function to generate the choices. 
    Uses try and except to handle the exceptions arising from sense2vec library and spacy version incompatibility in most_similar function.
    '''
    choices = []
    try:
        processed_tokens = token.translate(token.maketrans("","", string.punctuation)).lower()
        token_edits = one_apart(processed_tokens)
        token = token.replace(" ", "_")
        sense = sense_object.get_best_sense(token)
        similar_tokens = sense_object.most_similar(sense, n = 15)
        corelated_words = []
        corelated_words.append(processed_tokens)

        for each_token in similar_tokens:
            temp = each_token[0]
            temp = temp.split("|")[0].replace("_", " ")
            temp = temp.strip()
            processed_token = temp.lower().translate(temp.maketrans("","", string.punctuation))
            if processed_token not in corelated_words:
                if processed_tokens not in processed_token:
                    if processed_token not in token_edits:
                        corelated_words.append(processed_token)
                        choices.append(temp.title())
        choices = OrderedDict.fromkeys(choices)
        choices = list(choices)

        if len(choices) > 0:
            print("Similar choices genertated for answer: ", token)
            return choices, "sense2vec"
    except:
        print("Similar choices could not be generated for answer: ", token)
    return choices, "None"



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


def select_main(ph_keys, max_phrases):
    """
    Filter a list of phrases to a maximum number based on a scoring metric.
    Args:
        ph_keys (list): A list of phrases to filter.
        max_phrases (int): The maximum number of phrases to return.
    Returns:
        list: The filtered list of phrases, containing at most max_phrases phrases.
    """
    ans = [ph_keys[0]] if len(ph_keys) > 0 else []
    for phrase in ph_keys[1:]:
        if distance(ans, phrase, 0.7) and len(ans) < max_phrases:
            ans.append(phrase)
        if len(ans) >= max_phrases:
            break
    return ans


def find_subjects(text):
    """
    Extract the top 10 noun and proper noun phrases from a text using the MultipartiteRank algorithm.
    Args:
        text (str): The input text.
    Returns:
        list: The top 10 noun and proper noun phrases extracted from the text.
    """
    ans = []

    try:
        e = pke.unsupervised.MultipartiteRank()
        e.load_document(input=text, language='en',stoplist=list(string.punctuation) + stopwords.words('english'))
        e.candidate_selection(pos={'PROPN', 'NOUN'})
        e.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyph = e.get_n_best(n=10)
        ans=[key[0] for key in keyph]
    except:
        pass

    return ans

def generate_phrasal_words(doc):
    """
    Extract up to 50 longest noun phrases from a spaCy document object.
    Args:
        doc (spacy.tokens.Doc): The spaCy document object.
    Returns:
        list: The up to 50 most frequent noun phrases in the document.
    """
    ph = {}
    for noun in doc.noun_chunks:
        p = noun.text
        if len(p.split()) > 1:
            if p not in ph:
                ph[p] = 1
            else:
                ph[p] += 1

    ph_keys = sorted(list(ph.keys()), key=lambda x: len(x), reverse=True)[:50]
    return ph_keys

def find_keys(nlp, text, max, s2v, fd, nos):
    """
    Extract up to max_keywords keywords from a given text using a combination of
    approaches, including MultipartiteRank, noun phrases, and filtering.
    Args:
        nlp : The  model to use for text processing.
        text (str): The input text to extract keywords from.
        max (int): The maximum number of keywords to return.
        s2v : The sense2vec model to use for filtering out irrelevant keywords.
        fd : A frequency distribution of words in the text.
        nos (int): The number of sentences in the input text.
    Returns:
        list: The up to max_keywords most relevant keywords extracted from the text.
    """

    tp = select_main(sorted(find_subjects(text), key=lambda x: fd[x]), int(max)) + select_main(generate_phrasal_words(nlp(text)), int(max))

    tpf = select_main(tp, min(int(max), 2*nos))

    ans = []
    for answer in tpf:
        if answer not in ans:
          if is_possible(answer, s2v):
            ans.append(answer)
    return ans[:int(max)]

def MCQs_formulate(keyword_sent_mapping, sense2vec):
    '''
    function takes keyword_sent_mapping as a dictionary and sense2vec model as input
    and returns the list of questions and answers in the form of a dictionary along with the choices and reason for the answer.
    Uses T5 model for question generation and spacy for answer extraction and choice generation, 
    and sense2vec for filtering out irrelevant keywords.
    Also uses nltk for sentence tokenization and wordnet for synonym generation.
    '''

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('Parth/result')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    questions = []
    
    for answer, text in keyword_sent_mapping.items():
        question_context = "context: " + text + " " + "answer: " + answer + " </s>"
        encoding = tokenizer.encode_plus(question_context, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(input_ids=encoding["input_ids"].to(device), 
                                    attention_mask=encoding["attention_mask"].to(device), 
                                    max_length=150)
        question = tokenizer.decode(output[0], skip_special_tokens=True).replace("question:", "").strip()
        choices, algorithm = generate_choices(answer, sense2vec)
        #diffiiculty level can be set here based on algorithm used
        if(len(choices) < 3):
            continue
        additional_choices = choices[3:]
        choices = choices[:3]
        choices.append(answer)
        random.shuffle(choices)
        question_data = {
            "question": question,
            "answer": answer,
            "choices": choices,
            "additional_choices": additional_choices,
            "reasoning": text
        }
        questions.append(question_data)
        
    return questions

def start_generating_quiz(quiz_content):
    s2v=Sense2Vec().from_disk("s2v_old")
    
    nlp = spacy.load('en_core_web_md')
    doc = nlp(quiz_content)
    f=FreqDist(brown.words())
    k= find_keys(nlp,quiz_content,10,s2v,f,10)
    sentences=sent_tokenize(quiz_content)
    ksm=generate_sentences(k, sentences)
    for tempvar in ksm.keys():
        text_snippet = " ".join(ksm[tempvar][:3])
        ksm[tempvar] = text_snippet
    quiz=MCQs_formulate(ksm,s2v)
    return quiz

def interface():
    print("#"*100)
    print(" "*45+"QUIZZLE")
    print("Welcome to the content based quiz generator")
    #print("Paste the quiz content below")
    #quiz_content = input()
    quiz_content="Diophantus, the “father of algebra,” is best known for his book Arithmetica, a work on the solution of algebraic equations and the theory"
    quiz_content+=" of numbers. However, essentially nothing is known of his life, and"
    quiz_content+=" there has been much debate regarding precisely the years in which"
    quiz_content+=" he lived. Diophantus did his work in the great city of Alexandria. At"
    quiz_content+=" this time, Alexandria was the center of mathematical learning. The period "
    quiz_content+="from 250 bce to 350 ce in Alexandria is known as the Silver Age, also the Later "
    quiz_content+="Alexandrian Age. This was a time when mathematicians were discovering many ideas "
    quiz_content+="that led to our current conception of mathematics. The era is considered silver "
    quiz_content+="because it came after the Golden Age, a time of great development in the field "
    quiz_content+="of mathematics. This Golden Age encompasses the lifetime of Euclid."
    quiz = start_generating_quiz(quiz_content)
    print("Would you like to save the quiz content and questions to a file? (y/n)")
    save = input()
    with open("quiz.txt", "w") as f:
        if save == 'y':
            sys.stdout = f
        print("The quiz will be based on the following content")
        print(quiz_content)
        print("The quiz will have the following questions")
        choice_id = ord('a')
        print()
        for question in quiz:
            print("Q) ", question["question"])
            print("Choices are: ")
            for choice in question["choices"]:
                #print choice in a new line with alphabetical indexing
                print(chr(choice_id) + ". " + choice)
                choice_id += 1
            choice_id = ord('a')
            print("Additional choices are: ")
            for choice in question["additional_choices"]:
                print(choice, end = ", ")
            print()
            print("Correct answer is: ", question["answer"])
            print("Reasoning for the answer is: ")
            print(question["reasoning"])
            #generate a line across the screen to separate questions
            print("-"*100)
            print()
        sys.stdout = sys.__stdout__
    print("Thank you for generating the quiz")    
    print("#"*100)
    
interface()
