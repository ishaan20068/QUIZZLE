from flask import Flask , render_template, url_for, request,redirect
print("Importing libraries...")
print("This will take time. Please wait...")

import requests
from bs4 import BeautifulSoup
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
import openai
import MCQ_GENERATION as mcq
import Descriptive_Ques as dq
import fill_ups as fp
import true_false_generator as tf
global c

openai.api_key = "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"

def start_generating_quiz(quiz_content,key):
    s2v=Sense2Vec().from_disk("s2v_old")
    
    nlp = spacy.load('en_core_web_md')
    doc = nlp(quiz_content)
    f=FreqDist(brown.words())
    k= mcq.find_keys(nlp,quiz_content,10,s2v,f,10)
    sentences=sent_tokenize(quiz_content)
    ksm= mcq.generate_sentences(k, sentences)
    for tempvar in ksm.keys():
        text_snippet = " ".join(ksm[tempvar][:3])
        ksm[tempvar] = text_snippet
    if key==1:
        quiz= mcq.MCQs_formulate(ksm,s2v)
    elif key==2:
        quiz= dq.generate_descriptive(ksm)
    elif key==3:
        quiz=fp.generate_fillups(ksm)
    elif key==4:
        quiz=tf.generate_true_false(ksm)
    return quiz
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('b_index.html')
@app.route("/quiz")
def quiz():
    return render_template('quiz.html')

@app.route("/uploading_text")
def uploading_text():
    return render_template('uploading_text.html')

@app.route("/quiz_using_maths")
def quiz_using_maths():
    return render_template('quiz_using_maths.html')

@app.route("/quiz_using_maths", methods=['POST'])
def upp():
    n_q = request.form['NoOfQuestions']
    print('--------------->',n_q)
    
    return 1

@app.route("/uploading_text", methods=['POST'])
def up():
    t=request.form["text"]
    if request.form["fruit"]=="apple":
        d=start_generating_quiz(t,1)
        global c
        c=[]
        for k in d:
            c.append((k["question"],"Answer = "+k["answer"],k["choices"][0],k["choices"][1],k["choices"][2],k["choices"][3]))
        return render_template("mcq.html",e=c)
    elif request.form["fruit"]=="mango":
        d=start_generating_quiz(t,2)
        c=[]
        for k in d:
            c.append((k[0],"Answer = "+k[1]))
        return render_template("short.html",e=c)
    elif request.form["fruit"]=="guava":
        d=start_generating_quiz(t,3)
        c=[]
        for k in d:
            c.append((k[0],"Answer = "+k[1]))
    elif request.form["fruit"]=="banana":
        d=start_generating_quiz(t,4)
        c=[]
        for k in d:
            c.append((k[0],"Answer = "+k[1]))
        return render_template("short.html",e=c)
    return 1
@app.route("/quiz_using_keywords")
def quiz_using_keywords():
    return render_template('quiz_using_keywords.html')



def scrape(text):
    text=text.replace(" ","_")
    search_string="https://en.wikipedia.org/wiki/"+text
    webpage=BeautifulSoup((requests.get(search_string)).content, 'html.parser')
    list(webpage.children)
    sentences=webpage.find_all('p')
    final_text=""
    for i in range(len(sentences)):
        final_text+=sentences[i].get_text()
    return final_text
@app.route("/quiz_using_keywords", methods=['POST'])
def u():
    t=request.form["text"]
    t=scrape(t)
    if request.form["fruit"]=="apple":
        d=start_generating_quiz(t,1)
        c=[]
        for k in d:
            c.append((k["question"],"Answer = "+k["answer"],k["choices"][0],k["choices"][1],k["choices"][2],k["choices"][3]))
        return render_template("mcq.html",e=c)
    elif request.form["fruit"]=="mango":
        d=start_generating_quiz(t,2)
        c=[]
        for k in d:
            c.append((k[0],"Answer = "+k[1]))
        return render_template("short.html",e=c)
    elif request.form["fruit"]=="guava":
        d=start_generating_quiz(t,3)
        c=[]
        for k in d:
            c.append((k[0],"Answer = "+k[1]))
        return render_template("short.html",e=c)
    elif request.form["fruit"]=="banana":
        d=start_generating_quiz(t,4)
        c=[]
        for k in d:
            c.append((k[0],"Answer = "+k[1]))
        return render_template("short.html",e=c)
    return 1

@app.route("/uploading_lecture")
def uploading_lecture():
    return render_template('uploading_lecture.html')

from PyPDF2 import PdfReader
  

@app.route("/uploading_lecture", methods=['POST'])
def uu():
    t=request.files["fileToUpload"]
    reader = PdfReader(t)
    page = reader.pages[0]
    text = page.extract_text()
    with open('lecture02-intro-boolean.txt', 'wb') as f:
        for i in range(len(reader.pages)):
           page = reader.pages[i]
           text = text + page.extract_text() + '\n'+'\n'
        f.write(text.encode('utf-8'))
    if request.form["fruit"]=="apple":
        d=start_generating_quiz(text,1)
        c=[]
        for k in d:
            c.append((k["question"],"Answer = "+k["answer"],k["choices"][0],k["choices"][1],k["choices"][2],k["choices"][3]))
        return render_template("mcq.html",e=c)
    elif request.form["fruit"]=="mango":
        d=start_generating_quiz(text,2)
        c=[]
        for k in d:
            c.append((k[0],"Answer = "+k[1]))
        return render_template("short.html",e=c)
    elif request.form["fruit"]=="guava":
        d=start_generating_quiz(text,3)
        c=[]
        for k in d:
            c.append((k[0],"Answer = "+k[1]))
        return render_template("short.html",e=c)
    elif request.form["fruit"]=="banana":
        d=start_generating_quiz(t,4)
        c=[]
        for k in d:
            c.append((k[0],"Answer = "+k[1]))
        return render_template("short.html",e=c)
    return 1
if __name__=="__main__":
    app.run(debug=True)

