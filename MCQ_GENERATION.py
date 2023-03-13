import TEXT_PROCESSING
from sense2vec import Sense2Vec
from nltk.tokenize import sent_tokenize
from nltk import FreqDist
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from nltk.corpus import stopwords
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import pke.unsupervised
import spacy

s2v=Sense2Vec().from_disk("s2v_old")
text="Diophantus, the “father of algebra,” is best known for his book Arithmetica, a work on the solution of algebraic equations and the theory"
text+=" of numbers. However, essentially nothing is known of his life, and"
text+=" there has been much debate regarding precisely the years in which"
text+=" he lived. Diophantus did his work in the great city of Alexandria. At"
text+=" this time, Alexandria was the center of mathematical learning. The period "
text+="from 250 bce to 350 ce in Alexandria is known as the Silver Age, also the Later "
text+="Alexandrian Age. This was a time when mathematicians were discovering many ideas "
text+="that led to our current conception of mathematics. The era is considered silver "
text+="because it came after the Golden Age, a time of great development in the field "
text+="of mathematics. This Golden Age encompasses the lifetime of Euclid."
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
f=FreqDist(brown.words())
normalized_levenshtein = NormalizedLevenshtein()
k= find_keys(nlp,text,10,s2v,f,10)
sentences=sent_tokenize(text)
ksm=generate_sentences(k, sentences)
for tempvar in ksm.keys():
    text_snippet = " ".join(ksm[tempvar][:3])
    ksm[tempvar] = text_snippet
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('Parth/result')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
final=mcq(ksm,device,tokenizer,model,s2v, normalized_levenshtein)
print(final)
