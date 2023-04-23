
print("Importing libraries...")
print("This will take time. Please wait...")


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
import fill_ups as fups

openai.api_key = "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"

def start_generating_quiz(quiz_content, quiz_type):
    s2v=Sense2Vec().from_disk("s2v_old")
    
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(quiz_content)
    f=FreqDist(brown.words())
    k= mcq.find_keys(nlp,quiz_content,10,s2v,f,10)
    sentences=sent_tokenize(quiz_content)
    ksm= mcq.generate_sentences(k, sentences)
    for tempvar in ksm.keys():
        text_snippet = " ".join(ksm[tempvar][:3])
        ksm[tempvar] = text_snippet
    print("-------------------------")
    print(ksm)
    print("-------------------------")
    if quiz_type == 0:
        quiz= mcq.MCQs_formulate(ksm,s2v)
    elif quiz_type == 1:
        quiz = dq.generate_descriptive(ksm)
    elif quiz_type == 2:
        quiz = fups.generate_fillups(ksm)
    else:
        pass
    
    return quiz

def interface():
    print("#"*100)
    print(" "*45+"QUIZZLE")
    print("Welcome to the content based quiz generator")
    #print("Paste the quiz content below")
    
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
    quiz_content = "Photosynthesis definition states that the process exclusively takes place in the chloroplasts through photosynthetic pigments such as chlorophyll a, chlorophyll b, carotene and xanthophyll. All green plants and a few other autotrophic organisms utilize photosynthesis to synthesize nutrients by using carbon dioxide, water and sunlight. The by-product of the photosynthesis process is oxygen.Let us have a detailed look at the process, reaction and importance of photosynthesis. Photosynthesis reaction involves two reactants, carbon dioxide and water. These two reactants yield two products, namely, oxygen and glucose. Hence, the photosynthesis reaction is considered to be an endothermic reaction."
    quiz_content = "There is a lot of volcanic activity at divergent plate boundaries in the oceans. For example, many undersea volcanoes are found along the Mid-Atlantic Ridge. This is a divergent plate boundary that runs north-south through the middle of the Atlantic Ocean. As tectonic plates pull away from each other at a divergent plate boundary, they create deep fissures, or cracks, in the crust. Molten rock, called magma, erupts through these cracks onto Earth’s surface. At the surface, the molten rock is called lava. It cools and hardens, forming rock. Divergent plate boundaries also occur in the continental crust. Volcanoes form at these boundaries, but less often than in ocean crust. That’s because continental crust is thicker than oceanic crust. This makes it more difficult for molten rock to push up through the crust. Many volcanoes form along convergent plate boundaries where one tectonic plate is pulled down beneath another at a subduction zone. The leading edge of the plate melts as it is pulled into the mantle, forming magma that erupts as volcanoes. When a line of volcanoes forms along a subduction zone, they make up a volcanic arc. The edges of the Pacific plate are long subduction zones lined with volcanoes. This is why the Pacific rim is called the Pacific Ring of Fire."
    mcq_quiz = None
    desc_quiz = None
    fill_ups_quiz = None
    # mcq_quiz = start_generating_quiz(quiz_content, 0)
    # desc_quiz = start_generating_quiz(quiz_content, 1)
    fill_ups_quiz = start_generating_quiz(quiz_content, 2)
    print("Would you like to save the quiz content and questions to a file? (y/n)")
    save = input()
    with open("quiz.txt", "w") as f:
        if save == 'y':
            sys.stdout = f
        print("The quiz will be based on the following content")
        print(quiz_content)
        print("The quiz will have the following questions")
        if mcq_quiz is not None:
            choice_id = ord('a')
            print()
            for question in mcq_quiz:
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
                print("-"*100)
                print()
        if desc_quiz is not None:
            for question in desc_quiz:
                print("Q) ", question[0])
                print("Ans: ", question[1])
                print("-"*100)
                print()
        if fill_ups_quiz is not None:
            for question in fill_ups_quiz:
                print("Q) ", question[0])
                print("Ans: ", question[1])
                print("-"*100)
                print()
        sys.stdout = sys.__stdout__
    print("Thank you for generating the quiz")    
    print("#"*100)
    
interface()
