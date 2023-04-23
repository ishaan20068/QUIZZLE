import MCQ_GENERATION as mcq
import spacy
import sys
import openai
from nltk import FreqDist
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize
from sense2vec import Sense2Vec
from collections import OrderedDict

openai.api_key = "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"

def generate_short_questions(context):
    input_prompt = "Sentence: " + context
    template="""
    Sentence: Pandemic is a disease that spreads in a large area. One such pandemic is the COVID-19 pandemic. It was caused by the SARS-CoV-2 virus. It was first identified in December 2019 in Wuhan, China. It has since spread worldwide, leading to an ongoing pandemic. "
    Question: What is COVID-19? Answer: COVID-19 is a disease caused by the SARS-CoV-2 virus which was first identified in December 2019 in Wuhan, China.
    Sentence: Newton's laws of motion are three physical laws that, together, laid the foundation for classical mechanics. They describe the relationship between a body and the forces acting upon it, and its motion in response to those forces. 
    Question: What are Newton's laws of motion? Answer: There are three laws of motion that were developed by Sir Isaac Newton, which describe how objects move when they are acted upon by various forces.
    Sentence: ChatGPT is an advanced AI chatbot that can answer questions about any topic. It uses the latest technology to provide you with a natural conversation experience.
    Question: How is ChatGPT helpful? Answer: ChAtGPT helps you find answers to your questions easily, just like chatting, by using artificial intelligence and natural language processing.
    """
    prompt = template + input_prompt 
    completion = openai.Completion.create(engine="davinci", 
                                    prompt=prompt, 
                                    max_tokens=64, 
                                    temperature=0.7)

    message = completion.choices[0].text
    output_list = message.split("\n")
    out_index = []
    for idx, sentence in enumerate(output_list):
        if "Question" in sentence:
            out_index.append(idx)
    
    if out_index:
        return output_list[min(out_index)]

def generate_descriptive(ksm, n_questions=4):
    """A function to genereate descriptive questions from a given text snippet."""
    descriptive_questions = []
    questions = []
    while len(descriptive_questions) < n_questions:
        for term, text in ksm.items():
            ques_ans = generate_short_questions(context=text)
            #Answer is the keyword after "Answer:" and question is the sentence after "Question:" and before "?". Check if they exist and extract them.
            if ques_ans is not None and "Answer:" in ques_ans and "Question:" in ques_ans:
                answer = ques_ans.split("Answer: ")[1]
                question = ques_ans.split("Question: ")[1].split("?")[0]
                #Check if question is already present in the list of questions. If not, append it. Ignore cases
                if question.lower() in [q.lower() for q in questions]:
                    print("Question already present", question)
                    continue
            else:
                continue
                # answer = generate_questions(question, "")
            print("\n\n")
            print("#############################################")
            print(text)
            print(question)
            print("------------>Answer:",answer, "length: ", len(answer.split()))
            print("#############################################")
            descriptive_questions.append((question, answer))
    return descriptive_questions
