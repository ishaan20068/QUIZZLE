#Read m.txt and extract the top k links
import random

def question_math(category,topic,k):
    #Read m.txt and extract the top k links
    with open("math.txt", "r") as file:
        lines = file.readlines()
        links = []
        for i in range(len(lines)):
            if lines[i].strip() == str(category+' '+topic):
                for j in range(i+1, i+k+1):
                    links.append(lines[j].strip())
                break
    return links
l = question_math("numberTheory","IntegerFactorization",5)
for i in l:
    print(i)