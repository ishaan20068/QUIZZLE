import requests
from bs4 import BeautifulSoup

def scrape(text):
    text=text.replace(" ","_")
    search_string="https://en.wikipedia.org/wiki/"+text
    print(search_string)
    webpage=BeautifulSoup((requests.get(search_string)).content, 'html.parser')
    list(webpage.children)
    sentences=webpage.find_all('p')
    final_text=""
    for i in range(len(sentences)):
        final_text+=sentences[i].get_text()
    return final_text

print(scrape("photosynthesis"))
