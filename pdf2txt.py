from PyPDF2 import PdfReader
  
# creating a pdf reader object
# reader = PdfReader('Sociology of India March 13th.pdf')
reader = PdfReader('templecnotes.pdf')
  
# printing number of pages in pdf file
print(len(reader.pages))
  
# getting a specific page from the pdf file
page = reader.pages[0]
  
# extracting text from page
text = page.extract_text()
print(text)

#save to file
with open('lecture02-intro-boolean.txt', 'wb') as f:
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        text = text + page.extract_text() + '\n'+'\n'
    f.write(text.encode('utf-8'))
