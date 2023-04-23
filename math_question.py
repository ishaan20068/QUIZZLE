from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Create a new ChromeDriver instance
options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(options=options)

def generate_link(category, topic):
    pg_lnk = 'https://www.wolframalpha.com/problem-generator/quiz/?category=' + category + '&topic=' + topic
    driver.get(pg_lnk)

    # Wait for the page to load
    try:
        wait = WebDriverWait(driver, 4)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'blahblah')))
    except:
        print("TimedOut")

    page_source = driver.page_source
    
    # print(page_source)

    # Extract ww6 from the page source
    ww6 = 'https://www6b3.wolframalpha.com/' + page_source.split('https://www6b3.wolframalpha.com/')[1]

    ww6 = ww6.replace('amp;', '')
    return ww6

d={"Number%20theory":["IntegerFactorization","IntegerFactors","GCDLCMSummary","RelativelyPrimeTest"],
   "Algebra":["ComplexAlgebraSummary","PolynomialAlgebraSummary","QuadraticPolynomialSummary","SystemsOf2Equations",""],
   "Calculus":["RationalDiscontinuities","DerivativeSummary","IntegrateSummary"],
   "Linear%20algebra":["2DVectorSummary","3DVectorSummary","3by3MatrixSummary",'OtherMatrixSummary'],
   "Statistics":["StatisticsSummary"]} 
link_dictionary={}
try:
    with open("math1.txt", "w") as file:
        for i in d.keys():
            for j in d[i]:
                file.write(i+' '+ (j + '\n'))
                link_dictionary[(i,j)]=[]
                for k in range(15):
                    try:
                        link = generate_link(i,j)
                    except:
                        continue
                    link_dictionary[(i,j)].append(link)
                    file.write(link + '\n')
                link_dictionary[(i,j)]=list(set(link_dictionary[(i,j)]))
                file.write('\n')
                file.flush()
except:
    print("Error")
l_final=[]
print(link_dictionary)
print('------------------------------------------')
""" 
#Print it on a file with headings as the tuple in dictionary keys and the links as the list of values
with open("math.txt", "w") as file:
    for i in link_dictionary:
        file.write(str(i) + '\n')
        for j in link_dictionary[i]:
            file.write(j + '\n')
        file.write('\n') """