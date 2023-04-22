from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Create a new ChromeDriver instance
driver = webdriver.Chrome()

# Navigate to the URL
driver.get('https://www.wolframalpha.com/problem-generator/quiz/?category=Calculus&topic=TrigSubIntegrate')

# Wait for the page to load
try:
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'blahblah')))
except:
    print("TimedOut")

page_source = driver.page_source
# Close the browser
driver.quit()
# print(page_source)

print(type(page_source))
