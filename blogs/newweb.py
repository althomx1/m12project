import re

from bs4 import BeautifulSoup
from selenium.webdriver.firefox.options import Options
from selenium import webdriver
import time
import pandas as pd

pattern = r'(?<![A-Z][a-z]\.)(?<!\w\.\w.)(?<!\d\.)(?<=\.|\?|!)\s+(?=[A-Z]|\d)'
pattern = re.compile(pattern)



url = 'https://thehackernews.com/search/label/Vulnerability'
options = Options()
options.add_argument("--headless")
driver = webdriver.Firefox(options=options)

driver.get(url)
# wait to load the page
time.sleep(2)

blog_posts = []
current_page = 1
all_links = []
while current_page < 11:

    html_page = driver.page_source

    soup = BeautifulSoup(html_page, "html.parser")

    links = soup.find_all("a", class_="story-link")

    for link in links:
        href = link['href']
        all_links.append(href)

    next_button_link = soup.find("a", class_="blog-pager-older-link-mobile")

    next_button_href = next_button_link['href']

    driver.get(next_button_href)

    current_page += 1


for link in all_links:


    driver.get(link)
    source = driver.page_source
    soup2 = BeautifulSoup(source, "html.parser")
    article = soup2.find("div", id="articlebody")


    if article is None:
        continue

    text = article.find_all("p")
    sentences = []
    for para in text:
        for a in para.find_all("a"):
            inside_text = a.get_text()
            a.replace_with(inside_text)
        paragraphs = para.get_text()
        sentences.extend(re.split(pattern, paragraphs))
    cleaned_sentences = [sentence.strip() for sentence in sentences if sentence]


    for sentence in cleaned_sentences:
        blog_posts.append(
            {
                "link": link,
                "data": sentence
            }
        )
df = pd.DataFrame(blog_posts)
df.to_csv("blog posts6.csv", index=False)

print(df.head())
print(len(df))


















