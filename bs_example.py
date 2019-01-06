import requests
from bs4 import BeautifulSoup
import pprint
import re
import urllib.parse as urlparse
'''
# page = requests.get("http://dataquestio.github.io/web-scraping-pages/simple.html")
# print(page.status_code)

soup = BeautifulSoup(page.content, 'html.parser')

# print(list(soup.children))
# print([type(item) for item in list(soup.children)])

html = list(soup.children)[2]
# print(list(html.children))

body = list(html.children)[3]
p = list(body.children)[1]
# print(p.get_text())

pp = pprint.PrettyPrinter(indent=4)
for d in list(soup.find_all('div', class_='mw-indicator')):
    dd = list(d.children)
    print(dd)
    
'''
url = "https://www.ncbi.nlm.nih.gov/nuccore/321399722"
# parsed_url = urlparse.urlparse(url)
# gene_id = urlparse.parse_qs(parsed_url.query)['term']
# print(gene_id[0])

page = requests.get(url=url)
print(page.text)
# soup = BeautifulSoup(page.content, 'html.parser')
# print(soup.find_all('td'))



