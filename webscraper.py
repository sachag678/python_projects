import requests
from bs4 import BeautifulSoup

r = requests.get("https://en.wikipedia.org/wiki/Future")

soup = BeautifulSoup(r.content, "lxml")

dict1 = {}
for link in soup.find_all("a"):
		title = link.text
		dict1[title] = link.get("href")


