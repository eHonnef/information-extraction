import requests
from bs4 import BeautifulSoup
import subprocess
import re
import os
import stanfordnlp
from keras.utils import get_file


def download_articles(limit=None):
  base_url = 'https://dumps.wikimedia.org/enwiki/'
  index = requests.get(base_url).text
  soup_index = BeautifulSoup(index, 'html.parser')  # Find the links on the page
  dumps = [a['href'] for a in soup_index.find_all('a') if a.has_attr('href')]

  dump_url = base_url + dumps[-1]
  dump_html = requests.get(dump_url).text

  # Convert to a soup
  soup_dump = BeautifulSoup(dump_html, 'html.parser')

  file_folder = "C:/projects/information-extraction/new/data/"
  # Downloading only the articles
  count = 0
  for link in soup_dump.find_all(
      'a', {'href': re.compile(r".*pages-articles\d*\.xml-p.*\.bz2$")},
      limit=limit):
    count += 1
    path = file_folder + link["href"]
    if not os.path.exists(path):
      print("Downloading: " + link["href"])
      # get_file(path, dump_url + link["href"])
      # print("File size: " + str(os.stat(path).st_size / 1e6) + " MB")
  
  print(count)


def download_dictionary(language, location):
  stanfordnlp.download(language, resource_dir=location)


# file = "./data/enwiki-latest-pages-articles15.xml-p7744803p9244803.bz2"

# for line in subprocess.Popen(["bzcat"], stdin=open(file), stdout=subprocess.PIPE).stdout:
#   print(line)

download_articles()
'''
nlp = stanfordnlp.Pipeline(models_dir="./stanfordnlp_resources/")
doc = nlp("I just wanted an ice-cream. Because she likes it.")
for sentence in doc.sentences:
  print(sentence.print_dependencies())'''