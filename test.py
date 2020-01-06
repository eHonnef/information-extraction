import requests
from bs4 import BeautifulSoup
import subprocess
import re
import os
import stanfordnlp


def download_articles(limit=None):
  # It's faster if you don't use "tensorflow backend". Import when necessary.
  from keras.utils import get_file

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
  for link in soup_dump.find_all(
      'a', {'href': re.compile(r".*pages-articles\d*\.xml-p.*\.bz2$")},
      limit=limit):
    path = file_folder + link["href"]
    if not os.path.exists(path):
      print("Downloading: " + link["href"])
      get_file(path, dump_url + link["href"])
      print("File size: " + str(os.stat(path).st_size / 1e6) + " MB")


def download_dictionary(language, location):
  stanfordnlp.download(language, resource_dir=location)


from WikiXmlHandler import WikiXmlHandler
import xml.sax
import mwparserfromhell


def categorize():

  files = os.listdir("./data")

  handler = WikiXmlHandler()
  parser = xml.sax.make_parser()
  parser.setContentHandler(handler)

  handler._pages
  # for file in files:
  for line in subprocess.Popen(["bzcat"],
                               stdin=open("data/" + files[0]),
                               stdout=subprocess.PIPE).stdout:
    if len(handler._pages) > 1:
      break

    parser.feed(line)

  # print the page title
  print(handler._pages[1][0])

  # parse the wiki page
  wiki = mwparserfromhell.parse(handler._pages[1][1])

  # print the page text (or at least its first 1000 words)
  # print(wiki.strip_code())

  # download_articles()

  nlp = stanfordnlp.Pipeline(
      models_dir="./stanfordnlp_resources/", use_gpu=False)
  doc = nlp(wiki.strip_code())
  for sentence in doc.sentences:
    print(sentence.print_dependencies())

categorize()
