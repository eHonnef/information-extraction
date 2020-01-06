import requests
from bs4 import BeautifulSoup
import re
import os
import stanfordnlp
import pandas as pd


def get_infobox_list(filename):
  url = "https://en.wikipedia.org/wiki/Wikipedia:List_of_infoboxes"
  dump_html = requests.get(url).text
  soup_dump = BeautifulSoup(dump_html, 'html.parser')

  df = pd.DataFrame([
      link.get_text().split(":")[1]
      for link in soup_dump.find_all("a",
                                     {"title": re.compile(r"^Template:.*")})
      if re.match("^Template:.*", link.get_text()) != None
  ],
                    columns=["infobox template name"])

  df.to_csv(filename, index=False)


def download_dictionary(language, location):
  stanfordnlp.download(language, resource_dir=location)


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
      'a',
      {'href': re.compile(r".*pages-articles-multistream\d*\.xml-p.*\.bz2$")},
      limit=limit):
    path = file_folder + link["href"]
    if not os.path.exists(path):
      print("Downloading: " + link["href"])
      get_file(path, dump_url + link["href"])
      print("File size: " + str(os.stat(path).st_size / 1e6) + " MB")