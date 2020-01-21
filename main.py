import subprocess
import os
import gc
import multiprocessing
import tqdm
import xml.sax

import pandas as pd

from ProcessArticle import ProcessArticle
from WikiArticleXmlHandler import WikiArticleXmlHandler
from WikiDataXmlHandler import WikiDataXmlHandler
from timeit import default_timer as timer

import stanfordnlp

import warnings
warnings.filterwarnings('ignore')

# List of lists to single list
from itertools import chain
# Sending keyword arguments in map
# from functools import partial

infoboxes = pd.read_csv(
    "./2.data/infobox-list.csv")["infobox template name"].values.tolist()

folder_a = "./0.articles"
files_a = [folder_a + "/" + x for x in os.listdir(folder_a)]

folder_w = "./1.wikidata"
files_w = [folder_w + "/" + x for x in os.listdir(folder_w)]
w_cols = ["title", "label", "type"]

out_file = "./saved/out_parallel"

n_write = 5000

# nlp = stanfordnlp.Pipeline(models_dir="./stanfordnlp_resources/", use_gpu=False)
# func = ProcessArticle(infoboxes, nlp)

# lock
l_articles = multiprocessing.Lock()
l_wikidata = multiprocessing.Lock()


def read_wikidata(args):
  data_path = args

  handler = WikiDataXmlHandler(out_file + "_wikidata.csv")
  # parse obj
  parser = xml.sax.make_parser()
  parser.setContentHandler(handler)

  l_wikidata.acquire()
  print("[Started] Process PID: {}. Wikidata: {}\n".format(
      multiprocessing.current_process().pid, data_path))
  l_wikidata.release()

  for line in subprocess.Popen(["bzcat"],
                               stdin=open(data_path),
                               stdout=subprocess.PIPE).stdout:
    try:
      parser.feed(line)

      if handler._ready and (handler._counter % n_write) == 0:
        l_wikidata.acquire()
        handler.write_df()
        l_wikidata.release()
    except StopIteration:
      break

  # acquire lock to write file
  l_wikidata.acquire()

  # write file
  handler.write_df()

  print("[Finished] Process PID: {}, wikidata readed {}. File: {}\n".format(
      multiprocessing.current_process().pid, handler._counter, data_path))

  # release lock
  l_wikidata.release()
  return None


def read_articles(args):
  data_path = args

  nlp = stanfordnlp.Pipeline(models_dir="./stanfordnlp_resources/", use_gpu=False)
  func = ProcessArticle(infoboxes, nlp)
  
  handler = WikiArticleXmlHandler(out_file + ".csv", func)

  # parse obj
  parser = xml.sax.make_parser()
  parser.setContentHandler(handler)

  l_articles.acquire()
  print("[Started] Process PID: {}. Article: {}\n".format(
      multiprocessing.current_process().pid, data_path))
  l_articles.release()

  for line in subprocess.Popen(["bzcat"],
                               stdin=open(data_path),
                               stdout=subprocess.PIPE).stdout:
    try:
      parser.feed(line)

      if handler._ready and (handler._counter % n_write) == 0:
        l_articles.acquire()
        handler.write_df()
        l_articles.release()
    except StopIteration:
      break

  # acquire lock to write file
  l_articles.acquire()

  # write file
  handler.write_df()

  print("[Finished] Process PID: {}, articles readed {}. File: {}\n".format(
      multiprocessing.current_process().pid, handler._counter, data_path))

  # release lock
  l_articles.release()

  del handler
  del parser
  gc.collect()
  return None


if __name__ == "__main__":
  if not os.path.exists(out_file + ".csv"):
    pd.DataFrame(columns=ProcessArticle.cols).to_csv(
        out_file + ".csv", header=True, index=False, sep="\t")

  if not os.path.exists(out_file + "_wikidata.csv"):
    pd.DataFrame(columns=w_cols).to_csv(
        out_file + "_wikidata.csv", header=True, index=False, sep="\t")

  # Create a pool of workers to execute processes
  pool1 = multiprocessing.Pool(processes=2)
  pool2 = multiprocessing.Pool(processes=2)

  start = timer()

  # Map (service, tasks), applies function to each partition
  # f = [(x, None) for x in files_a]
  pool1.map_async(read_articles, files_a)

  # f = [(x, None) for x in files_w]
  pool2.map_async(read_wikidata, files_w)

  pool1.close()
  pool2.close()
  pool1.join()
  pool2.join()

  end = timer()
  print("{} seconds elapsed.".format(end - start))
