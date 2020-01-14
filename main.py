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
nlp = stanfordnlp.Pipeline(models_dir="./stanfordnlp_resources/", use_gpu=False)
func = ProcessArticle(infoboxes, nlp)

# lock
l_articles = multiprocessing.Lock()
l_wikidata = multiprocessing.Lock()


def read_wikidata(args):
  data_path, output_file = args

  handler = WikiDataXmlHandler()
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
    except StopIteration:
      break

  # acquire lock to write file
  l_wikidata.acquire()

  # write file
  handler.write_df(output_file)

  print("[Finished] Process PID: {}, wikidata readed {}. File: {}\n".format(
      multiprocessing.current_process().pid, handler._counter, data_path))

  # release lock
  l_wikidata.release()
  return None


def read_articles(args):
  data_path, limit = args

  handler = WikiArticleXmlHandler(out_file, func, None)

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
    except StopIteration:
      break

    if limit is not None and handler._counter >= limit:
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
    pd.DataFrame(columns=func.cols).to_csv(
        out_file + ".csv", header=True, index=False, sep="\t")

  if not os.path.exists(out_file + "_wikidata.csv"):
    pd.DataFrame(columns=w_cols).to_csv(
        out_file + "_wikidata.csv", header=True, index=False, sep="\t")

  # Create a pool of workers to execute processes
  pool1 = multiprocessing.Pool(processes=2)
  pool2 = multiprocessing.Pool(processes=2)

  start = timer()

  # Map (service, tasks), applies function to each partition
  f = [(x, None) for x in files_a]
  pool1.map_async(read_articles, f)

  f = [(x, None) for x in files_w]
  pool2.map_async(read_wikidata, f)

  pool1.close()
  pool2.close()
  pool1.join()
  pool2.join()

  end = timer()
  print("{} seconds elapsed.".format(end - start))
