import subprocess
import os
import gc
import multiprocessing
import tqdm
import xml.sax

import pandas as pd

from ProcessArticle import ProcessArticle
from WikiXmlHandler import WikiXmlHandler
from timeit import default_timer as timer

# List of lists to single list
from itertools import chain
# Sending keyword arguments in map
# from functools import partial

infoboxes = pd.read_csv(
    "./saved/infobox-list.csv")["infobox template name"].values.tolist()
folder = "./data"
files = [folder + "/" + x for x in os.listdir(folder)]
out_file = "./saved/out_parallel.csv"

func = ProcessArticle(infoboxes)

# lock
l = multiprocessing.Lock()


def read_articles(args):
  data_path, limit = args

  handler = WikiXmlHandler(out_file, func, None)

  # parse obj
  parser = xml.sax.make_parser()
  parser.setContentHandler(handler)

  l.acquire()
  print("[Started] Process PID: {}. File: {}\n".format(
      multiprocessing.current_process().pid, data_path))
  l.release()

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
  l.acquire()

  # write file
  handler.write_df()

  print("[Finished] Process PID: {}, articles readed {}. File: {}\n".format(
      multiprocessing.current_process().pid, handler._counter, data_path))

  # release lock
  l.release()

  del handler
  del parser
  gc.collect()
  return None


if __name__ == "__main__":
  if not os.path.exists(out_file):
    pd.DataFrame(columns=func.cols).to_csv(
        out_file, header=True, index=False, sep="\t")

  # Create a pool of workers to execute processes
  pool = multiprocessing.Pool(processes=2)

  start = timer()

  # Map (service, tasks), applies function to each partition
  f = [(x, None) for x in files]
  results = pool.map_async(read_articles, f)

  pool.close()
  pool.join()

  end = timer()
  print("{} seconds elapsed.".format(end - start))
