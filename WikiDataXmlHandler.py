import xml.sax
import os
import pandas as pd
import json


class WikiDataXmlHandler(xml.sax.handler.ContentHandler):
  """Content handler for Wikidata XML data using SAX"""

  def __init__(self, fileout):
    xml.sax.handler.ContentHandler.__init__(self)
    self._buffer = None
    self._values = {}
    self._current_tag = None
    self._counter = 0
    self._redirect = False

    # self._write_after = write_after
    self._fileout = fileout

    self._data = {"title": [], "label": [], "type": []}

    self._ready = False

  def characters(self, content):
    """Characters between opening and closing tags"""
    if self._current_tag:
      self._buffer.append(content)

  def write_df(self):
    pd.DataFrame(data=self._data).to_csv(
        self._fileout, mode="a", header=False, index=False, sep="\t")

    print("Saved to csv, current number of wikidata items <{}>".format(self._counter))
    # remove data and rebuild
    del self._data
    self._data = {"title": [], "label": [], "type": []}

  def startElement(self, name, attrs):
    """Opening tag of element"""
    self._ready = False
    
    if name in ('title', 'text'):
      self._buffer = []
      self._current_tag = name

  def endElement(self, name):
    """Closing tag of element"""
    if name == self._current_tag:
      self._values[name] = ''.join(self._buffer)

    if name == 'page' and self._values['title'].startswith("Q"):
      j = json.loads(self._values['text'])

      if "labels" not in j.keys():
        # ignore pages without label
        # print("Label key not found in {}".format(self._values['title']))
        return
      elif "en" not in j["labels"].keys():
        # ignore pages without english labels
        # print("No english (en) available for {}".format(self._values['title']))
        return

      ran = 0
      k = "P31"

      if isinstance(j["claims"], list):
        # print("it's a list (wtf?) {}".format(self._values['title']))
        return

      if "P31" in j["claims"].keys():
        ran = len(j["claims"]["P31"])
        k = "P31"
      elif "P279" in j["claims"].keys():
        ran = len(j["claims"]["P279"])
        k = "P279"
      elif "P361" in j["claims"].keys():
        ran = len(j["claims"]["P361"])
        k = "P361"
      elif "Q5299" in j["claims"].keys():
        ran = len(j["claims"]["P361"])
        k = "P361"
      else:
        print("No type key in: " + self._values['title'])
        k = None

      self._data["title"].append(self._values['title'])
      self._data["label"].append(j["labels"]["en"]["value"])

      tmp = []
      if k != None:
        for r in range(ran):
          try:
            tmp.append(
                j["claims"][k][r]["mainsnak"]["datavalue"]["value"]["id"])
          except Exception:
            print("ID or datavalue not found {}".format(self._values['title']))
            continue
      else:
        tmp.append("-")

      self._data["type"].append("|".join(tmp))

      self._counter += 1

      # sending a signal saying that it is safe to write to the file
      self._ready = True
      # if (self._counter % self._write_after) == 0:
        # self.write_df()
