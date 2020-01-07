import xml.sax
import ProcessArticle as pa
import pandas as pd
import os


class WikiXmlHandler(xml.sax.handler.ContentHandler):
  """Content handler for Wiki XML data using SAX"""

  def __init__(self, fileout, infoboxFile, write_after=100):
    xml.sax.handler.ContentHandler.__init__(self)
    self._buffer = None
    self._values = {}
    self._current_tag = None
    self._counter = 0
    self._redirect = False

    # self._data = {col: [] for col in pa.cols}
    self._data = list()
    self._pa = pa.ProcessArticle(infoboxFile)
    self._write_after = write_after
    self._fileout = fileout

    # create file to store the parsed information
    if not os.path.exists(fileout):
      pd.DataFrame(columns=self._pa.cols).to_csv(
          self._fileout, header=True, index=False, sep="\t")

    self._DEBUG = False

  def characters(self, content):
    """Characters between opening and closing tags"""
    if self._current_tag:
      self._buffer.append(content)

  def startElement(self, name, attrs):
    """Opening tag of element"""
    if name in ('title', 'text', 'timestamp'):
      self._buffer = []
      self._current_tag = name

    elif name == "redirect":
      # self._type = attrs.get("title")
      self._redirect = True

  def write_df(self):
    if self._DEBUG:
      return

    print("Saved to csv, current number of articles <{}>".format(self._counter))
    pd.DataFrame(
        data=self._data, columns=self._pa.cols).to_csv(
            self._fileout, mode="a", header=False, index=False, sep="\t")
    # remove data and rebuild
    del self._data
    self._data = list()

  def endElement(self, name):
    """Closing tag of element"""
    if name == self._current_tag:
      self._values[name] = ' '.join(self._buffer)

    if name == 'page':
      # self._pages.append((self._values['title'], self._values['text'], self._redirect))
      # when the reader finish reding, process da'beach
      if not self._redirect:
        lst = self._pa.process_article(
            [self._values["title"], self._values["text"]])

        if lst != None:
          # skip (list of / redirect) articles types
          # [self._data[pa.cols[i]].append(lst[i]) for i in range(len(lst))]
          self._data.append(lst)
          self._counter += 1

        # @TODO: logic: get a lock to write to the file (multiprocessing)
        if (self._counter % self._write_after) == 0:
          self.write_df()

      self._redirect = False
