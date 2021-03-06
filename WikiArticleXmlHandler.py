import xml.sax
import ProcessArticle as pa
import pandas as pd
import os


class WikiArticleXmlHandler(xml.sax.handler.ContentHandler):
  """Content handler for Wiki XML data using SAX"""

  def __init__(self, fileout, func):
    xml.sax.handler.ContentHandler.__init__(self)
    self._buffer = None
    self._values = {}
    self._current_tag = None
    self._counter = 0
    self._redirect = False

    # self._data = {col: [] for col in pa.cols}
    self._data = list()
    self._pa = func
    self._fileout = fileout

    self._ready = False

    # create file to store the parsed information
    # if write_after != None:
    #   if not os.path.exists(fileout):
    #     pd.DataFrame(columns=self._pa.cols).to_csv(
    #         self._fileout, header=True, index=False, sep="\t")

    # save text for debug purposes
    self._text = []
    self._DEBUG = False

  def characters(self, content):
    """Characters between opening and closing tags"""
    if self._current_tag:
      self._buffer.append(content)

  def startElement(self, name, attrs):
    """Opening tag of element"""
    self._ready = False
    if name in ('title', 'text', 'timestamp'):
      self._buffer = []
      self._current_tag = name

    elif name == "redirect":
      # self._type = attrs.get("title")
      self._redirect = True

  def write_df(self):
    pd.DataFrame(
        data=self._data, columns=self._pa.cols).to_csv(
            self._fileout, mode="a", header=False, index=False, sep="\t")
    print("Saved to csv, current number of articles <{}>".format(self._counter))
    # remove data and rebuild
    del self._data
    self._data = list()

  def clear(self):
    self._buffer = None
    self._values = {}
    self._current_tag = None
    self._redirect = False

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
          if self._DEBUG:
            self._text.append(self._values["text"])

          self._data.append(lst)
          self._counter += 1

          self._ready = True
          # if self._write_after != None and not self._DEBUG:
          #   if (self._counter % self._write_after) == 0:
          #     self.write_df()

      self._redirect = False
