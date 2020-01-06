import xml.sax
import ProcessArticle as pa
import pandas as pd
import os

class WikiXmlHandler(xml.sax.handler.ContentHandler):
  """Content handler for Wiki XML data using SAX"""

  def __init__(self, fileout, infoboxFile):
    xml.sax.handler.ContentHandler.__init__(self)
    self._buffer = None
    self._values = {}
    self._current_tag = None
    self._counter = 0
    self._redirect = None

    pa.init(infoboxFile)
    self._fileout = fileout
    # create file to store the parsed information
    if not os.path.exists(fileout):
      with open(fileout, "w"): pass

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
      self._redirect = attrs.get("title")
      # self._redirect = True

  def endElement(self, name):
    """Closing tag of element"""
    if name == self._current_tag:
      self._values[name] = ' '.join(self._buffer)

    if name == 'page':
      # self._pages.append((self._values['title'], self._values['text'], self._redirect))
      # when the reader finish reding, process da'beach
      df = pa.process_article(
          [self._values["title"], self._values["text"], self._redirect])
      
      #logic: get a lock to write to the file
      df.to_csv(self._fileout, mode="a", header=False, index=False)

      self._counter += 1
      self._redirect = None
