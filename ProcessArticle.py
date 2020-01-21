import pandas as pd
import mwparserfromhell
import Utils
import datetime


class ProcessArticle:
  cols = [
      "title", "type", "def words", "infobox_name", "infobox", "categories",
      "wikilinks"
  ]

  def __init__(self, infoboxes, nlp):
    self.infoboxes = infoboxes

    self.no_no = ["punct", "nummod", "appos", "det"]

    self.months = [
        datetime.date(2008, i, 1).strftime("%B").lower() for i in range(1, 13)
    ]

    self.nlp = nlp

  def process_nlp(self, text):
    doc = self.nlp(text)
    def_set = set()
    sentence = doc.sentences[0]

    be_index = None
    be_gov = None

    # find the "be" lemma
    for word in sentence.words:
      if word.lemma == "be":
        be_index = int(word.index.rjust(2))
        be_gov = word.governor
        def_set.add(sentence.words[be_gov - 1].lemma)
        break

    if be_index == None or be_gov == None:
      return "-"

    # now find the realted words with the be lemma
    for word in sentence.words:
      if word.governor == be_gov and word.dependency_relation in [
          "amod", "conj", "compound"
      ]:
        def_set.add(word.lemma)

    return "|".join(def_set)

  def process_article(self, page):
    """
      page[0] = title
      page[1] = text
      page[2] = page redirection (if it is a redirect article)

      return a list object ordered like cols, returns None if it's not an article
    """
    # if page[2] != None:
    #   return None

    if page[0].lower().startswith("list of"):
      return None

    # parse page
    wiki = mwparserfromhell.parse(page[1])

    # soft redirect pages
    if (len(wiki.filter_templates(matches="softredirect")) > 0):
      return None

    # [0] page title
    lst = [page[0]]

    # [1] defining article type
    # if "list of" in wiki.filter_templates(matches="DEFAULTSORT")[0].name.lower():
    #   return None
    # lst.append("list of")
    # else:
    lst.append("article")

    # [2] definition words
    # getting paragraphs, titles and other strings are included in the list
    # eg.: == head title ==
    # eg.: <!-- this comment -->
    fst = wiki.strip_code().split("\n")
    fst = [
        " ".join(Utils.cleanhtml(x.strip()).split())
        for x in fst
        if not x.isspace()
    ]

    # getting 1st paragraph, sometimes the 1xt element is not the 1st paragraph, e.g: apollo page
    lst.append("-")
    for p in fst:
      if p != "":
        lst[-1] = self.process_nlp(p)
        break

    # search for infobox template name
    box = None
    lst.append("-")  # [3] tmp append the infobox name

    for template in wiki.filter_templates():
      if template.name.strip_code().strip() in self.infoboxes:
        box = wiki.filter_templates(matches=template.name.strip())[0]
        lst[-1] = template.name.strip()
        break

    lst.append("-")  # [4] tmp append infobox items
    if box != None:
      box_items = "|".join([
          "{}:{}".format(
              param.name.strip_code().strip(),
              param.value.strip_code(keep_template_params=True).strip())
          for param in box.params
      ])
      lst[-1] = Utils.cleanhtml(box_items)

    # [5] Getting article categories
    lst.append("-")  # tmp append categories
    lst[-1] = "|".join([
        list(filter(None, x.title.split(":")))[-1]
        for x in wiki.filter_wikilinks(matches="category")
    ])

    # [6] wikilinks
    lst.append("-")  # tmp append wikilinks
    lst[-1] = "|".join([
        link.title.strip_code().strip()
        for link in wiki.filter_wikilinks()
        if "Category:" not in link.title.strip_code().strip()
    ])

    return lst
