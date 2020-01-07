import pandas as pd
import mwparserfromhell
import Utils
import datetime


class ProcessArticle:

  def __init__(self, infoboxes, nlp):
    self.infoboxes = infoboxes
    self.cols = [
        "title", "type", "def words", "infobox_name", "infobox", "categories",
        "wikilinks"
    ]

    self.no_no = [
        "punct", "nummod", "appos", "det", "nmod:poss", "cop", "cc", "nsubj",
        "case"
    ]

    self.months = [
        datetime.date(2008, i, 1).strftime("%B").lower() for i in range(1, 13)
    ]

    self.nlp = nlp

  def process_nlp(self, text):
    doc = self.nlp(text)
    def_set = set()

    for sentence in doc.sentences:
      for word in sentence.words:
        if word.dependency_relation in self.no_no or word.lemma.lower(
        ) in self.months:
          continue

        governor = sentence.words[word.governor -
                                  1].lemma if word.governor > 0 else ""

        def_set.add(governor)
        def_set.add(word.lemma.lower())

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

    if page[1].lower().startswith("list of"):
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
