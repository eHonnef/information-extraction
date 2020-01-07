import pandas as pd
import mwparserfromhell
import Utils


class ProcessArticle:

  def __init__(self, infobox_file):
    self.infoboxes = pd.read_csv(
        infobox_file)["infobox template name"].values.tolist()
    self.cols = [
        "title", "type", "def words", "infobox_name", "infobox", "categories",
        "wikilinks"
    ]

  def process_nlp(self, text):
    return None

  def process_article(self, page):
    """
      page[0] = title
      page[1] = text
      page[2] = page redirection (if it is a redirect article)

      return a list object ordered like cols, returns None if it's not an article
    """
    # if page[2] != None:
    #   return None

    # parse page
    wiki = mwparserfromhell.parse(page[1])

    # [0] page title
    lst = [page[0]]

    # defining article type
    lst.append("article")

    # [1] definition words
    # lst.append(process_nlp(""))
    lst.append("-")

    # search for infobox template name
    box = None
    lst.append("-")  # [2] tmp append the infobox name

    for template in wiki.filter_templates():
      if template.name.strip_code().strip() in self.infoboxes:
        box = wiki.filter_templates(matches=template.name.strip())[0]
        lst[-1] = template.name.strip()
        break

    lst.append("-")  # [3] tmp append infobox items
    if box != None:
      box_items = "|".join([
          "{}:{}".format(
              param.name.strip_code().strip(),
              param.value.strip_code(keep_template_params=True).strip())
          for param in box.params
      ])
      lst[-1] = Utils.cleanhtml(box_items)

    # [4] Getting article categories
    lst.append("-")  # tmp append categories
    lst[-1] = "|".join([
        list(filter(None, x.title.split(":")))[-1]
        for x in wiki.filter_wikilinks(matches="category")
    ])

    # [5] wikilinks
    lst.append("-")  # tmp append wikilinks
    lst[-1] = "|".join([
        link.title.strip_code().strip()
        for link in wiki.filter_wikilinks()
        if "Category:" not in link.title.strip_code().strip()
    ])

    return lst
