import pandas as pd
import mwparserfromhell

# just a stupid way to setup a "global variable", i don't want to make a class just for that
infoboxes = []


def init(infobox_file):
  infoboxes = pd.read_csv(infobox_file)["infobox template name"].values.tolist()


def process_article(page):
  """
    page[0] = title
    page[1] = text
    page[2] = page redirection (if it is a redirect article)

    return a pandas.DataFrame object
  """
  # Creating dataframe
  df = pd.DataFrame(columns=[
      "title", "type", "def words", "infobox", "categories", "wikilinks"
  ])
  # parse page
  wiki = mwparserfromhell.parse(page[1])

  # page title
  df["title"] = [page[0]]

  # defining article type
  if page[2] != None:
    df["type"] = ["redirect"]
  else:
    df["type"] = ["article"]

  # Getting article categories
  df["categories"] = [
      "|".join([
          list(filter(None, x.title.split(":")))[1]
          for x in wiki.filter_wikilinks(matches="category")
      ])
  ]

  # wikilinks
  df["wikilinks"] = "|".join(
      [link.title.strip_code().strip() for link in wiki.filter_wikilinks()])

  # external links
  # df["ext_links"] = "|".join([link.url.strip_code().strip() for link in wiki.filter_external_links()])

  # search for infobox template name
  box = None
  for template in wiki.filter_templates():
    if template.name.strip() in infoboxes:
      box = wiki.filter_templates(matches=template.name.strip())[0]
      break

  if box != None:
    props = ""
    for param in box.params:
      props += "|{}:{}".format(param.name.strip_code().strip(),
                               param.value.strip_code().strip())

    df["infobox"] = props[1:]

  return df
