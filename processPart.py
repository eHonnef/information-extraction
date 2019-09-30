import html
import re
from xml.etree import ElementTree
import os.path
from time import strftime, time
import spacy
import sys
import json
import nltk
import re
import codecs
import types
from WikiExtractor import clean
spacyModel =""
props={'annotators': 'tokenize,pos,depparse,lemma'} #these annotators will be used
#now configure variables
cf = open(sys.argv[2])
linesOfCf = cf.readlines()
langFolder = sys.argv[2]
categoryInLang = linesOfCf[0].strip()
spacyModel = linesOfCf[1].strip()
picklePath = linesOfCf[12].strip()
abbreviationsPath = linesOfCf[13].strip()
nationalitiesPath = linesOfCf[14].strip()
countriesPath = linesOfCf[15].strip()
beFroms = linesOfCf[2].strip().lower().split(",")
articles=linesOfCf[3].strip().lower().split(",")
upperCaseAllowed=linesOfCf[4].strip().lower().split(",")
pronouns=linesOfCf[5].strip().lower().split(",")
setOfMonths=linesOfCf[6].strip().lower().split(",")
phrasesInNotArticleTitle=linesOfCf[7].strip().split(",")
phrasesInNotArticleSentencesStart=linesOfCf[8].strip().split(",")
phrasesInNotArticleSentencesMiddle=linesOfCf[9].strip().split(",")
from stanfordcorenlp import StanfordCoreNLP
nlpStanford = StanfordCoreNLP('http://localhost', port=int(sys.argv[3]))
nlpStanford.logging_level = "none"
nlpSpacy = spacy.load(spacyModel)
sent_detector = nltk.data.load(picklePath)
for line in codecs.open(abbreviationsPath, 'r', 'utf-8'):
    sent_detector._params.abbrev_types.add(line.strip())

def createSetFromFile(file): #creates set() from file with one entry on one line
    settoRet = set()
    with open(file) as f:
        for line in f:
            settoRet.add(line.rstrip().lower()) #make everything lowercase
    return settoRet

acceptedAsNouns={"NN","NNS"}
setOfNationalities = createSetFromFile(nationalitiesPath)
setOfCountries = createSetFromFile(countriesPath)

def preprocess(sentence):
    replaced = re.sub("\((.*?)\)","",sentence) #regex to match all brackets and its content and replace them with empty string
    return replaced
def get_data(title, text):
    result = ""
    if "#" == text[0]: #redirects, filter them out
        return ["filtered_out"]
    text2 = text
    replaceInParentess = re.search("\{\{(.*?)(?=(lead=yes))",text2)
    if replaceInParentess:
        text2 = text.replace(replaceInParentess.group(0),replaceInParentess.group(1))
    text2 = re.sub("^(\{\{.*(\||\}\}).*\n){1,}","",text2)
    text2 = re.sub("^.*?\]\]\n\n","",text2)
    text2 = re.sub("\{\{Infobox.*?\n\}\}(?:\{|\n)\n?","",text2,flags=re.DOTALL)
    text2 = re.sub("^\n?\[\[(|File|Image):.*\]\n+(?=\'\'\')","",text2,flags=re.DOTALL)

    if not re.search("(?<=^)\'|[A-Z]|\d|\<",text2[0]) and text2[:3]!="'''": #if there is still something to remove in the begenning
        text2 = re.sub("^\s?.*?(\}|\|)\}\n+(?=\'|[A-Z]|\d|\<)","",text2,flags=re.DOTALL)
    infoboxAll = "" #save the whole unformatted infobox
    infoboxRaw = "" #save just the name of the first infobox on page
    infoboxName = "" #save just the name of the first infobox on page
    tmp = re.search("(\{\{(?:Speciesbox|Automatic taxobox|Taxobox|Infobox).*?\}\}\n)(?!\s?\|)",text,flags=re.DOTALL)
    if tmp:
        infoboxAll = tmp.group(0).replace("\n","")
        infoboxAll = re.sub("\s{2,}"," ",infoboxAll)
        infoboxNameFound = re.search("\{\{(?:Speciesbox|Automatic taxobox|Taxobox|Infobox)\s(.*?)(?=\|)",infoboxAll)
        if infoboxNameFound:
            infoboxName = infoboxNameFound.group(0)
            infoboxName = re.sub("\{\{(Speciesbox|Automatic taxobox|Taxobox|Infobox)\s","",infoboxName)
            infoboxName = re.sub("<[^>]+>", "", infoboxName).strip()
    text2 = re.sub("^\|.*?\}\}\n+","",text2,flags=re.DOTALL)
    firstParagraphWiki = text2.split("\n\n")[0]

    partOfTextWithCategories = text.split(categoryInLang)
    categories = []
    if len(partOfTextWithCategories) > 1:
        for part in partOfTextWithCategories[1:]:
            categories.append(part.lstrip(":").split('|')[0].split(']')[0].split('}')[0])

    firstParagraphClean = firstParagraphWiki.replace("\n\n", "S_%neW-paragraph%_E").replace("\n=", "S_%neW-paragraph%_E").replace("\n*", "S_%neW-paragraph%_E")
    firstParagraphClean = re.sub("from ?{{", "{{", firstParagraphClean)
    firstParagraphClean = re.sub("\({{lang-[^)]+(}}|/)\)", "", firstParagraphClean)
    firstParagraphClean = firstParagraphClean.strip().replace("<br>", "").replace("<br/>", "").replace("<br />", "")
    firstParagraphClean = re.sub("(__TOC__|__NOTOC__)","",firstParagraphClean,flags=re.IGNORECASE)
    firstParagraphClean = re.sub("^:?\"[^\n\.\!\?]*[\.\!\?]?\"\n", "", firstParagraphClean)
    firstParagraphClean = re.sub("S?_?%neW-paragraph%_?E?", "\n\n", firstParagraphClean)
    firstParagraphClean = re.sub(".*&quot;.*For the .*, see .*&quot;.*", "", firstParagraphClean)
    try:
        regex = "(\{\{.*?" + re.escape(title) + ".*?lead=yes\}\})"
        firstParagraphClean = re.sub("^\{\{.*'''.*\}\}",title,firstParagraphClean)
        firstParagraphClean = re.sub(regex,title,firstParagraphClean)#if the title is in nested {{ }}, the clean will remove it, se lets extract the title before that happends
        firstParagraphClean = re.sub("(?<=\'\'\'\s)\(.*?\)","",firstParagraphClean)
    except: pass
    langToRepl = re.search("\({{lang-.*?\|(.*?)}}",firstParagraphClean)
    if langToRepl:
        firstParagraphClean = firstParagraphClean.replace(langToRepl.group(0),"("+langToRepl.group(1)+")")
    firstParagraphClean = clean(firstParagraphClean,title)
    firstParagraphClean = firstParagraphClean.lstrip("\n")
    firstParagraphClean = firstParagraphClean.split("\n\n")[0]
    while len(firstParagraphClean.strip()) > 0 and firstParagraphClean.strip()[:1] == "=":
        firstParagraphClean = (firstParagraphClean.split("\n", 1)[1] if len(firstParagraphClean.split("\n", 1)) > 1 else "").strip()
    if firstParagraphClean[:2] == "* ":
        firstParagraphClean = firstParagraphClean[2:]
    if firstParagraphClean[:1] == "*":
        firstParagraphClean = firstParagraphClean[1:]
        firstParagraphClean = replace_inside_parentheses(firstParagraphClean, ".", ":")
        firstParagraphClean = replace_inside_parentheses(firstParagraphClean, "?", "+")
    firstParagraphClean = firstParagraphClean.replace("&quot;","'").replace("&nbsp;"," ").replace("&ndash;","-").replace(".'",".")

    bracketsWithContent = re.search("\((.*?)\)",firstParagraphClean) #the tokenizer may have problems with brackets, temporairly replace them
    if bracketsWithContent:
        firstParagraphClean= firstParagraphClean.replace(bracketsWithContent.group(0),"bracketsanditscontent")
    temp = sent_detector.tokenize(firstParagraphClean) #

    if len(temp) > 0: #replace the brackets with its content back as it was
        j=0
        if bracketsWithContent:
            for j in range(0, len(temp)):
                if "bracketsanditscontent" in temp[j]:
                    temp[j] = temp[j].replace("bracketsanditscontent",
                                              bracketsWithContent.group(0))  # replace it back
            firstParagraphClean = firstParagraphClean.replace("bracketsanditscontent",
                                              bracketsWithContent.group(0))
        bracketsWithContent = None
        result = temp[0].strip().split('\n')[0]

    else:
        result = str()
    a = result.strip()

    if title.startswith('\"') and title.endswith('\"'):
        title = title[1:-1]
    if title.startswith("\'") and title.endswith("\'"):
        title = title[1:-1]

    if a.startswith('\"') and a.endswith('\"'):
        a = a[1:-1]
    if a.startswith("\'") and a.endswith("\'"):
        a = a[1:-1]

    if a == title:
        result = result + " " + (
            temp[1].strip() if len(temp) > 1 else "")

    result = replace_inside_parentheses(result, ":", ".")
    result = replace_inside_parentheses(result, "+", "?")

    result = re.sub("\s*\([\s\W]*(or)?[\s\W]*\)", "", result)
    result = re.sub("\([^\w&]+", "(", result)
    result = re.sub("[^\w&.]+\)", ")", result)
    result = re.sub("(\s?)[,;:'\\-–\s]+(\\s?)([,;:'\\-–])", "\\1\\2\\3", result)
    result = re.sub("<[^>]+>", "", result)
    result = result.strip().strip('"')
    title = title.strip().strip('"')
    if result.split(" ")[0] in beFroms: #sometimes the title is removed no matter the regexes trying to fix it, just add it manually
        result = title +" "+ result
    splittedOnDot = firstParagraphWiki.split(".")
    lastWord = result.split(" ")[len(result.split(" "))-1]
    firstSenteceUnformatted = []
    for one in splittedOnDot: #split the paragraph on dots and check if the last word matches th last one from first sentece
        oneFormatted = one.replace("[[", "").replace("]]", "")
        lastWordToMatch = oneFormatted.split(" ")[len(oneFormatted.split(" "))-1]+"."
        firstSenteceUnformatted.append(one)
        if lastWord==lastWordToMatch:
            break

    if "has the following meaning" in firstParagraphClean: #special case where sentences ending with colon might have some infomration on the page
        result = []
        sentences = re.findall("(?<=#)(.*?)(?=\n)", text2)
        for oneSentence in sentences:
            result.append(clean(oneSentence,title).replace("\t"," ").replace("\n",""))
        toReturn = [result]
        toReturn.append("")
        toReturn.append('|'.join(categories).replace("\t"," ").replace("\n",""))
        toReturn.append("")
        toReturn.append(' '.join(firstSenteceUnformatted).replace("\t"," ").replace("\n",""))
        toReturn.append(firstParagraphClean.replace("\t"," ").replace("\n",""))
        toReturn.append(firstParagraphWiki.replace("\t"," ").replace("\n",""))
        return toReturn


    if checkIfArticle(title,result):
        toReturn = [result.replace("\t"," ").replace("\n","")]
        toReturn.append(infoboxName.replace("\t"," ").replace("\n",""))
        toReturn.append('|'.join(categories).replace("\t"," ").replace("\n",""))
        toReturn.append(infoboxAll.replace("\t"," ").replace("\n",""))
        toReturn.append(' '.join(firstSenteceUnformatted).replace("\t"," ").replace("\n",""))
        toReturn.append(firstParagraphClean.replace("\t"," ").replace("\n",""))
        toReturn.append(firstParagraphWiki.replace("\t"," ").replace("\n",""))
        return toReturn
    return ["filtered_out"]
def replace_inside_parentheses(string, toReplace, replaceWith):
    bracket_count = 0
    return_string = ""
    for one in string:
        if one == "(":
            bracket_count += 1
        elif one == ")":
            bracket_count -= 1
        if bracket_count > 0:
            return_string += one.replace(toReplace, replaceWith)
        else:
            return_string += one
    return return_string

def checkIfArticle(title,sentence):
    if len(sentence) > 0:
        splitted_title = title.split(" ")
        if splitted_title[0].lower() in setOfMonths:
            if len(splitted_title)==2:
                if re.search("\d",splitted_title[1]):
                    return False #articles like April 1999,May 1989...

        if "#" == sentence[0]: #redirect
            return False
        for phr in phrasesInNotArticleSentencesMiddle:
            if phr in sentence:
                return False
        for phr2 in phrasesInNotArticleSentencesStart:
            if sentence.startswith(phr2):
                return False
        for phr3 in phrasesInNotArticleTitle:
            if phr3 in title:
                return False
        return True
    return False
def findDefwordsStanford(sentence):
    defwords = []
    try:
        annotated = nlpStanford.annotate(sentence,properties=props) #annotate the sentence
        parsed = json.loads(annotated)
    except:
        return ""
    if len(parsed.get('sentences'))<1:
        return ""
    annotedInJSON =  parsed.get('sentences')[0]
    positionOfBe = -1
    for token in annotedInJSON.get('tokens'): #find be in sentence
        if token.get('lemma').lower() in beFroms:
            positionOfBe = token.get('index',-1)
            break

    if positionOfBe != -1: #if there was be in some form
        if positionOfBe < len(annotedInJSON.get('basicDependencies',"")):
            firstDefWordPosotion = annotedInJSON.get('basicDependencies')[positionOfBe].get('governor')
            if annotedInJSON.get('tokens')[firstDefWordPosotion-1].get('pos') in acceptedAsNouns or str(annotedInJSON.get('tokens')[firstDefWordPosotion-1].get('lemma')).lower() in upperCaseAllowed:
                defwords.append(annotedInJSON.get('tokens')[firstDefWordPosotion - 1].get('lemma').lower())
            for oneDep in annotedInJSON.get('basicDependencies'):  # find the other defwords
                if oneDep.get('dep') == "conj":
                    if oneDep.get('governor')==firstDefWordPosotion:
                        if annotedInJSON.get('tokens')[oneDep.get('dependent') - 1].get('pos') in acceptedAsNouns or str(annotedInJSON.get('tokens')[firstDefWordPosotion-1].get('lemma')).lower() in upperCaseAllowed:
                            defwords.append(annotedInJSON.get('tokens')[oneDep.get('dependent') - 1].get('lemma').lower())


    if len(defwords) == 0: #additional filtering
        cont = True
        for i in range(0, len(annotedInJSON.get('tokens')) - 3):
            if cont:
                if annotedInJSON.get('tokens')[i].get('lemma').lower() in beFroms:
                    if annotedInJSON.get('tokens')[i + 1].get('pos') == "CD" and annotedInJSON.get('tokens')[i + 2].get(
                            'pos') == "IN":
                        j = i + 2
                        while j + 1 <= len(annotedInJSON.get('tokens')) - 4:
                            j = j + 1
                            if annotedInJSON.get('tokens')[j].get('pos') in acceptedAsNouns or \
                                    str(annotedInJSON.get('tokens')[j].get('lemma')).lower() in upperCaseAllowed:
                                defwords.append(annotedInJSON.get('tokens')[j].get('lemma').lower())
                                cont = False
                                break


    return '|'.join(sorted(defwords))


# funkce zpracuje vetu a vrati vysledek ve tvaru [Veta,[definicni_slovo1,definicni_slovo2...]]
def findDefwordsSpacy(sentence):
    chunkerDidNotFound = False
    analyzedSentence = nlpSpacy(sentence)
    chunks = iter(analyzedSentence.noun_chunks)
    next(chunks, None)
    chunkDicts = []
    # Nactou se vsechny chunky, ktere ve vete jsou do listu k dalsimu zpracovani
    for chunk in chunks:
        dict = {"chunk.text": chunk.text, "chunk.root.text": chunk.root.text, "chunk.root.dep_": chunk.root.dep_,
                "chunk.root.head.text": chunk.root.head.text}
        chunkDicts.append(dict)
    nationalities = []
    words = []
    allDefWords = []
    definitions = []
    exclude = set()
    include = set()
    keypos = 0
    for token in analyzedSentence:
        if keypos == 0:
            exclude.add(token.text)
            if str(token.lemma_).lower() in beFroms:
                keypos = token.i
        else: #after be, lets include the words
            include.add(token.text)
         #exlude everything before be
    toExclude = exclude - include

    for chunk in chunkDicts:  # Pro každý chunk
        if  str(nlpSpacy(chunk['chunk.root.head.text'])[0].lemma_).lower() in beFroms:  # Pokud je kořen chunku potomkem slovesa být
            words.append(chunk['chunk.root.head.text'].lower())  # Přidej slovo do seznamu definičních slov
    for chunk in chunkDicts:  # Pro každý chunk
        if chunk['chunk.root.head.text'] in words:  # Pokud je kořen chunku v seznamu definičních slov
            if chunk['chunk.root.text'].lower() in upperCaseAllowed or chunk[
                'chunk.root.text'].islower():  # Nepřidávat vlastní jména
                if (chunk['chunk.root.text'].lower() not in pronouns):  # Nepřidávat vztážná zájmena
                    words.append(chunk['chunk.root.text'].lower())  # Přidej slovo do seznamu

    lenOfSentence = len(analyzedSentence) - 1
    if len(words) <= 1:
        chunkerDidNotFound = True  # chunker nic nenašel
        for token in analyzedSentence:  # Iterace přes všechny token ve větě
            j = 0
            if str(token.lemma_).lower() in beFroms:
                while (token.i + j + 1 < lenOfSentence):  # Pokud existuje následující token
                    j = j + 1  # Přesune se kontrola na další token
                    if (token.doc[token.i + j].pos_ == "NOUN" or
                        str(token.doc[token.i + j].lemma_ ).lower()in upperCaseAllowed) \
                            and token.doc[
                        token.i + j] in token.children:  # Token je označen jako podst. jméno, nebo je ze seznamu povolených(King,Bishop...)
                        words.append(token.doc[token.i + j].lemma_.lower())
    if len(words) == 0:
        cont = True
        for i in range(0, len(analyzedSentence) - 3):  # is one of ...
            if cont:
                if str(analyzedSentence[i].lemma_).lower() in beFroms:
                    if analyzedSentence[i + 1].pos_ == "NUM" and analyzedSentence[i + 2].pos_ == "ADP":
                        j = i + 2
                        while j + 1 <= len(analyzedSentence) - 4:
                            j = j + 1
                            if analyzedSentence[j].pos_ == "NOUN" or str(analyzedSentence[j].lemma_).lower() in upperCaseAllowed:
                                words.append(str(analyzedSentence[j].lemma_).lower())
                                cont = False
                                break

    splitted = []
    #Fix of some problems with hyphens that spacy has
    foundwithhypen = re.findall('\S+.-\S+', sentenceToAnalyze)
    for word in foundwithhypen:
        splitted = word.split("-")
        toExclude.update(splitted)
    foundwithhypen = re.findall('\S+.\'\S+', sentenceToAnalyze)
    for word in foundwithhypen:
        splitted = word.split("'")
        toExclude.update(splitted)

    for chunk in chunkDicts:
        if (chunk['chunk.root.head.text'] in words): #Search the chunk and remove articles from phrases and detect nationalities
            if (chunk['chunk.text'].split()[0].lower() in articles):
                withoutarticles = chunk['chunk.text'].split()[1:]
                withoutarticles = ' '.join(withoutarticles)
                if (len(chunk['chunk.text'].split()) > 2):
                    definitions.append(withoutarticles)
                    for word in withoutarticles.split():
                        if (word.lower() in setOfNationalities):
                            nationalities.append(word.title())
            else:
                if (len(chunk['chunk.text'].split()) > 1):
                    definitions.append(chunk['chunk.text'])
                    for word in chunk['chunk.text'].split():
                        if (word.lower() in setOfNationalities):
                            nationalities.append(word.title())

    words = set(words) - toExclude
    wordsToRet = []
    for word in words:
        if str(nlpSpacy(word)[0].lemma_).lower() not in beFroms and word.lower() not in pronouns:
            wordsToRet.append(word)

    formattedwords = '|'.join(sorted(wordsToRet))
    definitions = '|'.join(sorted(definitions))

    for onelist in definitions:
        i = 0
        for i in range(0, len(onelist.split())):
            if onelist.split()[i].lower() in setOfNationalities:
                allDefWords.append(str(onelist.split()[i]).title())

    if (len(nationalities)):
        allDefWords.insert(0, nationalities[0])
    else:
        allDefWords.insert(0, "")
    allDefWords.insert(0, definitions)
    allDefWords.insert(0, formattedwords)
    return allDefWords
inputxml = ""
inputxml = sys.argv[1]
count = 0
i = 0
x = 0
ns = None
inPage = False
inBuffer = 0
buffer = '<root>'
title = None
page = 10000
lastTime = time()
totalTime = 0
numberOfClient = inputxml.split(".")[len(inputxml.split("."))-1]
tsv_output = []
colon_output = []
sys.stderr.write("[INFO][CLIENT "+str(numberOfClient)+"][" + str(strftime("%m-%d %H:%M") + "] Started with PID: "+str(os.getpid())+", processing file '"+inputxml+"'\n"))
for line in open(inputxml, 'r'):
    if line.strip() == '<page>':
        inPage = True
    if inPage:
        buffer = buffer + line
    if line.strip() == '</page>' and inPage:
        inBuffer += 1
        inPage = False
    if inBuffer >= 1000 or line.strip() == '</mediawiki>':
        buffer = buffer + '<page></page></root>'
        for elem in ElementTree.fromstring(buffer).iter():
            if elem.tag == 'title':
                title = elem.text
            if elem.tag == 'text':
                text = elem.text
            if elem.tag == 'ns' and int(elem.text) == 0:
                ns = True
            if elem.tag == 'page':
                if title and text and ns:
                    i += 1
                    result = get_data(title, text)  # index0 = sentence, index1 = infobox
                    toout = []
                    if result[0]!="filtered_out":
                        stanfordRes = []
                        spacyRes = [[],[],[]]
                        spacyResTmp = []
                        sentences = []
                        if(type(result[0]) is list): #special case with more than one sentence(e.g. https://en.wikipedia.org/wiki/Call_set-up_time)
                            infoboxName = result[1]
                            categories = result[2]
                            toout.append(title.replace("\t", " ").replace("\n", ""))  # title of article
                            for sentence in result[0]:
                                sentenceToAnalyze = html.unescape(sentence)
                                sentences.append(sentenceToAnalyze)
                                preprocessedSentence = preprocess(title+" is "+sentenceToAnalyze)
                                stanfrodResTmp = findDefwordsStanford(preprocessedSentence)
                                if len(stanfrodResTmp): stanfordRes.append(stanfrodResTmp)  # stanfrod defword
                                spacyResTmp = findDefwordsSpacy(preprocessedSentence)
                                if len(spacyResTmp[0]):spacyRes[0].append(spacyResTmp[0])
                                if len(spacyResTmp[1]):spacyRes[1].append(spacyResTmp[1])
                                if len(spacyResTmp[2]):spacyRes[2].append(spacyResTmp[2])

                            tsv_output.append([title,'|'.join(sentences),"",categories,'|'.join(stanfordRes),'|'.join(spacyRes[0]),'|'.join(spacyRes[1]),'|'.join(spacyRes[2]),"",result[4],result[5],result[6]])

                        else:
                            sentenceToAnalyze = html.unescape(result[0])
                            preprocessedSentence = preprocess(sentenceToAnalyze)
                            infoboxName = result[1]
                            categories = result[2]
                            toout.append(title)#title of article
                            toout.append(sentenceToAnalyze)#first sentence formatted
                            toout.append(infoboxName)#name of first infobox
                            toout.append(categories)#wikicategories
                            toout.append(findDefwordsStanford(preprocessedSentence))#stanfrod defword
                            spacyRes = findDefwordsSpacy(preprocessedSentence)
                            toout.append(spacyRes[0])#defwords
                            toout.append(spacyRes[1])#defphrases
                            toout.append(spacyRes[2]) #nationality from noun chunk
                            toout.append(result[3])#infobox string
                            toout.append(result[4])#first sentence unformatted
                            toout.append(result[5])#first paragraph formatted
                            toout.append(result[6])#first paragraph unformatted
                            if len(title) and len(preprocessedSentence):
                                if sentenceToAnalyze[-1:] == ":":
                                    colon_output.append(toout)
                                else:
                                    tsv_output.append(toout)
                    title = None
                    text = None
                    ns = None
                if i >= page:
                    x += 1
                    i = 0
                inBuffer = 0
                buffer = '<root>'
with open(inputxml+".out", mode='w') as outFile:
    for article in tsv_output:
        outFile.write("\t".join(article)+"\n")
with open(inputxml+".col", mode='w') as colFile:
    for article in colon_output:
        colFile.write("\t".join(article)+"\n")
sys.stderr.write("[INFO][CLIENT"+str(numberOfClient)+"][" + str(strftime("%m-%d %H:%M") + "] Finished processing file '"+inputxml+"'\n"))
