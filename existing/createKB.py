import collections
import math
import operator
import random
import re
import sys
from multiprocessing.dummy import Process
from time import strftime
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
CORRECTNUMBEROFCOLUMNTS = 12
features = ['infobox','wikicategories','defwordStanford','defwordSpaCy','defphrSpaCy']
#load all configuration from the config file
conf = open(sys.argv[2])
linesOfCf = conf.readlines()
nationalitiesPath = linesOfCf[14].strip()
countriesPath = linesOfCf[15].strip()
categoryInLang = linesOfCf[0].strip()
setOfMonths=linesOfCf[6].strip().split(",")
conf2 = open(sys.argv[3])
linesOfCf2 = conf2.readlines()
MINOCCURENCE = linesOfCf2[2].strip().split(",")
percentileLimit = linesOfCf2[3].strip().split(",")
hardLimitLines = int(linesOfCf2[4].strip())
categoryFilter = linesOfCf2[1].strip()
onlyAllMatched = int(linesOfCf2[7].strip())
learnFile = linesOfCf2[0].strip()
iterationsForExpandingTrainigSet = int(linesOfCf2[5].strip())
iterationsForTrainingClassifiers = int(linesOfCf2[6].strip())
failedClassifiers = []
classifierLimits = {'infobox': {}, 'wikicategories': {}, 'defwordStanford': {}, 'defwordSpaCy': {}, 'defphrSpaCy': {}}
for i in range(0,len(features)): #build a dictionary containing the limits for each feature
    classifierLimits[features[i]]['min'] = int(MINOCCURENCE[i])
    classifierLimits[features[i]]['percentage'] = int(percentileLimit[0])

def createSetFromFile(file): #creates set() from file with one entry on one line
    settoRet = set()
    with open(file) as f:
        for line in f:
            settoRet.add(line.rstrip().lower()) #make everything lowercase
    return settoRet
aliases = ['native_name','nickname','fullname','official_name','other_name','character_name','alter_ego']
infoboxInfoToKeep={'name','native_name','nickname','birth_place',
                   'native_name_lang','birth_date','birth_date','death_date','death_date','fullname','subdivision_name','country','population','occupation','coordinates',
                   'official_name','other_name','area_total_km2','population','area','length','streamflow','source1_location','character_name','alter_ego','nationality','subdivision_name1'}
setOfNationalities = createSetFromFile(nationalitiesPath)
setOfCountries = createSetFromFile(countriesPath)
categoriesToPrint = ["artwork","organism","organisation","event","person","settlement","person:fictional","person:group","country","country:former","watercourse","waterarea","geo:relief","geo:waterfall","geo:island","geo:peninsula","geo:continent"]
classifiers = {}
toPrint = []

def convertlong(longtitude): #converts longitute from degree format to decimal
    W = 'W' in longtitude
    if len(longtitude[:-2].split(';')) == 3:
        d, m, s = map(float, longtitude[:-2].split(';'))
    else:
        d, m = map(float, longtitude[:-2].split(';'))
        s = 0
    return (d + m / 60. + s / 3600.) * (-1 if W else 1)


def convertlat(latitude): #converts latitude from degree format to decimal
    N = 'N' in latitude
    if len(latitude[:-2].split(';'))==3:
        d, m, s = map(float, latitude[:-2].split(';'))
    else:
        d, m = map(float, latitude[:-2].split(';'))
        s=0
    return (d + m / 60. + s / 3600.) * (1 if N else -1)

def preprocessCategories(wholeTsv): #preprocesses wikicategories according to filter settings
    preprocessedCategories = {}
    for splitted_line in wholeTsv:
        if len(splitted_line)>3:
            if splitted_line[3]:
                splitted = splitted_line[3].split("|")
                for i in range (0,len(splitted)):
                    oneCatOptimized = splitted[i]
                    if "y" in categoryFilter:
                        oneCatOptimized = ''.join([j for j in splitted[i] if not j.isdigit()])
                    if "n" in categoryFilter:
                        for nationality in setOfNationalities:
                            oneCatOptimized= oneCatOptimized.replace(nationality, "")
                    if "c" in categoryFilter:
                        for country in setOfCountries:
                            oneCatOptimized=oneCatOptimized.replace(country, "")
                    if splitted[i] not in preprocessedCategories:
                        preprocessedCategories.update({splitted[i]:oneCatOptimized})
                    splitted[i] = oneCatOptimized

    return preprocessedCategories

def filterLowOccurences(toTrainFrom,feature): #filters feature values with low occurencies
    maxEntries = hardLimitLines
    percentile = int(classifierLimits.get(feature).get('percentile',100)) #get the % that should be kept for this feature
    normalize = {}
    data = []
    target = []
    for one in toTrainFrom:
        for cat,word in one.items():
            if len(word):
                if list(one.keys())[0] not in normalize:
                    normalize[list(one.keys())[0]] = []
                normalize[list(one.keys())[0]].append(word)
                data.append(word)
                target.append(cat)
    occurences = {}
    percentageOccurencesOfDefwords = {}
    for cat in normalize:
        occurences[cat]  = Counter(normalize[cat]) #for each category count occurences of defwrods
        percentageOccurencesOfDefwords[cat] = [(i, occurences[cat][i] / len(normalize[cat]) * 100.0) for i in occurences[cat]] #convert occurences into percentage coverage
        percentageOccurencesOfDefwords[cat].sort(key=lambda tup: tup[1],reverse=True) #sort from largest coverage

    wordsToKeep = set() #add from the biggest group and stop when the required percentile is reached
    for cat in percentageOccurencesOfDefwords:
        percentage = 0
        for entry in percentageOccurencesOfDefwords[cat]: #keep adding the feature values until limit is reached
            if percentage>percentile:
                break
            if occurences[cat][entry[0]] >= int(classifierLimits.get(feature).get('min',1)): #check if the value has occurency bigger than treshold
                percentage+=entry[1]
                wordsToKeep.add(entry[0])
    i = 0
    filteredData = [] #now we have the most common words for each category that cover at least "percentile" % of the category
    filteredTarget = []

    for i in range(i,len(data)):
        if data[i] in wordsToKeep: #if the entry is in words we want to keep, add it
            if len(data[i])>=3:
                filteredData.append(data[i])
                filteredTarget.append(target[i])
    filterDataToRtet = []
    filterTargetoRtet = []
    if len(filteredData) - 1 > maxEntries:
        indexesToKeep = random.sample(range(0, len(filteredData) - 1), maxEntries - 1)  # save the required percentage of lines, these will be learned from
        for i in indexesToKeep:
            filterDataToRtet.append(filteredData[i])
            filterTargetoRtet.append(filteredTarget[i])
    else:
        filterDataToRtet = filteredData
        filterTargetoRtet = filteredTarget

    return [filterDataToRtet,filterTargetoRtet,wordsToKeep] #return filtered data and target and also the feature values that passed the filter

def getLinesForLearning(lines,linesToUseForLearning): #returns list with lines that will be used in the learning process

    titlesWithCategoriesClassifiedSoFar = {}  # load grand truths from file into variable
    with open(learnFile, 'r') as LEARNfile:
        for line in LEARNfile:
            splitted_line = line.split('\t')
            titlesWithCategoriesClassifiedSoFar[splitted_line[0]] = splitted_line[1].rstrip()

    linesToLearnFrom = []

    # to speed up the training process, lets take just some of the lines from the whole file to learn from
    if not linesToUseForLearning>= len(lines):
        indexesToKeep = random.sample(range(0, len(lines) - 1),linesToUseForLearning-1)  # save the required percentage of lines, these will be learned from
        for i in indexesToKeep:
            linesToLearnFrom.append(lines[i])
        for line in lines:  # exclude article with ground truths from filtering and add them explicitly after
            splitted_line = line
            if splitted_line[0] in titlesWithCategoriesClassifiedSoFar:
                linesToLearnFrom.append(line)
    else:
        linesToLearnFrom = lines

    return linesToLearnFrom

def expandTrainingSet(interations, preprocessedCategories, linesToLearnFrom): #
    titlesWithCategoriesClassifiedSoFar = {}
    titlesWithCategoriesFromFileToAdd = {}
    with open(learnFile, 'r') as LEARNfile:
        for line in LEARNfile:
            splitted_line = line.split('\t')
            titlesWithCategoriesClassifiedSoFar[splitted_line[0]] = splitted_line[1].rstrip()

    allLinesFromTSV = {}
    allLinesFromTSVList = []
    for line in linesToLearnFrom: #load the lines that will be used for learining into different strucutres
        splitted_line = line
        allLinesFromTSVList.append(splitted_line)
        allLinesFromTSV[splitted_line[0]] = [splitted_line[1:]]
    learned = {}
    for i in range(0,interations+1): # in each iteration look at infoboxes and categories from last it. and add the others from the line, this will serve in the next it.
        #print(str(strftime("%m-%d %H:%M")+"LEARN ITER:"+str(i)))
        for one in titlesWithCategoriesClassifiedSoFar: #get informations from already classified lines
            if one in allLinesFromTSV:
                splitted_line = list([one])
                splitted_line.extend(allLinesFromTSV[one][0])
                if len(splitted_line)>6:
                    if titlesWithCategoriesClassifiedSoFar[one] not in learned: #if has not already been classified
                        learned[titlesWithCategoriesClassifiedSoFar[one]] = {'infobox': set(), 'wikicategories': set(), 'defwordStanford': set(), 'defwordSpaCy': set(), 'defphrSpaCy': set()}
                    learned[titlesWithCategoriesClassifiedSoFar[one]]['infobox'].update(splitted_line[2].split("|"))
                    for oneCategory in splitted_line[3].split("|"):
                        toAdd = preprocessedCategories.get(oneCategory)
                        learned[titlesWithCategoriesClassifiedSoFar[one]]['wikicategories'].add(toAdd)

        infobox = []
        wikicategories = []
        for category in learned: #append learned now to already learned
            infobox.extend(learned[category]['infobox'])
            wikicategories.extend(learned[category]['wikicategories'])
        toDelDictCat = Counter(wikicategories) #get number of categories in which the article was classified(by category)
        toDelCat = set()
        for key, value in toDelDictCat.items():
            if value != 1:
                toDelCat.add(key) #if it is in more than one, delte it, this prevents learning wrong category
        toDelDictInf = Counter(infobox) #get number of categories in which the article was classified(by infobox)
        toDelInf = set()
        for key, value in toDelDictInf.items():
            if value != 1:
                toDelInf.add(key) #if it is in more than one, delte it, this prevents learning wrong category
        tmp = {}
        for category,data in learned.items(): #delete what is market to be deleted
            if category not in tmp:
                tmp[category] = {}
            tmp[category]['infobox'] = data['infobox'] - toDelInf
            tmp[category]['wikicategories'] = data['wikicategories'] - toDelCat

        learned = tmp

        for splitted_line in allLinesFromTSVList:
            detectedCategory = set()
            for title, category in titlesWithCategoriesClassifiedSoFar.items():
                if splitted_line[0] not in titlesWithCategoriesClassifiedSoFar:
                    if len(splitted_line) >= 7: #if the line has all columns
                        if splitted_line[2]:
                            if category in tmp:
                                if splitted_line[2] in tmp[category]['infobox']:
                                    detectedCategory.add(category)

                        if splitted_line[3]:
                            if category in tmp:
                                for oneCat in splitted_line[3].split("|"):
                                    oneCat = preprocessedCategories.get(oneCat)
                                    if oneCat in tmp[category]['wikicategories']:
                                        detectedCategory.add(category)

                if len(detectedCategory) == 1: #if the detected category is not different(infobox vs wikicategories), add it
                    titlesWithCategoriesFromFileToAdd[splitted_line[0]] = list(detectedCategory)[0]
        titlesWithCategoriesClassifiedSoFar.update(titlesWithCategoriesFromFileToAdd)
        #print("TIT:"+str(len(titlesWithCategoriesClassifiedSoFar)))
    return learned


def parseInfobox(infobox_string):
    information = {}
    infobox_string =infobox_string.replace("<br>",", ").replace("</br>",", ")
    infobox_string = re.sub("<ref.*?>.*?</ref>","",infobox_string)
    infobox_string = re.sub("^\{\{.*?box.*?(?=\|)","",infobox_string)
    infobox_string = re.sub("(\s?\|\s?)", "|", infobox_string)
    infobox_string = re.sub("(\<.*?\>)", "", infobox_string)
    infobox_string = re.sub("\[http.*?\]", "", infobox_string)
    pipeAdjust = re.findall("(?<=\{\{)(.*?)(?=\}\})",infobox_string)
    if pipeAdjust:
        for one in pipeAdjust:
            infobox_string = infobox_string.replace(one,';'.join(one.split("|")))
    pipeAdjust = re.findall("(?<=\[\[)(.*?)(?=\]\])", infobox_string)
    if pipeAdjust:
        for one in pipeAdjust:
            infobox_string = infobox_string.replace(one, ';'.join(one.split("|")))

    infobox_string = infobox_string.replace("{","").replace("}","")
    infobox_string = re.sub("(\s+=\s+)","=",infobox_string)
    foundInformation = re.findall("(?<=\|)(.*?)(?=\|)",infobox_string)
    for one in foundInformation:
        splitted = one.split("=")
        if len(splitted) >= 2:
            if splitted[0] in infoboxInfoToKeep:
                one = stylePartOfInfobox(one,information)
                if one is not None:
                    for oneInformation in one:
                        information.update(oneInformation)
    return information

def convertToSI(string):
    string = string.lower()
    if re.search("(?<=convert;)(\d+)(?=;)",string) and re.search("(?<=convert;)\d+;(.*?)(?=;)",string):
        try:
            value = int(re.search("(?<=convert;)(\d+)(?=;)",string).group(1))
            unit = re.search("(?<=convert;)\d+;(.*?)(?=;)",string).group(1)
            if unit =="lb":
                value = value*0.4535
            if unit == "ft":
                value = value*30.48
            if unit=="mi":
                value = value *1.609
            if unit=="acre":
                value = value * 0.004046
            if unit =="ha":
                value = value*0.01
            if unit =="mi2":
                value = value*2.59
        except:
            return ""
        return round(value,3)
    return ""
def stylePartOfInfobox(one,information):
    splitted = one.split("=",1)
    splitted[0] = splitted[0].replace("-", "_")  # sometimes birth_place is writeen as birth-place and so on
    parsed = splitted[1]
    found = re.findall("\[\[.*?\]\]",parsed)
    if found:
        for onepart in found:
            parsed = parsed.replace(onepart,onepart.split(";")[len(onepart.split(";"))-1])
        parsed = parsed.replace("]]","").replace("[[","").replace(",,",",")
        splitted[1] = parsed
    if "date" in splitted[0]:
        if "death" in splitted[0]: #sometimes, birth date is also in death date, lets delete it
            splitted[1] = splitted[1].replace(information.get("birth_date",""),"")
        splitted[1] = re.sub("^.*(date;|year;|age;|=yes;|=no;|=y;)","",splitted[1])
        splitted[1] = re.sub(";[^;]+?(date|year|age|=yes|=no|=y)$","",splitted[1])
        if re.search("\d{4};\d{1,2};\d{1,2}",splitted[1]): splitted[1] = re.search("\d{4};\d{1,2};\d{1,2}",splitted[1]).group(0)
        splitted[1] = splitted[1].strip(";").replace(";","-")
    if "occupation" in splitted[0]:
        if re.search("(?<=list;)(.*)", splitted[1]): splitted[1] = re.search("(?<=list;)(.*)",splitted[1]).group(0)
        splitted[1] = splitted[1].replace(";",", ").replace("*",",,")
    if "coordinates" in splitted[0]:
        lat=""
        long=""
        if re.search("(?:\d{1,3};){2,3}[N|S]",splitted[1]):lat=re.search("(?:\d{1,3};){2,3}[N|S]",splitted[1]).group(0)
        if re.search("(?:\d{1,3};){2,3}[W|E]",splitted[1]):long=re.search("(?:\d{1,3};){2,3}[W|E]",splitted[1]).group(0)
        if lat and long:
            lat = str(round(convertlat(lat),3))
            long = str(round(convertlong(long),3))
            return [{"latitude": lat},{"longtitude": long}]
    if "convert" in splitted[1]:
        splitted[1] = str(convertToSI(splitted[1]))
    return [{splitted[0]:splitted[1]}]

def formatInformation(line,category):
    information = ""
    lineToRet = []
    informationDict = {}
    if category in categoriesToPrint:
        if len(line)>=9:
            lineToRet.append(category)
            aliasList = []
            name=line[0]
            if len(line[8]):
                informationDict = parseInfobox(line[8])
            for oneinfo in informationDict.items():
                if oneinfo[0] == 'name':
                    name=oneinfo[1]#if it has name, append it now
                if oneinfo[0] in aliases: #store aliases, but add them when we have all of them
                    aliasList.append(oneinfo[1])
            lineToRet.append(name)
            lineToRet.append('|'.join(aliasList))
            lineToRet.append(line[1])
            lineToRet.append(line[0])
            if category =="person":
                lineToRet.append(informationDict.get("birth_date", ""))
                lineToRet.append(informationDict.get("birth_place", ""))
                lineToRet.append(informationDict.get("death_date", ""))
                lineToRet.append(informationDict.get("death_place", ""))
                lineToRet.append(informationDict.get("occupation", ""))
                lineToRet.append(line[7]) #nationality
            if category=="country" or category == "settlement":
                lineToRet.append(informationDict.get("area", ""))
                lineToRet.append(informationDict.get("population", ""))
            if category == "watercourse":
                lineToRet.append(informationDict.get("length", ""))
                lineToRet.append(informationDict.get("area", ""))
                lineToRet.append(informationDict.get("streamflow", ""))
                lineToRet.append(informationDict.get("source1_location", ""))
            if category == "waterarea":
                lineToRet.append(informationDict.get("area", ""))
            if category[:4]=="geo:":
                latitude = informationDict.get("latitude", "")
                longtitude = informationDict.get("longtitude", "")
                if latitude and longtitude:
                    lineToRet.append(latitude)
                    lineToRet.append(longtitude)
            if category in ("geo:continent","geo:island"):
                lineToRet.append(informationDict.get("area", ""))
                lineToRet.append(informationDict.get("population", ""))

    return lineToRet

def classify(toKeep ,line):
    classified = {}
    if len(line[2]) and line[2]in toKeep['infobox']:
        resInfobox = classifiers['infobox'].predict([line[2]])[0]
        if resInfobox not in classified:
            classified[resInfobox] = 11
        else:
            classified[resInfobox] += 11

    if len(line[3]) and line[3]in toKeep['wikicategories']:
        resWikicategories = classifiers['wikicategories'].predict([line[3]])[0]
        if resWikicategories not in classified:
            classified[resWikicategories] = 1
        else:
            classified[resWikicategories] += 1

    if len(line[4]) and line[4]in toKeep['defwordStanford']:
        resStanforddefword = classifiers['defwordStanford'].predict([line[4]])[0]
        if resStanforddefword not in classified:
            classified[resStanforddefword] = 2
        else:
            classified[resStanforddefword] += 2
    if len(line[5]) and line[5]in toKeep['defwordSpaCy']:
        resSpaCydefword = classifiers['defwordSpaCy'].predict([line[5]])[0]
        if resSpaCydefword not in classified:
            classified[resSpaCydefword] = 2
        else:
            classified[resSpaCydefword] =+ 2
    if len(line[6]) and line[6]in toKeep['defphrSpaCy']:
        resSpaCydefphr = classifiers['defphrSpaCy'].predict([line[6]])[0]
        if resSpaCydefphr not in classified:
            classified[resSpaCydefphr] = 5
        else:
            classified[resSpaCydefphr] += 5
    if len(classified):
        maxValues = max(classified.items(), key=operator.itemgetter(1))
        if len([i for i,j in classified.items() if j == maxValues[1]])==1: #check if only one category has the highest score
            return maxValues[0]
    if onlyAllMatched==0:
        if len(classified):
            maxValues = max(classified.items(), key=operator.itemgetter(1))
            if len([i for i,j in classified.items() if j == maxValues[1]])==1: #check if only one category has the highest score
                return maxValues[0]
    else: #mode where all classifiers must give the same result
        if len(classified.keys())==1:
            try:
                return next(iter(classified))
            except:
                pass
    return ""

def trainClassifier(feature,toTrainFrom):
    filteredTrainData = filterLowOccurences(toTrainFrom[feature], feature)  # filter data with required occurences and limit number of training entries
    data = filteredTrainData[0]
    #print("DATA LEN:"+str(len(data)))
    target = filteredTrainData[1]
    toKeepTmp= filteredTrainData[2]
    #print(str(strftime("%m-%d %H:%M") + "TRAINING " + feature))
    classifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SVC(C=20.0, cache_size=200, coef0=0.0,
                    decision_function_shape='ovo', gamma=2, kernel='rbf',
                    max_iter=iterationsForTrainingClassifiers, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False, class_weight=None))])
    classifier.fit(data, target)
    failedClassifiers.append(feature)
    classifiers[feature] = classifier
    toKeep[feature] = toKeepTmp

def classifyPart(part):

    for line in part:
        if len(line) >= 8:
            line[4] = ' '.join(line[4].split("|")) #prepare the data for count vectorizer
            line[5] = ' '.join(line[5].split("|"))
            cat = str(classify(toKeep, line))
            if len(cat):


                toOut = '\t'.join(formatInformation(line,cat))
                if toOut:
                    toPrint.append(toOut.replace("\n",""))

    return 1

def splitList(l, parts):#splits list into smaller lists
    return [l[i::parts] for i in range(parts)]
#Now we have dictionary of categories with their articles in titlesInCategory, lets connect it with TSV file
titlesWithTSVData = {}
allLinesFromTSV2 = {}
lines = []
#print(str(strftime("%m-%d %H:%M")+"READING FILE"))
allLinesFromTSVList2 = []
radkyok = 0
with open(sys.argv[1], 'r') as tsvFile:
    for line in tsvFile:
        if len(line.split("\t"))==CORRECTNUMBEROFCOLUMNTS:
            splitted_line = line.split("\t")[0:9] #no need to store big unformatted strings and whole paragraph
            lines.append(splitted_line)
            allLinesFromTSVList2.append(splitted_line)
            allLinesFromTSV2[splitted_line[0]] = [splitted_line[1:]]
linesForLearning = getLinesForLearning(lines,hardLimitLines) #choose number of random lines that will be used for learning
preprocessedCategories = preprocessCategories(linesForLearning) #for those lines, preprocess categories
expandedTrainingSet = expandTrainingSet(iterationsForExpandingTrainigSet,preprocessedCategories,linesForLearning)


for line in linesForLearning: #load features from lines to learn from
    splitted_line = line
    title = splitted_line[0]
    if len(splitted_line)>6:
        titlesWithTSVData[splitted_line[0]] = {}
        titlesWithTSVData[splitted_line[0]]['infobox'] = splitted_line[2]
        titlesWithTSVData[splitted_line[0]]['wikicategories'] = splitted_line[3]
        titlesWithTSVData[splitted_line[0]]['defwordStanford'] = splitted_line[4]
        titlesWithTSVData[splitted_line[0]]['defwordSpaCy'] = splitted_line[5]
        titlesWithTSVData[splitted_line[0]]['defphrSpaCy'] = splitted_line[6]

categoriesAndDefwords = {}
toClassify ={'infobox': [], 'wikicategories': [], 'defwordStanford': [], 'defwordSpaCy': [], 'defphrSpaCy': []}
toExtractFeaturesTmp = {}
toExtractFeatures = []
toTrainFrom = {'infobox': [], 'wikicategories': [], 'defwordStanford': [], 'defwordSpaCy': [], 'defphrSpaCy': []}
allCategories = set(expandedTrainingSet.keys())
for title,data in titlesWithTSVData.items():
    detectedCategoryCatBox = []
    for category in allCategories:
        if data.get('infobox'):
            if data['infobox'] in expandedTrainingSet[category].get('infobox'):
                detectedCategoryCatBox.append(category)

        if data.get('wikicategories'):
            for onewikicategory in data.get('wikicategories').split('|'):
                if onewikicategory in expandedTrainingSet[category].get('wikicategories'):
                    detectedCategoryCatBox.append(category)

    if len(set(detectedCategoryCatBox))==1: #if we know the category of the line
        foundCategory = str(detectedCategoryCatBox[0])
        for feature in features:
            if len(' '.join(data[feature].split("|"))):
                toTrainFrom[feature].append({foundCategory:' '.join(data[feature].split("|"))})

target = {'infobox': [], 'wikicategories': [], 'defwordStanford': [], 'defwordSpaCy': [], 'defphrSpaCy': []}
data = {'infobox': [], 'wikicategories': [], 'defwordStanford': [], 'defwordSpaCy': [], 'defphrSpaCy': []}
toKeep = {'infobox': set(), 'wikicategories': set(), 'defwordStanford': set(), 'defwordSpaCy': set(), 'defphrSpaCy': set()}
i=0
classifierToTrain = {}
for feature in features: #train all classifiers in parallel
    classifierToTrain[feature] = Process(target = trainClassifier,args=(feature,toTrainFrom, ))
    failedClassifiers.append(feature)
for classifier in classifierToTrain:
    classifierToTrain[classifier].start()
for classifier in classifierToTrain:
    classifierToTrain[classifier].join()
partsOfWholeFile = splitList(allLinesFromTSVList2, int(sys.argv[4]))

listOfProcessing = []
for i in range(0,int(sys.argv[4])): #classify parts of TSV in parallel
    listOfProcessing.append(Process(target=classifyPart, args=(partsOfWholeFile[i],)))
for process in listOfProcessing:
    process.start()
for process in listOfProcessing:
    process.join()

toPrint = sorted(toPrint)#sort the result and print it
for line in toPrint:
    print(line)
