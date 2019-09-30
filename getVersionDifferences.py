import sys
titlesKB1 = set()
titlesKB2 = set()
entriesKB1 = {}
entriesKB2 = {}
with open(sys.argv[1] , 'r') as kb1:
    for line in kb1:
        splitted_line = line.strip().split("\t")
        if len(splitted_line)>=5:
            restofline = splitted_line[:4]
            if len(splitted_line)>5:
                restofline+=splitted_line[5:]
            entriesKB1[splitted_line[4]]=restofline
with open(sys.argv[2] , 'r') as kb2:
    for line in kb2:
        splitted_line = line.strip().split("\t")
        if len(splitted_line)>=5:
            restofline = splitted_line[:4]
            if len(splitted_line)>5:
                restofline+=splitted_line[5:]
            entriesKB2[splitted_line[4]]=restofline

titlesKB1 = set(entriesKB1.keys())
titlesKB2 = set(entriesKB2.keys())
titlesUnion = set(titlesKB1.union(titlesKB2))
addedTitles = set()
deletedTitles = set()
for title in titlesUnion:
    if title not in titlesKB1 and title in titlesKB2:
        addedTitles.add(title)
    elif title in titlesKB1 and title not in titlesKB2:
        deletedTitles.add(title)
    else:
        if entriesKB1.get(title)!=entriesKB2.get(title):
           if entriesKB1.get(title)[0] != entriesKB2.get(title)[0]:
               print("[DIF-TYPE]["+title+"]: '"+entriesKB1.get(title)[0]+"' => '"+entriesKB2.get(title)[0])
           if entriesKB1.get(title)[1:] != entriesKB2.get(title)[1:]:
               print("[DIF-INFO]["+title+"]: '"+'\t'.join(entriesKB1.get(title)[1:])+"' => '"+'\t'.join(entriesKB2.get(title)[1:]))
for title in addedTitles:
    print("[NEW]["+title+"]")
for title in deletedTitles:
    print("[DELETED]["+title+"]")


