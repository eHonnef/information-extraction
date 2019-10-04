import os
import shutil
import signal
import subprocess
import sys
import time

from time import strftime
langFolder = ""
configuration = {}
dumpLocation = sys.argv[6]
sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] Control script started with PID: " + str(os.getpid())+", will process "+dumpLocation)+".\n")
sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] Reading config file.\n"))
try: #load config file
    cf = open(sys.argv[5])
    linesOfCf = cf.readlines()
    configuration['spacyModel'] = linesOfCf[1].strip()
    configuration['stanfordDir'] = linesOfCf[10].strip()
    pathToModel = linesOfCf[11].strip()
except:
    sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] Error while reading config file.\n"))
filesInLangDir = []
folderWithModel = '/'.join(pathToModel.split("/")[0:len(pathToModel.split("/"))-1])

sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] Preparing models for NLP.\n"))

subprocess.run(["jar", "xf", pathToModel],cwd=folderWithModel) #unpack jar with stanfrod model
for file in os.listdir(folderWithModel):
    if file.endswith(".properties"): propertiesFile = os.path.join(folderWithModel, file)

subprocess.run(["cp", pathToModel, configuration['stanfordDir']])


numberOfSubprocessesServer = int(sys.argv[1])
numberOfSubprocessesClients = int(sys.argv[2])
memoryForServer = int(sys.argv[3])
port = sys.argv[4]
sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] Models ready, starting splitting XML dump for client subprocesses with PID: " + str(os.getpid()) + ".\n"))

#create hosts file, which will be used during splitting process
with open('hosts.txt', 'w') as hostsFile:
    for i in range(0, numberOfSubprocessesClients):
        hostsFile.write(str(i)+'\n')
subprocess.call(['python', sys.argv[7], dumpLocation])

FNULL = open(os.devnull, 'w')
serverProc = subprocess.Popen(['java','-mx'+str(memoryForServer)+'g', '-cp' , "*", 'edu.stanford.nlp.pipeline.StanfordCoreNLPServer','-threads', str(numberOfSubprocessesServer),'-serverProperties',propertiesFile, '-port ',str(port), '-timeout ','30000','-quiet'],cwd=configuration['stanfordDir'],stderr=FNULL)

if serverProc.errors == None:
    sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] XML Splitted, Stanford NLP Server successfully started with PID: " + str(serverProc.pid) + ", listeninig on port "+str(port)+".\n"))
#spawn child processes that process the xml dump
sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] Starting child processes for parsing.\n"))

listOfProcesses = []
for i in range (0,numberOfSubprocessesClients):
    listOfProcesses.append(subprocess.Popen(['python', sys.argv[8], dumpLocation+"."+str(i),sys.argv[5],port]))
for process in listOfProcesses: #start all subprocesses
    process.communicate()
for process in listOfProcesses: #wait for all subprocesses before proceeding
    process.wait()
#get the generated files together and sort it alphabetically

sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] All clients finished, stopping Stanford NLP Server.\n"))
os.kill(serverProc.pid, signal.SIGINT) #stop the server for annotations
FNULL.close()
sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] Stanford NLP Server stopped, starting sorting whole TSV.\n"))
allLines = []
allLinesCol = []
for i in range (0,numberOfSubprocessesClients):
    with open(dumpLocation+"."+str(i)+".out", 'r') as onePart:
        for line in onePart:
            allLines.append(line)
allLinesSorted = sorted(allLines)

for i in range (0,numberOfSubprocessesClients):  #sentences ending with ":" will be printed to different file
    with open(dumpLocation+"."+str(i)+".col", 'r') as onePartCol:
        for line in onePartCol:
            allLinesCol.append(line)
allLinesSortedCol = sorted(allLinesCol)
sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] TSV sorted, cleaning up temp files.\n"))

with open(dumpLocation[:-4] + ".tsv", 'w') as outputFile:
    for line in allLinesSorted:
        outputFile.write(line)

with open(dumpLocation[:-4] + ".colon_ending_first_sentence", 'w') as outputFileCol:
    for line in allLinesSortedCol:
        outputFileCol.write(line)

import glob
# remove all temp files
fileList = glob.glob(dumpLocation+".*")
for filePath in fileList:
    try:
        os.remove(filePath)
    except OSError:
        pass
os.remove("hosts.txt")
os.remove(configuration['stanfordDir']+"/"+str(pathToModel.split("/")[len(pathToModel.split("/"))-1]))
shutil.rmtree(folderWithModel+"/edu")
shutil.rmtree(folderWithModel+"/META-INF")
os.remove(propertiesFile)
sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] Finished, TSV file saved as '"+dumpLocation[:-4] + ".tsv \n"))
sys.stderr.write("[INFO][CONTROL][" + str(strftime("%m-%d %H:%M:%S") + "] Finished, sentences ending with colon were saved to file: "+dumpLocation[:-4] + ".colon_ending_first_sentence \n"))
