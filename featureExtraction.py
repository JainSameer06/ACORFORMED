from ipuseg import IPUdriver
from entropy import videoEntropyMatrix
import os
import sys
import shutil
from pos import POSfeatures, avgSentenceLength, PunctuatedFile
from wavSplitter import duration
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

def filePaths():
    #the function collects 4 files for each sample from the two sources in the paths below. The 4 files are: unity coordinates, xra transcription, wav participant mic audio, and the mp4 extracted from the video.
    gregCorpusPath = "/home/sameer/Projects/ACORFORMED/Data/corpus2017"
    profBCorpusPath = "/home/sameer/Projects/ACORFORMED/Data/Data"

    outerArr = []
    for subdir in os.listdir(gregCorpusPath):
        #print subdir
        if(os.path.isdir(os.path.join(gregCorpusPath, subdir))):
            for envDir in os.listdir(os.path.join(gregCorpusPath, subdir)):
                #print envDir
                innerArr = []
                if(os.path.isdir(os.path.join(gregCorpusPath, subdir,envDir))) and (os.path.isdir(os.path.join(profBCorpusPath, subdir, envDir))):
                    #print os.path.join(subdir, envDir)
                    for dirs, subdirs, files in os.walk(os.path.join(gregCorpusPath, subdir, envDir), topdown = True, onerror = None, followlinks = False):
                        for file in files:
                            name, exten = os.path.splitext(file)

                            if(os.path.basename(os.path.normpath(dirs)) == 'data') and (exten == ".wav"):
                                innerArr.append(os.path.join(dirs, file))
                                for trsFile in os.listdir(os.path.join(dirs, "asr-trans")):
                                    if trsFile.endswith(".xra"):
                                        innerArr.append(os.path.join(dirs, "asr-trans", trsFile))

                    for dirs, subdirs, files in os.walk(os.path.join(profBCorpusPath, subdir, envDir), topdown = True, onerror = None, followlinks = False):
                        for file in files:
                            name, exten = os.path.splitext(file)    

                            if(os.path.basename(os.path.normpath(dirs)) == 'Video') and (exten == ".mp4"):
                                #print os.path.join(dirs, file)
                                innerArr.append(os.path.join(dirs, file))
                            if(os.path.basename(os.path.normpath(dirs)) == 'Unity') and (exten == ".txt"):
                                #print os.path.join(dirs, file)
                                innerArr.append(os.path.join(dirs, file))                
                    outerArr.append(innerArr)
    return outerArr

def computePOStags(pathsList, splitratios):
    crashlist = ["/home/sameer/Projects/ACORFORMED/Data/corpus2017/N1A/Casque/data/asr-trans/N1A-02-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N2B/PC/data/asr-trans/N2B-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E6F/Cave/data/asr-trans/E6F-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N15C/Casque/data/asr-trans/N15C-01-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N3C/Casque/data/asr-trans/N3C-01-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E2B/Cave/data/asr-trans/E2B-02-Cave-micro.E2B-latin1.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/PC/data/asr-trans/N6F-05-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/PC/data/asr-trans/N6F-01-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/Casque/data/asr-trans/N6F-04-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/Casque/data/asr-trans/N6F-02-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N21C/PC/data/asr-trans/N21C-03-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N22D/PC/data/asr-trans/N22D-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N22D/PC/data/asr-trans/N22D-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N22D/Cave/data/asr-trans/N22D-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N4D/Casque/data/asr-trans/N4D-01-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E7A/Casque/data/asr-trans/E7A-02-Casque-micro.E1-5.xra"]
    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if(fileext == ".xra" and path not in crashlist):# and path not in throughList):
                for wavPath in paths:
                    _, extWav = os.path.splitext(wavPath)
                    if(extWav == ".wav"):
                        #print path, wavPath
                        POSfreqArr = POSfeatures(path, wavPath, splitratios)
                        
                        envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
                        candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))))
                        dest = os.path.join("/home/sameer/Projects/ACORFORMED/Data/Data", candidate, envType, "Features", "pos.txt")
                        np.savetxt(dest, POSfreqArr)


def computeSentenceLengths(pathsList, splitratios):
    #code cleanup: computeSentenceLengths and computePOStags do redundant work. Could be improved
    crashlist = ["/home/sameer/Projects/ACORFORMED/Data/corpus2017/N1A/Casque/data/asr-trans/N1A-02-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N2B/PC/data/asr-trans/N2B-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E6F/Cave/data/asr-trans/E6F-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N15C/Casque/data/asr-trans/N15C-01-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N3C/Casque/data/asr-trans/N3C-01-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E2B/Cave/data/asr-trans/E2B-02-Cave-micro.E2B-latin1.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/PC/data/asr-trans/N6F-05-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/PC/data/asr-trans/N6F-01-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/Casque/data/asr-trans/N6F-04-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/Casque/data/asr-trans/N6F-02-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N21C/PC/data/asr-trans/N21C-03-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N22D/PC/data/asr-trans/N22D-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N22D/PC/data/asr-trans/N22D-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N22D/Cave/data/asr-trans/N22D-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N4D/Casque/data/asr-trans/N4D-01-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E7A/Casque/data/asr-trans/E7A-02-Casque-micro.E1-5.xra"]

    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if(fileext == ".xra" and path not in crashlist):# and path not in throughList):
                for wavPath in paths:
                    _, extWav = os.path.splitext(wavPath)
                    if(extWav == ".wav"):
                        #print path, wavPath
                        sentenceLengthArray = avgSentenceLength(path, wavPath, splitratios, "/home/sameer/Downloads/sppas-1.8.6", "1.8.6")
                        envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
                        candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))))
                        dest = os.path.join("/home/sameer/Projects/ACORFORMED/Data/Data", candidate, envType, "Features", "slength.txt")
                        np.savetxt(dest, sentenceLengthArray)
            

def computeEntropies(pathsList, splitratios):
    #the function computes entropies for all unity files, replacing nan for non-working trackers
    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if(fileext == ".txt"):
                entArr = videoEntropyMatrix(path, splitratios)

                nvType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
                candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))))
                dest = os.path.join("/home/sameer/Projects/ACORFORMED/Data/Data", candidate, envType, "Features", "entropy.txt")
                np.savetxt(dest, entArr)

def computeIPUlengths(pathsList, splitratios):
    crashlist = ["/home/sameer/Projects/ACORFORMED/Data/corpus2017/N21C/PC/data/N21C-03-PC-micro.wav"]#, "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E13A/Casque/data/E13A-02-Casque-micro.wav", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E13A/Cave/data/E13A-01-Cave-micro.wav"]
    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if(fileext == ".wav" and path not in crashlist):
                entArr = IPUdriver(path, splitratios)
    
                envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(path))))
                candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
                dest = os.path.join("/home/sameer/Projects/ACORFORMED/Data/Data", candidate, envType, "Features", "ipu.txt")
                np.savetxt(dest, entArr)

def sum_nan_arrays(a, b):
    #from https://stackoverflow.com/questions/42209838/treat-nan-as-zero-in-numpy-array-summation-except-for-nan-in-all-arrays
    ma = np.isnan(a)
    mb = np.isnan(b)
    return  np.where(ma & mb, np.NaN, np.where(ma, 0, a) + np.where(mb, 0, b))

def updateFrequencies(current, newArr):
    #not an optimal solution, attempt improvement
    for i in range(newArr.shape[0]):
        for j in range(newArr.shape[1]):
            if(not np.isnan(newArr[i][j])):
                current[i][j] += 1
    return current

def removeNaN():
    #the following function replaces the nan values in the entropy matrices with average values
    profBCorpusPath = "/home/sameer/Projects/ACORFORMED/Data/Data"
    sums = np.zeros((15, 3))            #this array eventually contains the sums of the values of the valid entries of each feature 
    currentFreq = np.zeros((15,3))      #supporting array to count the number of valid entries for each feature
    count = 0
    for dirs, subdirs, files in os.walk(profBCorpusPath, topdown = True, onerror = None, followlinks = False):  #this loop computes the sums and currentFreq arrays
        for file in files:
            if file == "entropy.txt" and os.path.basename(os.path.normpath(dirs)) == "Features":
                completePath = os.path.join(dirs, file)
                sums = sum_nan_arrays(averages, np.loadtxt(completePath))
                currentFreq = updateFrequencies(currentFreq, np.loadtxt(completePath))
                count += 1

    averages = np.divide(sums, currentFreq)     #the average array is used in the next loop to replace nan values with corresponding averages

    for dirs, subdirs, files in os.walk(profBCorpusPath, topdown = True, onerror = None, followlinks = False):
        for file in files:
            if file == "entropy.txt" and os.path.basename(os.path.normpath(dirs)) == "Features":
                completePath = os.path.join(dirs, file)
                origMat = np.loadtxt(completePath)
                ma = np.isnan(origMat)

                modifiedMat = np.where(ma, averages, origMat)
                np.savetxt(os.path.join(dirs, "usableEntropies.txt"), modifiedMat)

def extractClass(dframe, candidate, env):
    #the function maps a score in [1, 2) to class '1', a score in [2, 4) to class '2', and a score in [4, 5] to class '3'
    #yet to optimize
    print candidate, env
    for i in range(dframe.shape[0]):
        #print dframe["Candidate"][i], dframe["Environment"][i]
        if(dframe["Candidate"][i] == candidate) and (dframe["Environment"][i] == env):

            if(dframe["Value"][i] >= 1 and dframe["Value"][i] < 2.5):
                return dframe["Value"][i], 1
            elif(dframe["Value"][i] >= 2.5 and dframe["Value"][i] < 3.5):
                return dframe["Value"][i], 2
            else:
                return dframe["Value"][i], 3

def movePOSfiles(pathsList):
    #following copies pos txt files to the features folder, which were generated in the asr-trans folder, to the Features folder
    for files in pathsList:
        for file in files:
            name, exten = os.path.splitext(file)
            if exten == ".xra":
                posFile = os.path.join(os.path.dirname(file), name + ".txt")
                if os.path.isfile(posFile):
                    envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(posFile)))))
                    candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(posFile))))))
                    dest = os.path.join("/home/sameer/Projects/ACORFORMED/Data/Data", candidate, envType, "Features", "pos.txt")
                    shutil.copy(posFile, dest)

def moveIPUfiles():
    for dirs, subdirs, files in os.walk("/home/sameer/Projects/ACORFORMED/Data/corpus2017", topdown = True, onerror = None, followlinks = False):
        if(os.path.basename(os.path.normpath(dirs)) == "Features" and os.path.isfile(os.path.join(dirs, 'ipu.txt'))):
            envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.join(dirs, 'ipu.txt')))))
            candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(dirs, 'ipu.txt'))))))
            target = os.path.join("/home/sameer/Projects/ACORFORMED/Data/Data", candidate, envType,"Features")
            if os.path.exists(target):
                print "yes"
                shutil.copy(os.path.join(dirs, "ipu.txt"), target)

def compressEntropy(entMat):
    head = np.mean(entMat[0:3, :], axis = 0)
    leftW = np.mean(entMat[3:6, :], axis = 0)
    rightW = np.mean(entMat[6:9, :], axis = 0)
    leftE = np.mean(entMat[9:12, :], axis = 0)
    rightE = np.mean(entMat[12:15, :], axis = 0)

    compressed = np.vstack((head, leftW, rightW, leftE, rightE))
    return compressed

def prepareMatrix():
    featureMat = []

    candidateVec = []
    envVec = []
    expertVec = []
    durationVec = []

    classVec = []
    pScoreVec = []
    copClassVec = []
    copScoreVec = []
    print "in prepare"
    for dirs, subdirs, files in os.walk("/home/sameer/Projects/ACORFORMED/Data/Data", topdown = True, onerror = None, followlinks = False):
        if(os.path.basename(os.path.normpath(dirs)) == "Features"):
            entropyFile = os.path.join(dirs, "usableEntropies.txt")
            posFile = os.path.join(dirs, "pos.txt")
            slengthFile = os.path.join(dirs, "slength.txt")
            ipuFile = os.path.join(dirs, "ipu.txt")

            if(os.path.isfile(entropyFile) and os.path.isfile(posFile) and os.path.isfile(slengthFile) and os.path.isfile(ipuFile)):
                #print "in condition"
                expert = 1
                envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(posFile))))
                candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(posFile)))))
                candidateLevel = candidate[0]
                
                labels = pd.read_excel("/home/sameer/Projects/ACORFORMED/Data/presence.xlsx")
                labelsCo = pd.read_excel("/home/sameer/Projects/ACORFORMED/Data/copresence.xlsx")

                if(extractClass(labels, candidate, envType)) is None:
                    continue

                if(extractClass(labelsCo, candidate, envType)) is None:
                    continue
                

                audioTarget = os.path.join("/home/sameer/Projects/ACORFORMED/Data/corpus2017", candidate, envType, "data")

                for file in os.listdir(audioTarget):
                    #print file, "here"
                    if(file.endswith(".wav")):
                        #print file
                        dur = duration(os.path.join(audioTarget, file))
                        if(durationVec == []):
                            durationVec = [dur]
                        else:
                            durationVec = np.append(durationVec, dur)
                        break

                print len(durationVec)

                entMat = np.loadtxt(entropyFile)
                compressedEntMat = compressEntropy(entMat)

                posMat = np.loadtxt(posFile)
                slengthMat = np.loadtxt(slengthFile)
                ipuMat = np.loadtxt(ipuFile)
                #print ipuMat
                combinedMat = np.vstack((compressedEntMat, posMat, slengthMat, ipuMat))

                adjecPOSVec = posMat[0, :]
                advPOSVec = posMat[1, :]

                sum1 = np.add(adjecPOSVec, advPOSVec)

                conjPOSVec = posMat[3, :]
                prepPOSVec = posMat[6, :]
                pronPOSVec = posMat[7, :]

                sum2 = np.add(conjPOSVec,prepPOSVec, pronPOSVec)
                
                allSum = np.sum(posMat, axis = 0)

                ratio1 = np.divide(sum1, allSum)
                ratio2 = np.divide(sum2, allSum)


                combinedVec = combinedMat.flatten()

                if(candidate[0] == 'N'):
                    expert = 0

                if(featureMat == []):
                    featureMat = combinedVec
                    pScore, pClass = extractClass(labels, candidate, envType)
                    pScoreVec = [pScore]
                    classVec = [pClass]

                    copScore, copClass = extractClass(labelsCo, candidate, envType)
                    copScoreVec = [copScore]
                    copClassVec = [copClass]

                    candidateVec = [candidate]
                    envVec = [envType]
                    expertVec = [expert]

                    rat1Vec = ratio1
                    rat2Vec = ratio2
                else:
                    #print "in this"
                    featureMat = np.vstack((featureMat, combinedVec))
                    pScore, pClass = extractClass(labels, candidate, envType)
                    copScore, copClass = extractClass(labelsCo, candidate, envType)

                    classVec = np.append(classVec, pClass)
                    copClassVec = np.append(copClassVec, copClass)
                    pScoreVec = np.append(pScoreVec, pScore)
                    copScoreVec = np.append(copScoreVec, copScore)

                    candidateVec = np.append(candidateVec, candidate)
                    envVec = np.append(envVec, envType)
                    expertVec = np.append(expertVec, expert)

                    rat1Vec = np.vstack((rat1Vec, ratio1))
                    rat2Vec = np.vstack((rat2Vec, ratio2))

    classes = np.stack((np.transpose(durationVec), np.transpose(pScoreVec), np.transpose(classVec), np.transpose(copScoreVec), np.transpose(copClassVec)), axis = -1)
    print featureMat.shape
    print classes.shape
    mat =  np.hstack((featureMat, rat1Vec, rat2Vec, classes))

    pdDump = pd.DataFrame(mat)
    dumpPath = "/home/sameer/Projects/ACORFORMED/Data/matrix.xlsx"
    pdDump.columns = ['Head_Entropy_Start', 'Head_Entropy_Mid', 'Head_Entropy_End', 'LeftWrist_Entropy_Start', 'LeftWrist_Entropy_Mid', 'LeftWrist_Entropy_End', 'RightWrist_Entropy_Start', 'RightWrist_Entropy_Mid', 'RightWrist_Entropy_End', 'LeftElbow_Entropy_Start', 'LeftElbow_Entropy_Mid', 'LeftElbow_Entropy_End', 'RightElbow_Entropy_Start', 'RightElbow_Entropy_Mid', 'RightElbow_Entropy_End', 'Freq_Adjective_Begin', 'Freq_Adjective_Mid', 'Freq_Adjective_End', 'Freq_Adverb_Begin', 'Freq_Adverb_Mid', 'Freq_Adverb_End','Freq_Auxiliary_Begin', 'Freq_Auxiliary_Mid', 'Freq_Auxiliary_End', 'Freq_Conjunction_Begin', 'Freq_Conjunction_Mid', 'Freq_Conjunction_End', 'Freq_Determiner_Begin', 'Freq_Determiner_Mid', 'Freq_Determiner_End', 'Freq_Noun_Begin', 'Freq_Noun_Mid', 'Freq_Noun_End', 'Freq_Preposition_Begin', 'Freq_Preposition_Mid', 'Freq_Preposition_End', 'Freq_Pronoun_Begin', 'Freq_Pronoun_Mid', 'Freq_Pronoun_End', 'Freq_Verb_Begin', 'Freq_Verb_Mid', 'Freq_Verb_End', 'Avg_SentenceLength_Begin', 'Avg_SentenceLength_Mid', 'Avg_SentenceLength_End', 'Avg_IPUlen_Begin', 'Avg_IPUlen_Middle', 'Avg_IPUlen_End', 'Ratio1_Begin', 'Ratio1_Mid', 'Ratio1_End', 'Ratio2_Begin', 'Ratio2_Mid', 'Ratio2_End', 'Duration', 'Presence Score', 'Presence Class', 'Co-presence Score', 'Co-presence Class']

    pdDump.insert(0, 'Candidate', candidateVec)
    pdDump.insert(1, 'Environment', envVec)
    pdDump.insert(2, 'Expert', expertVec)
    pdDump.to_excel(dumpPath, index = False)
    return mat

def randomForest(dataFile, modelTarget):
    samples = pd.read_excel(dataFile)

    samples_split = []
    if(modelTarget == "presence"):
        samples_split.append(samples[samples.PresenceClass == 1])
        samples_split.append(samples[samples.PresenceClass == 2])
        samples_split.append(samples[samples.PresenceClass == 3])

    elif(modelTarget == "copresence"):
        samples_split.append(samples[samples.CopresenceClass == 1])
        samples_split.append(samples[samples.CopresenceClass == 2])
        samples_split.append(samples[samples.CopresenceClass == 3])
    else:
        sys.exit("Invalid input. Please pick between presence and copresence")

    maxClassSize = max(samples_split[0].shape[0], samples_split[1].shape[0], samples_split[2].shape[0])

    upsampled = []

    for samples in samples_split:
        if(samples.shape[0] == maxClassSize):
            upsampled.append(samples)
        else:
            upsampled.append(resample(samples, replace=True, n_samples=maxClassSize, random_state=None))

    balanced_set = pd.concat(upsampled)

    forest = RandomForestClassifier()
    names = ("Expert", "Head_Entropy_Start", "Head_Entropy_Mid", "Head_Entropy_End", "Avg_HandEntropy_Begin", "Avg_HandEntropy_Mid", "Avg_HandEntropy_End", "Avg_SentenceLength_Begin", "Avg_SentenceLength_Mid", "Avg_SentenceLength_End", "Avg_IPUlen_Begin", "Avg_IPUlen_Middle", "Avg_IPUlen_End", "Ratio1_Begin", "Ratio1_Mid","Ratio1_End", "Ratio2_Begin", "Ratio2_Mid", "Ratio2_End", "Duration")

    """
    Xorig = np.nan_to_num(balanced_set.as_matrix(names))
    if(modelTarget == "presence"):
        yorig = np.array(balanced_set["PresenceClass"].tolist())

    else:
        yorig = np.array(balanced_set["CopresenceClass"].tolist())

    print modelTarget, "original"
    print "f1_micro", np.mean(cross_val_score(forest, Xorig, yorig, cv=10, scoring = "f1_micro"))
    print "precision", np.mean(cross_val_score(forest, Xorig, yorig, cv=10, scoring = "precision_micro"))
    print "recall", np.mean(cross_val_score(forest, Xorig, yorig, cv=10, scoring = "recall_micro"))
    """
    X = np.nan_to_num(balanced_set.as_matrix(names))

    if(modelTarget == "presence"):
        y = np.array(balanced_set["PresenceClass"].tolist())

    else:
        y = np.array(balanced_set["CopresenceClass"].tolist())

    #print X.shape
    print modelTarget
    print "f1_micro", np.mean(cross_val_score(forest, X, y, cv=10, scoring = "f1_micro"))
    print "precision", np.mean(cross_val_score(forest, X, y, cv=10, scoring = "precision_micro"))
    print "recall", np.mean(cross_val_score(forest, X, y, cv=10, scoring = "recall_micro"))
    #preds = cross_val_predict(forest, X, y, cv=10)
    #print metrics.accuracy_score(y, preds)

    forest.fit(X, y)
    print sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), names), reverse=True)
    
   #writer = pd.ExcelWriter('/home/sameer/Projects/ACORFORMED/Data/upsampled.xlsx')
    #balanced_set.to_excel(writer,'Sheet1')
    #writer.save()

def main():
    profBCorpusPath = "/home/sameer/Projects/ACORFORMED/Data/Data"

    pathsList = filePaths()
    splitratios = [0.15, 0.70, 0.15]
    """
    computePOStags(pathsList, splitratios)
    computeSentenceLengths(pathsList, splitratios)
    computeEntropies(pathsList, splitratios)
    removeNaN()
    computeIPUlengths(pathsList, splitratios)
    """
    
    #prepareMatrix()
    
    randomForest("/home/sameer/Projects/ACORFORMED/Data/mlMat.xlsx", "presence")
    print "\n\n"
    randomForest("/home/sameer/Projects/ACORFORMED/Data/mlMat.xlsx", "copresence")
    


    """
    splitPt = int(0.8 * featureMat.shape[0])
    print featureMat.shape
    print classVec
    rfc = RandomForestClassifier()
    rfc.fit(featureMat[0 : splitPt], classVec[0 : splitPt])

    print rfc.predict(featureMat[splitPt:])
    print classVec[splitPt:] 

    """
    """
    for paths in pathsList:
        for path in paths:
            _, fileext = os.path.splitext(path)
            if(fileext == ".txt"):
                videoEntropyMatrix(path, splitratios)
    """
    
    """
    crashlist = ["/home/sameer/Projects/ACORFORMED/Data/corpus2017/N1A/Casque/data/asr-trans/N1A-02-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N2B/PC/data/asr-trans/N2B-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E6F/Cave/data/asr-trans/E6F-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N15C/Casque/data/asr-trans/N15C-01-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N3C/Casque/data/asr-trans/N3C-01-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E2B/Cave/data/asr-trans/E2B-02-Cave-micro.E2B-latin1.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/PC/data/asr-trans/N6F-05-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/PC/data/asr-trans/N6F-01-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/Casque/data/asr-trans/N6F-04-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N6F/Casque/data/asr-trans/N6F-02-Casque-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N21C/PC/data/asr-trans/N21C-03-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N22D/PC/data/asr-trans/N22D-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N22D/PC/data/asr-trans/N22D-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N22D/Cave/data/asr-trans/N22D-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N4D/Casque/data/asr-trans/N4D-01-Casque-micro.E1-5.xra"]


    #N6 PC has two transcription files. Why? The one with "5" in the name crashed
    #N6 Casque also has 2 files.  Infinte loop occurs 

    throughList = ["/home/sameer/Projects/ACORFORMED/Data/corpus2017/N12F/PC/data/asr-trans/N12F-01-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N12F/Cave/data/asr-trans/N12F-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N12F/Casque/data/asr-trans/N12F-02-Casque-micro.E1-5.xra", 
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N1A/PC/data/asr-trans/N1A-03-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N1A/Cave/data/asr-trans/N1A-01-Cave-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N9C/PC/data/asr-trans/N9C-03-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N9C/Cave/data/asr-trans/N9C-02-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N9C/Casque/data/asr-trans/N9C-01-Casque-micro.E1-5.xra", 
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N2B/Cave/data/asr-trans/N2B-01-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N2B/Casque/data/asr-trans/N2B-03-Casque-micro.E1-5.xra", 
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N13A/PC/data/asr-trans/N13A-03-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N13A/Cave/data/asr-trans/N13A-01-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N13A/Casque/data/asr-trans/N13A-02-Casque-micro.E1-5.xra", 
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E6F/PC/data/asr-trans/E6F-01-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E6F/Casque/data/asr-trans/E6F-02-Casque-micro.E1-5.xra", 
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N10D/PC/data/asr-trans/N10D-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N10D/Cave/data/asr-trans/N10D-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N10D/Casque/data/asr-trans/N10D-01-Casque-micro.E1-5.xra", 
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N26D/PC/data/asr-trans/N26D-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N26D/Cave/data/asr-trans/N26D-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N26D/Casque/data/asr-trans/N26D-01-Casque-micro.E1-5.xra", 
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N15C/PC/data/asr-trans/N15C-03-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N15C/Cave/data/asr-trans/N15C-02-Cave-micro.E1-5.xra", 
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E3C/PC/data/asr-trans/E3C-03-PC-micro.E3C-latin1.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E3C/Cave/data/asr-trans/E3C-02-Cave-micro.E3C-latin1.xra", 
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E7A/PC/data/asr-trans/E7A-03-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E7A/Cave/data/asr-trans/E7A-01-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E7A/Casque/data/asr-trans/E7A-02-Casque-micro.E1-5.xra", 
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E2B/PC/data/asr-trans/E2B-03-PC-micro.E2B-latin1.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E2B/Casque/data/asr-trans/E2B-01-Casque-micro.E1A-latin1.xra"
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N18F/PC/data/asr-trans/N18F-01-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N18F/Cave/data/asr-trans/N18F-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N18F/Casque/data/asr-trans/N18F-02-Casque-micro.E1-5.xra", 
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E10D/PC/data/asr-trans/E10D-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E10D/Cave/data/asr-trans/E10D-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E10D/Casque/data/asr-trans/E10D-01-Casque-micro.E1-5.xra"
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E4D/PC/data/asr-trans/E4D-02-PC-micro.E4D-latin1.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E4D/Cave/data/asr-trans/E4D-03-Cave-micro.E4D-latin1.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E4D/Casque/data/asr-trans/E4D-01-Casque-micro.E1-3-latin1.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E5E/PC/data/asr-trans/E5E-01-PC-micro.E1-4-latin1.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E5E/Cave/data/asr-trans/E5E-02-Cave-micro.E5E-latin1.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E5E/Casque/data/asr-trans/E5E-03-Casque-micro.E5E-latin1.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E9C/PC/data/asr-trans/E9C-03-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E9C/Cave/data/asr-trans/E9C-02-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/E9C/Casque/data/asr-trans/E9C-01-Casque-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N14B/PC/data/asr-trans/N14B-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N14B/Cave/data/asr-trans/N14B-01-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N14B/Casque/data/asr-trans/N14B-03-Casque-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N12F/PC/data/asr-trans/N12F-01-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N12F/Cave/data/asr-trans/N12F-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N12F/Casque/data/asr-trans/N12F-02-Casque-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N21C/Cave/data/asr-trans/N21C-02-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N21C/Casque/data/asr-trans/N21C-01-Casque-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N7A/PC/data/asr-trans/N7A-03-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N7A/Cave/data/asr-trans/N7A-01-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N7A/Casque/data/asr-trans/N7A-02-Casque-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N19A/PC/data/asr-trans/N19A-03-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N19A/Cave/data/asr-trans/N19A-01-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N19A/Casque/data/asr-trans/N19A-02-Casque-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N20B/PC/data/asr-trans/N20B-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N20B/Cave/data/asr-trans/N20B-01-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N20B/Cave/data/asr-trans/N20B-03-Casque-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N23E/PC/data/asr-trans/N23E-01-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N23E/Cave/data/asr-trans/N23E-02-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N23E/Casque/data/asr-trans/N23E-03-Casque-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N24F/PC/data/asr-trans/N24F-01-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N24F/Cave/data/asr-trans/N24F-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N24F/Casque/data/asr-trans/N24F-02-Casque-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N22D/Casque/data/asr-trans/N22D-01-Casque-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N11E/PC/data/asr-trans/N11E-01-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N11E/Cave/data/asr-trans/N11E-02-Cave-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N17E/PC/data/asr-trans/N17E-01-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N17E/Cave/data/asr-trans/N17E-02-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N17E/Casque/data/asr-trans/N17E-03-Casque-micro.E1-5.xra",
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N16D/PC/data/asr-trans/N16D-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N16D/Cave/data/asr-trans/N16D-03-Cave-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N16D/Casque/data/asr-trans/N16D-01-Casque-micro.E1-5.xra"
    "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N4D/PC/data/asr-trans/N4D-02-PC-micro.E1-5.xra", "/home/sameer/Projects/ACORFORMED/Data/corpus2017/N4D/Cave/data/asr-trans/N4D-03-Cave-micro.E1-5.xra"]

    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if(fileext == ".xra" and path not in crashlist and path not in throughList):
                for wavPath in paths:
                    _, extWav = os.path.splitext(wavPath)
                    if(extWav == ".wav"):
                        #print path, wavPath
                        POSfreqArr = POSfeatures(path, wavPath, splitratios)
                        #POSfreqArr = POSfreqArr.astype(int)
                        np.savetxt(fileName + ".txt", POSfreqArr)
    """
    """
    for dirs, subdirs, files in os.walk("/home/sameer/Projects/ACORFORMED/Data/corpus2017", topdown = True, onerror = None, followlinks = False):
        for file in files:
            name, exten = os.path.splitext(file)  
            if(os.path.basename(os.path.normpath(dirs)) == 'asr-trans') and (exten == ".xra"):  
                shutil.copy(os.path.join(dirs, file), "/home/sameer/Projects/ACORFORMED/Transcription Files 2")
    """
if(__name__ == "__main__"):
    main()


