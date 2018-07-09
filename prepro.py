import numpy as np
import os
from collections import defaultdict
from matplotlib import pyplot

def getCoordArray(coordFile):
    coordArr = np.loadtxt(coordFile, skiprows=1)
    return coordArr

def getLabels(labelFile):
    labels = np.loadtxt(labelFile, dtype='string')
    return labels


def getFilteredDict(coordArr, allLabels, subjectLabels):                #returns a dictionary of arrays where each array corersponds to the time series of a subject coordinate point
    d = defaultdict(list)

    for i in range(allLabels.size):
        if(allLabels[i] in subjectLabels):
            d[allLabels[i]] = coordArr[:, i]

    return d

def getNumTimesteps(coordArr):
    return coordArr.shape[0]

def getNumLabels(labels):
    return labels.size

def getFilteredArray(dictionary, allLabels, subjectLabels, coordArr):
    i = 0
    filtArr = np.random.rand(getNumTimesteps(coordArr), getNumLabels(subjectLabels))
    for label in allLabels:
        if(label in dictionary):
            filtArr[:, i] = dictionary[label]  
            i += 1
    return filtArr

def graphCoord(label, dict):
    pyplot.plot(dict[label], label = label)
    #pyplot.ylabel(label)
    pyplot.legend()


def splitAndCompare(array1, array2):
    step = 500

    frame1 = 0
    comparisons = 0
    distance = 0
    #while(frame1 < array1.shape[0]):
    while (frame1 < 5000):
        pass
        frame2 = 0
        #while(frame2 < array2.shape[0]):
        while(frame2 < 5000):
            tempDist, path = ucrdtw(array1[frame1:frame1 + step, :], array2[frame2:frame2 + step, :], 0.05, False)
            comparisons += 1
            distance = distance + tempDist
            frame2 = frame2 + step
        frame1 = frame1 + step

    distance = distance / comparisons
    print "distance", distance
    print "number of comparisons", comparisons
    return distance

def main():

    allLabels = getLabels("coordLabels")
    subjectLabels = getLabels("subjectLabels")

    for dirs, subdirs, files in os.walk("/home/sameer/Projects/ACORFORMED/Data/Data", topdown = True, onerror = None, followlinks = False):
        for file in files:
            name, exten = os.path.splitext(file)  
            if(os.path.basename(os.path.normpath(dirs)) == 'Unity') and (exten == ".txt"):  
                filepath = os.path.join(dirs, file)
                env = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(filepath))))
                sampleNum = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(filepath)))))

                coords = getCoordArray(filepath)
                diction = getFilteredDict(coords, allLabels, subjectLabels)
                print filepath
                print env
                print sampleNum
                graphCoord("HeadSubject_posx", diction)
                graphCoord("HeadSubject_posy", diction)
                graphCoord("HeadSubject_posz", diction)
                graphCoord("LeftElbowSubject_posx", diction)
                graphCoord("LeftElbowSubject_posy", diction)
                graphCoord("LeftElbowSubject_posz", diction)
                graphCoord("RightElbowSubject_posx", diction)
                graphCoord("RightElbowSubject_posy", diction)
                graphCoord("RightElbowSubject_posz", diction)
                graphCoord("LeftWristSubject_posx", diction)
                graphCoord("LeftWristSubject_posy", diction)
                graphCoord("LeftWristSubject_posz", diction)
                graphCoord("RightWristSubject_posx", diction)
                graphCoord("RightWristSubject_posy", diction)
                graphCoord("RightWristSubject_posz", diction)

                pyplot.savefig(os.path.join("/home/sameer/Projects/ACORFORMED/Transcription Files", sampleNum + "_" + env + ".png"))
                pyplot.close()


    

    """
    coords = getCoordArray("E3C manque video-Casque-Unity-out_record_DATE17-4-21_12-16-59.txt")
    diction = getFilteredDict(coords, allLabels, subjectLabels)
    graphCoord("HeadSubject_posx", diction)
    graphCoord("HeadSubject_posy", diction)
    graphCoord("HeadSubject_posz", diction)
    pyplot.title("E3C")
    pyplot.show()


    coords = getCoordArray("E4D-Casque-Unity-out_record_DATE17-4-21_13-35-20.txt")
    diction = getFilteredDict(coords, allLabels, subjectLabels)
    graphCoord("HeadSubject_posx", diction)
    graphCoord("HeadSubject_posy", diction)
    graphCoord("HeadSubject_posz", diction)
    pyplot.title("E4D")
    pyplot.show()

    coordinateArrayRef = normalize(getCoordArray("casref.txt"), axis = 0)
    coordinateArrayCas5 = normalize(getCoordArray("castest.txt"), axis = 0)
    coordinateArrayCav5 = normalize(getCoordArray("cavtest.txt"), axis = 0)
    coordinateArrayPC5 = normalize(getCoordArray("pctest.txt"), axis = 0)

    dictionary = getFilteredDict(coordinateArrayRef, allLabels, subjectLabels)
    filteredArray = getFilteredArray(dictionary, allLabels, subjectLabels, coordinateArrayRef)

    diccas5 = getFilteredDict(coordinateArrayCas5, allLabels, subjectLabels)
    filteredArraycas = getFilteredArray(diccas5, allLabels, subjectLabels, coordinateArrayCas5)

    diccav5 = getFilteredDict(coordinateArrayCav5, allLabels, subjectLabels)
    filteredArraycav = getFilteredArray(diccav5, allLabels, subjectLabels, coordinateArrayCav5)
    
    dicpc5 = getFilteredDict(coordinateArrayPC5, allLabels, subjectLabels)
    filteredArraypc = getFilteredArray(dicpc5, allLabels, subjectLabels, coordinateArrayPC5)

    
    print filteredArray.shape

    

    print splitAndCompare(coordinateArrayRef, coordinateArrayCas5)
    print splitAndCompare(coordinateArrayRef, coordinateArrayCav5)
    print splitAndCompare(coordinateArrayRef, coordinateArrayPC5)

    """


if __name__ == '__main__':
    main()