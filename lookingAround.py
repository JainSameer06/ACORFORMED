import numpy as np
from collections import defaultdict
from matplotlib import pyplot

class Coordinate:
    def __init__(self, allLabelFile, subjectLabelFile, coordFile):
        self.allLabels = np.loadtxt(allLabelFile, dtype='string')
        self.subjectLabels = np.loadtxt(subjectLabelFile, dtype='string')
        self.d = defaultdict(list)
        self.coordArr = np.loadtxt(coordFile, skiprows=1)
        self.numTimeSteps = self.coordArr.shape[0]
        self.numSubjectLabels = self.subjectLabels.size
        self.filtArr = np.random.rand(self.numTimeSteps, self.numSubjectLabels)

    def dictArray(self):                            #returns a dictionary of arrays where each array corersponds to the time series of a subject coordinate point
        for i in range(self.allLabels.shape[0]):
            if(self.allLabels[i] in self.subjectLabels):
                self.d[allLabels[i]] = self.coordArr[:, i]

    def filteredCoords(self):
        i = 0
        for label in self.allLabels:
            if(label in self.d):
                self.filtArr[:, i] = self.d[label]  

    def graphCoord(self, label, dict):
        pyplot.plot(d[label])
        pyplot.ylabel(label)
        pyplot.show()

c = Coordinate("coordLabels", "subjectLabels", "CasqueP1")
c.dictArray()
print c.allLabels
print c.d