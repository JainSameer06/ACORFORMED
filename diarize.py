#from pyAudioAnalysis import convertToWav
#from pyAudioAnalysis import audioSegmentation
import sys
from subprocess import call
import numpy as np

pyAudioAnalysisPath = "/home/sameer/Downloads/pyAudioAnalysis-master"
sys.path.append(pyAudioAnalysisPath)
import convertToWav
import audioSegmentation
import audioAnalysis

def generateDiarization(videopath, filename, flsd, numSpeakers):
	convertToWav.main([videopath, "44100","2"])
	#audioFile = videopath + '/' + filename + ".wav"
	audioFile = videopath + '/' + 'combined.wav'
	print audioFile
	cls, mtStep = audioSegmentation.speakerDiarization(audioFile, numOfSpeakers = numSpeakers, LDAdim = 0, PLOT = True)
	#cls, mtStep = audioSegmentation.speakerDiarization(audioFile, numOfSpeakers = numSpeakers, mtSize=1.0, mtStep=0.5, stWin=0.01, LDAdim=0, PLOT=True)

	
	#spokenIntervals = audioSegmentation.silenceRemoval(audioFile, 0.05, 0.05, smoothWindow = 0.1, Weight = 0.6, plot = True)
	np.set_printoptions(threshold=np.nan)
	print mtStep, cls
	cls2 = []
	for i in range(len(cls)):
		for j in range(int(mtStep / 0.05)):
			cls2.append(cls[i])

	mtStep = mtStep/4
	cls = np.asarray(cls2)
	#print "new classes: ", cls

	spokenIntervals = audioSegmentation.silenceRemoval(audioFile, 0.05, 0.05, 0.1, 0.6, True)
	print spokenIntervals
	spokenSegIndex = 0
	nextDialogueStart = spokenIntervals[spokenSegIndex][0]
	nextDialogueStop = spokenIntervals[spokenSegIndex][1]

	"""
	i = 0
	while(i * mtStep < spokenIntervals[len(spokenIntervals) - 1][1]):
	    while(i * mtStep < nextDialogueStart):
	    	i += 1
	    	cls[i] = -1
	    while(i * mtStep < nextDialogueStop):
	    	i += 1

	    if(i < cls.size - 1 and spokenSegIndex < len(spokenIntervals) - 1):
	    	spokenSegIndex += 1
	    	nextDialogueStart = spokenIntervals[spokenSegIndex][0] 
	    	nextDialogueStop = spokenIntervals[spokenSegIndex][1]

	    if(i < cls.size):
	    	while(i < cls.size):
	    		cls[i] = -1
	    		i += 1
	    np.set_printoptions(threshold=np.nan)
	    print cls
	"""

	i = 0
	while(i < cls.size):
		#print 'iteration #', i, nextDialogueStart, nextDialogueStop,
		if(i * mtStep >= nextDialogueStart and i * mtStep < nextDialogueStop):
			#print 'Enters the if, i * mtStep = ', i * mtStep
			i += 1

			if(i * mtStep >= nextDialogueStop):
				spokenSegIndex += 1
				if(spokenSegIndex == len(spokenIntervals)):
					break
				else:	
					nextDialogueStart = spokenIntervals[spokenSegIndex][0]
					nextDialogueStop = spokenIntervals[spokenSegIndex][1]

		else:
			#print 'Enters the else, i * mtStep = ', i* mtStep
			"""
			if(i > 0 and (i - 1) * mtStep < nextDialogueStop):
				spokenSegIndex += 1
				if(spokenSegIndex == len(spokenIntervals)):
					break
				else:	
					nextDialogueStart = spokenIntervals[spokenSegIndex][0]
					nextDialogueStop = spokenIntervals[spokenSegIndex][1]
			"""

			cls[i] = -1
			i += 1
		#print "\n"
	np.set_printoptions(threshold=np.nan)
	print "cls: ", cls

	
	intervals = []
	intervals.append([])
	intervals.append([])

	i = 0
	while(i < len(cls)):
		if(cls[i] != -1):
			beginTime = i * mtStep
			while(i < len(cls) and cls[i] != -1):
				i += 1;
			endTime = i * mtStep
			if(i == len(cls)):
				break		#fix this
			intervals[int(cls[i])].append([beginTime, endTime])
		else:
			i += 1
	
	print spokenIntervals
	print intervals
	
	
	#audioAnalysis.main(["speakerDiarization", "-i", "/home/sameer/Projects/ACORFORMED/Data/Data/E6F/Casque/Video/E6F-Casque-Video-00025.wav", "--num", "2"])
	#audioAnalysisFilePath = pyAudioAnalysisPath + '/' + 'audioAnalysis.py'
	#call(["python", audioAnalysisFilePath, "speakerDiarization", "-i", audioFile, "--num", numSpeakers])

def main():
	"""
	videopath = raw_input("Enter the video path")
	filename = raw_input("Enter the file name")
	"""
	videopath = '/home/sameer/Projects/ACORFORMED/Data/Data/E6F/Casque/Video'
	filename = 'E6F-Casque-Video-00025'
	numSpeakers = eval(raw_input("Enter the number of speakers"))

	generateDiarization(videopath, filename, True, numSpeakers)

if __name__ == '__main__':
	main()