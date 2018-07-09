import sys
import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
from wavSplitter import splitInThree

sp_globPath = "/home/sameer/Downloads/sppas-1.8.6/sppas/src"
sys.path.append(sp_globPath)
from annotations.IPUs.ipusseg import sppasIPUs



def convertToMono(audioFileLoc, audioFileName):
	audioFile = os.path.join(audioFileLoc, audioFileName)
	fileName, fileExt = os.path.splitext(audioFileName)

	stereo = AudioSegment.from_wav(audioFile)
	mono = stereo.set_channels(1)

	monoFileName = os.path.join(audioFileLoc, fileName + "_mono" + fileExt)
	monoFile = mono.export(monoFileName, format="wav")
	return monoFileName

def avgIPU_length(audioFileLoc, audioFileName):

	f = convertToMono(audioFileLoc, audioFileName)
	#f = os.path.join(audioFileLoc, audioFileName)
	out_dir = os.path.join(audioFileLoc, "outputDir", audioFileName)

	IPUobj = sppasIPUs()
	IPUobj.run(audiofile = f, trsinputfile=None, trstieridx=None, ntracks=None, diroutput=out_dir, tracksext="xra", trsoutput="output.xra")

	index_file = os.path.join(out_dir, "index.txt")

	"""
	with open(index_file) as ind:
		summary = ind.readlines()
	"""

	summary = pd.read_csv(index_file, sep = ' ', header = None)
	summary = summary.dropna()
	print summary

	IPU_lengths = summary[1] - summary[0]
	return IPU_lengths.mean()


def IPUdriver(audioFilePath, splitUp):
	splitInThree(audioFilePath,splitUp)

	#print os.path.join(os.path.dirname(audioFilePath), "IPUtemp")
	avg_begin = avgIPU_length(os.path.join(os.path.dirname(audioFilePath), "IPUtemp"), "begin.wav")
	avg_mid = avgIPU_length(os.path.join(os.path.dirname(audioFilePath), "IPUtemp"), "middle.wav")
	avg_end = avgIPU_length(os.path.join(os.path.dirname(audioFilePath), "IPUtemp"), "end.wav")

	avgIPUarr = np.array([avg_begin, avg_mid, avg_end])

	print avgIPUarr
	print avgIPUarr[0], avgIPUarr[1], avgIPUarr[2]
	return avgIPUarr

#compute standard deviation


#avgIPU_length("/home/sameer/Projects/ACORFORMED/Data/Data/E6F/Casque/Video", "E6F-Casque-Video-00025.wav")
#convertToMono("/home/sameer/Projects/ACORFORMED/Data/Data/E6F/Casque/Video", "E6F-Casque-Video-00025.wav")
#IPUdriver("/home/sameer/Projects/ACORFORMED/Data/corpus2017/E7A/Casque/data/E7A-02-Casque-micro.wav", [0.15, 0.70, 0.15])