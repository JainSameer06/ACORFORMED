from __future__ import division
import sys
import os
import subprocess
from pydub import AudioSegment
from collections import defaultdict
import numpy as np


sp_globPath = "/home/sameer/Downloads/sppas-1.8.6/sppas/src"
greg_path = "/home/sameer/Downloads/Gregoire SPPAS Scripts/"


sys.path.append(sp_globPath)
sys.path.append(greg_path)

import annotationdata.aio
from annotationdata import Transcription
from annotationdata import Tier
from annotationdata import Annotation
from annotationdata import Label
from annotationdata import TimePoint
from annotationdata import TimeInterval

def alignmentFile(transcriptionFile, wavFile, sppaspath, sppasver):
	"""
	Inputs 	:	xra transcription
				wav audio
				SPPAS path
				SPPAS version
	Output 	: 	eaf transcription aligned with audio
	"""

	fileName, fileExt = os.path.splitext(transcriptionFile)
	print subprocess.check_output(["python", os.path.join(greg_path, "sppas_afterClosedCap.py"), transcriptionFile, "-D", sppaspath, "-V", sppasver])
	elanFile = os.path.join(fileName + ".eaf")		#ELAN file which created by the above call

	trs = annotationdata.aio.read(elanFile)

	#The SPPAS normalization routine works on the tier with the name 'transcription', or the first tier with the string 'trans' occuring in the name. 
	#Hence, the tier named 'ASR-Revised', on which normalization and subsequent processing is desired, is renamed. So is the tier 'ASR-Transcription', which is to be left untouched.

	for tier in trs:
		if(tier.GetName() == "ASR-Transcription"):
			tier.SetName("ASR-Orig")
		if(tier.GetName() == "ASR-Revised"):
			tier.SetName("transcription")			
	annotationdata.aio.write(elanFile, trs)

	wavFileName, wavExt = os.path.splitext(wavFile)
	tokFileName = wavFileName +"-token.eaf"
	phonFileName = wavFileName + "-phon.eaf"
	alignFileName = wavFileName + "-palign.eaf"

	#Subprocess call to create tokenized transcription file.
	print subprocess.check_output([os.path.join(sppaspath, "sppas", "bin", "tokenize.py"), '-r', os.path.join(sppaspath, "resources", "vocab", "fra.vocab"), "-i", elanFile, "-o", tokFileName])		
	
	#Subprocess call to create phonetized transcription file.	
	print subprocess.check_output([os.path.join(sppaspath, "sppas", "bin", "phonetize.py"), '-r', os.path.join(sppaspath, "resources", "dict", "fra.dict"), "-i", tokFileName, "-o", phonFileName])		

	#Subprocess call to create transcription file aligned with audio.
	print subprocess.check_output([os.path.join(sppaspath, "sppas", "bin", "alignment.py"), '-w', wavFile, "-i", phonFileName, "-I", tokFileName, "-o", alignFileName, "-r", os.path.join(sppaspath, "resources", "models", "models-fra")])		

	return alignFileName

def POStaggedFile(alignFileName):
	#Inputs 	: eaf transcription aligned with audio
	#Output 	: eaf transcription with a tier for part-of-speech labels

	print subprocess.check_output(["/home/sameer/MarsaTag/MarsaTag-UI.sh", '-cli', '-pt', "TokensAlign", "-oral", "-P", "-p", "lpl-oral-no-punct", "-r", "elan-lite", "-w", "elan-lite", "-in-ext", ".eaf", "--out-ext", "-marsatag.eaf", alignFileName])

	fileName, fileExt = os.path.splitext(alignFileName)
	return os.path.join(fileName + "-marsatag.eaf")

def PunctuatedFile(alignFileName):
	#Inputs 	: eaf transcription aligned with audio
	#Output 	: eaf transcription with a tier for part-of-speech and punctuation labels

	print subprocess.check_output(["/home/sameer/MarsaTag/MarsaTag-UI.sh", '-cli', '-pt', "TokensAlign", "-oral", "-P", "-p", "lpl-oral-with-punct", "-r", "elan-lite", "-w", "elan-lite", "-in-ext", ".eaf", "--out-ext", "-marsatagPunc.eaf", alignFileName])

	fileName, fileExt = os.path.splitext(alignFileName)
	return os.path.join(fileName + "-marsatagPunc.eaf")		

def avgSentenceLength(transcriptionFile, wavFile, splitUp, sppaspath, sppasver):
	"""
	Inputs 	:	xra transcription
				wav audio 
				phase-wise split of the interaction in the form of a list with 3 elements adding to 1
				SPPAS path
				SPPAS version
	#Output :	3-element numpy array contining average sentence lengths of each phase
	"""

	segment = AudioSegment.from_file(wavFile)												
	duration = segment.duration_seconds

	#splitPoint_1 : time (in seconds) at which the first phase ends and the second begins
	splitPoint_1 = (splitUp[0]) * duration

	#splitPoint_2 : time (in seconds) at which the second phase ends and the third begins	
	splitPoint_2 = (splitUp[0] + splitUp[1]) * duration

	taggedTransFile = PunctuatedFile(alignmentFile(transcriptionFile, wavFile, sppaspath, sppasver))

	trs = annotationdata.aio.read(taggedTransFile)
	tier = trs.Find("category", case_sensitive=False)

	#sentenceLength : temporary variable counting the length of each sentence from one punctuation annotation to the next 
	sentenceLength = 0

	#sentenceCount : 3-element array storing the number of sentences in each phase 
	sentenceCount = np.zeros(3)

	#numWords : 3-element array storing the number of words in each phase
	numWords = np.zeros(3)
	
	for annotation in tier:
		if(annotation.GetLabel().GetValue() == "punctuation"):
			if (annotation.GetLocation().GetBeginMidpoint() < splitPoint_1):
				sentenceCount[0] += 1
				numWords[0] = numWords[0] + sentenceLength
				sentenceLength = 0

			elif (annotation.GetLocation().GetBeginMidpoint() < splitPoint_2):
				sentenceCount[1] += 1
				numWords[1] = numWords[1] + sentenceLength
				sentenceLength = 0
			
			else:
				sentenceCount[2] += 1
				numWords[2] = numWords[2] + sentenceLength
				sentenceLength = 0

		else:
			sentenceLength += 1

	#avgLengths : 3-element array storing the average sentence length in each phase
	avgLengths = np.zeros(3)

	for i in range(3):
		if(sentenceCount[i] == 0):
			avgLengths[i] = 0
		else:
			avgLengths[i] = numWords[i] / sentenceCount[i]
	return avgLengths


def POSfreq(taggedTransFile, wavFile, splitUp):			#takes in the file generated by MarsaTag, along with a 3-element array for splitup, and returns a list of dicionaries with keys as parts-of-speech and values as their frequencies.

	dictList = []
	for i in range(3):
		dictList.append(defaultdict(list))
	segment = AudioSegment.from_file(wavFile)												
	duration = segment.duration_seconds
	#duration_ms = duration * 1000

	splitPoint_1 = (splitUp[0]) * duration
	splitPoint_2 = (splitUp[0] + splitUp[1]) * duration


	print "s1 : ", splitPoint_1
	print "s2 : ", splitPoint_2

	trs = annotationdata.aio.read(taggedTransFile)

	for tier in trs:
		if(tier.GetName() == "category"):
			break

	for annotation in tier:
		print annotation.GetLocation().GetBeginMidpoint() 
		print annotation.GetLabel().GetValue()

		if(annotation.GetLocation().GetBeginMidpoint() < splitPoint_1):
			if(not dictList[0][annotation.GetLabel().GetValue()]):
				dictList[0][annotation.GetLabel().GetValue()] = 1
			else:
				dictList[0][annotation.GetLabel().GetValue()] += 1
		elif(annotation.GetLocation().GetBeginMidpoint() < splitPoint_2):
			if(not dictList[1][annotation.GetLabel().GetValue()]):
				dictList[1][annotation.GetLabel().GetValue()] = 1
			else:
				dictList[1][annotation.GetLabel().GetValue()] += 1
		else:
			if(not dictList[2][annotation.GetLabel().GetValue()]):
				dictList[2][annotation.GetLabel().GetValue()] = 1
			else:
				dictList[2][annotation.GetLabel().GetValue()] += 1

	return dictList

def POSfeatures(transcriptionFile, wavFile, splitUp, sppaspath, sppasver):
	"""
	Inputs 	:	xra transcription
				wav audio 
				phase-wise split of the interaction in the form of a list with 3 elements adding to 1
				SPPAS path
				SPPAS version
	#Output :	9x3 numpy array contining the frequency of each POS tag in each phase (with rows arranged in lexicographic order of tag name)
	"""
	
	posFile = POStaggedFile(alignmentFile(transcriptionFile, wavFile, sppaspath, sppasver))
	POSdict = POSfreq(posFile, wavFile, splitUp)
	features = np.zeros((9, 3))

	for i in range(3):
		if(POSdict[i]["adjective"]):
			features[0][i] = POSdict[i]["adjective"]
		if(POSdict[i]["adverb"]):
			features[1][i] = POSdict[i]["adverb"]
		if(POSdict[i]["auxiliary"]):
			features[2][i] = POSdict[i]["auxiliary"]
		if(POSdict[i]["conjunction"]):
			features[3][i] = POSdict[i]["conjunction"]
		if(POSdict[i]["determiner"]):
			features[4][i] = POSdict[i]["determiner"]
		if(POSdict[i]["noun"]):
			features[5][i] = POSdict[i]["noun"]
		if(POSdict[i]["preposition"]):
			features[6][i] = POSdict[i]["preposition"]
		if(POSdict[i]["pronoun"]):
			features[7][i] = POSdict[i]["pronoun"]
		if(POSdict[i]["verb"]):
			features[8][i] = POSdict[i]["verb"]
				
	return features	