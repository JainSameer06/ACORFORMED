#from pyAudioAnalysis import convertToWav
#from pyAudioAnalysis import audioSegmentation
import sys
from subprocess import call
import numpy as np
from pydub.audio_segment import AudioSegment
from pydub.silence import split_on_silence

pyAudioAnalysisPath = "/home/sameer/Downloads/pyAudioAnalysis-master"
sys.path.append(pyAudioAnalysisPath)
import convertToWav
import audioSegmentation
import audioAnalysis

def generateDiarization(videopath, filename, flsd, numSpeakers):
	convertToWav.main([videopath, "44100","2"])
	audioFile = videopath + '/' + filename + ".wav"
	#audioFile = videopath + '/' + 'combined.wav'
	print audioFile

	audioSeg = AudioSegment.from_wav(audioFile)
	speechSegments = split_on_silence(audio_segment = audioSeg, min_silence_len = 1000, silence_thresh = -23, keep_silence=100, seek_step=1)

	for i in range(1, len(speechSegments)):
		speechSegments[0] = speechSegments[0].append(speechSegments[i])

	f = speechSegments[0].export("SilenceRemoved.wav", format = 'wav')

	cls, mtStep = audioSegmentation.speakerDiarization("SilenceRemoved.wav", numOfSpeakers = numSpeakers, LDAdim = 0, PLOT = True)
	

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