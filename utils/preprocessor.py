import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch
import numpy as np
from utils import audio

silence_audio_size = 200 * 8

def build_from_path_mydata(hparams, input_dir, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):

	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	# data_set = 'LJSpeech'
	data_set = 'biaobei'
	# biaobei
	if (data_set == 'biaobei'):
		with open(os.path.join(input_dir, 'biaobei_phone.txt'), encoding='utf-8') as f:
			for line in f:
				parts = line.strip().split(',')
				basename = parts[0]
				wav_path = os.path.join(input_dir, 'Wave', '{}.wav'.format(basename))
				text = parts[-1]
				futures.append(executor.submit(
					partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
				index += 1

	if (data_set == 'LJSpeech'):
		with open(os.path.join(input_dir, 'metadata.csv'),encoding='utf-8') as f:
			for line in f:
				parts = line.strip().split('|')
				basename = parts[0]
				wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(basename))
				text = parts[-1]
				futures.append(executor.submit(
					partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text,hparams)))
				index += 1


	return [future.result() for future in tqdm(futures) if future.result() is not None]




def build_from_path(hparams, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	for input_dir in input_dirs:
		with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
			for line in f:
				parts = line.strip().split('|')
				basename = parts[0]
				wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(basename))
				text = parts[2]
				futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
				index += 1

	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path)
		wav = np.clip(wav,-1,1)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
		return None

	wav = audio.trim_silence(wav)

	if (hparams.preemphasize):
		wav = audio.preemphasis(wav)

	wav = np.append([0.] * silence_audio_size, wav)
	wav = np.append(wav, [0.] * silence_audio_size)

	#[-1, 1]
	# out = wav
	# out_dtype = np.float32

	# Compute the mel scale spectrogram from the wav
	mel_spectrogram = audio.melspectrogram(wav)
	mel_frames = mel_spectrogram.shape[1]

	if mel_frames > hparams.max_mel_frames:
		return None

	# #Compute the linear scale spectrogram from the wav
	linear_spectrogram = audio.spectrogram(wav)
	linear_frames = linear_spectrogram.shape[1]

	#sanity check
	assert linear_frames == mel_frames

	out = wav
	out_dtype = np.float32
	time_steps = len(out)

	# Write the spectrogram and audio to disk
	audio_filename = 'audio-{}.npy'.format(index)
	mel_filename = 'mel-{}.npy'.format(index)
	linear_filename = 'linear-{}.npy'.format(index)
	np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram, allow_pickle=False)
	np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram, allow_pickle=False)

	# Return a tuple describing this training example
	return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)


