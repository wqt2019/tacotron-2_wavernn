import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
from text import text_to_sequence
from model.model import Tacotron2
from hparams import hparams as hps
from utils.util import mode, to_arr
from utils.audio import save_wav, inv_melspectrogram
import time,os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def load_model(ckpt_pth):
	ckpt_dict = torch.load(ckpt_pth,map_location=torch.device('cpu'))
	model = Tacotron2()
	model.load_state_dict(ckpt_dict['model'])
	model = mode(model, True).eval()
	return model


def infer(text, model):
	sequence = text_to_sequence(text, hps.text_cleaners)
	print('sequence:',len(sequence))
	sequence = mode(torch.IntTensor(sequence)[None, :]).long()
	for i in range(1):
		t1 = time.time()
		mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
		print('mel_outputs_postnet:', mel_outputs_postnet.shape)
		print('alignments:', alignments.shape)
		print('sess.run time:', (time.time() - t1))
	return (mel_outputs, mel_outputs_postnet, alignments)


def plot_data(data, figsize = (16, 4)):
	fig, axes = plt.subplots(1, len(data), figsize = figsize)
	for i in range(len(data)):
		axes[i].imshow(data[i], aspect = 'auto', origin = 'bottom')


def plot(output, pth):
	mel_outputs, mel_outputs_postnet, alignments = output
	plot_data((to_arr(mel_outputs[0]),
				to_arr(mel_outputs_postnet[0]),
				to_arr(alignments[0]).T))
	plt.savefig(pth+'.png')


def audio(output, pth):
	mel_outputs, mel_outputs_postnet, _ = output
	wav_postnet = inv_melspectrogram(to_arr(mel_outputs_postnet[0]))
	save_wav(wav_postnet, pth+'.wav')


def save_mel(output, pth):
	mel_outputs, mel_outputs_postnet, _ = output
	np.save(pth+'.npy', to_arr(mel_outputs[0]).T)


if __name__ == '__main__':

	hps.is_cuda = False

	ckpt_pth = './saved_model/biaobei3/biaobei_ckpt_200000.pt'
	name = 'biaobei3_200000_'

	texts = ['k a2 er2 p u3 #2 p ei2 w uai4 s uen1 #1 w uan2 h ua2 t i1 #4 。',
			'j ia2 y v3 c uen1 y ian2 #2 b ie2 z ai4 #1 y iong1 b ao4 w uo3 #4 。',
			'b ao2 m a3 #1 p ei4 g ua4 #1 b o3 l uo2 an1 #3 ， d iao1 ch an2 #1 y van4 zh en3 #2 d ong3 w uen1 t a4 #4 。',
			]

	model = load_model(ckpt_pth)
	for i in range(len(texts)):
		print('texts[i]:', texts[i])
		output = infer(texts[i], model)
		save_name = name + str(i)
		img_pth = './result/img/' + save_name
		wav_pth = './result/wav/' + save_name
		npy_pth = './result/npy/' + save_name

		if img_pth != '':
			plot(output, img_pth)
		if wav_pth != '':
			audio(output, wav_pth)
		if npy_pth != '':
			save_mel(output, npy_pth)
		print('-------------------------')

