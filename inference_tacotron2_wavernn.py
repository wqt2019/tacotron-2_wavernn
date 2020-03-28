import argparse
import os
import time
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer import Synthesizer

# #
import torch
from wavernn_vocoder.models.fatchord_version import WaveRNN
from wavernn_vocoder.utils import hparams as wavernn_hp
from wavernn_vocoder.utils.paths import Paths
from wavernn_vocoder.utils.display import simple_table
from wavernn_vocoder.utils.dsp import reconstruct_waveform, save_wav
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_sentences(args):
	if args.text_list != '':
		with open(args.text_list, 'rb') as f:
			sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
	else:
		sentences = hparams.sentences
	return sentences

def init_tacotron2(args):
	# t2
	print('\n#####################################')
	if args.model == 'Tacotron':
		print('\nInitialising Tacotron Model...\n')
		t2_hparams = hparams.parse(args.hparams)
		try:
			checkpoint_path = tf.train.get_checkpoint_state(args.taco_checkpoint).model_checkpoint_path
			log('loaded model at {}'.format(checkpoint_path))
		except:
			raise RuntimeError('Failed to load checkpoint at {}'.format(args.taco_checkpoint))

		output_dir = 'tacotron_' + args.output_dir
		eval_dir = os.path.join(output_dir, 'eval')
		log_dir = os.path.join(output_dir, 'logs-eval')
		print('eval_dir:', eval_dir)
		print('args.mels_dir:', args.mels_dir)

		# Create output path if it doesn't exist
		os.makedirs(eval_dir, exist_ok=True)
		os.makedirs(log_dir, exist_ok=True)
		os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
		os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
		log(hparams_debug_string())
		synth = Synthesizer()
		synth.load(checkpoint_path, t2_hparams)

	return synth,eval_dir,log_dir

def init_wavernn(args):
	##########################################
	# wavernn
	print('\n#####################################')
	if args.vocoder == 'wavernn' or args.vocoder == 'wr':
		wavernn_hp.configure(args.hp_file)
		paths = Paths(wavernn_hp.data_path, wavernn_hp.voc_model_id, wavernn_hp.tts_model_id)
		if not args.force_cpu and torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		print('Using device:', device)

		print('\nInitialising WaveRNN Model...\n')
		# Instantiate WaveRNN Model
		voc_model = WaveRNN(rnn_dims=wavernn_hp.voc_rnn_dims,
							fc_dims=wavernn_hp.voc_fc_dims,
							bits=wavernn_hp.bits,
							pad=wavernn_hp.voc_pad,
							upsample_factors=wavernn_hp.voc_upsample_factors,
							feat_dims=wavernn_hp.num_mels,
							compute_dims=wavernn_hp.voc_compute_dims,
							res_out_dims=wavernn_hp.voc_res_out_dims,
							res_blocks=wavernn_hp.voc_res_blocks,
							hop_length=wavernn_hp.hop_length,
							sample_rate=wavernn_hp.sample_rate,
							mode=wavernn_hp.voc_mode).to(device)

		voc_load_path = args.voc_weights if args.voc_weights else paths.voc_latest_weights
		voc_model.load(voc_load_path)

		voc_k = voc_model.get_step() // 1000
		simple_table([
			('Vocoder Type', 'WaveRNN'),
			('WaveRNN', str(voc_k) + 'k'),
			('Generation Mode', 'Batched' if wavernn_hp.voc_gen_batched else 'Unbatched'),
			('Target Samples', wavernn_hp.voc_target if wavernn_hp.voc_gen_batched else 'N/A'),
			('Overlap Samples', wavernn_hp.voc_overlap if wavernn_hp.voc_gen_batched else 'N/A')])

		if args.vocoder == 'griffinlim' or args.vocoder == 'gl':
			v_type = args.vocoder
		elif (args.vocoder == 'wavernn' or args.vocoder == 'wr') and wavernn_hp.voc_gen_batched:
			v_type = 'wavernn_batched'
		else:
			v_type = 'wavernn_unbatched'
	else:
		return None,None,None

	return voc_model,voc_k,v_type


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--taco_checkpoint',
			default='./logs-Tacotron-2/taco_pretrained/',
			help='Path to model checkpoint')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--vocoder', default='wr')
	parser.add_argument('--mels_dir', default='tacotron_output/eval/', help='folder to contain mels to synthesize audio from using the Wavenet')
	parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--text_list', default='sentences.txt', help='Text file contains list of texts to be synthesized. Valid if mode=eval')
	parser.add_argument('--speaker_id', default=None, help='Defines the speakers ids to use when running standalone Wavenet on a folder of mels. this variable must be a comma-separated list of ids')

	# wavernn
	parser.add_argument('--force_cpu', '-c', action='store_true',
						help='Forces CPU-only training, even when in CUDA capable environment')
	parser.add_argument('--hp_file', metavar='FILE', default='wavernn_vocoder/hparams.py',
						help='The file to use for the hyperparameters')
	parser.add_argument('--voc_weights', type=str, help='[string/path] Load in different WaveRNN weights')
	parser.add_argument('--iters', type=int, default=32, help='[int] number of griffinlim iterations')
	args = parser.parse_args()
	sentences = get_sentences(args)

	############################
	synth,eval_dir,log_dir= init_tacotron2(args)
	if args.vocoder == 'wavernn' or args.vocoder == 'wr':
		voc_model,voc_k,v_type = init_wavernn(args)
	output_wr_dir = 'tacotron_' + args.output_dir + 'wavernn/'
	os.makedirs(output_wr_dir, exist_ok=True)

	# ###################################
	# Set inputs batch wise
	sentences = [sentences[i: i + hparams.tacotron_synthesis_batch_size] for i in
				 range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

	log('Starting Synthesis')
	with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
		for i, texts in enumerate(tqdm(sentences)):
			start = time.time()
			print('\nsynthesis mel:' + str(i))
			basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
			mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)
			for elems in zip(texts, mel_filenames, speaker_ids):
				file.write('|'.join([str(x) for x in elems]) + '\n')
			print('\nsynthesis mel done')

			# wavernn
			print('\nstart wavernn')

			mel_filenames = mel_filenames[0]
			# print('\n'+ mel_filenames)
			m = np.load(mel_filenames)
			m = (m + 4) / 8
			m = np.transpose(m, (1, 0))

			if args.vocoder == 'wavernn' or args.vocoder == 'wr':
				save_wr_file = output_wr_dir + str(i) + '_' + str(v_type) + str(voc_k) + 'k.wav'
				m = torch.tensor(m).unsqueeze(0)
				voc_model.generate(m, save_wr_file, wavernn_hp.voc_gen_batched, wavernn_hp.voc_target, wavernn_hp.voc_overlap, wavernn_hp.mu_law)
			# elif args.vocoder == 'griffinlim' or args.vocoder == 'gl':
			# 	save_wr_file = output_wr_dir + str(i) + '_gl.wav'
			# 	wav = reconstruct_waveform(m, n_iter=args.iters)
			# 	save_wav(wav, save_wr_file)
			print('\nwavernn done')
			print('#####################\n')

	log('\nsynthesized done at {}'.format(output_wr_dir))


if __name__ == '__main__':
	main()
