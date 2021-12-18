import os
import time
import torch
import argparse
import numpy as np
from inference import infer
from utils.util import mode
from hparams import hparams as hps
from torch.utils.data import DataLoader
from utils.logger import Tacotron2Logger
from utils.dataset import ljdataset, ljcollate
from model.model import Tacotron2, Tacotron2Loss
from numpy import finfo
import math,random
from inference import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
np.random.seed(hps.seed)
torch.manual_seed(hps.seed)
torch.cuda.manual_seed(hps.seed)

def prepare_dataloaders(fdir):
	trainset = ljdataset(fdir)
	collate_fn = ljcollate(hps.n_frames_per_step)
	train_loader = DataLoader(trainset, num_workers = hps.n_workers, shuffle = True,
							  batch_size = hps.batch_size, pin_memory = hps.pin_mem,
							  drop_last = True, collate_fn = collate_fn)
	return train_loader


def load_checkpoint(ckpt_pth, model, optimizer):
	ckpt_dict = torch.load(ckpt_pth)
	model.load_state_dict(ckpt_dict['model'])
	optimizer.load_state_dict(ckpt_dict['optimizer'])
	iteration = ckpt_dict['iteration']
	return model, optimizer, iteration


def save_checkpoint(model, optimizer, iteration, ckpt_pth):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iteration': iteration}, ckpt_pth)

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
	np.save(pth+'.npy', to_arr(mel_outputs_postnet[0]).T)

def warm_start_model(checkpoint_path, model, ignore_layers,optimeizer):
	assert os.path.isfile(checkpoint_path)
	print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
	checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
	model_dict = checkpoint_dict['model']
	if len(ignore_layers) > 0:
		model_dict = {k: v for k, v in model_dict.items()
					  if k not in ignore_layers}
		dummy_dict = model.state_dict()
		dummy_dict.update(model_dict)
		model_dict = dummy_dict
	model.load_state_dict(model_dict)
	optimeizer.load_state_dict(checkpoint_dict['optimizer'])
	return model


def train(args):
	# build model
	model = Tacotron2()
	mode(model, True)
	if hps.fp16_run:
		model.decoder.attention_layer.score_mask_value = finfo('float16').min

	optimizer = torch.optim.Adam(model.parameters(), lr = hps.lr,
								betas = hps.betas, eps = hps.eps,
								weight_decay = hps.weight_decay)

	if hps.fp16_run:
		from apex import amp
		model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

	criterion = Tacotron2Loss()
	
	# load checkpoint
	iteration = 1
	if args.ckpt_pth != '':
		if hps.warm_start:
			model = warm_start_model(args.ckpt_pth, model, hps.ignore_layers,optimizer)
			iteration += 1
		else:
			model, optimizer, iteration = load_checkpoint(args.ckpt_pth, model, optimizer)
			print('load from :', args.ckpt_pth)
			iteration += 1  # next iteration is iteration+1
			# iteration = 1
	
	# get scheduler
	if hps.sch:
		lr_lambda = lambda step: hps.sch_step**0.5*min((step+1)*hps.sch_step**-1.5, (step+1)**-0.5)
		if args.ckpt_pth != '':
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch = iteration)
		else:
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
	
	# make dataset
	train_loader = prepare_dataloaders(args.data_dir)
	
	# get logger ready
	if args.log_dir != '':
		if not os.path.isdir(args.log_dir):
			os.makedirs(args.log_dir)
			os.chmod(args.log_dir, 0o775)
		logger = Tacotron2Logger(args.log_dir)

	# get ckpt_dir ready
	if args.ckpt_dir != '' and not os.path.isdir(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
		os.chmod(args.ckpt_dir, 0o775)
	
	model.train()
	# ================ MAIN TRAINNIG LOOP! ===================
	while iteration <= hps.max_iter:
		for batch in train_loader:
			if iteration > hps.max_iter:
				break
			start = time.perf_counter()
			x, y = model.parse_batch(batch)
			y_pred = model(x)

			# loss
			loss, item = criterion(y_pred, y, iteration)

			# zero grad
			model.zero_grad()

			if hps.fp16_run:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			if hps.fp16_run:
				grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), hps.grad_clip_thresh)
				is_overflow = math.isnan(grad_norm)
			else:
				grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.grad_clip_thresh)
			
			# backward, grad_norm, and update
			# loss.backward()
			# grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.grad_clip_thresh)

			optimizer.step()
			if hps.sch:
				scheduler.step()
			
			# info
			dur = time.perf_counter()-start
			if(iteration % 10 == 0):
				print('Iter: {} Loss: {:.6f} Grad Norm: {:.6f} {:.2f}s/it'.format(
					iteration, item, grad_norm, dur))
			
			# log
			if args.log_dir != '' and (iteration % hps.iters_per_log == 0):
				learning_rate = optimizer.param_groups[0]['lr']
				logger.log_training(item, grad_norm, learning_rate, iteration)
			
			# sample
			# iteration = 0
			if args.log_dir != '' and (iteration % hps.iters_per_sample == 0):
				model.eval()
				i = random.randint(0, len(hps.eg_text) - 1)
				text = hps.eg_text[i]
				print('text:', text)
				output = infer(text, model)
				model.train()
				logger.sample_training(output, iteration)

				plot(output, os.path.join(args.ckpt_dir,str(iteration)))
				audio(output, os.path.join(args.ckpt_dir,str(iteration)))
				save_mel(output, os.path.join(args.ckpt_dir,str(iteration)))
			
			# save ckpt
			if args.ckpt_dir != '' and (iteration % hps.iters_per_ckpt == 0):
				ckpt_pth = os.path.join(args.ckpt_dir, 'biaobei_{}.pt'.format(iteration))
				print('hps.n_frames_per_step:',hps.n_frames_per_step)
				print('ckpt_pth:',ckpt_pth)
				save_checkpoint(model, optimizer, iteration, ckpt_pth)

			iteration += 1
	if args.log_dir != '':
		logger.close()
	print('train done')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	data_dir = './tacotron2/biaobei/training/'
	print('data_dir:',data_dir)
	parser.add_argument('--data_dir', type = str, default = data_dir,help = 'directory to load data')
	parser.add_argument('--log_dir', type = str, default = './biaobei3/log',help = 'directory to save tensorboard logs')
	parser.add_argument('--ckpt_dir', type = str, default = './biaobei3/ckpt',help = 'directory to save checkpoints')
	parser.add_argument('--ckpt_pth', type = str, default = '',help = 'path to load checkpoints')

	args = parser.parse_args()
	print("FP16 Run:", hps.fp16_run)
	print("hps.n_symbols:", hps.n_symbols)
	print('hps.n_frames_per_setp:',hps.n_frames_per_step)
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = False # faster due to dynamic input shape
	train(args)
