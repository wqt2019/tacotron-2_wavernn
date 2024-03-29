import torch
from torch import nn
from math import sqrt
from hparams import hparams as hps
from torch.autograd import Variable
from torch.nn import functional as F
from model.layers import ConvNorm, LinearNorm
from utils.util import mode, get_mask_from_lengths
import numpy as np

class Tacotron2Loss(nn.Module):
	def __init__(self):
		super(Tacotron2Loss, self).__init__()

	def forward(self, model_output, targets, iteration):
		mel_target, gate_target = targets[0], targets[1]
		mel_target.requires_grad = False
		gate_target.requires_grad = False
		slice = torch.arange(0, gate_target.size(1), hps.n_frames_per_step)
		gate_target = gate_target[:, slice].view(-1, 1)

		mel_out, mel_out_postnet, gate_out, _ = model_output
		gate_out = gate_out.view(-1, 1)
		p = hps.p
		mel_loss = nn.MSELoss()(p*mel_out, p*mel_target) + \
			nn.MSELoss()(p*mel_out_postnet, p*mel_target)
		gate_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hps.gate_positive_weight,
														 device = gate_out.device))(gate_out, gate_target)
		return mel_loss+gate_loss, (mel_loss/(p**2)+gate_loss).item(),gate_loss


class LocationLayer(nn.Module):
	def __init__(self, attention_n_filters, attention_kernel_size,
				 attention_dim):
		super(LocationLayer, self).__init__()
		padding = int((attention_kernel_size - 1) / 2)
		self.location_conv = ConvNorm(2, attention_n_filters,
									  kernel_size=attention_kernel_size,
									  padding=padding, bias=False, stride=1,
									  dilation=1)
		self.location_dense = LinearNorm(attention_n_filters, attention_dim,
										 bias=False, w_init_gain='tanh')

	def forward(self, attention_weights_cat):
		processed_attention = self.location_conv(attention_weights_cat)
		processed_attention = processed_attention.transpose(1, 2)
		processed_attention = self.location_dense(processed_attention)
		return processed_attention


class StepwiseMonotonicAttention(nn.Module):
	"""
    StepwiseMonotonicAttention (SMA)

    This attention is described in:
        M. He, Y. Deng, and L. He, "Robust Sequence-to-Sequence Acoustic Modeling with Stepwise Monotonic Attention for Neural TTS,"
        in Annual Conference of the International Speech Communication Association (INTERSPEECH), 2019, pp. 1293-1297.
        https://arxiv.org/abs/1906.00672

    See:
        https://gist.github.com/mutiann/38a7638f75c21479582d7391490df37c
        https://github.com/keonlee9420/Stepwise_Monotonic_Multihead_Attention
    """

	def __init__(self, sigmoid_noise=2.0):
		"""
        Args:
            sigmoid_noise: Standard deviation of pre-sigmoid noise.
                           Setting this larger than 0 will encourage the model to produce
                           large attention scores, effectively making the choosing probabilities
                           discrete and the resulting attention distribution one-hot.
        """
		super(StepwiseMonotonicAttention, self).__init__()

		self.alignment = None  # alignment in previous query time step
		self.sigmoid_noise = sigmoid_noise

	def init_attention(self, processed_memory):
		# Initial alignment with [1, 0, ..., 0]
		b, t, c = processed_memory.size()
		self.alignment = processed_memory.new_zeros(b, t)
		self.alignment[:, 0:1] = 1

	def stepwise_monotonic_attention(self, p_i, prev_alignment):
		"""
        Compute stepwise monotonic attention
            - p_i: probability to keep attended to the last attended entry
            - Equation (8) in section 3 of the paper
        """
		pad = prev_alignment.new_zeros(prev_alignment.size(0), 1)
		alignment = prev_alignment * p_i + torch.cat((pad, prev_alignment[:, :-1] * (1.0 - p_i[:, :-1])), dim=1)
		return alignment

	def get_selection_probability(self, e, std):
		"""
        Compute selecton/sampling probability `p_i` from energies `e`
            - Equation (4) and the tricks in section 2.2 of the paper
        """
		# Add Gaussian noise to encourage discreteness
		if self.training:
			noise = e.new_zeros(e.size()).normal_()
			e = e + noise * std

		# Compute selecton/sampling probability p_i
		# (batch, max_time)
		return torch.sigmoid(e)

	def get_probabilities(self, energies):
		# Selecton/sampling probability p_i
		p_i = self.get_selection_probability(energies, self.sigmoid_noise)
		# Stepwise monotonic attention
		alignment = self.stepwise_monotonic_attention(p_i, self.alignment)
		# (batch, max_time)
		self.alignment = alignment
		return alignment


class Attention(nn.Module):
	def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
				 attention_location_n_filters, attention_location_kernel_size):
		super(Attention, self).__init__()
		self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
									  bias=False, w_init_gain='tanh')
		self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
									   w_init_gain='tanh')
		self.v = LinearNorm(attention_dim, 1, bias=False)
		self.location_layer = LocationLayer(attention_location_n_filters,
											attention_location_kernel_size,
											attention_dim)
		self.score_mask_value = -float('inf')
		self.sma = StepwiseMonotonicAttention()

	def get_alignment_energies(self, query, processed_memory,
							   attention_weights_cat):
		'''
		PARAMS
		------
		query: decoder output (batch, num_mels * n_frames_per_step)
		processed_memory: processed encoder outputs (B, T_in, attention_dim)
		attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

		RETURNS
		-------
		alignment (batch, max_time)
		'''

		processed_query = self.query_layer(query.unsqueeze(1))
		processed_attention_weights = self.location_layer(attention_weights_cat)
		energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))

		energies = energies.squeeze(-1)
		return energies

	def get_alignment_energies_basic(self, query, processed_memory):

		processed_query = self.query_layer(query.unsqueeze(1))
		energies = self.v(torch.tanh(processed_query + processed_memory))

		energies = energies.squeeze(-1)
		return energies

	def forward(self, attention_hidden_state, memory, processed_memory,
				attention_weights_cat, mask):
		'''
		PARAMS
		------
		attention_hidden_state: attention rnn last output
		memory: encoder outputs
		processed_memory: processed encoder outputs
		attention_weights_cat: previous and cummulative attention weights
		mask: binary mask for padded data
		'''

		######## lsa ###############################################################
		# alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)
		# attention_weights = F.softmax(alignment, dim=1)

		######## sma ###############################################################
		alignment = self.get_alignment_energies_basic(attention_hidden_state,processed_memory)
		alignment = self.sma.get_probabilities(alignment)
		attention_weights = alignment  # sma

		############################################################################
		if mask is not None:
			alignment.data.masked_fill_(mask, self.score_mask_value)
		attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
		attention_context = attention_context.squeeze(1)

		return attention_context, attention_weights


class Prenet(nn.Module):
	def __init__(self, in_dim, sizes):
		super(Prenet, self).__init__()
		in_sizes = [in_dim] + sizes[:-1]
		self.layers = nn.ModuleList(
			[LinearNorm(in_size, out_size, bias=False)
			 for (in_size, out_size) in zip(in_sizes, sizes)])

	def forward(self, x):
		for linear in self.layers:
			x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
		return x

class Postnet(nn.Module):
	'''Postnet
		- Five 1-d convolution with 512 channels and kernel size 5
	'''

	def __init__(self):
		super(Postnet, self).__init__()
		self.convolutions = nn.ModuleList()

		self.convolutions.append(
			nn.Sequential(
				ConvNorm(hps.num_mels, hps.postnet_embedding_dim,
						 kernel_size=hps.postnet_kernel_size, stride=1,
						 padding=int((hps.postnet_kernel_size - 1) / 2),
						 dilation=1, w_init_gain='tanh'),
				nn.BatchNorm1d(hps.postnet_embedding_dim))
		)

		for i in range(1, hps.postnet_n_convolutions - 1):
			self.convolutions.append(
				nn.Sequential(
					ConvNorm(hps.postnet_embedding_dim,
							 hps.postnet_embedding_dim,
							 kernel_size=hps.postnet_kernel_size, stride=1,
							 padding=int((hps.postnet_kernel_size - 1) / 2),
							 dilation=1, w_init_gain='tanh'),
					nn.BatchNorm1d(hps.postnet_embedding_dim))
			)

		self.convolutions.append(
			nn.Sequential(
				ConvNorm(hps.postnet_embedding_dim, hps.num_mels,
						 kernel_size=hps.postnet_kernel_size, stride=1,
						 padding=int((hps.postnet_kernel_size - 1) / 2),
						 dilation=1, w_init_gain='linear'),
				nn.BatchNorm1d(hps.num_mels))
			)

	def forward(self, x):
		for i in range(len(self.convolutions) - 1):
			x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
		x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

		return x


class Encoder(nn.Module):
	'''Encoder module:
		- Three 1-d convolution banks
		- Bidirectional LSTM
	'''
	def __init__(self):
		super(Encoder, self).__init__()

		convolutions = []
		for _ in range(hps.encoder_n_convolutions):
			conv_layer = nn.Sequential(
				ConvNorm(hps.encoder_embedding_dim,
						 hps.encoder_embedding_dim,
						 kernel_size=hps.encoder_kernel_size, stride=1,
						 padding=int((hps.encoder_kernel_size - 1) / 2),
						 dilation=1, w_init_gain='relu'),
				nn.BatchNorm1d(hps.encoder_embedding_dim))
			convolutions.append(conv_layer)
		self.convolutions = nn.ModuleList(convolutions)

		self.lstm = nn.LSTM(hps.encoder_embedding_dim,
							int(hps.encoder_embedding_dim / 2), 1,
							batch_first=True, bidirectional=True)

	def forward(self, x, input_lengths):
		for conv in self.convolutions:
			x = F.dropout(F.relu(conv(x)), 0.5, self.training)

		x = x.transpose(1, 2)

		# pytorch tensor are not reversible, hence the conversion
		input_lengths = input_lengths.cpu().numpy()
		x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

		self.lstm.flatten_parameters()
		outputs, _ = self.lstm(x)

		outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

		return outputs

	def inference(self, x):
		for conv in self.convolutions:
			x = F.dropout(F.relu(conv(x)), 0.5, self.training)

		x = x.transpose(1, 2)

		self.lstm.flatten_parameters()
		outputs, _ = self.lstm(x)

		return outputs


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.num_att_mixtures = 1 #hps.num_att_mixtures
		self.num_mels = hps.num_mels
		self.n_frames_per_step = hps.n_frames_per_step
		self.encoder_embedding_dim = hps.encoder_embedding_dim
		self.attention_rnn_dim = hps.attention_rnn_dim
		self.decoder_rnn_dim = hps.decoder_rnn_dim
		self.prenet_dim = hps.prenet_dim
		self.max_decoder_steps = hps.max_decoder_steps
		self.gate_threshold = hps.gate_threshold
		self.p_attention_dropout = hps.p_attention_dropout
		self.p_decoder_dropout = hps.p_decoder_dropout
		self.teacher_force_till = hps.teacher_force_till
		self.p_teacher_forcing = hps.p_teacher_forcing

		self.prenet = Prenet(
			hps.num_mels * hps.n_frames_per_step,
			[hps.prenet_dim, hps.prenet_dim])

		self.attention_rnn = nn.LSTMCell(
			hps.prenet_dim + hps.encoder_embedding_dim,
			hps.attention_rnn_dim)

		self.attention_layer = Attention(
			hps.attention_rnn_dim, hps.encoder_embedding_dim,
			hps.attention_dim, hps.attention_location_n_filters,
			hps.attention_location_kernel_size)

		self.decoder_rnn = nn.LSTMCell(
			hps.attention_rnn_dim + hps.encoder_embedding_dim,
			hps.decoder_rnn_dim, 1)

		self.linear_projection = LinearNorm(
			hps.decoder_rnn_dim + hps.encoder_embedding_dim,
			hps.num_mels * hps.n_frames_per_step)

		self.gate_layer = LinearNorm(
			hps.decoder_rnn_dim + hps.encoder_embedding_dim, 1,
			bias=True, w_init_gain='sigmoid')

	def get_go_frame(self, memory):
		''' Gets all zeros frames to use as first decoder input
		PARAMS
		------
		memory: decoder outputs

		RETURNS
		-------
		decoder_input: all zeros frames
		'''
		B = memory.size(0)
		decoder_input = Variable(memory.data.new(B, self.num_mels * self.n_frames_per_step).zero_())
		return decoder_input

	def initialize_decoder_states(self, memory, mask):
		''' Initializes attention rnn states, decoder rnn states, attention
		weights, attention cumulative weights, attention context, stores memory
		and stores processed memory
		PARAMS
		------
		memory: Encoder outputs
		mask: Mask for padded data if training, expects None for inference
		'''
		B = memory.size(0)
		MAX_TIME = memory.size(1)

		self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
		self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())

		self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
		self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())

		self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
		self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
		self.attention_context = Variable(memory.data.new(B, self.encoder_embedding_dim).zero_())

		self.memory = memory
		self.processed_memory = self.attention_layer.memory_layer(memory)
		self.mask = mask


	def parse_decoder_inputs(self, decoder_inputs):
		''' Prepares decoder inputs, i.e. mel outputs
		PARAMS
		------
		decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

		RETURNS
		-------
		inputs: processed decoder inputs

		'''
		# (B, num_mels, T_out) -> (B, T_out, num_mels)
		decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
		decoder_inputs = decoder_inputs.view(decoder_inputs.size(0),
											int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
		# (B, T_out, num_mels) -> (T_out, B, num_mels)
		decoder_inputs = decoder_inputs.transpose(0, 1)
		return decoder_inputs

	def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
		''' Prepares decoder outputs for output
		PARAMS
		------
		mel_outputs:
		gate_outputs: gate output energies
		alignments:

		RETURNS
		-------
		mel_outputs:
		gate_outpust: gate output energies
		alignments:
		'''
		# (T_out, B) -> (B, T_out)
		alignments = torch.stack(alignments).transpose(0, 1)
		# (T_out, B) -> (B, T_out)
		gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
		gate_outputs = gate_outputs.contiguous()
		# (T_out, B, num_mels) -> (B, T_out, num_mels)
		mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
		# decouple frames per step
		mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.num_mels)
		# (B, T_out, num_mels) -> (B, num_mels, T_out)
		mel_outputs = mel_outputs.transpose(1, 2)

		return mel_outputs, gate_outputs, alignments

	def decode(self, decoder_input):
		''' Decoder step using stored states, attention and memory
		PARAMS
		------
		decoder_input: previous mel output

		RETURNS
		-------
		mel_output:
		gate_output: gate output energies
		attention_weights:
		'''
		cell_input = torch.cat((decoder_input, self.attention_context), -1)
		self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))
		self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)

		attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1),self.attention_weights_cum.unsqueeze(1)), dim=1)
		self.attention_context, self.attention_weights = self.attention_layer(self.attention_hidden, self.memory, self.processed_memory,
																				attention_weights_cat, self.mask)

		self.attention_weights_cum += self.attention_weights
		decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
		self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))
		self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)

		decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
		decoder_output = self.linear_projection(decoder_hidden_attention_context)

		gate_prediction = self.gate_layer(decoder_hidden_attention_context)
		return decoder_output, gate_prediction, self.attention_weights

	def forward(self, memory, decoder_inputs, memory_lengths):
		''' Decoder forward pass for training
		PARAMS
		------
		memory: Encoder outputs
		decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
		memory_lengths: Encoder output lengths for attention masking.

		RETURNS
		-------
		mel_outputs: mel outputs from the decoder
		gate_outputs: gate outputs from the decoder
		alignments: sequence of attention weights from the decoder
		'''
		decoder_input = self.get_go_frame(memory).unsqueeze(0)
		decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
		decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
		decoder_inputs = self.prenet(decoder_inputs)

		self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths))

		self.attention_layer.sma.init_attention(self.processed_memory)

		mel_outputs, gate_outputs, alignments = [], [], []
		while len(mel_outputs) < decoder_inputs.size(0) - 1:
			# decoder_input = decoder_inputs[len(mel_outputs)]
			if len(mel_outputs) <= self.teacher_force_till or np.random.uniform(0.0, 1.0) <= self.p_teacher_forcing:
				decoder_input = decoder_inputs[len(mel_outputs)]  # use all-in-one processed output for next step
			else:
				decoder_input = self.prenet(mel_outputs[-1])  # use last output for next step (like inference)

			mel_output, gate_output, attention_weights = self.decode(decoder_input)
			mel_outputs += [mel_output.squeeze(1)]
			gate_outputs += [gate_output.squeeze()]
			alignments += [attention_weights]
		mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

		return mel_outputs, gate_outputs, alignments

	def inference(self, memory):
		''' Decoder inference
		PARAMS
		------
		memory: Encoder outputs

		RETURNS
		-------
		mel_outputs: mel outputs from the decoder
		gate_outputs: gate outputs from the decoder
		alignments: sequence of attention weights from the decoder
		'''
		decoder_input = self.get_go_frame(memory)

		self.initialize_decoder_states(memory, mask=None)

		self.attention_layer.sma.init_attention(self.memory)

		mel_outputs, gate_outputs, alignments = [], [], []

		while True:
			decoder_input = self.prenet(decoder_input)
			mel_output, gate_output, alignment = self.decode(decoder_input)

			mel_outputs += [mel_output.squeeze(1)]
			gate_outputs += [gate_output]
			alignments += [alignment]

			if torch.sigmoid(gate_output.data) > self.gate_threshold and len(mel_outputs)>10:
				print('Terminated by gate.:',torch.sigmoid(gate_output.data)[0][0])
				break
			# elif len(mel_outputs) > 1 and is_end_of_frames(mel_output):
			# 	print('torch.sigmoid(gate_output.data):', torch.sigmoid(gate_output.data))
			# 	print('Warning: End with low power.')
			# 	break
			elif len(mel_outputs) == self.max_decoder_steps:
				print('Warning: Reached max decoder steps.',self.max_decoder_steps)
				break

			decoder_input = mel_output

		mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
		return mel_outputs, gate_outputs, alignments

def is_end_of_frames(output, eps = 0.2):
	return (output.data <= eps).all()

class Tacotron2(nn.Module):
	def __init__(self):
		super(Tacotron2, self).__init__()
		self.num_mels = hps.num_mels
		self.mask_padding = hps.mask_padding
		self.n_frames_per_step = hps.n_frames_per_step
		self.embedding = nn.Embedding(
			hps.n_symbols, hps.symbols_embedding_dim)
		std = sqrt(2.0/(hps.n_symbols+hps.symbols_embedding_dim))
		val = sqrt(3.0)*std  # uniform bounds for std
		self.embedding.weight.data.uniform_(-val, val)
		self.encoder = Encoder()
		self.decoder = Decoder()
		self.postnet = Postnet()

	def parse_batch(self, batch):
		text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
		text_padded = mode(text_padded).long()
		input_lengths = mode(input_lengths).long()
		max_len = torch.max(input_lengths.data).item()
		mel_padded = mode(mel_padded).float()
		gate_padded = mode(gate_padded).float()
		output_lengths = mode(output_lengths).long()

		return ((text_padded, input_lengths, mel_padded, max_len, output_lengths),(mel_padded, gate_padded))

	def parse_output(self, outputs, output_lengths=None):
		if self.mask_padding and output_lengths is not None:
			mask = ~get_mask_from_lengths(output_lengths, True) # (B, T)
			mask = mask.expand(self.num_mels, mask.size(0), mask.size(1)) # (80, B, T)
			mask = mask.permute(1, 0, 2) # (B, 80, T)
			
			outputs[0].data.masked_fill_(mask, 0.0) # (B, 80, T)
			outputs[1].data.masked_fill_(mask, 0.0) # (B, 80, T)
			slice = torch.arange(0, mask.size(2), self.n_frames_per_step)
			outputs[2].data.masked_fill_(mask[:, 0, slice], 1e3)  # gate energies (B, T//n_frames_per_step)

		return outputs

	def forward(self, inputs):
		text_inputs, text_lengths, mels, max_len, output_lengths = inputs
		text_lengths, output_lengths = text_lengths.data, output_lengths.data

		embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

		encoder_outputs = self.encoder(embedded_inputs, text_lengths)

		mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths)

		mel_outputs_postnet = self.postnet(mel_outputs)
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet

		return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments],output_lengths)

	def inference(self, inputs):
		embedded_inputs = self.embedding(inputs).transpose(1, 2)
		encoder_outputs = self.encoder.inference(embedded_inputs)
		mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

		mel_outputs_postnet = self.postnet(mel_outputs)
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet

		outputs = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

		return outputs

	def teacher_infer(self, inputs, mels):
		il, _ =  torch.sort(torch.LongTensor([len(x) for x in inputs]),dim = 0, descending = True)
		text_lengths = mode(il)

		embedded_inputs = self.embedding(inputs).transpose(1, 2)

		encoder_outputs = self.encoder(embedded_inputs, text_lengths)

		mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths)
		
		mel_outputs_postnet = self.postnet(mel_outputs)
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet

		return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
