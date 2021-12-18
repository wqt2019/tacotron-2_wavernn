from text import symbols


class hparams:
	seed = 0

	################################
	# Data Parameters              #
	################################
	text_cleaners = ['basic_cleaners']

	################################
	# Audio                        #
	################################
	num_mels = 80
	num_freq = 513
	n_fft = 1024
	sample_rate = 16000
	frame_shift = 200
	hop_size = frame_shift
	frame_length = 800
	win_size = frame_length
	preemphasize = True
	preemphasis = 0.97
	min_level_db = -100
	ref_level_db = 20
	fmin = 0
	fmax = 8000
	power = 1.5
	gl_iters = 100

	max_mel_frames = 1200
	trim_silence = True,  # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	trim_fft_size = 1024  # Trimming window size
	trim_hop_size = 200  # Trimmin hop length
	trim_top_db = 40  # Trimming db difference from reference db (smaller==harder trim.)
	magnitude_power = 2.
	max_abs_value = 1.
	fp16_run = True
	warm_start = True
	ignore_layers = ['embedding.weight']

	################################
	# Train                        #
	################################
	is_cuda = True
	pin_mem = True
	n_workers = 8
	prep = True
	pth = 'lj-22k.pth'
	lr = 1e-3
	betas = (0.9, 0.999)
	eps = 1e-5
	sch = True
	sch_step = 4000
	max_iter = 200e3
	batch_size = 64
	iters_per_log = 10
	iters_per_sample = 2000
	iters_per_ckpt = 2000
	weight_decay = 1e-6
	grad_clip_thresh = 1.0
	mask_padding = True
	p = 10 # mel spec loss penalty
	gate_positive_weight = 3 #default 1

	################################
	# Model Parameters             #
	################################
	n_symbols = len(symbols)
	symbols_embedding_dim = 512

	# Encoder parameters
	encoder_kernel_size = 5
	encoder_n_convolutions = 3
	encoder_embedding_dim = 512

	# Decoder parameters
	n_frames_per_step = 3 #3
	decoder_rnn_dim = 1024 #1024
	prenet_dim = 256
	max_decoder_steps = 1000
	gate_threshold = 0.5
	p_attention_dropout = 0.1
	p_decoder_dropout = 0.1

	# Attention parameters
	attention_rnn_dim = 1024 #1024
	attention_dim = 256

	# Location Layer parameters
	attention_location_n_filters = 32
	attention_location_kernel_size = 31

	# Mel-post processing network parameters
	postnet_embedding_dim = 512
	postnet_kernel_size = 5
	postnet_n_convolutions = 5

	eg_text = [
		'k a2 er2 p u3 #2 p ei2 w uai4 s uen1 #1 w uan2 h ua2 t i1 #4 。',
		'j ia2 y v3 c uen1 y ian2 #2 b ie2 z ai4 #1 y iong1 b ao4 w uo3 #4 。',
		'b ao2 m a3 #1 p ei4 g ua4 #1 b o3 l uo2 an1 #3 ， d iao1 ch an2 #1 y van4 zh en3 #2 d ong3 w uen1 t a4 #4 。',
		]

