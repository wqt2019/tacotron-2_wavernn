
# CONFIG -----------------------------------------------------------------------------------------------------------#
train_data_base = 'D:/datasets/tacotron2/'
data_path = train_data_base + 'training/'

voc_model_id = 'biaobei_mol'
tts_model_id = 'biaobei_lsa_smooth_attention'

# DSP --------------------------------------------------------------------------------------------------------------#

# Settings for all models
sample_rate = 16000
n_fft = 1024
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 200                    # 12.5ms - in line with Tacotron 2 paper
win_length = 800                   # 50ms - same reason as above
fmin = 40   #40
min_level_db = -100
ref_level_db = 20
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode below
peak_norm = False                   # Normalise to the peak of each wav file
preemphasize = True
preemphasis = 0.97

# WAVERNN / VOCODER ------------------------------------------------------------------------------------------------#


# Model Hparams
voc_mode = 'RAW'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_upsample_factors = (5,5,8)  #(4, 5, 10)   # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 256  #256
voc_fc_dims = 256
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 64*2
voc_lr = 1e-4
voc_checkpoint_every = 20_000
voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
voc_total_steps = 1_000_000         # Total number of training steps
voc_test_samples = 500               # How many unseen samples to put aside for testing
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider than input length
voc_seq_len = hop_length * 5        # must be a multiple of hop_length
voc_clip_grad_norm = 4              # set to None if no gradient clipping needed

# Generating / Synthesizing
voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
voc_target = 4000                 # target number of samples to be generated in each batch entry
voc_overlap = 500                   # number of samples for crossfading between batches

# ------------------------------------------------------------------------------------------------------------------#

