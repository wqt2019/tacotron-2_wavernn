"""Convert trained model for libwavernn

usage: convert_model.py [options] <checkpoint.pth>

options:
    --output-dir=<dir>           Output Directory [default: model_outputs]
    -h, --help                   Show this help message and exit
"""
#    --mel=<file>                 Mel file input for testing.


import torch
import struct
import numpy as np
import scipy as sp

from models.fatchord_version import WaveRNN
from utils import hparams as wavernn_hp
from utils.display import num_params_count


def compress(W):
    N = W.shape[1]
    W_nz = W.copy()
    W_nz[W_nz!=0]=1
    L = W_nz.reshape([-1, N // wavernn_hp.sparse_group, wavernn_hp.sparse_group])
    S = L.max(axis=-1)
    #convert to compressed index
    #compressed representation has position in each row. "255" denotes row end.
    (row,col)=np.nonzero(S)
    idx=[]
    for i in range(S.shape[0]+1):
        idx += list(col[row==i])
        idx += [255]
    mask = np.repeat(S, wavernn_hp.sparse_group, axis=1)
    idx = np.asarray(idx, dtype='uint8')
    return (W[mask!=0], idx)

def writeCompressed(f, W):
    weights, idx = compress(W)
    f.write(struct.pack('@i',weights.size))
    f.write(weights.tobytes(order='C'))
    f.write(struct.pack('@i',idx.size))
    f.write(idx.tobytes(order='C'))
    return


def linear_saver(f, layer):
    weight = layer.weight.cpu().detach().numpy()

    bias = layer.bias.cpu().detach().numpy()
    nrows, ncols = weight.shape
    v = struct.pack('@iii', elSize, nrows, ncols)
    f.write(v)
    # writeCompressed(f, weight)
    f.write(weight.tobytes(order='C'))
    f.write(bias.tobytes(order='C'))

def conv1d_saver(f, layer):
    weight = layer.weight.cpu().detach().numpy()
    out_channels, in_channels, nkernel = weight.shape
    v = struct.pack('@iiiii', elSize, not(layer.bias is None), in_channels, out_channels, nkernel)
    f.write(v)
    f.write(weight.tobytes(order='C'))
    if not (layer.bias is None ):
        bias = layer.bias.cpu().detach().numpy()
        f.write(bias.tobytes(order='C'))
    return

def conv2d_saver(f, layer):
    weight = layer.weight.cpu().detach().numpy()
    assert(weight.shape[0]==weight.shape[1]==weight.shape[2]==1) #handles only specific type used in WaveRNN
    weight = weight.squeeze()
    nkernel = weight.shape[0]

    v = struct.pack('@ii', elSize, nkernel)
    f.write(v)
    f.write(weight.tobytes(order='C'))
    return

def batchnorm1d_saver(f, layer):

    v = struct.pack('@iif', elSize, layer.num_features, layer.eps)
    f.write(v)
    weight=layer.weight.detach().numpy()
    bias=layer.bias.detach().numpy()
    running_mean = layer.running_mean.detach().numpy()
    running_var = layer.running_var.detach().numpy()

    f.write(weight.tobytes(order='C'))
    f.write(bias.tobytes(order='C'))
    f.write(running_mean.tobytes(order='C'))
    f.write(running_var.tobytes(order='C'))

    return

def gru_saver(f, layer):
    weight_ih_l0 = layer.weight_ih_l0.detach().cpu().numpy()
    weight_hh_l0 = layer.weight_hh_l0.detach().cpu().numpy()
    bias_ih_l0 = layer.bias_ih_l0.detach().cpu().numpy()
    bias_hh_l0 = layer.bias_hh_l0.detach().cpu().numpy()

    W_ir,W_iz,W_in=np.vsplit(weight_ih_l0, 3)
    W_hr,W_hz,W_hn=np.vsplit(weight_hh_l0, 3)

    b_ir,b_iz,b_in=np.split(bias_ih_l0, 3)
    b_hr,b_hz,b_hn=np.split(bias_hh_l0, 3)

    hidden_size, input_size = W_ir.shape
    v = struct.pack('@iii', elSize, hidden_size, input_size)
    f.write(v)
    # writeCompressed(f, W_ir)
    # writeCompressed(f, W_iz)
    # writeCompressed(f, W_in)
    # writeCompressed(f, W_hr)
    # writeCompressed(f, W_hz)
    # writeCompressed(f, W_hn)

    f.write(W_ir.tobytes(order='C'))
    f.write(W_iz.tobytes(order='C'))
    f.write(W_in.tobytes(order='C'))
    f.write(W_hr.tobytes(order='C'))
    f.write(W_hz.tobytes(order='C'))
    f.write(W_hn.tobytes(order='C'))

    f.write(b_ir.tobytes(order='C'))
    f.write(b_iz.tobytes(order='C'))
    f.write(b_in.tobytes(order='C'))
    f.write(b_hr.tobytes(order='C'))
    f.write(b_hz.tobytes(order='C'))
    f.write(b_hn.tobytes(order='C'))
    return

def stretch2d_saver(f, layer):
    v = struct.pack('@ii', layer.x_scale, layer.y_scale)
    f.write(v)
    return

savers = { 'Conv1d':conv1d_saver, 'Conv2d':conv2d_saver, 'BatchNorm1d':batchnorm1d_saver,
           'Linear':linear_saver, 'GRU':gru_saver, 'Stretch2d':stretch2d_saver }
layer_enum = { 'Conv1d':1, 'Conv2d':2, 'BatchNorm1d':3, 'Linear':4, 'GRU':5, 'Stretch2d':6 }

def save_layer(f, layer):
    layer_type_name = layer._get_name()
    v = struct.pack('@i64s', layer_enum[layer_type_name], layer.__str__().encode() )
    f.write(v)
    savers[layer_type_name](f, layer)
    return


def save_resnet_block(f, layers):
    for l in layers:
        save_layer(f, l.conv1)
        save_layer(f, l.batch_norm1)
        save_layer(f, l.conv2)
        save_layer(f, l.batch_norm2)

def save_resnet( f, model ):
    try:
        model.upsample.resnet = model.upsample.resnet1  #temp hack
    except:
        pass

    save_layer(f, model.upsample.resnet.conv_in)
    save_layer(f, model.upsample.resnet.batch_norm)
    save_resnet_block( f, model.upsample.resnet.layers )  #save the resblock stack
    save_layer(f, model.upsample.resnet.conv_out)
    save_layer(f, model.upsample.resnet_stretch)
    return

def save_upsample(f, model):
    for l in model.upsample.up_layers:
        save_layer(f, l)
    return

def save_main(f, model):
    save_layer(f, model.I)
    save_layer(f, model.rnn1)
    save_layer(f, model.rnn2)
    save_layer(f, model.fc1)
    save_layer(f, model.fc2)
    save_layer(f, model.fc3)
    return

if __name__ == "__main__":

    elSize = 4  # change to 2 for fp16

    output_path = './'
    checkpoint_file_name = './checkpoints/biaobei_mol.wavernn/latest_weights.pyt'
    save_model = output_path+'/biaobei_wavernn_256.bin'

    device = torch.device("cpu")

    wavernn_hp.configure('./hparams.py')
    model = WaveRNN(rnn_dims=wavernn_hp.voc_rnn_dims,
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

    voc_load_path = checkpoint_file_name
    model.load(voc_load_path)
    model = model.eval()

    print("Parameter Count:")
    print("I: %.3f million" % (num_params_count(model.I)))
    print("Upsample: %.3f million" % (num_params_count(model.upsample)))
    print("rnn1: %.3f million" % (num_params_count(model.rnn1)))
    print("rnn2: %.3f million" % (num_params_count(model.rnn2)))
    print("fc1: %.3f million" % (num_params_count(model.fc1)))
    print("fc2: %.3f million" % (num_params_count(model.fc2)))
    print("fc3: %.3f million" % (num_params_count(model.fc3)))
    print(model)

    with open(save_model,'wb') as f:
        v = struct.pack('@iiii', wavernn_hp.voc_res_blocks, len(wavernn_hp.voc_upsample_factors),
                        np.prod(wavernn_hp.voc_upsample_factors), wavernn_hp.voc_pad)
        f.write(v)
        save_resnet(f, model)
        save_upsample(f, model)
        save_main(f, model)
    print('\ndone')
