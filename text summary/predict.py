FN = 'predict'
import os
os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'

import keras
keras.__version__

FN0 = 'vocabulary-embedding'
FN1 = 'train'

maxlend=50 
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 512
rnn_layers = 3  # match FN1
batch_norm=False

activation_rnn_size = 40 if maxlend else 0

# training parameters
seed=42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
batch_size=64

nb_train_samples = 30000
nb_val_samples = 3000


