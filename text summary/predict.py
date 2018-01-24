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

