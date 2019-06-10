import mxnet as mx
import numpy as np

# mb1_ic3oc64_ih1024oh1024kh3sh1dh0ph1_iw1024ow1024kw3sw1dw0pw1
data = mx.sym.Variable(name='data',shape=(1,3,1024,1024))
weight = mx.sym.Variable(name='weight',shape=(64,3,3,3))
bias = mx.sym.Variable(name='bias',shape=(64))
conv = mx.sym.Convolution(data=data, weight=weight, bias=bias, num_filter=64, kernel=(3,3), pad=(1,1))
sym = conv.get_backend_symbol('MKLDNN')

#data = mx.sym.Variable(name='data',shape=(1,64,512,512))
#weight = mx.sym.Variable(name='weight',shape=(128,64,3,3))
#bias = mx.sym.Variable(name='bias',shape=(128))
#conv = mx.sym.Convolution(data=data, weight=weight, bias=bias, num_filter=128, kernel=(3,3), pad=(1,1))

data_v = mx.nd.ones([1,3,1024,1024])
weight_v = mx.nd.random.normal(shape=(64,3,3,3))
bias_v = mx.nd.random.normal(shape=(64,))

executor = sym.bind(ctx=mx.cpu(0),args={'data': data_v,
                                         'weight':weight_v,
                                         'bias':bias_v})
executor.forward()
#print("-----------weight-----------")
#print(weight_v)
#print("------------bais------------")
#print(bias_v)
print("-----------output-----------")
executor.outputs[0].wait_to_read()
print(executor.outputs[0])
