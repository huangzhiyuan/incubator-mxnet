import numpy as np
import mxnet as mx
import copy
import math
import random
import itertools
from distutils.version import LooseVersion
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
from mxnet.base import py_str, MXNetError, _as_list
# from common import setup_module, with_seed, teardown, assert_raises_cudnn_not_satisfied, assertRaises
# from common import run_in_spawned_process
from nose.tools import assert_raises
import unittest
import os

def check_quantized_upsample(data_shape, qdtype):
    # if is_test_for_native_cpu():
    #     print('skipped testing _contrib_quantized_upsampling for native cpu since it is not supported yet')
    #     return
    # elif qdtype == 'int8' and is_test_for_mkldnn():
    #     print('skipped testing _contrib_quantized_upsampling for mkldnn cpu int8 since it is not supported yet')
    #     return
    # elif is_test_for_gpu():
    #     print('skipped testing _contrib_quantized_upsampling for gpu since it is not supported yet')
    #     return
    data = mx.sym.Variable(name='data', shape=data_shape, dtype='float32')
    # act_fp32 = mx.sym.Activation(data=data, act_type='relu', name='relu')
    act_fp32 = mx.symbol.UpSampling(data, scale=2, sample_type='nearest')
    arg_shapes, _, _ = act_fp32.infer_shape(data=data_shape)
    arg_names = act_fp32.list_arguments()
    act_fp32_exe = act_fp32.simple_bind(ctx=mx.current_context(), grad_req='null')
    if qdtype == 'uint8':
        data_low = 0.0
        data_high = 127.0
    else:
        data_low = -127.0
        data_high = 127.0

    arg = mx.nd.random.uniform(low=data_low, high=data_high, shape=data_shape).astype(qdtype)
    act_fp32_exe.arg_dict[arg_names[0]][:] = arg
    output = act_fp32_exe.forward()[0]

    qdata = mx.sym.Variable(name='qdata', shape=data_shape, dtype=qdtype)
    min_data = mx.sym.Variable(name='min_data')
    max_data = mx.sym.Variable(name='max_data')
    quantized_up = mx.sym.contrib.quantized_upsampling(data=qdata, num_args=1, min_data=min_data, max_data=max_data, scale=2, sample_type='nearest')
    up_int8_exe = quantized_up.simple_bind(ctx=mx.current_context(), grad_req='null')
    qarg_names = quantized_up.list_arguments()

    up_int8_exe.arg_dict[qarg_names[0]][:] = act_fp32_exe.arg_dict[arg_names[0]].astype(qdtype)
    quantized_range_min = mx.nd.min(up_int8_exe.arg_dict[qarg_names[0]][:])
    quantized_range_max = mx.nd.max(up_int8_exe.arg_dict[qarg_names[0]][:])
    up_int8_exe.arg_dict[qarg_names[1]][:] = quantized_range_min.astype(qdtype)
    up_int8_exe.arg_dict[qarg_names[2]][:] = quantized_range_max.astype(qdtype)
    qoutput, min_range, max_range = up_int8_exe.forward()
    print(qoutput[0])

    assert_almost_equal(output.asnumpy(), qoutput.asnumpy())
    assert_almost_equal(min_range.asscalar(), quantized_range_min.asscalar())
    assert_almost_equal(max_range.asscalar(), quantized_range_max.asscalar())


for qdtype in ['uint8']:
    # check_quantized_upsample((10,), qdtype)
    # check_quantized_upsample((10, 15), qdtype)
    # check_quantized_upsample((10, 15, 18), qdtype)
    check_quantized_upsample((1, 1, 4, 4), qdtype)