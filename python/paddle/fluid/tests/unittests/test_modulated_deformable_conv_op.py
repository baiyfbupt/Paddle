#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np

import paddle.fluid.core as core
from op_test import OpTest


def dmc_bilinear(data_im, height, width, h, w):
    h_low = int(np.floor(h))
    w_low = int(np.floor(w))
    h_high = h_low + 1
    w_high = w_low + 1

    lh = h - h_low
    lw = w - w_low
    hh = 1 - lh
    hw = 1 - lw

    v1 = 0
    if h_low >= 0 and w_low >= 0:
        v1 = data_im[h_low, w_low]
    v2 = 0
    if h_low >= 0 and w_high <= width - 1:
        v2 = data_im[h_low, w_high]
    v3 = 0
    if h_high <= height - 1 and w_low >= 0:
        v3 = data_im[h_high, w_low]
    v4 = 0
    if h_high <= height - 1 and w_high <= width - 1:
        v4 = data_im[h_high, w_high]

    w1, w2, w3, w4 = hh * hw, hh * lw, lh * hw, lh * lw
    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4

    return val


def dconv_im2col_gemm(input, offset, mask, filter, conv2d_param):
    in_n, in_c, in_h, in_w = input.shape
    out_c, f_c, f_h, f_w = filter.shape

    assert offset.shape == (in_n, 2 * f_h * f_w, in_h, in_w)
    assert mask.shape == (in_n, f_h * f_w, in_h, in_w)
    assert f_c == in_c
    
    out_h = in_h
    out_w = in_w
    out = np.zeros((in_n, out_c, out_h * out_w))
    input_pad = np.pad(input, ((0, ), (0, ), ((f_h - 1) // 2, ), (
        (f_w - 1) // 2, )),
                       mode='constant',
                       constant_values=0)
    col_buffer = np.zeros((in_n, in_c * f_h * f_w, in_h * in_w))
    in_n, in_c, in_pad_h, in_pad_w = input_pad.shape

    for n in range(in_n):
        for c in range(in_c):
            for h in range(in_h):
                for w in range(in_w):
                    for kh in range(f_h):
                        for kw in range(f_w):
                            offset_h_table = \
                                    offset[n, ::2, h, w].reshape(f_h, f_w)
                            offset_w_table = \
                                    offset[n, 1::2, h, w].reshape(f_h, f_w)
                            mask_table = \
                                mask[n, :, h, w].reshape(f_h, f_w)
                            offset_h = offset_h_table[kh, kw]
                            offset_w = offset_w_table[kh, kw]
                            val = 0
                            im_h = h + kh + offset_h - (f_h - 1) // 2
                            im_w = w + kw + offset_w - (f_w - 1) // 2
                            if im_h > -1 and im_w > -1 and \
                                im_h < in_h and im_w < in_h:
                                val = dmc_bilinear(input[n, c],
                                    in_h, in_w, im_h, im_w)
                            val_out = val * mask_table[kh, kw]
                            col_buffer[n, c * f_h * f_w + kh *f_w + kw, h * in_w + w] = val_out
    weight = filter.reshape(out_c, f_c * f_h * f_w)
    for n in range(in_n):
        out[n] = np.matmul(weight, col_buffer[n])
    out = out.reshape(in_n, out_c, out_h, out_w)
    return out


class TestModulatedDeformableConvOp(OpTest):
    def setUp(self):
        self.op_type = "modulated_deformable_conv"
        self.dtype = np.float32
        self.init_group()
        self.init_dilation()
        self.init_test_case()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        offset = 10 * np.random.random(self.offset_size).astype(self.dtype)
        mask = 10 * np.random.random(self.mask_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)

        output = dconv_im2col_gemm(input, offset, mask, filter, conv2d_param)
        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Offset': OpTest.np_dtype_to_fluid_dtype(offset),
            'Mask': OpTest.np_dtype_to_fluid_dtype(mask),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'deformable_groups': self.deformable_groups,
            'im2col_step': self.im2col_step,
            'dilations': self.dilations,
        }
        self.outputs = {'Output': output}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=1e-5)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, {'Input', 'Offset', 'Mask', 'Filter'},
            'Output',
            max_relative_error=0.05)
     
    def test_check_grad_no_filter(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['Input', 'Offset', 'Mask'],
            'Output',
            max_relative_error=0.05,
            no_grad_set=set(['Filter']))
    

    def test_check_grad_no_input(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['Filter', 'Offset', 'Mask'],
            'Output',
            max_relative_error=0.1,
            no_grad_set=set(['Input']))

    def test_check_grad_no_offset_no_mask(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['Input', 'Filter'],
            'Output',
            max_relative_error=0.05,
            no_grad_set=set(['Offset', 'Mask']))

    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.offset_size = [2, 18, 5, 5]
        self.mask_size = [2, 9, 5, 5]
        self.im2col_step = 1
        self.deformable_groups = 1

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1

if __name__ == '__main__':
    unittest.main()
