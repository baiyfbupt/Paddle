// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/modulated_deformable_conv_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T>
__device__ T dmcn_im2col_bilinear(const T* bottom_data, const int data_width,
        const int height, const int width, T h, T w) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  T v2 = 0;
  if (h_low >=0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
__global__ void modulated_deformable_im2col(
    const int nthreads, const T* input, const T* offset, const T* mask,
    const int inage_h, const int image_w, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int out_h, const int out_w,
    const int deformable_groups, const int channel_per_deformable_group,
    T* col) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t i = index; i < nthreads; i += offset){
    // index of output matrix
    const int w_col = i % out_w;
    const int h_col = (i / out_w) % out_h;
    const int c_im = (i / out_w) / out_h;
    const int c_col = c_im * kernel_h * kernel *w;

    // conpute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    T* col_ptr = col + (c_col * out_h + h_col) * out_w + w_col;
    const T* input_ptr = input + (c_im * image_h + h_in) * image_w + w_in;
    const T* offset_ptr = offset
        + deformable_group_index * 2 * kernel_h * kernel_w * out_h * out_w;
    const T* mask_ptr = mask
        + deformable_group_index * kernel_h * kernel_w * out_h * out_w;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int offset_h_ptr = ((2 * (i * kernel_w + j)) * out_h + h_col)
                               * out_w + w_col;
        const int offset_w_ptr = 
            ((2 * (i * kernel_w +j) + 1) * height_col + h_col) * out_w + w_col;
        const int mask_hw_ptr = ((i * kernel_w + j) * out_h + h_col)
                               * out_w + w_col;
        const T offset_h = offset_ptr[offset_h_ptr];
        const T offset_w = offset_ptr[offset_w_ptr];
        const T mask = mask_ptr[mask_hw_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        
        if (h_im > -1 && w_im > -1 && h_im < image_h && w_im < image_w) {
            val = dmcn_im2col_bilinear(input_ptr, image_w, image_h, image_w,
                                       h_im, w_im);
        }
        *col_ptr = val * mask;
        data_col_ptr += out_h * out_w;
      }
    }
  }
}


template <typename DeviceContext, typename T>
class ModulatedDeformableConvCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {

    const Tensor* input = ctx.Input<Tensor>("Input");
    Tensor offset = *ctx.Input<Tensor>("Offset");
    Tensor mask = *ctx.Input<Tensor>("Mask");
    Tensor filter = *ctx.Input<Tensor>("Filter");
    Tensor* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());

    int groups = ctx.Attr<int>("groups");
    int deformable_groups = ctx.Attr<int>("deformable_groups");
    int im2col_step = ctx.Attr<int>("im2col_step");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");

    auto &dev_ctx = ctx.cuda_device_context();

    const int batch_size = static_cast<int>(input->dims()[0]);

    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
    std::vector<int64_t> offset_shape_vec(framework::vectorize(offset.dims()));
    std::vector<int64_t> mask_shape_vec(framework::vectorize(mask.dims()));
    std::vector<int64_t> output_shape_vec(framework::vectorize(output->dims()));

    // get col_shape in the im2col calculation
    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2*data_dim);
    col_shape_vec[0] = input->dims[1] / groups;

    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
    }
    framework::DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    framework::DDim col_matrix_shape =
        framework::flatten_to_2d(col_shape, data_dim + 1);

    bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
    Tensor col;
    Tensor col_matrix;
    if (is_expand) {
        col = ctx.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
        col_matrix.ShareDataWith(col);
        col_matrix.Resize(col_matrix_shape);
    }

    // input
    framework::DDim input_shape =
        framework::slice_ddim(input->dims(), 1, input->dims().size());
    // offset
    framework::DDim offset_shape =
        framework::slice_ddim(offset.dims(), 1, offset.dims().size());
    // mask
    framework::DDim mask_shape =
        framework::slice_ddim(mask.dims(), 1, mask->dims().size());
    // filter
    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);
    // output
    framework::DDim output_matrix_shape = {
        output->dims()[1],
        output->numel() / (output->dims()[0] * output->dims()[1])};

    // convolution operator: im2col(or vol2col) + gemm
    int in_step = static_cast<int>(input->dims()[1]) / groups;
    int offset_step = static_cast<int>(offset.dims()[1]) / groups;
    int mask_step = static_cast<int>(mask.dims()[1]) / groups;
    int out_step = static_cast<int>(output->dims()[1]) / groups;

    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);



    for (int i = 0; i < batch_size; i++) {
      Tensor in_batch = input->Slice(i, i + 1).Resize(input_shape);
      Tensor offset_batch = offset.Slice(i, i + 1).Resize(offset_shape);
      Tensor mask_shape = mask.Slice(i, i + 1).Resize(mask_shape);
      Tensor out_batch = output->Slice(i, i + 1).Resize(output_matrix_shape);

      for (int g = 0; g < groups; g++){
        Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);
        Tensor offset_slice = offset_batch.Slice(g * offset_step,
                                                 (g + 1) * offset_step);
        Tensor mask_slice = mask_batch.Slice(g * mask_step,
                                                 (g + 1) * mask_step);
        int num_kernels = in_slice.dims()[0]  * out_batch.numel()
                          / out_batch.dims()[0];
        int blocks = NumBlocks(num_kernels);
        int threads = kNumCUDAThreads;
        size_t channel_per_deformable_group = in_step / deformable_groups;

        // im2col
        // TODO: Need to check no expand is correct
        modulated_deformable_im2col<T><<<blocks,
            threads, 0, dev_ctx.stream()>>>(num_kernels,
                                            in_slice.data<T>(),
                                            offset_slice.data<T>(),
                                            mask_slice.data<T>(),
                                            in_slice.dims[1],
                                            in_slice.dims[2],
                                            filter_shape_vec[2],
                                            filter_shape_vec[3],
                                            paddings[0],
                                            paddings[1],
                                            strides[0],
                                            strides[1],
                                            dilations[0],
                                            dilations[1],
                                            output_shape_vec[2],
                                            output_shape_vec[3],
                                            deformable_groups,
                                            channel_per_deformable_group,
                                            &col);

        // gemm
        Tensor out_slice = out_batch.Slice(g * out_step, (g + 1) * out_step);
        Tensor filter_slice = filter.Slice(g * out_step, (g + 1) * out_step);
        blas.MatMul(filter_slice, false, col_matrix, false, T(1.0), &out_slice,
                    T(0.0));
      }
    }
    //TODO: check bias
  }
};

template <typename DeviceContext, typename T>
class TreeConvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *out_g = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *in_g = ctx.Output<Tensor>(framework::GradVarName("NodesVector"));
    auto *filter_g = ctx.Output<Tensor>(framework::GradVarName("Filter"));
    int max_depth = ctx.Attr<int>("max_depth");
    auto *Embeddings = ctx.Input<Tensor>("NodesVector");
    auto *edges = ctx.Input<Tensor>("EdgeSet");
    auto *Filter = ctx.Input<Tensor>("Filter");
    math::Tree2ColFunctor<DeviceContext, T> tree2col;
    math::Col2TreeFunctor<DeviceContext, T> col2tree;
    math::SetConstant<DeviceContext, T> constant;
    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

    Tensor W;
    W.ShareDataWith(*Filter);
    W.Resize(framework::flatten_to_2d(Filter->dims(), 1));

    int batch_size = static_cast<int>(Embeddings->dims()[0]);

    auto edge_set_slicedim = framework::slice_ddim(
        edges->dims(), 1, static_cast<int>(edges->dims().size()));

    auto embedding_slicedim = framework::slice_ddim(
        Embeddings->dims(), 1, static_cast<int>(Embeddings->dims().size()));

    auto out_grad_dims = framework::slice_ddim(
        out_g->dims(), 1, static_cast<int>(out_g->dims().size()));
    out_grad_dims = framework::flatten_to_2d(out_grad_dims, 1);
    if (filter_g) {
      filter_g->mutable_data<T>(Filter->dims(), ctx.GetPlace());
      Tensor f_g;
      f_g.ShareDataWith(*filter_g);
      f_g.Resize(framework::flatten_to_2d(Filter->dims(), 2));
      constant(dev_ctx, filter_g, 0);
      for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        auto edge_set =
            edges->Slice(batch_id, batch_id + 1).Resize(edge_set_slicedim);
        auto embeddings = Embeddings->Slice(batch_id, batch_id + 1)
                              .Resize(embedding_slicedim);
        auto out_grad =
            out_g->Slice(batch_id, batch_id + 1).Resize(out_grad_dims);
        Tensor patch;
        tree2col(dev_ctx, edge_set, embeddings, &patch, max_depth);
        blas.MatMul(patch, true, out_grad, false, T(1.0), &f_g, T(1.0));
      }
    }
    if (in_g) {
      auto input_grad_dims = framework::slice_ddim(
          in_g->dims(), 1, static_cast<int>(in_g->dims().size()));
      in_g->mutable_data<T>(Embeddings->dims(), ctx.GetPlace());
      constant(dev_ctx, in_g, 0);
      for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        auto edge_set =
            edges->Slice(batch_id, batch_id + 1).Resize(edge_set_slicedim);
        auto out_grad =
            out_g->Slice(batch_id, batch_id + 1).Resize(out_grad_dims);
        auto in_grad =
            in_g->Slice(batch_id, batch_id + 1).Resize(input_grad_dims);
        Tensor in_grad_temp;
        col2tree(dev_ctx, edge_set, out_grad, &in_grad_temp, max_depth);
        blas.MatMul(in_grad_temp, false, W, true, &in_grad);
      }
    }
  }
};

} // namespace operators
} // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(
    modulated_deformable_conv,
    ops::ModulatedDeformableConvCUDAKernel<CUDA, float>,
    ops::ModulatedDeformableConvCUDAKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(
    modulated_deformable_conv_grad,
    ops::ModulatedDeformableConvGradCUDAKernel<CUDA, float>,
    ops::ModulatedDeformableConvGradCUDAKernel<CUDA, double>);
