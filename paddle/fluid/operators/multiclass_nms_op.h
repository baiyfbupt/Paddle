/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

constexpr int64_t kOutputDim = 6;
constexpr int64_t kBBoxSize = 4;

using Tensor = paddle::framework::Tensor;
using LoDTensor = paddle::framework::LoDTensor;


template <class T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;         
}

template <class T>
static inline void GetMaxScoreIndex(
    const std::vector<T>& scores, const T threshold, int top_k,
    std::vector<std::pair<T, int>>* sorted_indices) {
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      sorted_indices->push_back(std::make_pair(scores[i], i));//将大于分数阈值的（分数，序号）存入sorted_indices
    }
  }
  std::stable_sort(sorted_indices->begin(), sorted_indices->end(),
                   SortScorePairDescend<int>);//stable_sort use self-defined compare func
  if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(top_k);//keep top_k, dump the others
  }
}

template <class T>
static inline T BBoxArea(const T* box, const bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<T>(0.);
  } else {
    const T w = box[2] - box[0];
    const T h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

template <class T>
static inline T JaccardOverlap(const T* box1, const T* box2,
                               const bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = std::max(box1[0], box2[0]);
    const T inter_ymin = std::max(box1[1], box2[1]);
    const T inter_xmax = std::min(box1[2], box2[2]);
    const T inter_ymax = std::min(box1[3], box2[3]);
    const T inter_w = inter_xmax - inter_xmin;
    const T inter_h = inter_ymax - inter_ymin;
    const T inter_area = inter_w * inter_h;
    const T bbox1_area = BBoxArea<T>(box1, normalized);
    const T bbox2_area = BBoxArea<T>(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template<typename T>
inline HOSTDEVICE void NMSFast(const Tensor& bbox, const Tensor& scores,//SLICED_bbox[M,4], S_SLICED_scores[M]
               const T score_threshold, const T nms_threshold, const T eta,// M bounding box number
               const int64_t top_k, std::vector<int>* selected_indices) {
    // The total boxes for each instance.
    int64_t num_boxes = bbox.dims()[0];//M
    // 4: [xmin ymin xmax ymax]
    int64_t box_size = bbox.dims()[1];//4

    std::vector<T> scores_data(num_boxes);
    std::copy_n(scores.data<T>(), num_boxes, scores_data.begin());//M bbox scores
    std::vector<std::pair<T, int>> sorted_indices;
    GetMaxScoreIndex(scores_data, score_threshold, top_k, &sorted_indices);//get top_k <scores,index of M bbox>

    selected_indices->clear();//OUTPUT
    T adaptive_threshold = nms_threshold;
    const T* bbox_data = bbox.data<T>();

    while (sorted_indices.size() != 0) {            
      const int idx = sorted_indices.front().second;//idx : current highest score index
      bool keep = true;
      for (size_t k = 0; k < selected_indices->size(); ++k) {//M bbox number
        if (keep) {
          const int kept_idx = (*selected_indices)[k];
          T overlap = JaccardOverlap<T>(bbox_data + idx * box_size,
                                        bbox_data + kept_idx * box_size, true);
          keep = overlap <= adaptive_threshold;       //iou > threshold, dump the box
        } else {
          break;
        }
      }
      if (keep) {
        selected_indices->push_back(idx);       //NMS output
      }
      sorted_indices.erase(sorted_indices.begin());
      if (keep && eta < 1 && adaptive_threshold > 0.5) {
        adaptive_threshold *= eta;
      }
    }
  }

template<typename T>
inline HOSTDEVICE void MultiClassNMS(const paddle::framework::ExecutionContext& ctx,
                     const Tensor& scores, const Tensor& bboxes,
                     std::map<int, std::vector<int>>* indices,
                     int* num_nmsed_out) {
    int64_t background_label = ctx.Attr<int>("background_label");
    int64_t nms_top_k = ctx.Attr<int>("nms_top_k");
    int64_t keep_top_k = ctx.Attr<int>("keep_top_k");
    T nms_threshold = static_cast<T>(ctx.Attr<float>("nms_threshold"));
    T nms_eta = static_cast<T>(ctx.Attr<float>("nms_eta"));
    T score_threshold = static_cast<T>(ctx.Attr<float>("score_threshold"));

    int64_t class_num = scores.dims()[0];//SLICED_bbox[M,4], SLICED_scores[C,M]
    int64_t predict_dim = scores.dims()[1];//C class number, M bounding box number
    int num_det = 0;
    for (int64_t c = 0; c < class_num; ++c) {     //loop process multiclass                     
      if (c == background_label) continue;
      Tensor score = scores.Slice(c, c + 1);//M bounding box
      NMSFast<T>(bboxes, score, score_threshold, nms_threshold, nms_eta, nms_top_k,
              &((*indices)[c]));//number c class nms-output
      num_det += (*indices)[c].size();//total class indices number
    }

    *num_nmsed_out = num_det;
    const T* scores_data = scores.data<T>();
    if (keep_top_k > -1 && num_det > keep_top_k) {
      std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
      for (const auto& it : *indices) {//indices <class,<list of indices>>
        int label = it.first;
        const T* sdata = scores_data + label * predict_dim;
        const std::vector<int>& label_indices = it.second;
        for (size_t j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          PADDLE_ENFORCE_LT(idx, predict_dim);
          score_index_pairs.push_back(
              std::make_pair(sdata[idx], std::make_pair(label, idx)));//scores,(label,idx)
        }
      }
      // Keep top k results per image.
      std::stable_sort(score_index_pairs.begin(), score_index_pairs.end(),
                       SortScorePairDescend<std::pair<int, int>>);
      score_index_pairs.resize(keep_top_k);

      // Store the new indices.
      std::map<int, std::vector<int>> new_indices;
      for (size_t j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      new_indices.swap(*indices);
      *num_nmsed_out = keep_top_k;
    }
  }

template<typename T>
inline HOSTDEVICE void MultiClassOutput(const Tensor& scores, const Tensor& bboxes,
                        const std::map<int, std::vector<int>>& selected_indices,
                        Tensor* outs) {
    int predict_dim = scores.dims()[1];
    auto* scores_data = scores.data<T>();
    auto* bboxes_data = bboxes.data<T>();
    auto* odata = outs->data<T>();

    int count = 0;
    for (const auto& it : selected_indices) {
      int label = it.first;
      const T* sdata = scores_data + label * predict_dim;
      const std::vector<int>& indices = it.second;
      for (size_t j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        const T* bdata = bboxes_data + idx * kBBoxSize;
        odata[count * kOutputDim] = label;           // label
        odata[count * kOutputDim + 1] = sdata[idx];  // score
        // xmin, ymin, xmax, ymax
        std::memcpy(odata + count * kOutputDim + 2, bdata, 4 * sizeof(T));
        count++;
      }
    }
  }

namespace paddle {
namespace operators {


template <typename DeviceContext, typename T>
class MultiClassNMSKernel : public framework::OpKernel<T> {
 public:

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* boxes = ctx.Input<Tensor>("BBoxes");
    auto* scores = ctx.Input<Tensor>("Scores");
    auto* outs = ctx.Output<LoDTensor>("Out");

    auto score_dims = scores->dims();//bbox[N,M,4], scores[N,C,M] N-batch-size
                                     //C class number, M bounding box number           
    int64_t batch_size = score_dims[0];
    int64_t class_num = score_dims[1];
    int64_t predict_dim = score_dims[2];
    int64_t box_dim = boxes->dims()[2];//4

    std::vector<std::map<int, std::vector<int>>> all_indices;
    std::vector<size_t> batch_starts = {0};
    for (int64_t i = 0; i < batch_size; ++i) {
      Tensor ins_score = scores->Slice(i, i + 1);//i-st batch
      ins_score.Resize({class_num, predict_dim});//[C,M]

      Tensor ins_boxes = boxes->Slice(i, i + 1);//i-st batch
      ins_boxes.Resize({predict_dim, box_dim});//[M,4]

      std::map<int, std::vector<int>> indices;//OUTPUT:CURRENT BATCH INDICES
      int num_nmsed_out = 0;//OUTPUT
      MultiClassNMS<T>(ctx, ins_score, ins_boxes, &indices, &num_nmsed_out);
      all_indices.push_back(indices);//ALL INDICES
      batch_starts.push_back(batch_starts.back() + num_nmsed_out);//every batch output
    }

    int num_kept = batch_starts.back();
    if (num_kept == 0) {
      T* od = outs->mutable_data<T>({1}, ctx.GetPlace());
      od[0] = -1;
    } else {
      outs->mutable_data<T>({num_kept, kOutputDim}, ctx.GetPlace());
      for (int64_t i = 0; i < batch_size; ++i) {
        Tensor ins_score = scores->Slice(i, i + 1);//every batch
        ins_score.Resize({class_num, predict_dim});

        Tensor ins_boxes = boxes->Slice(i, i + 1);//every batch
        ins_boxes.Resize({predict_dim, box_dim});

        int64_t s = batch_starts[i];
        int64_t e = batch_starts[i + 1];
        if (e > s) {
          Tensor out = outs->Slice(s, e);
          MultiClassOutput<T>(ins_score, ins_boxes, all_indices[i], &out);
        }
      }
    }

    framework::LoD lod;
    lod.emplace_back(batch_starts);

    outs->set_lod(lod);
  }
};

}//namespace operator
}//namespace paddle
