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

#include "paddle/fluid/operators/multiclass_nms_op.h"


namespace paddle {
namespace operators {

class MultiClassNMSOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("BBoxes"),
                   "Input(BBoxes) of MultiClassNMS should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Scores"),
                   "Input(Scores) of MultiClassNMS should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MultiClassNMS should not be null.");

    auto box_dims = ctx->GetInputDim("BBoxes");
    auto score_dims = ctx->GetInputDim("Scores");

    PADDLE_ENFORCE_EQ(box_dims.size(), 3,
                      "The rank of Input(BBoxes) must be 3.");
    PADDLE_ENFORCE_EQ(score_dims.size(), 3,
                      "The rank of Input(Scores) must be 3.");
    PADDLE_ENFORCE_EQ(box_dims[2], 4,
                      "The 2nd dimension of Input(BBoxes) must be 4, "
                      "represents the layout of coordinate "
                      "[xmin, ymin, xmax, ymax]");
    PADDLE_ENFORCE_EQ(box_dims[1], score_dims[2],
                      "The 1st dimensiong of Input(BBoxes) must be equal to "
                      "3rd dimension of Input(Scores), which represents the "
                      "predicted bboxes.");

    // Here the box_dims[0] is not the real dimension of output.
    // It will be rewritten in the computing kernel.
    ctx->SetOutputDim("Out", {box_dims[1], 6});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(
            ctx.Input<framework::LoDTensor>("Scores")->type()),
        platform::CPUPlace());
  }
};

class MultiClassNMSOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MultiClassNMSOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("BBoxes",
             "(Tensor) A 3-D Tensor with shape [N, M, 4] represents the "
             "predicted locations of M bounding bboxes, N is the batch size. "
             "Each bounding box has four coordinate values and the layout is "
             "[xmin, ymin, xmax, ymax].");
    AddInput("Scores",
             "(Tensor) A 3-D Tensor with shape [N, C, M] represents the "
             "predicted confidence predictions. N is the batch size, C is the "
             "class number, M is number of bounding boxes. For each category "
             "there are total M scores which corresponding M bounding boxes. "
             " Please note, M is equal to the 1st dimension of BBoxes. ");
    AddAttr<int>(
        "background_label",
        "(int, defalut: 0) "
        "The index of background label, the background label will be ignored. "
        "If set to -1, then all categories will be considered.")
        .SetDefault(0);
    AddAttr<float>("score_threshold",
                   "(float) "
                   "Threshold to filter out bounding boxes with low "
                   "confidence score. If not provided, consider all boxes.");
    AddAttr<int>("nms_top_k",
                 "(int64_t) "
                 "Maximum number of detections to be kept according to the "
                 "confidences aftern the filtering detections based on "
                 "score_threshold");
    AddAttr<float>("nms_threshold",
                   "(float, defalut: 0.3) "
                   "The threshold to be used in NMS.")
        .SetDefault(0.3);
    AddAttr<float>("nms_eta",
                   "(float) "
                   "The parameter for adaptive NMS.")
        .SetDefault(1.0);
    AddAttr<int>("keep_top_k",
                 "(int64_t) "
                 "Number of total bboxes to be kept per image after NMS "
                 "step. -1 means keeping all bboxes after NMS step.");
    AddOutput("Out",
              "(LoDTensor) A 2-D LoDTensor with shape [No, 6] represents the "
              "detections. Each row has 6 values: "
              "[label, confidence, xmin, ymin, xmax, ymax], No is the total "
              "number of detections in this mini-batch. For each instance, "
              "the offsets in first dimension are called LoD, the number of "
              "offset is N + 1, if LoD[i + 1] - LoD[i] == 0, means there is "
              "no detected bbox.");
    AddComment(R"DOC(
This operator is to do multi-class non maximum suppression (NMS) on a batched
of boxes and scores.

In the NMS step, this operator greedily selects a subset of detection bounding
boxes that have high scores larger than score_threshold, if providing this
threshold, then selects the largest nms_top_k confidences scores if nms_top_k
is larger than -1. Then this operator pruns away boxes that have high IOU
(intersection over union) overlap with already selected boxes by adaptive
threshold NMS based on parameters of nms_threshold and nms_eta.

Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
per image if keep_top_k is larger than -1.

This operator support multi-class and batched inputs. It applying NMS
independently for each class. The outputs is a 2-D LoDTenosr, for each
image, the offsets in first dimension of LoDTensor are called LoD, the number
of offset is N + 1, where N is the batch size. If LoD[i + 1] - LoD[i] == 0,
means there is no detected bbox for this image. If there is no detected boxes
for all images, all the elements in LoD are 0, and the Out only contains one
value which is -1.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(multiclass_nms, ops::MultiClassNMSOp,
                  ops::MultiClassNMSOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    multiclass_nms,
    ops::MultiClassNMSKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MultiClassNMSKernel<paddle::platform::CPUDeviceContext, double>);
