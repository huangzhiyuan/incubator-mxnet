/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

 /*!
 * Copyright (c) 2019 by Contributors
 * \file quantized_upsample.cc
 * \author Zhiyuan Huang
*/

#include "../upsampling-inl.h"
#include <nnvm/op_attr_types.h>

namespace mxnet {
namespace op {

namespace quantized_upsample_enum {
enum QuantizedUpsampleInputs { kData, kInMin, kInMax };
enum QuantizedUpsampleOutputs { kOut, kOutMin, kOutMax };
}  // namespace quantized_upsample_enum

bool QuantizedUpSamplingShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape> *in_shape,
                              std::vector<TShape> *out_shape) {
  const UpSamplingParam param_ = nnvm::get<UpSamplingParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args * 3));
  CHECK_GE(out_shape->size(), 3U);
  const mxnet::TShape &dshape = (*in_shape)[0];
  mxnet::TShape oshape = dshape;
  oshape[1] = 0;
  auto& shape = (*in_shape)[0];
  CHECK_EQ(shape.ndim(), 4U) << "UpSamplingNearest: Input data should be 4D "
                                "in (batch, channel, y, x)";
  int oh = dshape[2] * param_.scale, ow = dshape[3] * param_.scale;
  CHECK_EQ(oh % shape[2], 0U)
      << "UpSamplingNearest: input height of " << shape[2]
      << "does not divide output height of " << oh;
  CHECK_EQ(ow % shape[3], 0U)
      << "UpSamplingNearest: input width of " << shape[3]
      << "does not divide output width of " << ow;
  if (param_.multi_input_mode == up_enum::kSum) {
    CHECK(oshape[1] == 0 || oshape[1] == shape[1])
        << "Number of channels must be the same when multi_input_mode==sum";
    oshape[1] = shape[1];
  } else {
    oshape[1] += shape[1];
  }
  // }
  oshape[2] = dshape[2] * param_.scale;
  oshape[3] = dshape[3] * param_.scale;

  SHAPE_ASSIGN_CHECK(*in_shape, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*in_shape, 2, mxnet::TShape{1});

  out_shape->clear();
  out_shape->push_back(oshape);
  out_shape->push_back(mxnet::TShape{1});
  out_shape->push_back(mxnet::TShape{1});
  return true;
}

bool QuantizedUpSamplingType(const nnvm::NodeAttrs& attrs,
                             std::vector<int> *in_type,
                             std::vector<int> *out_type) {
  const UpSamplingParam& param = nnvm::get<UpSamplingParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), static_cast<size_t>(param.num_args * 3));
  CHECK_GE(out_type->size(), 3U);
  int dtype = mshadow::kUint8;

  for (int i = 0; i < param.num_args; ++i) {
    if (in_type->at(i) == mshadow::kInt8) {
      dtype = mshadow::kInt8;
    } else {
      TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kUint8);
    }
  }

  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);
  return true;
}

void QuantizedUpSamplingCompute(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const UpSamplingParam& param = nnvm::get<UpSamplingParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), static_cast<size_t>(param.num_args * 3));
  CHECK_GE(inputs.size(), 3U);
  Stream<cpu>* s = ctx.get_stream<cpu>();

  CHECK(inputs[quantized_upsample_enum::kData].type_flag_ == mshadow::kUint8 ||
        inputs[quantized_upsample_enum::kData].type_flag_ == mshadow::kInt8)
      << "_contrib_quantized_upsampling op only supports uint8 and int8 as "
         "input type";

  if (param.sample_type == up_enum::kNearest && param.num_args == 1) {
    if (inputs[quantized_upsample_enum::kData].type_flag_ == mshadow::kUint8) {
      UpSamplingForward<cpu, uint8_t>(ctx, param, inputs, req, outputs);
    } else {
      UpSamplingForward<cpu, int8_t>(ctx, param, inputs, req, outputs);
    }
    float* in_min =
        inputs[quantized_upsample_enum::kInMin].get<cpu, 1, float>(s).dptr_;
    float* in_max =
        inputs[quantized_upsample_enum::kInMax].get<cpu, 1, float>(s).dptr_;
    float* out_min =
        outputs[quantized_upsample_enum::kOutMin].get<cpu, 1, float>(s).dptr_;
    float* out_max =
        outputs[quantized_upsample_enum::kOutMax].get<cpu, 1, float>(s).dptr_;

    out_min[0] = in_min[0];
    out_max[0] = in_max[0];
  } else {
    // (TODO) support upsample type kBilinear and upsample multi-inputmode sum
    LOG(FATAL) << "No implementation";
  }
}

NNVM_REGISTER_OP(_contrib_quantized_upsampling)
.describe(R"code(upsampling operator for input and output data type of int8.
Performs nearest neighbor/bilinear up sampling to inputs.
The input and output data comes with min and max thresholds for quantizing
the float32 data into int8.
 .. Note::
     This operator only supports forward propogation. DO NOT use it in training.
     This operator only supports `nearest`)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const UpSamplingParam& params = nnvm::get<UpSamplingParam>(attrs.parsed);
  return params.num_args * 3;
})
// .set_num_inputs(3)
.set_num_outputs(3)
.set_attr_parser(ParamParser<UpSamplingParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  // const UpSamplingParam& params = nnvm::get<UpSamplingParam>(attrs.parsed);
  std::vector<std::string> ret;
  ret.push_back(std::string("data"));
  ret.push_back(std::string("min_data"));
  ret.push_back(std::string("max_data"));
  // for (int i = 0; i < params.num_args; ++i) {
  //   ret.push_back(std::string("arg") + std::to_string(i));
  // }
  // for (int i = 0; i < params.num_args; ++i) {
  //   ret.push_back(std::string("arg") + std::to_string(i) + "_min");
  //   ret.push_back(std::string("arg") + std::to_string(i) + "_max");
  // }
  return ret;
})
// .set_attr<nnvm::FListInputNames>("FListInputNames",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::string>{"data", "min_data", "max_data"};
//   })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
})
.set_attr<FCompute>("FCompute<cpu>", QuantizedUpSamplingCompute)
.set_attr<mxnet::FInferShape>("FInferShape", QuantizedUpSamplingShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedUpSamplingType)
.set_attr<FNeedRequantize>("FNeedRequantize",
  [](const NodeAttrs& attrs) {
  const UpSamplingParam& param = nnvm::get<UpSamplingParam>(attrs.parsed);
  CHECK(param.sample_type == up_enum::kNearest)
    << "_contrib_quantized_upsampling only supports sample_type=kNearest for now";
  return false;
})
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("data", "NDArray-or-Symbol[]", "Array of tensors to upsample. ")
// .add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
// .add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  const UpSamplingParam& param = nnvm::get<UpSamplingParam>(n.parsed);
  if (param.sample_type == up_enum::kNearest) {
    return std::vector<ResourceRequest>();
  } else {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  }
})
.add_arguments(UpSamplingParam::__FIELDS__());

NNVM_REGISTER_OP(UpSampling)
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
    UpSamplingParam param;
    param.Init(attrs.dict);
    nnvm::NodePtr node = nnvm::Node::Create();
    if (param.sample_type == up_enum::kNearest) {
      node->attrs.op = Op::Get("_contrib_quantized_upsampling");
      node->attrs.name = "quantized_" + attrs.name;
    } else {
      node->attrs.op = nullptr;
      node->attrs.name = attrs.name;
    }
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });

}  // namespace op
}  // namespace mxnet
