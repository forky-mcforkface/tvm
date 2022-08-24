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
 * \file reduce.cc
 * \brief Reduction operators.
 */
#include "reduce.h"

#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/reduction.h>

#include <limits>
#include <numeric>

#include "../make_op.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ReduceAttrs);
TVM_REGISTER_NODE_TYPE(ArgReduceAttrs);
TVM_REGISTER_NODE_TYPE(VarianceAttrs);

template <class T>
bool GenericReduceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  ICHECK(static_cast<int>(data->shape.size()) != 0);
  std::vector<IndexExpr> in_shape(data->shape.begin(), data->shape.end());

  const T* param = attrs.as<T>();
  ICHECK(param != nullptr);

  // assign output type and shape
  auto oshape = ReduceShapeImpl(in_shape, param, reporter);
  reporter->Assign(types[1], TensorType(oshape, data->shape[0].dtype()));
  return true;
}
/*!
 * \brief ArgReduceRel Output type and shape relation evaluation function.
 * \param num_inputs Number of input types in the args.
 * \param attrs The additional attributes of the operator.
 * \param reporter The reporter to report solution to.
 * \return false if This relation cannot be resolved. true if this relation has been resolved.
 */
bool ArgReduceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  return GenericReduceRel<ReduceAttrs>(types, num_inputs, attrs, reporter);
}

/*!
 * \brief ReduceRel Output type and shape relation evaluation function.
 * \param num_inputs Number of input types in the args.
 * \param attrs The additional attributes of the operator.
 * \param reporter The reporter to report solution to.
 * \return false if This relation cannot be resolved. true if this relation has been resolved.
 */
bool ReduceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  std::vector<IndexExpr> in_shape(data->shape.begin(), data->shape.end());

  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  ICHECK(param != nullptr);

  // assign output type and shape
  auto oshape = ReduceShapeImpl(in_shape, param, reporter);
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeReduce(Expr data, Array<Integer> axis, bool keepdims, bool exclude, String op_name) {
  auto attrs = make_object<ReduceAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;
  return Call(Op::Get(op_name), {data}, Attrs(attrs), {});
}

Expr MakeOneElementReduce(Expr data, Array<Integer> axis, bool keepdims, bool exclude,
                          bool select_last_index, String op_name) {
  auto attrs = make_object<ArgReduceAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;
  attrs->select_last_index = select_last_index;
  return Call(Op::Get(op_name), {data}, Attrs(attrs), {});
}

#define RELAY_REGISTER_REDUCE_OP(OpName)                                                \
  TVM_REGISTER_GLOBAL("relay.op._make." OpName)                                         \
      .set_body_typed([](Expr data, Array<Integer> axis, bool keepdims, bool exclude) { \
        return MakeReduce(data, axis, keepdims, exclude, OpName);                       \
      });                                                                               \
  RELAY_REGISTER_OP(OpName).set_num_inputs(1).add_argument("data", "Tensor", "The input tensor.")

#define RELAY_REGISTER_ONE_ELEMENT_REDUCE_OP(OpName)                                           \
  TVM_REGISTER_GLOBAL("relay.op._make." OpName)                                                \
      .set_body_typed([](Expr data, Array<Integer> axis, bool keepdims, bool exclude,          \
                         bool select_last_index) {                                             \
        return MakeOneElementReduce(data, axis, keepdims, exclude, select_last_index, OpName); \
      });                                                                                      \
  RELAY_REGISTER_OP(OpName).set_num_inputs(1).add_argument("data", "Tensor", "The input tensor.")

Array<te::Tensor> ArgMaxCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  return ArgReduceCompute(attrs, inputs, out_type, topi::argmax);
}

RELAY_REGISTER_ONE_ELEMENT_REDUCE_OP("argmax")
    .describe(R"code(Creates an operation that finds the indices of the maximum
values over a given axis.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ArgReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("ArgReduce", GenericReduceRel<ArgReduceAttrs>)
    .set_attr<FTVMCompute>("FTVMCompute", ArgMaxCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ArgReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> ArgMinCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  return ArgReduceCompute(attrs, inputs, out_type, topi::argmin);
}

RELAY_REGISTER_ONE_ELEMENT_REDUCE_OP("argmin")
    .describe(R"code(Creates an operation that finds the indices of the minimum
values over a given axis.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ArgReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("ArgReduce", GenericReduceRel<ArgReduceAttrs>)
    .set_attr<FTVMCompute>("FTVMCompute", ArgMinCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ArgReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> SumCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::sum);
}

RELAY_REGISTER_REDUCE_OP("sum")
    .describe(R"code(Computes the sum of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  sum(data, axis=1)
  [[  4.   8.]
   [ 10.   9.]
   [ 21.   6.]]

  sum(data, axis=[1,2])
  [ 12.  19.  27.]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<FTVMCompute>("FTVMCompute", SumCompute)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> AllCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::all);
}

RELAY_REGISTER_REDUCE_OP("all")
    .describe(R"code(Computes the logical AND of boolean array elements over given axes.

Example::

  data = [[[ True,  True,  True],
           [ True,  True,  True],
           [False,  True, False]],
          [[ True, False, False],
           [ True,  True, False],
           [False,  True,  True]]]

  all(data, axis=1)
  [[False,  True, False],
   [False, False, False]]

  all(data, axis=0)
  [[ True, False, False],
   [ True,  True, False],
   [False,  True, False]]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", AllCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> AnyCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::any);
}

RELAY_REGISTER_REDUCE_OP("any")
    .describe(R"code(Computes the logical OR of boolean array elements over given axes.

Example::

  data = [[[ True,  True,  True],
           [ True,  True,  True],
           [False,  True, False]],
          [[ True, False, False],
           [ True,  True, False],
           [False,  True,  True]]]

  any(data, axis=1)
  [[True,  True, True],
   [True,  True, True]]

  any(data, axis=0)
  [[ True,  True, True],
   [ True,  True, True],
   [False,  True, True]]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", AnyCompute)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> MaxCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::max);
}

RELAY_REGISTER_REDUCE_OP("max")
    .describe(R"code(Computes the max of array elements over given axes.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", MaxCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> MinCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::min);
}

RELAY_REGISTER_REDUCE_OP("min")
    .describe(R"code(Computes the min of array elements over given axes.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", MinCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> ProdCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::prod);
}

TVM_REGISTER_GLOBAL("relay.op._make.prod").set_body_typed(Prod);

RELAY_REGISTER_OP("prod")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .describe(R"code(Computes the products of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  prod(data, axis=1)
  [35562240]

  prod(data, axis=[1,2])
  [ 36  480  2058]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", ProdCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> MeanCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
  IndexExpr count = tir::make_const(inputs[0]->dtype, 1);
  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  ICHECK(param != nullptr);
  auto axes = param->axis;
  for (int64_t i : GetReduceAxes(inputs[0]->shape.size(), param->axis, param->exclude)) {
    count *= inputs[0]->shape[i];
  }
  // Although count is created as inputs[0]->dtype,
  // its type may be changed (promoted) during multiplication
  count = cast(inputs[0]->dtype, count);
  auto res = ReduceCompute(attrs, inputs, out_type, topi::sum);
  return {topi::divide(res[0], count)};
}

RELAY_REGISTER_REDUCE_OP("mean")
    .describe(R"code(Computes the mean of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  mean(data)
  [3.22]

  mean(data, axis=[1,2])
  [ 2.  3.16666667  4.5]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", MeanCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

bool VarianceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  ICHECK(static_cast<int>(data->shape.size()) != 0);
  const auto* mean = types[1].as<TensorTypeNode>();
  if (mean == nullptr) return false;

  std::vector<IndexExpr> in_shape(data->shape.begin(), data->shape.end());
  std::vector<IndexExpr> mean_shape(mean->shape.begin(), mean->shape.end());
  ICHECK_EQ(in_shape.size(), mean_shape.size());

  const VarianceAttrs* param = attrs.as<VarianceAttrs>();
  ICHECK(param != nullptr);

  // assign output type and shape
  auto oshape = ReduceShapeImpl(in_shape, param, reporter);
  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> VarianceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                  const Type& out_type) {
  IndexExpr count = tir::make_const(inputs[0]->dtype, 1);
  const VarianceAttrs* param = attrs.as<VarianceAttrs>();
  ICHECK(param != nullptr);
  auto axes = param->axis;
  bool unbiased = param->unbiased;
  auto data = inputs[0];
  auto mean = inputs[1];
  for (int64_t i : GetReduceAxes(data->shape.size(), param->axis, param->exclude)) {
    count *= data->shape[i];
  }
  if (unbiased) {
    count -= 1;
  }
  std::vector<Integer> expand_shape;
  auto diff = topi::subtract(data, mean);
  auto sq_diff = topi::multiply(diff, diff);
  if (param->exclude) {
    axes = GetExcludeAxes(sq_diff->shape.size(), param->axis);
    ICHECK_NE(axes.size(), 0);
  }
  auto var = topi::divide(topi::sum(sq_diff, axes, param->keepdims, false), count);

  return {var};
}

Expr MakeVariance(Expr data, Expr mean, Array<Integer> axis, bool keepdims, bool exclude,
                  bool unbiased = false) {
  auto attrs = make_object<VarianceAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;
  attrs->unbiased = unbiased;
  static const Op& op = Op::Get("variance");
  return Call(op, {data, mean}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make._variance").set_body_typed(MakeVariance);

RELAY_REGISTER_OP("variance")
    .describe(R"code(Computes the variance of array elements over given axes.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<VarianceAttrs>()
    .set_support_level(4)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("mean", "Tensor", "The mean tensor.")
    .add_type_rel("Variance", VarianceRel)
    .set_attr<FTVMCompute>("FTVMCompute", VarianceCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<VarianceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

}  // namespace relay
}  // namespace tvm
