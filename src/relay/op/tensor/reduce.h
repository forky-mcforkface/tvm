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

#ifndef TVM_RELAY_OP_TENSOR_REDUCE_H
#define TVM_RELAY_OP_TENSOR_REDUCE_H

namespace tvm {
namespace relay {
/*! \brief GetReduceAxes,
    get the new axis from indim and other arguments * \param indim Number of dimensions of input
            data.* \param axis The input axis
                vector.* \param exclude Whether 'axis' input given is the excluded
                    axis.* \return r_axes The new reduced axes of the output. */
inline std::vector<int64_t> GetReduceAxes(const uint32_t indim, const Array<Integer>& inaxis,
                                          bool exclude) {
  if (!inaxis.defined() || inaxis.empty()) {
    std::vector<int64_t> r_axes(indim);
    std::iota(r_axes.begin(), r_axes.end(), 0);
    return r_axes;
  }

  std::vector<int64_t> in_axes;
  for (auto i : inaxis) {
    int64_t axis = i->value;
    if (axis < 0) {
      axis = axis + indim;
    }

    // Check out of bounds error
    ICHECK(axis >= 0) << "Axis out of bounds in reduce operator.";
    ICHECK(axis < indim) << "Axis out of bounds in reduce operator.";
    in_axes.push_back(axis);
  }

  ICHECK(in_axes[in_axes.size() - 1] < indim)
      << "Reduction axis " << in_axes[in_axes.size() - 1] << " exceeds input dimensions " << indim;

  std::sort(in_axes.begin(), in_axes.end());

  if (!exclude) {
    return in_axes;
  }

  auto r_size = indim - in_axes.size();
  std::vector<int64_t> r_axes(r_size);
  for (uint32_t i = 0, j = 0, k = 0; i < indim; ++i) {
    if (j < in_axes.size() && in_axes[j] == i) {
      ++j;
      continue;
    }
    r_axes[k++] = i;
  }
  return r_axes;
}

// Get axis under exclude condition.
Array<Integer> GetExcludeAxes(size_t indim, const Array<Integer>& inaxis) {
  ICHECK(inaxis.defined()) << "Cannot set exclude when axis=None";
  std::vector<bool> axis_flag(indim, true);
  for (auto i : inaxis) {
    int64_t axis = i->value;
    if (axis < 0) {
      axis = axis + static_cast<int64_t>(indim);
    }
    // Check out of bounds error
    ICHECK_GE(axis, 0) << "Axis out of bounds in reduce operator.";
    ICHECK_LT(axis, static_cast<int64_t>(indim)) << "Axis out of bounds in reduce operator.";
    axis_flag[axis] = false;
  }

  Array<Integer> r_axes;

  for (size_t i = 0; i < axis_flag.size(); ++i) {
    if (axis_flag[i]) {
      r_axes.push_back(static_cast<int>(i));
    }
  }
  return r_axes;
}

// Return the modified layout for AlterOpLayout pass.
template <typename T>
InferCorrectLayoutOutput ReduceInferCorrectLayout(const Attrs& attrs,
                                                  const Array<Layout>& new_in_layouts,
                                                  const Array<Layout>& old_in_layouts,
                                                  const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<T>();
  ICHECK(attrs_ptr);
  ObjectPtr<T> params = make_object<T>(*attrs_ptr);

  // Get the reduce axes.
  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    ICHECK(old_in_t.as<TensorTypeNode>());
    old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
  }
  uint32_t indim = old_in_shapes[0].size();
  auto r_axes = GetReduceAxes(indim, params->axis, params->exclude);

  Layout inferred_in = Layout::Undef();
  Layout inferred_out = Layout::Undef();

  // Infer [in_layout, out_layout, new_r_axes] from old_in_layout or new_in_layout
  auto infer = [&](const Layout& layout) {
    // 1) Collect the original axes
    std::unordered_set<std::string> old_r_dims;
    for (auto r_axis : r_axes) {
      old_r_dims.emplace(old_in_layouts[0][r_axis].name());
    }

    // 2) Collect the new axes by walking new_layout.
    tvm::Array<tvm::Integer> new_r_axes;
    std::string inferred_in_string = "";
    std::string inferred_out_string = "";
    auto push_new_axis = [&](const std::string& layout_dim, int axis) {
      if ((old_r_dims.count(layout_dim) && !params->exclude) ||
          (!old_r_dims.count(layout_dim) && params->exclude)) {
        new_r_axes.push_back(tvm::Integer(axis));
        return true;
      }
      return false;
    };
    for (size_t axis_index = 0; axis_index < layout->axes.size(); ++axis_index) {
      const auto& layout_axis = LayoutAxis::Get(layout->axes[axis_index]);
      const std::string& layout_dim = layout_axis.name();
      if (layout_axis.IsPrimal()) {
        push_new_axis(layout_dim, axis_index);
        inferred_in_string += layout_dim;
        if (!old_r_dims.count(layout_dim) || params->keepdims) {
          inferred_out_string += layout_dim;
        }
      } else {
        // For example, if the original layout is NCHW, the new layout is NCHW8c, and the original
        // reduce axes is [1], the new reduce axes become [1, 4].
        auto primal_dim = layout_axis.ToPrimal().name();
        auto packed_dim = std::to_string(layout.FactorOf(layout_axis)) + layout_dim;
        inferred_in_string += packed_dim;
        if (push_new_axis(primal_dim, axis_index)) {
          if (params->exclude) {
            // The primal axis is not reduced, so keep the input packed dim.
            inferred_out_string += packed_dim;
          } else if (params->keepdims) {
            // If the primal axis is part of reduce axes in the original layout, the inner dim
            // becomes 1 after reduction.
            inferred_out_string += "1" + layout_dim;
          }
        } else {
          inferred_out_string += packed_dim;
        }
      }
    }

    // 3) Set the new axis and layout.
    return std::make_tuple(Layout(inferred_in_string), Layout(inferred_out_string), new_r_axes);
  };

  std::string new_layout_string;
  Array<Integer> new_r_axes;
  Array<Layout> new_input_layouts;

  auto check_num_input_layouts = [](Array<Layout> in_layouts) {
    // The second case is for variance op
    ICHECK(in_layouts.size() == 1 || in_layouts.size() == 2);
  };

  if (new_in_layouts.defined() && r_axes.size()) {
    // Adapt to new layout. The axis has to change. Record original reduce axes. Convert to the
    // modified layout axes.
    check_num_input_layouts(new_in_layouts);
    check_num_input_layouts(old_in_layouts);

    // Get inferred_in and inferred_out from new_in_layout.
    std::tie(inferred_in, inferred_out, new_r_axes) = infer(new_in_layouts[0]);
    params->axis = new_r_axes;
  } else if (old_in_layouts.defined()) {
    check_num_input_layouts(old_in_layouts);

    // If the new layout is undefined, get inferred_in and inferred_out from old_in_layout.
    if (old_in_layouts[0].defined()) {
      std::tie(inferred_in, inferred_out, std::ignore) = infer(old_in_layouts[0]);
    }
  }

  new_input_layouts.push_back(inferred_in);

  if (old_in_layouts.size() == 2) {
    new_input_layouts.push_back(inferred_in);
  }

  return InferCorrectLayoutOutput(new_input_layouts, {inferred_out}, Attrs(params));
}

template <typename F>
Array<te::Tensor> ReduceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type, F f) {
  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  ICHECK(param != nullptr);
  if (inputs[0]->shape.size() == 0) {
    return {topi::identity(inputs[0])};
  }
  auto axes = param->axis;
  if (param->exclude) {
    axes = GetExcludeAxes(inputs[0]->shape.size(), param->axis);
    if (axes.size() == 0) {
      return {topi::identity(inputs[0])};
    }
  }

  return {f(inputs[0], axes, param->keepdims, false)};
}

template <typename F>
Array<te::Tensor> ArgReduceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                   const Type& out_type, F f) {
  const ArgReduceAttrs* param = attrs.as<ArgReduceAttrs>();
  ICHECK(param != nullptr);
  if (inputs[0]->shape.size() == 0) {
    return {topi::identity(inputs[0])};
  }
  auto axes = param->axis;
  if (param->exclude) {
    axes = GetExcludeAxes(inputs[0]->shape.size(), param->axis);
    if (axes.size() == 0) {
      return {topi::identity(inputs[0])};
    }
  }

  return {f(inputs[0], axes, param->keepdims, false, param->select_last_index)};
}

/*!
 * \brief ReduceShapeImpl get the outshape for the reduction operator
 * \param in_shape Shape of input data.
 * \param param Attrs details.
 * \param reporter The reporter to report solution to.
 * \return oshape Output shape inferred.
 * \tparam AttrsType The attribute type.
 */
template <typename AttrsType>
inline std::vector<IndexExpr> ReduceShapeImpl(const std::vector<IndexExpr>& in_shape,
                                              const AttrsType* param,
                                              const TypeReporter& reporter) {
  uint32_t indim = in_shape.size();
  auto r_axes = GetReduceAxes(indim, param->axis, param->exclude);
  if (!r_axes.size()) {
    return in_shape;
  }

  auto max_shape = tir::make_const(DataType::Int(64), 1);
  bool is_dynamic_input = false;
  for (int64_t axis : r_axes) {
    if (in_shape[axis].as<IntImmNode>()) {
      max_shape *= in_shape[axis];
    } else {
      is_dynamic_input = true;
      break;
    }
  }

  if (is_dynamic_input) {
    ICHECK(reporter->Assert(
        max_shape < tir::make_const(DataType::Int(64), std::numeric_limits<int32_t>::max())))
        << "The maximum possible index of reduced shape cannot be more than int32 max.";
  }

  if (param->keepdims) {
    std::vector<IndexExpr> oshape(in_shape);
    for (unsigned i = 0, j = 0; i < indim; ++i) {
      if (j >= r_axes.size() || !(r_axes[j] == i)) {
        continue;
      }
      oshape[i] = 1;
      ++j;
    }
    return oshape;
  } else {
    auto osize = indim - r_axes.size();
    std::vector<IndexExpr> oshape(osize);
    for (unsigned i = 0, j = 0, k = 0; i < indim; ++i) {
      if (j < r_axes.size() && (r_axes[j] == i)) {
        ++j;
        continue;
      }
      oshape[k++] = in_shape[i];
    }
    return oshape;
  }
}

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_TENSOR_REDUCE_H
