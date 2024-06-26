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
#ifndef TVM_META_SCHEDULE_SPACE_GENERATOR_H_
#define TVM_META_SCHEDULE_SPACE_GENERATOR_H_

#include <tvm/ir/module.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace meta_schedule {

// Forward declaration
class TuneContext;

/*!
 * \brief The abstract class for design space generation.
 * \note The relationship between SpaceGenerator and other classes are as follows:
      ┌──────────────────────────────────────────────────────────────┐
   ┌──┴───────────────────────────────────────────────────────────┐  │
┌──┴────────────────── Tune Context ───────────────────────────┐  │  │
│                ┌─────────────────────┐                       │  │  │
│                │                     │   Generate            │  │  │
│                │   Space Generator   ├──────────────┐        │  │  │
│                │                     │              │        │  │  │
│                └─────────────────────┘              ▼        │  │  │
│                                                Design Space  │  │  │
│                ┌─────────────────────┐              │        │  │  │
│      Generate  │                     │   Pretuning  │        │  │  │
│    ┌───────────┤   Search Strategy   │◄─────────────┘        │  │  │
│    │           │                     │                       │  ├──┘
│    │           └─────────────────────┘                       ├──┘
└────┼─────────────────────────────────────────────────────────┘
     │
     │
┌────┼──────────────── Managed By Task Scheduler ─────────────────────┐
│    │                                 ┌───────────┐                  │
│    │                      Send to    │           │  Send to         │
│    ▼                  ┌─────────────►│  Builder  ├──────────┐       │
│ Measure Candidate     │   Builder    │           │  Runner  │       │
│    │                  │              └───────────┘          │       │
│    │     ┌────────────┴────────┐                            │       │
│    │     │                     │     ┌───────────┐          │       │
│    └────►│   Task Scheduler    │     │           │          │       │
│          │                     │     │  Runner   │◄─────────┘       │
│          └─────────────────────┘     │           │                  │
│                   ▲                  └─────┬─────┘                  │
│                   │                        │                        │
│                   └───  Runner Future ◄────┘                        │
└─────────────────────────────────────────────────────────────────────┘
*/
class SpaceGeneratorNode : public runtime::Object {
 public:
  /*! \brief Default destructor */
  virtual ~SpaceGeneratorNode() = default;

  /*!
   * \brief Initialize the design space generator with tuning context.
   * \param context The tuning context for initialization.
   * \note This method is supposed to be called only once before every other method.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Generate design spaces given a module.
   * \param mod The module used for design space generation.
   * \return The generated design spaces, i.e., schedules.
   */
  virtual Array<tir::Schedule> GenerateDesignSpace(const IRModule& mod) = 0;

  static constexpr const char* _type_key = "meta_schedule.SpaceGenerator";
  TVM_DECLARE_BASE_OBJECT_INFO(SpaceGeneratorNode, Object);
};

/*! \brief The design space generator with customized methods on the python-side. */
class PySpaceGeneratorNode : public SpaceGeneratorNode {
 public:
  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*!
   * \brief The function type of `GenerateDesignSpace` method.
   * \param mod The module used for design space generation.
   * \return The generated design spaces, i.e., schedules.
   */
  using FGenerateDesignSpace = runtime::TypedPackedFunc<Array<tir::Schedule>(const IRModule&)>;

  /*! \brief The packed function to the `InitializeWithTuneContext` function. */
  FInitializeWithTuneContext f_initialize_with_tune_context;
  /*! \brief The packed function to the `GenerateDesignSpace` function. */
  FGenerateDesignSpace f_generate_design_space;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_initialize_with_tune_context` is not visited
    // `f_generate_design_space` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& context) final;
  Array<tir::Schedule> GenerateDesignSpace(const IRModule& mod) final;

  static constexpr const char* _type_key = "meta_schedule.PySpaceGenerator";
  TVM_DECLARE_FINAL_OBJECT_INFO(PySpaceGeneratorNode, SpaceGeneratorNode);
};

/*!
 * \brief Managed reference to SpaceGeneratorNode.
 * \sa SpaceGeneratorNode
 */
class SpaceGenerator : public runtime::ObjectRef {
 protected:
  SpaceGenerator() = default;

 public:
  /*!
   * \brief Create a design space generator with customized methods on the python-side.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_generate_design_space The packed function of `GenerateDesignSpace`.
   * \return The design space generator created.
   */
  TVM_DLL static SpaceGenerator PySpaceGenerator(
      PySpaceGeneratorNode::FInitializeWithTuneContext f_initialize_with_tune_context,
      PySpaceGeneratorNode::FGenerateDesignSpace f_generate_design_space);
  /*!
   * \brief Create a design space generator with customized schedule function.
   * \param schedule_fn The schedule function, which can have the following signatures:
   * 1) void(Schedule)
   * 2) Schedule(Schedule)
   * 3) Array<Schedule>(Schedule)
   */
  TVM_DLL static SpaceGenerator ScheduleFn(PackedFunc schedule_fn);
  /*!
   * \brief Create a design space generator that is union of multiple design space generators.
   * \param space_generators An array of design space generators to be unioned.
   * \return The design space generator created.
   */
  TVM_DLL static SpaceGenerator SpaceGeneratorUnion(Array<SpaceGenerator, void> space_generators);
  /*!
   * \brief Create a design space generator that generates design spaces by applying schedule rules
   *  to blocks in post-DFS order.
   * \return The design space generator created.
   */
  TVM_DLL static SpaceGenerator PostOrderApply(runtime::PackedFunc f_block_filter = nullptr);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(SpaceGenerator, ObjectRef, SpaceGeneratorNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_SPACE_GENERATOR_H_
