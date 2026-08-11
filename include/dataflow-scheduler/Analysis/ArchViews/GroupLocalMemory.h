//===-- GroupLocalMemory.h --------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
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
//
//===----------------------------------------------------------------------===//
//
// GroupLocalMemory
//
// Captures the local memory resource present directly inside of a
// ktdf_arch.group across the different groups.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_GROUPLOCALMEMORY_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_GROUPLOCALMEMORY_H_

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Attributes.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"

namespace scheduler::arch_view {

/// Maps exec_unit kind -> local memory kind for groups that contain both.
///
/// Constructed as a DeviceView child of a DeviceOp
class GroupLocalMemory : public mlir::ktdf_arch::DeviceView {
 public:
  explicit GroupLocalMemory(const mlir::ktdf_arch::Device& device);

  /// Returns the kind attribute of the local memory co-located with the
  /// exec_unit identified by @p exec_unit_kind, or nullptr if no such mapping
  /// exists.
  [[nodiscard]] mlir::Attribute getLocalMemoryKind(
      mlir::Attribute exec_unit_kind) const;

  /// Returns the local memory kind for the single exec_unit kind declared on
  /// @p stage, or nullptr if the stage has an ambiguous / missing exec_unit
  /// kind or if no local memory is mapped to it.
  ///
  /// Emits an error on @p stage on failure so the caller can propagate it
  /// without reconstructing the message.
  [[nodiscard]] mlir::Attribute getLocalMemoryKindForStage(
      mlir::ktdf::StageOp stage) const;

 private:
  /// exec_unit kind -> local memory kind.
  llvm::DenseMap<mlir::Attribute, mlir::Attribute> exec_to_mem_kind_;

  void initialize();
};

}  // namespace scheduler::arch_view

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_GROUPLOCALMEMORY_H_
