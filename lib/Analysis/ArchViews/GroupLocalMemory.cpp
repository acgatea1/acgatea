//===----------------------------------------------------------------------===//
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

#include "dataflow-scheduler/Analysis/ArchViews/GroupLocalMemory.h"

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

using namespace scheduler::arch_view;

GroupLocalMemory::GroupLocalMemory(const mlir::ktdf_arch::Device& device)
    : DeviceView(device) {
  initialize();
}

void GroupLocalMemory::initialize() {
  // Walk every group in the device definition.  For each group, scan its
  // immediate children for exec_unit ops and memory ops.  Build the map
  //   exec_unit.kind -> memory.kind
  // for every exec_unit that shares a group body with at least one memory.
  getDevice().getDefinition().walk([&](mlir::ktdf_arch::GroupOp group) {
    // Collect the kinds of exec_units and the first memory kind seen in this
    // group's immediate body (no recursion into nested groups).
    llvm::SmallVector<mlir::Attribute> exec_kinds;
    mlir::Attribute mem_kind;

    for (auto& op : group.getRegion().front().getOperations()) {
      if (auto exec_op =
              mlir::dyn_cast<mlir::ktdf_arch::ExecutionUnitOp>(&op)) {
        if (mlir::Attribute k = exec_op.getKind()) exec_kinds.push_back(k);
      } else if (auto mem_op = mlir::dyn_cast<mlir::ktdf_arch::MemoryOp>(&op)) {
        if (!mem_kind) mem_kind = mem_op.getKind();
      }
    }

    if (mem_kind) {
      for (mlir::Attribute ek : exec_kinds)
        exec_to_mem_kind_.insert({ek, mem_kind});
    }

    return mlir::WalkResult::advance();
  });
}

mlir::Attribute GroupLocalMemory::getLocalMemoryKind(
    mlir::Attribute exec_unit_kind) const {
  auto it = exec_to_mem_kind_.find(exec_unit_kind);
  if (it == exec_to_mem_kind_.end()) return nullptr;
  return it->second;
}
