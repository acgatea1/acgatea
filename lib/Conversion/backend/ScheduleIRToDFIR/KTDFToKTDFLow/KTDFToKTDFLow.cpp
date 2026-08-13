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
//
/// Phase 1: Operand-Based KTDF-to-DFIR Lowering
///
/// Converts KTDF pipelines/stages with applicable_units attributes to
/// operand-based ktdf_lowering IR using dataflow.get_unit, uniform maps,
/// and uniform queries.
///
//
//===----------------------------------------------------------------------===//

#include <map>

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Analysis/WriteSetScan.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ComponentClassifier.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/PipelineExecutionTransform.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ScratchpadConflicts.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/SignalInsertion.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UniformInfra.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UnitMaterializer.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/GlobalStageDAG.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "ktdf-to-ktdflowering"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTDFTOKTDFLOWERINGPASS
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"
}  // namespace scheduler

namespace {

// ---------------------------------------------------------------------------
// Check whether there is a genuine cross-iteration scratchpad dependency
// between load_stage and store_stage.
//
// The load stage reads from a shared scratchpad (memref source) into FIFOs or
// other local buffers.  The store stage writes the computed result back into
// that same scratchpad on the next iteration, creating a loop-carried RAW.
//
// Algorithm:
//   1. Collect every ktdf.data_transfer inside load_stage whose source
//      is a memref (i.e. not a FIFO).
//   2. For each such transfer, ask scheduler::regionWritesTo whether
//      store_stage's region writes through the same source memref.
//   3. If any transfer matches, a back-edge is needed.  All matching
//      transfers are returned via transfers_with_dependency so the caller
//      can attach them to BackEdgeInfo for later use during signal insertion.
//
// Returns true iff at least one transfer with a dependency was found.
// ---------------------------------------------------------------------------
static bool stageHasBackedgeDependency(
    mlir::ktdf::StageOp load_stage, mlir::ktdf::StageOp store_stage,
    llvm::SmallVectorImpl<mlir::ktdf::DataTransferOp>&
        transfers_with_dependency) {
  // Collect data transfers in load_stage whose source is a memref (not a
  // FIFO) — these are the reads from the shared scratchpad.
  llvm::SmallVector<mlir::ktdf::DataTransferOp, 4> load_transfers;
  load_stage.walk([&](mlir::ktdf::DataTransferOp xfer) {
    if (!xfer.isSourceFifo()) load_transfers.push_back(xfer);
  });

  // For each load transfer, check whether store_stage writes to the same
  // scratchpad memref that load_stage reads from.
  mlir::Region& store_region = store_stage.getRegion();
  for (mlir::ktdf::DataTransferOp xfer : load_transfers) {
    if (scheduler::regionWritesTo(store_region, xfer.getSource()))
      transfers_with_dependency.push_back(xfer);
  }
  return !transfers_with_dependency.empty();
}

// ---------------------------------------------------------------------------
// Walk the def-use chains of each transfer in transfers_with_dependency and
// collect the distinct non-trivial scf.for loops whose induction variables
// appear as BlockArguments in those chains.
//
// Algorithm:
//   - Maintain a worklist of Values, seeded with the index operands of each
//     qualifying transfer (not the destination memref — only the address
//     computation operands can reveal which loop IVs govern the scratchpad).
//   - For each Value: if it is a BlockArgument that is the induction variable
//     of an scf.for, record that ForOp (once) unless its static trip count
//     is <= 1.
//   - If it is an Operation result, push all operands of the defining op onto
//     the worklist to follow the chain transitively.
// ---------------------------------------------------------------------------
static llvm::SmallVector<mlir::scf::ForOp, 2> collectDependentLoops(
    llvm::ArrayRef<mlir::ktdf::DataTransferOp> transfers_with_dependency) {
  llvm::SmallVector<mlir::scf::ForOp, 2> result;
  llvm::DenseSet<mlir::Block*> seen_blocks;
  llvm::DenseSet<mlir::Value> visited;
  llvm::SmallVector<mlir::Value> worklist;

  for (mlir::ktdf::DataTransferOp xfer : transfers_with_dependency)
    for (mlir::Value idx : xfer.getSourceIndices()) worklist.push_back(idx);

  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) continue;

    if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      mlir::Block* owner = block_arg.getOwner();
      if (!seen_blocks.insert(owner).second) continue;
      auto for_op =
          mlir::dyn_cast_or_null<mlir::scf::ForOp>(owner->getParentOp());
      if (!for_op) continue;
      if (for_op.getInductionVar() != v) continue;
      // Discard trivial loops (trip count statically known to be <= 1).
      if (auto tc = for_op.getStaticTripCount(); tc && tc->ule(1)) continue;
      result.push_back(for_op);
    } else if (mlir::Operation* def = v.getDefiningOp()) {
      for (mlir::Value operand : def->getOperands())
        worklist.push_back(operand);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Augment global_dag with cross-iteration back-edges and populate
// back_edges_out for every pipeline whose linalg.generic compute stage has a
// genuine scratchpad RAW dependency between its load and store stages.
//
// For each linalg.generic, stageHasBackedgeDependency verifies that
// store_stage actually writes to a memref that load_stage loads from before
// recording the back-edge.  collectDependentLoops then walks the def-use
// chains of the qualifying transfers to find all non-trivial loops whose IVs
// index the scratchpad — these are stored in BackEdgeInfo as dependent_loops
// and drive the multi-loop conjunction guards emitted by insertSignals.
// If no dependent loops are found the candidate is skipped (no back-edge
// needed).
//
// The back-edge is:
//   store_stage (token-successor of compute_stage) → load_stage (predecessor)
// injected into global_dag so computeScratchpadConflicts fires on the edge,
// while back_edges_out lets insertSignals emit guarded signals instead of the
// unconditional post-producer signal used for normal edges.
// ---------------------------------------------------------------------------
static void addScfForPipelineBackEdges(
    mlir::func::FuncOp func, mlir::ktdf::StageDependencyDAG& global_dag,
    llvm::SmallVectorImpl<BackEdgeInfo>& back_edges_out) {
  func.walk([&](mlir::linalg::GenericOp generic) {
    // Step 1: find the immediately enclosing ktdf.stage (compute_stage).
    auto compute_stage = generic->getParentOfType<mlir::ktdf::StageOp>();
    assert(compute_stage && "linalg.generic must be inside a ktdf.stage");

    // Step 2: look up load_stage and store_stage directly from global_dag.
    // compute_stage is a leaf node in global_dag (no nested pipeline), so its
    // predecessor entry is the load leaf stage and its successor is the store
    // leaf stage.
    mlir::Operation* compute_op = compute_stage.getOperation();

    auto pred_it = global_dag.predecessors.find(compute_op);
    if (pred_it == global_dag.predecessors.end() || pred_it->second.empty())
      return;
    auto* load_stage_op = pred_it->second.front();

    auto succ_it = global_dag.successors.find(compute_op);
    if (succ_it == global_dag.successors.end() || succ_it->second.empty())
      return;
    auto* store_stage_op = succ_it->second.front();

    // Step 3: verify a genuine scratchpad dependency exists between the
    // candidate load/store stage pair before committing the back-edge.
    auto load_stage = mlir::cast<mlir::ktdf::StageOp>(load_stage_op);
    auto store_stage = mlir::cast<mlir::ktdf::StageOp>(store_stage_op);
    llvm::SmallVector<mlir::ktdf::DataTransferOp, 4> transfers_with_dependency;
    if (!stageHasBackedgeDependency(load_stage, store_stage,
                                    transfers_with_dependency))
      return;

    // Step 4: derive the set of non-trivial loops that govern the scratchpad
    // address by walking the def-use chains of the qualifying transfers.
    llvm::SmallVector<mlir::scf::ForOp, 2> dependent_loops =
        collectDependentLoops(transfers_with_dependency);
    if (dependent_loops.empty()) return;

    LDBG(1) << "  Adding scf.for pipeline back-edge: store_stage -> "
               "load_stage (cross-iteration scratchpad RAW)";
    global_dag.successors[store_stage_op].push_back(load_stage_op);
    global_dag.predecessors[load_stage_op].push_back(store_stage_op);

    back_edges_out.emplace_back(store_stage_op, load_stage_op,
                                std::move(dependent_loops),
                                std::move(transfers_with_dependency));
  });
}

struct KTDFToKTDFLoweringPass
    : public impl::KTDFToKTDFLoweringPassBase<KTDFToKTDFLoweringPass> {
  KTDFToKTDFLoweringPass()
      : scheduler_ctx_(SchedulerExtContext::dummyContext()) {}

  KTDFToKTDFLoweringPass(const SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void runOnOperation() override {
    LDBG(1) << "========= " PASS_NAME " =========";
    mlir::ModuleOp module_op = getOperation();

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module_op->emitError(
          "Unable to import the device specification. This could happen if the "
          "device spec file is empty or contains multiple devices");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<arch_view::ResourceKinds>(*device);

    llvm::SmallVector<mlir::func::FuncOp, 4> funcs;
    module_op.walk([&](mlir::func::FuncOp func) {
      funcs.push_back(func);
      return mlir::WalkResult::skip();
    });
    for (auto func : funcs) {
      LDBG(1) << "Running " << PASS_NAME << " on " << func.getName();

      // Pre-compute stages for reuse across multiple steps
      llvm::SmallVector<mlir::ktdf::StageOp, 8> stages;
      mlir::ktdf::collectStages(func, stages);

      if (stages.empty()) {
        LDBG(1) << "  No stages found - skipped";
        continue;
      }

      // Step 1: Classify components
      ComponentClassifier classifier(func);
      ComponentClassification components;
      if (mlir::failed(classifier.classify(stages, components))) {
        return signalPassFailure();
      }

      // Step 2: Extract grid size
      int grid_size = 0;
      if (mlir::failed(extractGridSize(func, grid_size))) {
        return signalPassFailure();
      }

      // Step 3: Materialize units
      UnitSSAMap unit_ssa_map;
      mlir::OpBuilder builder(&func.getBody().front(),
                              func.getBody().front().begin());

      UnitMaterializer materializer(func);
      if (mlir::failed(materializer.materialize(components, grid_size,
                                                unit_ssa_map, builder))) {
        return signalPassFailure();
      }

      // Step 4: Create uniform maps and queries
      QueriedUnitsMap queried_units;
      UniformMapsStorage uniform_maps;

      UniformInfra uniform_infra(func);
      if (mlir::failed(uniform_infra.createMapsAndQueries(
              components, grid_size, unit_ssa_map, queried_units, uniform_maps,
              builder))) {
        return signalPassFailure();
      }

      // Step 5: Wire queried units from Step 4 to stages based on
      // applicable_units
      StageToUnitsMap stage_to_units;
      int wired_queries = 0;
      for (auto stage : stages) {
        auto applicable_units = stage.getApplicableUnitsAttr();
        assert(applicable_units && "Stage should have applicable units");

        for (auto component : applicable_units.getValue()) {
          // Check if stage is in a parallel region
          mlir::Operation* parallel_parent =
              mlir::ktdf::findParallelParent(stage);

          if (parallel_parent) {
            // Parallel stage: look in queried_units.parallel for all corelets
            auto parallel_op =
                mlir::dyn_cast<mlir::ktdf::ParallelOp>(parallel_parent);
            int num_corelets = parallel_op.getNumInstances();
            for (int corelet = 0; corelet < num_corelets; ++corelet) {
              auto parallel_key = std::make_pair(
                  std::make_pair(parallel_parent, component), corelet);
              auto query_it = queried_units.parallel.find(parallel_key);
              if (query_it != queried_units.parallel.end()) {
                stage_to_units.mapping[stage.getOperation()].push_back(
                    query_it->second);
                wired_queries++;
              }
            }
          } else {
            // Non-parallel stage: first try queried_units.non_parallel
            auto query_it = queried_units.non_parallel.find(component);
            if (query_it != queried_units.non_parallel.end()) {
              stage_to_units.mapping[stage.getOperation()].push_back(
                  query_it->second);
              wired_queries++;
            } else {
              // If not found in non_parallel, this component might be in a
              // parallel region Find any parallel region that has this
              // component and use those units
              for (auto& [parallel_op, parallel_comps] :
                   components.parallel_components_map) {
                if (parallel_comps.contains(component)) {
                  auto parallel_parent_op =
                      mlir::dyn_cast<mlir::ktdf::ParallelOp>(parallel_op);
                  int num_corelets = parallel_parent_op.getNumInstances();
                  for (int corelet = 0; corelet < num_corelets; ++corelet) {
                    auto parallel_key = std::make_pair(
                        std::make_pair(parallel_op, component), corelet);
                    auto parallel_query_it =
                        queried_units.parallel.find(parallel_key);
                    if (parallel_query_it != queried_units.parallel.end()) {
                      stage_to_units.mapping[stage.getOperation()].push_back(
                          parallel_query_it->second);
                      wired_queries++;
                    }
                  }
                  break;  // Found the component in a parallel region, stop
                          // searching
                }
              }
            }
          }
        }
      }

      if (wired_queries == 0) {
        func.emitError("no queries wired to stages");
        return signalPassFailure();
      }

      // Step 6: Build the global flat stage DAG once, spanning all nesting
      // levels. Nodes are leaf StageOps only; used for conflict detection (Step
      // 7) and signal insertion (Step 8).
      mlir::ktdf::StageDependencyDAG global_dag;
      if (mlir::failed(mlir::ktdf::buildGlobalStageDAG(func, global_dag))) {
        func.emitError("failed to build global stage DAG");
        return signalPassFailure();
      }

      // Step 6b: Augment global_dag with cross-iteration back-edges for every
      // pipeline that has a genuine scratchpad RAW dependency between its load
      // and store stages.  Also collects BackEdgeInfo so insertSignals can emit
      // loop-IV-guarded signals for these edges.
      llvm::SmallVector<BackEdgeInfo> back_edges;
      addScfForPipelineBackEdges(func, global_dag, back_edges);

      // Step 7: Compute scratchpad conflicts across all pipelines using the
      // global leaf-stage DAG.
      std::map<std::pair<mlir::Operation*, mlir::Operation*>,
               llvm::SmallVector<scheduler::ResourceType, 2>>
          conflicts;
      if (mlir::failed(computeScratchpadConflicts(stage_to_units, global_dag,
                                                  resource_kinds, conflicts))) {
        func.emitError("failed to compute scratchpad conflicts");
        return signalPassFailure();
      }
      LDBG(1) << "Number of scratchpad conflicts found: " << conflicts.size();

      // Step 8: Insert signal operations for all conflicting global DAG edges,
      // before any pipeline transformation mutates the IR.  Back-edge signals
      // are wrapped in scf.if guards (iv != lb / iv != ub-step).
      if (mlir::failed(insertSignals(func.getLoc(), stage_to_units, global_dag,
                                     conflicts, back_edges))) {
        return signalPassFailure();
      }

      // Steps 9-12: Process each pipeline independently (innermost
      // first).
      llvm::SmallVector<mlir::ktdf::PipelineOp, 8> pipelines;
      func.walk([&](mlir::ktdf::PipelineOp pipeline) {
        pipelines.push_back(pipeline);
      });

      if (!pipelines.empty()) {
        LDBG(1) << "  Processing " << pipelines.size() << " pipelines";

        // Process pipelines in reverse order (post-order walk for nested
        // pipelines)
        for (auto pipeline : llvm::reverse(pipelines)) {
          LDBG(1) << "  Processing pipeline at " << pipeline.getLoc() << "";

          // Collect stages that are direct children of this pipeline (not
          // nested)
          llvm::SmallVector<mlir::ktdf::StageOp, 8> pipeline_stages;
          for (auto& op : pipeline.getBodyRegion().front()) {
            if (auto stage = mlir::dyn_cast<mlir::ktdf::StageOp>(op)) {
              pipeline_stages.push_back(stage);
            }
          }

          if (pipeline_stages.empty()) {
            // Skip pipelines that have no direct stages (already transformed or
            // empty)
            LDBG(1) << "  Skipping pipeline with no direct stages";
            continue;
          }

          // Step 9: Analyze per-pipeline stage dependencies for topo-sort.
          mlir::ktdf::StageDependencyDAG dag;
          if (mlir::failed(
                  mlir::ktdf::analyzeStageDependencies(pipeline_stages, dag))) {
            return signalPassFailure();
          }

          // Step 10: Topologically sort stages
          llvm::SmallVector<mlir::ktdf::StageOp, 8> sorted_stages;
          if (mlir::failed(mlir::ktdf::topologicalSortStages(
                  pipeline_stages, dag, sorted_stages))) {
            pipeline.emitError("topological sort of stages failed");
            return signalPassFailure();
          }

          // Step 11: Transform stages to execute_on (inside-out: stages first)
          if (mlir::failed(transformStagesToExecuteOn(pipeline, sorted_stages,
                                                      stage_to_units))) {
            return signalPassFailure();
          }

          // Step 12: Transform pipeline to execute_on (wrapping everything)
          if (mlir::failed(transformPipelineToExecuteOn(pipeline, sorted_stages,
                                                        stage_to_units))) {
            return signalPassFailure();
          }

          // erasing the stages at the very end.
          for (auto& stage : sorted_stages) {
            stage.erase();
          }
        }

        LDBG(1) << "  Phase 2 complete";
      }

      // Remove loop_type attributes from all scf.for loops now that
      // lowering is complete and the attribute is no longer needed.
      func.walk(
          [](mlir::scf::ForOp for_op) { for_op->removeAttr("loop_type"); });

      LDBG(1) << "Lowering complete for " << func.getName() << "";
    }
  }

 private:
  const SchedulerExtContext& schedulerExtContext() const {
    return scheduler_ctx_;
  }

  const SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createKTDFToKTDFLoweringPass() {
  return std::make_unique<KTDFToKTDFLoweringPass>();
}

std::unique_ptr<mlir::Pass> scheduler::createKTDFToKTDFLoweringPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<KTDFToKTDFLoweringPass>(scheduler_ctx);
}
