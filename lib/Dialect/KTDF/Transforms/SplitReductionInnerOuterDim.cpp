//===-- SplitReductionInnerOuterDim.cpp -------------------------*- c++ -*-===//
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
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "split-reduction-inner-outer-dim"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_SPLITREDUCTIONINNEROUTERDIMPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Split Reduction Inner/Outer Dim pass"),
    llvm::cl::init(false));

namespace {

// ---------------------------------------------------------------------------
// A reduction dim expressed in loop-space together with its size in the input
// tensor type.
// ---------------------------------------------------------------------------
struct ReductionDimInfo {
  unsigned loop_dim;  // index into the iterator-types list
  int64_t size;       // static size of this dim in the input tensor
};

// ---------------------------------------------------------------------------
// Per-candidate analysis result.
// ---------------------------------------------------------------------------
struct CandidateInfo {
  linalg::GenericOp generic_op;
  // In-stick: exactly one dim — the rightmost reduction dim in input order.
  SmallVector<ReductionDimInfo> in_stick;
  // Across-stick: all remaining (leftward) reduction dims, same ordering.
  SmallVector<ReductionDimInfo> across_stick;
};

// ---------------------------------------------------------------------------
// Collect all reduction dims of `generic_op` ordered by their position in the
// first input tensor (leftmost first), together with their static sizes.
// Returns failure (and emits a debug note) if any reduction dim has a dynamic
// size or a non-trivial (non-AffineDimExpr) map result.
// ---------------------------------------------------------------------------
static LogicalResult collectReductionDims(
    linalg::GenericOp generic_op, SmallVectorImpl<ReductionDimInfo>& dims) {
  auto iter_types = generic_op.getIteratorTypesArray();
  auto input_type =
      cast<RankedTensorType>(generic_op.getInputs().front().getType());
  AffineMap input_map = generic_op.getIndexingMapsArray().front();
  ArrayRef<int64_t> shape = input_type.getShape();

  // Build a map: loop_dim -> input tensor dim index (only for reduction dims).
  // Iterate input dims left-to-right so the result list is in input order.
  for (int64_t d = 0; d < static_cast<int64_t>(shape.size()); ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(input_map.getResult(d));
    if (!dim_expr) continue;
    unsigned loop_dim = dim_expr.getPosition();
    if (iter_types[loop_dim] != utils::IteratorType::reduction) continue;

    int64_t sz = shape[d];
    if (sz == ShapedType::kDynamic) {
      LDBG(1) << PASS_NAME ": dynamic reduction dim size — skipping";
      return failure();
    }
    dims.push_back({loop_dim, sz});
  }
  return success();
}

// ---------------------------------------------------------------------------
// Return true when `generic_op` qualifies for the split:
//   1. It has at least two reduction iterator types (one in-stick + at least
//      one across-stick).
//   2. The rightmost non-1 input dimension maps to a reduction loop dim
//      (i.e. the innermost dim is the in-stick dimension).
// ---------------------------------------------------------------------------
static bool isEligible(linalg::GenericOp generic_op) {
  auto iter_types = generic_op.getIteratorTypesArray();

  // Condition 1: count reduction dims.
  int num_reductions = 0;
  for (auto it : iter_types)
    if (it == utils::IteratorType::reduction) ++num_reductions;
  if (num_reductions < 2) return false;

  // Condition 2: find the rightmost input dimension that is not size 1, and
  // check that its corresponding loop dim (via the input's indexing map) is a
  // reduction.
  auto input_type =
      dyn_cast<RankedTensorType>(generic_op.getInputs().front().getType());
  if (!input_type) return false;

  AffineMap input_map = generic_op.getIndexingMapsArray().front();
  ArrayRef<int64_t> shape = input_type.getShape();

  // Walk input dims from rightmost to leftmost.
  for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
    if (shape[d] == 1) continue;  // skip size-1 dims

    // Find the loop dim that this input dim maps to.
    // input_map.getResult(d) must be a pure dim expression (AffineDimExpr).
    AffineExpr expr = input_map.getResult(d);
    auto dim_expr = dyn_cast<AffineDimExpr>(expr);
    if (!dim_expr) return false;

    unsigned loop_dim = dim_expr.getPosition();
    return iter_types[loop_dim] == utils::IteratorType::reduction;
  }

  return false;
}

// ---------------------------------------------------------------------------
// Partition the reduction dims of `generic_op` into:
//   in-stick   — exactly the rightmost reduction dim (in input order).
//   across-stick — all remaining (leftward) reduction dims.
// ---------------------------------------------------------------------------
static FailureOr<CandidateInfo> partitionReductionDims(
    linalg::GenericOp generic_op, int64_t vector_length) {
  CandidateInfo info;
  info.generic_op = generic_op;

  SmallVector<ReductionDimInfo> all_dims;
  if (failed(collectReductionDims(generic_op, all_dims))) return failure();

  // The rightmost reduction dim (last in all_dims, ordered by input position)
  // is the single in-stick dim.
  assert(!all_dims.empty());
  ReductionDimInfo& in_stick_dim = all_dims.back();
  assert(in_stick_dim.size == vector_length &&
         "in-stick (innermost) reduction dim size must equal stick size");

  // all_dims[0 .. N-2] are across-stick; all_dims[N-1] is in-stick.
  for (int i = 0; i < static_cast<int>(all_dims.size()) - 1; ++i)
    info.across_stick.push_back(all_dims[i]);
  info.in_stick.push_back(in_stick_dim);

  LDBG(1) << PASS_NAME ": generic at " << generic_op.getLoc()
          << " vector_length=" << vector_length
          << " in-stick dims (loop_dim:size):";
  for (auto& d : info.in_stick)
    LDBG(1) << "  [" << d.loop_dim << "]=" << d.size;
  LDBG(1) << " across-stick dims (loop_dim:size):";
  for (auto& d : info.across_stick)
    LDBG(1) << "  [" << d.loop_dim << "]=" << d.size;

  return info;
}

// ---------------------------------------------------------------------------
// Clone the body region of `src` into a freshly-created (but not yet emitted)
// GenericOp `dst`.  The cloneInto helper prepends an empty placeholder block
// that must be removed after the copy.
// ---------------------------------------------------------------------------
static void cloneGenericBody(linalg::GenericOp src, linalg::GenericOp dst) {
  IRMapping mapping;
  src.getRegion().cloneInto(&dst.getRegion(), mapping);
  // cloneInto prepends an empty placeholder block; drop it.
  Block& placeholder = dst.getRegion().front();
  if (&placeholder != &dst.getRegion().back()) placeholder.erase();
}

// ---------------------------------------------------------------------------
// Split a reduction linalg.generic into two:
//
//   Generic 1 (across-stick reduction): reduces the across-stick dims.
//     Input:  original input tensor
//     Output: intermediate tensor whose dims = original parallel dims +
//             in-stick dims (across-stick dims are reduced away).
//     Iterator types: across-stick dims → reduction, in-stick dims → parallel,
//                     original parallel dims unchanged.
//
//   Generic 2 (in-stick reduction): reduces the in-stick dims.
//     Input:  intermediate tensor (result of Generic 1)
//     Output: same type as the original output
//     Iterator types: in-stick dims → reduction, parallel dims → parallel.
//
// The original generic is replaced: all uses of its result are replaced by
// the result of Generic 2, then it is erased.
// ---------------------------------------------------------------------------
static void splitDim(CandidateInfo& info) {
  linalg::GenericOp generic_op = info.generic_op;
  LDBG(1) << PASS_NAME ": splitDim on generic at " << generic_op.getLoc();

  MLIRContext* ctx = generic_op.getContext();
  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  // ── Collect structural info ──────────────────────────────────────────────
  SmallVector<utils::IteratorType> orig_iter_types =
      generic_op.getIteratorTypesArray();
  unsigned N = static_cast<unsigned>(orig_iter_types.size());

  AffineMap input_map = generic_op.getIndexingMapsArray().front();
  AffineMap output_map = generic_op.getIndexingMapsArray().back();

  auto input_type =
      cast<RankedTensorType>(generic_op.getInputs().front().getType());
  auto output_type =
      cast<RankedTensorType>(generic_op.getOutputs().front().getType());
  ArrayRef<int64_t> input_shape = input_type.getShape();
  Type elem_type = output_type.getElementType();

  // Build sets of in-stick and across-stick loop dim indices for quick lookup.
  llvm::SmallDenseSet<unsigned> in_stick_loop_dims;
  llvm::SmallDenseSet<unsigned> across_stick_loop_dims;
  for (auto& d : info.in_stick) in_stick_loop_dims.insert(d.loop_dim);
  for (auto& d : info.across_stick) across_stick_loop_dims.insert(d.loop_dim);

  // ── Build intermediate tensor shape ─────────────────────────────────────
  // The intermediate tensor has one dim per input tensor dim whose
  // corresponding loop dim is NOT an across-stick reduction (i.e. parallel or
  // in-stick).  We walk the input map results left-to-right to preserve order.
  SmallVector<int64_t> inter_shape;
  for (int64_t d = 0; d < static_cast<int64_t>(input_shape.size()); ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(input_map.getResult(d));
    if (!dim_expr) continue;  // non-trivial expression — skip
    unsigned ld = dim_expr.getPosition();
    if (!across_stick_loop_dims.count(ld))
      inter_shape.push_back(input_shape[d]);
  }

  // ── Build Generic 1 indexing maps ───────────────────────────────────────
  // Generic 1 keeps all N loop dims of the original op.  Its output is the
  // intermediate tensor, which has one dim per loop dim that survives G1
  // (i.e. every dim that is NOT an across-stick reduction).
  //
  // Step 1: assign intermediate-tensor slots for every loop dim reachable
  // through the input map, walking input dims left-to-right so the slot
  // ordering matches inter_shape.
  unsigned inter_dim = 0;
  llvm::SmallDenseMap<unsigned, unsigned> loop_dim_to_inter_dim;
  for (int64_t d = 0; d < static_cast<int64_t>(input_shape.size()); ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(input_map.getResult(d));
    if (!dim_expr) continue;
    unsigned ld = dim_expr.getPosition();
    if (!across_stick_loop_dims.count(ld))
      loop_dim_to_inter_dim[ld] = inter_dim++;
  }
  // Step 2: also assign slots for parallel loop dims whose only result is in
  // the output map (they never appear as a result in the input map, so Step 1
  // never visits them).  Concrete example from the failing IR:
  //   iterator_types = [reduction, parallel, reduction, parallel]
  //   input:  (d0,d1,d2,d3) -> (d0,d1,d2)   tensor<2x1x64>
  //   output: (d0,d1,d2,d3) -> (d1,d3)       tensor<1x64>
  // d0 is across-stick, d2 is in-stick, d1 appears in both maps, and d3
  // appears only as output_map.getResult(1).  Step 1 walks input dims 0..2
  // and produces {d1->0, d2->1}; d3 is never encountered.
  // linalg.generic requires every parallel iterator to be anchored in at
  // least one indexing map result, so G1's output map must include d3 or the
  // verifier rejects the op with "non-invertible indexing maps".
  ArrayRef<int64_t> output_shape = output_type.getShape();
  for (unsigned r = 0; r < output_map.getNumResults(); ++r) {
    auto dim_expr = dyn_cast<AffineDimExpr>(output_map.getResult(r));
    if (!dim_expr) continue;
    unsigned ld = dim_expr.getPosition();
    if (!loop_dim_to_inter_dim.count(ld) && !across_stick_loop_dims.count(ld)) {
      loop_dim_to_inter_dim[ld] = inter_dim++;
      inter_shape.push_back(output_shape[r]);
    }
  }
  // Build the g1 output map: one result per intermediate-tensor dim, in order.
  SmallVector<AffineExpr> g1_out_exprs(inter_dim);
  for (auto& [ld, id] : loop_dim_to_inter_dim)
    g1_out_exprs[id] = getAffineDimExpr(ld, ctx);
  AffineMap g1_out_map = AffineMap::get(N, 0, g1_out_exprs, ctx);

  // ── Generic 1 iterator types ─────────────────────────────────────────────
  // across-stick → reduction, in-stick → parallel, others → same as original.
  SmallVector<utils::IteratorType> g1_iter_types(orig_iter_types);
  for (unsigned i = 0; i < N; ++i) {
    if (in_stick_loop_dims.count(i))
      g1_iter_types[i] = utils::IteratorType::parallel;
    else if (across_stick_loop_dims.count(i))
      g1_iter_types[i] = utils::IteratorType::reduction;
    // parallel dims keep their original type
  }

  // ── Build Generic 2 indexing maps ────────────────────────────────────────
  // Generic 2 loops over M dims: original parallel dims + in-stick dims.
  // We build a renaming: original loop dim → new G2 loop dim index,
  // skipping across-stick dims.
  llvm::SmallDenseMap<unsigned, unsigned> loop_dim_to_g2_dim;
  unsigned g2_dim = 0;
  for (unsigned i = 0; i < N; ++i) {
    if (!across_stick_loop_dims.count(i)) loop_dim_to_g2_dim[i] = g2_dim++;
  }
  unsigned M = g2_dim;

  // G2 input map: maps the G2 loop dim for each intermediate dim (identity).
  // The intermediate dims are ordered the same way as loop_dim_to_inter_dim.
  SmallVector<AffineExpr> g2_in_exprs(inter_dim);
  for (auto& [ld, id] : loop_dim_to_inter_dim) {
    unsigned g2d = loop_dim_to_g2_dim[ld];
    g2_in_exprs[id] = getAffineDimExpr(g2d, ctx);
  }
  AffineMap g2_in_map = AffineMap::get(M, 0, g2_in_exprs, ctx);

  // G2 output map: remap the original output map using the G2 loop dim indices.
  SmallVector<AffineExpr> g2_out_exprs;
  for (unsigned r = 0; r < output_map.getNumResults(); ++r) {
    auto dim_expr = dyn_cast<AffineDimExpr>(output_map.getResult(r));
    assert(dim_expr && "output map result is not a pure dim expression");
    unsigned ld = dim_expr.getPosition();
    assert(loop_dim_to_g2_dim.count(ld) &&
           "output dim maps to an across-stick loop dim — unexpected");
    g2_out_exprs.push_back(getAffineDimExpr(loop_dim_to_g2_dim[ld], ctx));
  }
  AffineMap g2_out_map = AffineMap::get(M, 0, g2_out_exprs, ctx);

  // ── Generic 2 iterator types ─────────────────────────────────────────────
  SmallVector<utils::IteratorType> g2_iter_types(M);
  for (unsigned i = 0; i < N; ++i) {
    if (across_stick_loop_dims.count(i)) continue;
    unsigned g2d = loop_dim_to_g2_dim[i];
    g2_iter_types[g2d] = in_stick_loop_dims.count(i)
                             ? utils::IteratorType::reduction
                             : utils::IteratorType::parallel;
  }

  // ── Emit intermediate tensor initialiser ─────────────────────────────────
  auto inter_tensor_type = RankedTensorType::get(inter_shape, elem_type);
  auto inter_empty =
      tensor::EmptyOp::create(builder, loc, inter_shape, elem_type);

  // ── Emit Generic 1 ───────────────────────────────────────────────────────
  auto g1 = linalg::GenericOp::create(
      builder, loc,
      /*resultTensorTypes=*/TypeRange{inter_tensor_type},
      /*inputs=*/generic_op.getInputs(),
      /*outputs=*/ValueRange{inter_empty.getResult()},
      ArrayRef<AffineMap>{input_map, g1_out_map}, g1_iter_types);
  cloneGenericBody(generic_op, g1);

  // ── Emit output tensor initialiser for Generic 2 ─────────────────────────
  auto out_empty =
      tensor::EmptyOp::create(builder, loc, output_type.getShape(), elem_type);

  // ── Emit Generic 2 ───────────────────────────────────────────────────────
  auto g2 = linalg::GenericOp::create(
      builder, loc,
      /*resultTensorTypes=*/TypeRange{output_type},
      /*inputs=*/ValueRange{g1.getResult(0)},
      /*outputs=*/ValueRange{out_empty.getResult()},
      ArrayRef<AffineMap>{g2_in_map, g2_out_map}, g2_iter_types);
  cloneGenericBody(generic_op, g2);

  // ── Replace and erase original generic and its now-dead output init ──────
  generic_op.getResult(0).replaceAllUsesWith(g2.getResult(0));
  Value orig_out_init = generic_op.getOutputs().front();
  generic_op.erase();
  if (orig_out_init.use_empty())
    if (auto* def = orig_out_init.getDefiningOp()) def->erase();
}

struct SplitReductionInnerOuterDimPass
    : public mlir::ktdf::impl::SplitReductionInnerOuterDimPassBase<
          SplitReductionInnerOuterDimPass> {
  using SplitReductionInnerOuterDimPassBase<
      SplitReductionInnerOuterDimPass>::SplitReductionInnerOuterDimPassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    ModuleOp module = getOperation();

    // Obtain the device manager and derive vector_length from the SIMD feature.
    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module->emitError(PASS_NAME
                        ": unable to import the device specification");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<scheduler::arch_view::ResourceKinds>(
            *device);

    // Collect eligible linalg.generic ops: at least two reduction iterator
    // types, with one reduction dim mapping to the rightmost non-1 input dim.
    SmallVector<linalg::GenericOp> candidates;
    module.walk([&](linalg::GenericOp generic_op) {
      if (isEligible(generic_op)) candidates.push_back(generic_op);
      return WalkResult::advance();
    });

    if (candidates.empty()) {
      LDBG(1) << PASS_NAME ": no eligible linalg.generic found — skipping";
      return;
    }

    auto simd_feature =
        resource_kinds.getFeature<mlir::ktdf_arch::feature::SIMD>(
            resource_kinds.getComputeKind());

    for (linalg::GenericOp generic_op : candidates) {
      // Derive the element type from the first output (the accumulator).
      auto output_type =
          dyn_cast<ShapedType>(generic_op.getOutputs().front().getType());
      assert(output_type);
      Type elem_type = output_type.getElementType();

      const int64_t vector_length =
          std::max(simd_feature.getLanes(elem_type), int64_t(1));

      FailureOr<CandidateInfo> info =
          partitionReductionDims(generic_op, vector_length);
      if (failed(info)) continue;
      splitDim(*info);
    }
  }
};

}  // namespace

auto mlir::ktdf::createSplitReductionInnerOuterDimPass()
    -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<SplitReductionInnerOuterDimPass>();
}
