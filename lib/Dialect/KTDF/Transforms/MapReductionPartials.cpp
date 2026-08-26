//===-- MapReductionPartials.cpp --------------------------------*- c++ -*-===//
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
// MapReductionPartials: lower reduction linalg.generic ops to buffer-semantics
// form backed by pre-allocated memrefs.  Two kinds of generics are handled:
//
// ── Across-stick generic (loop-exposed by ReductionLoopExposure) ─────────────
//
// After ReductionLoopExposure the compute stage looks like:
//
//   %empty = tensor.empty() : tensor<CxDxf16>
//   %result = scf.for %r = 0 to R iter_args(%carry = %empty)
//                 {loop_type = reduction_loop}
//     %slice = ktdf.read_from_fifo %fifo -> tensor<CxDxf16>
//     %new_carry = linalg.generic ins(%slice) outs(%carry)
//     scf.if (%r == R-1): ktdf.write_to_fifo %new_carry, %fifo_out
//     scf.yield %new_carry
//
// This pass rewrites that to:
//
//   %alloc = memref.alloc() : memref<CxDxf16, compute_kind>
//   linalg.fill(%zero, %alloc)
//   scf.for %r = 0 to R {
//     %slice = ktdf.read_from_fifo %fifo -> memref<CxDxf16>
//     linalg.generic ins(%slice) outs(%alloc)
//     scf.if (%r == R-1): ktdf.write_to_fifo %alloc, %fifo_out
//   }
//
// When the iter_arg initializer is a conditional (scf.if), the pass recurses
// into each branch and handles the yielded value:
//
//   %alloc = memref.alloc() : memref<CxDxf16, compute_kind>
//   scf.if %cond {                     // no result; was -> (tensor<...>)
//     linalg.fill(%zero, %alloc)       // was: tensor.empty + yield
//   } else {
//     %r = ktdf.read_from_fifo ...     // memref form of read_from_fifo
//     memref.copy %r, %alloc
//   }
//   scf.for %r = 0 to R { ... }       // iter_arg stripped
//
// The iter_arg / loop result / scf.yield operand are stripped, and any
// write_to_fifo that consumed the loop result is patched to use %alloc.
//
// ── In-stick generic (produced by SplitReductionInnerOuterDim) ───────────────
//
// After the across-stick rewrite above, %alloc_G1 (memref<D0x...xDNxf16, ms>)
// has been RAUW'd in place of the loop result, so the in-stick generic sees:
//
//   %empty = tensor.empty() : tensor<D0x...xDN-1xf16>
//   %result = linalg.generic ins(%alloc_G1: memref<D0x...xDNxf16, ms>)
//                            outs(%empty: tensor<D0x...xDN-1xf16>)
//   ktdf.write_to_fifo %result, %fifo_out
//
// A rank-reducing subview of %alloc_G1 (size=1 on reduction dims, original
// size on parallel dims) is used as the outs buffer; ins is the full %alloc_G1.
// After the reduction, the full %alloc_G1 — not the subview — is written to
// the FIFO so the downstream stage receives all partial sums.
// The FIFO slot type and the paired memref.alloc in the ktdf.private block are
// widened to match %alloc_G1's shape.
//
//   %sv = memref.subview %alloc_G1[0,...][D0,...,1,...][1,...]
//             : memref<D0x...xDNxf16, ms> to memref<D0x...xDN-1xf16, strided,
//             ms>
//   linalg.generic ins(%alloc_G1) outs(%sv)  {original maps & iter_types}
//   ktdf.write_to_fifo %alloc_G1, %fifo_out  // full buffer
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Analysis/ArchViews/GroupLocalMemory.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "map-reduction-partials"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Map Reduction Partials pass"),
    llvm::cl::init(false));

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_MAPREDUCTIONPARTIALSPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

namespace {

// ---------------------------------------------------------------------------
// Return true if `generic_op` has at least one reduction iterator.
// ---------------------------------------------------------------------------
static bool hasReductionIterator(linalg::GenericOp generic_op) {
  for (auto it : generic_op.getIteratorTypesArray())
    if (it == utils::IteratorType::reduction) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Transform the initializer `init_val` of a loop-carried accumulator in-place,
// emitting linalg.fill (or memref.copy) into `alloc_val` wherever the original
// tensor value was produced.  `alloc_val` is a memref.alloc already emitted at
// the top of the stage.
//
// Three leaf cases are handled; scf.if recurses into both branches:
//
//   tensor.empty()
//     → linalg.fill(%zero, %alloc) inserted just before the tensor.empty,
//       then the tensor.empty is erased.
//
//   ktdf.read_from_fifo ... -> tensor<...>
//     → emit memref-typed read_from_fifo + memref.copy into %alloc just
//       before the original op, then erase it.
//
//   scf.if %cond -> (tensor<...>) { yield A } else { yield B }
//     → recurse on A (then-branch yield operand 0) and B (else-branch yield
//       operand 0), drop the yield operands so both yields become result-less,
//       then rebuild the scf.if with no result types.
// ---------------------------------------------------------------------------
static LogicalResult lowerIterArgInitializer(Value init_val, Value alloc_val) {
  Operation* defining_op = init_val.getDefiningOp();
  assert(defining_op &&
         "lowerIterArgInitializer: init_val must be an op result");

  Type elem_type = cast<MemRefType>(alloc_val.getType()).getElementType();

  // ── Case: tensor.empty ────────────────────────────────────────────────────
  if (auto empty_op = dyn_cast<tensor::EmptyOp>(defining_op)) {
    OpBuilder builder(empty_op);
    Location loc = empty_op.getLoc();
    Value zero = arith::ConstantOp::create(
        builder, loc, builder.getFloatAttr(elem_type, 0.0));
    linalg::FillOp::create(builder, loc, ValueRange{zero},
                           ValueRange{alloc_val});
    empty_op->erase();
    return success();
  }

  // ── Case: ktdf.read_from_fifo returning a tensor ──────────────────────────
  if (auto read_op = dyn_cast<ktdf::ReadFromFifoOp>(defining_op)) {
    auto tensor_type = cast<RankedTensorType>(read_op.getResult().getType());
    auto memref_type =
        MemRefType::get(tensor_type.getShape(), tensor_type.getElementType());
    OpBuilder builder(read_op);
    Location loc = read_op.getLoc();
    Value new_read = ktdf::ReadFromFifoOp::create(builder, loc, memref_type,
                                                  read_op.getFifoSlot())
                         .getResult();
    memref::CopyOp::create(builder, loc, new_read, alloc_val);
    read_op->erase();
    return success();
  }

  // ── Case: scf.if returning a tensor — recurse into both branches ──────────
  if (auto if_op = dyn_cast<scf::IfOp>(defining_op)) {
    // Drop the yield operand *before* recursing so that when the recursive
    // call erases the defining op (e.g. tensor.empty) no uses remain.
    {
      auto then_yield =
          cast<scf::YieldOp>(if_op.getThenRegion().front().getTerminator());
      Value then_init = then_yield.getOperand(0);
      then_yield->setOperands({});
      if (failed(lowerIterArgInitializer(then_init, alloc_val)))
        return failure();
    }

    {
      auto else_yield =
          cast<scf::YieldOp>(if_op.getElseRegion().front().getTerminator());
      Value else_init = else_yield.getOperand(0);
      else_yield->setOperands({});
      if (failed(lowerIterArgInitializer(else_init, alloc_val)))
        return failure();
    }

    // Rebuild the scf.if without result types and transfer the rewritten
    // regions.
    OpBuilder builder(if_op);
    auto new_if =
        scf::IfOp::create(builder, if_op.getLoc(),
                          /*resultTypes=*/TypeRange{}, if_op.getCondition(),
                          /*withElseRegion=*/true);
    new_if.getThenRegion().takeBody(if_op.getThenRegion());
    new_if.getElseRegion().takeBody(if_op.getElseRegion());
    if_op.erase();
    return success();
  }

  return defining_op->emitError(
      "lowerIterArgInitializer: unrecognised initializer form — expected "
      "tensor.empty, ktdf.read_from_fifo, or scf.if");
}

// ---------------------------------------------------------------------------
// Walk up the iter_arg chain rooted at `acc_iter_arg` (outs[0] of the
// original linalg.generic, captured before erasure) and rebuild each
// enclosing scf.for without that iter_arg slot, replacing every use of
// the dropped iter_arg and its loop result with `alloc_val`.
//
// After the innermost loop is rebuilt the old loop result may itself be an
// iter_arg of an outer scf.for — we keep climbing until the chain exits a
// scf.for.
//
// The initializer at the top of the chain has already been handled by
// lowerIterArgInitializer before this function is called, so it is NOT
// erased here.
//
// Cannot remove an iter_arg in-place: getInitArgsMutable().erase() only drops
// the operand but leaves the region block argument and loop result intact,
// producing an inconsistent op that fails the verifier.  We clone instead.
// ---------------------------------------------------------------------------
static void removeUnusedIterArgChain(BlockArgument acc_iter_arg,
                                     Value alloc_val) {
  // `current` starts as the innermost iter_arg and advances to the init of
  // that iter_arg at each level.
  Value current = acc_iter_arg;
  while (auto iter_arg = dyn_cast<BlockArgument>(current)) {
    auto for_op = dyn_cast<scf::ForOp>(iter_arg.getOwner()->getParentOp());
    assert(for_op);

    // Block arg 0 of an scf.for is the induction variable; iter_args start
    // at index 1, so subtract 1 to get the iter_arg slot index.
    unsigned iter_idx = iter_arg.getArgNumber() - 1;

    // Capture the init before rebuilding — this is what we advance to next.
    Value iter_init = for_op.getInits()[iter_idx];

    // Replace all uses of the dropped iter_arg and its loop result.
    Value loop_result = for_op.getResult(iter_idx);
    loop_result.replaceAllUsesWith(alloc_val);
    iter_arg.replaceAllUsesWith(alloc_val);

    // Rebuild the loop without the iter_arg at `iter_idx`.
    OpBuilder builder(for_op);
    Location loc = for_op.getLoc();

    SmallVector<Value> new_inits;
    for (auto [i, init] : llvm::enumerate(for_op.getInits()))
      if (i != iter_idx) new_inits.push_back(init);

    auto new_for =
        scf::ForOp::create(builder, loc, for_op.getLowerBound(),
                           for_op.getUpperBound(), for_op.getStep(), new_inits);

    for (auto attr : for_op->getAttrs())
      if (attr.getName() != "operandSegmentSizes")
        new_for->setAttr(attr.getName(), attr.getValue());

    // Map old to new iter args, skipping over iter_idx.
    IRMapping body_map;
    body_map.map(for_op.getInductionVar(), new_for.getInductionVar());
    unsigned new_arg_idx = 0;
    for (auto [i, arg] : llvm::enumerate(for_op.getRegionIterArgs()))
      if (i != iter_idx)
        body_map.map(arg, new_for.getRegionIterArgs()[new_arg_idx++]);

    // Copy old scf.for yield operands skipping over iter_idx.
    auto* old_yield = for_op.getBody()->getTerminator();
    Operation* new_yield = new_for.getBody()->getTerminator();
    OpBuilder body_builder(new_yield);
    for (auto& op : for_op.getBody()->without_terminator())
      body_builder.clone(op, body_map);

    SmallVector<Value> new_yield_operands;
    for (auto [i, operand] : llvm::enumerate(old_yield->getOperands()))
      if (i != iter_idx)
        new_yield_operands.push_back(body_map.lookupOrDefault(operand));
    new_yield->setOperands(new_yield_operands);

    unsigned new_res_idx = 0;
    for (auto [i, res] : llvm::enumerate(for_op.getResults()))
      if (i != iter_idx)
        res.replaceAllUsesWith(new_for.getResult(new_res_idx++));

    // Advance to next parent loop in iter arg chain.
    current = iter_init;
    for_op.erase();
  }
}

// ---------------------------------------------------------------------------
// Emit a new ktdf.read_from_fifo with a memref result type that mirrors the
// tensor-typed input at position 0 of `generic_op`.
//
// Asserts that `generic_op` has exactly one input and that the input is
// defined by a ktdf.read_from_fifo — no other producer is supported.
// ---------------------------------------------------------------------------
static Value convertInputToMemref(OpBuilder& builder,
                                  linalg::GenericOp generic_op) {
  assert(generic_op.getInputs().size() == 1 &&
         "convertInputToMemref: expected exactly one input on linalg.generic");
  Value input = generic_op.getInputs()[0];
  auto orig_read = input.getDefiningOp<ktdf::ReadFromFifoOp>();
  assert(orig_read &&
         "convertInputToMemref: input[0] must be a ktdf.read_from_fifo");

  auto in_tensor_type = cast<RankedTensorType>(input.getType());
  auto in_memref_type = MemRefType::get(in_tensor_type.getShape(),
                                        in_tensor_type.getElementType());
  return ktdf::ReadFromFifoOp::create(builder, generic_op.getLoc(),
                                      in_memref_type, orig_read.getFifoSlot())
      .getResult();
}

// ---------------------------------------------------------------------------
// Lower a single reduction linalg.generic to buffer semantics.
// `generic_op` must have at least one reduction iterator and its input must
// be a ktdf.read_from_fifo.
// ---------------------------------------------------------------------------
static LogicalResult rewriteGeneric(
    linalg::GenericOp generic_op,
    scheduler::arch_view::GroupLocalMemory& group_local_mem) {
  auto stage = generic_op->getParentOfType<ktdf::StageOp>();
  assert(stage && "expected enclosing ktdf.stage");

  // Step 1: decide which memory kind backs the accumulator.
  Attribute mem_space = group_local_mem.getLocalMemoryKindForStage(stage);
  if (!mem_space) return failure();

  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  auto out_tensor_type =
      cast<RankedTensorType>(generic_op.getDpsInitOperand(0)->get().getType());
  auto alloc_type = MemRefType::get(out_tensor_type.getShape(),
                                    out_tensor_type.getElementType(),
                                    MemRefLayoutAttrInterface{}, mem_space);

  // Step 2: allocate the accumulator memref at the top of the stage.
  builder.setInsertionPointToStart(stage.getBody());
  auto alloc = memref::AllocOp::create(builder, loc, alloc_type);

  // Step 3: capture the iter_arg and walk up to find the outermost initializer
  // before anything is erased.
  assert(generic_op.getOutputs().size() == 1 &&
         "rewriteGeneric: expected exactly one output on linalg.generic");
  auto acc_iter_arg = dyn_cast<BlockArgument>(generic_op.getOutputs()[0]);
  assert(acc_iter_arg &&
         isa<scf::ForOp>(acc_iter_arg.getOwner()->getParentOp()) &&
         "rewriteGeneric: outs[0] must be an iter_arg of an scf.for");

  Value outermost_init;
  {
    Value cursor = acc_iter_arg;
    while (auto ba = dyn_cast<BlockArgument>(cursor)) {
      auto for_op = cast<scf::ForOp>(ba.getOwner()->getParentOp());
      assert(for_op);
      outermost_init = for_op.getInits()[ba.getArgNumber() - 1];
      cursor = outermost_init;
    }
  }
  assert(outermost_init && "could not find iter_arg initializer");

  // Step 4: replace all uses of the outermost initializer (e.g. the scf.for
  // init operand) with alloc_val *before* lowering it, so that when
  // lowerIterArgInitializer erases the defining op no uses remain.
  outermost_init.replaceAllUsesWith(alloc.getResult());

  // Step 4b: lower the initializer — emit linalg.fill (and/or memref.copy) in
  // the right place and clean up tensor ops.
  if (failed(lowerIterArgInitializer(outermost_init, alloc.getResult())))
    return failure();

  // Step 5: emit a new ktdf.read_from_fifo with a memref result type so the
  // buffer-semantics linalg.generic below has a pure-buffer input.
  builder.setInsertionPoint(generic_op);
  Value new_read = convertInputToMemref(builder, generic_op);

  // Step 6: pure-buffer linalg.generic — memref ins + memref outs, no result.
  auto buf_generic = linalg::GenericOp::create(
      builder, loc,
      /*resultTensorTypes=*/TypeRange{},
      /*inputs=*/ValueRange{new_read},
      /*outputs=*/ValueRange{alloc.getResult()},
      generic_op.getIndexingMapsAttr(), generic_op.getIteratorTypesAttr(),
      /*doc=*/StringAttr{},
      /*library_call=*/StringAttr{});
  IRMapping mapping;
  generic_op.getRegion().cloneInto(&buf_generic.getRegion(), mapping);
  // cloneInto prepends an empty placeholder block; drop it, keep the clone.
  Block& placeholder = buf_generic.getRegion().front();
  if (&placeholder != &buf_generic.getRegion().back()) placeholder.erase();

  // Step 7: replace generic result with alloc, patch write_to_fifo users,
  // then erase the generic and its now-dead tensor read_from_fifo input.
  Value generic_result = generic_op.getResult(0);
  for (Operation* user : generic_result.getUsers())
    if (auto write_op = dyn_cast<ktdf::WriteToFifoOp>(user))
      write_op.getDataMutable().assign(alloc.getResult());
  generic_result.replaceAllUsesWith(alloc.getResult());

  Value orig_input = generic_op.getInputs()[0];
  generic_op.erase();
  if (orig_input.use_empty())
    if (auto* orig_input_op = orig_input.getDefiningOp())
      orig_input_op->erase();

  // Step 8: walk up the iter_arg chain and rebuild each scf.for without it.
  // The initializer has already been transformed; no erasure is needed here.
  removeUnusedIterArgChain(acc_iter_arg, alloc.getResult());
  return success();
}

// ---------------------------------------------------------------------------
// Set `new_type` on a PrivateOp result and its corresponding inner value
// (the private_yield operand at the same index).
//
// Always called with a direct PrivateOp result; the inner value is derived
// by indexing into private_yield.
//
// Example:
//   %2#2 = ktdf.private -> (!ktdf.fifo.slot<"A"->"B", 4xf16>, ...)
//     → widenPrivateResult(%2#2, !ktdf.fifo.slot<"A"->"B", 32xf16>)
//   %2#3 = ktdf.private -> (memref<4xf16, ms>, ...)
//     → widenPrivateResult(%2#3, memref<2x16xf16, ms>)
// ---------------------------------------------------------------------------
static void widenPrivateResult(Value priv_res, Type new_type) {
  auto priv_op = cast<ktdf::PrivateOp>(cast<OpResult>(priv_res).getOwner());
  unsigned idx = cast<OpResult>(priv_res).getResultNumber();
  Value inner =
      cast<ktdf::PrivateYieldOp>(priv_op.getRegion().front().getTerminator())
          .getOperand(idx);
  inner.setType(new_type);
  priv_res.setType(new_type);
}

// ---------------------------------------------------------------------------
// Iterate the direct uses of `fifo_slot` and for each data_transfer that
// references it:
//   - update both static size fields (fifo side and alloc side),
//   - rebuild the alloc-side affine map to match the new rank,
//   - widen the paired PrivateOp result and its inner alloc.
// Early-exits per transfer if the alloc is already the target type.
// ---------------------------------------------------------------------------
static void widenFifoUses(Value fifo_slot, ArrayRef<int64_t> new_shape,
                          MemRefType new_alloc_type, MLIRContext* ctx) {
  auto new_sizes_attr = DenseI64ArrayAttr::get(ctx, new_shape);

  // Build a zero-constant affine map for the alloc side of a transfer,
  // preserving the number of dim inputs to match the existing dynamic indices.
  auto makeZeroMap = [&](unsigned num_dims) -> AffineMapAttr {
    SmallVector<AffineExpr> zero_exprs(new_shape.size(),
                                       getAffineConstantExpr(0, ctx));
    return AffineMapAttr::get(AffineMap::get(num_dims, 0, zero_exprs, ctx));
  };

  for (Operation* user : fifo_slot.getUsers()) {
    auto xfer = dyn_cast<ktdf::DataTransferOp>(user);
    if (!xfer) continue;

    // Determine which end carries the alloc; update both size fields and
    // rebuild the alloc-side map to match the widened rank, preserving the
    // number of dynamic index operands on that side.
    Value alloc_side;
    if (xfer.isSourceFifo()) {
      xfer.setStaticSourceSizesAttr(new_sizes_attr);
      xfer.setStaticDestSizesAttr(new_sizes_attr);
      xfer.setDestMapAttr(makeZeroMap(xfer.getDestIndices().size()));
      alloc_side = xfer.getDestination();
    } else if (xfer.isDestFifo()) {
      xfer.setStaticDestSizesAttr(new_sizes_attr);
      xfer.setStaticSourceSizesAttr(new_sizes_attr);
      xfer.setSourceMapAttr(makeZeroMap(xfer.getSourceIndices().size()));
      alloc_side = xfer.getSource();
    } else {
      continue;
    }

    // alloc_side must be a PrivateOp result; skip anything else.
    auto priv_result = dyn_cast<OpResult>(alloc_side);
    if (!priv_result) continue;
    if (!isa<ktdf::PrivateOp>(priv_result.getOwner())) continue;
    if (alloc_side.getType() == new_alloc_type) continue;  // no widening needed
    widenPrivateResult(alloc_side, new_alloc_type);

    // Fix any other data_transfer that references alloc_side as source or
    // destination but was not reached via the fifo_slot use-list (e.g. a
    // memref→memref store-out transfer downstream of the widened alloc).
    for (Operation* alloc_user : alloc_side.getUsers()) {
      auto other_xfer = dyn_cast<ktdf::DataTransferOp>(alloc_user);
      if (!other_xfer || other_xfer == xfer) continue;
      if (other_xfer.getSource() == alloc_side) {
        if (auto src_map = other_xfer.getSourceMapAttr())
          if (src_map.getValue().getNumResults() != new_shape.size())
            other_xfer.setSourceMapAttr(
                makeZeroMap(other_xfer.getSourceIndices().size()));
        other_xfer.setStaticSourceSizesAttr(new_sizes_attr);
      }
      if (other_xfer.getDestination() == alloc_side) {
        if (auto dst_map = other_xfer.getDestMapAttr())
          if (dst_map.getValue().getNumResults() != new_shape.size())
            other_xfer.setDestMapAttr(
                makeZeroMap(other_xfer.getDestIndices().size()));
        other_xfer.setStaticDestSizesAttr(new_sizes_attr);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Lower an in-stick reduction linalg.generic to buffer semantics.
//
// At this point (after rewriteGeneric has run on G1) the stage contains:
//
//   %result = linalg.generic ins(%alloc_G1: memref<D0x...xDNxf16, ms>)
//                            outs(%empty:   tensor<D0x...xDN-1xf16>)
//   ktdf.write_to_fifo %result, %fifo_out
//
// %alloc_G1 is already a memref (RAUW'd by rewriteGeneric).  We reuse it as
// both input and output buffer:
//
//   ins  = %alloc_G1 (full memref<D0x...xDNxf16>)
//   outs = a rank-reducing subview of %alloc_G1 with size=1 on all reduction
//          dims (rank-reduced away) and original size on parallel dims.  This
//          gives a strided view into alloc_G1 where reduced results are
//          written back in-place.
//
// After the in-stick reduction, the full %alloc_G1 buffer — not the subview —
// is sent to the FIFO so the downstream stage receives all partial sums.
// The FIFO slot type and the paired memref.alloc in the ktdf.private block are
// widened to match %alloc_G1's shape.
//
// Example (memref<2x64xf16, ms>, reduction over dim 1):
//
//   %sv = memref.subview %alloc_G1[0, 0][2, 1][1, 1]
//             : memref<2x64xf16, ms> to memref<2xf16, strided<[64]>, ms>
//   linalg.generic ins(%alloc_G1) outs(%sv)
//   ktdf.write_to_fifo %alloc_G1, %fifo_out    // full buffer, not subview
//
//   // ktdf.private widened (flat count 2 → 128, shape 2xf16 → 2x64xf16):
//   %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"A" -> "B", 128xf16>
//   %acc  = memref.alloc() : memref<2x64xf16, ct_local>
// ---------------------------------------------------------------------------
static LogicalResult rewriteInStickGeneric(linalg::GenericOp generic_op) {
  auto stage = generic_op->getParentOfType<ktdf::StageOp>();
  assert(stage && "expected enclosing ktdf.stage");

  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  // ins[0] is a memref — either the across-stick alloc from rewriteGeneric, or
  // a memref-typed read_from_fifo emitted by the caller when no across-stick
  // loop exists.
  Value alloc_in = generic_op.getInputs()[0];
  auto in_memref_type = cast<MemRefType>(alloc_in.getType());
  unsigned in_rank = in_memref_type.getRank();

  // Inspect input indexing map and iterator types to classify each input dim.
  auto iter_types = generic_op.getIteratorTypesArray();
  AffineMap in_map = generic_op.getIndexingMapsArray().front();
  ArrayRef<int64_t> in_shape = in_memref_type.getShape();

  // Build subview sizes: 1 on reduction dims (rank-reduced away), original
  // size on parallel dims.  Offsets and strides are all 0/1.
  SmallVector<int64_t> sv_sizes(in_rank);
  for (unsigned d = 0; d < in_rank; ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(in_map.getResult(d));
    assert(dim_expr && "in-stick input map must be identity-like");
    unsigned loop_dim = dim_expr.getPosition();
    sv_sizes[d] = (iter_types[loop_dim] == utils::IteratorType::reduction)
                      ? 1
                      : in_shape[d];
  }

  SmallVector<OpFoldResult> sv_offsets(in_rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sv_sizes_ofr;
  for (int64_t sz : sv_sizes) sv_sizes_ofr.push_back(builder.getIndexAttr(sz));
  SmallVector<OpFoldResult> sv_strides(in_rank, builder.getIndexAttr(1));

  // Rank-reduced result shape: drop only reduction dims (those set to size 1
  // above).  Parallel dims that happen to be size 1 must be kept so the result
  // rank matches the output indexing map.
  SmallVector<int64_t> sv_result_shape;
  for (unsigned d = 0; d < in_rank; ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(in_map.getResult(d));
    unsigned loop_dim = dim_expr ? dim_expr.getPosition() : in_rank;
    if (iter_types[loop_dim] != utils::IteratorType::reduction)
      sv_result_shape.push_back(in_shape[d]);
  }

  // Let SubViewOp infer the correct strided layout from the source memref.
  auto sv_result_type =
      cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          sv_result_shape, in_memref_type, sv_offsets, sv_sizes_ofr,
          sv_strides));

  auto subview =
      memref::SubViewOp::create(builder, loc, sv_result_type, alloc_in,
                                sv_offsets, sv_sizes_ofr, sv_strides);

  // Buffer linalg.generic: ins = full alloc_in, outs = rank-reduced subview.
  // Original maps and iterator_types are preserved — they already express the
  // correct ins/outs rank relationship.
  auto buf_generic = linalg::GenericOp::create(
      builder, loc,
      /*resultTensorTypes=*/TypeRange{},
      /*inputs=*/ValueRange{alloc_in},
      /*outputs=*/ValueRange{subview.getResult()},
      generic_op.getIndexingMapsAttr(), generic_op.getIteratorTypesAttr(),
      /*doc=*/StringAttr{},
      /*library_call=*/StringAttr{});
  IRMapping mapping;
  generic_op.getRegion().cloneInto(&buf_generic.getRegion(), mapping);
  Block& placeholder = buf_generic.getRegion().front();
  if (&placeholder != &buf_generic.getRegion().back()) placeholder.erase();

  // Patch write_to_fifo to send the full alloc_in buffer (not the subview),
  // and capture the old fifo slot so we can widen its type below.
  Value old_fifo_slot;
  Value generic_result = generic_op.getResult(0);
  for (Operation* user :
       llvm::make_early_inc_range(generic_result.getUsers())) {
    if (auto write_op = dyn_cast<ktdf::WriteToFifoOp>(user)) {
      old_fifo_slot = write_op.getFifoSlot();
      write_op.getDataMutable().assign(alloc_in);
    }
  }
  generic_result.replaceAllUsesWith(subview.getResult());

  // Widen the FIFO slot type and all downstream allocs/transfers that use it.
  // The new element count is the product of all input dimensions.
  if (old_fifo_slot) {
    auto old_slot_type = cast<ktdf::FifoSlotType>(old_fifo_slot.getType());
    Type elem_type = old_slot_type.getElementType();

    int64_t new_num_elems = 1;
    for (int64_t sz : in_shape) new_num_elems *= sz;

    auto new_slot_type = ktdf::FifoSlotType::get(
        generic_op.getContext(), old_slot_type.getSrc(),
        old_slot_type.getDest(), new_num_elems, elem_type);
    auto new_alloc_type =
        MemRefType::get(in_shape, elem_type, MemRefLayoutAttrInterface{},
                        in_memref_type.getMemorySpace());

    // Widen the PrivateOp result for the fifo slot (and its inner allocate).
    widenPrivateResult(old_fifo_slot, new_slot_type);

    // Update all data_transfer uses of the fifo slot and widen the paired
    // memref.alloc in the ktdf.private block for each transfer.
    widenFifoUses(old_fifo_slot, in_shape, new_alloc_type,
                  generic_op.getContext());
  }

  // Erase the original tensor.empty output initializer and the tensor generic.
  Value orig_out_init = generic_op.getDpsInitOperand(0)->get();
  generic_op.erase();
  if (orig_out_init.use_empty())
    if (auto* def = orig_out_init.getDefiningOp()) def->erase();

  return success();
}

struct MapReductionPartialsPass
    : public ktdf::impl::MapReductionPartialsPassBase<
          MapReductionPartialsPass> {
  using MapReductionPartialsPassBase<
      MapReductionPartialsPass>::MapReductionPartialsPassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp module = getOperation();

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module->emitError("Unable to import the device specification.");
      signalPassFailure();
      return;
    }
    auto& group_local_mem =
        device_manager.getOrCreateView<scheduler::arch_view::GroupLocalMemory>(
            *device);

    // Collect generics in two buckets.  Loop-exposed (across-stick) generics
    // must be processed first so their loop results are RAUW'd to alloc
    // memrefs before the in-stick generics are processed (which expect ins[0]
    // to already be a memref).
    SmallVector<linalg::GenericOp> loop_exposed, in_stick;
    module.walk([&](linalg::GenericOp generic_op) {
      if (!hasReductionIterator(generic_op)) return;
      if (generic_op.getOutputs().empty()) return;
      auto iter_arg = dyn_cast<BlockArgument>(generic_op.getOutputs()[0]);
      if (iter_arg && isa<scf::ForOp>(iter_arg.getOwner()->getParentOp()))
        loop_exposed.push_back(generic_op);
      else
        in_stick.push_back(generic_op);
    });

    for (auto generic_op : loop_exposed) {
      if (failed(rewriteGeneric(generic_op, group_local_mem))) {
        signalPassFailure();
        return;
      }
    }
    for (auto generic_op : in_stick) {
      // ins[0] is a tensor-typed ktdf.read_from_fifo when there was no
      // across-stick loop feeding this generic (i.e. rewriteGeneric did not run
      // on its pipeline).  Emit a memref-typed read in its place so that
      // rewriteInStickGeneric can assume ins[0] is already a memref.
      Operation* stale_tensor_read = nullptr;
      if (generic_op.getInputs()[0].getDefiningOp<ktdf::ReadFromFifoOp>()) {
        OpBuilder builder(generic_op);
        stale_tensor_read = generic_op.getInputs()[0].getDefiningOp();
        Value new_read = convertInputToMemref(builder, generic_op);
        generic_op.getInputsMutable().assign(new_read);
      }
      // Transform in-stick generic into memref-typed generic.
      if (failed(rewriteInStickGeneric(generic_op))) {
        signalPassFailure();
        return;
      }
      if (stale_tensor_read && stale_tensor_read->use_empty())
        stale_tensor_read->erase();
    }
  }
};

}  // namespace

auto mlir::ktdf::createMapReductionPartialsPass() -> std::unique_ptr<Pass> {
  return std::make_unique<MapReductionPartialsPass>();
}
