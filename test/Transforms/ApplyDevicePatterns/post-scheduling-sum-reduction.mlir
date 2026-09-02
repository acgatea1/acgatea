// RUN: dataflow-scheduler-opt %s -allow-unregistered-dialect \
// RUN:   -apply-device-patterns='groups=post_scheduling' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @sum_reduction
// CHECK-NOT:     linalg.generic
// CHECK:         "test.simd_reduction_sum"(%[[IN:.*]], %[[ACC:.*]]) : (memref<64x32xf16, "L1">, memref<64xf16, "L1">) -> ()
// CHECK-NOT:     memref.subview

// Verify that the post_scheduling PDL pattern in the sample device rewrites a
// linalg.generic sum-reduction (inner-dim, output via subview) into a bare
// test.simd_reduction op whose second operand is the subview's *source*.

#map_in  = affine_map<(d0, d1) -> (d0, d1)>
#map_out = affine_map<(d0, d1) -> (d0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  func.func @sum_reduction() {
    %alloc_in  = memref.alloc() : memref<64x32xf16, "L1">
    %alloc_acc = memref.alloc() : memref<64xf16, "L1">

    // The pattern requires the output to be a memref.subview of the accumulator.
    %subview = memref.subview %alloc_acc[0][64][1]
        : memref<64xf16, "L1"> to memref<64xf16, "L1">

    linalg.generic {
      indexing_maps = [#map_in, #map_out],
      iterator_types = ["parallel", "reduction"]
    } ins(%alloc_in : memref<64x32xf16, "L1">)
      outs(%subview  : memref<64xf16, "L1">) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %out : f16
      linalg.yield %sum : f16
    }

    return
  }
}
