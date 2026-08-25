// RUN: dataflow-scheduler-opt --split-reduction-inner-outer-dim %s | FileCheck %s

// Test: parallel loop dim (d3) that appears only in the output map.
//
// The compute stage holds a linalg.generic that reduces tensor<2x1x64xf16>
// to tensor<1x64xf16> with iterator types [reduction, parallel, reduction, parallel]:
//   d0 (size 2,  reduction) — across-stick
//   d1 (size 1,  parallel)  — appears in both maps
//   d2 (size 64, reduction) — in-stick (vector_length=64 for f16)
//   d3 (size 64, parallel)  — appears ONLY in the output map result
//
// input_map  = (d0,d1,d2,d3) -> (d0,d1,d2)   tensor<2x1x64xf16>
// output_map = (d0,d1,d2,d3) -> (d1,d3)       tensor<1x64xf16>
//
// Before the fix the pass omitted d3 from G1's output map (the input-map walk
// never visits it), producing a non-invertible map and a verifier error.
//
// After the split:
//   Generic 1 (across-stick, reduces d0):
//     input:  (d0,d1,d2,d3) -> (d0,d1,d2)     tensor<2x1x64xf16>
//     output: (d0,d1,d2,d3) -> (d1,d2,d3)     tensor<1x64x64xf16>
//     iterator_types = ["reduction", "parallel", "parallel", "parallel"]
//
//   Generic 2 (in-stick, reduces d2 — renamed to new d1):
//     input:  (d0,d1,d2) -> (d0,d1,d2)         tensor<1x64x64xf16>
//     output: (d0,d1,d2) -> (d0,d2)             tensor<1x64xf16>
//     iterator_types = ["parallel", "reduction", "parallel"]

// CHECK: #[[$G1_IN:.+]]  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: #[[$G1_OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK: #[[$G2_IN:.+]]  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$G2_OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: module @local_schedule_0
// CHECK:         ktdf.stage depends_in({{.*}}) depends_out({{.*}}) {
// CHECK:           %[[IN:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<2x1x64xf16>
// CHECK:           %[[INTER_EMPTY:.*]] = tensor.empty() : tensor<1x64x64xf16>
// CHECK:           %[[G1:.*]] = linalg.generic
// CHECK-SAME:        indexing_maps = [#[[$G1_IN]], #[[$G1_OUT]]]
// CHECK-SAME:        iterator_types = ["reduction", "parallel", "parallel", "parallel"]
// CHECK-SAME:        ins(%[[IN]] : tensor<2x1x64xf16>)
// CHECK-SAME:        outs(%[[INTER_EMPTY]] : tensor<1x64x64xf16>)
// CHECK:           %[[OUT_EMPTY:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK:           %[[G2:.*]] = linalg.generic
// CHECK-SAME:        indexing_maps = [#[[$G2_IN]], #[[$G2_OUT]]]
// CHECK-SAME:        iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME:        ins(%[[G1]] : tensor<1x64x64xf16>)
// CHECK-SAME:        outs(%[[OUT_EMPTY]] : tensor<1x64xf16>)
// CHECK:           ktdf.write_to_fifo %[[G2]], {{.*}}

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#set  = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 0 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  module {
    func.func @sum_onstick_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0          = arith.constant 0 : index
      %c1          = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 1, 64], strides: [64, 64, 1]
           {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x1x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [1, 64], strides: [64, 1]
           {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
      %memspacecast   = memref.memory_space_cast %0 : memref<2x1x64xf16> to memref<2x1x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast
           to offset: [0], sizes: [2, 1, 64], strides: [64, 64, 1]
           : memref<2x1x64xf16, "DDR"> to memref<2x1x64xf16, strided<[64, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast
           : memref<2x1x64xf16, strided<[64, 64, 1]>, "DDR">
          to memref<2x1x64xf16, strided<[64, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<1x64xf16> to memref<1x64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0
           to offset: [0], sizes: [1, 64], strides: [64, 1]
           : memref<1x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1
           : memref<1x64xf16, strided<[64, 1]>, "DDR">
          to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
      ktdf.pipeline {
        %priv:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>,
                                   !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
                                   !ktdf.token, !ktdf.token) {
          %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
          %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %f0, %f1, %t0, %t1
              : !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>,
                !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
                !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%priv#2) {
          ktdf.data_transfer from %cast[%c0, %c0, %c0] size [2, 1, 64] to %priv#0 size [128]
              : memref<2x1x64xf16, strided<[64, 64, 1], offset: ?>, "DDR">,
                !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%priv#2) depends_out(%priv#3) {
          %in   = ktdf.read_from_fifo %priv#0 : <"L1LU" -> "SFU", 128xf16> -> tensor<2x1x64xf16>
          %init = tensor.empty() : tensor<1x64xf16>
          %res  = linalg.generic {
              indexing_maps = [#map, #map1],
              iterator_types = ["reduction", "parallel", "reduction", "parallel"]
          } ins(%in : tensor<2x1x64xf16>) outs(%init : tensor<1x64xf16>) {
          ^bb0(%a: f16, %acc: f16):
            %s = arith.addf %a, %acc : f16
            linalg.yield %s : f16
          } -> tensor<1x64xf16>
          ktdf.write_to_fifo %res, %priv#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%priv#3) depends_out(none) {
          ktdf.data_transfer from %priv#1 size [64] to %cast_2[%c0, %c0] size [1, 64]
              : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
                memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
        } {applicable_units = ["L1SU"]}
      }
      return
    }
  }
}
