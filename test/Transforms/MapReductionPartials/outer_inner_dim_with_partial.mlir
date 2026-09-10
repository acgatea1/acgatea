// RUN: dataflow-scheduler-opt --map-reduction-partials %s | FileCheck %s

// Tests the case where ReductionLoopExposure has produced:
//   G1 (outer-dim, loop-exposed): absmax reduction tensor<1x1x64xf16> -> tensor<1x64xf16>
//      carried as iter_arg through a scf.for reduction loop.
//   combine scf.if: on the first chunk yields the loop result directly; on
//      subsequent chunks reads the previous partial from a FIFO and adds it.
//   G2 (inner-dim): absmax reduction tensor<1x64xf16> -> tensor<1x64xf16>
//      whose input is the combine scf.if result.
//
// MapReductionPartials must:
//   - Lower G1: alloc + linalg.fill(0.0) + buffer scf.for (no iter_arg).
//   - Lower the combine scf.if to a result-less scf.if:
//       then: empty (alloc already holds the loop result)
//       else: memref read_from_fifo + buffer linalg.generic addf into alloc
//   - Lower G2 using the G1 alloc as ins, a rank-reducing subview as outs,
//     and write the full alloc to the widened FIFO.

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: module @local_schedule_0

// Outer pipeline private: unchanged
// CHECK:         %[[PRIV0:.*]]:4 = ktdf.private -> (memref<2x1x256x64xf16
// CHECK:           memref.alloc() : memref<2x1x256x64xf16
// CHECK:           memref.alloc() : memref<2x1x64xf16

// Inner pipeline private: slots unchanged (G2 widens slot #2 below)
// CHECK:         %[[PRIV1:.*]]:5 = ktdf.private
// CHECK-SAME:      !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-SAME:      !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-SAME:      !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>

// L1LU load stage: unchanged
// CHECK:         ktdf.stage depends_in(none)
// CHECK:           scf.if

// SFU compute stage — key transformations:
// CHECK:         ktdf.stage depends_in(%[[PRIV1]]#3)
//
// G1: alloc + fill(0.0) hoisted to top of stage
// CHECK-NEXT:      %[[ALLOC:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:      %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      linalg.fill ins(%[[ZERO]] : f16) outs(%[[ALLOC]] : memref<1x64xf16, "SFU_REG">)
//
// G1: buffer scf.for reduction loop (no iter_arg)
// CHECK:           scf.for %[[IV:.*]] = {{.*}} {
// CHECK-NEXT:        %[[RD0:.*]] = ktdf.read_from_fifo %[[PRIV1]]#0 : <"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
// CHECK-NEXT:        linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[RD0]] : memref<1x1x64xf16>) outs(%[[ALLOC]] : memref<1x64xf16, "SFU_REG">)
// CHECK:           } {loop_type = #ktdf.loop_type<reduction_loop>}
//
// combine scf.if: result-less; then empty, else read_from_fifo + addf into alloc
// CHECK:           scf.if %[[IS_FIRST:.*]] {
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[RD1:.*]] = ktdf.read_from_fifo %[[PRIV1]]#1 : <"L1LU" -> "SFU", 64xf16> -> memref<1x64xf16>
// CHECK-NEXT:        linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%[[RD1]] : memref<1x64xf16>) outs(%[[ALLOC]] : memref<1x64xf16, "SFU_REG">)
// CHECK:           }
//
// G2: rank-reducing subview of alloc (reduction dim d1 collapsed to size 1)
// CHECK:           %[[SV:.*]] = memref.subview %[[ALLOC]][0, 0] [1, 1] [1, 1]
// CHECK-SAME:        memref<1x64xf16, "SFU_REG"> to memref<1x1xf16, strided<[64, 1]>, "SFU_REG">
// CHECK-NEXT:      linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[ALLOC]] : memref<1x64xf16, "SFU_REG">) outs(%[[SV]] : memref<1x1xf16
//
// write_to_fifo sends full alloc (not subview)
// CHECK:           ktdf.write_to_fifo %[[ALLOC]], %[[PRIV1]]#2


#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  module {
    func.func @absmax_onstick_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c2 = arith.constant 2 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>>
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>
      %memspacecast_0 = memref.memory_space_cast %1 : memref<2x64xf16> to memref<2x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, #ktdp.memory_space<global>> to memref<2x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>>
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<2x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>> to memref<2x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
      ktdf.pipeline {
        %3:4 = ktdf.private -> (memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>
          %alloc_3 = memref.alloc() : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
          %4 = ktdf.create_token : !ktdf.token
          %5 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %4, %5 : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%3#2) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %4 = arith.subi %arg0, %c0 : index
            %5 = arith.divsi %4, %c1 : index
            ktdf.data_transfer from %cast[%arg0, 0, 0] size [1, 256, 64] to %3#0[%5, 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%3#2) depends_out(%3#3) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %c4 = arith.constant 4 : index
            %c0_3 = arith.constant 0 : index
            %c1_4 = arith.constant 1 : index
            scf.for %arg1 = %c0_3 to %c4 step %c1_4 {
              %4 = arith.cmpi eq, %arg1, %c0_3 : index
              %c0_5 = arith.constant 0 : index
              %c1_6 = arith.constant 1 : index
              %c63 = arith.constant 63 : index
              ktdf.pipeline {
                %5:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                  %6 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  %7 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  %8 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                  %9 = ktdf.create_token : !ktdf.token
                  %10 = ktdf.create_token : !ktdf.token
                  ktdf.private_yield %6, %7, %8, %9, %10 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
                }
                ktdf.stage depends_in(none) depends_out(%5#3) {
                  %c0_7 = arith.constant 0 : index
                  %c1_8 = arith.constant 1 : index
                  %6 = arith.subi %arg0, %c0_7 : index
                  %7 = arith.divsi %6, %c1_8 : index
                  scf.if %4 {
                  } else {
                    ktdf.data_transfer from %3#1[%7, %c0_7, %c0_7] size [1, 1, 64] to %5#1 size [64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  }
                  %c64 = arith.constant 64 : index
                  scf.for %arg2 = %c0_5 to %c64 step %c1_6 {
                    ktdf.data_transfer from %3#0[%7, %c0_7, %arg1 * 64 + %arg2, %c0_7] size [1, 1, 1, 64] to %5#0 size [64] : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  } {loop_type = #ktdf.loop_type<reduction_loop>}
                } {applicable_units = ["L1LU"]}
                ktdf.stage depends_in(%5#3) depends_out(%5#4) {
                  // G1: outer-dim absmax reduction loop, iter_arg initialized
                  // to tensor.empty.
                  %6 = tensor.empty() : tensor<1x64xf16>
                  %c64 = arith.constant 64 : index
                  %7 = scf.for %arg2 = %c0_5 to %c64 step %c1_6 iter_args(%arg3 = %6) -> (tensor<1x64xf16>) {
                    %11 = ktdf.read_from_fifo %5#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%11 : tensor<1x1x64xf16>) outs(%arg3 : tensor<1x64xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %13 = math.absf %in : f16
                      %14 = math.absf %out : f16
                      %15 = arith.maxnumf %13, %14 : f16
                      linalg.yield %15 : f16
                    } -> tensor<1x64xf16>
                    scf.yield %12 : tensor<1x64xf16>
                  } {loop_type = #ktdf.loop_type<reduction_loop>}
                  // combine scf.if: first chunk -> pass loop result through;
                  // subsequent chunks -> add previous partial from FIFO.
                  %8 = scf.if %4 -> (tensor<1x64xf16>) {
                    scf.yield %7 : tensor<1x64xf16>
                  } else {
                    %11 = ktdf.read_from_fifo %5#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
                    %12 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<1x64xf16>) outs(%7 : tensor<1x64xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %13 = arith.addf %in, %out : f16
                      linalg.yield %13 : f16
                    } -> tensor<1x64xf16>
                    scf.yield %12 : tensor<1x64xf16>
                  }
                  // G2: inner-dim absmax reduction over the combined result.
                  %9 = tensor.empty() : tensor<1x64xf16>
                  %10 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%8 : tensor<1x64xf16>) outs(%9 : tensor<1x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %11 = math.absf %in : f16
                    %12 = math.absf %out : f16
                    %13 = arith.maxnumf %11, %12 : f16
                    linalg.yield %13 : f16
                  } -> tensor<1x64xf16>
                  ktdf.write_to_fifo %10, %5#2 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                } {applicable_units = ["SFU"]}
                ktdf.stage depends_in(%5#4) depends_out(none) {
                  %c64 = arith.constant 64 : index
                  scf.for %arg2 = %c0_5 to %c64 step %c1_6 {
                    %c0_7 = arith.constant 0 : index
                    %c1_8 = arith.constant 1 : index
                    %6 = arith.subi %arg0, %c0_7 : index
                    %7 = arith.divsi %6, %c1_8 : index
                    %8 = arith.cmpi eq, %arg2, %c63 : index
                    scf.if %8 {
                      ktdf.data_transfer from %5#2 size [64] to %3#1[%7, %c0_7, %c0_7] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
                    }
                  } {loop_type = #ktdf.loop_type<reduction_loop>}
                } {applicable_units = ["L1SU"]}
              }
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%3#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %4 = arith.subi %arg0, %c0 : index
            %5 = arith.divsi %4, %c1 : index
            ktdf.data_transfer from %3#1[%5, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<2x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
