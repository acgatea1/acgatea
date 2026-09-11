// RUN: dataflow-scheduler-opt --map-reduction-partials %s | FileCheck %s

// Two accumulators through one combine.  A compute that accumulates several
// things carries one accumulator per result, and the combine-with-partial
// scf.if wraps all of them: it has a result per accumulator, each with its own
// partial FIFO slot.  Nothing here may assume a single result -- the bypass has
// to short-circuit every result, and the lowering needs a destination per
// result.
//
// The combine is one 2-result linalg.generic, so both of its results have to be
// lowered together: buffer form drops all results at once, and rewriting only
// one would leave the other without a reader.
//
// Same placement as outer_inner_dim_with_partial.mlir (combine after the
// outer-dim reduction, ahead of the inner-dim one), only widened to two.

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: module @local_schedule_0

// Inner pipeline private: in slot, two partial slots, two out slots, 2 tokens
// CHECK:         %[[PRIV1:.*]]:7 = ktdf.private

// SFU compute stage
// CHECK:         ktdf.stage depends_in(%[[PRIV1]]#5)
//
// G1: one accumulator alloc + fill(0.0) per result
// CHECK-NEXT:      %[[ACC0:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:      %[[ACC1:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:      %[[Z0:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      linalg.fill ins(%[[Z0]] : f16) outs(%[[ACC0]] : memref<1x64xf16, "SFU_REG">)
// CHECK-NEXT:      %[[Z1:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      linalg.fill ins(%[[Z1]] : f16) outs(%[[ACC1]] : memref<1x64xf16, "SFU_REG">)
//
// G1: buffer scf.for reduction loop, both accumulators as outs, no iter_args
// CHECK:           scf.for
// CHECK-NEXT:        %[[RD0:.*]] = ktdf.read_from_fifo %[[PRIV1]]#0 : <"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
// CHECK-NEXT:        linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[RD0]] : memref<1x1x64xf16>) outs(%[[ACC0]], %[[ACC1]] : memref<1x64xf16, "SFU_REG">, memref<1x64xf16, "SFU_REG">)
// CHECK:           } {loop_type = #ktdf.loop_type<reduction_loop>}
//
// combine: result-less scf.if; the else branch reads both partials as memrefs
// and combines them into both accumulators in one buffer-form generic
// CHECK:           scf.if %[[IS_FIRST:.*]] {
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[P0:.*]] = ktdf.read_from_fifo %[[PRIV1]]#1 : <"L1LU" -> "SFU", 64xf16> -> memref<1x64xf16>
// CHECK-NEXT:        %[[P1:.*]] = ktdf.read_from_fifo %[[PRIV1]]#2 : <"L1LU" -> "SFU", 64xf16> -> memref<1x64xf16>
// CHECK-NEXT:        linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%[[P0]], %[[P1]] : memref<1x64xf16>, memref<1x64xf16>) outs(%[[ACC0]], %[[ACC1]] : memref<1x64xf16, "SFU_REG">, memref<1x64xf16, "SFU_REG">)
// CHECK:           }
//
// G2: a rank-reducing subview per accumulator, both as outs of one generic
// CHECK:           %[[SV0:.*]] = memref.subview %[[ACC0]][0, 0] [1, 1] [1, 1]
// CHECK-NEXT:      %[[SV1:.*]] = memref.subview %[[ACC1]][0, 0] [1, 1] [1, 1]
// CHECK-NEXT:      linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP3]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[ACC0]], %[[ACC1]] : memref<1x64xf16, "SFU_REG">, memref<1x64xf16, "SFU_REG">) outs(%[[SV0]], %[[SV1]] :
//
// each write_to_fifo sends its whole accumulator, not its subview
// CHECK:           ktdf.write_to_fifo %[[ACC0]], %[[PRIV1]]#3
// CHECK-NEXT:      ktdf.write_to_fifo %[[ACC1]], %[[PRIV1]]#4


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
                %5:7 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                  %6 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  %7 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  %8 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  %9 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                  %10 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                  %11 = ktdf.create_token : !ktdf.token
                  %12 = ktdf.create_token : !ktdf.token
                  ktdf.private_yield %6, %7, %8, %9, %10, %11, %12 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
                }
                ktdf.stage depends_in(none) depends_out(%5#5) {
                  %c0_7 = arith.constant 0 : index
                  %c1_8 = arith.constant 1 : index
                  %6 = arith.subi %arg0, %c0_7 : index
                  %7 = arith.divsi %6, %c1_8 : index
                  scf.if %4 {
                  } else {
                    ktdf.data_transfer from %3#1[%7, %c0_7, %c0_7] size [1, 1, 64] to %5#1 size [64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    ktdf.data_transfer from %3#1[%7, %c0_7, %c0_7] size [1, 1, 64] to %5#2 size [64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  }
                  %c64 = arith.constant 64 : index
                  scf.for %arg2 = %c0_5 to %c64 step %c1_6 {
                    ktdf.data_transfer from %3#0[%7, %c0_7, %arg1 * 64 + %arg2, %c0_7] size [1, 1, 1, 64] to %5#0 size [64] : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  } {loop_type = #ktdf.loop_type<reduction_loop>}
                } {applicable_units = ["L1LU"]}
                ktdf.stage depends_in(%5#5) depends_out(%5#6) {
                  // G1: two outer-dim absmax accumulators in one loop nest,
                  // each iter_arg initialized to tensor.empty.
                  %6 = tensor.empty() : tensor<1x64xf16>
                  %7 = tensor.empty() : tensor<1x64xf16>
                  %c64 = arith.constant 64 : index
                  %8:2 = scf.for %arg2 = %c0_5 to %c64 step %c1_6 iter_args(%arg3 = %6, %arg4 = %7) -> (tensor<1x64xf16>, tensor<1x64xf16>) {
                    %13 = ktdf.read_from_fifo %5#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                    %14:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%13 : tensor<1x1x64xf16>) outs(%arg3, %arg4 : tensor<1x64xf16>, tensor<1x64xf16>) {
                    ^bb0(%in: f16, %out0: f16, %out1: f16):
                      %15 = math.absf %in : f16
                      %16 = math.absf %out0 : f16
                      %17 = arith.maxnumf %15, %16 : f16
                      %18 = math.absf %out1 : f16
                      %19 = arith.maxnumf %15, %18 : f16
                      linalg.yield %17, %19 : f16, f16
                    } -> (tensor<1x64xf16>, tensor<1x64xf16>)
                    scf.yield %14#0, %14#1 : tensor<1x64xf16>, tensor<1x64xf16>
                  } {loop_type = #ktdf.loop_type<reduction_loop>}
                  // combine scf.if with two results: one 2-result generic
                  // combines both accumulators against their own partials.
                  %9:2 = scf.if %4 -> (tensor<1x64xf16>, tensor<1x64xf16>) {
                    scf.yield %8#0, %8#1 : tensor<1x64xf16>, tensor<1x64xf16>
                  } else {
                    %13 = ktdf.read_from_fifo %5#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
                    %14 = ktdf.read_from_fifo %5#2 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
                    %15:2 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%13, %14 : tensor<1x64xf16>, tensor<1x64xf16>) outs(%8#0, %8#1 : tensor<1x64xf16>, tensor<1x64xf16>) {
                    ^bb0(%in0: f16, %in1: f16, %out0: f16, %out1: f16):
                      %16 = arith.addf %in0, %out0 : f16
                      %17 = arith.maxnumf %in1, %out1 : f16
                      linalg.yield %16, %17 : f16, f16
                    } -> (tensor<1x64xf16>, tensor<1x64xf16>)
                    scf.yield %15#0, %15#1 : tensor<1x64xf16>, tensor<1x64xf16>
                  }
                  // G2: two inner-dim absmax reductions over the combined
                  // results.
                  %10 = tensor.empty() : tensor<1x64xf16>
                  %11 = tensor.empty() : tensor<1x64xf16>
                  %12:2 = linalg.generic {indexing_maps = [#map3, #map3, #map1, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%9#0, %9#1 : tensor<1x64xf16>, tensor<1x64xf16>) outs(%10, %11 : tensor<1x64xf16>, tensor<1x64xf16>) {
                  ^bb0(%in0: f16, %in1: f16, %out0: f16, %out1: f16):
                    %13 = math.absf %in0 : f16
                    %14 = math.absf %out0 : f16
                    %15 = arith.maxnumf %13, %14 : f16
                    %16 = math.absf %in1 : f16
                    %17 = math.absf %out1 : f16
                    %18 = arith.maxnumf %16, %17 : f16
                    linalg.yield %15, %18 : f16, f16
                  } -> (tensor<1x64xf16>, tensor<1x64xf16>)
                  ktdf.write_to_fifo %12#0, %5#3 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                  ktdf.write_to_fifo %12#1, %5#4 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                } {applicable_units = ["SFU"]}
                ktdf.stage depends_in(%5#6) depends_out(none) {
                  %c64 = arith.constant 64 : index
                  scf.for %arg2 = %c0_5 to %c64 step %c1_6 {
                    %c0_7 = arith.constant 0 : index
                    %c1_8 = arith.constant 1 : index
                    %6 = arith.subi %arg0, %c0_7 : index
                    %7 = arith.divsi %6, %c1_8 : index
                    %8 = arith.cmpi eq, %arg2, %c63 : index
                    scf.if %8 {
                      ktdf.data_transfer from %5#3 size [64] to %3#1[%7, %c0_7, %c0_7] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
                      ktdf.data_transfer from %5#4 size [64] to %3#1[%7, %c0_7, %c0_7] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
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
