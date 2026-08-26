// RUN: dataflow-scheduler-opt --map-reduction-partials %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// CHECK-LABEL:   module {
// CHECK:     func.func @sum_1core() attributes {grid = [1]} {
// CHECK:       call @local_schedule_0() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @local_schedule_0()
// CHECK:   }
// CHECK:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-NEXT:     func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_1]], sizes: [2], strides: [1] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.memory_space<global>} : memref<2xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2xf16> to memref<2xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2], strides: [1] : memref<2xf16, "DDR"> to memref<2xf16, strided<[1]>, "DDR">
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<2xf16, strided<[1]>, "DDR"> to memref<2xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 255 : index
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:8 = ktdf.private -> (memref<2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>, memref<2x64xf16, "SFU_REG">, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x256x64xf16, "L1">
// CHECK-NEXT:           %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK-NEXT:           %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<2x64xf16, "SFU_REG">
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[FIFO_0]], %[[FIFO_1]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : memref<2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>, memref<2x64xf16, "SFU_REG">, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#4) {
// CHECK-NEXT:           %[[CONSTANT_5:.*]] = arith.constant 256 : index
// CHECK-NEXT:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_5]] step %[[CONSTANT_3]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[CONSTANT_0]] * 2, 0, 0] size [2, 256, 64] to %[[VAL_0]]#0[0, 0, 0] size [2, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x256x64xf16, "L1">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_2:.*]]#4) depends_out(%[[VAL_2]]#5) {
// CHECK-NEXT:           %[[CONSTANT_6:.*]] = arith.constant 256 : index
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_6]] step %[[CONSTANT_3]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_2]]#0[0, %[[VAL_3]], 0] size [2, 1, 64] to %[[VAL_2]]#1 size [128] : memref<2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1LU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_4:.*]]#5) depends_out(%[[VAL_4]]#6) {
// CHECK-NEXT:           %[[ALLOC_2:.*]] = memref.alloc() : memref<2x64xf16, "SFU_REG">
// CHECK-NEXT:           %[[CONSTANT_7:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:           linalg.fill ins(%[[CONSTANT_7]] : f16) outs(%[[ALLOC_2]] : memref<2x64xf16, "SFU_REG">)
// CHECK-NEXT:           %[[CONSTANT_8:.*]] = arith.constant 256 : index
// CHECK-NEXT:           scf.for %[[VAL_5:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_8]] step %[[CONSTANT_3]] {
// CHECK-NEXT:             %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_4]]#1 : <"L1LU" -> "SFU", 128xf16> -> memref<2x1x64xf16>
// CHECK-NEXT:             linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : memref<2x1x64xf16>) outs(%[[ALLOC_2]] : memref<2x64xf16, "SFU_REG">) {
// CHECK-NEXT:             ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK-NEXT:               %[[ADDF_0:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f16
// CHECK-NEXT:               linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:           %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_2]][0, 0] [2, 1] [1, 1] : memref<2x64xf16, "SFU_REG"> to memref<2xf16, strided<[64]>, "SFU_REG">
// CHECK-NEXT:           linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["parallel", "reduction"]} ins(%[[ALLOC_2]] : memref<2x64xf16, "SFU_REG">) outs(%[[SUBVIEW_0]] : memref<2xf16, strided<[64]>, "SFU_REG">) {
// CHECK-NEXT:           ^bb0(%[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16):
// CHECK-NEXT:             %[[ADDF_1:.*]] = arith.addf %[[VAL_8]], %[[VAL_9]] : f16
// CHECK-NEXT:             linalg.yield %[[ADDF_1]] : f16
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.write_to_fifo %[[ALLOC_2]], %[[VAL_4]]#2 : memref<2x64xf16, "SFU_REG">, <"SFU" -> "L1SU", 128xf16>
// CHECK-NEXT:         } {applicable_units = ["SFU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_10:.*]]#6) depends_out(%[[VAL_10]]#7) {
// CHECK-NEXT:           %[[CONSTANT_9:.*]] = arith.constant 256 : index
// CHECK-NEXT:           scf.for %[[VAL_11:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_9]] step %[[CONSTANT_3]] {
// CHECK-NEXT:             %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_11]], %[[CONSTANT_4]] : index
// CHECK-NEXT:             scf.if %[[CMPI_0]] {
// CHECK-NEXT:               ktdf.data_transfer from %[[VAL_10]]#2 size [2, 64] to %[[VAL_10]]#3[0, 0] size [2, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>, memref<2x64xf16, "SFU_REG">
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_12:.*]]#7) depends_out(none) {
// CHECK-NEXT:           %[[CONSTANT_10:.*]] = arith.constant 256 : index
// CHECK-NEXT:           scf.for %[[VAL_13:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_10]] step %[[CONSTANT_3]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_12]]#3[0, 0] size [2, 64] to %[[CAST_1]]{{\[}}%[[CONSTANT_0]] * 2] size [2] : memref<2x64xf16, "SFU_REG">, memref<2xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNISU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }



// Tests the in-stick generic path: after SplitReductionInnerOuterDim +
// ReductionLoopExposure the SFU compute stage contains two linalg.generic ops:
//   G1 (across-stick, loop-exposed): tensor<2x1x64xf16> -> tensor<2x64xf16>
//   G2 (in-stick, plain):            tensor<2x64xf16>   -> tensor<2xf16>
//
// MapReductionPartials must:
//   - Lower G1 into a memref.alloc + linalg.fill + scf.for with buffer generic.
//   - Lower G2 using the G1 alloc as ins, a rank-reducing subview as outs,
//     and write the subview to the FIFO.



#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>

module {
  module {
    func.func @sum_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c8589934592 = arith.constant 8589934592 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<2xf16> to memref<2xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2], strides: [1] : memref<2xf16, "DDR"> to memref<2xf16, strided<[1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<2xf16, strided<[1]>, "DDR"> to memref<2xf16, strided<[1], offset: ?>, "DDR">
      %c0_3 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c255 = arith.constant 255 : index
      ktdf.pipeline {
        %2:8 = ktdf.private -> (memref<2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, memref<2xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<2x256x64xf16, "L1">
          %3 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
          %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>
          %alloc_4 = memref.alloc() : memref<2xf16, "L1">
          %5 = ktdf.create_token : !ktdf.token
          %6 = ktdf.create_token : !ktdf.token
          %7 = ktdf.create_token : !ktdf.token
          %8 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %3, %4, %alloc_4, %5, %6, %7, %8 : memref<2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, memref<2xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#4) {
          %c256 = arith.constant 256 : index
          scf.for %arg0 = %c0_3 to %c256 step %c1 {
            ktdf.data_transfer from %cast[%c0 * 2, 0, 0] size [2, 256, 64] to %2#0[0, 0, 0] size [2, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x256x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<reduction_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#4) depends_out(%2#5) {
          %c256 = arith.constant 256 : index
          scf.for %arg0 = %c0_3 to %c256 step %c1 {
            ktdf.data_transfer from %2#0[0, %arg0, 0] size [2, 1, 64] to %2#1 size [128] : memref<2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
          } {loop_type = #ktdf.loop_type<reduction_loop>}
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%2#5) depends_out(%2#6) {
          %3 = tensor.empty() : tensor<2x64xf16>
          %c256 = arith.constant 256 : index
          %4 = scf.for %arg0 = %c0_3 to %c256 step %c1 iter_args(%arg1 = %3) -> (tensor<2x64xf16>) {
            %7 = ktdf.read_from_fifo %2#1 : <"L1LU" -> "SFU", 128xf16> -> tensor<2x1x64xf16>
            %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%7 : tensor<2x1x64xf16>) outs(%arg1 : tensor<2x64xf16>) {
            ^bb0(%in: f16, %out: f16):
              %9 = arith.addf %in, %out : f16
              linalg.yield %9 : f16
            } -> tensor<2x64xf16>
            scf.yield %8 : tensor<2x64xf16>
          } {loop_type = #ktdf.loop_type<reduction_loop>}
          %5 = tensor.empty() : tensor<2xf16>
          %6 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%4 : tensor<2x64xf16>) outs(%5 : tensor<2xf16>) {
          ^bb0(%in: f16, %out: f16):
            %7 = arith.addf %in, %out : f16
            linalg.yield %7 : f16
          } -> tensor<2xf16>
          ktdf.write_to_fifo %6, %2#2 : tensor<2xf16>, <"SFU" -> "L1SU", 2xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%2#6) depends_out(%2#7) {
          %c256 = arith.constant 256 : index
          scf.for %arg0 = %c0_3 to %c256 step %c1 {
            %3 = arith.cmpi eq, %arg0, %c255 : index
            scf.if %3 {
              ktdf.data_transfer from %2#2 size [2] to %2#3[0] size [2] : !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, memref<2xf16, "L1">
            }
          } {loop_type = #ktdf.loop_type<reduction_loop>}
        } {applicable_units = ["L1SU"]}
        ktdf.stage depends_in(%2#7) depends_out(none) {
          %c256 = arith.constant 256 : index
          scf.for %arg0 = %c0_3 to %c256 step %c1 {
            ktdf.data_transfer from %2#3[0] size [2] to %cast_2[%c0 * 2] size [2] : memref<2xf16, "L1">, memref<2xf16, strided<[1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<reduction_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
