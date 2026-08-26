// RUN: dataflow-scheduler-opt --map-reduction-partials %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
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
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_1]], sizes: [2], strides: [1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<2xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x64xf16> to memref<2x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2xf16> to memref<2xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2], strides: [1] : memref<2xf16, "DDR"> to memref<2xf16, strided<[1]>, "DDR">
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<2xf16, strided<[1]>, "DDR"> to memref<2xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:8 = ktdf.private -> (memref<2x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>, memref<2x64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x64xf16, "L1">
// CHECK-NEXT:           %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK-NEXT:           %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<2x64xf16>
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[FIFO_0]], %[[FIFO_1]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : memref<2x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>, memref<2x64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#4) {
// CHECK-NEXT:           ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[CONSTANT_0]] * 2, 0] size [2, 64] to %[[VAL_0]]#0[0, 0] size [2, 64] : memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<2x64xf16, "L1">
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_1:.*]]#4) depends_out(%[[VAL_1]]#5) {
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_1]]#0[0, 0] size [2, 64] to %[[VAL_1]]#1 size [128] : memref<2x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK-NEXT:         } {applicable_units = ["L1LU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_2:.*]]#5) depends_out(%[[VAL_2]]#6) {
// CHECK-NEXT:           %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_2]]#1 : <"L1LU" -> "SFU", 128xf16> -> memref<2x64xf16>
// CHECK-NEXT:           %[[SUBVIEW_0:.*]] = memref.subview %[[READ_FROM_FIFO_0]][0, 0] [2, 1] [1, 1] : memref<2x64xf16> to memref<2xf16, strided<[64]>>
// CHECK-NEXT:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction"]} ins(%[[READ_FROM_FIFO_0]] : memref<2x64xf16>) outs(%[[SUBVIEW_0]] : memref<2xf16, strided<[64]>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK-NEXT:             %[[ADDF_0:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f16
// CHECK-NEXT:             linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.write_to_fifo %[[READ_FROM_FIFO_0]], %[[VAL_2]]#2 : memref<2x64xf16>, <"SFU" -> "L1SU", 128xf16>
// CHECK-NEXT:         } {applicable_units = ["SFU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_5:.*]]#6) depends_out(%[[VAL_5]]#7) {
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_5]]#2 size [2, 64] to %[[VAL_5]]#3[0, 0] size [2, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>, memref<2x64xf16>
// CHECK-NEXT:         } {applicable_units = ["L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_6:.*]]#7) depends_out(none) {
// CHECK-NEXT:           ktdf.data_transfer from %[[VAL_6]]#3[0, 0] size [2, 64] to %[[CAST_1]]{{\[}}%[[CONSTANT_0]] * 2] size [2] : memref<2x64xf16>, memref<2xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:         } {applicable_units = ["MNISU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }


// Tests the pure in-stick-only path.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
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
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x64xf16> to memref<2x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<2xf16> to memref<2xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2], strides: [1] : memref<2xf16, "DDR"> to memref<2xf16, strided<[1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<2xf16, strided<[1]>, "DDR"> to memref<2xf16, strided<[1], offset: ?>, "DDR">
      ktdf.pipeline {
        %2:8 = ktdf.private -> (memref<2x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, memref<2xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<2x64xf16, "L1">
          %3 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
          %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>
          %alloc_3 = memref.alloc() : memref<2xf16, "L1">
          %5 = ktdf.create_token : !ktdf.token
          %6 = ktdf.create_token : !ktdf.token
          %7 = ktdf.create_token : !ktdf.token
          %8 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %3, %4, %alloc_3, %5, %6, %7, %8 : memref<2x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, memref<2xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#4) {
          ktdf.data_transfer from %cast[%c0 * 2, 0] size [2, 64] to %2#0[0, 0] size [2, 64] : memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<2x64xf16, "L1">
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#4) depends_out(%2#5) {
          ktdf.data_transfer from %2#0[0, 0] size [2, 64] to %2#1 size [128] : memref<2x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%2#5) depends_out(%2#6) {
          %3 = ktdf.read_from_fifo %2#1 : <"L1LU" -> "SFU", 128xf16> -> tensor<2x64xf16>
          %4 = tensor.empty() : tensor<2xf16>
          %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<2x64xf16>) outs(%4 : tensor<2xf16>) {
          ^bb0(%in: f16, %out: f16):
            %6 = arith.addf %in, %out : f16
            linalg.yield %6 : f16
          } -> tensor<2xf16>
          ktdf.write_to_fifo %5, %2#2 : tensor<2xf16>, <"SFU" -> "L1SU", 2xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%2#6) depends_out(%2#7) {
          ktdf.data_transfer from %2#2 size [2] to %2#3[0] size [2] : !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, memref<2xf16, "L1">
        } {applicable_units = ["L1SU"]}
        ktdf.stage depends_in(%2#7) depends_out(none) {
          ktdf.data_transfer from %2#3[0] size [2] to %cast_2[%c0 * 2] size [2] : memref<2xf16, "L1">, memref<2xf16, strided<[1], offset: ?>, "DDR">
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
