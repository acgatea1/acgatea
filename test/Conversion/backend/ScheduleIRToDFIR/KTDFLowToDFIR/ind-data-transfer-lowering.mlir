// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Verify that ktdf.ind_data_transfer is lowered to
// agen.composite_indirect_load_and_store for all three modes:
//   1. Gather → memref: IAB drives the source address; result goes into a
//      local staging buffer.  No body beyond agen.yield.
//   2. Gather → FIFO: IAB drives the source address; result is forwarded
//      via dataflow.send in the body.
//   3. Scatter: IAB drives the destination address; source is a local buffer.
//      No body beyond agen.yield.
//
// In all cases the original ktdf.ind_data_transfer must not survive.

// CHECK: #[[$ZERO_2D:.+]] = affine_map<(d0) -> (0, 0)>
// CHECK: #[[$ZERO_1D:.+]] = affine_map<(d0) -> (0)>
// CHECK: #[[$ID_2D:.+]]   = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$EMPTY:.+]]   = affine_map<() -> ()>
// CHECK: #[[$ID_1D:.+]]   = affine_map<(d0) -> (d0)>
// CHECK: #[[$VEC_SET:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$TIME_SET:.+]] = affine_set<(d0) : (d0 == 0)>

// ---------------------------------------------------------------------------
// Test 1: gather → memref
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ind_transfer_gather_to_memref
// CHECK:         dataflow.program_unit
// CHECK:           agen.composite_indirect_load_and_store
// CHECK-SAME:        indirect_src:{{.+}} direct_src:{{.+}} direct_dst:{{.+}}
// CHECK-NEXT:        time_symbols(), load_iv({{.+}}:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ZERO_2D]], load_indirect_time_addr_map = #[[$ZERO_1D]], load_order = #[[$ID_2D]], load_set = #[[$VEC_SET]], store_direct_time_addr_map = #[[$ZERO_2D]], store_indirect_time_addr_map = #[[$EMPTY]], store_order = #[[$ID_2D]], store_set = #[[$VEC_SET]], time_order = #[[$ID_1D]], time_set = #[[$TIME_SET]]}
// CHECK-NEXT:      {
// CHECK-NEXT:        agen.yield
// CHECK-NEXT:      }
// CHECK-NOT:       ktdf.ind_data_transfer

// ---------------------------------------------------------------------------
// Test 2: gather → FIFO
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ind_transfer_gather_to_fifo
// CHECK:         dataflow.program_unit
// CHECK:           agen.composite_indirect_load_and_store
// CHECK-SAME:        indirect_src:{{.+}} direct_src:{{.+}} direct_dst:{{.+}}
// CHECK-NEXT:        time_symbols(), load_iv([[IV:%[a-z0-9_]+]]:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ZERO_2D]], load_indirect_time_addr_map = #[[$ZERO_1D]], load_order = #[[$ID_2D]], load_set = #[[$VEC_SET]], store_direct_time_addr_map = #[[$ZERO_1D]], store_indirect_time_addr_map = #[[$EMPTY]], store_order = #[[$ID_2D]], store_set = #[[$VEC_SET]], time_order = #[[$ID_1D]], time_set = #[[$TIME_SET]]}
// CHECK-NEXT:      {
// CHECK-NEXT:        dataflow.send %{{.+}}, [[IV]] : vector<64xf16>
// CHECK-NEXT:        agen.yield
// CHECK-NEXT:      }
// CHECK-NOT:       ktdf.ind_data_transfer

// ---------------------------------------------------------------------------
// Test 3: scatter
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ind_transfer_scatter
// CHECK:         dataflow.program_unit
// CHECK:           agen.composite_indirect_load_and_store
// CHECK-SAME:        direct_src:{{.+}} indirect_dst:{{.+}} direct_dst:{{.+}}
// CHECK-NEXT:        time_symbols(), load_iv({{.+}}:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ZERO_2D]], load_indirect_time_addr_map = #[[$EMPTY]], load_order = #[[$ID_2D]], load_set = #[[$VEC_SET]], store_direct_time_addr_map = #[[$ZERO_2D]], store_indirect_time_addr_map = #[[$ZERO_1D]], store_order = #[[$ID_2D]], store_set = #[[$VEC_SET]], time_order = #[[$ID_1D]], time_set = #[[$TIME_SET]]}
// CHECK-NEXT:      {
// CHECK-NEXT:        agen.yield
// CHECK-NEXT:      }
// CHECK-NOT:       ktdf.ind_data_transfer

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // -------------------------------------------------------------------
  // Gather to memref: IAB entry drives the source base address;
  // a [1, 64] tile is loaded and written into a local staging buffer.
  // -------------------------------------------------------------------
  func.func @ind_transfer_gather_to_memref() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index

    ktdf_lowering.execute_on %u_l1lu {
      %iab     = memref.alloc() : memref<32xindex, "L1">
      %data    = memref.alloc() : memref<64x64xf16, "L1">
      %staging = memref.alloc() : memref<1x64xf16, "L1">
      ktdf.ind_data_transfer
          ind_src = %iab[%c1]
          dir_src = %data[%c0, %c0] size [1, 64]
          ind_dst = none
          dir_dst = %staging[%c0, %c0] size [1, 64]
          : memref<32xindex, "L1">,
            memref<64x64xf16, "L1">,
            none,
            memref<1x64xf16, "L1">
    }
    return
  }

  // -------------------------------------------------------------------
  // Gather to FIFO: IAB entry drives the source base address; the loaded
  // vector is forwarded to the SFU via dataflow.send in the body.
  // The outer execute_on spans both L1LU and SFU so the FIFO endpoint
  // ("SFU") can be resolved.
  // -------------------------------------------------------------------
  func.func @ind_transfer_gather_to_fifo() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu, %u_sfu {
      ktdf_lowering.execute_on %u_l1lu {
        %iab  = memref.alloc() : memref<32xindex, "L1">
        %data = memref.alloc() : memref<64x64xf16, "L1">
        ktdf.ind_data_transfer
            ind_src = %iab[%c1]
            dir_src = %data[%c0, %c0] size [1, 64]
            ind_dst = none
            dir_dst = %fifo           size [64]
            : memref<32xindex, "L1">,
              memref<64x64xf16, "L1">,
              none,
              !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
    }
    return
  }

  // -------------------------------------------------------------------
  // Scatter: IAB entry drives the destination base address; a [1, 64]
  // tile is written from a local staging buffer to global memory.
  // -------------------------------------------------------------------
  func.func @ind_transfer_scatter() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index

    ktdf_lowering.execute_on %u_l1lu {
      %iab     = memref.alloc() : memref<32xindex, "L1">
      %staging = memref.alloc() : memref<1x64xf16, "L1">
      %dst     = memref.alloc() : memref<64x64xf16, "L1">
      ktdf.ind_data_transfer
          ind_src = none
          dir_src = %staging[%c0, %c0] size [1, 64]
          ind_dst = %iab[%c1]
          dir_dst = %dst[%c0, %c0]     size [1, 64]
          : none,
            memref<1x64xf16, "L1">,
            memref<32xindex, "L1">,
            memref<64x64xf16, "L1">
    }
    return
  }
}
