// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (0, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<() -> ()>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @gather_with_iab_fill() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<32xindex, "DDR">
// CHECK-NEXT:       %[[ALLOC_1:.*]] = memref.alloc() : memref<32xindex, "IAB">
// CHECK-NEXT:       %[[ALLOC_2:.*]] = memref.alloc() : memref<64x64xf16, "DDR">
// CHECK-NEXT:       %[[ALLOC_3:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       affine.for %[[VAL_1:.*]] = 0 to 32 {
// CHECK-NEXT:         %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-NEXT:         scf.if %[[CMPI_0]] {
// CHECK-NEXT:           agen.composite_load_and_store src:%[[ALLOC_0]]{{\[}}%[[CONSTANT_0]]] dst:%[[ALLOC_1]]{{\[}}%[[CONSTANT_0]]]
// CHECK-NEXT:            time_symbols(), load_iv(%[[VAL_2:.*]]:vector<1xindex>)
// CHECK-NEXT:            {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_4]], load_time_addr_map = #[[$ATTR_0]], store_order = #[[$ATTR_0]], store_set = #[[$ATTR_4]], store_time_addr_map = #[[$ATTR_0]], time_order = #[[$ATTR_0]], time_set = #[[$ATTR_5]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             agen.yield
// CHECK-NEXT:           } : memref<32xindex, "DDR">, memref<32xindex, "IAB">
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:         %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:         dataflow.sync_send %[[QUERY_MAP_0]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:         agen.composite_indirect_load_and_store indirect_src:%[[ALLOC_1]]{{\[}}%[[VAL_1]]] direct_src:%[[ALLOC_2]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] direct_dst:%[[ALLOC_3]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK-NEXT:          time_symbols(), load_iv(%[[VAL_3:.*]]:vector<64xf16>)
// CHECK-NEXT:          {load_direct_time_addr_map = #[[$ATTR_1]], load_indirect_time_addr_map = #[[$ATTR_0]], load_order = #[[$ATTR_2]], load_set = #[[$ATTR_6]], store_direct_time_addr_map = #[[$ATTR_1]], store_indirect_time_addr_map = #[[$ATTR_3]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_6]], time_order = #[[$ATTR_0]], time_set = #[[$ATTR_4]]}
// CHECK-NEXT:         {
// CHECK-NEXT:           agen.yield
// CHECK-NEXT:         } : memref<32xindex, "IAB">, memref<64x64xf16, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @scatter_with_iab_fill() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<32xindex, "DDR">
// CHECK-NEXT:       %[[ALLOC_1:.*]] = memref.alloc() : memref<32xindex, "IAB">
// CHECK-NEXT:       %[[ALLOC_2:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       %[[ALLOC_3:.*]] = memref.alloc() : memref<64x64xf16, "DDR">
// CHECK-NEXT:       agen.composite_load_and_store src:%[[ALLOC_0]]{{\[}}%[[CONSTANT_1]]] dst:%[[ALLOC_1]]{{\[}}%[[CONSTANT_1]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_1:.*]]:vector<1xindex>)
// CHECK-NEXT:        {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_4]], load_time_addr_map = #[[$ATTR_0]], store_order = #[[$ATTR_0]], store_set = #[[$ATTR_4]], store_time_addr_map = #[[$ATTR_0]], time_order = #[[$ATTR_0]], time_set = #[[$ATTR_5]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<32xindex, "DDR">, memref<32xindex, "IAB">
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_0]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       agen.composite_indirect_load_and_store direct_src:%[[ALLOC_2]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]] indirect_dst:%[[ALLOC_1]]{{\[}}%[[CONSTANT_0]]] direct_dst:%[[ALLOC_3]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_1]]:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ATTR_1]], load_indirect_time_addr_map = #[[$ATTR_3]], load_order = #[[$ATTR_2]], load_set = #[[$ATTR_6]], store_direct_time_addr_map = #[[$ATTR_1]], store_indirect_time_addr_map = #[[$ATTR_0]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_6]], time_order = #[[$ATTR_0]], time_set = #[[$ATTR_4]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<1x64xf16, "L1">, memref<32xindex, "IAB">, memref<64x64xf16, "DDR">
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @iab_fill_gather_and_scatter() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<32xindex, "DDR">
// CHECK-NEXT:       %[[ALLOC_1:.*]] = memref.alloc() : memref<32xindex, "IAB">
// CHECK-NEXT:       %[[ALLOC_2:.*]] = memref.alloc() : memref<64x64xf16, "DDR">
// CHECK-NEXT:       %[[ALLOC_3:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       agen.composite_load_and_store src:%[[ALLOC_0]]{{\[}}%[[CONSTANT_1]]] dst:%[[ALLOC_1]]{{\[}}%[[CONSTANT_1]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_1:.*]]:vector<1xindex>)
// CHECK-NEXT:        {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_4]], load_time_addr_map = #[[$ATTR_0]], store_order = #[[$ATTR_0]], store_set = #[[$ATTR_4]], store_time_addr_map = #[[$ATTR_0]], time_order = #[[$ATTR_0]], time_set = #[[$ATTR_5]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<32xindex, "DDR">, memref<32xindex, "IAB">
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_0]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       agen.composite_indirect_load_and_store indirect_src:%[[ALLOC_1]]{{\[}}%[[CONSTANT_0]]] direct_src:%[[ALLOC_2]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]] direct_dst:%[[ALLOC_3]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_1]]:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ATTR_1]], load_indirect_time_addr_map = #[[$ATTR_0]], load_order = #[[$ATTR_2]], load_set = #[[$ATTR_6]], store_direct_time_addr_map = #[[$ATTR_1]], store_indirect_time_addr_map = #[[$ATTR_3]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_6]], time_order = #[[$ATTR_0]], time_set = #[[$ATTR_4]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<32xindex, "IAB">, memref<64x64xf16, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_2:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:       %[[ALLOC_4:.*]] = memref.alloc() : memref<32xindex, "DDR">
// CHECK-NEXT:       %[[ALLOC_5:.*]] = memref.alloc() : memref<32xindex, "IAB">
// CHECK-NEXT:       %[[ALLOC_6:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       %[[ALLOC_7:.*]] = memref.alloc() : memref<64x64xf16, "DDR">
// CHECK-NEXT:       agen.composite_load_and_store src:%[[ALLOC_4]]{{\[}}%[[CONSTANT_1]]] dst:%[[ALLOC_5]]{{\[}}%[[CONSTANT_1]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_3:.*]]:vector<1xindex>)
// CHECK-NEXT:        {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_4]], load_time_addr_map = #[[$ATTR_0]], store_order = #[[$ATTR_0]], store_set = #[[$ATTR_4]], store_time_addr_map = #[[$ATTR_0]], time_order = #[[$ATTR_0]], time_set = #[[$ATTR_5]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<32xindex, "DDR">, memref<32xindex, "IAB">
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_2]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_1]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       agen.composite_indirect_load_and_store direct_src:%[[ALLOC_6]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]] indirect_dst:%[[ALLOC_5]]{{\[}}%[[CONSTANT_0]]] direct_dst:%[[ALLOC_7]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_3]]:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ATTR_1]], load_indirect_time_addr_map = #[[$ATTR_3]], load_order = #[[$ATTR_2]], load_set = #[[$ATTR_6]], store_direct_time_addr_map = #[[$ATTR_1]], store_indirect_time_addr_map = #[[$ATTR_0]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_6]], time_order = #[[$ATTR_0]], time_set = #[[$ATTR_4]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<1x64xf16, "L1">, memref<32xindex, "IAB">, memref<64x64xf16, "DDR">
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Verify that when an IAB fill (ktdf.data_transfer DDR→IAB) and an indirect
// load (ktdf.ind_data_transfer) coexist in the same execute_on body, both are
// lowered correctly and no ktdf ops survive.  A self-sync is emitted after
// the IAB fill and before the composite_indirect_load_and_store:
//
//   1. gather_with_iab_fill: the IAB is first populated from DDR via a
//      conditional data_transfer (only on the first loop iteration), then used
//      as the indirect source for a gather to a local staging buffer.  The
//      self-sync (MNILU→MNILU) is placed inside the loop, after the fill and
//      before the indirect op.
//
//   2. scatter_with_iab_fill: same pattern on the scatter side — the IAB is
//      filled first, then a self-sync (MNISU→MNISU) is emitted, then the
//      indirect scatter.
//
//   3. iab_fill_gather_and_scatter: a single function contains an IAB fill,
//      an indirect gather (MNILU), and an indirect scatter (MNISU).  Each
//      program_unit emits its own self-sync after its IAB fill and before its
//      composite_indirect_load_and_store.

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // -------------------------------------------------------------------
  // Gather with IAB fill: fill the IAB from DDR on the first iteration
  // of a loop, then use it to gather a [1, 64] tile from DDR into L1.
  // -------------------------------------------------------------------
  func.func @gather_with_iab_fill() attributes {grid = [2]} {
    %mnilu0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %mnilu1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_mnilu = uniform.def_immutable_mapping([%c0 -> %mnilu0], [%c1 -> %mnilu1]) : index
    %u_mnilu   = uniform.query_map(map:%map_mnilu, key:%tile_id) : index

    ktdf_lowering.execute_on %u_mnilu {
      %addr_buf = memref.alloc() : memref<32xindex, "DDR">
      %iab      = memref.alloc() : memref<32xindex, "IAB">
      %data     = memref.alloc() : memref<64x64xf16, "DDR">
      %staging  = memref.alloc() : memref<1x64xf16, "L1">
      affine.for %iv = 0 to 32 {
        // Fill the IAB from the address buffer on the first iteration only.
        %eq0 = arith.cmpi eq, %iv, %c0 : index
        scf.if %eq0 {
          ktdf.data_transfer
              from %addr_buf[%c0] size [32]
              to   %iab[%c0]      size [32]
              : memref<32xindex, "DDR">, memref<32xindex, "IAB">
        }
        // Use the filled IAB to gather a tile from DDR.
        ktdf.ind_data_transfer
            ind_src = %iab[%iv]
            dir_src = %data[%c0, %c0] size [1, 64]
            ind_dst = none
            dir_dst = %staging[%c0, %c0] size [1, 64]
            : memref<32xindex, "IAB">,
              memref<64x64xf16, "DDR">,
              none,
              memref<1x64xf16, "L1">
      }
    }
    return
  }

  // -------------------------------------------------------------------
  // Scatter with IAB fill: fill the IAB from DDR unconditionally, then
  // use it to scatter a [1, 64] tile from L1 out to DDR.
  // -------------------------------------------------------------------
  func.func @scatter_with_iab_fill() attributes {grid = [2]} {
    %mnisu0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
    %mnisu1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_mnisu = uniform.def_immutable_mapping([%c0 -> %mnisu0], [%c1 -> %mnisu1]) : index
    %u_mnisu   = uniform.query_map(map:%map_mnisu, key:%tile_id) : index

    ktdf_lowering.execute_on %u_mnisu {
      %addr_buf = memref.alloc() : memref<32xindex, "DDR">
      %iab      = memref.alloc() : memref<32xindex, "IAB">
      %staging  = memref.alloc() : memref<1x64xf16, "L1">
      %dst      = memref.alloc() : memref<64x64xf16, "DDR">
      // Fill the IAB once.
      ktdf.data_transfer
          from %addr_buf[%c0] size [32]
          to   %iab[%c0]      size [32]
          : memref<32xindex, "DDR">, memref<32xindex, "IAB">
      // Scatter using the IAB-driven destination address.
      ktdf.ind_data_transfer
          ind_src = none
          dir_src = %staging[%c0, %c0] size [1, 64]
          ind_dst = %iab[%c1]
          dir_dst = %dst[%c0, %c0]     size [1, 64]
          : none,
            memref<1x64xf16, "L1">,
            memref<32xindex, "IAB">,
            memref<64x64xf16, "DDR">
    }
    return
  }

  // -------------------------------------------------------------------
  // IAB fill + gather + scatter in a single function: both indirect load
  // and indirect store are present.  The IAB is filled once from DDR,
  // then the MNILU block uses it as the indirect source (gather) and the
  // MNISU block uses it as the indirect destination (scatter).
  // -------------------------------------------------------------------
  func.func @iab_fill_gather_and_scatter() attributes {grid = [2]} {
    %mnilu0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %mnilu1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %mnisu0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
    %mnisu1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_mnilu = uniform.def_immutable_mapping([%c0 -> %mnilu0], [%c1 -> %mnilu1]) : index
    %u_mnilu   = uniform.query_map(map:%map_mnilu, key:%tile_id) : index
    %map_mnisu = uniform.def_immutable_mapping([%c0 -> %mnisu0], [%c1 -> %mnisu1]) : index
    %u_mnisu   = uniform.query_map(map:%map_mnisu, key:%tile_id) : index

    ktdf_lowering.execute_on %u_mnilu, %u_mnisu {
      %addr_buf  = memref.alloc() : memref<32xindex, "DDR">
      %iab_ld    = memref.alloc() : memref<32xindex, "IAB">
      %iab_st    = memref.alloc() : memref<32xindex, "IAB">
      %src_data  = memref.alloc() : memref<64x64xf16, "DDR">
      %staging   = memref.alloc() : memref<1x64xf16, "L1">
      %dst_data  = memref.alloc() : memref<64x64xf16, "DDR">

      // MNILU: fill IAB for load, then perform indirect gather.
      ktdf_lowering.execute_on %u_mnilu {
        ktdf.data_transfer
            from %addr_buf[%c0] size [32]
            to   %iab_ld[%c0]   size [32]
            : memref<32xindex, "DDR">, memref<32xindex, "IAB">
        ktdf.ind_data_transfer
            ind_src = %iab_ld[%c1]
            dir_src = %src_data[%c0, %c0] size [1, 64]
            ind_dst = none
            dir_dst = %staging[%c0, %c0]  size [1, 64]
            : memref<32xindex, "IAB">,
              memref<64x64xf16, "DDR">,
              none,
              memref<1x64xf16, "L1">
      }

      // MNISU: fill IAB for store, then perform indirect scatter.
      ktdf_lowering.execute_on %u_mnisu {
        ktdf.data_transfer
            from %addr_buf[%c0] size [32]
            to   %iab_st[%c0]   size [32]
            : memref<32xindex, "DDR">, memref<32xindex, "IAB">
        ktdf.ind_data_transfer
            ind_src = none
            dir_src = %staging[%c0, %c0] size [1, 64]
            ind_dst = %iab_st[%c1]
            dir_dst = %dst_data[%c0, %c0] size [1, 64]
            : none,
              memref<1x64xf16, "L1">,
              memref<32xindex, "IAB">,
              memref<64x64xf16, "DDR">
      }
    }
    return
  }
}
