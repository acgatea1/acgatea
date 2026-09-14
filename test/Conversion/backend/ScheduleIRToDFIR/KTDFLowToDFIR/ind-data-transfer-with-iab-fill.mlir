// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// CHECK: #[[$LAYOUT_1D:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$LAYOUT_2D:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (0, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<() -> ()>
// CHECK: #[[$IAB_SET:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @gather_with_iab_fill() attributes {grid = [2]} {
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[DDR_UNIT:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[LMV_ADDR:.*]] = dataflow.get_logical_memory_view %[[DDR_UNIT]], %[[C0]] {layout_map = #[[$LAYOUT_1D]]} : index, index, memref<32xindex>
// CHECK-NEXT:       %[[LMV_DATA:.*]] = dataflow.get_logical_memory_view %[[DDR_UNIT]], %[[C0]] {layout_map = #[[$LAYOUT_2D]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:       %[[IAB:.*]] = ktdp_lowering.construct_memory_view %[[C0]], sizes: [32], strides: [1] {coordinate_set = #[[$IAB_SET]], memory_space = "IAB"} : memref<32xindex, "IAB">
// CHECK-NEXT:       %[[ALLOC_L1:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       affine.for %[[IV:.*]] = 0 to 32 {
// CHECK-NEXT:         %[[CMPI:.*]] = arith.cmpi eq, %[[IV]], %[[C0]] : index
// CHECK-NEXT:         scf.if %[[CMPI]] {
// CHECK-NEXT:           agen.composite_load_and_store src:%[[LMV_ADDR]]{{\[}}%[[C0]]] dst:%[[IAB]]{{\[}}%[[C0]]]
// CHECK-NEXT:            time_symbols(), load_iv(%[[VAL_1:.*]]:vector<1xindex>)
// CHECK-NEXT:            {load_order = #[[$LAYOUT_1D]], load_set = #[[$ATTR_3]], load_time_addr_map = #[[$LAYOUT_1D]], store_order = #[[$LAYOUT_1D]], store_set = #[[$ATTR_3]], store_time_addr_map = #[[$LAYOUT_1D]], time_order = #[[$LAYOUT_1D]], time_set = #[[$IAB_SET]]}
// CHECK-NEXT:           {
// CHECK-NEXT:             agen.yield
// CHECK-NEXT:           } : memref<32xindex>, memref<32xindex, "IAB">
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[DEF_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:         %[[QUERY_MAP:.*]] = uniform.query_map(map:%[[DEF_MAP]], key:%[[VAL_0]]) : index
// CHECK-NEXT:         dataflow.sync_send %[[QUERY_MAP]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:         agen.composite_indirect_load_and_store indirect_src:%[[IAB]]{{\[}}%[[IV]]] direct_src:%[[LMV_DATA]]{{\[}}%[[C0]], %[[C0]]] direct_dst:%[[ALLOC_L1]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:          time_symbols(), load_iv(%[[VAL_1]]:vector<64xf16>)
// CHECK-NEXT:          {load_direct_time_addr_map = #[[$ATTR_0]], load_indirect_time_addr_map = #[[$LAYOUT_1D]], load_order = #[[$ATTR_2]], load_set = #[[$ATTR_4]], store_direct_time_addr_map = #[[$ATTR_0]], store_indirect_time_addr_map = #[[$ATTR_1]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_4]], time_order = #[[$LAYOUT_1D]], time_set = #[[$ATTR_3]]}
// CHECK-NEXT:         {
// CHECK-NEXT:           agen.yield
// CHECK-NEXT:         } : memref<32xindex, "IAB">, memref<64x64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @scatter_with_iab_fill() attributes {grid = [2]} {
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
// CHECK-NEXT:     %[[DDR_UNIT:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[LMV_ADDR:.*]] = dataflow.get_logical_memory_view %[[DDR_UNIT]], %[[C0]] {layout_map = #[[$LAYOUT_1D]]} : index, index, memref<32xindex>
// CHECK-NEXT:       %[[LMV_DST:.*]] = dataflow.get_logical_memory_view %[[DDR_UNIT]], %[[C0]] {layout_map = #[[$LAYOUT_2D]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:       %[[IAB:.*]] = ktdp_lowering.construct_memory_view %[[C0]], sizes: [32], strides: [1] {coordinate_set = #[[$IAB_SET]], memory_space = "IAB"} : memref<32xindex, "IAB">
// CHECK-NEXT:       %[[ALLOC_L1:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       agen.composite_load_and_store src:%[[LMV_ADDR]]{{\[}}%[[C0]]] dst:%[[IAB]]{{\[}}%[[C0]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_1:.*]]:vector<1xindex>)
// CHECK-NEXT:        {load_order = #[[$LAYOUT_1D]], load_set = #[[$ATTR_3]], load_time_addr_map = #[[$LAYOUT_1D]], store_order = #[[$LAYOUT_1D]], store_set = #[[$ATTR_3]], store_time_addr_map = #[[$LAYOUT_1D]], time_order = #[[$LAYOUT_1D]], time_set = #[[$IAB_SET]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<32xindex>, memref<32xindex, "IAB">
// CHECK-NEXT:       %[[DEF_MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP:.*]] = uniform.query_map(map:%[[DEF_MAP]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       agen.composite_indirect_load_and_store direct_src:%[[ALLOC_L1]]{{\[}}%[[C0]], %[[C0]]] indirect_dst:%[[IAB]]{{\[}}%[[C1]]] direct_dst:%[[LMV_DST]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_1]]:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ATTR_0]], load_indirect_time_addr_map = #[[$ATTR_1]], load_order = #[[$ATTR_2]], load_set = #[[$ATTR_4]], store_direct_time_addr_map = #[[$ATTR_0]], store_indirect_time_addr_map = #[[$LAYOUT_1D]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_4]], time_order = #[[$LAYOUT_1D]], time_set = #[[$ATTR_3]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<1x64xf16, "L1">, memref<32xindex, "IAB">, memref<64x64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @iab_fill_gather_and_scatter() attributes {grid = [2]} {
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
// CHECK-NEXT:     %[[DDR_UNIT:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[LMV_ADDR_LD:.*]] = dataflow.get_logical_memory_view %[[DDR_UNIT]], %[[C0]] {layout_map = #[[$LAYOUT_1D]]} : index, index, memref<32xindex>
// CHECK-NEXT:       %[[LMV_SRC:.*]] = dataflow.get_logical_memory_view %[[DDR_UNIT]], %[[C0]] {layout_map = #[[$LAYOUT_2D]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:       %[[ALLOC_L1:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       %[[IAB_LD:.*]] = ktdp_lowering.construct_memory_view %[[C0]], sizes: [32], strides: [1] {coordinate_set = #[[$IAB_SET]], memory_space = "IAB"} : memref<32xindex, "IAB">
// CHECK-NEXT:       agen.composite_load_and_store src:%[[LMV_ADDR_LD]]{{\[}}%[[C0]]] dst:%[[IAB_LD]]{{\[}}%[[C0]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_1:.*]]:vector<1xindex>)
// CHECK-NEXT:        {load_order = #[[$LAYOUT_1D]], load_set = #[[$ATTR_3]], load_time_addr_map = #[[$LAYOUT_1D]], store_order = #[[$LAYOUT_1D]], store_set = #[[$ATTR_3]], store_time_addr_map = #[[$LAYOUT_1D]], time_order = #[[$LAYOUT_1D]], time_set = #[[$IAB_SET]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<32xindex>, memref<32xindex, "IAB">
// CHECK-NEXT:       %[[DEF_MAP_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_MAP_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_0]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       agen.composite_indirect_load_and_store indirect_src:%[[IAB_LD]]{{\[}}%[[C1]]] direct_src:%[[LMV_SRC]]{{\[}}%[[C0]], %[[C0]]] direct_dst:%[[ALLOC_L1]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_1]]:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ATTR_0]], load_indirect_time_addr_map = #[[$LAYOUT_1D]], load_order = #[[$ATTR_2]], load_set = #[[$ATTR_4]], store_direct_time_addr_map = #[[$ATTR_0]], store_indirect_time_addr_map = #[[$ATTR_1]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_4]], time_order = #[[$LAYOUT_1D]], time_set = #[[$ATTR_3]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<32xindex, "IAB">, memref<64x64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_2:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:       %[[LMV_ADDR_ST:.*]] = dataflow.get_logical_memory_view %[[DDR_UNIT]], %[[C0]] {layout_map = #[[$LAYOUT_1D]]} : index, index, memref<32xindex>
// CHECK-NEXT:       %[[LMV_DST:.*]] = dataflow.get_logical_memory_view %[[DDR_UNIT]], %[[C0]] {layout_map = #[[$LAYOUT_2D]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:       %[[ALLOC_L1_ST:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       %[[IAB_ST:.*]] = ktdp_lowering.construct_memory_view %[[C0]], sizes: [32], strides: [1] {coordinate_set = #[[$IAB_SET]], memory_space = "IAB"} : memref<32xindex, "IAB">
// CHECK-NEXT:       agen.composite_load_and_store src:%[[LMV_ADDR_ST]]{{\[}}%[[C0]]] dst:%[[IAB_ST]]{{\[}}%[[C0]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_3:.*]]:vector<1xindex>)
// CHECK-NEXT:        {load_order = #[[$LAYOUT_1D]], load_set = #[[$ATTR_3]], load_time_addr_map = #[[$LAYOUT_1D]], store_order = #[[$LAYOUT_1D]], store_set = #[[$ATTR_3]], store_time_addr_map = #[[$LAYOUT_1D]], time_order = #[[$LAYOUT_1D]], time_set = #[[$IAB_SET]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<32xindex>, memref<32xindex, "IAB">
// CHECK-NEXT:       %[[DEF_MAP_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_MAP_1]], key:%[[VAL_2]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_1]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       agen.composite_indirect_load_and_store direct_src:%[[ALLOC_L1_ST]]{{\[}}%[[C0]], %[[C0]]] indirect_dst:%[[IAB_ST]]{{\[}}%[[C1]]] direct_dst:%[[LMV_DST]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_3]]:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ATTR_0]], load_indirect_time_addr_map = #[[$ATTR_1]], load_order = #[[$ATTR_2]], load_set = #[[$ATTR_4]], store_direct_time_addr_map = #[[$ATTR_0]], store_indirect_time_addr_map = #[[$LAYOUT_1D]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_4]], time_order = #[[$LAYOUT_1D]], time_set = #[[$ATTR_3]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<1x64xf16, "L1">, memref<32xindex, "IAB">, memref<64x64xf16>
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
//
// DDR buffers arrive as ktdp.construct_memory_view chains (outside execute_on)
// and are lowered to dataflow.get_logical_memory_view inside the program_unit.
// IAB buffers use ktdp_lowering.construct_memory_view and pass through unchanged.
// L1 buffers remain as memref.alloc.

#ddr_coord_1d = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#iab_coord_1d = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#ddr_coord_2d = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>

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

    %addr_buf_mv  = ktdp.construct_memory_view %c0, sizes: [32], strides: [1]
                        {coordinate_set = #ddr_coord_1d, memory_space = #ktdp.memory_space<global>}
                        : memref<32xindex>
    %addr_buf_msc = memref.memory_space_cast %addr_buf_mv : memref<32xindex> to memref<32xindex, "DDR">
    %addr_buf     = memref.reinterpret_cast %addr_buf_msc to offset: [%c0], sizes: [32], strides: [1]
                        : memref<32xindex, "DDR"> to memref<32xindex, strided<[1], offset: ?>, "DDR">

    %data_mv  = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1]
                    {coordinate_set = #ddr_coord_2d, memory_space = #ktdp.memory_space<global>}
                    : memref<64x64xf16>
    %data_msc = memref.memory_space_cast %data_mv : memref<64x64xf16> to memref<64x64xf16, "DDR">
    %data     = memref.reinterpret_cast %data_msc to offset: [%c0], sizes: [64, 64], strides: [64, 1]
                    : memref<64x64xf16, "DDR"> to memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">

    ktdf_lowering.execute_on %u_mnilu {
      %iab     = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
                     {coordinate_set = #iab_coord_1d, memory_space = "IAB"}
                     : memref<32xindex, "IAB">
      %staging = memref.alloc() : memref<1x64xf16, "L1">
      affine.for %iv = 0 to 32 {
        // Fill the IAB from the address buffer on the first iteration only.
        %eq0 = arith.cmpi eq, %iv, %c0 : index
        scf.if %eq0 {
          ktdf.data_transfer
              from %addr_buf[%c0] size [32]
              to   %iab[%c0]      size [32]
              : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, "IAB">
        }
        // Use the filled IAB to gather a tile from DDR.
        ktdf.ind_data_transfer
            ind_src = %iab[%iv]
            dir_src = %data[%c0, %c0] size [1, 64]
            ind_dst = none
            dir_dst = %staging[%c0, %c0] size [1, 64]
            : memref<32xindex, "IAB">,
              memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">,
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

    %addr_buf_mv  = ktdp.construct_memory_view %c0, sizes: [32], strides: [1]
                        {coordinate_set = #ddr_coord_1d, memory_space = #ktdp.memory_space<global>}
                        : memref<32xindex>
    %addr_buf_msc = memref.memory_space_cast %addr_buf_mv : memref<32xindex> to memref<32xindex, "DDR">
    %addr_buf     = memref.reinterpret_cast %addr_buf_msc to offset: [%c0], sizes: [32], strides: [1]
                        : memref<32xindex, "DDR"> to memref<32xindex, strided<[1], offset: ?>, "DDR">

    %dst_mv  = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1]
                   {coordinate_set = #ddr_coord_2d, memory_space = #ktdp.memory_space<global>}
                   : memref<64x64xf16>
    %dst_msc = memref.memory_space_cast %dst_mv : memref<64x64xf16> to memref<64x64xf16, "DDR">
    %dst     = memref.reinterpret_cast %dst_msc to offset: [%c0], sizes: [64, 64], strides: [64, 1]
                   : memref<64x64xf16, "DDR"> to memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">

    ktdf_lowering.execute_on %u_mnisu {
      %iab     = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
                     {coordinate_set = #iab_coord_1d, memory_space = "IAB"}
                     : memref<32xindex, "IAB">
      %staging = memref.alloc() : memref<1x64xf16, "L1">
      // Fill the IAB once.
      ktdf.data_transfer
          from %addr_buf[%c0] size [32]
          to   %iab[%c0]      size [32]
          : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, "IAB">
      // Scatter using the IAB-driven destination address.
      ktdf.ind_data_transfer
          ind_src = none
          dir_src = %staging[%c0, %c0] size [1, 64]
          ind_dst = %iab[%c1]
          dir_dst = %dst[%c0, %c0]     size [1, 64]
          : none,
            memref<1x64xf16, "L1">,
            memref<32xindex, "IAB">,
            memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">
    }
    return
  }

  // -------------------------------------------------------------------
  // IAB fill + gather + scatter in a single function: both indirect load
  // and indirect store are present.  The shared addr_buf is from DDR,
  // the MNILU block uses it to fill its IAB for gather, and the MNISU
  // block uses it to fill its IAB for scatter.
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

    %addr_buf_mv  = ktdp.construct_memory_view %c0, sizes: [32], strides: [1]
                        {coordinate_set = #ddr_coord_1d, memory_space = #ktdp.memory_space<global>}
                        : memref<32xindex>
    %addr_buf_msc = memref.memory_space_cast %addr_buf_mv : memref<32xindex> to memref<32xindex, "DDR">
    %addr_buf     = memref.reinterpret_cast %addr_buf_msc to offset: [%c0], sizes: [32], strides: [1]
                        : memref<32xindex, "DDR"> to memref<32xindex, strided<[1], offset: ?>, "DDR">

    %src_data_mv  = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1]
                        {coordinate_set = #ddr_coord_2d, memory_space = #ktdp.memory_space<global>}
                        : memref<64x64xf16>
    %src_data_msc = memref.memory_space_cast %src_data_mv : memref<64x64xf16> to memref<64x64xf16, "DDR">
    %src_data     = memref.reinterpret_cast %src_data_msc to offset: [%c0], sizes: [64, 64], strides: [64, 1]
                        : memref<64x64xf16, "DDR"> to memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">

    %dst_data_mv  = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1]
                        {coordinate_set = #ddr_coord_2d, memory_space = #ktdp.memory_space<global>}
                        : memref<64x64xf16>
    %dst_data_msc = memref.memory_space_cast %dst_data_mv : memref<64x64xf16> to memref<64x64xf16, "DDR">
    %dst_data     = memref.reinterpret_cast %dst_data_msc to offset: [%c0], sizes: [64, 64], strides: [64, 1]
                        : memref<64x64xf16, "DDR"> to memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">

    ktdf_lowering.execute_on %u_mnilu, %u_mnisu {
      %staging = memref.alloc() : memref<1x64xf16, "L1">

      // MNILU: fill IAB for load, then perform indirect gather.
      ktdf_lowering.execute_on %u_mnilu {
        %iab_ld = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
                      {coordinate_set = #iab_coord_1d, memory_space = "IAB"}
                      : memref<32xindex, "IAB">
        ktdf.data_transfer
            from %addr_buf[%c0] size [32]
            to   %iab_ld[%c0]   size [32]
            : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, "IAB">
        ktdf.ind_data_transfer
            ind_src = %iab_ld[%c1]
            dir_src = %src_data[%c0, %c0] size [1, 64]
            ind_dst = none
            dir_dst = %staging[%c0, %c0]  size [1, 64]
            : memref<32xindex, "IAB">,
              memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">,
              none,
              memref<1x64xf16, "L1">
      }

      // MNISU: fill IAB for store, then perform indirect scatter.
      ktdf_lowering.execute_on %u_mnisu {
        %iab_st = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
                      {coordinate_set = #iab_coord_1d, memory_space = "IAB"}
                      : memref<32xindex, "IAB">
        ktdf.data_transfer
            from %addr_buf[%c0] size [32]
            to   %iab_st[%c0]   size [32]
            : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, "IAB">
        ktdf.ind_data_transfer
            ind_src = none
            dir_src = %staging[%c0, %c0] size [1, 64]
            ind_dst = %iab_st[%c1]
            dir_dst = %dst_data[%c0, %c0] size [1, 64]
            : none,
              memref<1x64xf16, "L1">,
              memref<32xindex, "IAB">,
              memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">
      }
    }
    return
  }
}
