// Lean compiler output
// Module: Mathlib.Topology.MetricSpace.ProperSpace.Lemmas
// Imports: public import Init public import Mathlib.Topology.Order.Compact public import Mathlib.Topology.MetricSpace.Bounded public import Mathlib.Topology.Order.IntermediateValue public import Mathlib.Topology.Order.LocalExtr public import Mathlib.Topology.Maps.Proper.CompactlyGenerated
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_Compact(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Bounded(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_IntermediateValue(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_LocalExtr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Maps_Proper_CompactlyGenerated(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_ProperSpace_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_Compact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_IntermediateValue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_LocalExtr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Maps_Proper_CompactlyGenerated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
