// Lean compiler output
// Module: Mathlib.Topology.Order.IntermediateValue
// Imports: public import Init public import Mathlib.Order.Interval.Set.Image public import Mathlib.Order.CompleteLatticeIntervals public import Mathlib.Topology.Order.DenselyOrdered public import Mathlib.Topology.Order.Monotone public import Mathlib.Topology.Connected.TotallyDisconnected
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
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_DenselyOrdered(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_Monotone(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Connected_TotallyDisconnected(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Order_IntermediateValue(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_DenselyOrdered(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_Monotone(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Connected_TotallyDisconnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
