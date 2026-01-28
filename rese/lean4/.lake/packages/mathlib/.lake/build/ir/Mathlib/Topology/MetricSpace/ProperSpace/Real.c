// Lean compiler output
// Module: Mathlib.Topology.MetricSpace.ProperSpace.Real
// Imports: public import Init public import Mathlib.Data.Rat.Encodable public import Mathlib.Topology.MetricSpace.Isometry public import Mathlib.Topology.MetricSpace.ProperSpace public import Mathlib.Topology.Order.Compact public import Mathlib.Topology.Order.MonotoneContinuity public import Mathlib.Topology.Order.Real public import Mathlib.Topology.UniformSpace.Real
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
lean_object* initialize_mathlib_Mathlib_Data_Rat_Encodable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Isometry(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_ProperSpace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_Compact(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_MonotoneContinuity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_Real(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_Real(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_ProperSpace_Real(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Encodable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Isometry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_ProperSpace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_Compact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_MonotoneContinuity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_Real(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_Real(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
