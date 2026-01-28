// Lean compiler output
// Module: Mathlib.Topology.MetricSpace.Algebra
// Imports: public import Init public import Mathlib.Topology.Algebra.MulAction public import Mathlib.Topology.Algebra.Order.Field public import Mathlib.Topology.Algebra.SeparationQuotient.Basic public import Mathlib.Topology.Algebra.UniformMulAction public import Mathlib.Topology.MetricSpace.Lipschitz
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
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_MulAction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Order_Field(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_SeparationQuotient_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_UniformMulAction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Lipschitz(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Algebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_MulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Order_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_SeparationQuotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_UniformMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Lipschitz(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
