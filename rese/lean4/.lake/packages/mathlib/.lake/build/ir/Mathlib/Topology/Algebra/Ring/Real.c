// Lean compiler output
// Module: Mathlib.Topology.Algebra.Ring.Real
// Imports: public import Init public import Mathlib.Data.EReal.Operations public import Mathlib.Topology.Algebra.Order.Field public import Mathlib.Topology.Algebra.IsUniformGroup.Defs public import Mathlib.Topology.Bornology.Real public import Mathlib.Topology.Instances.Int public import Mathlib.Topology.Order.MonotoneContinuity public import Mathlib.Topology.Order.Real public import Mathlib.Topology.UniformSpace.Real
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
lean_object* initialize_mathlib_Mathlib_Data_EReal_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Order_Field(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_IsUniformGroup_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Bornology_Real(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Instances_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_MonotoneContinuity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_Real(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_Real(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Ring_Real(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_EReal_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Order_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_IsUniformGroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Bornology_Real(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Instances_Int(builtin);
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
