// Lean compiler output
// Module: Mathlib.MeasureTheory.OuterMeasure.Defs
// Imports: public import Init public import Mathlib.Topology.Algebra.InfiniteSum.Defs public import Mathlib.Topology.Order.Real
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
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_OuterMeasure_instFunLikeSetENNReal(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_OuterMeasure_instFunLikeSetENNReal___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_OuterMeasure_instFunLikeSetENNReal___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, lean_box(0));
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_OuterMeasure_instFunLikeSetENNReal(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_OuterMeasure_instFunLikeSetENNReal___lam__0), 2, 0);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_InfiniteSum_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_Real(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_OuterMeasure_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_InfiniteSum_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_Real(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
