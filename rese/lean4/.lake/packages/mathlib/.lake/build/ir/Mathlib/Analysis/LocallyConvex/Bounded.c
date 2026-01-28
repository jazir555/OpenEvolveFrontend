// Lean compiler output
// Module: Mathlib.Analysis.LocallyConvex.Bounded
// Imports: public import Init public import Mathlib.GroupTheory.GroupAction.Pointwise public import Mathlib.Analysis.LocallyConvex.Basic public import Mathlib.Analysis.LocallyConvex.BalancedCoreHull public import Mathlib.Analysis.Seminorm public import Mathlib.LinearAlgebra.Basis.VectorSpace public import Mathlib.Topology.Bornology.Basic public import Mathlib.Topology.Algebra.IsUniformGroup.Basic public import Mathlib.Topology.UniformSpace.Cauchy
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
LEAN_EXPORT lean_object* lp_mathlib_Bornology_vonNBornology(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bornology_vonNBornology___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bornology_vonNBornology(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_box(0);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bornology_vonNBornology___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Bornology_vonNBornology(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_LocallyConvex_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_LocallyConvex_BalancedCoreHull(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Seminorm(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Basis_VectorSpace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Bornology_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_IsUniformGroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_Cauchy(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_LocallyConvex_Bounded(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_LocallyConvex_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_LocallyConvex_BalancedCoreHull(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Seminorm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Basis_VectorSpace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Bornology_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_IsUniformGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_Cauchy(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
