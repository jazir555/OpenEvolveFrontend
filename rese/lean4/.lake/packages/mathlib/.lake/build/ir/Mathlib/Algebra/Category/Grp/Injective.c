// Lean compiler output
// Module: Mathlib.Algebra.Category.Grp.Injective
// Imports: public import Init public import Mathlib.Algebra.Category.Grp.ZModuleEquivalence public import Mathlib.Algebra.Category.ModuleCat.Injective public import Mathlib.Algebra.EuclideanDomain.Int public import Mathlib.GroupTheory.Divisible public import Mathlib.RingTheory.PrincipalIdealDomain
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
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Grp_ZModuleEquivalence(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Injective(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Divisible(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_PrincipalIdealDomain(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_Grp_Injective(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Grp_ZModuleEquivalence(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Injective(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Divisible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_PrincipalIdealDomain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
