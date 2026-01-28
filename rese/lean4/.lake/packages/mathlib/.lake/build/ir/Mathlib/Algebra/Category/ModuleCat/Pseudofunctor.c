// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Pseudofunctor
// Imports: public import Init public import Mathlib.Algebra.Category.ModuleCat.ChangeOfRings public import Mathlib.Algebra.Category.Ring.Basic public import Mathlib.CategoryTheory.Bicategory.Functor.LocallyDiscrete public import Mathlib.CategoryTheory.Adjunction.Mates
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
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_ChangeOfRings(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Ring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Bicategory_Functor_LocallyDiscrete(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Mates(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Pseudofunctor(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_ChangeOfRings(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Bicategory_Functor_LocallyDiscrete(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Mates(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
