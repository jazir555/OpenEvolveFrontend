// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Presheaf.Sheafification
// Imports: public import Init public import Mathlib.Algebra.Category.ModuleCat.Presheaf.Abelian public import Mathlib.Algebra.Category.ModuleCat.Presheaf.Sheafify public import Mathlib.Algebra.Category.ModuleCat.Presheaf.Limits public import Mathlib.Algebra.Category.ModuleCat.Sheaf.Limits public import Mathlib.CategoryTheory.Sites.LocallyBijective public import Mathlib.CategoryTheory.Sites.Sheafification public import Mathlib.CategoryTheory.Functor.ReflectsIso.Balanced
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
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf_Abelian(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf_Sheafify(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf_Limits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Sheaf_Limits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_LocallyBijective(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_Sheafification(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Functor_ReflectsIso_Balanced(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf_Sheafification(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf_Abelian(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf_Sheafify(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf_Limits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Sheaf_Limits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Sites_LocallyBijective(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Sites_Sheafification(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Functor_ReflectsIso_Balanced(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
