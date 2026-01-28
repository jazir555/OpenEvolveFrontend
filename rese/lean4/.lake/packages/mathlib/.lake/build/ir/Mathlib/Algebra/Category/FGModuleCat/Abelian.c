// Lean compiler output
// Module: Mathlib.Algebra.Category.FGModuleCat.Abelian
// Imports: public import Init public import Mathlib.Algebra.Category.FGModuleCat.Colimits public import Mathlib.Algebra.Category.FGModuleCat.Limits public import Mathlib.Algebra.Category.ModuleCat.Abelian public import Mathlib.CategoryTheory.Limits.Preserves.Shapes.AbelianImages
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
lean_object* lp_mathlib_CategoryTheory_Preadditive_fullSubcategory___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleCat_instAbelian___redArg(lean_object*);
lean_object* lp_mathlib_ModuleCat_instAddCommGroupHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ModuleCat_moduleCategory(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleCat_instAbelian(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGModuleCat_instAbelian___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_ModuleCat_moduleCategory(lean_box(0), x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_instAddCommGroupHom___boxed), 4, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lp_mathlib_CategoryTheory_Preadditive_fullSubcategory___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGModuleCat_instAbelian(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FGModuleCat_instAbelian___redArg(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_FGModuleCat_Colimits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_FGModuleCat_Limits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Abelian(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_AbelianImages(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_FGModuleCat_Abelian(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_FGModuleCat_Colimits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_FGModuleCat_Limits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Abelian(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_AbelianImages(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
