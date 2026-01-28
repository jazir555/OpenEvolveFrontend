// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Presheaf.Limits
// Imports: public import Init public import Mathlib.Algebra.Category.ModuleCat.Presheaf public import Mathlib.Algebra.Category.ModuleCat.ChangeOfRings public import Mathlib.CategoryTheory.Limits.Preserves.Limits public import Mathlib.CategoryTheory.Limits.FunctorCategory.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_PresheafOfModules_evaluationJointlyReflectsLimits___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PresheafOfModules_evaluationJointlyReflectsLimits___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PresheafOfModules_evaluationJointlyReflectsLimits(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PresheafOfModules_evaluationJointlyReflectsLimits___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_PresheafOfModules_evaluation___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCone___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PresheafOfModules_evaluationJointlyReflectsLimits___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc(x_4);
x_6 = lp_mathlib_PresheafOfModules_evaluation___redArg(x_4);
x_7 = lp_mathlib_CategoryTheory_Functor_mapCone___redArg(x_6, x_1, x_3);
x_8 = lean_apply_3(x_2, x_4, x_7, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PresheafOfModules_evaluationJointlyReflectsLimits___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_PresheafOfModules_evaluationJointlyReflectsLimits___redArg___lam__0), 5, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PresheafOfModules_evaluationJointlyReflectsLimits(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_PresheafOfModules_evaluationJointlyReflectsLimits___redArg(x_6, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PresheafOfModules_evaluationJointlyReflectsLimits___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_PresheafOfModules_evaluationJointlyReflectsLimits(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_ChangeOfRings(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Limits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_FunctorCategory_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf_Limits(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Presheaf(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_ChangeOfRings(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Limits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_FunctorCategory_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
