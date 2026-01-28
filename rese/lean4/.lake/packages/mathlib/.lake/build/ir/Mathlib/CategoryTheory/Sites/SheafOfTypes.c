// Lean compiler output
// Module: Mathlib.CategoryTheory.Sites.SheafOfTypes
// Imports: public import Init public import Mathlib.CategoryTheory.Sites.Pretopology public import Mathlib.CategoryTheory.Sites.IsSheafFor
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_yonedaFamilyOfElements__fromCocone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_compatibleYonedaFamily__toCocone___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Over_forget___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_yonedaFamilyOfElements__fromCocone___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_ObjectProperty_00_u03b9(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_instCategoryOver___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_compatibleYonedaFamily__toCocone___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_compatibleYonedaFamily__toCocone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_yonedaFamilyOfElements__fromCocone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_compatibleYonedaFamily__toCocone___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_ctor_get(x_3, 2);
lean_inc(x_5);
x_6 = lean_apply_1(x_4, x_3);
x_7 = lean_apply_3(x_2, x_6, x_5, lean_box(0));
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_compatibleYonedaFamily__toCocone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_1);
x_5 = lp_mathlib_CategoryTheory_instCategoryOver___redArg(x_1);
x_6 = lp_mathlib_CategoryTheory_ObjectProperty_00_u03b9(lean_box(0), x_5, lean_box(0));
lean_dec_ref(x_5);
x_7 = lp_mathlib_CategoryTheory_Over_forget___redArg(x_1, x_2);
x_8 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_6, x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Presieve_compatibleYonedaFamily__toCocone___redArg___lam__0), 3, 2);
lean_closure_set(x_9, 0, x_8);
lean_closure_set(x_9, 1, x_4);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_3);
lean_ctor_set(x_10, 1, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_compatibleYonedaFamily__toCocone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Presieve_compatibleYonedaFamily__toCocone___redArg(x_2, x_3, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_yonedaFamilyOfElements__fromCocone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_2, x_3);
x_6 = lean_apply_1(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_yonedaFamilyOfElements__fromCocone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Presieve_yonedaFamilyOfElements__fromCocone___redArg(x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_yonedaFamilyOfElements__fromCocone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Presieve_yonedaFamilyOfElements__fromCocone(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_Pretopology(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_IsSheafFor(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_SheafOfTypes(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Sites_Pretopology(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Sites_IsSheafFor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
