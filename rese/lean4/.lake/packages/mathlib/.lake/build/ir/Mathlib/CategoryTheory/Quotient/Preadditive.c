// Lean compiler output
// Module: Mathlib.CategoryTheory.Quotient.Preadditive
// Imports: public import Init public import Mathlib.CategoryTheory.Quotient public import Mathlib.CategoryTheory.Preadditive.AdditiveFunctor
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
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_neg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_preadditive___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_preadditive___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_preadditive(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_neg___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_add___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_add___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_add(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_add___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_apply_2(x_1, x_2, x_3);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_2(x_8, x_4, x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_add(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Quotient_Preadditive_add___redArg(x_3, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_add___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Quotient_Preadditive_add(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_neg___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_apply_2(x_1, x_2, x_3);
x_6 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_5);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_7, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_neg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Quotient_Preadditive_neg___redArg(x_3, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_Preadditive_neg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Quotient_Preadditive_neg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_preadditive___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_inc_ref(x_1);
lean_inc(x_4);
lean_inc(x_3);
x_5 = lean_apply_2(x_1, x_3, x_4);
x_6 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_5);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Quotient_Preadditive_add___boxed), 10, 8);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, x_1);
lean_closure_set(x_8, 3, lean_box(0));
lean_closure_set(x_8, 4, lean_box(0));
lean_closure_set(x_8, 5, lean_box(0));
lean_closure_set(x_8, 6, x_3);
lean_closure_set(x_8, 7, x_4);
x_9 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Quotient_Preadditive_neg___boxed), 9, 8);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_2);
lean_closure_set(x_9, 2, x_1);
lean_closure_set(x_9, 3, lean_box(0));
lean_closure_set(x_9, 4, lean_box(0));
lean_closure_set(x_9, 5, lean_box(0));
lean_closure_set(x_9, 6, x_3);
lean_closure_set(x_9, 7, x_4);
lean_inc_ref(x_8);
lean_inc(x_7);
x_10 = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_7);
lean_closure_set(x_10, 2, x_8);
lean_inc_ref(x_10);
lean_inc(x_7);
lean_inc_ref(x_8);
x_11 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_11, 0, x_8);
lean_ctor_set(x_11, 1, x_7);
lean_ctor_set(x_11, 2, x_10);
lean_inc_ref(x_9);
lean_inc_ref(x_11);
x_12 = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_11);
lean_closure_set(x_12, 2, x_9);
lean_inc_ref(x_9);
x_13 = lean_alloc_closure((void*)(lp_mathlib_zsmulRec___boxed), 7, 5);
lean_closure_set(x_13, 0, lean_box(0));
lean_closure_set(x_13, 1, x_7);
lean_closure_set(x_13, 2, x_8);
lean_closure_set(x_13, 3, x_9);
lean_closure_set(x_13, 4, x_10);
x_14 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_14, 0, x_11);
lean_ctor_set(x_14, 1, x_9);
lean_ctor_set(x_14, 2, x_12);
lean_ctor_set(x_14, 3, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_preadditive___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Quotient_preadditive___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Quotient_preadditive(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Quotient_preadditive___redArg(x_2, x_3);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Quotient(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Preadditive_AdditiveFunctor(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Quotient_Preadditive(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Quotient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Preadditive_AdditiveFunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
