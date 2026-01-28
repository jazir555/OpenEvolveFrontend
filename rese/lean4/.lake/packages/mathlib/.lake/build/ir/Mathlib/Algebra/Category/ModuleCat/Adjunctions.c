// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Adjunctions
// Imports: public import Init public import Mathlib.Algebra.Category.ModuleCat.Monoidal.Basic public import Mathlib.CategoryTheory.Monoidal.Types.Basic public import Mathlib.LinearAlgebra.DirectSum.Finsupp public import Mathlib.CategoryTheory.Linear.LinearFunctor
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_ext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_ext___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_ext(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_of___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_of___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_of(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_app___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_of___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_ext___redArg(lean_object*);
lean_object* lp_mathlib_Finsupp_sum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_of(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_of___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_of___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Free_of(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_of___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_Free_of___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_apply_3(x_1, x_2, x_3, x_7);
x_10 = lean_apply_4(x_4, x_5, x_6, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
lean_dec_ref(x_1);
lean_inc(x_7);
lean_inc(x_4);
x_9 = lean_apply_1(x_7, x_4);
lean_inc(x_5);
x_10 = lean_apply_1(x_7, x_5);
lean_inc(x_10);
lean_inc(x_9);
x_11 = lean_apply_2(x_2, x_9, x_10);
x_12 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_11);
x_13 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Free_lift___redArg___lam__1), 8, 6);
lean_closure_set(x_13, 0, x_8);
lean_closure_set(x_13, 1, x_4);
lean_closure_set(x_13, 2, x_5);
lean_closure_set(x_13, 3, x_3);
lean_closure_set(x_13, 4, x_9);
lean_closure_set(x_13, 5, x_10);
x_14 = lp_mathlib_Finsupp_sum___redArg(x_12, x_6, x_13);
lean_dec_ref(x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_3);
x_4 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Free_lift___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Free_lift___redArg___lam__2), 6, 3);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_1);
lean_closure_set(x_5, 2, x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Free_lift___redArg(x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_lift___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Free_lift(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_ext___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Iso_app___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_ext___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Free_ext___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_ext(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CategoryTheory_Free_ext___redArg(x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Free_ext___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CategoryTheory_Free_ext(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_16;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Monoidal_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Types_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_DirectSum_Finsupp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Linear_LinearFunctor(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Adjunctions(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Monoidal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Types_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_DirectSum_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Linear_LinearFunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
