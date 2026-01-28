// Lean compiler output
// Module: Mathlib.CategoryTheory.Comma.CardinalArrow
// Imports: public import Init public import Mathlib.CategoryTheory.Comma.Arrow public import Mathlib.CategoryTheory.FinCategory.Basic public import Mathlib.CategoryTheory.EssentiallySmall public import Mathlib.Data.Set.Finite.Basic public import Mathlib.SetTheory.Cardinal.HasCardinalLT
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
lean_object* lp_mathlib_CategoryTheory_Functor_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_CategoryStruct_opposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
x_8 = lean_apply_1(x_4, x_7);
x_9 = lean_apply_1(x_4, x_6);
lean_ctor_set(x_2, 1, x_9);
lean_ctor_set(x_2, 0, x_8);
return x_2;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
x_12 = lean_ctor_get(x_2, 2);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_2);
lean_inc(x_4);
x_13 = lean_apply_1(x_4, x_11);
x_14 = lean_apply_1(x_4, x_10);
x_15 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
lean_ctor_set(x_15, 2, x_12);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__0(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
x_8 = lean_apply_1(x_4, x_7);
x_9 = lean_apply_1(x_4, x_6);
lean_ctor_set(x_2, 1, x_9);
lean_ctor_set(x_2, 0, x_8);
return x_2;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
x_12 = lean_ctor_get(x_2, 2);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_2);
lean_inc(x_4);
x_13 = lean_apply_1(x_4, x_11);
x_14 = lean_apply_1(x_4, x_10);
x_15 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
lean_ctor_set(x_15, 2, x_12);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__1(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_CategoryTheory_CategoryStruct_opposite___redArg(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Arrow_opEquiv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Arrow_opEquiv___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Comma_Arrow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_FinCategory_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_EssentiallySmall(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_HasCardinalLT(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Comma_CardinalArrow(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Comma_Arrow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_FinCategory_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_EssentiallySmall(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_HasCardinalLT(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
