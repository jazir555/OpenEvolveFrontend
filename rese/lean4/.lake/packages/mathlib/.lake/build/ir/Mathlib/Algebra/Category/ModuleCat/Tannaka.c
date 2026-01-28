// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Tannaka
// Imports: public import Init public import Mathlib.Algebra.Category.ModuleCat.Basic public import Mathlib.LinearAlgebra.Span.Basic
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
extern lean_object* lp_mathlib_AddCommGrpCat_instConcreteCategoryAddMonoidHomCarrier;
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ringEquivEndForget_u2082___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ringEquivEndForget_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ringEquivEndForget_u2082___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ringEquivEndForget_u2082(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ringEquivEndForget_u2082___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0(x_4, x_1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ringEquivEndForget_u2082___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_1);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 1);
lean_dec(x_8);
lean_inc_ref(x_2);
lean_ctor_set(x_1, 1, x_3);
lean_ctor_set(x_1, 0, x_2);
x_9 = lean_apply_1(x_5, x_1);
lean_inc_ref(x_2);
x_10 = lean_apply_4(x_7, x_2, x_2, x_9, x_4);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_1, 0);
lean_inc(x_11);
lean_dec(x_1);
lean_inc_ref(x_2);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_3);
x_13 = lean_apply_1(x_5, x_12);
lean_inc_ref(x_2);
x_14 = lean_apply_4(x_11, x_2, x_2, x_13, x_4);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ringEquivEndForget_u2082___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_2 = lp_mathlib_AddCommGrpCat_instConcreteCategoryAddMonoidHomCarrier;
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_4 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_5 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 2);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_ringEquivEndForget_u2082___redArg___lam__0), 3, 0);
x_9 = lp_mathlib_Semiring_toModule___redArg(x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ringEquivEndForget_u2082___redArg___lam__1), 5, 4);
lean_closure_set(x_10, 0, x_2);
lean_closure_set(x_10, 1, x_4);
lean_closure_set(x_10, 2, x_9);
lean_closure_set(x_10, 3, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_8);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ringEquivEndForget_u2082(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ringEquivEndForget_u2082___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Tannaka(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
