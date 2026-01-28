// Lean compiler output
// Module: Mathlib.LinearAlgebra.Matrix.Adjugate
// Imports: public import Init public import Mathlib.Algebra.Regular.Basic public import Mathlib.LinearAlgebra.Matrix.MvPolynomial public import Mathlib.LinearAlgebra.Matrix.Polynomial
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
lean_object* lp_mathlib_Matrix_detRowAlternating___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_adjugate___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cramerMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cramer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
static lean_object* lp_mathlib_Matrix_adjugate___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_adjugate___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cramer___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_adjugate___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_transpose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_adjugate___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_adjugate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cramerMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_single___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Matrix_updateCol(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cramerMap___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_updateCol), 9, 7);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, lean_box(0));
lean_closure_set(x_7, 3, x_1);
lean_closure_set(x_7, 4, x_4);
lean_closure_set(x_7, 5, x_6);
lean_closure_set(x_7, 6, x_5);
x_8 = lp_mathlib_Matrix_detRowAlternating___redArg(x_1, x_2, x_3);
x_9 = lean_apply_1(x_8, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cramerMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Matrix_cramerMap___redArg(x_3, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cramer(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_cramerMap), 8, 6);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, x_3);
lean_closure_set(x_7, 3, x_4);
lean_closure_set(x_7, 4, x_5);
lean_closure_set(x_7, 5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_cramer___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Matrix_cramerMap), 8, 6);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, x_1);
lean_closure_set(x_5, 3, x_2);
lean_closure_set(x_5, 4, x_3);
lean_closure_set(x_5, 5, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_adjugate___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Matrix_adjugate___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_adjugate___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Matrix_adjugate___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_adjugate___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_transpose), 6, 4);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, lean_box(0));
lean_closure_set(x_9, 3, x_1);
lean_inc_ref(x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Pi_single___boxed), 7, 6);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, x_2);
lean_closure_set(x_10, 3, x_3);
lean_closure_set(x_10, 4, x_7);
lean_closure_set(x_10, 5, x_4);
x_11 = lp_mathlib_Matrix_cramerMap___redArg(x_3, x_5, x_6, x_9, x_10, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_adjugate___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_inc_ref(x_3);
x_7 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_3);
x_8 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_7);
x_9 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_8);
x_10 = lean_ctor_get(x_9, 1);
lean_inc(x_10);
lean_dec_ref(x_9);
lean_inc_ref(x_3);
x_11 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_3);
x_12 = lean_ctor_get(x_11, 1);
lean_inc_ref(x_12);
lean_dec_ref(x_11);
x_13 = lean_ctor_get(x_12, 2);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_Matrix_adjugate___redArg___closed__0;
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Matrix_adjugate___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_16, 0, x_10);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Matrix_adjugate___redArg___lam__1), 8, 6);
lean_closure_set(x_17, 0, x_4);
lean_closure_set(x_17, 1, x_16);
lean_closure_set(x_17, 2, x_1);
lean_closure_set(x_17, 3, x_13);
lean_closure_set(x_17, 4, x_2);
lean_closure_set(x_17, 5, x_3);
x_18 = lean_apply_3(x_15, x_17, x_5, x_6);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_adjugate(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Matrix_adjugate___redArg(x_3, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_MvPolynomial(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Polynomial(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Adjugate(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_MvPolynomial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Polynomial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Matrix_adjugate___redArg___closed__0 = _init_lp_mathlib_Matrix_adjugate___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Matrix_adjugate___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
