// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.Reverse
// Imports: public import Init public import Mathlib.Algebra.Polynomial.Degree.TrailingDegree public import Mathlib.Algebra.Polynomial.EraseLead
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAtFun___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAtFun(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAtFun(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_le(x_2, x_1);
if (x_3 == 0)
{
lean_inc(x_2);
return x_2;
}
else
{
lean_object* x_4; 
x_4 = lean_nat_sub(x_1, x_2);
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAtFun___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Polynomial_revAtFun(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_le(x_2, x_1);
if (x_3 == 0)
{
lean_inc(x_2);
return x_2;
}
else
{
lean_object* x_4; 
x_4 = lean_nat_sub(x_1, x_2);
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Polynomial_revAt___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Polynomial_revAt___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_apply_1(x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_EraseLead(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Reverse(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_EraseLead(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
