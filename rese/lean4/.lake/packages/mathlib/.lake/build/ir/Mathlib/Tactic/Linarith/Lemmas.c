// Lean compiler output
// Module: Mathlib.Tactic.Linarith.Lemmas
// Imports: public import Init public meta import Batteries.Tactic.Lint.Basic public meta import Mathlib.Algebra.Order.Monoid.Unbundled.Basic public meta import Mathlib.Algebra.Order.Ring.Defs public meta import Mathlib.Algebra.Order.ZeroLEOne public meta import Mathlib.Data.Nat.Cast.Order.Ring public meta import Mathlib.Data.Int.Order.Basic public meta import Mathlib.Data.Ineq
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
static lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2;
static lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__7;
static lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___boxed(lean_object*);
static lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6;
static lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__3;
static lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0;
static lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1;
static lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName(uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__5;
static lean_object* _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Mathlib", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Linarith", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mul_eq", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__3;
x_2 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2;
x_3 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1;
x_4 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mul_nonpos", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__5;
x_2 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2;
x_3 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1;
x_4 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mul_neg", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__7;
x_2 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2;
x_3 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1;
x_4 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName(uint8_t x_1) {
_start:
{
switch (x_1) {
case 0:
{
lean_object* x_2; 
x_2 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4;
return x_2;
}
case 1:
{
lean_object* x_3; 
x_3 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6;
return x_3;
}
default: 
{
lean_object* x_4; 
x_4 = lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8;
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_Mathlib_Ineq_toConstMulName(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Ineq(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0 = _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0();
lean_mark_persistent(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0);
lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1 = _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1();
lean_mark_persistent(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1);
lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2 = _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2();
lean_mark_persistent(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2);
lp_mathlib_Mathlib_Ineq_toConstMulName___closed__3 = _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__3();
lean_mark_persistent(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__3);
lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4 = _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4();
lean_mark_persistent(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4);
lp_mathlib_Mathlib_Ineq_toConstMulName___closed__5 = _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__5();
lean_mark_persistent(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__5);
lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6 = _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6();
lean_mark_persistent(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6);
lp_mathlib_Mathlib_Ineq_toConstMulName___closed__7 = _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__7();
lean_mark_persistent(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__7);
lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8 = _init_lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8();
lean_mark_persistent(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
