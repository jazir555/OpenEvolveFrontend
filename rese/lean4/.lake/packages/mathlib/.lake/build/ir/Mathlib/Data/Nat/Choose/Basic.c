// Lean compiler output
// Module: Mathlib.Data.Nat.Choose.Basic
// Imports: public import Init public import Mathlib.Data.Nat.Factorial.Basic public import Mathlib.Order.Monotone.Defs
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_choose(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_fast__choose___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_multichoose___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_descFactorialBinary(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_choose___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_factorialBinarySplitting(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_fast__choose(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_multichoose(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_choose(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_nat_dec_eq(x_2, x_3);
if (x_4 == 1)
{
lean_object* x_5; 
x_5 = lean_unsigned_to_nat(1u);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = lean_nat_dec_eq(x_1, x_3);
if (x_6 == 1)
{
return x_3;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_sub(x_2, x_7);
x_9 = lean_nat_sub(x_1, x_7);
x_10 = lp_mathlib_Nat_choose(x_9, x_8);
x_11 = lean_nat_add(x_8, x_7);
lean_dec(x_8);
x_12 = lp_mathlib_Nat_choose(x_9, x_11);
lean_dec(x_11);
lean_dec(x_9);
x_13 = lean_nat_add(x_10, x_12);
lean_dec(x_12);
lean_dec(x_10);
return x_13;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_choose___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_choose(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_nat_dec_eq(x_2, x_6);
if (x_7 == 1)
{
lean_object* x_8; 
lean_dec(x_5);
lean_dec(x_4);
x_8 = lean_apply_1(x_3, x_1);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; uint8_t x_11; 
lean_dec(x_3);
x_9 = lean_unsigned_to_nat(1u);
x_10 = lean_nat_sub(x_2, x_9);
x_11 = lean_nat_dec_eq(x_1, x_6);
if (x_11 == 1)
{
lean_object* x_12; 
lean_dec(x_5);
lean_dec(x_1);
x_12 = lean_apply_1(x_4, x_10);
return x_12;
}
else
{
lean_object* x_13; lean_object* x_14; 
lean_dec(x_4);
x_13 = lean_nat_sub(x_1, x_9);
lean_dec(x_1);
x_14 = lean_apply_2(x_5, x_13, x_10);
return x_14;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_fast__choose(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Nat_descFactorialBinary(x_1, x_2);
x_4 = lp_mathlib_Nat_factorialBinarySplitting(x_2);
x_5 = lean_nat_div(x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_fast__choose___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_fast__choose(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_multichoose(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_nat_dec_eq(x_2, x_3);
if (x_4 == 1)
{
lean_object* x_5; 
x_5 = lean_unsigned_to_nat(1u);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = lean_nat_dec_eq(x_1, x_3);
if (x_6 == 1)
{
return x_3;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_sub(x_2, x_7);
x_9 = lean_nat_sub(x_1, x_7);
x_10 = lean_nat_add(x_8, x_7);
x_11 = lp_mathlib_Nat_multichoose(x_9, x_10);
lean_dec(x_10);
x_12 = lean_nat_add(x_9, x_7);
lean_dec(x_9);
x_13 = lp_mathlib_Nat_multichoose(x_12, x_8);
lean_dec(x_8);
lean_dec(x_12);
x_14 = lean_nat_add(x_11, x_13);
lean_dec(x_13);
lean_dec(x_11);
return x_14;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_multichoose___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_multichoose(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Monotone_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Monotone_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
