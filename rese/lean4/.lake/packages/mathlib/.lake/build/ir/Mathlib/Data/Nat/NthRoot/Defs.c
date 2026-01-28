// Lean compiler output
// Module: Mathlib.Data.Nat.NthRoot.Defs
// Imports: public import Init public import Mathlib.Init
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
LEAN_EXPORT lean_object* lp_mathlib_Nat_nthRoot(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_nthRoot_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_nthRoot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_nthRoot_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_nthRoot_go(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_nat_dec_eq(x_3, x_5);
if (x_6 == 1)
{
lean_dec(x_3);
return x_4;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_add(x_1, x_7);
x_9 = lean_nat_pow(x_4, x_8);
x_10 = lean_nat_div(x_2, x_9);
lean_dec(x_9);
x_11 = lean_nat_mul(x_8, x_4);
lean_dec(x_8);
x_12 = lean_nat_add(x_10, x_11);
lean_dec(x_11);
lean_dec(x_10);
x_13 = lean_unsigned_to_nat(2u);
x_14 = lean_nat_add(x_1, x_13);
x_15 = lean_nat_div(x_12, x_14);
lean_dec(x_14);
lean_dec(x_12);
x_16 = lean_nat_dec_lt(x_15, x_4);
if (x_16 == 0)
{
lean_dec(x_15);
lean_dec(x_3);
return x_4;
}
else
{
lean_object* x_17; 
lean_dec(x_4);
x_17 = lean_nat_sub(x_3, x_7);
lean_dec(x_3);
x_3 = x_17;
x_4 = x_15;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_nthRoot_go___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Nat_nthRoot_go(x_1, x_2, x_3, x_4);
lean_dec(x_2);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_nthRoot(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_nat_dec_eq(x_1, x_3);
if (x_4 == 1)
{
lean_object* x_5; 
lean_dec(x_2);
x_5 = lean_unsigned_to_nat(1u);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_6 = lean_unsigned_to_nat(1u);
x_7 = lean_nat_sub(x_1, x_6);
x_8 = lean_nat_dec_eq(x_7, x_3);
if (x_8 == 1)
{
lean_dec(x_7);
return x_2;
}
else
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_nat_sub(x_7, x_6);
lean_dec(x_7);
lean_inc_n(x_2, 2);
x_10 = lp_mathlib_Nat_nthRoot_go(x_9, x_2, x_2, x_2);
lean_dec(x_2);
lean_dec(x_9);
return x_10;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_nthRoot___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_nthRoot(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_NthRoot_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
