// Lean compiler output
// Module: Mathlib.NumberTheory.Padics.PadicVal.Defs
// Imports: public import Init public import Mathlib.RingTheory.Multiplicity public import Mathlib.Data.Nat.Factors
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
uint8_t l_Nat_decidable__dvd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_padicValNat(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_padicValNat___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_padicValNat___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_findX___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_padicValNat___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4, uint8_t x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_nat_add(x_6, x_1);
x_8 = lean_nat_pow(x_2, x_7);
lean_dec(x_7);
x_9 = l_Nat_decidable__dvd(x_8, x_3);
lean_dec(x_8);
if (x_9 == 0)
{
return x_4;
}
else
{
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_padicValNat___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; uint8_t x_8; uint8_t x_9; lean_object* x_10; 
x_7 = lean_unbox(x_4);
x_8 = lean_unbox(x_5);
x_9 = lp_mathlib_padicValNat___lam__0(x_1, x_2, x_3, x_7, x_8, x_6);
lean_dec(x_6);
lean_dec(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_10 = lean_box(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_padicValNat(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_unsigned_to_nat(1u);
x_4 = lean_nat_dec_eq(x_1, x_3);
if (x_4 == 0)
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_nat_dec_lt(x_5, x_2);
if (x_6 == 0)
{
lean_dec(x_2);
lean_dec(x_1);
return x_5;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_box(x_6);
x_8 = lean_box(x_4);
x_9 = lean_alloc_closure((void*)(lp_mathlib_padicValNat___lam__0___boxed), 6, 5);
lean_closure_set(x_9, 0, x_3);
lean_closure_set(x_9, 1, x_1);
lean_closure_set(x_9, 2, x_2);
lean_closure_set(x_9, 3, x_7);
lean_closure_set(x_9, 4, x_8);
x_10 = lp_mathlib_Nat_findX___redArg(x_9);
return x_10;
}
}
else
{
lean_object* x_11; 
lean_dec(x_2);
lean_dec(x_1);
x_11 = lean_unsigned_to_nat(0u);
return x_11;
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Multiplicity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Factors(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_NumberTheory_Padics_PadicVal_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Multiplicity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Factors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
