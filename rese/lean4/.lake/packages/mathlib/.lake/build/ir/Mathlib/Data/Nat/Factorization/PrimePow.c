// Lean compiler output
// Module: Mathlib.Data.Nat.Factorization.PrimePow
// Imports: public import Init public import Mathlib.Algebra.IsPrimePow public import Mathlib.Data.Nat.Factorization.Basic public import Mathlib.Data.Nat.Prime.Pow public import Mathlib.NumberTheory.Divisors
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
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_prodNatEquiv___lam__1(lean_object*);
lean_object* lp_mathlib_Nat_minFac(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_prodNatEquiv;
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_prodNatEquiv___lam__0(lean_object*);
lean_object* lp_mathlib_Nat_factorization(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_prodNatEquiv___lam__0___boxed(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_prodNatEquiv___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_unsigned_to_nat(1u);
x_5 = lean_nat_add(x_3, x_4);
x_6 = lean_nat_pow(x_2, x_5);
lean_dec(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_prodNatEquiv___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nat_Primes_prodNatEquiv___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_prodNatEquiv___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
lean_inc(x_1);
x_2 = lp_mathlib_Nat_factorization(x_1);
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_ctor_get(x_2, 1);
x_5 = lean_ctor_get(x_2, 0);
lean_dec(x_5);
x_6 = lp_mathlib_Nat_minFac(x_1);
lean_dec(x_1);
lean_inc(x_6);
x_7 = lean_apply_1(x_4, x_6);
x_8 = lean_unsigned_to_nat(1u);
x_9 = lean_nat_sub(x_7, x_8);
lean_dec(x_7);
lean_ctor_set(x_2, 1, x_9);
lean_ctor_set(x_2, 0, x_6);
return x_2;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = lean_ctor_get(x_2, 1);
lean_inc(x_10);
lean_dec(x_2);
x_11 = lp_mathlib_Nat_minFac(x_1);
lean_dec(x_1);
lean_inc(x_11);
x_12 = lean_apply_1(x_10, x_11);
x_13 = lean_unsigned_to_nat(1u);
x_14 = lean_nat_sub(x_12, x_13);
lean_dec(x_12);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_11);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
}
static lean_object* _init_lp_mathlib_Nat_Primes_prodNatEquiv() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_Primes_prodNatEquiv___lam__0___boxed), 1, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Nat_Primes_prodNatEquiv___lam__1), 1, 0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_IsPrimePow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorization_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Prime_Pow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_NumberTheory_Divisors(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorization_PrimePow(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_IsPrimePow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Factorization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Prime_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_NumberTheory_Divisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_Primes_prodNatEquiv = _init_lp_mathlib_Nat_Primes_prodNatEquiv();
lean_mark_persistent(lp_mathlib_Nat_Primes_prodNatEquiv);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
