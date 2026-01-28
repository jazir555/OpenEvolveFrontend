// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.Identities
// Imports: public import Init public import Mathlib.Algebra.Polynomial.Derivative public import Mathlib.Tactic.LinearCombination public import Mathlib.Tactic.Ring
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
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powAddExpansion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_evalSubFactor___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_evalSubFactor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powAddExpansion___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powSubPowFactor___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Polynomial_sum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_binomExpansion___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powAddExpansion___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_evalSubFactor___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powSubPowFactor___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_binomExpansion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_binomExpansion___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_binomExpansion___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powSubPowFactor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_evalSubFactor___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powAddExpansion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powSubPowFactor___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powAddExpansion___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_5 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_inc_ref(x_6);
x_7 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_6);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_6);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_9, 1);
lean_inc(x_11);
lean_dec_ref(x_9);
x_12 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_5);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_12, 2);
lean_inc(x_14);
lean_dec_ref(x_12);
x_15 = lean_unsigned_to_nat(0u);
x_16 = lean_nat_dec_eq(x_4, x_15);
if (x_16 == 1)
{
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_8;
}
else
{
lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_17 = lean_unsigned_to_nat(1u);
x_18 = lean_nat_sub(x_4, x_17);
x_19 = lean_nat_dec_eq(x_18, x_15);
if (x_19 == 1)
{
lean_dec(x_18);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_8;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
lean_dec(x_8);
x_20 = lean_ctor_get(x_1, 3);
lean_inc(x_20);
x_21 = lean_nat_sub(x_18, x_17);
lean_dec(x_18);
x_22 = lean_nat_add(x_21, x_17);
lean_inc(x_3);
lean_inc(x_2);
x_23 = lp_mathlib_Polynomial_powAddExpansion___redArg(x_1, x_2, x_3, x_22);
lean_dec(x_22);
lean_inc(x_10);
lean_inc(x_23);
lean_inc(x_2);
x_24 = lean_apply_2(x_10, x_2, x_23);
lean_inc(x_21);
x_25 = lean_apply_1(x_13, x_21);
lean_inc(x_11);
x_26 = lean_apply_2(x_11, x_25, x_14);
x_27 = lean_apply_2(x_20, x_21, x_2);
lean_inc(x_10);
x_28 = lean_apply_2(x_10, x_26, x_27);
lean_inc(x_11);
x_29 = lean_apply_2(x_11, x_24, x_28);
x_30 = lean_apply_2(x_10, x_23, x_3);
x_31 = lean_apply_2(x_11, x_29, x_30);
return x_31;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powAddExpansion(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Polynomial_powAddExpansion___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powAddExpansion___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Polynomial_powAddExpansion(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powAddExpansion___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Polynomial_powAddExpansion___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
x_6 = lp_mathlib_Polynomial_powAddExpansion___redArg(x_5, x_2, x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1___redArg(x_2, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_binomExpansion___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib___private_Mathlib_Algebra_Polynomial_Identities_0__Polynomial_polyBinomAux1___redArg(x_1, x_2, x_3, x_5);
x_8 = lean_apply_2(x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_binomExpansion___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Polynomial_binomExpansion___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_binomExpansion___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_1);
x_5 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_1);
x_6 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_6);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Polynomial_binomExpansion___redArg___lam__0___boxed), 6, 4);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_3);
lean_closure_set(x_10, 2, x_4);
lean_closure_set(x_10, 3, x_9);
x_11 = lp_mathlib_Polynomial_sum___redArg(x_7, x_2, x_10);
lean_dec_ref(x_7);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_binomExpansion(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Polynomial_binomExpansion___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powSubPowFactor___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; 
lean_inc_ref(x_1);
x_5 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_1);
x_6 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_5);
lean_inc_ref(x_6);
x_7 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_6);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
lean_dec_ref(x_7);
lean_inc_ref(x_1);
x_9 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_10 = lean_ctor_get(x_9, 1);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
x_11 = lean_ctor_get(x_10, 2);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_6);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_12, 1);
lean_inc(x_14);
lean_dec_ref(x_12);
x_15 = lean_ctor_get(x_1, 0);
x_16 = lean_unsigned_to_nat(0u);
x_17 = lean_nat_dec_eq(x_4, x_16);
if (x_17 == 1)
{
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_11);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_8;
}
else
{
lean_object* x_18; lean_object* x_19; uint8_t x_20; 
lean_dec(x_8);
x_18 = lean_unsigned_to_nat(1u);
x_19 = lean_nat_sub(x_4, x_18);
x_20 = lean_nat_dec_eq(x_19, x_16);
if (x_20 == 1)
{
lean_dec(x_19);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_11;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
lean_dec(x_11);
x_21 = lean_ctor_get(x_15, 3);
lean_inc(x_21);
x_22 = lean_nat_sub(x_19, x_18);
lean_dec(x_19);
x_23 = lean_nat_add(x_22, x_18);
lean_dec(x_22);
lean_inc(x_3);
lean_inc(x_2);
x_24 = lp_mathlib_Polynomial_powSubPowFactor___redArg(x_1, x_2, x_3, x_23);
x_25 = lean_apply_2(x_13, x_24, x_2);
x_26 = lean_apply_2(x_21, x_23, x_3);
x_27 = lean_apply_2(x_14, x_25, x_26);
return x_27;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powSubPowFactor(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Polynomial_powSubPowFactor___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powSubPowFactor___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Polynomial_powSubPowFactor(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_powSubPowFactor___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Polynomial_powSubPowFactor___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_evalSubFactor___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_Polynomial_powSubPowFactor___redArg(x_1, x_2, x_3, x_5);
x_8 = lean_apply_2(x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_evalSubFactor___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Polynomial_evalSubFactor___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_evalSubFactor___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_1);
x_5 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_1);
x_6 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_6);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Polynomial_evalSubFactor___redArg___lam__0___boxed), 6, 4);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_3);
lean_closure_set(x_10, 2, x_4);
lean_closure_set(x_10, 3, x_9);
x_11 = lp_mathlib_Polynomial_sum___redArg(x_7, x_2, x_10);
lean_dec_ref(x_7);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_evalSubFactor(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Polynomial_evalSubFactor___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Derivative(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_LinearCombination(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Identities(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Derivative(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_LinearCombination(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
