// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.Degree.TrailingDegree
// Imports: public import Init public import Mathlib.Algebra.Polynomial.Degree.Support public import Mathlib.Data.ENat.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_ENat_toNat(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Finset_min___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Polynomial_coeff___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff___boxed(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_instLinearOrder;
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Nat_instLinearOrder;
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lp_mathlib_Finset_min___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Polynomial_trailingDegree___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Polynomial_trailingDegree(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Polynomial_trailingDegree___redArg(x_1);
x_3 = lp_mathlib_ENat_toNat(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Polynomial_natTrailingDegree___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Polynomial_natTrailingDegree(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_Polynomial_natTrailingDegree___redArg(x_1);
x_3 = lp_mathlib_Polynomial_coeff___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Polynomial_trailingCoeff___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Polynomial_trailingCoeff(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_5 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_4);
x_6 = lean_ctor_get(x_5, 2);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_Polynomial_trailingCoeff___redArg(x_2);
x_8 = lean_apply_2(x_3, x_7, x_6);
x_9 = lean_unbox(x_8);
return x_9;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = lp_mathlib_Polynomial_TrailingMonic_decidable___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Polynomial_TrailingMonic_decidable(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_Polynomial_TrailingMonic_decidable___redArg(x_1, x_2, x_3);
lean_dec_ref(x_1);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
lean_inc_ref(x_2);
x_3 = lp_mathlib_Polynomial_natTrailingDegree___redArg(x_2);
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_nat_dec_eq(x_3, x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_unsigned_to_nat(1u);
x_7 = lean_nat_add(x_3, x_6);
lean_dec(x_3);
x_8 = lp_mathlib_Polynomial_coeff___redArg(x_2, x_7);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_dec(x_3);
lean_dec_ref(x_2);
x_9 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_10);
x_12 = lean_ctor_get(x_11, 1);
lean_inc(x_12);
lean_dec_ref(x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Polynomial_nextCoeffUp___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Polynomial_nextCoeffUp(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Polynomial_nextCoeffUp___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Support(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ENat_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
