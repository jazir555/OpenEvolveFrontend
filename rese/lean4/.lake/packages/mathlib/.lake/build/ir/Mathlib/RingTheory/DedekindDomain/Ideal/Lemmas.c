// Lean compiler output
// Module: Mathlib.RingTheory.DedekindDomain.Ideal.Lemmas
// Imports: public import Init public import Mathlib.Algebra.Polynomial.FieldDivision public import Mathlib.Algebra.Squarefree.Basic public import Mathlib.RingTheory.ChainOfDivisors public import Mathlib.RingTheory.DedekindDomain.Ideal.Basic public import Mathlib.RingTheory.Spectrum.Maximal.Localization public import Mathlib.Algebra.Order.GroupWithZero.Unbundled.OrderIso
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
LEAN_EXPORT lean_object* lp_mathlib_idealFactorsFunOfQuotHom___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_ofPrime___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_idealFactorsFunOfQuotHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_idealFactorsFunOfQuotHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_ofPrime___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__2;
static lean_object* lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__0;
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_ofPrime(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_equivMaximalSpectrum(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderIso_ofHomInv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_equivMaximalSpectrum___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderHom_instFunLike___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_idealFactorsEquivOfQuotEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_equivMaximalSpectrum___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_idealFactorsEquivOfQuotEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_ofPrime(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_ofPrime___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_ofPrime___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_IsDedekindDomain_HeightOneSpectrum_ofPrime(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_equivMaximalSpectrum___lam__0(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_equivMaximalSpectrum(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_IsDedekindDomain_HeightOneSpectrum_equivMaximalSpectrum___lam__0), 1, 0);
lean_inc_ref(x_5);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsDedekindDomain_HeightOneSpectrum_equivMaximalSpectrum___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_IsDedekindDomain_HeightOneSpectrum_equivMaximalSpectrum(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_idealFactorsFunOfQuotHom___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_idealFactorsFunOfQuotHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_idealFactorsFunOfQuotHom___lam__0), 1, 0);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_idealFactorsFunOfQuotHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_idealFactorsFunOfQuotHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_8);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_10;
}
}
static lean_object* _init_lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_OrderHom_instFunLike___lam__0), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_idealFactorsFunOfQuotHom___lam__0), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__1;
x_2 = lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__0;
x_3 = lp_mathlib_OrderIso_ofHomInv___redArg(x_2, x_2, x_1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_idealFactorsEquivOfQuotEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__2;
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_idealFactorsEquivOfQuotEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_idealFactorsEquivOfQuotEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_9);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_mathlib_idealFactorsEquivOfQuotEquiv(lean_box(0), lean_box(0), x_1, x_2, lean_box(0), x_3, x_4, lean_box(0), x_5);
x_8 = lp_mathlib_Equiv_toEmbedding___redArg(x_7);
x_9 = lean_apply_1(x_8, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lp_mathlib_idealFactorsEquivOfQuotEquiv(lean_box(0), lean_box(0), x_1, x_2, lean_box(0), x_3, x_4, lean_box(0), x_5);
x_8 = lp_mathlib_Equiv_symm___redArg(x_7);
x_9 = lp_mathlib_Equiv_toEmbedding___redArg(x_8);
x_10 = lean_apply_1(x_9, x_6);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__1(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_5);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__0___boxed), 6, 5);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_2);
lean_closure_set(x_6, 2, x_3);
lean_closure_set(x_6, 3, x_4);
lean_closure_set(x_6, 4, x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg___lam__1___boxed), 6, 5);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_2);
lean_closure_set(x_7, 2, x_3);
lean_closure_set(x_7, 3, x_4);
lean_closure_set(x_7, 4, x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalizedFactorsEquivOfQuotEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_normalizedFactorsEquivOfQuotEquiv___redArg(x_3, x_4, x_6, x_7, x_9);
return x_12;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_FieldDivision(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Squarefree_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_ChainOfDivisors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_DedekindDomain_Ideal_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Spectrum_Maximal_Localization(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Unbundled_OrderIso(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_DedekindDomain_Ideal_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_FieldDivision(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Squarefree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_ChainOfDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_DedekindDomain_Ideal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Spectrum_Maximal_Localization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Unbundled_OrderIso(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__0 = _init_lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__0();
lean_mark_persistent(lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__0);
lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__1 = _init_lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__1();
lean_mark_persistent(lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__1);
lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__2 = _init_lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__2();
lean_mark_persistent(lp_mathlib_idealFactorsEquivOfQuotEquiv___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
