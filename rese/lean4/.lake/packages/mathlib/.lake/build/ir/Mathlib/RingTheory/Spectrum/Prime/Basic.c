// Lean compiler output
// Module: Mathlib.RingTheory.Spectrum.Prime.Basic
// Imports: public import Init public import Mathlib.RingTheory.Ideal.MinimalPrime.Basic public import Mathlib.RingTheory.Nilpotent.Lemmas public import Mathlib.RingTheory.Noetherian.Basic public import Mathlib.RingTheory.Spectrum.Prime.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_vanishingIdeal(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_instUnique(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_instOrderBotOfIsDomain(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Spectrum_Prime_Basic_0__PrimeSpectrum_primeSpectrumProdOfSum_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_primeSpectrumProdOfSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_PrimeSpectrum_vanishingIdeal___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_instOrderBotOfIsDomain___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_vanishingIdeal___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_instUnique___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_instInfSet___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Spectrum_Prime_Basic_0__PrimeSpectrum_primeSpectrumProdOfSum_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Spectrum_Prime_Basic_0__PrimeSpectrum_primeSpectrumProdOfSum_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_primeSpectrumProdOfSum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_primeSpectrumProdOfSum(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_primeSpectrumProdOfSum___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_PrimeSpectrum_primeSpectrumProdOfSum(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Spectrum_Prime_Basic_0__PrimeSpectrum_primeSpectrumProdOfSum_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_3);
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_2, x_4, lean_box(0));
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_2);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_2(x_3, x_6, lean_box(0));
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Spectrum_Prime_Basic_0__PrimeSpectrum_primeSpectrumProdOfSum_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib___private_Mathlib_RingTheory_Spectrum_Prime_Basic_0__PrimeSpectrum_primeSpectrumProdOfSum_match__1_splitter___redArg(x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Spectrum_Prime_Basic_0__PrimeSpectrum_primeSpectrumProdOfSum_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib___private_Mathlib_RingTheory_Spectrum_Prime_Basic_0__PrimeSpectrum_primeSpectrumProdOfSum_match__1_splitter(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_9;
}
}
static lean_object* _init_lp_mathlib_PrimeSpectrum_vanishingIdeal___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Submodule_instInfSet___lam__0(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_vanishingIdeal(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_PrimeSpectrum_vanishingIdeal___closed__0;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_vanishingIdeal___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_PrimeSpectrum_vanishingIdeal(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_instOrderBotOfIsDomain(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_instOrderBotOfIsDomain___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_PrimeSpectrum_instOrderBotOfIsDomain(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_instUnique(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrimeSpectrum_instUnique___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PrimeSpectrum_instUnique(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_MinimalPrime_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Nilpotent_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Noetherian_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Spectrum_Prime_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Spectrum_Prime_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_MinimalPrime_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Nilpotent_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Noetherian_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Spectrum_Prime_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PrimeSpectrum_vanishingIdeal___closed__0 = _init_lp_mathlib_PrimeSpectrum_vanishingIdeal___closed__0();
lean_mark_persistent(lp_mathlib_PrimeSpectrum_vanishingIdeal___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
