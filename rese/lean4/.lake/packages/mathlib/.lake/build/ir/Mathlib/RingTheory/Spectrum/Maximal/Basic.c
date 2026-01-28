// Lean compiler output
// Module: Mathlib.RingTheory.Spectrum.Maximal.Basic
// Imports: public import Init public import Mathlib.RingTheory.Spectrum.Maximal.Defs public import Mathlib.RingTheory.Spectrum.Prime.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_toPrimeSpectrum___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_toPrimeSpectrum(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_equivSubtype___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_equivSubtype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_toPrimeSpectrum___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_equivSubtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_equivSubtype___lam__0(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_equivSubtype(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MaximalSpectrum_equivSubtype___lam__0), 1, 0);
lean_inc_ref(x_3);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_equivSubtype___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MaximalSpectrum_equivSubtype(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_toPrimeSpectrum(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_toPrimeSpectrum___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MaximalSpectrum_toPrimeSpectrum___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MaximalSpectrum_toPrimeSpectrum(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Spectrum_Maximal_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Spectrum_Prime_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Spectrum_Maximal_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Spectrum_Maximal_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Spectrum_Prime_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
