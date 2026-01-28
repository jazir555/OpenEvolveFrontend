// Lean compiler output
// Module: Mathlib.RingTheory.Artinian.Module
// Imports: public import Init public import Mathlib.Algebra.Group.Units.Opposite public import Mathlib.Algebra.Regular.Opposite public import Mathlib.Data.SetLike.Fintype public import Mathlib.LinearAlgebra.FreeModule.Finite.Basic public import Mathlib.Order.Filter.EventuallyConst public import Mathlib.RingTheory.Ideal.Prod public import Mathlib.RingTheory.Ideal.Quotient.Operations public import Mathlib.RingTheory.Jacobson.Semiprimary public import Mathlib.RingTheory.Nilpotent.Lemmas public import Mathlib.RingTheory.Noetherian.Defs public import Mathlib.RingTheory.Spectrum.Maximal.Basic public import Mathlib.RingTheory.Spectrum.Prime.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_IsArtinianRing_primeSpectrumEquivMaximalSpectrum___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsArtinianRing_primeSpectrumEquivMaximalSpectrum___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsArtinianRing_primeSpectrumEquivMaximalSpectrum(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsArtinianRing_primeSpectrumEquivMaximalSpectrum___lam__0(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsArtinianRing_primeSpectrumEquivMaximalSpectrum(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_IsArtinianRing_primeSpectrumEquivMaximalSpectrum___lam__0), 1, 0);
lean_inc_ref(x_4);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsArtinianRing_primeSpectrumEquivMaximalSpectrum___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_IsArtinianRing_primeSpectrumEquivMaximalSpectrum(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SetLike_Fintype(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_FreeModule_Finite_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_EventuallyConst(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Jacobson_Semiprimary(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Nilpotent_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Noetherian_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Spectrum_Maximal_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Spectrum_Prime_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Artinian_Module(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SetLike_Fintype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_FreeModule_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_EventuallyConst(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Jacobson_Semiprimary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Nilpotent_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Noetherian_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Spectrum_Maximal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Spectrum_Prime_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
