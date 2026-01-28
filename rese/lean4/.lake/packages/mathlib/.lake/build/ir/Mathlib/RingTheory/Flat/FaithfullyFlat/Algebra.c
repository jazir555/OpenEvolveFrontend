// Lean compiler output
// Module: Mathlib.RingTheory.Flat.FaithfullyFlat.Algebra
// Imports: public import Init public import Mathlib.RingTheory.Flat.FaithfullyFlat.Basic public import Mathlib.RingTheory.Ideal.Over public import Mathlib.RingTheory.LocalRing.RingHom.Basic public import Mathlib.RingTheory.Spectrum.Prime.RingHom public import Mathlib.RingTheory.TensorProduct.Quotient
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
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Flat_FaithfullyFlat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Over(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_LocalRing_RingHom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Spectrum_Prime_RingHom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_Quotient(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Flat_FaithfullyFlat_Algebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Flat_FaithfullyFlat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Over(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_LocalRing_RingHom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Spectrum_Prime_RingHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TensorProduct_Quotient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
