// Lean compiler output
// Module: Mathlib.RingTheory.LocalRing.ResidueField.Ideal
// Imports: public import Init public import Mathlib.RingTheory.LocalRing.ResidueField.Basic public import Mathlib.RingTheory.Localization.AtPrime.Basic public import Mathlib.RingTheory.Localization.FractionRing public import Mathlib.RingTheory.SurjectiveOnStalks
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
lean_object* initialize_mathlib_Mathlib_RingTheory_LocalRing_ResidueField_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_AtPrime_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_SurjectiveOnStalks(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_LocalRing_ResidueField_Ideal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_LocalRing_ResidueField_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_AtPrime_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_SurjectiveOnStalks(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
