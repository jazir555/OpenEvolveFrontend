// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.Roots
// Imports: public import Init public import Mathlib.Algebra.Polynomial.BigOperators public import Mathlib.Algebra.Polynomial.RingDivision public import Mathlib.Data.Set.Finite.Lemmas public import Mathlib.RingTheory.Coprime.Lemmas public import Mathlib.RingTheory.Localization.FractionRing public import Mathlib.SetTheory.Cardinal.Order
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
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_RingDivision(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Coprime_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_RingDivision(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Coprime_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
