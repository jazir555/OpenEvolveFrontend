// Lean compiler output
// Module: Mathlib.RingTheory.Finiteness.Defs
// Imports: public import Init public import Mathlib.Algebra.Algebra.Hom public import Mathlib.Data.Set.Finite.Lemmas public import Mathlib.Data.Finsupp.Defs public import Mathlib.GroupTheory.Finiteness public import Mathlib.RingTheory.Ideal.Span public import Mathlib.Tactic.Algebraize
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Finiteness(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Span(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Algebraize(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Finiteness(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Span(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Algebraize(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
