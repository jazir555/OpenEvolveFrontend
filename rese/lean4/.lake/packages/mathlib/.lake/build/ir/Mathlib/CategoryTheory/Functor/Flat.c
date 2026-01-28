// Lean compiler output
// Module: Mathlib.CategoryTheory.Functor.Flat
// Imports: public import Init public import Mathlib.CategoryTheory.Filtered.Connected public import Mathlib.CategoryTheory.Limits.ConeCategory public import Mathlib.CategoryTheory.Limits.FilteredColimitCommutesFiniteLimit public import Mathlib.CategoryTheory.Limits.Preserves.Filtered public import Mathlib.CategoryTheory.Limits.Preserves.FunctorCategory public import Mathlib.CategoryTheory.Limits.Bicones public import Mathlib.CategoryTheory.Limits.Comma public import Mathlib.CategoryTheory.Limits.Preserves.Finite public import Mathlib.CategoryTheory.Limits.Preserves.Opposites public import Mathlib.CategoryTheory.Limits.Shapes.FiniteLimits
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
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Filtered_Connected(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_ConeCategory(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_FilteredColimitCommutesFiniteLimit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Filtered(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_FunctorCategory(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Bicones(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Comma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Opposites(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_FiniteLimits(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Functor_Flat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Filtered_Connected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_ConeCategory(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_FilteredColimitCommutesFiniteLimit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Filtered(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_FunctorCategory(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Bicones(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Comma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_FiniteLimits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
