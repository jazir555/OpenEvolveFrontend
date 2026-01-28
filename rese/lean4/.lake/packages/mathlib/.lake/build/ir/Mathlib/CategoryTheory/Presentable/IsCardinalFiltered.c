// Lean compiler output
// Module: Mathlib.CategoryTheory.Presentable.IsCardinalFiltered
// Imports: public import Init public import Mathlib.CategoryTheory.Filtered.Final public import Mathlib.CategoryTheory.Limits.Shapes.WideEqualizers public import Mathlib.CategoryTheory.Comma.CardinalArrow public import Mathlib.SetTheory.Cardinal.Cofinality public import Mathlib.SetTheory.Cardinal.HasCardinalLT public import Mathlib.SetTheory.Cardinal.Arithmetic
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
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Filtered_Final(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_WideEqualizers(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Comma_CardinalArrow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Cofinality(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_HasCardinalLT(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Arithmetic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Presentable_IsCardinalFiltered(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Filtered_Final(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_WideEqualizers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Comma_CardinalArrow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Cofinality(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_HasCardinalLT(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Arithmetic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
