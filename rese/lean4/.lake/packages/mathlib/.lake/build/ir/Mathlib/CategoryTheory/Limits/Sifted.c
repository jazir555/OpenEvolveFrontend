// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Sifted
// Imports: public import Init public import Mathlib.CategoryTheory.Monoidal.FunctorCategory public import Mathlib.CategoryTheory.Monoidal.ExternalProduct.Basic public import Mathlib.CategoryTheory.Monoidal.Closed.Types public import Mathlib.CategoryTheory.Monoidal.Limits.Preserves public import Mathlib.CategoryTheory.Limits.Preserves.Bifunctor public import Mathlib.CategoryTheory.Limits.Preserves.FunctorCategory public import Mathlib.CategoryTheory.Limits.IsConnected
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
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_FunctorCategory(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_ExternalProduct_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Closed_Types(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Limits_Preserves(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Bifunctor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_FunctorCategory(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_IsConnected(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Sifted(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_FunctorCategory(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_ExternalProduct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Closed_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Limits_Preserves(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Bifunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_FunctorCategory(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_IsConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
