// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Shapes.Diagonal
// Imports: public import Init public import Mathlib.CategoryTheory.Comma.Over.Pullback public import Mathlib.CategoryTheory.Limits.Shapes.KernelPair public import Mathlib.CategoryTheory.Limits.Shapes.Pullback.CommSq public import Mathlib.CategoryTheory.Limits.Shapes.Pullback.Assoc
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
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Comma_Over_Pullback(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_KernelPair(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_CommSq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_Assoc(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Diagonal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Comma_Over_Pullback(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_KernelPair(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_CommSq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_Assoc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
