// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Constructions.Equalizers
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Shapes.Equalizers public import Mathlib.CategoryTheory.Limits.Shapes.BinaryProducts public import Mathlib.CategoryTheory.Limits.Preserves.Shapes.Pullbacks public import Mathlib.CategoryTheory.Limits.Preserves.Shapes.BinaryProducts
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
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Equalizers(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryProducts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_Pullbacks(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_BinaryProducts(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_Equalizers(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Equalizers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryProducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_Pullbacks(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_BinaryProducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
