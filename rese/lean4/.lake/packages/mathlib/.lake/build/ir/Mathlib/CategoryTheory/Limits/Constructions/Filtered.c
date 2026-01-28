// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Constructions.Filtered
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Constructions.LimitsOfProductsAndEqualizers public import Mathlib.CategoryTheory.Limits.Shapes.Opposites.Filtered public import Mathlib.CategoryTheory.Limits.Shapes.Opposites.Products
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
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_LimitsOfProductsAndEqualizers(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Opposites_Filtered(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Opposites_Products(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_Filtered(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_LimitsOfProductsAndEqualizers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Opposites_Filtered(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Opposites_Products(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
