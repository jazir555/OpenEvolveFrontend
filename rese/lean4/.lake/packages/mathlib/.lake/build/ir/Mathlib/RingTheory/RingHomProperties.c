// Lean compiler output
// Module: Mathlib.RingTheory.RingHomProperties
// Imports: public import Init public import Mathlib.Algebra.Category.Ring.Constructions public import Mathlib.Algebra.Category.Ring.Colimits public import Mathlib.CategoryTheory.Iso public import Mathlib.CategoryTheory.MorphismProperty.Limits public import Mathlib.RingTheory.Localization.Away.Basic public import Mathlib.RingTheory.IsTensorProduct
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
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Ring_Constructions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Ring_Colimits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Iso(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_MorphismProperty_Limits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_Away_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_IsTensorProduct(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_RingHomProperties(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Ring_Constructions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Ring_Colimits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Iso(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_MorphismProperty_Limits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_Away_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_IsTensorProduct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
