// Lean compiler output
// Module: Mathlib.CategoryTheory.Abelian.SerreClass.Basic
// Imports: public import Init public import Mathlib.CategoryTheory.Abelian.Basic public import Mathlib.CategoryTheory.ObjectProperty.ContainsZero public import Mathlib.CategoryTheory.ObjectProperty.EpiMono public import Mathlib.CategoryTheory.ObjectProperty.Extensions public import Mathlib.Algebra.Homology.ShortComplex.ShortExact
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
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Abelian_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_ObjectProperty_ContainsZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_ObjectProperty_EpiMono(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_ObjectProperty_Extensions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_ShortComplex_ShortExact(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Abelian_SerreClass_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Abelian_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_ObjectProperty_ContainsZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_ObjectProperty_EpiMono(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_ObjectProperty_Extensions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_ShortComplex_ShortExact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
