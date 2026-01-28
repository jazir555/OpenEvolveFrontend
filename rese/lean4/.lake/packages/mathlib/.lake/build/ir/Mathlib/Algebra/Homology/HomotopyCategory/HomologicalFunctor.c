// Lean compiler output
// Module: Mathlib.Algebra.Homology.HomotopyCategory.HomologicalFunctor
// Imports: public import Init public import Mathlib.Algebra.Homology.HomologicalComplexAbelian public import Mathlib.Algebra.Homology.HomotopyCategory.DegreewiseSplit public import Mathlib.Algebra.Homology.HomologySequence public import Mathlib.CategoryTheory.Triangulated.HomologicalFunctor
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
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomologicalComplexAbelian(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_DegreewiseSplit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomologySequence(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Triangulated_HomologicalFunctor(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_HomologicalFunctor(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_HomologicalComplexAbelian(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_DegreewiseSplit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_HomologySequence(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Triangulated_HomologicalFunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
