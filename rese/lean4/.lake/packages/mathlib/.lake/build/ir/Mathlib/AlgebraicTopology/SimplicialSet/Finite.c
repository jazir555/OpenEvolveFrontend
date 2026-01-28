// Lean compiler output
// Module: Mathlib.AlgebraicTopology.SimplicialSet.Finite
// Imports: public import Init public import Mathlib.AlgebraicTopology.SimplicialSet.Dimension public import Mathlib.AlgebraicTopology.SimplicialSet.NonDegenerateSimplices public import Mathlib.Data.Finite.Sigma
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
lean_object* initialize_mathlib_Mathlib_AlgebraicTopology_SimplicialSet_Dimension(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_AlgebraicTopology_SimplicialSet_NonDegenerateSimplices(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finite_Sigma(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_AlgebraicTopology_SimplicialSet_Finite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_AlgebraicTopology_SimplicialSet_Dimension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_AlgebraicTopology_SimplicialSet_NonDegenerateSimplices(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finite_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
