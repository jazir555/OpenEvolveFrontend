// Lean compiler output
// Module: Mathlib.Analysis.Calculus.Deriv.Basic
// Imports: public import Init public import Mathlib.Analysis.Calculus.FDeriv.Const public import Mathlib.Analysis.Normed.Operator.NormedSpace public import Mathlib.Analysis.Calculus.TangentCone.DimOne public import Mathlib.Analysis.Calculus.TangentCone.Real
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
lean_object* initialize_mathlib_Mathlib_Analysis_Calculus_FDeriv_Const(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Operator_NormedSpace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Calculus_TangentCone_DimOne(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Calculus_TangentCone_Real(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Calculus_Deriv_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Calculus_FDeriv_Const(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Operator_NormedSpace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Calculus_TangentCone_DimOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Calculus_TangentCone_Real(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
