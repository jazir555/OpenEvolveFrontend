// Lean compiler output
// Module: Mathlib.Analysis.SpecialFunctions.Exp
// Imports: public import Init public import Mathlib.Analysis.Complex.Asymptotics public import Mathlib.Analysis.Complex.Trigonometric public import Mathlib.Analysis.SpecificLimits.Normed public import Mathlib.Topology.Algebra.MetricSpace.Lipschitz
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
lean_object* initialize_mathlib_Mathlib_Analysis_Complex_Asymptotics(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Complex_Trigonometric(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_SpecificLimits_Normed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_MetricSpace_Lipschitz(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_SpecialFunctions_Exp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Complex_Asymptotics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Complex_Trigonometric(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_SpecificLimits_Normed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_MetricSpace_Lipschitz(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
