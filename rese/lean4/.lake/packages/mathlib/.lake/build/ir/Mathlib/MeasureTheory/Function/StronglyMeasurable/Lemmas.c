// Lean compiler output
// Module: Mathlib.MeasureTheory.Function.StronglyMeasurable.Lemmas
// Imports: public import Init public import Mathlib.Analysis.Normed.Operator.BoundedLinearMaps public import Mathlib.Dynamics.Ergodic.MeasurePreserving public import Mathlib.MeasureTheory.Function.StronglyMeasurable.AEStronglyMeasurable public import Mathlib.MeasureTheory.Measure.WithDensity public import Mathlib.Topology.Algebra.Module.FiniteDimension
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
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Operator_BoundedLinearMaps(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Dynamics_Ergodic_MeasurePreserving(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Function_StronglyMeasurable_AEStronglyMeasurable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_WithDensity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Module_FiniteDimension(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_Function_StronglyMeasurable_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Operator_BoundedLinearMaps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Dynamics_Ergodic_MeasurePreserving(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Function_StronglyMeasurable_AEStronglyMeasurable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Measure_WithDensity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Module_FiniteDimension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
