// Lean compiler output
// Module: Mathlib.MeasureTheory.Measure.WithDensity
// Imports: public import Init public import Mathlib.MeasureTheory.Integral.Lebesgue.Countable public import Mathlib.MeasureTheory.Measure.Decomposition.Exhaustion public import Mathlib.MeasureTheory.Group.Convolution public import Mathlib.Analysis.LConvolution
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
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Integral_Lebesgue_Countable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_Decomposition_Exhaustion(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Group_Convolution(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_LConvolution(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_WithDensity(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Integral_Lebesgue_Countable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Measure_Decomposition_Exhaustion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Group_Convolution(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_LConvolution(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
