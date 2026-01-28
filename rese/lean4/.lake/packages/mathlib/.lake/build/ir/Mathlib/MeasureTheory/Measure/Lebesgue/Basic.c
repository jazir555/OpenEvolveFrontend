// Lean compiler output
// Module: Mathlib.MeasureTheory.Measure.Lebesgue.Basic
// Imports: public import Init public import Mathlib.Dynamics.Ergodic.MeasurePreserving public import Mathlib.LinearAlgebra.Determinant public import Mathlib.LinearAlgebra.Matrix.Diagonal public import Mathlib.LinearAlgebra.Matrix.Transvection public import Mathlib.MeasureTheory.Group.LIntegral public import Mathlib.MeasureTheory.Integral.Marginal public import Mathlib.MeasureTheory.Measure.Stieltjes public import Mathlib.MeasureTheory.Measure.Haar.OfBasis
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
lean_object* initialize_mathlib_Mathlib_Dynamics_Ergodic_MeasurePreserving(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Determinant(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Diagonal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Transvection(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Group_LIntegral(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Integral_Marginal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_Stieltjes(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_Haar_OfBasis(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_Lebesgue_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Dynamics_Ergodic_MeasurePreserving(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Determinant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Diagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Transvection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Group_LIntegral(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Integral_Marginal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Measure_Stieltjes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Measure_Haar_OfBasis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
