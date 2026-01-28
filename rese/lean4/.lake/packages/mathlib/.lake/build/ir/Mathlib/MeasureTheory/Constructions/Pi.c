// Lean compiler output
// Module: Mathlib.MeasureTheory.Constructions.Pi
// Imports: public import Init public import Mathlib.Algebra.BigOperators.Fin public import Mathlib.Logic.Encodable.Pi public import Mathlib.MeasureTheory.Group.Measure public import Mathlib.MeasureTheory.MeasurableSpace.Pi public import Mathlib.MeasureTheory.Measure.Prod public import Mathlib.Topology.Constructions
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
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Measure_FiniteSpanningSetsIn_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Measure_FiniteSpanningSetsIn_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Measure_FiniteSpanningSetsIn_pi(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_box(0);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Measure_FiniteSpanningSetsIn_pi___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_MeasureTheory_Measure_FiniteSpanningSetsIn_pi(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Fin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Encodable_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Group_Measure(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Constructions(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_Constructions_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Encodable_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Group_Measure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_MeasurableSpace_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Measure_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Constructions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
