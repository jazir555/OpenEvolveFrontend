// Lean compiler output
// Module: Mathlib.MeasureTheory.Group.Action
// Imports: public import Init public import Mathlib.Dynamics.Ergodic.MeasurePreserving public import Mathlib.Dynamics.Minimal public import Mathlib.GroupTheory.GroupAction.Hom public import Mathlib.MeasureTheory.Group.MeasurableEquiv public import Mathlib.MeasureTheory.Measure.Regular public import Mathlib.MeasureTheory.Group.Defs public import Mathlib.Order.Filter.EventuallyConst
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
lean_object* initialize_mathlib_Mathlib_Dynamics_Minimal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Group_MeasurableEquiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_Regular(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_EventuallyConst(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_Group_Action(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Dynamics_Ergodic_MeasurePreserving(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Dynamics_Minimal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Group_MeasurableEquiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Measure_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_EventuallyConst(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
