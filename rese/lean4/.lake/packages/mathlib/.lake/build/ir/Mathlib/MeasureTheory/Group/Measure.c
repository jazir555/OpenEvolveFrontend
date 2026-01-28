// Lean compiler output
// Module: Mathlib.MeasureTheory.Group.Measure
// Imports: public import Init public import Mathlib.Algebra.Group.Pointwise.Set.Card public import Mathlib.GroupTheory.Complement public import Mathlib.MeasureTheory.Group.Action public import Mathlib.MeasureTheory.Group.Pointwise public import Mathlib.MeasureTheory.Measure.Prod public import Mathlib.Topology.Algebra.Module.Equiv public import Mathlib.Topology.ContinuousMap.CocompactMap public import Mathlib.Topology.Algebra.ContinuousMonoidHom
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Complement(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Group_Action(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Group_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Measure_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Module_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_ContinuousMap_CocompactMap(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_ContinuousMonoidHom(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_Group_Measure(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Complement(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Group_Action(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Group_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Measure_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Module_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_ContinuousMap_CocompactMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_ContinuousMonoidHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
