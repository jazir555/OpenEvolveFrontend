// Lean compiler output
// Module: Mathlib.Analysis.Normed.Module.Ball.Homeomorph
// Imports: public import Init public import Mathlib.Topology.OpenPartialHomeomorph.Composition public import Mathlib.Analysis.Normed.Group.AddTorsor public import Mathlib.Analysis.Normed.Module.Ball.Pointwise public import Mathlib.Data.Real.Sqrt
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
lean_object* initialize_mathlib_Mathlib_Topology_OpenPartialHomeomorph_Composition(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_AddTorsor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Module_Ball_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Real_Sqrt(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Module_Ball_Homeomorph(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_OpenPartialHomeomorph_Composition(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Group_AddTorsor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Module_Ball_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Real_Sqrt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
