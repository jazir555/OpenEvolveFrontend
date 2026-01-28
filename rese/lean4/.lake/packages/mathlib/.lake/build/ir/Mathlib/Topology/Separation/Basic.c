// Lean compiler output
// Module: Mathlib.Topology.Separation.Basic
// Imports: public import Init public import Mathlib.Algebra.Notation.Support public import Mathlib.Topology.Inseparable public import Mathlib.Topology.Piecewise public import Mathlib.Topology.Separation.SeparatedNhds public import Mathlib.Topology.Compactness.LocallyCompact public import Mathlib.Topology.Bases public import Mathlib.Tactic.StacksAttribute
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
LEAN_EXPORT lean_object* lp_mathlib_specializationOrder___redArg(lean_object*);
lean_object* lp_mathlib_specializationPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_specializationOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bornology_relativelyCompact(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_specializationOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_specializationPreorder(lean_box(0), x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_specializationOrder___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_specializationPreorder(lean_box(0), x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bornology_relativelyCompact(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Support(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Inseparable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Piecewise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Separation_SeparatedNhds(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Compactness_LocallyCompact(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Bases(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_StacksAttribute(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Separation_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Inseparable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Separation_SeparatedNhds(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Compactness_LocallyCompact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Bases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_StacksAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
