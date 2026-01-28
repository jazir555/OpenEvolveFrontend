// Lean compiler output
// Module: Mathlib.Topology.DiscreteSubset
// Imports: public import Init public import Mathlib.Tactic.TautoSet public import Mathlib.Topology.Constructions public import Mathlib.Data.Set.Subset public import Mathlib.Topology.Separation.Basic
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
static lean_object* lp_mathlib_Filter_codiscreteWithin___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Filter_codiscreteWithin(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Filter_instSupSet___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_codiscrete(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Filter_codiscreteWithin___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Filter_instSupSet___lam__0(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_codiscreteWithin(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Filter_codiscreteWithin___closed__0;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_codiscrete(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Filter_codiscreteWithin___closed__0;
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TautoSet(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Constructions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Subset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Separation_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_DiscreteSubset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TautoSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Constructions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Subset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Separation_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Filter_codiscreteWithin___closed__0 = _init_lp_mathlib_Filter_codiscreteWithin___closed__0();
lean_mark_persistent(lp_mathlib_Filter_codiscreteWithin___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
