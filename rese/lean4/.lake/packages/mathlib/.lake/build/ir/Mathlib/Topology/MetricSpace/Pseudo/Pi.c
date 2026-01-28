// Lean compiler output
// Module: Mathlib.Topology.MetricSpace.Pseudo.Pi
// Imports: public import Init public import Mathlib.Data.ENNReal.Lemmas public import Mathlib.Topology.Bornology.Constructions public import Mathlib.Topology.EMetricSpace.Pi public import Mathlib.Topology.MetricSpace.Pseudo.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_pseudoEMetricSpacePi___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PseudoMetricSpace_toPseudoEMetricSpace___redArg(lean_object*);
extern lean_object* lp_mathlib_Real_definition_00___x40_Mathlib_Data_Real_Basic_1850581184____hygCtx___hyg_8_;
extern lean_object* lp_mathlib_instSemilatticeSupNNReal;
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_ENNReal_instOrderBot;
extern lean_object* lp_mathlib_instSemilatticeSupENNReal;
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_PseudoMetricSpace_toPseudoEMetricSpace___redArg(x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_4);
x_5 = lean_apply_1(x_1, x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
lean_inc(x_4);
x_7 = lean_apply_1(x_2, x_4);
x_8 = lean_apply_1(x_3, x_4);
x_9 = lean_apply_2(x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_pseudoMetricSpacePi___redArg___lam__1), 4, 3);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_5);
lean_closure_set(x_7, 2, x_6);
x_8 = lp_mathlib_Finset_sup___redArg(x_2, x_3, x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_4);
x_5 = lean_apply_1(x_1, x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
lean_inc(x_4);
x_7 = lean_apply_1(x_2, x_4);
x_8 = lean_apply_1(x_3, x_4);
x_9 = lean_apply_2(x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_pseudoMetricSpacePi___redArg___lam__3), 4, 3);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_5);
lean_closure_set(x_7, 2, x_6);
x_8 = lp_mathlib_Finset_sup___redArg(x_2, x_3, x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_pseudoMetricSpacePi___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_instSemilatticeSupNNReal;
lean_inc_ref(x_3);
lean_inc(x_1);
x_5 = lp_mathlib_pseudoEMetricSpacePi___redArg(x_1, x_3);
x_6 = lp_mathlib_instSemilatticeSupENNReal;
x_7 = lean_ctor_get(x_5, 1);
lean_inc_ref(x_7);
lean_dec_ref(x_5);
x_8 = lp_mathlib_Real_definition_00___x40_Mathlib_Data_Real_Basic_1850581184____hygCtx___hyg_8_;
lean_inc(x_1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_pseudoMetricSpacePi___redArg___lam__2), 6, 4);
lean_closure_set(x_9, 0, x_2);
lean_closure_set(x_9, 1, x_4);
lean_closure_set(x_9, 2, x_8);
lean_closure_set(x_9, 3, x_1);
x_10 = lp_mathlib_ENNReal_instOrderBot;
x_11 = lean_alloc_closure((void*)(lp_mathlib_pseudoMetricSpacePi___redArg___lam__4), 6, 4);
lean_closure_set(x_11, 0, x_3);
lean_closure_set(x_11, 1, x_6);
lean_closure_set(x_11, 2, x_10);
lean_closure_set(x_11, 3, x_1);
x_12 = lean_box(0);
x_13 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_13, 0, x_9);
lean_ctor_set(x_13, 1, x_11);
lean_ctor_set(x_13, 2, x_7);
lean_ctor_set(x_13, 3, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pseudoMetricSpacePi(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_pseudoMetricSpacePi___redArg(x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ENNReal_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Bornology_Constructions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_EMetricSpace_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Pseudo_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Pseudo_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENNReal_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Bornology_Constructions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_EMetricSpace_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Pseudo_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
