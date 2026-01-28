// Lean compiler output
// Module: Mathlib.Topology.EMetricSpace.Pi
// Imports: public import Init public import Mathlib.Topology.EMetricSpace.Basic public import Mathlib.Topology.UniformSpace.Pi
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
LEAN_EXPORT lean_object* lp_mathlib_emetricSpacePi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pseudoEMetricSpacePi___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEDistForall___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_emetricSpacePi___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_emetricSpacePi___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pseudoEMetricSpacePi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pseudoEMetricSpacePi___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_uniformSpace___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEDistForall___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEDistForall(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pseudoEMetricSpacePi___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_ENNReal_instOrderBot;
LEAN_EXPORT lean_object* lp_mathlib_instEDistForall___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_instSemilatticeSupENNReal;
LEAN_EXPORT lean_object* lp_mathlib_instEDistForall___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
x_5 = lean_apply_1(x_1, x_4);
lean_inc(x_4);
x_6 = lean_apply_1(x_2, x_4);
x_7 = lean_apply_3(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEDistForall___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_instEDistForall___redArg___lam__0), 4, 3);
lean_closure_set(x_7, 0, x_5);
lean_closure_set(x_7, 1, x_6);
lean_closure_set(x_7, 2, x_1);
x_8 = lp_mathlib_Finset_sup___redArg(x_2, x_3, x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEDistForall___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_instSemilatticeSupENNReal;
x_4 = lp_mathlib_ENNReal_instOrderBot;
x_5 = lean_alloc_closure((void*)(lp_mathlib_instEDistForall___redArg___lam__1), 6, 4);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_3);
lean_closure_set(x_5, 2, x_4);
lean_closure_set(x_5, 3, x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEDistForall(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instEDistForall___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pseudoEMetricSpacePi___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_1, x_2);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, x_3, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pseudoEMetricSpacePi___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pseudoEMetricSpacePi___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_pseudoEMetricSpacePi___redArg___lam__0), 4, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_pseudoEMetricSpacePi___redArg___lam__1), 2, 1);
lean_closure_set(x_4, 0, x_2);
x_5 = lp_mathlib_instEDistForall___redArg(x_1, x_3);
x_6 = lp_mathlib_Pi_uniformSpace___redArg(x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pseudoEMetricSpacePi(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_pseudoEMetricSpacePi___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_emetricSpacePi___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_emetricSpacePi___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_emetricSpacePi___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_pseudoEMetricSpacePi___redArg(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_emetricSpacePi(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_emetricSpacePi___redArg(x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_EMetricSpace_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_Pi(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_EMetricSpace_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_EMetricSpace_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
