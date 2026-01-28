// Lean compiler output
// Module: Mathlib.Analysis.Normed.Group.Uniform
// Imports: public import Init public import Mathlib.Analysis.Normed.Group.Continuity public import Mathlib.Topology.Algebra.IsUniformGroup.Basic public import Mathlib.Topology.MetricSpace.Algebra public import Mathlib.Topology.MetricSpace.IsometricSMul
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
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNormedAddCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNormedCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instMulNorm(lean_object*, lean_object*);
lean_object* lp_mathlib_SeparationQuotient_instAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instMulNorm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNorm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNormedCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_SeparationQuotient_instMetricSpace___redArg(lean_object*);
lean_object* lp_mathlib_SeparationQuotient_instCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNormedAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNorm(lean_object*, lean_object*);
lean_object* lp_mathlib_SeparationQuotient_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instMulNorm___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lean_ctor_get(x_1, 2);
x_3 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_SeparationQuotient_lift), 6, 5);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_5);
lean_closure_set(x_6, 3, x_4);
lean_closure_set(x_6, 4, lean_box(0));
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instMulNorm(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SeparationQuotient_instMulNorm___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNorm___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lean_ctor_get(x_1, 2);
x_3 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_SeparationQuotient_lift), 6, 5);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_5);
lean_closure_set(x_6, 3, x_4);
lean_closure_set(x_6, 4, lean_box(0));
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNorm(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SeparationQuotient_instNorm___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNormedCommGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lean_ctor_get(x_1, 1);
x_3 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_SeparationQuotient_instCommGroup___redArg(x_2);
x_5 = lp_mathlib_SeparationQuotient_instMulNorm___redArg(x_1);
x_6 = lp_mathlib_SeparationQuotient_instMetricSpace___redArg(x_3);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_4);
lean_ctor_set(x_7, 2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNormedCommGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SeparationQuotient_instNormedCommGroup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNormedAddCommGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lean_ctor_get(x_1, 1);
x_3 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_SeparationQuotient_instAddCommGroup___redArg(x_2);
x_5 = lp_mathlib_SeparationQuotient_instNorm___redArg(x_1);
x_6 = lp_mathlib_SeparationQuotient_instMetricSpace___redArg(x_3);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_4);
lean_ctor_set(x_7, 2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SeparationQuotient_instNormedAddCommGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SeparationQuotient_instNormedAddCommGroup___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_Continuity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_IsUniformGroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Algebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_IsometricSMul(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_Uniform(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Group_Continuity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_IsUniformGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Algebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_IsometricSMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
