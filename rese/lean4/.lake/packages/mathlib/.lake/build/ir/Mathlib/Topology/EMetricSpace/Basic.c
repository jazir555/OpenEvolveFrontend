// Lean compiler output
// Module: Mathlib.Topology.EMetricSpace.Basic
// Imports: public import Init public import Mathlib.Algebra.Order.BigOperators.Group.Finset public import Mathlib.Algebra.Order.Interval.Finset.SuccPred public import Mathlib.Data.Nat.SuccPred public import Mathlib.Order.Interval.Finset.Nat public import Mathlib.Topology.EMetricSpace.Defs public import Mathlib.Topology.UniformSpace.Compact public import Mathlib.Topology.UniformSpace.LocallyUniformConvergence public import Mathlib.Topology.UniformSpace.UniformEmbedding
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
lean_object* lp_mathlib_SeparationQuotient_instUniformSpace(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EMetricSpace_ofT0PseudoEMetricSpace(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_pseudoEMetricSpaceMax___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEDistSeparationQuotient___redArg(lean_object*);
lean_object* lp_mathlib_SeparationQuotient_lift_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EMetricSpace_ofT0PseudoEMetricSpace___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EMetricSpace_ofT0PseudoEMetricSpace___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EMetricSpace_ofT0PseudoEMetricSpace___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEMetricSpaceSeparationQuotient(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_emetricSpaceMax(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEMetricSpaceSeparationQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_emetricSpaceMax___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEDistSeparationQuotient(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EMetricSpace_ofT0PseudoEMetricSpace(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EMetricSpace_ofT0PseudoEMetricSpace___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EMetricSpace_ofT0PseudoEMetricSpace___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_EMetricSpace_ofT0PseudoEMetricSpace(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EMetricSpace_ofT0PseudoEMetricSpace___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_EMetricSpace_ofT0PseudoEMetricSpace___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_emetricSpaceMax(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_pseudoEMetricSpaceMax___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_emetricSpaceMax___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_pseudoEMetricSpaceMax___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEDistSeparationQuotient___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_SeparationQuotient_lift_u2082), 9, 7);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, lean_box(0));
lean_closure_set(x_5, 3, x_4);
lean_closure_set(x_5, 4, x_4);
lean_closure_set(x_5, 5, x_3);
lean_closure_set(x_5, 6, lean_box(0));
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEDistSeparationQuotient(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instEDistSeparationQuotient___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEMetricSpaceSeparationQuotient___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_2);
x_3 = lp_mathlib_instEDistSeparationQuotient___redArg(x_1);
x_4 = lp_mathlib_SeparationQuotient_instUniformSpace(lean_box(0), x_2);
lean_dec_ref(x_2);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEMetricSpaceSeparationQuotient(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instEMetricSpaceSeparationQuotient___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Interval_Finset_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_EMetricSpace_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_Compact(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_LocallyUniformConvergence(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_UniformEmbedding(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_EMetricSpace_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Interval_Finset_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_EMetricSpace_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_Compact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_LocallyUniformConvergence(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_UniformEmbedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
