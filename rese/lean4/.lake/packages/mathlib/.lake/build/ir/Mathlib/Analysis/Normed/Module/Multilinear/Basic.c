// Lean compiler output
// Module: Mathlib.Analysis.Normed.Module.Multilinear.Basic
// Imports: public import Init public import Mathlib.Analysis.Normed.Operator.NormedSpace public import Mathlib.Logic.Embedding.Basic public import Mathlib.Data.Fintype.CardEmbedding public import Mathlib.Topology.Algebra.MetricSpace.Lipschitz public import Mathlib.Topology.Algebra.Module.Multilinear.Topology
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
lean_object* lp_mathlib_ContinuousMultilinearMap_mkPiRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NormedField_toNormedCommRing___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_ContinuousMultilinearMap_mkPiRing___redArg(x_1, x_2, x_3, x_4);
x_7 = lean_apply_1(x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_4);
x_5 = lp_mathlib_NormedField_toNormedCommRing___redArg(x_1);
x_6 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_4);
x_7 = lean_ctor_get(x_5, 1);
lean_inc_ref(x_7);
lean_dec_ref(x_5);
x_8 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_7);
x_9 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_ctor_get(x_9, 2);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_11, 0, x_3);
lean_closure_set(x_11, 1, x_6);
lean_closure_set(x_11, 2, x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_12, 0, x_10);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg___lam__2), 2, 1);
lean_closure_set(x_13, 0, x_12);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_11);
lean_ctor_set(x_14, 1, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___redArg(x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMultilinearMap_piFieldEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_ContinuousMultilinearMap_piFieldEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_5);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Operator_NormedSpace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_CardEmbedding(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_MetricSpace_Lipschitz(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Module_Multilinear_Topology(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Module_Multilinear_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Operator_NormedSpace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_CardEmbedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_MetricSpace_Lipschitz(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Module_Multilinear_Topology(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
