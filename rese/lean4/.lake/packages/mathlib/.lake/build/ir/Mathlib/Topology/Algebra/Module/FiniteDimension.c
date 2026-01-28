// Lean compiler output
// Module: Mathlib.Topology.Algebra.Module.FiniteDimension
// Imports: public import Init public import Mathlib.Analysis.LocallyConvex.BalancedCoreHull public import Mathlib.Analysis.Normed.Module.Basic public import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas public import Mathlib.RingTheory.LocalRing.Basic public import Mathlib.Topology.Algebra.Module.Determinant public import Mathlib.Topology.Algebra.Module.ModuleTopology public import Mathlib.Topology.Algebra.Module.Simple public import Mathlib.Topology.Algebra.SeparationQuotient.FiniteDimensional
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
LEAN_EXPORT lean_object* lp_mathlib_Module_End_toContinuousLinearMap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toContinuousLinearEquiv___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_toContinuousLinearMap___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toContinuousLinearEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toContinuousLinearMap___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_toContinuousLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toContinuousLinearEquiv___redArg(lean_object*);
lean_object* lp_mathlib_NormedAddCommGroup_toSeminormedAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toContinuousLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_toContinuousLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toContinuousLinearMap___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toContinuousLinearMap___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toContinuousLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toContinuousLinearMap___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toContinuousLinearMap___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toContinuousLinearMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_18 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toContinuousLinearMap___lam__0), 2, 0);
x_19 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toContinuousLinearMap___lam__1), 2, 0);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_19);
lean_ctor_set(x_20, 1, x_18);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toContinuousLinearMap___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_LinearMap_toContinuousLinearMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_toContinuousLinearMap___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_4 = lp_mathlib_NormedAddCommGroup_toSeminormedAddCommGroup___redArg(x_2);
x_5 = lean_ctor_get(x_4, 2);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_5, 2);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_7);
lean_dec_ref(x_4);
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lp_mathlib_LinearMap_toContinuousLinearMap(lean_box(0), x_1, lean_box(0), x_7, x_3, x_8, lean_box(0), lean_box(0), lean_box(0), x_7, x_3, x_8, lean_box(0), lean_box(0), lean_box(0), lean_box(0), lean_box(0));
lean_dec_ref(x_7);
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
return x_9;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_11 = lean_ctor_get(x_9, 0);
x_12 = lean_ctor_get(x_9, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_9);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_toContinuousLinearMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Module_End_toContinuousLinearMap___redArg(x_2, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_toContinuousLinearMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Module_End_toContinuousLinearMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_toContinuousLinearMap___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Module_End_toContinuousLinearMap___redArg(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toContinuousLinearEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_inc_ref(x_19);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toContinuousLinearEquiv___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toContinuousLinearEquiv___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
lean_object* x_19 = _args[18];
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_LinearEquiv_toContinuousLinearEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec_ref(x_19);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toContinuousLinearEquiv___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_LinearEquiv_toContinuousLinearEquiv___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_LocallyConvex_BalancedCoreHull(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Module_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_FiniteDimensional_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_LocalRing_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Module_Determinant(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Module_ModuleTopology(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Module_Simple(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_SeparationQuotient_FiniteDimensional(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Module_FiniteDimension(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_LocallyConvex_BalancedCoreHull(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Module_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_FiniteDimensional_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_LocalRing_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Module_Determinant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Module_ModuleTopology(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Module_Simple(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_SeparationQuotient_FiniteDimensional(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
