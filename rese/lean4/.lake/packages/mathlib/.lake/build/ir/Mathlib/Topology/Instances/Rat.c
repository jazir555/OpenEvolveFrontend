// Lean compiler output
// Module: Mathlib.Topology.Instances.Rat
// Imports: public import Init public import Mathlib.Algebra.Algebra.Rat public import Mathlib.Algebra.Module.Rat public import Mathlib.Data.NNRat.Order public import Mathlib.Topology.Algebra.Order.Archimedean public import Mathlib.Topology.Algebra.Ring.Real public import Mathlib.Topology.Instances.Nat
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
extern lean_object* lp_mathlib_Real_instRatCast;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instMetricSpace;
extern lean_object* lp_mathlib_Real_instAddGroup;
lean_object* lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_1138242547____hygCtx___hyg_8_(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_metricSpace___redArg(lean_object*);
lean_object* lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_2451848184____hygCtx___hyg_8_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instMetricSpace___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_abs___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instMetricSpace;
extern lean_object* lp_mathlib_Real_instDistribLattice;
static lean_object* lp_mathlib_Rat_instMetricSpace___closed__0;
static lean_object* lp_mathlib_NNRat_instMetricSpace___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instMetricSpace___lam__1(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Rat_instMetricSpace___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instMetricSpace___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_3 = lp_mathlib_Real_instDistribLattice;
x_4 = lp_mathlib_Real_instAddGroup;
x_5 = lp_mathlib_Real_instRatCast;
x_6 = lean_apply_1(x_5, x_1);
x_7 = lp_mathlib_Real_instRatCast;
x_8 = lean_apply_1(x_7, x_2);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_2451848184____hygCtx___hyg_8_), 2, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_1138242547____hygCtx___hyg_8_), 3, 2);
lean_closure_set(x_10, 0, x_6);
lean_closure_set(x_10, 1, x_9);
x_11 = lp_mathlib_abs___redArg(x_3, x_4, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instMetricSpace___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_3 = lp_mathlib_Real_instDistribLattice;
x_4 = lp_mathlib_Real_instAddGroup;
x_5 = lp_mathlib_Real_instRatCast;
x_6 = lean_apply_1(x_5, x_1);
x_7 = lp_mathlib_Real_instRatCast;
x_8 = lean_apply_1(x_7, x_2);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_2451848184____hygCtx___hyg_8_), 2, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_1138242547____hygCtx___hyg_8_), 3, 2);
lean_closure_set(x_10, 0, x_6);
lean_closure_set(x_10, 1, x_9);
x_11 = lp_mathlib_abs___redArg(x_3, x_4, x_10);
x_12 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
}
static lean_object* _init_lp_mathlib_Rat_instMetricSpace() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Rat_instMetricSpace___lam__0), 2, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Rat_instMetricSpace___lam__1), 2, 0);
x_3 = lean_box(0);
x_4 = lp_mathlib_Rat_instMetricSpace___closed__0;
x_5 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_2);
lean_ctor_set(x_5, 2, x_4);
lean_ctor_set(x_5, 3, x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_NNRat_instMetricSpace___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Rat_instMetricSpace;
x_2 = lp_mathlib_Subtype_metricSpace___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_NNRat_instMetricSpace() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_NNRat_instMetricSpace___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_NNRat_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Order_Archimedean(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Ring_Real(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Instances_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Instances_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_NNRat_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Order_Archimedean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Ring_Real(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Instances_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_instMetricSpace___closed__0 = _init_lp_mathlib_Rat_instMetricSpace___closed__0();
lean_mark_persistent(lp_mathlib_Rat_instMetricSpace___closed__0);
lp_mathlib_Rat_instMetricSpace = _init_lp_mathlib_Rat_instMetricSpace();
lean_mark_persistent(lp_mathlib_Rat_instMetricSpace);
lp_mathlib_NNRat_instMetricSpace___closed__0 = _init_lp_mathlib_NNRat_instMetricSpace___closed__0();
lean_mark_persistent(lp_mathlib_NNRat_instMetricSpace___closed__0);
lp_mathlib_NNRat_instMetricSpace = _init_lp_mathlib_NNRat_instMetricSpace();
lean_mark_persistent(lp_mathlib_NNRat_instMetricSpace);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
