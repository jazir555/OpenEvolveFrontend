// Lean compiler output
// Module: Mathlib.Topology.Instances.Int
// Imports: public import Init public import Mathlib.Data.Int.Interval public import Mathlib.Data.Int.ConditionallyCompleteOrder public import Mathlib.Topology.Instances.Discrete public import Mathlib.Topology.MetricSpace.Bounded public import Mathlib.Order.Filter.AtTopBot.Archimedean public import Mathlib.Topology.MetricSpace.Basic public import Mathlib.Topology.Order.Bornology
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
LEAN_EXPORT lean_object* lp_mathlib_Int_instDist___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_instBotUniformSpace(lean_object*);
extern lean_object* lp_mathlib_Real_instAddGroup;
lean_object* lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_1138242547____hygCtx___hyg_8_(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_2451848184____hygCtx___hyg_8_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instMetricSpace___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_abs___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Real_instIntCast;
LEAN_EXPORT lean_object* lp_mathlib_Int_instDist;
extern lean_object* lp_mathlib_Real_instDistribLattice;
LEAN_EXPORT lean_object* lp_mathlib_Int_instMetricSpace;
static lean_object* lp_mathlib_Int_instMetricSpace___closed__1;
static lean_object* lp_mathlib_Int_instMetricSpace___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_instDist___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_3 = lp_mathlib_Real_instDistribLattice;
x_4 = lp_mathlib_Real_instAddGroup;
x_5 = lp_mathlib_Real_instIntCast;
x_6 = lean_apply_1(x_5, x_1);
x_7 = lp_mathlib_Real_instIntCast;
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
static lean_object* _init_lp_mathlib_Int_instDist() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Int_instDist___lam__0), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instMetricSpace___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Int_instDist___lam__0), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instMetricSpace___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_instBotUniformSpace(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instMetricSpace___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_3 = lp_mathlib_Real_instDistribLattice;
x_4 = lp_mathlib_Real_instAddGroup;
x_5 = lp_mathlib_Real_instIntCast;
x_6 = lean_apply_1(x_5, x_1);
x_7 = lp_mathlib_Real_instIntCast;
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
static lean_object* _init_lp_mathlib_Int_instMetricSpace() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Int_instMetricSpace___closed__0;
x_2 = lean_alloc_closure((void*)(lp_mathlib_Int_instMetricSpace___lam__1), 2, 0);
x_3 = lp_mathlib_Int_instMetricSpace___closed__1;
x_4 = lean_box(0);
x_5 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_2);
lean_ctor_set(x_5, 2, x_3);
lean_ctor_set(x_5, 3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Interval(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_ConditionallyCompleteOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Instances_Discrete(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Bounded(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_AtTopBot_Archimedean(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_Bornology(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Instances_Int(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Interval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_ConditionallyCompleteOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Instances_Discrete(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_AtTopBot_Archimedean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_Bornology(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_instDist = _init_lp_mathlib_Int_instDist();
lean_mark_persistent(lp_mathlib_Int_instDist);
lp_mathlib_Int_instMetricSpace___closed__0 = _init_lp_mathlib_Int_instMetricSpace___closed__0();
lean_mark_persistent(lp_mathlib_Int_instMetricSpace___closed__0);
lp_mathlib_Int_instMetricSpace___closed__1 = _init_lp_mathlib_Int_instMetricSpace___closed__1();
lean_mark_persistent(lp_mathlib_Int_instMetricSpace___closed__1);
lp_mathlib_Int_instMetricSpace = _init_lp_mathlib_Int_instMetricSpace();
lean_mark_persistent(lp_mathlib_Int_instMetricSpace);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
