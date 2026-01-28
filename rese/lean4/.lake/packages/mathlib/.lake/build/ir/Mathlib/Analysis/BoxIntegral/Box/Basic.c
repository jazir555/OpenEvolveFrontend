// Lean compiler output
// Module: Mathlib.Analysis.BoxIntegral.Box.Basic
// Imports: public import Init public import Mathlib.Data.NNReal.Basic public import Mathlib.Order.Fin.Tuple public import Mathlib.Order.Interval.Set.Monotone public import Mathlib.Topology.MetricSpace.Basic public import Mathlib.Topology.MetricSpace.Bounded public import Mathlib.Topology.MetricSpace.Pseudo.Real public import Mathlib.Topology.Order.MonotoneConvergence
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
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_3793047190____hygCtx___hyg_8_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_withBotCoe(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_Icc(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_BoxIntegral_Box_instSemilatticeSup___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instSemilatticeSup___lam__0(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Real_definition_00___x40_Mathlib_Data_Real_Basic_1279875089____hygCtx___hyg_8_;
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instPartialOrder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instSemilatticeSup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instInhabited___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instCoeTCSetForallReal(lean_object*);
extern lean_object* lp_mathlib_Real_definition_00___x40_Mathlib_Data_Real_Basic_1850581184____hygCtx___hyg_8_;
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instMembershipForallReal(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instSemilatticeSup___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instSemilatticeSup___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_Ioo(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instLE(lean_object*);
lean_object* lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_1934218611____hygCtx___hyg_8_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instInhabited___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instInhabited___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_BoxIntegral_Box_instPartialOrder___closed__0;
lean_object* lp_mathlib_Fin_succAbove___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instInhabited___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instInhabited___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Real_definition_00___x40_Mathlib_Data_Real_Basic_1850581184____hygCtx___hyg_8_;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instInhabited___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Real_definition_00___x40_Mathlib_Data_Real_Basic_1279875089____hygCtx___hyg_8_;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instInhabited___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_BoxIntegral_Box_instInhabited___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instInhabited___lam__1___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_BoxIntegral_Box_instInhabited___lam__1(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instInhabited(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_BoxIntegral_Box_instInhabited___lam__0___boxed), 1, 0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_BoxIntegral_Box_instInhabited___lam__1___boxed), 1, 0);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instMembershipForallReal(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instCoeTCSetForallReal(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instLE(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
static lean_object* _init_lp_mathlib_BoxIntegral_Box_instPartialOrder___closed__0() {
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
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instPartialOrder(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_BoxIntegral_Box_instPartialOrder___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_Icc(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
static lean_object* _init_lp_mathlib_BoxIntegral_Box_instSemilatticeSup___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_BoxIntegral_Box_instPartialOrder(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instSemilatticeSup___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_1(x_2, x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_3793047190____hygCtx___hyg_8_), 3, 2);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instSemilatticeSup___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_1(x_2, x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_1934218611____hygCtx___hyg_8_), 3, 2);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instSemilatticeSup___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_BoxIntegral_Box_instSemilatticeSup___lam__0), 3, 2);
lean_closure_set(x_8, 0, x_4);
lean_closure_set(x_8, 1, x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_BoxIntegral_Box_instSemilatticeSup___lam__1), 3, 2);
lean_closure_set(x_9, 0, x_3);
lean_closure_set(x_9, 1, x_6);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_9);
return x_2;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib_BoxIntegral_Box_instSemilatticeSup___lam__0), 3, 2);
lean_closure_set(x_12, 0, x_4);
lean_closure_set(x_12, 1, x_11);
x_13 = lean_alloc_closure((void*)(lp_mathlib_BoxIntegral_Box_instSemilatticeSup___lam__1), 3, 2);
lean_closure_set(x_13, 0, x_3);
lean_closure_set(x_13, 1, x_10);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_12);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_instSemilatticeSup(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_BoxIntegral_Box_instSemilatticeSup___lam__2), 2, 0);
x_3 = lp_mathlib_BoxIntegral_Box_instSemilatticeSup___closed__0;
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_withBotCoe(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Fin_succAbove___redArg(x_1, x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_BoxIntegral_Box_face___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Fin_succAbove___redArg(x_1, x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_BoxIntegral_Box_face___redArg___lam__1(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_BoxIntegral_Box_face___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_BoxIntegral_Box_face___redArg___lam__1___boxed), 3, 2);
lean_closure_set(x_7, 0, x_2);
lean_closure_set(x_7, 1, x_4);
lean_ctor_set(x_1, 1, x_6);
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_1);
lean_inc(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_BoxIntegral_Box_face___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_10, 0, x_2);
lean_closure_set(x_10, 1, x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_BoxIntegral_Box_face___redArg___lam__1___boxed), 3, 2);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_8);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_10);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_BoxIntegral_Box_face___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_face___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_BoxIntegral_Box_face(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoxIntegral_Box_Ioo(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_NNReal_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Fin_Tuple(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Monotone(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Bounded(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Pseudo_Real(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Order_MonotoneConvergence(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_BoxIntegral_Box_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_NNReal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Fin_Tuple(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Monotone(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Pseudo_Real(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Order_MonotoneConvergence(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_BoxIntegral_Box_instPartialOrder___closed__0 = _init_lp_mathlib_BoxIntegral_Box_instPartialOrder___closed__0();
lean_mark_persistent(lp_mathlib_BoxIntegral_Box_instPartialOrder___closed__0);
lp_mathlib_BoxIntegral_Box_instSemilatticeSup___closed__0 = _init_lp_mathlib_BoxIntegral_Box_instSemilatticeSup___closed__0();
lean_mark_persistent(lp_mathlib_BoxIntegral_Box_instSemilatticeSup___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
