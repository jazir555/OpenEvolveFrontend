// Lean compiler output
// Module: Mathlib.Order.CompleteLatticeIntervals
// Imports: public import Init public import Mathlib.Order.ConditionallyCompleteLattice.Basic public import Mathlib.Order.LatticeIntervals public import Mathlib.Order.Interval.Set.OrdConnected
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
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Set_Iic_instLatticeElem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, lean_box(0));
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_2, lean_box(0));
x_7 = lean_apply_2(x_5, x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 2);
lean_inc(x_6);
lean_dec_ref(x_3);
lean_inc_ref(x_4);
x_7 = lp_mathlib_Set_Iic_instLatticeElem___redArg(x_4);
x_8 = !lean_is_exclusive(x_1);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_9 = lean_ctor_get(x_1, 3);
x_10 = lean_ctor_get(x_1, 2);
lean_dec(x_10);
x_11 = lean_ctor_get(x_1, 1);
lean_dec(x_11);
x_12 = lean_ctor_get(x_1, 0);
lean_dec(x_12);
x_13 = !lean_is_exclusive(x_9);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lean_ctor_get(x_9, 0);
lean_dec(x_14);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__0), 2, 1);
lean_closure_set(x_15, 0, x_5);
lean_inc(x_2);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__1), 4, 3);
lean_closure_set(x_16, 0, x_4);
lean_closure_set(x_16, 1, x_6);
lean_closure_set(x_16, 2, x_2);
lean_ctor_set(x_9, 0, x_2);
lean_ctor_set(x_1, 2, x_16);
lean_ctor_set(x_1, 1, x_15);
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_17 = lean_ctor_get(x_9, 1);
lean_inc(x_17);
lean_dec(x_9);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__0), 2, 1);
lean_closure_set(x_18, 0, x_5);
lean_inc(x_2);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__1), 4, 3);
lean_closure_set(x_19, 0, x_4);
lean_closure_set(x_19, 1, x_6);
lean_closure_set(x_19, 2, x_2);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_2);
lean_ctor_set(x_20, 1, x_17);
lean_ctor_set(x_1, 3, x_20);
lean_ctor_set(x_1, 2, x_19);
lean_ctor_set(x_1, 1, x_18);
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_21 = lean_ctor_get(x_1, 3);
lean_inc(x_21);
lean_dec(x_1);
x_22 = lean_ctor_get(x_21, 1);
lean_inc(x_22);
if (lean_is_exclusive(x_21)) {
 lean_ctor_release(x_21, 0);
 lean_ctor_release(x_21, 1);
 x_23 = x_21;
} else {
 lean_dec_ref(x_21);
 x_23 = lean_box(0);
}
x_24 = lean_alloc_closure((void*)(lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__0), 2, 1);
lean_closure_set(x_24, 0, x_5);
lean_inc(x_2);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__1), 4, 3);
lean_closure_set(x_25, 0, x_4);
lean_closure_set(x_25, 1, x_6);
lean_closure_set(x_25, 2, x_2);
if (lean_is_scalar(x_23)) {
 x_26 = lean_alloc_ctor(0, 2, 0);
} else {
 x_26 = x_23;
}
lean_ctor_set(x_26, 0, x_2);
lean_ctor_set(x_26, 1, x_22);
x_27 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_27, 0, x_7);
lean_ctor_set(x_27, 1, x_24);
lean_ctor_set(x_27, 2, x_25);
lean_ctor_set(x_27, 3, x_26);
return x_27;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Set_Iic_instCompleteLattice___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_LatticeIntervals(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_LatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
