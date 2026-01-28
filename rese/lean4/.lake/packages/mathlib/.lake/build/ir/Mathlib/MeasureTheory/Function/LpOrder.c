// Lean compiler output
// Module: Mathlib.MeasureTheory.Function.LpOrder
// Imports: public import Init public import Mathlib.Analysis.Normed.Order.Lattice public import Mathlib.MeasureTheory.Function.ConvergenceInMeasure public import Mathlib.MeasureTheory.Function.LpSpace.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MeasureTheory_AEEqFun_instLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MeasureTheory_AEEqFun_comp_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__3(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NormedAddCommGroup_toSeminormedAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_Subtype_partialOrder(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__1), 3, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lp_mathlib_MeasureTheory_AEEqFun_comp_u2082___redArg(x_5, x_2, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MeasureTheory_AEEqFun_comp_u2082___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_5 = lp_mathlib_NormedAddCommGroup_toSeminormedAddCommGroup___redArg(x_3);
x_6 = lean_ctor_get(x_5, 2);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_7, 1);
lean_dec(x_10);
lean_inc_ref(x_4);
x_11 = lp_mathlib_MeasureTheory_AEEqFun_instLattice___redArg(x_1, x_2, x_9, x_4);
x_12 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_11);
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_14 = lean_ctor_get(x_12, 0);
x_15 = lean_ctor_get(x_12, 1);
lean_dec(x_15);
lean_inc_ref(x_4);
x_16 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__0), 3, 1);
lean_closure_set(x_16, 0, x_4);
x_17 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__2), 3, 1);
lean_closure_set(x_17, 0, x_4);
x_18 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__3), 3, 1);
lean_closure_set(x_18, 0, x_16);
x_19 = lp_mathlib_Subtype_partialOrder(lean_box(0), x_14, lean_box(0));
lean_dec_ref(x_14);
lean_ctor_set(x_12, 1, x_17);
lean_ctor_set(x_12, 0, x_19);
lean_ctor_set(x_7, 1, x_18);
lean_ctor_set(x_7, 0, x_12);
return x_7;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_20 = lean_ctor_get(x_12, 0);
lean_inc(x_20);
lean_dec(x_12);
lean_inc_ref(x_4);
x_21 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__0), 3, 1);
lean_closure_set(x_21, 0, x_4);
x_22 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__2), 3, 1);
lean_closure_set(x_22, 0, x_4);
x_23 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__3), 3, 1);
lean_closure_set(x_23, 0, x_21);
x_24 = lp_mathlib_Subtype_partialOrder(lean_box(0), x_20, lean_box(0));
lean_dec_ref(x_20);
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_24);
lean_ctor_set(x_25, 1, x_22);
lean_ctor_set(x_7, 1, x_23);
lean_ctor_set(x_7, 0, x_25);
return x_7;
}
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_26 = lean_ctor_get(x_7, 0);
lean_inc(x_26);
lean_dec(x_7);
lean_inc_ref(x_4);
x_27 = lp_mathlib_MeasureTheory_AEEqFun_instLattice___redArg(x_1, x_2, x_26, x_4);
x_28 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_27);
x_29 = lean_ctor_get(x_28, 0);
lean_inc_ref(x_29);
if (lean_is_exclusive(x_28)) {
 lean_ctor_release(x_28, 0);
 lean_ctor_release(x_28, 1);
 x_30 = x_28;
} else {
 lean_dec_ref(x_28);
 x_30 = lean_box(0);
}
lean_inc_ref(x_4);
x_31 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__0), 3, 1);
lean_closure_set(x_31, 0, x_4);
x_32 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__2), 3, 1);
lean_closure_set(x_32, 0, x_4);
x_33 = lean_alloc_closure((void*)(lp_mathlib_MeasureTheory_Lp_instLattice___redArg___lam__3), 3, 1);
lean_closure_set(x_33, 0, x_31);
x_34 = lp_mathlib_Subtype_partialOrder(lean_box(0), x_29, lean_box(0));
lean_dec_ref(x_29);
if (lean_is_scalar(x_30)) {
 x_35 = lean_alloc_ctor(0, 2, 0);
} else {
 x_35 = x_30;
}
lean_ctor_set(x_35, 0, x_34);
lean_ctor_set(x_35, 1, x_32);
x_36 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_36, 0, x_35);
lean_ctor_set(x_36, 1, x_33);
return x_36;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MeasureTheory_Lp_instLattice___redArg(x_3, x_4, x_6, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MeasureTheory_Lp_instLattice(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MeasureTheory_Lp_instLattice___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_MeasureTheory_Lp_instLattice___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Order_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Function_ConvergenceInMeasure(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_MeasureTheory_Function_LpSpace_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_MeasureTheory_Function_LpOrder(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Order_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Function_ConvergenceInMeasure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_MeasureTheory_Function_LpSpace_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
