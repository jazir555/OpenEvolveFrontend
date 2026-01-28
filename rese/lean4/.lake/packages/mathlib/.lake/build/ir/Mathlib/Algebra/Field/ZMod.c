// Lean compiler output
// Module: Mathlib.Algebra.Field.ZMod
// Imports: public import Init public import Mathlib.Algebra.Field.Basic public import Mathlib.Data.ZMod.Basic
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
lean_object* lp_mathlib_Rat_castRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZMod_commRing(lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZMod_inv___boxed(lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField(lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_Rat_castRec___redArg(x_1, x_2, x_3, x_5);
x_8 = lean_apply_2(x_4, x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_NNRat_castRec___redArg(x_1, x_2, x_4);
x_7 = lean_apply_2(x_3, x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
lean_inc(x_1);
x_2 = lp_mathlib_ZMod_commRing(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_2, 4);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 2);
lean_inc(x_7);
x_8 = lean_ctor_get(x_3, 3);
lean_inc(x_8);
lean_dec_ref(x_3);
x_9 = lean_ctor_get(x_4, 1);
lean_inc(x_9);
lean_dec_ref(x_4);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ZMod_inv___boxed), 2, 1);
lean_closure_set(x_10, 0, x_1);
lean_inc(x_6);
lean_inc(x_9);
x_11 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_6);
lean_ctor_set(x_11, 2, x_8);
lean_inc_ref(x_10);
x_12 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_11);
lean_closure_set(x_12, 2, x_10);
lean_inc(x_9);
lean_inc_ref(x_12);
lean_inc(x_5);
lean_inc(x_7);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ZMod_instField___redArg___lam__0), 6, 4);
lean_closure_set(x_13, 0, x_7);
lean_closure_set(x_13, 1, x_5);
lean_closure_set(x_13, 2, x_12);
lean_closure_set(x_13, 3, x_9);
lean_inc(x_9);
lean_inc_ref(x_12);
lean_inc(x_7);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ZMod_instField___redArg___lam__1), 5, 3);
lean_closure_set(x_14, 0, x_7);
lean_closure_set(x_14, 1, x_12);
lean_closure_set(x_14, 2, x_9);
lean_inc(x_9);
lean_inc(x_6);
x_15 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_15, 0, lean_box(0));
lean_closure_set(x_15, 1, x_6);
lean_closure_set(x_15, 2, x_9);
lean_inc_ref(x_10);
x_16 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_16, 0, lean_box(0));
lean_closure_set(x_16, 1, x_6);
lean_closure_set(x_16, 2, x_9);
lean_closure_set(x_16, 3, x_10);
lean_closure_set(x_16, 4, x_15);
lean_inc_ref(x_12);
lean_inc(x_7);
x_17 = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, x_7);
lean_closure_set(x_17, 2, x_12);
lean_inc_ref(x_12);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, x_7);
lean_closure_set(x_18, 2, x_5);
lean_closure_set(x_18, 3, x_12);
x_19 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_19, 0, x_2);
lean_ctor_set(x_19, 1, x_10);
lean_ctor_set(x_19, 2, x_12);
lean_ctor_set(x_19, 3, x_16);
lean_ctor_set(x_19, 4, x_17);
lean_ctor_set(x_19, 5, x_18);
lean_ctor_set(x_19, 6, x_14);
lean_ctor_set(x_19, 7, x_13);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ZMod_instField___redArg(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_ZMod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
