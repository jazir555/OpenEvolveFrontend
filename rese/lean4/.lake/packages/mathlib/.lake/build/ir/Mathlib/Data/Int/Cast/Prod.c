// Lean compiler output
// Module: Mathlib.Data.Int.Cast.Prod
// Imports: public import Init public import Mathlib.Data.Int.Cast.Basic public import Mathlib.Data.Nat.Cast.Prod
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
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instAddMonoidWithOne___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_subNegMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_inc(x_3);
x_8 = lean_apply_2(x_1, x_3, x_6);
x_9 = lean_apply_2(x_2, x_3, x_7);
lean_ctor_set(x_4, 1, x_9);
lean_ctor_set(x_4, 0, x_8);
return x_4;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_4, 0);
x_11 = lean_ctor_get(x_4, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_4);
lean_inc(x_3);
x_12 = lean_apply_2(x_1, x_3, x_10);
x_13 = lean_apply_2(x_2, x_3, x_11);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_1(x_2, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 4);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_2, 4);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_4);
x_9 = lp_mathlib_Prod_instAddMonoidWithOne___redArg(x_4, x_7);
x_10 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_1);
lean_dec_ref(x_1);
x_11 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_2);
x_12 = !lean_is_exclusive(x_2);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_13 = lean_ctor_get(x_2, 4);
lean_dec(x_13);
x_14 = lean_ctor_get(x_2, 3);
lean_dec(x_14);
x_15 = lean_ctor_get(x_2, 2);
lean_dec(x_15);
x_16 = lean_ctor_get(x_2, 1);
lean_dec(x_16);
x_17 = lean_ctor_get(x_2, 0);
lean_dec(x_17);
x_18 = lp_mathlib_Prod_subNegMonoid___redArg(x_10, x_11);
x_19 = lean_ctor_get(x_18, 1);
lean_inc(x_19);
x_20 = lean_ctor_get(x_18, 2);
lean_inc(x_20);
lean_dec_ref(x_18);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Prod_instAddGroupWithOne___redArg___lam__0), 4, 2);
lean_closure_set(x_21, 0, x_5);
lean_closure_set(x_21, 1, x_8);
x_22 = lean_alloc_closure((void*)(lp_mathlib_Prod_instAddGroupWithOne___redArg___lam__1), 3, 2);
lean_closure_set(x_22, 0, x_3);
lean_closure_set(x_22, 1, x_6);
lean_ctor_set(x_2, 4, x_21);
lean_ctor_set(x_2, 3, x_20);
lean_ctor_set(x_2, 2, x_19);
lean_ctor_set(x_2, 1, x_9);
lean_ctor_set(x_2, 0, x_22);
return x_2;
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
lean_dec(x_2);
x_23 = lp_mathlib_Prod_subNegMonoid___redArg(x_10, x_11);
x_24 = lean_ctor_get(x_23, 1);
lean_inc(x_24);
x_25 = lean_ctor_get(x_23, 2);
lean_inc(x_25);
lean_dec_ref(x_23);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Prod_instAddGroupWithOne___redArg___lam__0), 4, 2);
lean_closure_set(x_26, 0, x_5);
lean_closure_set(x_26, 1, x_8);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Prod_instAddGroupWithOne___redArg___lam__1), 3, 2);
lean_closure_set(x_27, 0, x_3);
lean_closure_set(x_27, 1, x_6);
x_28 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_28, 0, x_27);
lean_ctor_set(x_28, 1, x_9);
lean_ctor_set(x_28, 2, x_24);
lean_ctor_set(x_28, 3, x_25);
lean_ctor_set(x_28, 4, x_26);
return x_28;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instAddGroupWithOne___redArg(x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
