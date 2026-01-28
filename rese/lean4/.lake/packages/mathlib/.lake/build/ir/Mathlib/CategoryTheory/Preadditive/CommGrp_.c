// Lean compiler output
// Module: Mathlib.CategoryTheory.Preadditive.CommGrp_
// Imports: public import Init public import Mathlib.CategoryTheory.Monoidal.CommGrp_ public import Mathlib.CategoryTheory.Preadditive.Biproducts
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
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_instGrpObj___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CommGrp_forget___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalence(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CommGrp_instCategory___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalenceAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalence___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalenceAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_category___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalenceAux___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CommGrp_mkIso_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_instGrpObj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_instGrpObj___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_3);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_5, 2);
lean_inc(x_7);
x_8 = lean_ctor_get(x_5, 3);
lean_inc(x_8);
lean_dec_ref(x_5);
x_9 = lean_ctor_get(x_6, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_6, 4);
lean_inc(x_10);
lean_dec_ref(x_6);
lean_inc_ref(x_2);
lean_inc(x_4);
x_11 = lean_apply_2(x_2, x_10, x_4);
x_12 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_11);
lean_dec_ref(x_11);
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; 
x_14 = lean_ctor_get(x_12, 1);
lean_dec(x_14);
lean_inc_n(x_4, 2);
x_15 = lean_apply_2(x_9, x_4, x_4);
lean_inc_ref(x_2);
lean_inc(x_4);
x_16 = lean_apply_2(x_2, x_15, x_4);
x_17 = lean_ctor_get(x_16, 0);
lean_inc_ref(x_17);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
lean_inc_n(x_4, 2);
x_19 = lean_apply_2(x_7, x_4, x_4);
lean_inc_n(x_4, 2);
x_20 = lean_apply_2(x_8, x_4, x_4);
x_21 = lean_apply_2(x_18, x_19, x_20);
lean_ctor_set(x_12, 1, x_21);
lean_inc_n(x_4, 2);
x_22 = lean_apply_2(x_2, x_4, x_4);
x_23 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_22);
lean_dec_ref(x_22);
x_24 = !lean_is_exclusive(x_23);
if (x_24 == 0)
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_25 = lean_ctor_get(x_23, 1);
x_26 = lean_ctor_get(x_23, 0);
lean_dec(x_26);
x_27 = lean_ctor_get(x_1, 1);
lean_inc(x_27);
lean_dec_ref(x_1);
x_28 = lean_apply_1(x_27, x_4);
x_29 = lean_apply_1(x_25, x_28);
lean_ctor_set(x_23, 1, x_29);
lean_ctor_set(x_23, 0, x_12);
return x_23;
}
else
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_30 = lean_ctor_get(x_23, 1);
lean_inc(x_30);
lean_dec(x_23);
x_31 = lean_ctor_get(x_1, 1);
lean_inc(x_31);
lean_dec_ref(x_1);
x_32 = lean_apply_1(x_31, x_4);
x_33 = lean_apply_1(x_30, x_32);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_12);
lean_ctor_set(x_34, 1, x_33);
return x_34;
}
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; 
x_35 = lean_ctor_get(x_12, 0);
lean_inc(x_35);
lean_dec(x_12);
lean_inc_n(x_4, 2);
x_36 = lean_apply_2(x_9, x_4, x_4);
lean_inc_ref(x_2);
lean_inc(x_4);
x_37 = lean_apply_2(x_2, x_36, x_4);
x_38 = lean_ctor_get(x_37, 0);
lean_inc_ref(x_38);
lean_dec_ref(x_37);
x_39 = lean_ctor_get(x_38, 0);
lean_inc(x_39);
lean_dec_ref(x_38);
lean_inc_n(x_4, 2);
x_40 = lean_apply_2(x_7, x_4, x_4);
lean_inc_n(x_4, 2);
x_41 = lean_apply_2(x_8, x_4, x_4);
x_42 = lean_apply_2(x_39, x_40, x_41);
x_43 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_43, 0, x_35);
lean_ctor_set(x_43, 1, x_42);
lean_inc_n(x_4, 2);
x_44 = lean_apply_2(x_2, x_4, x_4);
x_45 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_44);
lean_dec_ref(x_44);
x_46 = lean_ctor_get(x_45, 1);
lean_inc(x_46);
if (lean_is_exclusive(x_45)) {
 lean_ctor_release(x_45, 0);
 lean_ctor_release(x_45, 1);
 x_47 = x_45;
} else {
 lean_dec_ref(x_45);
 x_47 = lean_box(0);
}
x_48 = lean_ctor_get(x_1, 1);
lean_inc(x_48);
lean_dec_ref(x_1);
x_49 = lean_apply_1(x_48, x_4);
x_50 = lean_apply_1(x_46, x_49);
if (lean_is_scalar(x_47)) {
 x_51 = lean_alloc_ctor(0, 2, 0);
} else {
 x_51 = x_47;
}
lean_ctor_set(x_51, 0, x_43);
lean_ctor_set(x_51, 1, x_50);
return x_51;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_instGrpObj(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Preadditive_instGrpObj___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
lean_inc(x_4);
x_5 = lp_mathlib_CategoryTheory_Preadditive_instGrpObj___redArg(x_1, x_2, x_3, x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg___lam__1(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg___lam__0), 4, 3);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
lean_closure_set(x_4, 2, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg___lam__1___boxed), 3, 0);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg(x_2, x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_toCommGrp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Preadditive_toCommGrp(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalenceAux___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
lean_dec_ref(x_2);
lean_inc_ref(x_4);
x_7 = lean_apply_1(x_5, x_4);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 1);
lean_inc_ref(x_9);
lean_dec_ref(x_7);
x_10 = lean_apply_1(x_6, x_4);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_10, 1);
lean_inc_ref(x_12);
lean_dec_ref(x_10);
lean_inc(x_8);
x_13 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_3, x_8);
x_14 = lp_mathlib_CategoryTheory_CommGrp_mkIso_x27___redArg(x_8, x_11, x_13, x_9, x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalenceAux___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_5 = lp_mathlib_CategoryTheory_CommGrp_instCategory___redArg(x_1, x_3, x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_6 = lp_mathlib_CategoryTheory_CommGrp_forget___redArg(x_1, x_3, x_4);
lean_inc_ref(x_1);
x_7 = lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg(x_1, x_2, x_3);
x_8 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_6, x_7);
x_9 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_5);
lean_dec_ref(x_5);
x_10 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalenceAux___redArg___lam__0), 4, 3);
lean_closure_set(x_10, 0, x_8);
lean_closure_set(x_10, 1, x_9);
lean_closure_set(x_10, 2, x_1);
x_11 = lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalenceAux(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalenceAux___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalence___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_5 = lp_mathlib_CategoryTheory_Preadditive_toCommGrp___redArg(x_1, x_2, x_3);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_6 = lp_mathlib_CategoryTheory_CommGrp_forget___redArg(x_1, x_3, x_4);
lean_inc_ref(x_1);
x_7 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_1);
x_8 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_1);
x_9 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_7, x_8);
x_10 = lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalenceAux___redArg(x_1, x_2, x_3, x_4);
x_11 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_11, 0, x_5);
lean_ctor_set(x_11, 1, x_6);
lean_ctor_set(x_11, 2, x_9);
lean_ctor_set(x_11, 3, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalence(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Preadditive_commGrpEquivalence___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_CommGrp__(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Preadditive_Biproducts(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Preadditive_CommGrp__(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_CommGrp__(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Preadditive_Biproducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
