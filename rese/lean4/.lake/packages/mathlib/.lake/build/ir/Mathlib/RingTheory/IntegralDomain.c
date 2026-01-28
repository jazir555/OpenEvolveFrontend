// Lean compiler output
// Module: Mathlib.RingTheory.IntegralDomain
// Imports: public import Init public import Mathlib.Algebra.Polynomial.Roots public import Mathlib.Data.Fintype.Inv public import Mathlib.GroupTheory.SpecificGroups.Cyclic public import Mathlib.Tactic.FieldSimp
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
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_fieldOfDomain___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_fieldOfDomain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_bijInv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; uint8_t x_8; 
lean_inc_ref(x_1);
lean_inc(x_2);
lean_inc(x_6);
x_7 = lean_apply_2(x_1, x_6, x_2);
x_8 = lean_unbox(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_dec(x_2);
x_9 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_3);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__0), 3, 2);
lean_closure_set(x_11, 0, x_4);
lean_closure_set(x_11, 1, x_6);
x_12 = lp_mathlib_Fintype_bijInv___redArg(x_5, x_1, x_11, x_10);
return x_12;
}
else
{
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_inc_ref(x_1);
x_4 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_1);
lean_inc_ref(x_4);
x_5 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_4);
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec_ref(x_5);
x_9 = lean_ctor_get(x_6, 0);
x_10 = lean_ctor_get(x_6, 1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__1), 6, 5);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_8);
lean_closure_set(x_11, 2, x_4);
lean_closure_set(x_11, 3, x_7);
lean_closure_set(x_11, 4, x_3);
lean_inc_ref(x_11);
lean_inc_ref(x_6);
x_12 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_6);
lean_closure_set(x_12, 2, x_11);
lean_inc(x_9);
lean_inc(x_10);
x_13 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_13, 0, lean_box(0));
lean_closure_set(x_13, 1, x_10);
lean_closure_set(x_13, 2, x_9);
lean_inc_ref(x_11);
lean_inc(x_9);
lean_inc(x_10);
x_14 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, x_10);
lean_closure_set(x_14, 2, x_9);
lean_closure_set(x_14, 3, x_11);
lean_closure_set(x_14, 4, x_13);
x_15 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_15, 0, x_1);
lean_ctor_set(x_15, 1, x_11);
lean_ctor_set(x_15, 2, x_12);
lean_ctor_set(x_15, 3, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Fintype_groupWithZeroOfCancel___redArg(x_2, x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_NNRat_castRec___redArg(x_1, x_2, x_4);
x_7 = lean_apply_2(x_3, x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_Rat_castRec___redArg(x_1, x_2, x_3, x_5);
x_8 = lean_apply_2(x_4, x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 4);
lean_inc_ref(x_4);
x_6 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_6);
x_7 = lp_mathlib_Fintype_groupWithZeroOfCancel___redArg(x_6, x_2, x_3);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 2);
lean_inc(x_9);
lean_dec_ref(x_7);
lean_inc_ref(x_6);
x_10 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_6);
lean_inc_ref(x_10);
x_11 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_10);
x_12 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_6);
x_13 = lean_ctor_get(x_4, 0);
x_14 = lean_ctor_get(x_11, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_11, 1);
lean_inc(x_15);
lean_dec_ref(x_11);
x_16 = lean_ctor_get(x_12, 0);
lean_inc(x_16);
x_17 = lean_ctor_get(x_12, 1);
lean_inc(x_17);
lean_dec_ref(x_12);
x_18 = lean_ctor_get(x_4, 2);
x_19 = lean_ctor_get(x_13, 1);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__1), 6, 5);
lean_closure_set(x_20, 0, x_2);
lean_closure_set(x_20, 1, x_15);
lean_closure_set(x_20, 2, x_10);
lean_closure_set(x_20, 3, x_14);
lean_closure_set(x_20, 4, x_3);
lean_inc(x_16);
lean_inc(x_17);
x_21 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_17);
lean_closure_set(x_21, 2, x_16);
x_22 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_22, 0, lean_box(0));
lean_closure_set(x_22, 1, x_17);
lean_closure_set(x_22, 2, x_16);
lean_closure_set(x_22, 3, x_20);
lean_closure_set(x_22, 4, x_21);
lean_inc(x_9);
lean_inc(x_18);
x_23 = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(x_23, 0, lean_box(0));
lean_closure_set(x_23, 1, x_18);
lean_closure_set(x_23, 2, x_9);
lean_inc(x_9);
lean_inc(x_5);
lean_inc(x_18);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(x_24, 0, lean_box(0));
lean_closure_set(x_24, 1, x_18);
lean_closure_set(x_24, 2, x_5);
lean_closure_set(x_24, 3, x_9);
lean_inc(x_19);
lean_inc(x_9);
lean_inc(x_18);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__2), 5, 3);
lean_closure_set(x_25, 0, x_18);
lean_closure_set(x_25, 1, x_9);
lean_closure_set(x_25, 2, x_19);
lean_inc(x_19);
lean_inc(x_9);
lean_inc(x_5);
lean_inc(x_18);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__0), 6, 4);
lean_closure_set(x_26, 0, x_18);
lean_closure_set(x_26, 1, x_5);
lean_closure_set(x_26, 2, x_9);
lean_closure_set(x_26, 3, x_19);
x_27 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_27, 0, x_1);
lean_ctor_set(x_27, 1, x_8);
lean_ctor_set(x_27, 2, x_9);
lean_ctor_set(x_27, 3, x_22);
lean_ctor_set(x_27, 4, x_23);
lean_ctor_set(x_27, 5, x_24);
lean_ctor_set(x_27, 6, x_25);
lean_ctor_set(x_27, 7, x_26);
return x_27;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Fintype_divisionRingOfIsDomain___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_fieldOfDomain___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_4 = lp_mathlib_Fintype_divisionRingOfIsDomain___redArg(x_1, x_2, x_3);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_6 = lean_ctor_get(x_4, 7);
lean_dec(x_6);
x_7 = lean_ctor_get(x_4, 6);
lean_dec(x_7);
x_8 = lean_ctor_get(x_4, 3);
lean_dec(x_8);
x_9 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_1, 4);
lean_inc(x_10);
lean_dec_ref(x_1);
lean_inc_ref(x_9);
x_11 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_9);
lean_inc_ref(x_11);
x_12 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_11);
lean_inc_ref(x_12);
x_13 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_12);
x_14 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_14);
x_15 = lean_ctor_get(x_13, 0);
lean_inc(x_15);
x_16 = lean_ctor_get(x_13, 1);
lean_inc(x_16);
lean_dec_ref(x_13);
x_17 = lean_ctor_get(x_14, 0);
lean_inc(x_17);
x_18 = lean_ctor_get(x_14, 1);
lean_inc(x_18);
lean_dec_ref(x_14);
lean_inc(x_3);
lean_inc_ref(x_2);
x_19 = lp_mathlib_Fintype_groupWithZeroOfCancel___redArg(x_11, x_2, x_3);
x_20 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_20);
x_21 = lean_ctor_get(x_19, 2);
lean_inc(x_21);
lean_dec_ref(x_19);
x_22 = lean_ctor_get(x_9, 2);
lean_inc(x_22);
lean_dec_ref(x_9);
x_23 = lean_ctor_get(x_20, 1);
lean_inc(x_23);
lean_dec_ref(x_20);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__1), 6, 5);
lean_closure_set(x_24, 0, x_2);
lean_closure_set(x_24, 1, x_16);
lean_closure_set(x_24, 2, x_12);
lean_closure_set(x_24, 3, x_15);
lean_closure_set(x_24, 4, x_3);
lean_inc(x_17);
lean_inc(x_18);
x_25 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, x_18);
lean_closure_set(x_25, 2, x_17);
x_26 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_26, 0, lean_box(0));
lean_closure_set(x_26, 1, x_18);
lean_closure_set(x_26, 2, x_17);
lean_closure_set(x_26, 3, x_24);
lean_closure_set(x_26, 4, x_25);
lean_inc(x_23);
lean_inc(x_21);
lean_inc(x_22);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__0), 6, 4);
lean_closure_set(x_27, 0, x_22);
lean_closure_set(x_27, 1, x_10);
lean_closure_set(x_27, 2, x_21);
lean_closure_set(x_27, 3, x_23);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__2), 5, 3);
lean_closure_set(x_28, 0, x_22);
lean_closure_set(x_28, 1, x_21);
lean_closure_set(x_28, 2, x_23);
lean_ctor_set(x_4, 7, x_27);
lean_ctor_set(x_4, 6, x_28);
lean_ctor_set(x_4, 3, x_26);
return x_4;
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; 
x_29 = lean_ctor_get(x_4, 0);
x_30 = lean_ctor_get(x_4, 1);
x_31 = lean_ctor_get(x_4, 2);
x_32 = lean_ctor_get(x_4, 4);
x_33 = lean_ctor_get(x_4, 5);
lean_inc(x_33);
lean_inc(x_32);
lean_inc(x_31);
lean_inc(x_30);
lean_inc(x_29);
lean_dec(x_4);
x_34 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_34);
x_35 = lean_ctor_get(x_1, 4);
lean_inc(x_35);
lean_dec_ref(x_1);
lean_inc_ref(x_34);
x_36 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_34);
lean_inc_ref(x_36);
x_37 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_36);
lean_inc_ref(x_37);
x_38 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_37);
x_39 = lean_ctor_get(x_36, 0);
lean_inc_ref(x_39);
x_40 = lean_ctor_get(x_38, 0);
lean_inc(x_40);
x_41 = lean_ctor_get(x_38, 1);
lean_inc(x_41);
lean_dec_ref(x_38);
x_42 = lean_ctor_get(x_39, 0);
lean_inc(x_42);
x_43 = lean_ctor_get(x_39, 1);
lean_inc(x_43);
lean_dec_ref(x_39);
lean_inc(x_3);
lean_inc_ref(x_2);
x_44 = lp_mathlib_Fintype_groupWithZeroOfCancel___redArg(x_36, x_2, x_3);
x_45 = lean_ctor_get(x_34, 0);
lean_inc_ref(x_45);
x_46 = lean_ctor_get(x_44, 2);
lean_inc(x_46);
lean_dec_ref(x_44);
x_47 = lean_ctor_get(x_34, 2);
lean_inc(x_47);
lean_dec_ref(x_34);
x_48 = lean_ctor_get(x_45, 1);
lean_inc(x_48);
lean_dec_ref(x_45);
x_49 = lean_alloc_closure((void*)(lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__1), 6, 5);
lean_closure_set(x_49, 0, x_2);
lean_closure_set(x_49, 1, x_41);
lean_closure_set(x_49, 2, x_37);
lean_closure_set(x_49, 3, x_40);
lean_closure_set(x_49, 4, x_3);
lean_inc(x_42);
lean_inc(x_43);
x_50 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_50, 0, lean_box(0));
lean_closure_set(x_50, 1, x_43);
lean_closure_set(x_50, 2, x_42);
x_51 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_51, 0, lean_box(0));
lean_closure_set(x_51, 1, x_43);
lean_closure_set(x_51, 2, x_42);
lean_closure_set(x_51, 3, x_49);
lean_closure_set(x_51, 4, x_50);
lean_inc(x_48);
lean_inc(x_46);
lean_inc(x_47);
x_52 = lean_alloc_closure((void*)(lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__0), 6, 4);
lean_closure_set(x_52, 0, x_47);
lean_closure_set(x_52, 1, x_35);
lean_closure_set(x_52, 2, x_46);
lean_closure_set(x_52, 3, x_48);
x_53 = lean_alloc_closure((void*)(lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__2), 5, 3);
lean_closure_set(x_53, 0, x_47);
lean_closure_set(x_53, 1, x_46);
lean_closure_set(x_53, 2, x_48);
x_54 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_54, 0, x_29);
lean_ctor_set(x_54, 1, x_30);
lean_ctor_set(x_54, 2, x_31);
lean_ctor_set(x_54, 3, x_51);
lean_ctor_set(x_54, 4, x_32);
lean_ctor_set(x_54, 5, x_33);
lean_ctor_set(x_54, 6, x_53);
lean_ctor_set(x_54, 7, x_52);
return x_54;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_fieldOfDomain(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Fintype_fieldOfDomain___redArg(x_2, x_4, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Inv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FieldSimp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_IntegralDomain(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Inv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FieldSimp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
