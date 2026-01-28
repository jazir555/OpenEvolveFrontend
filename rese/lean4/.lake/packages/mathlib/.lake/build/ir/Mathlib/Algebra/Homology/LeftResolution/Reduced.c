// Lean compiler output
// Module: Mathlib.Algebra.Homology.LeftResolution.Reduced
// Imports: public import Init public import Mathlib.Algebra.Homology.LeftResolution.Transport public import Mathlib.CategoryTheory.Idempotents.FunctorExtension public import Mathlib.CategoryTheory.MorphismProperty.Retract
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_retractArrow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Idempotents_functorExtension_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Idempotents_toKaroubi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_retractArrow___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Idempotents_functorExtension_u2081___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Idempotents_whiskeringLeftObjToKaroubiFullyFaithful___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Idempotents_Karoubi_instCategory___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_1);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 1);
lean_inc(x_9);
lean_dec_ref(x_7);
lean_inc(x_8);
lean_inc(x_4);
x_10 = lean_apply_1(x_8, x_4);
lean_inc(x_5);
x_11 = lean_apply_1(x_8, x_5);
x_12 = lean_apply_2(x_2, x_10, x_11);
x_13 = lean_ctor_get(x_12, 2);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc(x_5);
lean_inc(x_4);
x_14 = lean_apply_2(x_3, x_4, x_5);
x_15 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_14);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
lean_inc(x_9);
lean_inc(x_5);
lean_inc(x_4);
x_17 = lean_apply_3(x_9, x_4, x_5, x_6);
x_18 = lean_apply_3(x_9, x_4, x_5, x_16);
x_19 = lean_apply_2(x_13, x_17, x_18);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
lean_inc(x_5);
x_9 = lean_apply_1(x_7, x_5);
lean_inc_n(x_9, 2);
x_10 = lean_apply_2(x_2, x_9, x_9);
x_11 = lean_ctor_get(x_10, 2);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lean_ctor_get(x_3, 1);
lean_inc(x_12);
lean_dec_ref(x_3);
lean_inc_n(x_5, 2);
x_13 = lean_apply_2(x_4, x_5, x_5);
x_14 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_13);
lean_dec_ref(x_13);
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_16 = lean_ctor_get(x_14, 0);
x_17 = lean_ctor_get(x_14, 1);
lean_dec(x_17);
lean_inc(x_9);
x_18 = lean_apply_1(x_12, x_9);
lean_inc(x_5);
x_19 = lean_apply_3(x_8, x_5, x_5, x_16);
x_20 = lean_apply_2(x_11, x_18, x_19);
lean_ctor_set(x_14, 1, x_20);
lean_ctor_set(x_14, 0, x_9);
return x_14;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_21 = lean_ctor_get(x_14, 0);
lean_inc(x_21);
lean_dec(x_14);
lean_inc(x_9);
x_22 = lean_apply_1(x_12, x_9);
lean_inc(x_5);
x_23 = lean_apply_3(x_8, x_5, x_5, x_21);
x_24 = lean_apply_2(x_11, x_22, x_23);
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_9);
lean_ctor_set(x_25, 1, x_24);
return x_25;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___redArg___lam__0), 6, 3);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_3);
lean_closure_set(x_5, 2, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___redArg___lam__1), 5, 4);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_3);
lean_closure_set(x_6, 2, x_1);
lean_closure_set(x_6, 3, x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___redArg(x_3, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_1);
x_6 = lp_mathlib_CategoryTheory_Idempotents_functorExtension_u2081___redArg(x_2, x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___redArg(x_1, x_3, x_4, x_5);
x_9 = lean_apply_1(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F___redArg(x_3, x_4, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27___redArg(x_6);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_retractArrow___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; 
lean_inc_ref(x_2);
x_8 = lp_mathlib_CategoryTheory_Idempotents_toKaroubi___redArg(x_2);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_ctor_get(x_4, 0);
x_11 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_2);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lean_ctor_get(x_3, 1);
lean_inc(x_13);
lean_dec_ref(x_3);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
x_14 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F___redArg(x_1, x_2, x_4, x_5, x_6);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc(x_9);
lean_inc(x_7);
x_16 = lean_apply_1(x_9, x_7);
lean_inc_ref(x_16);
x_17 = lean_apply_1(x_15, x_16);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lean_ctor_get(x_10, 0);
lean_inc(x_19);
x_20 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F_x27___redArg(x_1, x_4, x_5, x_6);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
lean_inc(x_7);
x_22 = lean_apply_1(x_21, x_7);
x_23 = !lean_is_exclusive(x_22);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; uint8_t x_26; 
x_24 = lean_ctor_get(x_22, 1);
x_25 = lean_ctor_get(x_22, 0);
lean_dec(x_25);
x_26 = !lean_is_exclusive(x_16);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; uint8_t x_31; 
x_27 = lean_ctor_get(x_16, 1);
x_28 = lean_ctor_get(x_16, 0);
lean_dec(x_28);
lean_inc(x_7);
x_29 = lean_apply_1(x_12, x_7);
x_30 = lean_apply_1(x_9, x_29);
x_31 = !lean_is_exclusive(x_30);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_32 = lean_ctor_get(x_30, 1);
x_33 = lean_ctor_get(x_30, 0);
lean_dec(x_33);
x_34 = lean_apply_1(x_19, x_7);
lean_inc(x_13);
lean_inc(x_24);
lean_inc(x_34);
lean_inc(x_18);
x_35 = lean_apply_3(x_13, x_18, x_34, x_24);
lean_ctor_set(x_30, 1, x_27);
lean_ctor_set(x_30, 0, x_35);
x_36 = lean_apply_3(x_13, x_34, x_18, x_24);
lean_ctor_set(x_16, 1, x_32);
lean_ctor_set(x_16, 0, x_36);
lean_ctor_set(x_22, 1, x_16);
lean_ctor_set(x_22, 0, x_30);
return x_22;
}
else
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_37 = lean_ctor_get(x_30, 1);
lean_inc(x_37);
lean_dec(x_30);
x_38 = lean_apply_1(x_19, x_7);
lean_inc(x_13);
lean_inc(x_24);
lean_inc(x_38);
lean_inc(x_18);
x_39 = lean_apply_3(x_13, x_18, x_38, x_24);
x_40 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_40, 0, x_39);
lean_ctor_set(x_40, 1, x_27);
x_41 = lean_apply_3(x_13, x_38, x_18, x_24);
lean_ctor_set(x_16, 1, x_37);
lean_ctor_set(x_16, 0, x_41);
lean_ctor_set(x_22, 1, x_16);
lean_ctor_set(x_22, 0, x_40);
return x_22;
}
}
else
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; 
x_42 = lean_ctor_get(x_16, 1);
lean_inc(x_42);
lean_dec(x_16);
lean_inc(x_7);
x_43 = lean_apply_1(x_12, x_7);
x_44 = lean_apply_1(x_9, x_43);
x_45 = lean_ctor_get(x_44, 1);
lean_inc(x_45);
if (lean_is_exclusive(x_44)) {
 lean_ctor_release(x_44, 0);
 lean_ctor_release(x_44, 1);
 x_46 = x_44;
} else {
 lean_dec_ref(x_44);
 x_46 = lean_box(0);
}
x_47 = lean_apply_1(x_19, x_7);
lean_inc(x_13);
lean_inc(x_24);
lean_inc(x_47);
lean_inc(x_18);
x_48 = lean_apply_3(x_13, x_18, x_47, x_24);
if (lean_is_scalar(x_46)) {
 x_49 = lean_alloc_ctor(0, 2, 0);
} else {
 x_49 = x_46;
}
lean_ctor_set(x_49, 0, x_48);
lean_ctor_set(x_49, 1, x_42);
x_50 = lean_apply_3(x_13, x_47, x_18, x_24);
x_51 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_51, 0, x_50);
lean_ctor_set(x_51, 1, x_45);
lean_ctor_set(x_22, 1, x_51);
lean_ctor_set(x_22, 0, x_49);
return x_22;
}
}
else
{
lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; 
x_52 = lean_ctor_get(x_22, 1);
lean_inc(x_52);
lean_dec(x_22);
x_53 = lean_ctor_get(x_16, 1);
lean_inc(x_53);
if (lean_is_exclusive(x_16)) {
 lean_ctor_release(x_16, 0);
 lean_ctor_release(x_16, 1);
 x_54 = x_16;
} else {
 lean_dec_ref(x_16);
 x_54 = lean_box(0);
}
lean_inc(x_7);
x_55 = lean_apply_1(x_12, x_7);
x_56 = lean_apply_1(x_9, x_55);
x_57 = lean_ctor_get(x_56, 1);
lean_inc(x_57);
if (lean_is_exclusive(x_56)) {
 lean_ctor_release(x_56, 0);
 lean_ctor_release(x_56, 1);
 x_58 = x_56;
} else {
 lean_dec_ref(x_56);
 x_58 = lean_box(0);
}
x_59 = lean_apply_1(x_19, x_7);
lean_inc(x_13);
lean_inc(x_52);
lean_inc(x_59);
lean_inc(x_18);
x_60 = lean_apply_3(x_13, x_18, x_59, x_52);
if (lean_is_scalar(x_58)) {
 x_61 = lean_alloc_ctor(0, 2, 0);
} else {
 x_61 = x_58;
}
lean_ctor_set(x_61, 0, x_60);
lean_ctor_set(x_61, 1, x_53);
x_62 = lean_apply_3(x_13, x_59, x_18, x_52);
if (lean_is_scalar(x_54)) {
 x_63 = lean_alloc_ctor(0, 2, 0);
} else {
 x_63 = x_54;
}
lean_ctor_set(x_63, 0, x_62);
lean_ctor_set(x_63, 1, x_57);
x_64 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_64, 0, x_61);
lean_ctor_set(x_64, 1, x_63);
return x_64;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_retractArrow(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_retractArrow___redArg(x_3, x_4, x_5, x_6, x_7, x_8, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
lean_inc_ref(x_2);
x_7 = lp_mathlib_CategoryTheory_Idempotents_Karoubi_instCategory___redArg(x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_8 = lp_mathlib_CategoryTheory_Idempotents_functorExtension_u2082___redArg(x_1, x_2);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
lean_inc_ref(x_4);
lean_inc_ref(x_2);
x_10 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_F___redArg(x_1, x_2, x_4, x_5, x_6);
x_11 = lean_apply_1(x_9, x_3);
x_12 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_10, x_11);
x_13 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_7);
x_14 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0_x27___redArg___lam__0), 2, 1);
lean_closure_set(x_14, 0, x_4);
x_15 = lp_mathlib_CategoryTheory_Idempotents_whiskeringLeftObjToKaroubiFullyFaithful___redArg(x_2, x_7);
x_16 = lean_apply_3(x_15, x_12, x_13, x_14);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Abelian_LeftResolution_karoubi_00_u03c0___redArg(x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_LeftResolution_Transport(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Idempotents_FunctorExtension(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_MorphismProperty_Retract(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Homology_LeftResolution_Reduced(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_LeftResolution_Transport(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Idempotents_FunctorExtension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_MorphismProperty_Retract(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
