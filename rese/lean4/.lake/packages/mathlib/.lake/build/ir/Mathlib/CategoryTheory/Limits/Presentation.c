// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Presentation
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Connected public import Mathlib.CategoryTheory.Limits.Final
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
lean_object* lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_changeDiag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_cone___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_changeDiag___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_cocone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_const___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_cocone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_changeDiag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_ofIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_cone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_ofIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_ofIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_ofIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_ext___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_cone___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_changeDiag___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_cocone___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_cone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_ext___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_cocone___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_ofIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_ofIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_cocone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_cocone___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_1);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_cocone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Limits_ColimitPresentation_cocone(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_cocone___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Limits_ColimitPresentation_cocone___redArg(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_changeDiag___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; 
x_7 = !lean_is_exclusive(x_4);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_8 = lean_ctor_get(x_4, 0);
x_9 = lean_ctor_get(x_4, 1);
x_10 = lean_ctor_get(x_4, 2);
lean_inc_ref(x_1);
x_11 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_12 = !lean_is_exclusive(x_11);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_13 = lean_ctor_get(x_11, 0);
x_14 = lean_ctor_get(x_11, 1);
lean_dec(x_14);
x_15 = lean_ctor_get(x_6, 0);
lean_inc(x_15);
lean_inc(x_9);
lean_inc(x_3);
lean_ctor_set(x_11, 1, x_9);
lean_ctor_set(x_11, 0, x_3);
lean_inc_ref(x_8);
lean_inc_ref(x_5);
lean_inc_ref(x_1);
x_16 = lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(x_2, x_1, x_5, x_8, x_6, x_11);
x_17 = lean_ctor_get(x_16, 1);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lean_apply_1(x_13, x_3);
lean_inc_ref(x_5);
x_19 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_5, x_8, x_18, x_15, x_9);
x_20 = lean_apply_1(x_17, x_10);
lean_ctor_set(x_4, 2, x_20);
lean_ctor_set(x_4, 1, x_19);
lean_ctor_set(x_4, 0, x_5);
return x_4;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_21 = lean_ctor_get(x_11, 0);
lean_inc(x_21);
lean_dec(x_11);
x_22 = lean_ctor_get(x_6, 0);
lean_inc(x_22);
lean_inc(x_9);
lean_inc(x_3);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_3);
lean_ctor_set(x_23, 1, x_9);
lean_inc_ref(x_8);
lean_inc_ref(x_5);
lean_inc_ref(x_1);
x_24 = lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(x_2, x_1, x_5, x_8, x_6, x_23);
x_25 = lean_ctor_get(x_24, 1);
lean_inc(x_25);
lean_dec_ref(x_24);
x_26 = lean_apply_1(x_21, x_3);
lean_inc_ref(x_5);
x_27 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_5, x_8, x_26, x_22, x_9);
x_28 = lean_apply_1(x_25, x_10);
lean_ctor_set(x_4, 2, x_28);
lean_ctor_set(x_4, 1, x_27);
lean_ctor_set(x_4, 0, x_5);
return x_4;
}
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_29 = lean_ctor_get(x_4, 0);
x_30 = lean_ctor_get(x_4, 1);
x_31 = lean_ctor_get(x_4, 2);
lean_inc(x_31);
lean_inc(x_30);
lean_inc(x_29);
lean_dec(x_4);
lean_inc_ref(x_1);
x_32 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_33 = lean_ctor_get(x_32, 0);
lean_inc(x_33);
if (lean_is_exclusive(x_32)) {
 lean_ctor_release(x_32, 0);
 lean_ctor_release(x_32, 1);
 x_34 = x_32;
} else {
 lean_dec_ref(x_32);
 x_34 = lean_box(0);
}
x_35 = lean_ctor_get(x_6, 0);
lean_inc(x_35);
lean_inc(x_30);
lean_inc(x_3);
if (lean_is_scalar(x_34)) {
 x_36 = lean_alloc_ctor(0, 2, 0);
} else {
 x_36 = x_34;
}
lean_ctor_set(x_36, 0, x_3);
lean_ctor_set(x_36, 1, x_30);
lean_inc_ref(x_29);
lean_inc_ref(x_5);
lean_inc_ref(x_1);
x_37 = lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(x_2, x_1, x_5, x_29, x_6, x_36);
x_38 = lean_ctor_get(x_37, 1);
lean_inc(x_38);
lean_dec_ref(x_37);
x_39 = lean_apply_1(x_33, x_3);
lean_inc_ref(x_5);
x_40 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_5, x_29, x_39, x_35, x_30);
x_41 = lean_apply_1(x_38, x_31);
x_42 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_42, 0, x_5);
lean_ctor_set(x_42, 1, x_40);
lean_ctor_set(x_42, 2, x_41);
return x_42;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_changeDiag(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_ColimitPresentation_changeDiag___redArg(x_2, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_ofIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_7 = lean_ctor_get(x_3, 0);
x_8 = lean_ctor_get(x_3, 1);
x_9 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_1);
x_10 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
x_14 = lean_ctor_get(x_5, 0);
lean_inc(x_12);
lean_inc(x_2);
x_15 = lean_apply_1(x_12, x_2);
lean_inc(x_4);
x_16 = lean_apply_1(x_12, x_4);
lean_inc(x_14);
lean_inc(x_4);
lean_inc(x_2);
x_17 = lean_apply_3(x_13, x_2, x_4, x_14);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_1);
x_18 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_7, x_15, x_16, x_8, x_17);
lean_ctor_set(x_10, 1, x_8);
lean_ctor_set(x_10, 0, x_2);
lean_inc(x_18);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_4);
lean_ctor_set(x_19, 1, x_18);
x_20 = lp_mathlib_CategoryTheory_Limits_Cocones_ext___redArg(x_5);
x_21 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg(x_1, x_10, x_19, x_9, x_20);
lean_ctor_set(x_3, 2, x_21);
lean_ctor_set(x_3, 1, x_18);
return x_3;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_22 = lean_ctor_get(x_10, 0);
x_23 = lean_ctor_get(x_10, 1);
lean_inc(x_23);
lean_inc(x_22);
lean_dec(x_10);
x_24 = lean_ctor_get(x_5, 0);
lean_inc(x_22);
lean_inc(x_2);
x_25 = lean_apply_1(x_22, x_2);
lean_inc(x_4);
x_26 = lean_apply_1(x_22, x_4);
lean_inc(x_24);
lean_inc(x_4);
lean_inc(x_2);
x_27 = lean_apply_3(x_23, x_2, x_4, x_24);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_1);
x_28 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_7, x_25, x_26, x_8, x_27);
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_2);
lean_ctor_set(x_29, 1, x_8);
lean_inc(x_28);
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_4);
lean_ctor_set(x_30, 1, x_28);
x_31 = lp_mathlib_CategoryTheory_Limits_Cocones_ext___redArg(x_5);
x_32 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg(x_1, x_29, x_30, x_9, x_31);
lean_ctor_set(x_3, 2, x_32);
lean_ctor_set(x_3, 1, x_28);
return x_3;
}
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; 
x_33 = lean_ctor_get(x_3, 0);
x_34 = lean_ctor_get(x_3, 1);
x_35 = lean_ctor_get(x_3, 2);
lean_inc(x_35);
lean_inc(x_34);
lean_inc(x_33);
lean_dec(x_3);
lean_inc_ref(x_1);
x_36 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_37 = lean_ctor_get(x_36, 0);
lean_inc(x_37);
x_38 = lean_ctor_get(x_36, 1);
lean_inc(x_38);
if (lean_is_exclusive(x_36)) {
 lean_ctor_release(x_36, 0);
 lean_ctor_release(x_36, 1);
 x_39 = x_36;
} else {
 lean_dec_ref(x_36);
 x_39 = lean_box(0);
}
x_40 = lean_ctor_get(x_5, 0);
lean_inc(x_37);
lean_inc(x_2);
x_41 = lean_apply_1(x_37, x_2);
lean_inc(x_4);
x_42 = lean_apply_1(x_37, x_4);
lean_inc(x_40);
lean_inc(x_4);
lean_inc(x_2);
x_43 = lean_apply_3(x_38, x_2, x_4, x_40);
lean_inc(x_34);
lean_inc_ref(x_33);
lean_inc_ref(x_1);
x_44 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_33, x_41, x_42, x_34, x_43);
if (lean_is_scalar(x_39)) {
 x_45 = lean_alloc_ctor(0, 2, 0);
} else {
 x_45 = x_39;
}
lean_ctor_set(x_45, 0, x_2);
lean_ctor_set(x_45, 1, x_34);
lean_inc(x_44);
x_46 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_46, 0, x_4);
lean_ctor_set(x_46, 1, x_44);
x_47 = lp_mathlib_CategoryTheory_Limits_Cocones_ext___redArg(x_5);
x_48 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg(x_1, x_45, x_46, x_35, x_47);
x_49 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_49, 0, x_33);
lean_ctor_set(x_49, 1, x_44);
lean_ctor_set(x_49, 2, x_48);
return x_49;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_ofIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_ColimitPresentation_ofIso___redArg(x_2, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_ColimitPresentation_ofIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_ColimitPresentation_ofIso(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_cone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_cone___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_1);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_cone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Limits_LimitPresentation_cone(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_cone___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Limits_LimitPresentation_cone___redArg(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_changeDiag___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; uint8_t x_8; 
lean_inc_ref(x_1);
x_7 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_7, 1);
lean_dec(x_10);
x_11 = !lean_is_exclusive(x_4);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_12 = lean_ctor_get(x_4, 0);
x_13 = lean_ctor_get(x_4, 1);
x_14 = lean_ctor_get(x_4, 2);
x_15 = lean_ctor_get(x_6, 1);
lean_inc(x_15);
x_16 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_6);
lean_inc(x_13);
lean_inc(x_3);
lean_ctor_set(x_7, 1, x_13);
lean_ctor_set(x_7, 0, x_3);
lean_inc_ref(x_5);
lean_inc_ref(x_12);
lean_inc_ref(x_1);
x_17 = lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(x_2, x_1, x_12, x_5, x_16, x_7);
x_18 = lean_ctor_get(x_17, 1);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lean_apply_1(x_9, x_3);
lean_inc_ref(x_5);
x_20 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_19, x_12, x_5, x_13, x_15);
x_21 = lean_apply_1(x_18, x_14);
lean_ctor_set(x_4, 2, x_21);
lean_ctor_set(x_4, 1, x_20);
lean_ctor_set(x_4, 0, x_5);
return x_4;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_22 = lean_ctor_get(x_4, 0);
x_23 = lean_ctor_get(x_4, 1);
x_24 = lean_ctor_get(x_4, 2);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_22);
lean_dec(x_4);
x_25 = lean_ctor_get(x_6, 1);
lean_inc(x_25);
x_26 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_6);
lean_inc(x_23);
lean_inc(x_3);
lean_ctor_set(x_7, 1, x_23);
lean_ctor_set(x_7, 0, x_3);
lean_inc_ref(x_5);
lean_inc_ref(x_22);
lean_inc_ref(x_1);
x_27 = lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(x_2, x_1, x_22, x_5, x_26, x_7);
x_28 = lean_ctor_get(x_27, 1);
lean_inc(x_28);
lean_dec_ref(x_27);
x_29 = lean_apply_1(x_9, x_3);
lean_inc_ref(x_5);
x_30 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_29, x_22, x_5, x_23, x_25);
x_31 = lean_apply_1(x_28, x_24);
x_32 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_32, 0, x_5);
lean_ctor_set(x_32, 1, x_30);
lean_ctor_set(x_32, 2, x_31);
return x_32;
}
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_33 = lean_ctor_get(x_7, 0);
lean_inc(x_33);
lean_dec(x_7);
x_34 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_34);
x_35 = lean_ctor_get(x_4, 1);
lean_inc(x_35);
x_36 = lean_ctor_get(x_4, 2);
lean_inc(x_36);
if (lean_is_exclusive(x_4)) {
 lean_ctor_release(x_4, 0);
 lean_ctor_release(x_4, 1);
 lean_ctor_release(x_4, 2);
 x_37 = x_4;
} else {
 lean_dec_ref(x_4);
 x_37 = lean_box(0);
}
x_38 = lean_ctor_get(x_6, 1);
lean_inc(x_38);
x_39 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_6);
lean_inc(x_35);
lean_inc(x_3);
x_40 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_40, 0, x_3);
lean_ctor_set(x_40, 1, x_35);
lean_inc_ref(x_5);
lean_inc_ref(x_34);
lean_inc_ref(x_1);
x_41 = lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(x_2, x_1, x_34, x_5, x_39, x_40);
x_42 = lean_ctor_get(x_41, 1);
lean_inc(x_42);
lean_dec_ref(x_41);
x_43 = lean_apply_1(x_33, x_3);
lean_inc_ref(x_5);
x_44 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_43, x_34, x_5, x_35, x_38);
x_45 = lean_apply_1(x_42, x_36);
if (lean_is_scalar(x_37)) {
 x_46 = lean_alloc_ctor(0, 3, 0);
} else {
 x_46 = x_37;
}
lean_ctor_set(x_46, 0, x_5);
lean_ctor_set(x_46, 1, x_44);
lean_ctor_set(x_46, 2, x_45);
return x_46;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_changeDiag(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_LimitPresentation_changeDiag___redArg(x_2, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_ofIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_7 = lean_ctor_get(x_3, 0);
x_8 = lean_ctor_get(x_3, 1);
x_9 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_1);
x_10 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
x_14 = lean_ctor_get(x_5, 1);
lean_inc(x_12);
lean_inc(x_4);
x_15 = lean_apply_1(x_12, x_4);
lean_inc(x_2);
x_16 = lean_apply_1(x_12, x_2);
lean_inc(x_14);
lean_inc(x_2);
lean_inc(x_4);
x_17 = lean_apply_3(x_13, x_4, x_2, x_14);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_1);
x_18 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_15, x_16, x_7, x_17, x_8);
lean_ctor_set(x_10, 1, x_8);
lean_ctor_set(x_10, 0, x_2);
lean_inc(x_18);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_4);
lean_ctor_set(x_19, 1, x_18);
x_20 = lp_mathlib_CategoryTheory_Limits_Cones_ext___redArg(x_5);
x_21 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg(x_1, x_10, x_19, x_9, x_20);
lean_ctor_set(x_3, 2, x_21);
lean_ctor_set(x_3, 1, x_18);
return x_3;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_22 = lean_ctor_get(x_10, 0);
x_23 = lean_ctor_get(x_10, 1);
lean_inc(x_23);
lean_inc(x_22);
lean_dec(x_10);
x_24 = lean_ctor_get(x_5, 1);
lean_inc(x_22);
lean_inc(x_4);
x_25 = lean_apply_1(x_22, x_4);
lean_inc(x_2);
x_26 = lean_apply_1(x_22, x_2);
lean_inc(x_24);
lean_inc(x_2);
lean_inc(x_4);
x_27 = lean_apply_3(x_23, x_4, x_2, x_24);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_1);
x_28 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_25, x_26, x_7, x_27, x_8);
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_2);
lean_ctor_set(x_29, 1, x_8);
lean_inc(x_28);
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_4);
lean_ctor_set(x_30, 1, x_28);
x_31 = lp_mathlib_CategoryTheory_Limits_Cones_ext___redArg(x_5);
x_32 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg(x_1, x_29, x_30, x_9, x_31);
lean_ctor_set(x_3, 2, x_32);
lean_ctor_set(x_3, 1, x_28);
return x_3;
}
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; 
x_33 = lean_ctor_get(x_3, 0);
x_34 = lean_ctor_get(x_3, 1);
x_35 = lean_ctor_get(x_3, 2);
lean_inc(x_35);
lean_inc(x_34);
lean_inc(x_33);
lean_dec(x_3);
lean_inc_ref(x_1);
x_36 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_37 = lean_ctor_get(x_36, 0);
lean_inc(x_37);
x_38 = lean_ctor_get(x_36, 1);
lean_inc(x_38);
if (lean_is_exclusive(x_36)) {
 lean_ctor_release(x_36, 0);
 lean_ctor_release(x_36, 1);
 x_39 = x_36;
} else {
 lean_dec_ref(x_36);
 x_39 = lean_box(0);
}
x_40 = lean_ctor_get(x_5, 1);
lean_inc(x_37);
lean_inc(x_4);
x_41 = lean_apply_1(x_37, x_4);
lean_inc(x_2);
x_42 = lean_apply_1(x_37, x_2);
lean_inc(x_40);
lean_inc(x_2);
lean_inc(x_4);
x_43 = lean_apply_3(x_38, x_4, x_2, x_40);
lean_inc(x_34);
lean_inc_ref(x_33);
lean_inc_ref(x_1);
x_44 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_1, x_41, x_42, x_33, x_43, x_34);
if (lean_is_scalar(x_39)) {
 x_45 = lean_alloc_ctor(0, 2, 0);
} else {
 x_45 = x_39;
}
lean_ctor_set(x_45, 0, x_2);
lean_ctor_set(x_45, 1, x_34);
lean_inc(x_44);
x_46 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_46, 0, x_4);
lean_ctor_set(x_46, 1, x_44);
x_47 = lp_mathlib_CategoryTheory_Limits_Cones_ext___redArg(x_5);
x_48 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg(x_1, x_45, x_46, x_35, x_47);
x_49 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_49, 0, x_33);
lean_ctor_set(x_49, 1, x_44);
lean_ctor_set(x_49, 2, x_48);
return x_49;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_ofIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_LimitPresentation_ofIso___redArg(x_2, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_LimitPresentation_ofIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_LimitPresentation_ofIso(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_4);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Connected(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Final(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Presentation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Connected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Final(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
