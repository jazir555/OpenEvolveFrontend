// Lean compiler output
// Module: Mathlib.CategoryTheory.Subobject.Basic
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Skeleton public import Mathlib.CategoryTheory.Subobject.MonoOver public import Mathlib.CategoryTheory.Skeletal public import Mathlib.CategoryTheory.ConcreteCategory.Basic public import Mathlib.Tactic.ApplyFun public import Mathlib.Tactic.CategoryTheory.Elementwise
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
lean_object* lp_mathlib_CategoryTheory_toThinSkeleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mapPullbackAdj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_existsPullbackAdj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lift___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lowerEquivalence___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mapIsoToOrderIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_id(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_MonoOver_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lowerAdjunction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Preorder_smallCategory(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lowerAdjunction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPartialOrderSubobject(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lowerEquivalence(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mapPullbackAdj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_instCategoryOver___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_ThinSkeleton_map_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPartialOrderSubobject___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lowerEquivalence___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mk___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_category___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_ThinSkeleton_preorder(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_ObjectProperty_FullSubcategory_category___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_eqToIso___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPartialOrderSubobject___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_ThinSkeleton_map___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mapIsoToOrderIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_existsPullbackAdj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPartialOrderSubobject___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_CategoryTheory_instCategoryOver___redArg(x_1);
x_3 = lp_mathlib_CategoryTheory_ObjectProperty_FullSubcategory_category___redArg(x_2);
x_4 = lp_mathlib_CategoryTheory_ThinSkeleton_preorder(lean_box(0), x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPartialOrderSubobject(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_instPartialOrderSubobject___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPartialOrderSubobject___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_instPartialOrderSubobject(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mk___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lp_mathlib_CategoryTheory_instCategoryOver___redArg(x_1);
x_5 = lp_mathlib_CategoryTheory_ObjectProperty_FullSubcategory_category___redArg(x_4);
x_6 = lp_mathlib_CategoryTheory_toThinSkeleton___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lp_mathlib_CategoryTheory_CostructuredArrow_mk___redArg(x_2, x_3);
x_9 = lean_apply_1(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mk(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Subobject_mk___redArg(x_2, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mk___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Subobject_mk(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lift___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 2);
lean_inc(x_4);
lean_dec(x_2);
x_5 = lean_apply_3(x_1, x_3, x_4, lean_box(0));
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lift(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Subobject_lift___redArg(x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lift___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Subobject_lift(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_ThinSkeleton_map___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_ThinSkeleton_map___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Subobject_lower(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower_u2082___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_CategoryTheory_instCategoryOver___redArg(x_1);
x_4 = lp_mathlib_CategoryTheory_ObjectProperty_FullSubcategory_category___redArg(x_3);
lean_inc_ref_n(x_4, 2);
x_5 = lp_mathlib_CategoryTheory_ThinSkeleton_map_u2082___redArg(x_4, x_4, x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower_u2082(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Subobject_lower_u2082___redArg(x_2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lower_u2082___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Subobject_lower_u2082(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lowerAdjunction(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, lean_box(0));
lean_ctor_set(x_10, 1, lean_box(0));
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lowerAdjunction___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Subobject_lowerAdjunction(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lowerEquivalence___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_4 = lp_mathlib_CategoryTheory_instPartialOrderSubobject___redArg(x_1);
x_5 = lp_mathlib_Preorder_smallCategory(lean_box(0), x_4);
lean_dec_ref(x_4);
x_6 = lp_mathlib_CategoryTheory_instPartialOrderSubobject___redArg(x_2);
x_7 = lp_mathlib_Preorder_smallCategory(lean_box(0), x_6);
lean_dec_ref(x_6);
x_8 = !lean_is_exclusive(x_3);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_9 = lean_ctor_get(x_3, 0);
x_10 = lean_ctor_get(x_3, 1);
x_11 = lean_ctor_get(x_3, 3);
lean_dec(x_11);
x_12 = lean_ctor_get(x_3, 2);
lean_dec(x_12);
x_13 = lp_mathlib_CategoryTheory_ThinSkeleton_map___redArg(x_9);
x_14 = lp_mathlib_CategoryTheory_ThinSkeleton_map___redArg(x_10);
lean_inc_ref(x_5);
x_15 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_5);
x_16 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_5);
lean_dec_ref(x_5);
lean_inc_ref(x_14);
lean_inc_ref(x_13);
x_17 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_13, x_14);
x_18 = lp_mathlib_CategoryTheory_eqToIso___redArg(x_15, x_16, x_17);
lean_inc_ref(x_7);
x_19 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_7);
lean_inc_ref(x_13);
lean_inc_ref(x_14);
x_20 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_14, x_13);
x_21 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_7);
lean_dec_ref(x_7);
x_22 = lp_mathlib_CategoryTheory_eqToIso___redArg(x_19, x_20, x_21);
lean_ctor_set(x_3, 3, x_22);
lean_ctor_set(x_3, 2, x_18);
lean_ctor_set(x_3, 1, x_14);
lean_ctor_set(x_3, 0, x_13);
return x_3;
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_23 = lean_ctor_get(x_3, 0);
x_24 = lean_ctor_get(x_3, 1);
lean_inc(x_24);
lean_inc(x_23);
lean_dec(x_3);
x_25 = lp_mathlib_CategoryTheory_ThinSkeleton_map___redArg(x_23);
x_26 = lp_mathlib_CategoryTheory_ThinSkeleton_map___redArg(x_24);
lean_inc_ref(x_5);
x_27 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_5);
x_28 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_5);
lean_dec_ref(x_5);
lean_inc_ref(x_26);
lean_inc_ref(x_25);
x_29 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_25, x_26);
x_30 = lp_mathlib_CategoryTheory_eqToIso___redArg(x_27, x_28, x_29);
lean_inc_ref(x_7);
x_31 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_7);
lean_inc_ref(x_25);
lean_inc_ref(x_26);
x_32 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_26, x_25);
x_33 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_7);
lean_dec_ref(x_7);
x_34 = lp_mathlib_CategoryTheory_eqToIso___redArg(x_31, x_32, x_33);
x_35 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_35, 0, x_25);
lean_ctor_set(x_35, 1, x_26);
lean_ctor_set(x_35, 2, x_30);
lean_ctor_set(x_35, 3, x_34);
return x_35;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lowerEquivalence(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Subobject_lowerEquivalence___redArg(x_2, x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_lowerEquivalence___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Subobject_lowerEquivalence(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec(x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_map___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_CategoryTheory_MonoOver_map___redArg(x_1, x_2, x_3, x_4);
x_6 = lp_mathlib_CategoryTheory_ThinSkeleton_map___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Subobject_map___redArg(x_2, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mapIsoToOrderIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_dec_ref(x_4);
lean_inc(x_3);
lean_inc(x_2);
lean_inc_ref(x_1);
x_7 = lp_mathlib_CategoryTheory_Subobject_map___redArg(x_1, x_2, x_3, x_5);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lp_mathlib_CategoryTheory_Subobject_map___redArg(x_1, x_3, x_2, x_6);
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_ctor_get(x_9, 0);
x_12 = lean_ctor_get(x_9, 1);
lean_dec(x_12);
lean_ctor_set(x_9, 1, x_11);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
else
{
lean_object* x_13; lean_object* x_14; 
x_13 = lean_ctor_get(x_9, 0);
lean_inc(x_13);
lean_dec(x_9);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_8);
lean_ctor_set(x_14, 1, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mapIsoToOrderIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Subobject_mapIsoToOrderIso___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mapPullbackAdj(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, lean_box(0));
lean_ctor_set(x_8, 1, lean_box(0));
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_mapPullbackAdj___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Subobject_mapPullbackAdj(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_existsPullbackAdj(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, lean_box(0));
lean_ctor_set(x_8, 1, lean_box(0));
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subobject_existsPullbackAdj___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Subobject_existsPullbackAdj(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Skeleton(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Subobject_MonoOver(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Skeletal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_ConcreteCategory_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CategoryTheory_Elementwise(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Subobject_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Skeleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Subobject_MonoOver(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Skeletal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_ConcreteCategory_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CategoryTheory_Elementwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
