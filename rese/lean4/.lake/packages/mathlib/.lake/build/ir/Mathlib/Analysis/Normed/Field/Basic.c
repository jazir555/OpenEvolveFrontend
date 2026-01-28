// Lean compiler output
// Module: Mathlib.Analysis.Normed.Field.Basic
// Imports: public import Init public import Mathlib.Algebra.Field.Subfield.Defs public import Mathlib.Algebra.Order.Group.Pointwise.Interval public import Mathlib.Analysis.Normed.Ring.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___lam__2(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubfieldClass_toField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedField_induced(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedDivisionRing_toNormedRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedDivisionRing_induced___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedField_toNormedCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedField_induced___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NormedAddCommGroup_toNormedAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NontriviallyNormedField_ofNormNeOne___redArg___boxed(lean_object*);
lean_object* lp_mathlib_NormedRing_toNonUnitalNormedRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedField_toNormedDivisionRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DenselyNormedField_toNontriviallyNormedField___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NontriviallyNormedField_ofNormNeOne___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubringClass_subtype___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_DenselyNormedField_toNontriviallyNormedField___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedField_toNormedDivisionRing___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_instFunLike___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NontriviallyNormedField_ofNormNeOne(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedDivisionRing_induced(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedDivisionRing_toNormedRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DenselyNormedField_toNontriviallyNormedField(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedField_toNormedCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DenselyNormedField_toNontriviallyNormedField___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___closed__0;
lean_object* lp_mathlib_NonUnitalNormedRing_toNormedAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_NormedAddCommGroup_induced___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NontriviallyNormedField_ofNormNeOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormedDivisionRing_toNormedRing___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 1, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_1, 2);
lean_inc(x_7);
lean_inc(x_5);
lean_inc(x_6);
lean_dec(x_1);
x_8 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_5);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_8);
lean_ctor_set(x_9, 2, x_7);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedDivisionRing_toNormedRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NormedDivisionRing_toNormedRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DenselyNormedField_toNontriviallyNormedField(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DenselyNormedField_toNontriviallyNormedField___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DenselyNormedField_toNontriviallyNormedField___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_DenselyNormedField_toNontriviallyNormedField(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DenselyNormedField_toNontriviallyNormedField___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DenselyNormedField_toNontriviallyNormedField___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedField_toNormedDivisionRing___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_ctor_get(x_1, 1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
x_7 = lean_ctor_get(x_3, 2);
x_8 = lean_ctor_get(x_3, 3);
x_9 = lean_ctor_get(x_3, 4);
x_10 = lean_ctor_get(x_3, 5);
x_11 = lean_ctor_get(x_3, 6);
x_12 = lean_ctor_get(x_3, 7);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_3);
x_13 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_13, 0, x_5);
lean_ctor_set(x_13, 1, x_6);
lean_ctor_set(x_13, 2, x_7);
lean_ctor_set(x_13, 3, x_8);
lean_ctor_set(x_13, 4, x_9);
lean_ctor_set(x_13, 5, x_10);
lean_ctor_set(x_13, 6, x_11);
lean_ctor_set(x_13, 7, x_12);
lean_ctor_set(x_1, 1, x_13);
return x_1;
}
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_14 = lean_ctor_get(x_1, 1);
x_15 = lean_ctor_get(x_1, 0);
x_16 = lean_ctor_get(x_1, 2);
lean_inc(x_16);
lean_inc(x_14);
lean_inc(x_15);
lean_dec(x_1);
x_17 = lean_ctor_get(x_14, 0);
lean_inc_ref(x_17);
x_18 = lean_ctor_get(x_14, 1);
lean_inc(x_18);
x_19 = lean_ctor_get(x_14, 2);
lean_inc(x_19);
x_20 = lean_ctor_get(x_14, 3);
lean_inc(x_20);
x_21 = lean_ctor_get(x_14, 4);
lean_inc(x_21);
x_22 = lean_ctor_get(x_14, 5);
lean_inc(x_22);
x_23 = lean_ctor_get(x_14, 6);
lean_inc(x_23);
x_24 = lean_ctor_get(x_14, 7);
lean_inc(x_24);
if (lean_is_exclusive(x_14)) {
 lean_ctor_release(x_14, 0);
 lean_ctor_release(x_14, 1);
 lean_ctor_release(x_14, 2);
 lean_ctor_release(x_14, 3);
 lean_ctor_release(x_14, 4);
 lean_ctor_release(x_14, 5);
 lean_ctor_release(x_14, 6);
 lean_ctor_release(x_14, 7);
 x_25 = x_14;
} else {
 lean_dec_ref(x_14);
 x_25 = lean_box(0);
}
if (lean_is_scalar(x_25)) {
 x_26 = lean_alloc_ctor(0, 8, 0);
} else {
 x_26 = x_25;
}
lean_ctor_set(x_26, 0, x_17);
lean_ctor_set(x_26, 1, x_18);
lean_ctor_set(x_26, 2, x_19);
lean_ctor_set(x_26, 3, x_20);
lean_ctor_set(x_26, 4, x_21);
lean_ctor_set(x_26, 5, x_22);
lean_ctor_set(x_26, 6, x_23);
lean_ctor_set(x_26, 7, x_24);
x_27 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_27, 0, x_15);
lean_ctor_set(x_27, 1, x_26);
lean_ctor_set(x_27, 2, x_16);
return x_27;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedField_toNormedDivisionRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NormedField_toNormedDivisionRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedField_toNormedCommRing___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 1, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_1, 2);
lean_inc(x_7);
lean_inc(x_5);
lean_inc(x_6);
lean_dec(x_1);
x_8 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_5);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_8);
lean_ctor_set(x_9, 2, x_7);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedField_toNormedCommRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NormedField_toNormedCommRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NontriviallyNormedField_ofNormNeOne(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NontriviallyNormedField_ofNormNeOne___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NontriviallyNormedField_ofNormNeOne___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_NontriviallyNormedField_ofNormNeOne(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NontriviallyNormedField_ofNormNeOne___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NontriviallyNormedField_ofNormNeOne___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedDivisionRing_induced(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; 
x_10 = !lean_is_exclusive(x_5);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_11 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_11);
x_12 = lp_mathlib_Ring_toAddCommGroup___redArg(x_11);
x_13 = lp_mathlib_NormedDivisionRing_toNormedRing___redArg(x_6);
x_14 = lp_mathlib_NormedRing_toNonUnitalNormedRing___redArg(x_13);
x_15 = lp_mathlib_NonUnitalNormedRing_toNormedAddCommGroup___redArg(x_14);
x_16 = lp_mathlib_NormedAddCommGroup_toNormedAddGroup___redArg(x_15);
x_17 = lp_mathlib_NormedAddCommGroup_induced___redArg(x_4, x_12, x_16, x_8);
x_18 = lean_ctor_get(x_17, 1);
lean_inc_ref(x_18);
x_19 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_19);
x_20 = lean_ctor_get(x_19, 0);
lean_inc_ref(x_20);
x_21 = !lean_is_exclusive(x_17);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; 
x_22 = lean_ctor_get(x_17, 1);
lean_dec(x_22);
x_23 = lean_ctor_get(x_18, 0);
lean_inc_ref(x_23);
x_24 = lean_ctor_get(x_18, 1);
lean_inc(x_24);
x_25 = lean_ctor_get(x_18, 2);
lean_inc(x_25);
lean_dec_ref(x_18);
x_26 = !lean_is_exclusive(x_11);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; uint8_t x_30; 
x_27 = lean_ctor_get(x_11, 2);
lean_dec(x_27);
x_28 = lean_ctor_get(x_11, 1);
lean_dec(x_28);
x_29 = lean_ctor_get(x_11, 0);
lean_dec(x_29);
x_30 = !lean_is_exclusive(x_19);
if (x_30 == 0)
{
lean_object* x_31; uint8_t x_32; 
x_31 = lean_ctor_get(x_19, 0);
lean_dec(x_31);
x_32 = !lean_is_exclusive(x_20);
if (x_32 == 0)
{
lean_object* x_33; 
x_33 = lean_ctor_get(x_20, 0);
lean_dec(x_33);
lean_ctor_set(x_20, 0, x_23);
lean_ctor_set(x_11, 2, x_25);
lean_ctor_set(x_11, 1, x_24);
lean_ctor_set(x_17, 1, x_5);
return x_17;
}
else
{
lean_object* x_34; lean_object* x_35; 
x_34 = lean_ctor_get(x_20, 1);
lean_inc(x_34);
lean_dec(x_20);
x_35 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_35, 0, x_23);
lean_ctor_set(x_35, 1, x_34);
lean_ctor_set(x_19, 0, x_35);
lean_ctor_set(x_11, 2, x_25);
lean_ctor_set(x_11, 1, x_24);
lean_ctor_set(x_17, 1, x_5);
return x_17;
}
}
else
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_36 = lean_ctor_get(x_19, 1);
x_37 = lean_ctor_get(x_19, 2);
x_38 = lean_ctor_get(x_19, 3);
lean_inc(x_38);
lean_inc(x_37);
lean_inc(x_36);
lean_dec(x_19);
x_39 = lean_ctor_get(x_20, 1);
lean_inc(x_39);
if (lean_is_exclusive(x_20)) {
 lean_ctor_release(x_20, 0);
 lean_ctor_release(x_20, 1);
 x_40 = x_20;
} else {
 lean_dec_ref(x_20);
 x_40 = lean_box(0);
}
if (lean_is_scalar(x_40)) {
 x_41 = lean_alloc_ctor(0, 2, 0);
} else {
 x_41 = x_40;
}
lean_ctor_set(x_41, 0, x_23);
lean_ctor_set(x_41, 1, x_39);
x_42 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_42, 0, x_41);
lean_ctor_set(x_42, 1, x_36);
lean_ctor_set(x_42, 2, x_37);
lean_ctor_set(x_42, 3, x_38);
lean_ctor_set(x_11, 2, x_25);
lean_ctor_set(x_11, 1, x_24);
lean_ctor_set(x_11, 0, x_42);
lean_ctor_set(x_17, 1, x_5);
return x_17;
}
}
else
{
lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; 
x_43 = lean_ctor_get(x_11, 3);
x_44 = lean_ctor_get(x_11, 4);
lean_inc(x_44);
lean_inc(x_43);
lean_dec(x_11);
x_45 = lean_ctor_get(x_19, 1);
lean_inc(x_45);
x_46 = lean_ctor_get(x_19, 2);
lean_inc(x_46);
x_47 = lean_ctor_get(x_19, 3);
lean_inc(x_47);
if (lean_is_exclusive(x_19)) {
 lean_ctor_release(x_19, 0);
 lean_ctor_release(x_19, 1);
 lean_ctor_release(x_19, 2);
 lean_ctor_release(x_19, 3);
 x_48 = x_19;
} else {
 lean_dec_ref(x_19);
 x_48 = lean_box(0);
}
x_49 = lean_ctor_get(x_20, 1);
lean_inc(x_49);
if (lean_is_exclusive(x_20)) {
 lean_ctor_release(x_20, 0);
 lean_ctor_release(x_20, 1);
 x_50 = x_20;
} else {
 lean_dec_ref(x_20);
 x_50 = lean_box(0);
}
if (lean_is_scalar(x_50)) {
 x_51 = lean_alloc_ctor(0, 2, 0);
} else {
 x_51 = x_50;
}
lean_ctor_set(x_51, 0, x_23);
lean_ctor_set(x_51, 1, x_49);
if (lean_is_scalar(x_48)) {
 x_52 = lean_alloc_ctor(0, 4, 0);
} else {
 x_52 = x_48;
}
lean_ctor_set(x_52, 0, x_51);
lean_ctor_set(x_52, 1, x_45);
lean_ctor_set(x_52, 2, x_46);
lean_ctor_set(x_52, 3, x_47);
x_53 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_53, 0, x_52);
lean_ctor_set(x_53, 1, x_24);
lean_ctor_set(x_53, 2, x_25);
lean_ctor_set(x_53, 3, x_43);
lean_ctor_set(x_53, 4, x_44);
lean_ctor_set(x_5, 0, x_53);
lean_ctor_set(x_17, 1, x_5);
return x_17;
}
}
else
{
lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; 
x_54 = lean_ctor_get(x_17, 0);
x_55 = lean_ctor_get(x_17, 2);
lean_inc(x_55);
lean_inc(x_54);
lean_dec(x_17);
x_56 = lean_ctor_get(x_18, 0);
lean_inc_ref(x_56);
x_57 = lean_ctor_get(x_18, 1);
lean_inc(x_57);
x_58 = lean_ctor_get(x_18, 2);
lean_inc(x_58);
lean_dec_ref(x_18);
x_59 = lean_ctor_get(x_11, 3);
lean_inc(x_59);
x_60 = lean_ctor_get(x_11, 4);
lean_inc(x_60);
if (lean_is_exclusive(x_11)) {
 lean_ctor_release(x_11, 0);
 lean_ctor_release(x_11, 1);
 lean_ctor_release(x_11, 2);
 lean_ctor_release(x_11, 3);
 lean_ctor_release(x_11, 4);
 x_61 = x_11;
} else {
 lean_dec_ref(x_11);
 x_61 = lean_box(0);
}
x_62 = lean_ctor_get(x_19, 1);
lean_inc(x_62);
x_63 = lean_ctor_get(x_19, 2);
lean_inc(x_63);
x_64 = lean_ctor_get(x_19, 3);
lean_inc(x_64);
if (lean_is_exclusive(x_19)) {
 lean_ctor_release(x_19, 0);
 lean_ctor_release(x_19, 1);
 lean_ctor_release(x_19, 2);
 lean_ctor_release(x_19, 3);
 x_65 = x_19;
} else {
 lean_dec_ref(x_19);
 x_65 = lean_box(0);
}
x_66 = lean_ctor_get(x_20, 1);
lean_inc(x_66);
if (lean_is_exclusive(x_20)) {
 lean_ctor_release(x_20, 0);
 lean_ctor_release(x_20, 1);
 x_67 = x_20;
} else {
 lean_dec_ref(x_20);
 x_67 = lean_box(0);
}
if (lean_is_scalar(x_67)) {
 x_68 = lean_alloc_ctor(0, 2, 0);
} else {
 x_68 = x_67;
}
lean_ctor_set(x_68, 0, x_56);
lean_ctor_set(x_68, 1, x_66);
if (lean_is_scalar(x_65)) {
 x_69 = lean_alloc_ctor(0, 4, 0);
} else {
 x_69 = x_65;
}
lean_ctor_set(x_69, 0, x_68);
lean_ctor_set(x_69, 1, x_62);
lean_ctor_set(x_69, 2, x_63);
lean_ctor_set(x_69, 3, x_64);
if (lean_is_scalar(x_61)) {
 x_70 = lean_alloc_ctor(0, 5, 0);
} else {
 x_70 = x_61;
}
lean_ctor_set(x_70, 0, x_69);
lean_ctor_set(x_70, 1, x_57);
lean_ctor_set(x_70, 2, x_58);
lean_ctor_set(x_70, 3, x_59);
lean_ctor_set(x_70, 4, x_60);
lean_ctor_set(x_5, 0, x_70);
x_71 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_71, 0, x_54);
lean_ctor_set(x_71, 1, x_5);
lean_ctor_set(x_71, 2, x_55);
return x_71;
}
}
else
{
lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; 
x_72 = lean_ctor_get(x_5, 0);
x_73 = lean_ctor_get(x_5, 1);
x_74 = lean_ctor_get(x_5, 2);
x_75 = lean_ctor_get(x_5, 3);
x_76 = lean_ctor_get(x_5, 4);
x_77 = lean_ctor_get(x_5, 5);
x_78 = lean_ctor_get(x_5, 6);
x_79 = lean_ctor_get(x_5, 7);
lean_inc(x_79);
lean_inc(x_78);
lean_inc(x_77);
lean_inc(x_76);
lean_inc(x_75);
lean_inc(x_74);
lean_inc(x_73);
lean_inc(x_72);
lean_dec(x_5);
lean_inc_ref(x_72);
x_80 = lp_mathlib_Ring_toAddCommGroup___redArg(x_72);
x_81 = lp_mathlib_NormedDivisionRing_toNormedRing___redArg(x_6);
x_82 = lp_mathlib_NormedRing_toNonUnitalNormedRing___redArg(x_81);
x_83 = lp_mathlib_NonUnitalNormedRing_toNormedAddCommGroup___redArg(x_82);
x_84 = lp_mathlib_NormedAddCommGroup_toNormedAddGroup___redArg(x_83);
x_85 = lp_mathlib_NormedAddCommGroup_induced___redArg(x_4, x_80, x_84, x_8);
x_86 = lean_ctor_get(x_85, 1);
lean_inc_ref(x_86);
x_87 = lean_ctor_get(x_72, 0);
lean_inc_ref(x_87);
x_88 = lean_ctor_get(x_87, 0);
lean_inc_ref(x_88);
x_89 = lean_ctor_get(x_85, 0);
lean_inc(x_89);
x_90 = lean_ctor_get(x_85, 2);
lean_inc_ref(x_90);
if (lean_is_exclusive(x_85)) {
 lean_ctor_release(x_85, 0);
 lean_ctor_release(x_85, 1);
 lean_ctor_release(x_85, 2);
 x_91 = x_85;
} else {
 lean_dec_ref(x_85);
 x_91 = lean_box(0);
}
x_92 = lean_ctor_get(x_86, 0);
lean_inc_ref(x_92);
x_93 = lean_ctor_get(x_86, 1);
lean_inc(x_93);
x_94 = lean_ctor_get(x_86, 2);
lean_inc(x_94);
lean_dec_ref(x_86);
x_95 = lean_ctor_get(x_72, 3);
lean_inc(x_95);
x_96 = lean_ctor_get(x_72, 4);
lean_inc(x_96);
if (lean_is_exclusive(x_72)) {
 lean_ctor_release(x_72, 0);
 lean_ctor_release(x_72, 1);
 lean_ctor_release(x_72, 2);
 lean_ctor_release(x_72, 3);
 lean_ctor_release(x_72, 4);
 x_97 = x_72;
} else {
 lean_dec_ref(x_72);
 x_97 = lean_box(0);
}
x_98 = lean_ctor_get(x_87, 1);
lean_inc(x_98);
x_99 = lean_ctor_get(x_87, 2);
lean_inc(x_99);
x_100 = lean_ctor_get(x_87, 3);
lean_inc(x_100);
if (lean_is_exclusive(x_87)) {
 lean_ctor_release(x_87, 0);
 lean_ctor_release(x_87, 1);
 lean_ctor_release(x_87, 2);
 lean_ctor_release(x_87, 3);
 x_101 = x_87;
} else {
 lean_dec_ref(x_87);
 x_101 = lean_box(0);
}
x_102 = lean_ctor_get(x_88, 1);
lean_inc(x_102);
if (lean_is_exclusive(x_88)) {
 lean_ctor_release(x_88, 0);
 lean_ctor_release(x_88, 1);
 x_103 = x_88;
} else {
 lean_dec_ref(x_88);
 x_103 = lean_box(0);
}
if (lean_is_scalar(x_103)) {
 x_104 = lean_alloc_ctor(0, 2, 0);
} else {
 x_104 = x_103;
}
lean_ctor_set(x_104, 0, x_92);
lean_ctor_set(x_104, 1, x_102);
if (lean_is_scalar(x_101)) {
 x_105 = lean_alloc_ctor(0, 4, 0);
} else {
 x_105 = x_101;
}
lean_ctor_set(x_105, 0, x_104);
lean_ctor_set(x_105, 1, x_98);
lean_ctor_set(x_105, 2, x_99);
lean_ctor_set(x_105, 3, x_100);
if (lean_is_scalar(x_97)) {
 x_106 = lean_alloc_ctor(0, 5, 0);
} else {
 x_106 = x_97;
}
lean_ctor_set(x_106, 0, x_105);
lean_ctor_set(x_106, 1, x_93);
lean_ctor_set(x_106, 2, x_94);
lean_ctor_set(x_106, 3, x_95);
lean_ctor_set(x_106, 4, x_96);
x_107 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_107, 0, x_106);
lean_ctor_set(x_107, 1, x_73);
lean_ctor_set(x_107, 2, x_74);
lean_ctor_set(x_107, 3, x_75);
lean_ctor_set(x_107, 4, x_76);
lean_ctor_set(x_107, 5, x_77);
lean_ctor_set(x_107, 6, x_78);
lean_ctor_set(x_107, 7, x_79);
if (lean_is_scalar(x_91)) {
 x_108 = lean_alloc_ctor(0, 3, 0);
} else {
 x_108 = x_91;
}
lean_ctor_set(x_108, 0, x_89);
lean_ctor_set(x_108, 1, x_107);
lean_ctor_set(x_108, 2, x_90);
return x_108;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedDivisionRing_induced___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_6 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_6);
x_7 = lp_mathlib_Ring_toAddCommGroup___redArg(x_6);
x_8 = lp_mathlib_NormedDivisionRing_toNormedRing___redArg(x_3);
x_9 = lp_mathlib_NormedRing_toNonUnitalNormedRing___redArg(x_8);
x_10 = lp_mathlib_NonUnitalNormedRing_toNormedAddCommGroup___redArg(x_9);
x_11 = lp_mathlib_NormedAddCommGroup_toNormedAddGroup___redArg(x_10);
x_12 = lp_mathlib_NormedAddCommGroup_induced___redArg(x_1, x_7, x_11, x_4);
x_13 = lean_ctor_get(x_12, 1);
lean_inc_ref(x_13);
x_14 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_14);
x_15 = lean_ctor_get(x_14, 0);
lean_inc_ref(x_15);
x_16 = !lean_is_exclusive(x_12);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_17 = lean_ctor_get(x_12, 1);
lean_dec(x_17);
x_18 = lean_ctor_get(x_13, 0);
lean_inc_ref(x_18);
x_19 = lean_ctor_get(x_13, 1);
lean_inc(x_19);
x_20 = lean_ctor_get(x_13, 2);
lean_inc(x_20);
lean_dec_ref(x_13);
x_21 = !lean_is_exclusive(x_6);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; 
x_22 = lean_ctor_get(x_6, 2);
lean_dec(x_22);
x_23 = lean_ctor_get(x_6, 1);
lean_dec(x_23);
x_24 = lean_ctor_get(x_6, 0);
lean_dec(x_24);
x_25 = !lean_is_exclusive(x_14);
if (x_25 == 0)
{
lean_object* x_26; uint8_t x_27; 
x_26 = lean_ctor_get(x_14, 0);
lean_dec(x_26);
x_27 = !lean_is_exclusive(x_15);
if (x_27 == 0)
{
lean_object* x_28; 
x_28 = lean_ctor_get(x_15, 0);
lean_dec(x_28);
lean_ctor_set(x_15, 0, x_18);
lean_ctor_set(x_6, 2, x_20);
lean_ctor_set(x_6, 1, x_19);
lean_ctor_set(x_12, 1, x_2);
return x_12;
}
else
{
lean_object* x_29; lean_object* x_30; 
x_29 = lean_ctor_get(x_15, 1);
lean_inc(x_29);
lean_dec(x_15);
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_18);
lean_ctor_set(x_30, 1, x_29);
lean_ctor_set(x_14, 0, x_30);
lean_ctor_set(x_6, 2, x_20);
lean_ctor_set(x_6, 1, x_19);
lean_ctor_set(x_12, 1, x_2);
return x_12;
}
}
else
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_31 = lean_ctor_get(x_14, 1);
x_32 = lean_ctor_get(x_14, 2);
x_33 = lean_ctor_get(x_14, 3);
lean_inc(x_33);
lean_inc(x_32);
lean_inc(x_31);
lean_dec(x_14);
x_34 = lean_ctor_get(x_15, 1);
lean_inc(x_34);
if (lean_is_exclusive(x_15)) {
 lean_ctor_release(x_15, 0);
 lean_ctor_release(x_15, 1);
 x_35 = x_15;
} else {
 lean_dec_ref(x_15);
 x_35 = lean_box(0);
}
if (lean_is_scalar(x_35)) {
 x_36 = lean_alloc_ctor(0, 2, 0);
} else {
 x_36 = x_35;
}
lean_ctor_set(x_36, 0, x_18);
lean_ctor_set(x_36, 1, x_34);
x_37 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_37, 0, x_36);
lean_ctor_set(x_37, 1, x_31);
lean_ctor_set(x_37, 2, x_32);
lean_ctor_set(x_37, 3, x_33);
lean_ctor_set(x_6, 2, x_20);
lean_ctor_set(x_6, 1, x_19);
lean_ctor_set(x_6, 0, x_37);
lean_ctor_set(x_12, 1, x_2);
return x_12;
}
}
else
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; 
x_38 = lean_ctor_get(x_6, 3);
x_39 = lean_ctor_get(x_6, 4);
lean_inc(x_39);
lean_inc(x_38);
lean_dec(x_6);
x_40 = lean_ctor_get(x_14, 1);
lean_inc(x_40);
x_41 = lean_ctor_get(x_14, 2);
lean_inc(x_41);
x_42 = lean_ctor_get(x_14, 3);
lean_inc(x_42);
if (lean_is_exclusive(x_14)) {
 lean_ctor_release(x_14, 0);
 lean_ctor_release(x_14, 1);
 lean_ctor_release(x_14, 2);
 lean_ctor_release(x_14, 3);
 x_43 = x_14;
} else {
 lean_dec_ref(x_14);
 x_43 = lean_box(0);
}
x_44 = lean_ctor_get(x_15, 1);
lean_inc(x_44);
if (lean_is_exclusive(x_15)) {
 lean_ctor_release(x_15, 0);
 lean_ctor_release(x_15, 1);
 x_45 = x_15;
} else {
 lean_dec_ref(x_15);
 x_45 = lean_box(0);
}
if (lean_is_scalar(x_45)) {
 x_46 = lean_alloc_ctor(0, 2, 0);
} else {
 x_46 = x_45;
}
lean_ctor_set(x_46, 0, x_18);
lean_ctor_set(x_46, 1, x_44);
if (lean_is_scalar(x_43)) {
 x_47 = lean_alloc_ctor(0, 4, 0);
} else {
 x_47 = x_43;
}
lean_ctor_set(x_47, 0, x_46);
lean_ctor_set(x_47, 1, x_40);
lean_ctor_set(x_47, 2, x_41);
lean_ctor_set(x_47, 3, x_42);
x_48 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_48, 0, x_47);
lean_ctor_set(x_48, 1, x_19);
lean_ctor_set(x_48, 2, x_20);
lean_ctor_set(x_48, 3, x_38);
lean_ctor_set(x_48, 4, x_39);
lean_ctor_set(x_2, 0, x_48);
lean_ctor_set(x_12, 1, x_2);
return x_12;
}
}
else
{
lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; 
x_49 = lean_ctor_get(x_12, 0);
x_50 = lean_ctor_get(x_12, 2);
lean_inc(x_50);
lean_inc(x_49);
lean_dec(x_12);
x_51 = lean_ctor_get(x_13, 0);
lean_inc_ref(x_51);
x_52 = lean_ctor_get(x_13, 1);
lean_inc(x_52);
x_53 = lean_ctor_get(x_13, 2);
lean_inc(x_53);
lean_dec_ref(x_13);
x_54 = lean_ctor_get(x_6, 3);
lean_inc(x_54);
x_55 = lean_ctor_get(x_6, 4);
lean_inc(x_55);
if (lean_is_exclusive(x_6)) {
 lean_ctor_release(x_6, 0);
 lean_ctor_release(x_6, 1);
 lean_ctor_release(x_6, 2);
 lean_ctor_release(x_6, 3);
 lean_ctor_release(x_6, 4);
 x_56 = x_6;
} else {
 lean_dec_ref(x_6);
 x_56 = lean_box(0);
}
x_57 = lean_ctor_get(x_14, 1);
lean_inc(x_57);
x_58 = lean_ctor_get(x_14, 2);
lean_inc(x_58);
x_59 = lean_ctor_get(x_14, 3);
lean_inc(x_59);
if (lean_is_exclusive(x_14)) {
 lean_ctor_release(x_14, 0);
 lean_ctor_release(x_14, 1);
 lean_ctor_release(x_14, 2);
 lean_ctor_release(x_14, 3);
 x_60 = x_14;
} else {
 lean_dec_ref(x_14);
 x_60 = lean_box(0);
}
x_61 = lean_ctor_get(x_15, 1);
lean_inc(x_61);
if (lean_is_exclusive(x_15)) {
 lean_ctor_release(x_15, 0);
 lean_ctor_release(x_15, 1);
 x_62 = x_15;
} else {
 lean_dec_ref(x_15);
 x_62 = lean_box(0);
}
if (lean_is_scalar(x_62)) {
 x_63 = lean_alloc_ctor(0, 2, 0);
} else {
 x_63 = x_62;
}
lean_ctor_set(x_63, 0, x_51);
lean_ctor_set(x_63, 1, x_61);
if (lean_is_scalar(x_60)) {
 x_64 = lean_alloc_ctor(0, 4, 0);
} else {
 x_64 = x_60;
}
lean_ctor_set(x_64, 0, x_63);
lean_ctor_set(x_64, 1, x_57);
lean_ctor_set(x_64, 2, x_58);
lean_ctor_set(x_64, 3, x_59);
if (lean_is_scalar(x_56)) {
 x_65 = lean_alloc_ctor(0, 5, 0);
} else {
 x_65 = x_56;
}
lean_ctor_set(x_65, 0, x_64);
lean_ctor_set(x_65, 1, x_52);
lean_ctor_set(x_65, 2, x_53);
lean_ctor_set(x_65, 3, x_54);
lean_ctor_set(x_65, 4, x_55);
lean_ctor_set(x_2, 0, x_65);
x_66 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_66, 0, x_49);
lean_ctor_set(x_66, 1, x_2);
lean_ctor_set(x_66, 2, x_50);
return x_66;
}
}
else
{
lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; 
x_67 = lean_ctor_get(x_2, 0);
x_68 = lean_ctor_get(x_2, 1);
x_69 = lean_ctor_get(x_2, 2);
x_70 = lean_ctor_get(x_2, 3);
x_71 = lean_ctor_get(x_2, 4);
x_72 = lean_ctor_get(x_2, 5);
x_73 = lean_ctor_get(x_2, 6);
x_74 = lean_ctor_get(x_2, 7);
lean_inc(x_74);
lean_inc(x_73);
lean_inc(x_72);
lean_inc(x_71);
lean_inc(x_70);
lean_inc(x_69);
lean_inc(x_68);
lean_inc(x_67);
lean_dec(x_2);
lean_inc_ref(x_67);
x_75 = lp_mathlib_Ring_toAddCommGroup___redArg(x_67);
x_76 = lp_mathlib_NormedDivisionRing_toNormedRing___redArg(x_3);
x_77 = lp_mathlib_NormedRing_toNonUnitalNormedRing___redArg(x_76);
x_78 = lp_mathlib_NonUnitalNormedRing_toNormedAddCommGroup___redArg(x_77);
x_79 = lp_mathlib_NormedAddCommGroup_toNormedAddGroup___redArg(x_78);
x_80 = lp_mathlib_NormedAddCommGroup_induced___redArg(x_1, x_75, x_79, x_4);
x_81 = lean_ctor_get(x_80, 1);
lean_inc_ref(x_81);
x_82 = lean_ctor_get(x_67, 0);
lean_inc_ref(x_82);
x_83 = lean_ctor_get(x_82, 0);
lean_inc_ref(x_83);
x_84 = lean_ctor_get(x_80, 0);
lean_inc(x_84);
x_85 = lean_ctor_get(x_80, 2);
lean_inc_ref(x_85);
if (lean_is_exclusive(x_80)) {
 lean_ctor_release(x_80, 0);
 lean_ctor_release(x_80, 1);
 lean_ctor_release(x_80, 2);
 x_86 = x_80;
} else {
 lean_dec_ref(x_80);
 x_86 = lean_box(0);
}
x_87 = lean_ctor_get(x_81, 0);
lean_inc_ref(x_87);
x_88 = lean_ctor_get(x_81, 1);
lean_inc(x_88);
x_89 = lean_ctor_get(x_81, 2);
lean_inc(x_89);
lean_dec_ref(x_81);
x_90 = lean_ctor_get(x_67, 3);
lean_inc(x_90);
x_91 = lean_ctor_get(x_67, 4);
lean_inc(x_91);
if (lean_is_exclusive(x_67)) {
 lean_ctor_release(x_67, 0);
 lean_ctor_release(x_67, 1);
 lean_ctor_release(x_67, 2);
 lean_ctor_release(x_67, 3);
 lean_ctor_release(x_67, 4);
 x_92 = x_67;
} else {
 lean_dec_ref(x_67);
 x_92 = lean_box(0);
}
x_93 = lean_ctor_get(x_82, 1);
lean_inc(x_93);
x_94 = lean_ctor_get(x_82, 2);
lean_inc(x_94);
x_95 = lean_ctor_get(x_82, 3);
lean_inc(x_95);
if (lean_is_exclusive(x_82)) {
 lean_ctor_release(x_82, 0);
 lean_ctor_release(x_82, 1);
 lean_ctor_release(x_82, 2);
 lean_ctor_release(x_82, 3);
 x_96 = x_82;
} else {
 lean_dec_ref(x_82);
 x_96 = lean_box(0);
}
x_97 = lean_ctor_get(x_83, 1);
lean_inc(x_97);
if (lean_is_exclusive(x_83)) {
 lean_ctor_release(x_83, 0);
 lean_ctor_release(x_83, 1);
 x_98 = x_83;
} else {
 lean_dec_ref(x_83);
 x_98 = lean_box(0);
}
if (lean_is_scalar(x_98)) {
 x_99 = lean_alloc_ctor(0, 2, 0);
} else {
 x_99 = x_98;
}
lean_ctor_set(x_99, 0, x_87);
lean_ctor_set(x_99, 1, x_97);
if (lean_is_scalar(x_96)) {
 x_100 = lean_alloc_ctor(0, 4, 0);
} else {
 x_100 = x_96;
}
lean_ctor_set(x_100, 0, x_99);
lean_ctor_set(x_100, 1, x_93);
lean_ctor_set(x_100, 2, x_94);
lean_ctor_set(x_100, 3, x_95);
if (lean_is_scalar(x_92)) {
 x_101 = lean_alloc_ctor(0, 5, 0);
} else {
 x_101 = x_92;
}
lean_ctor_set(x_101, 0, x_100);
lean_ctor_set(x_101, 1, x_88);
lean_ctor_set(x_101, 2, x_89);
lean_ctor_set(x_101, 3, x_90);
lean_ctor_set(x_101, 4, x_91);
x_102 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_102, 0, x_101);
lean_ctor_set(x_102, 1, x_68);
lean_ctor_set(x_102, 2, x_69);
lean_ctor_set(x_102, 3, x_70);
lean_ctor_set(x_102, 4, x_71);
lean_ctor_set(x_102, 5, x_72);
lean_ctor_set(x_102, 6, x_73);
lean_ctor_set(x_102, 7, x_74);
if (lean_is_scalar(x_86)) {
 x_103 = lean_alloc_ctor(0, 3, 0);
} else {
 x_103 = x_86;
}
lean_ctor_set(x_103, 0, x_84);
lean_ctor_set(x_103, 1, x_102);
lean_ctor_set(x_103, 2, x_85);
return x_103;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedField_induced(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; 
lean_inc_ref(x_5);
x_10 = lp_mathlib_Field_toDivisionRing___redArg(x_5);
x_11 = lp_mathlib_NormedField_toNormedDivisionRing___redArg(x_6);
x_12 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_12);
x_13 = lean_ctor_get(x_10, 1);
lean_inc(x_13);
x_14 = lean_ctor_get(x_10, 2);
lean_inc(x_14);
x_15 = lean_ctor_get(x_10, 4);
lean_inc(x_15);
x_16 = lean_ctor_get(x_10, 5);
lean_inc(x_16);
lean_dec_ref(x_10);
lean_inc_ref(x_12);
x_17 = lp_mathlib_Ring_toAddCommGroup___redArg(x_12);
x_18 = lp_mathlib_NormedDivisionRing_toNormedRing___redArg(x_11);
x_19 = lp_mathlib_NormedRing_toNonUnitalNormedRing___redArg(x_18);
x_20 = lp_mathlib_NonUnitalNormedRing_toNormedAddCommGroup___redArg(x_19);
x_21 = lp_mathlib_NormedAddCommGroup_toNormedAddGroup___redArg(x_20);
x_22 = lp_mathlib_NormedAddCommGroup_induced___redArg(x_4, x_17, x_21, x_8);
x_23 = lean_ctor_get(x_22, 1);
lean_inc_ref(x_23);
x_24 = lean_ctor_get(x_12, 0);
lean_inc_ref(x_24);
x_25 = lean_ctor_get(x_24, 0);
lean_inc_ref(x_25);
x_26 = !lean_is_exclusive(x_22);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; uint8_t x_31; 
x_27 = lean_ctor_get(x_22, 1);
lean_dec(x_27);
x_28 = lean_ctor_get(x_23, 0);
lean_inc_ref(x_28);
x_29 = lean_ctor_get(x_23, 1);
lean_inc(x_29);
x_30 = lean_ctor_get(x_23, 2);
lean_inc(x_30);
lean_dec_ref(x_23);
x_31 = !lean_is_exclusive(x_12);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; uint8_t x_35; 
x_32 = lean_ctor_get(x_12, 2);
lean_dec(x_32);
x_33 = lean_ctor_get(x_12, 1);
lean_dec(x_33);
x_34 = lean_ctor_get(x_12, 0);
lean_dec(x_34);
x_35 = !lean_is_exclusive(x_24);
if (x_35 == 0)
{
lean_object* x_36; uint8_t x_37; 
x_36 = lean_ctor_get(x_24, 0);
lean_dec(x_36);
x_37 = !lean_is_exclusive(x_25);
if (x_37 == 0)
{
lean_object* x_38; uint8_t x_39; 
x_38 = lean_ctor_get(x_25, 0);
lean_dec(x_38);
lean_ctor_set(x_25, 0, x_28);
lean_ctor_set(x_12, 2, x_30);
lean_ctor_set(x_12, 1, x_29);
x_39 = !lean_is_exclusive(x_5);
if (x_39 == 0)
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; 
x_40 = lean_ctor_get(x_5, 5);
lean_dec(x_40);
x_41 = lean_ctor_get(x_5, 4);
lean_dec(x_41);
x_42 = lean_ctor_get(x_5, 2);
lean_dec(x_42);
x_43 = lean_ctor_get(x_5, 1);
lean_dec(x_43);
x_44 = lean_ctor_get(x_5, 0);
lean_dec(x_44);
lean_ctor_set(x_5, 5, x_16);
lean_ctor_set(x_5, 4, x_15);
lean_ctor_set(x_5, 2, x_14);
lean_ctor_set(x_5, 1, x_13);
lean_ctor_set(x_5, 0, x_12);
lean_ctor_set(x_22, 1, x_5);
return x_22;
}
else
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; 
x_45 = lean_ctor_get(x_5, 3);
x_46 = lean_ctor_get(x_5, 6);
x_47 = lean_ctor_get(x_5, 7);
lean_inc(x_47);
lean_inc(x_46);
lean_inc(x_45);
lean_dec(x_5);
x_48 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_48, 0, x_12);
lean_ctor_set(x_48, 1, x_13);
lean_ctor_set(x_48, 2, x_14);
lean_ctor_set(x_48, 3, x_45);
lean_ctor_set(x_48, 4, x_15);
lean_ctor_set(x_48, 5, x_16);
lean_ctor_set(x_48, 6, x_46);
lean_ctor_set(x_48, 7, x_47);
lean_ctor_set(x_22, 1, x_48);
return x_22;
}
}
else
{
lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; 
x_49 = lean_ctor_get(x_25, 1);
lean_inc(x_49);
lean_dec(x_25);
x_50 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_50, 0, x_28);
lean_ctor_set(x_50, 1, x_49);
lean_ctor_set(x_24, 0, x_50);
lean_ctor_set(x_12, 2, x_30);
lean_ctor_set(x_12, 1, x_29);
x_51 = lean_ctor_get(x_5, 3);
lean_inc(x_51);
x_52 = lean_ctor_get(x_5, 6);
lean_inc(x_52);
x_53 = lean_ctor_get(x_5, 7);
lean_inc(x_53);
if (lean_is_exclusive(x_5)) {
 lean_ctor_release(x_5, 0);
 lean_ctor_release(x_5, 1);
 lean_ctor_release(x_5, 2);
 lean_ctor_release(x_5, 3);
 lean_ctor_release(x_5, 4);
 lean_ctor_release(x_5, 5);
 lean_ctor_release(x_5, 6);
 lean_ctor_release(x_5, 7);
 x_54 = x_5;
} else {
 lean_dec_ref(x_5);
 x_54 = lean_box(0);
}
if (lean_is_scalar(x_54)) {
 x_55 = lean_alloc_ctor(0, 8, 0);
} else {
 x_55 = x_54;
}
lean_ctor_set(x_55, 0, x_12);
lean_ctor_set(x_55, 1, x_13);
lean_ctor_set(x_55, 2, x_14);
lean_ctor_set(x_55, 3, x_51);
lean_ctor_set(x_55, 4, x_15);
lean_ctor_set(x_55, 5, x_16);
lean_ctor_set(x_55, 6, x_52);
lean_ctor_set(x_55, 7, x_53);
lean_ctor_set(x_22, 1, x_55);
return x_22;
}
}
else
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; 
x_56 = lean_ctor_get(x_24, 1);
x_57 = lean_ctor_get(x_24, 2);
x_58 = lean_ctor_get(x_24, 3);
lean_inc(x_58);
lean_inc(x_57);
lean_inc(x_56);
lean_dec(x_24);
x_59 = lean_ctor_get(x_25, 1);
lean_inc(x_59);
if (lean_is_exclusive(x_25)) {
 lean_ctor_release(x_25, 0);
 lean_ctor_release(x_25, 1);
 x_60 = x_25;
} else {
 lean_dec_ref(x_25);
 x_60 = lean_box(0);
}
if (lean_is_scalar(x_60)) {
 x_61 = lean_alloc_ctor(0, 2, 0);
} else {
 x_61 = x_60;
}
lean_ctor_set(x_61, 0, x_28);
lean_ctor_set(x_61, 1, x_59);
x_62 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_62, 0, x_61);
lean_ctor_set(x_62, 1, x_56);
lean_ctor_set(x_62, 2, x_57);
lean_ctor_set(x_62, 3, x_58);
lean_ctor_set(x_12, 2, x_30);
lean_ctor_set(x_12, 1, x_29);
lean_ctor_set(x_12, 0, x_62);
x_63 = lean_ctor_get(x_5, 3);
lean_inc(x_63);
x_64 = lean_ctor_get(x_5, 6);
lean_inc(x_64);
x_65 = lean_ctor_get(x_5, 7);
lean_inc(x_65);
if (lean_is_exclusive(x_5)) {
 lean_ctor_release(x_5, 0);
 lean_ctor_release(x_5, 1);
 lean_ctor_release(x_5, 2);
 lean_ctor_release(x_5, 3);
 lean_ctor_release(x_5, 4);
 lean_ctor_release(x_5, 5);
 lean_ctor_release(x_5, 6);
 lean_ctor_release(x_5, 7);
 x_66 = x_5;
} else {
 lean_dec_ref(x_5);
 x_66 = lean_box(0);
}
if (lean_is_scalar(x_66)) {
 x_67 = lean_alloc_ctor(0, 8, 0);
} else {
 x_67 = x_66;
}
lean_ctor_set(x_67, 0, x_12);
lean_ctor_set(x_67, 1, x_13);
lean_ctor_set(x_67, 2, x_14);
lean_ctor_set(x_67, 3, x_63);
lean_ctor_set(x_67, 4, x_15);
lean_ctor_set(x_67, 5, x_16);
lean_ctor_set(x_67, 6, x_64);
lean_ctor_set(x_67, 7, x_65);
lean_ctor_set(x_22, 1, x_67);
return x_22;
}
}
else
{
lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; 
x_68 = lean_ctor_get(x_12, 3);
x_69 = lean_ctor_get(x_12, 4);
lean_inc(x_69);
lean_inc(x_68);
lean_dec(x_12);
x_70 = lean_ctor_get(x_24, 1);
lean_inc(x_70);
x_71 = lean_ctor_get(x_24, 2);
lean_inc(x_71);
x_72 = lean_ctor_get(x_24, 3);
lean_inc(x_72);
if (lean_is_exclusive(x_24)) {
 lean_ctor_release(x_24, 0);
 lean_ctor_release(x_24, 1);
 lean_ctor_release(x_24, 2);
 lean_ctor_release(x_24, 3);
 x_73 = x_24;
} else {
 lean_dec_ref(x_24);
 x_73 = lean_box(0);
}
x_74 = lean_ctor_get(x_25, 1);
lean_inc(x_74);
if (lean_is_exclusive(x_25)) {
 lean_ctor_release(x_25, 0);
 lean_ctor_release(x_25, 1);
 x_75 = x_25;
} else {
 lean_dec_ref(x_25);
 x_75 = lean_box(0);
}
if (lean_is_scalar(x_75)) {
 x_76 = lean_alloc_ctor(0, 2, 0);
} else {
 x_76 = x_75;
}
lean_ctor_set(x_76, 0, x_28);
lean_ctor_set(x_76, 1, x_74);
if (lean_is_scalar(x_73)) {
 x_77 = lean_alloc_ctor(0, 4, 0);
} else {
 x_77 = x_73;
}
lean_ctor_set(x_77, 0, x_76);
lean_ctor_set(x_77, 1, x_70);
lean_ctor_set(x_77, 2, x_71);
lean_ctor_set(x_77, 3, x_72);
x_78 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_78, 0, x_77);
lean_ctor_set(x_78, 1, x_29);
lean_ctor_set(x_78, 2, x_30);
lean_ctor_set(x_78, 3, x_68);
lean_ctor_set(x_78, 4, x_69);
x_79 = lean_ctor_get(x_5, 3);
lean_inc(x_79);
x_80 = lean_ctor_get(x_5, 6);
lean_inc(x_80);
x_81 = lean_ctor_get(x_5, 7);
lean_inc(x_81);
if (lean_is_exclusive(x_5)) {
 lean_ctor_release(x_5, 0);
 lean_ctor_release(x_5, 1);
 lean_ctor_release(x_5, 2);
 lean_ctor_release(x_5, 3);
 lean_ctor_release(x_5, 4);
 lean_ctor_release(x_5, 5);
 lean_ctor_release(x_5, 6);
 lean_ctor_release(x_5, 7);
 x_82 = x_5;
} else {
 lean_dec_ref(x_5);
 x_82 = lean_box(0);
}
if (lean_is_scalar(x_82)) {
 x_83 = lean_alloc_ctor(0, 8, 0);
} else {
 x_83 = x_82;
}
lean_ctor_set(x_83, 0, x_78);
lean_ctor_set(x_83, 1, x_13);
lean_ctor_set(x_83, 2, x_14);
lean_ctor_set(x_83, 3, x_79);
lean_ctor_set(x_83, 4, x_15);
lean_ctor_set(x_83, 5, x_16);
lean_ctor_set(x_83, 6, x_80);
lean_ctor_set(x_83, 7, x_81);
lean_ctor_set(x_22, 1, x_83);
return x_22;
}
}
else
{
lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; 
x_84 = lean_ctor_get(x_22, 0);
x_85 = lean_ctor_get(x_22, 2);
lean_inc(x_85);
lean_inc(x_84);
lean_dec(x_22);
x_86 = lean_ctor_get(x_23, 0);
lean_inc_ref(x_86);
x_87 = lean_ctor_get(x_23, 1);
lean_inc(x_87);
x_88 = lean_ctor_get(x_23, 2);
lean_inc(x_88);
lean_dec_ref(x_23);
x_89 = lean_ctor_get(x_12, 3);
lean_inc(x_89);
x_90 = lean_ctor_get(x_12, 4);
lean_inc(x_90);
if (lean_is_exclusive(x_12)) {
 lean_ctor_release(x_12, 0);
 lean_ctor_release(x_12, 1);
 lean_ctor_release(x_12, 2);
 lean_ctor_release(x_12, 3);
 lean_ctor_release(x_12, 4);
 x_91 = x_12;
} else {
 lean_dec_ref(x_12);
 x_91 = lean_box(0);
}
x_92 = lean_ctor_get(x_24, 1);
lean_inc(x_92);
x_93 = lean_ctor_get(x_24, 2);
lean_inc(x_93);
x_94 = lean_ctor_get(x_24, 3);
lean_inc(x_94);
if (lean_is_exclusive(x_24)) {
 lean_ctor_release(x_24, 0);
 lean_ctor_release(x_24, 1);
 lean_ctor_release(x_24, 2);
 lean_ctor_release(x_24, 3);
 x_95 = x_24;
} else {
 lean_dec_ref(x_24);
 x_95 = lean_box(0);
}
x_96 = lean_ctor_get(x_25, 1);
lean_inc(x_96);
if (lean_is_exclusive(x_25)) {
 lean_ctor_release(x_25, 0);
 lean_ctor_release(x_25, 1);
 x_97 = x_25;
} else {
 lean_dec_ref(x_25);
 x_97 = lean_box(0);
}
if (lean_is_scalar(x_97)) {
 x_98 = lean_alloc_ctor(0, 2, 0);
} else {
 x_98 = x_97;
}
lean_ctor_set(x_98, 0, x_86);
lean_ctor_set(x_98, 1, x_96);
if (lean_is_scalar(x_95)) {
 x_99 = lean_alloc_ctor(0, 4, 0);
} else {
 x_99 = x_95;
}
lean_ctor_set(x_99, 0, x_98);
lean_ctor_set(x_99, 1, x_92);
lean_ctor_set(x_99, 2, x_93);
lean_ctor_set(x_99, 3, x_94);
if (lean_is_scalar(x_91)) {
 x_100 = lean_alloc_ctor(0, 5, 0);
} else {
 x_100 = x_91;
}
lean_ctor_set(x_100, 0, x_99);
lean_ctor_set(x_100, 1, x_87);
lean_ctor_set(x_100, 2, x_88);
lean_ctor_set(x_100, 3, x_89);
lean_ctor_set(x_100, 4, x_90);
x_101 = lean_ctor_get(x_5, 3);
lean_inc(x_101);
x_102 = lean_ctor_get(x_5, 6);
lean_inc(x_102);
x_103 = lean_ctor_get(x_5, 7);
lean_inc(x_103);
if (lean_is_exclusive(x_5)) {
 lean_ctor_release(x_5, 0);
 lean_ctor_release(x_5, 1);
 lean_ctor_release(x_5, 2);
 lean_ctor_release(x_5, 3);
 lean_ctor_release(x_5, 4);
 lean_ctor_release(x_5, 5);
 lean_ctor_release(x_5, 6);
 lean_ctor_release(x_5, 7);
 x_104 = x_5;
} else {
 lean_dec_ref(x_5);
 x_104 = lean_box(0);
}
if (lean_is_scalar(x_104)) {
 x_105 = lean_alloc_ctor(0, 8, 0);
} else {
 x_105 = x_104;
}
lean_ctor_set(x_105, 0, x_100);
lean_ctor_set(x_105, 1, x_13);
lean_ctor_set(x_105, 2, x_14);
lean_ctor_set(x_105, 3, x_101);
lean_ctor_set(x_105, 4, x_15);
lean_ctor_set(x_105, 5, x_16);
lean_ctor_set(x_105, 6, x_102);
lean_ctor_set(x_105, 7, x_103);
x_106 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_106, 0, x_84);
lean_ctor_set(x_106, 1, x_105);
lean_ctor_set(x_106, 2, x_85);
return x_106;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormedField_induced___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; 
lean_inc_ref(x_2);
x_5 = lp_mathlib_Field_toDivisionRing___redArg(x_2);
x_6 = lp_mathlib_NormedField_toNormedDivisionRing___redArg(x_3);
x_7 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_5, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_5, 4);
lean_inc(x_10);
x_11 = lean_ctor_get(x_5, 5);
lean_inc(x_11);
lean_dec_ref(x_5);
lean_inc_ref(x_7);
x_12 = lp_mathlib_Ring_toAddCommGroup___redArg(x_7);
x_13 = lp_mathlib_NormedDivisionRing_toNormedRing___redArg(x_6);
x_14 = lp_mathlib_NormedRing_toNonUnitalNormedRing___redArg(x_13);
x_15 = lp_mathlib_NonUnitalNormedRing_toNormedAddCommGroup___redArg(x_14);
x_16 = lp_mathlib_NormedAddCommGroup_toNormedAddGroup___redArg(x_15);
x_17 = lp_mathlib_NormedAddCommGroup_induced___redArg(x_1, x_12, x_16, x_4);
x_18 = lean_ctor_get(x_17, 1);
lean_inc_ref(x_18);
x_19 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_19);
x_20 = lean_ctor_get(x_19, 0);
lean_inc_ref(x_20);
x_21 = !lean_is_exclusive(x_17);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; 
x_22 = lean_ctor_get(x_17, 1);
lean_dec(x_22);
x_23 = lean_ctor_get(x_18, 0);
lean_inc_ref(x_23);
x_24 = lean_ctor_get(x_18, 1);
lean_inc(x_24);
x_25 = lean_ctor_get(x_18, 2);
lean_inc(x_25);
lean_dec_ref(x_18);
x_26 = !lean_is_exclusive(x_7);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; uint8_t x_30; 
x_27 = lean_ctor_get(x_7, 2);
lean_dec(x_27);
x_28 = lean_ctor_get(x_7, 1);
lean_dec(x_28);
x_29 = lean_ctor_get(x_7, 0);
lean_dec(x_29);
x_30 = !lean_is_exclusive(x_19);
if (x_30 == 0)
{
lean_object* x_31; uint8_t x_32; 
x_31 = lean_ctor_get(x_19, 0);
lean_dec(x_31);
x_32 = !lean_is_exclusive(x_20);
if (x_32 == 0)
{
lean_object* x_33; uint8_t x_34; 
x_33 = lean_ctor_get(x_20, 0);
lean_dec(x_33);
lean_ctor_set(x_20, 0, x_23);
lean_ctor_set(x_7, 2, x_25);
lean_ctor_set(x_7, 1, x_24);
x_34 = !lean_is_exclusive(x_2);
if (x_34 == 0)
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_35 = lean_ctor_get(x_2, 5);
lean_dec(x_35);
x_36 = lean_ctor_get(x_2, 4);
lean_dec(x_36);
x_37 = lean_ctor_get(x_2, 2);
lean_dec(x_37);
x_38 = lean_ctor_get(x_2, 1);
lean_dec(x_38);
x_39 = lean_ctor_get(x_2, 0);
lean_dec(x_39);
lean_ctor_set(x_2, 5, x_11);
lean_ctor_set(x_2, 4, x_10);
lean_ctor_set(x_2, 2, x_9);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_7);
lean_ctor_set(x_17, 1, x_2);
return x_17;
}
else
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; 
x_40 = lean_ctor_get(x_2, 3);
x_41 = lean_ctor_get(x_2, 6);
x_42 = lean_ctor_get(x_2, 7);
lean_inc(x_42);
lean_inc(x_41);
lean_inc(x_40);
lean_dec(x_2);
x_43 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_43, 0, x_7);
lean_ctor_set(x_43, 1, x_8);
lean_ctor_set(x_43, 2, x_9);
lean_ctor_set(x_43, 3, x_40);
lean_ctor_set(x_43, 4, x_10);
lean_ctor_set(x_43, 5, x_11);
lean_ctor_set(x_43, 6, x_41);
lean_ctor_set(x_43, 7, x_42);
lean_ctor_set(x_17, 1, x_43);
return x_17;
}
}
else
{
lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; 
x_44 = lean_ctor_get(x_20, 1);
lean_inc(x_44);
lean_dec(x_20);
x_45 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_45, 0, x_23);
lean_ctor_set(x_45, 1, x_44);
lean_ctor_set(x_19, 0, x_45);
lean_ctor_set(x_7, 2, x_25);
lean_ctor_set(x_7, 1, x_24);
x_46 = lean_ctor_get(x_2, 3);
lean_inc(x_46);
x_47 = lean_ctor_get(x_2, 6);
lean_inc(x_47);
x_48 = lean_ctor_get(x_2, 7);
lean_inc(x_48);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 lean_ctor_release(x_2, 3);
 lean_ctor_release(x_2, 4);
 lean_ctor_release(x_2, 5);
 lean_ctor_release(x_2, 6);
 lean_ctor_release(x_2, 7);
 x_49 = x_2;
} else {
 lean_dec_ref(x_2);
 x_49 = lean_box(0);
}
if (lean_is_scalar(x_49)) {
 x_50 = lean_alloc_ctor(0, 8, 0);
} else {
 x_50 = x_49;
}
lean_ctor_set(x_50, 0, x_7);
lean_ctor_set(x_50, 1, x_8);
lean_ctor_set(x_50, 2, x_9);
lean_ctor_set(x_50, 3, x_46);
lean_ctor_set(x_50, 4, x_10);
lean_ctor_set(x_50, 5, x_11);
lean_ctor_set(x_50, 6, x_47);
lean_ctor_set(x_50, 7, x_48);
lean_ctor_set(x_17, 1, x_50);
return x_17;
}
}
else
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; 
x_51 = lean_ctor_get(x_19, 1);
x_52 = lean_ctor_get(x_19, 2);
x_53 = lean_ctor_get(x_19, 3);
lean_inc(x_53);
lean_inc(x_52);
lean_inc(x_51);
lean_dec(x_19);
x_54 = lean_ctor_get(x_20, 1);
lean_inc(x_54);
if (lean_is_exclusive(x_20)) {
 lean_ctor_release(x_20, 0);
 lean_ctor_release(x_20, 1);
 x_55 = x_20;
} else {
 lean_dec_ref(x_20);
 x_55 = lean_box(0);
}
if (lean_is_scalar(x_55)) {
 x_56 = lean_alloc_ctor(0, 2, 0);
} else {
 x_56 = x_55;
}
lean_ctor_set(x_56, 0, x_23);
lean_ctor_set(x_56, 1, x_54);
x_57 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_57, 0, x_56);
lean_ctor_set(x_57, 1, x_51);
lean_ctor_set(x_57, 2, x_52);
lean_ctor_set(x_57, 3, x_53);
lean_ctor_set(x_7, 2, x_25);
lean_ctor_set(x_7, 1, x_24);
lean_ctor_set(x_7, 0, x_57);
x_58 = lean_ctor_get(x_2, 3);
lean_inc(x_58);
x_59 = lean_ctor_get(x_2, 6);
lean_inc(x_59);
x_60 = lean_ctor_get(x_2, 7);
lean_inc(x_60);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 lean_ctor_release(x_2, 3);
 lean_ctor_release(x_2, 4);
 lean_ctor_release(x_2, 5);
 lean_ctor_release(x_2, 6);
 lean_ctor_release(x_2, 7);
 x_61 = x_2;
} else {
 lean_dec_ref(x_2);
 x_61 = lean_box(0);
}
if (lean_is_scalar(x_61)) {
 x_62 = lean_alloc_ctor(0, 8, 0);
} else {
 x_62 = x_61;
}
lean_ctor_set(x_62, 0, x_7);
lean_ctor_set(x_62, 1, x_8);
lean_ctor_set(x_62, 2, x_9);
lean_ctor_set(x_62, 3, x_58);
lean_ctor_set(x_62, 4, x_10);
lean_ctor_set(x_62, 5, x_11);
lean_ctor_set(x_62, 6, x_59);
lean_ctor_set(x_62, 7, x_60);
lean_ctor_set(x_17, 1, x_62);
return x_17;
}
}
else
{
lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; 
x_63 = lean_ctor_get(x_7, 3);
x_64 = lean_ctor_get(x_7, 4);
lean_inc(x_64);
lean_inc(x_63);
lean_dec(x_7);
x_65 = lean_ctor_get(x_19, 1);
lean_inc(x_65);
x_66 = lean_ctor_get(x_19, 2);
lean_inc(x_66);
x_67 = lean_ctor_get(x_19, 3);
lean_inc(x_67);
if (lean_is_exclusive(x_19)) {
 lean_ctor_release(x_19, 0);
 lean_ctor_release(x_19, 1);
 lean_ctor_release(x_19, 2);
 lean_ctor_release(x_19, 3);
 x_68 = x_19;
} else {
 lean_dec_ref(x_19);
 x_68 = lean_box(0);
}
x_69 = lean_ctor_get(x_20, 1);
lean_inc(x_69);
if (lean_is_exclusive(x_20)) {
 lean_ctor_release(x_20, 0);
 lean_ctor_release(x_20, 1);
 x_70 = x_20;
} else {
 lean_dec_ref(x_20);
 x_70 = lean_box(0);
}
if (lean_is_scalar(x_70)) {
 x_71 = lean_alloc_ctor(0, 2, 0);
} else {
 x_71 = x_70;
}
lean_ctor_set(x_71, 0, x_23);
lean_ctor_set(x_71, 1, x_69);
if (lean_is_scalar(x_68)) {
 x_72 = lean_alloc_ctor(0, 4, 0);
} else {
 x_72 = x_68;
}
lean_ctor_set(x_72, 0, x_71);
lean_ctor_set(x_72, 1, x_65);
lean_ctor_set(x_72, 2, x_66);
lean_ctor_set(x_72, 3, x_67);
x_73 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_73, 0, x_72);
lean_ctor_set(x_73, 1, x_24);
lean_ctor_set(x_73, 2, x_25);
lean_ctor_set(x_73, 3, x_63);
lean_ctor_set(x_73, 4, x_64);
x_74 = lean_ctor_get(x_2, 3);
lean_inc(x_74);
x_75 = lean_ctor_get(x_2, 6);
lean_inc(x_75);
x_76 = lean_ctor_get(x_2, 7);
lean_inc(x_76);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 lean_ctor_release(x_2, 3);
 lean_ctor_release(x_2, 4);
 lean_ctor_release(x_2, 5);
 lean_ctor_release(x_2, 6);
 lean_ctor_release(x_2, 7);
 x_77 = x_2;
} else {
 lean_dec_ref(x_2);
 x_77 = lean_box(0);
}
if (lean_is_scalar(x_77)) {
 x_78 = lean_alloc_ctor(0, 8, 0);
} else {
 x_78 = x_77;
}
lean_ctor_set(x_78, 0, x_73);
lean_ctor_set(x_78, 1, x_8);
lean_ctor_set(x_78, 2, x_9);
lean_ctor_set(x_78, 3, x_74);
lean_ctor_set(x_78, 4, x_10);
lean_ctor_set(x_78, 5, x_11);
lean_ctor_set(x_78, 6, x_75);
lean_ctor_set(x_78, 7, x_76);
lean_ctor_set(x_17, 1, x_78);
return x_17;
}
}
else
{
lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; 
x_79 = lean_ctor_get(x_17, 0);
x_80 = lean_ctor_get(x_17, 2);
lean_inc(x_80);
lean_inc(x_79);
lean_dec(x_17);
x_81 = lean_ctor_get(x_18, 0);
lean_inc_ref(x_81);
x_82 = lean_ctor_get(x_18, 1);
lean_inc(x_82);
x_83 = lean_ctor_get(x_18, 2);
lean_inc(x_83);
lean_dec_ref(x_18);
x_84 = lean_ctor_get(x_7, 3);
lean_inc(x_84);
x_85 = lean_ctor_get(x_7, 4);
lean_inc(x_85);
if (lean_is_exclusive(x_7)) {
 lean_ctor_release(x_7, 0);
 lean_ctor_release(x_7, 1);
 lean_ctor_release(x_7, 2);
 lean_ctor_release(x_7, 3);
 lean_ctor_release(x_7, 4);
 x_86 = x_7;
} else {
 lean_dec_ref(x_7);
 x_86 = lean_box(0);
}
x_87 = lean_ctor_get(x_19, 1);
lean_inc(x_87);
x_88 = lean_ctor_get(x_19, 2);
lean_inc(x_88);
x_89 = lean_ctor_get(x_19, 3);
lean_inc(x_89);
if (lean_is_exclusive(x_19)) {
 lean_ctor_release(x_19, 0);
 lean_ctor_release(x_19, 1);
 lean_ctor_release(x_19, 2);
 lean_ctor_release(x_19, 3);
 x_90 = x_19;
} else {
 lean_dec_ref(x_19);
 x_90 = lean_box(0);
}
x_91 = lean_ctor_get(x_20, 1);
lean_inc(x_91);
if (lean_is_exclusive(x_20)) {
 lean_ctor_release(x_20, 0);
 lean_ctor_release(x_20, 1);
 x_92 = x_20;
} else {
 lean_dec_ref(x_20);
 x_92 = lean_box(0);
}
if (lean_is_scalar(x_92)) {
 x_93 = lean_alloc_ctor(0, 2, 0);
} else {
 x_93 = x_92;
}
lean_ctor_set(x_93, 0, x_81);
lean_ctor_set(x_93, 1, x_91);
if (lean_is_scalar(x_90)) {
 x_94 = lean_alloc_ctor(0, 4, 0);
} else {
 x_94 = x_90;
}
lean_ctor_set(x_94, 0, x_93);
lean_ctor_set(x_94, 1, x_87);
lean_ctor_set(x_94, 2, x_88);
lean_ctor_set(x_94, 3, x_89);
if (lean_is_scalar(x_86)) {
 x_95 = lean_alloc_ctor(0, 5, 0);
} else {
 x_95 = x_86;
}
lean_ctor_set(x_95, 0, x_94);
lean_ctor_set(x_95, 1, x_82);
lean_ctor_set(x_95, 2, x_83);
lean_ctor_set(x_95, 3, x_84);
lean_ctor_set(x_95, 4, x_85);
x_96 = lean_ctor_get(x_2, 3);
lean_inc(x_96);
x_97 = lean_ctor_get(x_2, 6);
lean_inc(x_97);
x_98 = lean_ctor_get(x_2, 7);
lean_inc(x_98);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 lean_ctor_release(x_2, 3);
 lean_ctor_release(x_2, 4);
 lean_ctor_release(x_2, 5);
 lean_ctor_release(x_2, 6);
 lean_ctor_release(x_2, 7);
 x_99 = x_2;
} else {
 lean_dec_ref(x_2);
 x_99 = lean_box(0);
}
if (lean_is_scalar(x_99)) {
 x_100 = lean_alloc_ctor(0, 8, 0);
} else {
 x_100 = x_99;
}
lean_ctor_set(x_100, 0, x_95);
lean_ctor_set(x_100, 1, x_8);
lean_ctor_set(x_100, 2, x_9);
lean_ctor_set(x_100, 3, x_96);
lean_ctor_set(x_100, 4, x_10);
lean_ctor_set(x_100, 5, x_11);
lean_ctor_set(x_100, 6, x_97);
lean_ctor_set(x_100, 7, x_98);
x_101 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_101, 0, x_79);
lean_ctor_set(x_101, 1, x_100);
lean_ctor_set(x_101, 2, x_80);
return x_101;
}
}
}
static lean_object* _init_lp_mathlib_SubfieldClass_toNormedField___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_instFunLike___lam__0), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_SubfieldClass_toNormedField___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_SubringClass_subtype___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 3);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_NormedField_toNormedDivisionRing___redArg(x_1);
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; uint8_t x_31; 
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
lean_dec(x_5);
x_6 = lean_ctor_get(x_1, 0);
lean_dec(x_6);
lean_inc_ref(x_4);
x_7 = lp_mathlib_SubfieldClass_toField___redArg(x_4);
x_8 = lp_mathlib_Field_toDivisionRing___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_8, 1);
lean_inc(x_10);
x_11 = lean_ctor_get(x_8, 2);
lean_inc(x_11);
x_12 = lean_ctor_get(x_8, 4);
lean_inc(x_12);
x_13 = lean_ctor_get(x_8, 5);
lean_inc(x_13);
lean_dec_ref(x_8);
x_14 = lp_mathlib_SubfieldClass_toNormedField___redArg___closed__0;
x_15 = lp_mathlib_SubfieldClass_toNormedField___redArg___closed__1;
lean_inc_ref(x_9);
x_16 = lp_mathlib_Ring_toAddCommGroup___redArg(x_9);
x_17 = lp_mathlib_NormedDivisionRing_toNormedRing___redArg(x_2);
x_18 = lp_mathlib_NormedRing_toNonUnitalNormedRing___redArg(x_17);
x_19 = lp_mathlib_NonUnitalNormedRing_toNormedAddCommGroup___redArg(x_18);
x_20 = lp_mathlib_NormedAddCommGroup_toNormedAddGroup___redArg(x_19);
x_21 = lp_mathlib_NormedAddCommGroup_induced___redArg(x_14, x_16, x_20, x_15);
x_22 = lean_ctor_get(x_21, 1);
lean_inc_ref(x_22);
x_23 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_23);
x_24 = lean_ctor_get(x_23, 0);
lean_inc_ref(x_24);
x_25 = lean_ctor_get(x_21, 0);
lean_inc(x_25);
x_26 = lean_ctor_get(x_21, 2);
lean_inc_ref(x_26);
lean_dec_ref(x_21);
x_27 = lean_ctor_get(x_22, 0);
lean_inc_ref(x_27);
x_28 = lean_ctor_get(x_22, 1);
lean_inc(x_28);
x_29 = lean_ctor_get(x_22, 2);
lean_inc(x_29);
lean_dec_ref(x_22);
x_30 = lean_ctor_get(x_9, 4);
lean_inc(x_30);
lean_dec_ref(x_9);
x_31 = !lean_is_exclusive(x_23);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; uint8_t x_34; 
x_32 = lean_ctor_get(x_23, 3);
lean_dec(x_32);
x_33 = lean_ctor_get(x_23, 0);
lean_dec(x_33);
x_34 = !lean_is_exclusive(x_24);
if (x_34 == 0)
{
lean_object* x_35; uint8_t x_36; 
x_35 = lean_ctor_get(x_24, 0);
lean_dec(x_35);
lean_ctor_set(x_24, 0, x_27);
x_36 = !lean_is_exclusive(x_4);
if (x_36 == 0)
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; uint8_t x_45; 
x_37 = lean_ctor_get(x_4, 0);
x_38 = lean_ctor_get(x_4, 3);
x_39 = lean_ctor_get(x_4, 6);
x_40 = lean_ctor_get(x_4, 7);
x_41 = lean_ctor_get(x_4, 5);
lean_dec(x_41);
x_42 = lean_ctor_get(x_4, 4);
lean_dec(x_42);
x_43 = lean_ctor_get(x_4, 2);
lean_dec(x_43);
x_44 = lean_ctor_get(x_4, 1);
lean_dec(x_44);
x_45 = !lean_is_exclusive(x_37);
if (x_45 == 0)
{
lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; 
x_46 = lean_ctor_get(x_37, 0);
x_47 = lean_ctor_get(x_37, 3);
x_48 = lean_ctor_get(x_37, 4);
lean_dec(x_48);
x_49 = lean_ctor_get(x_37, 2);
lean_dec(x_49);
x_50 = lean_ctor_get(x_37, 1);
lean_dec(x_50);
x_51 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__0), 3, 1);
lean_closure_set(x_51, 0, x_38);
x_52 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__1), 3, 1);
lean_closure_set(x_52, 0, x_39);
x_53 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__2), 3, 1);
lean_closure_set(x_53, 0, x_40);
x_54 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__3), 3, 1);
lean_closure_set(x_54, 0, x_47);
x_55 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__4), 3, 1);
lean_closure_set(x_55, 0, x_46);
lean_ctor_set(x_23, 3, x_55);
lean_ctor_set(x_37, 4, x_30);
lean_ctor_set(x_37, 3, x_54);
lean_ctor_set(x_37, 2, x_29);
lean_ctor_set(x_37, 1, x_28);
lean_ctor_set(x_37, 0, x_23);
lean_ctor_set(x_4, 7, x_53);
lean_ctor_set(x_4, 6, x_52);
lean_ctor_set(x_4, 5, x_13);
lean_ctor_set(x_4, 4, x_12);
lean_ctor_set(x_4, 3, x_51);
lean_ctor_set(x_4, 2, x_11);
lean_ctor_set(x_4, 1, x_10);
lean_ctor_set(x_1, 2, x_26);
lean_ctor_set(x_1, 0, x_25);
return x_1;
}
else
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; 
x_56 = lean_ctor_get(x_37, 0);
x_57 = lean_ctor_get(x_37, 3);
lean_inc(x_57);
lean_inc(x_56);
lean_dec(x_37);
x_58 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__0), 3, 1);
lean_closure_set(x_58, 0, x_38);
x_59 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__1), 3, 1);
lean_closure_set(x_59, 0, x_39);
x_60 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__2), 3, 1);
lean_closure_set(x_60, 0, x_40);
x_61 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__3), 3, 1);
lean_closure_set(x_61, 0, x_57);
x_62 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__4), 3, 1);
lean_closure_set(x_62, 0, x_56);
lean_ctor_set(x_23, 3, x_62);
x_63 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_63, 0, x_23);
lean_ctor_set(x_63, 1, x_28);
lean_ctor_set(x_63, 2, x_29);
lean_ctor_set(x_63, 3, x_61);
lean_ctor_set(x_63, 4, x_30);
lean_ctor_set(x_4, 7, x_60);
lean_ctor_set(x_4, 6, x_59);
lean_ctor_set(x_4, 5, x_13);
lean_ctor_set(x_4, 4, x_12);
lean_ctor_set(x_4, 3, x_58);
lean_ctor_set(x_4, 2, x_11);
lean_ctor_set(x_4, 1, x_10);
lean_ctor_set(x_4, 0, x_63);
lean_ctor_set(x_1, 2, x_26);
lean_ctor_set(x_1, 0, x_25);
return x_1;
}
}
else
{
lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; 
x_64 = lean_ctor_get(x_4, 0);
x_65 = lean_ctor_get(x_4, 3);
x_66 = lean_ctor_get(x_4, 6);
x_67 = lean_ctor_get(x_4, 7);
lean_inc(x_67);
lean_inc(x_66);
lean_inc(x_65);
lean_inc(x_64);
lean_dec(x_4);
x_68 = lean_ctor_get(x_64, 0);
lean_inc_ref(x_68);
x_69 = lean_ctor_get(x_64, 3);
lean_inc(x_69);
if (lean_is_exclusive(x_64)) {
 lean_ctor_release(x_64, 0);
 lean_ctor_release(x_64, 1);
 lean_ctor_release(x_64, 2);
 lean_ctor_release(x_64, 3);
 lean_ctor_release(x_64, 4);
 x_70 = x_64;
} else {
 lean_dec_ref(x_64);
 x_70 = lean_box(0);
}
x_71 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__0), 3, 1);
lean_closure_set(x_71, 0, x_65);
x_72 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__1), 3, 1);
lean_closure_set(x_72, 0, x_66);
x_73 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__2), 3, 1);
lean_closure_set(x_73, 0, x_67);
x_74 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__3), 3, 1);
lean_closure_set(x_74, 0, x_69);
x_75 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__4), 3, 1);
lean_closure_set(x_75, 0, x_68);
lean_ctor_set(x_23, 3, x_75);
if (lean_is_scalar(x_70)) {
 x_76 = lean_alloc_ctor(0, 5, 0);
} else {
 x_76 = x_70;
}
lean_ctor_set(x_76, 0, x_23);
lean_ctor_set(x_76, 1, x_28);
lean_ctor_set(x_76, 2, x_29);
lean_ctor_set(x_76, 3, x_74);
lean_ctor_set(x_76, 4, x_30);
x_77 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_77, 0, x_76);
lean_ctor_set(x_77, 1, x_10);
lean_ctor_set(x_77, 2, x_11);
lean_ctor_set(x_77, 3, x_71);
lean_ctor_set(x_77, 4, x_12);
lean_ctor_set(x_77, 5, x_13);
lean_ctor_set(x_77, 6, x_72);
lean_ctor_set(x_77, 7, x_73);
lean_ctor_set(x_1, 2, x_26);
lean_ctor_set(x_1, 1, x_77);
lean_ctor_set(x_1, 0, x_25);
return x_1;
}
}
else
{
lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; 
x_78 = lean_ctor_get(x_24, 1);
lean_inc(x_78);
lean_dec(x_24);
x_79 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_79, 0, x_27);
lean_ctor_set(x_79, 1, x_78);
x_80 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_80);
x_81 = lean_ctor_get(x_4, 3);
lean_inc(x_81);
x_82 = lean_ctor_get(x_4, 6);
lean_inc(x_82);
x_83 = lean_ctor_get(x_4, 7);
lean_inc(x_83);
if (lean_is_exclusive(x_4)) {
 lean_ctor_release(x_4, 0);
 lean_ctor_release(x_4, 1);
 lean_ctor_release(x_4, 2);
 lean_ctor_release(x_4, 3);
 lean_ctor_release(x_4, 4);
 lean_ctor_release(x_4, 5);
 lean_ctor_release(x_4, 6);
 lean_ctor_release(x_4, 7);
 x_84 = x_4;
} else {
 lean_dec_ref(x_4);
 x_84 = lean_box(0);
}
x_85 = lean_ctor_get(x_80, 0);
lean_inc_ref(x_85);
x_86 = lean_ctor_get(x_80, 3);
lean_inc(x_86);
if (lean_is_exclusive(x_80)) {
 lean_ctor_release(x_80, 0);
 lean_ctor_release(x_80, 1);
 lean_ctor_release(x_80, 2);
 lean_ctor_release(x_80, 3);
 lean_ctor_release(x_80, 4);
 x_87 = x_80;
} else {
 lean_dec_ref(x_80);
 x_87 = lean_box(0);
}
x_88 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__0), 3, 1);
lean_closure_set(x_88, 0, x_81);
x_89 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__1), 3, 1);
lean_closure_set(x_89, 0, x_82);
x_90 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__2), 3, 1);
lean_closure_set(x_90, 0, x_83);
x_91 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__3), 3, 1);
lean_closure_set(x_91, 0, x_86);
x_92 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__4), 3, 1);
lean_closure_set(x_92, 0, x_85);
lean_ctor_set(x_23, 3, x_92);
lean_ctor_set(x_23, 0, x_79);
if (lean_is_scalar(x_87)) {
 x_93 = lean_alloc_ctor(0, 5, 0);
} else {
 x_93 = x_87;
}
lean_ctor_set(x_93, 0, x_23);
lean_ctor_set(x_93, 1, x_28);
lean_ctor_set(x_93, 2, x_29);
lean_ctor_set(x_93, 3, x_91);
lean_ctor_set(x_93, 4, x_30);
if (lean_is_scalar(x_84)) {
 x_94 = lean_alloc_ctor(0, 8, 0);
} else {
 x_94 = x_84;
}
lean_ctor_set(x_94, 0, x_93);
lean_ctor_set(x_94, 1, x_10);
lean_ctor_set(x_94, 2, x_11);
lean_ctor_set(x_94, 3, x_88);
lean_ctor_set(x_94, 4, x_12);
lean_ctor_set(x_94, 5, x_13);
lean_ctor_set(x_94, 6, x_89);
lean_ctor_set(x_94, 7, x_90);
lean_ctor_set(x_1, 2, x_26);
lean_ctor_set(x_1, 1, x_94);
lean_ctor_set(x_1, 0, x_25);
return x_1;
}
}
else
{
lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; 
x_95 = lean_ctor_get(x_23, 1);
x_96 = lean_ctor_get(x_23, 2);
lean_inc(x_96);
lean_inc(x_95);
lean_dec(x_23);
x_97 = lean_ctor_get(x_24, 1);
lean_inc(x_97);
if (lean_is_exclusive(x_24)) {
 lean_ctor_release(x_24, 0);
 lean_ctor_release(x_24, 1);
 x_98 = x_24;
} else {
 lean_dec_ref(x_24);
 x_98 = lean_box(0);
}
if (lean_is_scalar(x_98)) {
 x_99 = lean_alloc_ctor(0, 2, 0);
} else {
 x_99 = x_98;
}
lean_ctor_set(x_99, 0, x_27);
lean_ctor_set(x_99, 1, x_97);
x_100 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_100);
x_101 = lean_ctor_get(x_4, 3);
lean_inc(x_101);
x_102 = lean_ctor_get(x_4, 6);
lean_inc(x_102);
x_103 = lean_ctor_get(x_4, 7);
lean_inc(x_103);
if (lean_is_exclusive(x_4)) {
 lean_ctor_release(x_4, 0);
 lean_ctor_release(x_4, 1);
 lean_ctor_release(x_4, 2);
 lean_ctor_release(x_4, 3);
 lean_ctor_release(x_4, 4);
 lean_ctor_release(x_4, 5);
 lean_ctor_release(x_4, 6);
 lean_ctor_release(x_4, 7);
 x_104 = x_4;
} else {
 lean_dec_ref(x_4);
 x_104 = lean_box(0);
}
x_105 = lean_ctor_get(x_100, 0);
lean_inc_ref(x_105);
x_106 = lean_ctor_get(x_100, 3);
lean_inc(x_106);
if (lean_is_exclusive(x_100)) {
 lean_ctor_release(x_100, 0);
 lean_ctor_release(x_100, 1);
 lean_ctor_release(x_100, 2);
 lean_ctor_release(x_100, 3);
 lean_ctor_release(x_100, 4);
 x_107 = x_100;
} else {
 lean_dec_ref(x_100);
 x_107 = lean_box(0);
}
x_108 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__0), 3, 1);
lean_closure_set(x_108, 0, x_101);
x_109 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__1), 3, 1);
lean_closure_set(x_109, 0, x_102);
x_110 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__2), 3, 1);
lean_closure_set(x_110, 0, x_103);
x_111 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__3), 3, 1);
lean_closure_set(x_111, 0, x_106);
x_112 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__4), 3, 1);
lean_closure_set(x_112, 0, x_105);
x_113 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_113, 0, x_99);
lean_ctor_set(x_113, 1, x_95);
lean_ctor_set(x_113, 2, x_96);
lean_ctor_set(x_113, 3, x_112);
if (lean_is_scalar(x_107)) {
 x_114 = lean_alloc_ctor(0, 5, 0);
} else {
 x_114 = x_107;
}
lean_ctor_set(x_114, 0, x_113);
lean_ctor_set(x_114, 1, x_28);
lean_ctor_set(x_114, 2, x_29);
lean_ctor_set(x_114, 3, x_111);
lean_ctor_set(x_114, 4, x_30);
if (lean_is_scalar(x_104)) {
 x_115 = lean_alloc_ctor(0, 8, 0);
} else {
 x_115 = x_104;
}
lean_ctor_set(x_115, 0, x_114);
lean_ctor_set(x_115, 1, x_10);
lean_ctor_set(x_115, 2, x_11);
lean_ctor_set(x_115, 3, x_108);
lean_ctor_set(x_115, 4, x_12);
lean_ctor_set(x_115, 5, x_13);
lean_ctor_set(x_115, 6, x_109);
lean_ctor_set(x_115, 7, x_110);
lean_ctor_set(x_1, 2, x_26);
lean_ctor_set(x_1, 1, x_115);
lean_ctor_set(x_1, 0, x_25);
return x_1;
}
}
else
{
lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; 
x_116 = lean_ctor_get(x_1, 1);
lean_inc(x_116);
lean_dec(x_1);
lean_inc_ref(x_116);
x_117 = lp_mathlib_SubfieldClass_toField___redArg(x_116);
x_118 = lp_mathlib_Field_toDivisionRing___redArg(x_117);
x_119 = lean_ctor_get(x_118, 0);
lean_inc_ref(x_119);
x_120 = lean_ctor_get(x_118, 1);
lean_inc(x_120);
x_121 = lean_ctor_get(x_118, 2);
lean_inc(x_121);
x_122 = lean_ctor_get(x_118, 4);
lean_inc(x_122);
x_123 = lean_ctor_get(x_118, 5);
lean_inc(x_123);
lean_dec_ref(x_118);
x_124 = lp_mathlib_SubfieldClass_toNormedField___redArg___closed__0;
x_125 = lp_mathlib_SubfieldClass_toNormedField___redArg___closed__1;
lean_inc_ref(x_119);
x_126 = lp_mathlib_Ring_toAddCommGroup___redArg(x_119);
x_127 = lp_mathlib_NormedDivisionRing_toNormedRing___redArg(x_2);
x_128 = lp_mathlib_NormedRing_toNonUnitalNormedRing___redArg(x_127);
x_129 = lp_mathlib_NonUnitalNormedRing_toNormedAddCommGroup___redArg(x_128);
x_130 = lp_mathlib_NormedAddCommGroup_toNormedAddGroup___redArg(x_129);
x_131 = lp_mathlib_NormedAddCommGroup_induced___redArg(x_124, x_126, x_130, x_125);
x_132 = lean_ctor_get(x_131, 1);
lean_inc_ref(x_132);
x_133 = lean_ctor_get(x_119, 0);
lean_inc_ref(x_133);
x_134 = lean_ctor_get(x_133, 0);
lean_inc_ref(x_134);
x_135 = lean_ctor_get(x_131, 0);
lean_inc(x_135);
x_136 = lean_ctor_get(x_131, 2);
lean_inc_ref(x_136);
lean_dec_ref(x_131);
x_137 = lean_ctor_get(x_132, 0);
lean_inc_ref(x_137);
x_138 = lean_ctor_get(x_132, 1);
lean_inc(x_138);
x_139 = lean_ctor_get(x_132, 2);
lean_inc(x_139);
lean_dec_ref(x_132);
x_140 = lean_ctor_get(x_119, 4);
lean_inc(x_140);
lean_dec_ref(x_119);
x_141 = lean_ctor_get(x_133, 1);
lean_inc(x_141);
x_142 = lean_ctor_get(x_133, 2);
lean_inc(x_142);
if (lean_is_exclusive(x_133)) {
 lean_ctor_release(x_133, 0);
 lean_ctor_release(x_133, 1);
 lean_ctor_release(x_133, 2);
 lean_ctor_release(x_133, 3);
 x_143 = x_133;
} else {
 lean_dec_ref(x_133);
 x_143 = lean_box(0);
}
x_144 = lean_ctor_get(x_134, 1);
lean_inc(x_144);
if (lean_is_exclusive(x_134)) {
 lean_ctor_release(x_134, 0);
 lean_ctor_release(x_134, 1);
 x_145 = x_134;
} else {
 lean_dec_ref(x_134);
 x_145 = lean_box(0);
}
if (lean_is_scalar(x_145)) {
 x_146 = lean_alloc_ctor(0, 2, 0);
} else {
 x_146 = x_145;
}
lean_ctor_set(x_146, 0, x_137);
lean_ctor_set(x_146, 1, x_144);
x_147 = lean_ctor_get(x_116, 0);
lean_inc_ref(x_147);
x_148 = lean_ctor_get(x_116, 3);
lean_inc(x_148);
x_149 = lean_ctor_get(x_116, 6);
lean_inc(x_149);
x_150 = lean_ctor_get(x_116, 7);
lean_inc(x_150);
if (lean_is_exclusive(x_116)) {
 lean_ctor_release(x_116, 0);
 lean_ctor_release(x_116, 1);
 lean_ctor_release(x_116, 2);
 lean_ctor_release(x_116, 3);
 lean_ctor_release(x_116, 4);
 lean_ctor_release(x_116, 5);
 lean_ctor_release(x_116, 6);
 lean_ctor_release(x_116, 7);
 x_151 = x_116;
} else {
 lean_dec_ref(x_116);
 x_151 = lean_box(0);
}
x_152 = lean_ctor_get(x_147, 0);
lean_inc_ref(x_152);
x_153 = lean_ctor_get(x_147, 3);
lean_inc(x_153);
if (lean_is_exclusive(x_147)) {
 lean_ctor_release(x_147, 0);
 lean_ctor_release(x_147, 1);
 lean_ctor_release(x_147, 2);
 lean_ctor_release(x_147, 3);
 lean_ctor_release(x_147, 4);
 x_154 = x_147;
} else {
 lean_dec_ref(x_147);
 x_154 = lean_box(0);
}
x_155 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__0), 3, 1);
lean_closure_set(x_155, 0, x_148);
x_156 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__1), 3, 1);
lean_closure_set(x_156, 0, x_149);
x_157 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__2), 3, 1);
lean_closure_set(x_157, 0, x_150);
x_158 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__3), 3, 1);
lean_closure_set(x_158, 0, x_153);
x_159 = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toNormedField___redArg___lam__4), 3, 1);
lean_closure_set(x_159, 0, x_152);
if (lean_is_scalar(x_143)) {
 x_160 = lean_alloc_ctor(0, 4, 0);
} else {
 x_160 = x_143;
}
lean_ctor_set(x_160, 0, x_146);
lean_ctor_set(x_160, 1, x_141);
lean_ctor_set(x_160, 2, x_142);
lean_ctor_set(x_160, 3, x_159);
if (lean_is_scalar(x_154)) {
 x_161 = lean_alloc_ctor(0, 5, 0);
} else {
 x_161 = x_154;
}
lean_ctor_set(x_161, 0, x_160);
lean_ctor_set(x_161, 1, x_138);
lean_ctor_set(x_161, 2, x_139);
lean_ctor_set(x_161, 3, x_158);
lean_ctor_set(x_161, 4, x_140);
if (lean_is_scalar(x_151)) {
 x_162 = lean_alloc_ctor(0, 8, 0);
} else {
 x_162 = x_151;
}
lean_ctor_set(x_162, 0, x_161);
lean_ctor_set(x_162, 1, x_120);
lean_ctor_set(x_162, 2, x_121);
lean_ctor_set(x_162, 3, x_155);
lean_ctor_set(x_162, 4, x_122);
lean_ctor_set(x_162, 5, x_123);
lean_ctor_set(x_162, 6, x_156);
lean_ctor_set(x_162, 7, x_157);
x_163 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_163, 0, x_135);
lean_ctor_set(x_163, 1, x_162);
lean_ctor_set(x_163, 2, x_136);
return x_163;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_SubfieldClass_toNormedField___redArg(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toNormedField___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_SubfieldClass_toNormedField(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Pointwise_Interval(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Ring_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Field_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Pointwise_Interval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_SubfieldClass_toNormedField___redArg___closed__0 = _init_lp_mathlib_SubfieldClass_toNormedField___redArg___closed__0();
lean_mark_persistent(lp_mathlib_SubfieldClass_toNormedField___redArg___closed__0);
lp_mathlib_SubfieldClass_toNormedField___redArg___closed__1 = _init_lp_mathlib_SubfieldClass_toNormedField___redArg___closed__1();
lean_mark_persistent(lp_mathlib_SubfieldClass_toNormedField___redArg___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
