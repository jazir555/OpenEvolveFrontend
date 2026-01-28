// Lean compiler output
// Module: Mathlib.CategoryTheory.MorphismProperty.Basic
// Imports: public import Init public import Mathlib.CategoryTheory.Comma.Arrow public import Mathlib.Order.CompleteBooleanAlgebra
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
extern lean_object* lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_homFamily___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_homFamily(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_homFamily___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_homFamily___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instInhabited___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__1(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_3 = lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra;
x_4 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__0___boxed), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__1___boxed), 2, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___lam__1___boxed), 2, 1);
lean_closure_set(x_6, 0, x_5);
x_7 = lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg(x_6);
x_8 = lean_box(0);
x_9 = !lean_is_exclusive(x_7);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_10 = lean_ctor_get(x_7, 0);
x_11 = lean_ctor_get(x_7, 3);
lean_dec(x_11);
x_12 = lean_ctor_get(x_7, 2);
lean_dec(x_12);
x_13 = lean_ctor_get(x_7, 1);
lean_dec(x_13);
x_14 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_14);
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; uint8_t x_18; 
x_16 = lean_ctor_get(x_14, 0);
x_17 = lean_ctor_get(x_14, 1);
lean_dec(x_17);
x_18 = !lean_is_exclusive(x_16);
if (x_18 == 0)
{
lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_19 = lean_ctor_get(x_16, 0);
x_20 = lean_ctor_get(x_16, 1);
lean_dec(x_20);
x_21 = !lean_is_exclusive(x_10);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; 
x_22 = lean_ctor_get(x_10, 2);
lean_dec(x_22);
x_23 = lean_ctor_get(x_10, 1);
lean_dec(x_23);
x_24 = lean_ctor_get(x_10, 0);
lean_dec(x_24);
x_25 = !lean_is_exclusive(x_19);
if (x_25 == 0)
{
lean_object* x_26; 
x_26 = lean_ctor_get(x_19, 0);
lean_dec(x_26);
lean_ctor_set(x_19, 0, x_8);
lean_ctor_set(x_16, 1, lean_box(0));
lean_ctor_set(x_14, 1, lean_box(0));
lean_ctor_set(x_10, 2, lean_box(0));
lean_ctor_set(x_10, 1, lean_box(0));
lean_ctor_set(x_7, 3, lean_box(0));
lean_ctor_set(x_7, 2, lean_box(0));
lean_ctor_set(x_7, 1, lean_box(0));
return x_7;
}
else
{
lean_object* x_27; lean_object* x_28; 
x_27 = lean_ctor_get(x_19, 1);
lean_inc(x_27);
lean_dec(x_19);
x_28 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_28, 0, x_8);
lean_ctor_set(x_28, 1, x_27);
lean_ctor_set(x_16, 1, lean_box(0));
lean_ctor_set(x_16, 0, x_28);
lean_ctor_set(x_14, 1, lean_box(0));
lean_ctor_set(x_10, 2, lean_box(0));
lean_ctor_set(x_10, 1, lean_box(0));
lean_ctor_set(x_7, 3, lean_box(0));
lean_ctor_set(x_7, 2, lean_box(0));
lean_ctor_set(x_7, 1, lean_box(0));
return x_7;
}
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_29 = lean_ctor_get(x_10, 3);
lean_inc(x_29);
lean_dec(x_10);
x_30 = lean_ctor_get(x_19, 1);
lean_inc(x_30);
if (lean_is_exclusive(x_19)) {
 lean_ctor_release(x_19, 0);
 lean_ctor_release(x_19, 1);
 x_31 = x_19;
} else {
 lean_dec_ref(x_19);
 x_31 = lean_box(0);
}
if (lean_is_scalar(x_31)) {
 x_32 = lean_alloc_ctor(0, 2, 0);
} else {
 x_32 = x_31;
}
lean_ctor_set(x_32, 0, x_8);
lean_ctor_set(x_32, 1, x_30);
lean_ctor_set(x_16, 1, lean_box(0));
lean_ctor_set(x_16, 0, x_32);
lean_ctor_set(x_14, 1, lean_box(0));
x_33 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_33, 0, x_14);
lean_ctor_set(x_33, 1, lean_box(0));
lean_ctor_set(x_33, 2, lean_box(0));
lean_ctor_set(x_33, 3, x_29);
lean_ctor_set(x_7, 3, lean_box(0));
lean_ctor_set(x_7, 2, lean_box(0));
lean_ctor_set(x_7, 1, lean_box(0));
lean_ctor_set(x_7, 0, x_33);
return x_7;
}
}
else
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_34 = lean_ctor_get(x_16, 0);
lean_inc(x_34);
lean_dec(x_16);
x_35 = lean_ctor_get(x_10, 3);
lean_inc_ref(x_35);
if (lean_is_exclusive(x_10)) {
 lean_ctor_release(x_10, 0);
 lean_ctor_release(x_10, 1);
 lean_ctor_release(x_10, 2);
 lean_ctor_release(x_10, 3);
 x_36 = x_10;
} else {
 lean_dec_ref(x_10);
 x_36 = lean_box(0);
}
x_37 = lean_ctor_get(x_34, 1);
lean_inc(x_37);
if (lean_is_exclusive(x_34)) {
 lean_ctor_release(x_34, 0);
 lean_ctor_release(x_34, 1);
 x_38 = x_34;
} else {
 lean_dec_ref(x_34);
 x_38 = lean_box(0);
}
if (lean_is_scalar(x_38)) {
 x_39 = lean_alloc_ctor(0, 2, 0);
} else {
 x_39 = x_38;
}
lean_ctor_set(x_39, 0, x_8);
lean_ctor_set(x_39, 1, x_37);
x_40 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_40, 0, x_39);
lean_ctor_set(x_40, 1, lean_box(0));
lean_ctor_set(x_14, 1, lean_box(0));
lean_ctor_set(x_14, 0, x_40);
if (lean_is_scalar(x_36)) {
 x_41 = lean_alloc_ctor(0, 4, 0);
} else {
 x_41 = x_36;
}
lean_ctor_set(x_41, 0, x_14);
lean_ctor_set(x_41, 1, lean_box(0));
lean_ctor_set(x_41, 2, lean_box(0));
lean_ctor_set(x_41, 3, x_35);
lean_ctor_set(x_7, 3, lean_box(0));
lean_ctor_set(x_7, 2, lean_box(0));
lean_ctor_set(x_7, 1, lean_box(0));
lean_ctor_set(x_7, 0, x_41);
return x_7;
}
}
else
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; 
x_42 = lean_ctor_get(x_14, 0);
lean_inc(x_42);
lean_dec(x_14);
x_43 = lean_ctor_get(x_42, 0);
lean_inc_ref(x_43);
if (lean_is_exclusive(x_42)) {
 lean_ctor_release(x_42, 0);
 lean_ctor_release(x_42, 1);
 x_44 = x_42;
} else {
 lean_dec_ref(x_42);
 x_44 = lean_box(0);
}
x_45 = lean_ctor_get(x_10, 3);
lean_inc_ref(x_45);
if (lean_is_exclusive(x_10)) {
 lean_ctor_release(x_10, 0);
 lean_ctor_release(x_10, 1);
 lean_ctor_release(x_10, 2);
 lean_ctor_release(x_10, 3);
 x_46 = x_10;
} else {
 lean_dec_ref(x_10);
 x_46 = lean_box(0);
}
x_47 = lean_ctor_get(x_43, 1);
lean_inc(x_47);
if (lean_is_exclusive(x_43)) {
 lean_ctor_release(x_43, 0);
 lean_ctor_release(x_43, 1);
 x_48 = x_43;
} else {
 lean_dec_ref(x_43);
 x_48 = lean_box(0);
}
if (lean_is_scalar(x_48)) {
 x_49 = lean_alloc_ctor(0, 2, 0);
} else {
 x_49 = x_48;
}
lean_ctor_set(x_49, 0, x_8);
lean_ctor_set(x_49, 1, x_47);
if (lean_is_scalar(x_44)) {
 x_50 = lean_alloc_ctor(0, 2, 0);
} else {
 x_50 = x_44;
}
lean_ctor_set(x_50, 0, x_49);
lean_ctor_set(x_50, 1, lean_box(0));
x_51 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_51, 0, x_50);
lean_ctor_set(x_51, 1, lean_box(0));
if (lean_is_scalar(x_46)) {
 x_52 = lean_alloc_ctor(0, 4, 0);
} else {
 x_52 = x_46;
}
lean_ctor_set(x_52, 0, x_51);
lean_ctor_set(x_52, 1, lean_box(0));
lean_ctor_set(x_52, 2, lean_box(0));
lean_ctor_set(x_52, 3, x_45);
lean_ctor_set(x_7, 3, lean_box(0));
lean_ctor_set(x_7, 2, lean_box(0));
lean_ctor_set(x_7, 1, lean_box(0));
lean_ctor_set(x_7, 0, x_52);
return x_7;
}
}
else
{
lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; 
x_53 = lean_ctor_get(x_7, 0);
lean_inc(x_53);
lean_dec(x_7);
x_54 = lean_ctor_get(x_53, 0);
lean_inc_ref(x_54);
x_55 = lean_ctor_get(x_54, 0);
lean_inc_ref(x_55);
if (lean_is_exclusive(x_54)) {
 lean_ctor_release(x_54, 0);
 lean_ctor_release(x_54, 1);
 x_56 = x_54;
} else {
 lean_dec_ref(x_54);
 x_56 = lean_box(0);
}
x_57 = lean_ctor_get(x_55, 0);
lean_inc_ref(x_57);
if (lean_is_exclusive(x_55)) {
 lean_ctor_release(x_55, 0);
 lean_ctor_release(x_55, 1);
 x_58 = x_55;
} else {
 lean_dec_ref(x_55);
 x_58 = lean_box(0);
}
x_59 = lean_ctor_get(x_53, 3);
lean_inc_ref(x_59);
if (lean_is_exclusive(x_53)) {
 lean_ctor_release(x_53, 0);
 lean_ctor_release(x_53, 1);
 lean_ctor_release(x_53, 2);
 lean_ctor_release(x_53, 3);
 x_60 = x_53;
} else {
 lean_dec_ref(x_53);
 x_60 = lean_box(0);
}
x_61 = lean_ctor_get(x_57, 1);
lean_inc(x_61);
if (lean_is_exclusive(x_57)) {
 lean_ctor_release(x_57, 0);
 lean_ctor_release(x_57, 1);
 x_62 = x_57;
} else {
 lean_dec_ref(x_57);
 x_62 = lean_box(0);
}
if (lean_is_scalar(x_62)) {
 x_63 = lean_alloc_ctor(0, 2, 0);
} else {
 x_63 = x_62;
}
lean_ctor_set(x_63, 0, x_8);
lean_ctor_set(x_63, 1, x_61);
if (lean_is_scalar(x_58)) {
 x_64 = lean_alloc_ctor(0, 2, 0);
} else {
 x_64 = x_58;
}
lean_ctor_set(x_64, 0, x_63);
lean_ctor_set(x_64, 1, lean_box(0));
if (lean_is_scalar(x_56)) {
 x_65 = lean_alloc_ctor(0, 2, 0);
} else {
 x_65 = x_56;
}
lean_ctor_set(x_65, 0, x_64);
lean_ctor_set(x_65, 1, lean_box(0));
if (lean_is_scalar(x_60)) {
 x_66 = lean_alloc_ctor(0, 4, 0);
} else {
 x_66 = x_60;
}
lean_ctor_set(x_66, 0, x_65);
lean_ctor_set(x_66, 1, lean_box(0));
lean_ctor_set(x_66, 2, lean_box(0));
lean_ctor_set(x_66, 3, x_59);
x_67 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_67, 0, x_66);
lean_ctor_set(x_67, 1, lean_box(0));
lean_ctor_set(x_67, 2, lean_box(0));
lean_ctor_set(x_67, 3, lean_box(0));
return x_67;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_MorphismProperty_instCompleteBooleanAlgebra(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_instInhabited___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_MorphismProperty_instInhabited(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_homFamily(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_ctor_get(x_4, 2);
lean_inc(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_homFamily___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 2);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_homFamily___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_MorphismProperty_homFamily(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_homFamily___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_MorphismProperty_homFamily___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Comma_Arrow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_MorphismProperty_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Comma_Arrow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
