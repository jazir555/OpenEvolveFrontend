// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Final
// Imports: public import Init public import Mathlib.CategoryTheory.Category.Cat.AsSmall public import Mathlib.CategoryTheory.Comma.StructuredArrow.Basic public import Mathlib.CategoryTheory.IsConnected public import Mathlib.CategoryTheory.Limits.Preserves.Shapes.Terminal public import Mathlib.CategoryTheory.Limits.Types.Products public import Mathlib.CategoryTheory.Limits.Shapes.Grothendieck public import Mathlib.CategoryTheory.Filtered.Basic public import Mathlib.CategoryTheory.Limits.Yoneda public import Mathlib.CategoryTheory.PUnit public import Mathlib.CategoryTheory.Grothendieck
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_const___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_StructuredArrow_mk___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_StructuredArrow_homMk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_eqToHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; uint8_t x_20; 
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 2);
lean_inc(x_9);
lean_dec_ref(x_6);
x_10 = lean_ctor_get(x_1, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_1, 1);
lean_inc(x_11);
lean_dec_ref(x_1);
x_12 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_2);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc(x_3);
x_14 = lean_apply_1(x_13, x_3);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_4, 0);
lean_inc(x_16);
lean_dec_ref(x_4);
x_17 = lean_apply_1(x_15, x_7);
lean_inc(x_8);
x_18 = lean_apply_1(x_16, x_8);
lean_inc(x_11);
lean_inc(x_9);
lean_inc(x_18);
x_19 = lean_apply_3(x_11, x_17, x_18, x_9);
x_20 = !lean_is_exclusive(x_19);
if (x_20 == 0)
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; 
x_21 = lean_ctor_get(x_19, 0);
x_22 = lean_ctor_get(x_19, 1);
lean_dec(x_22);
lean_inc(x_18);
x_23 = lean_apply_1(x_10, x_18);
x_24 = lean_ctor_get(x_23, 1);
lean_inc(x_24);
lean_dec_ref(x_23);
lean_inc(x_9);
x_25 = lean_apply_3(x_11, x_3, x_18, x_9);
x_26 = !lean_is_exclusive(x_25);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_27 = lean_ctor_get(x_25, 0);
x_28 = lean_ctor_get(x_25, 1);
lean_dec(x_28);
lean_inc(x_5);
x_29 = lean_apply_1(x_21, x_5);
lean_ctor_set(x_25, 1, x_29);
lean_ctor_set(x_25, 0, x_8);
x_30 = lean_apply_1(x_27, x_5);
x_31 = lean_apply_1(x_24, x_30);
lean_ctor_set(x_19, 1, x_31);
lean_ctor_set(x_19, 0, x_9);
x_32 = lp_mathlib_CategoryTheory_StructuredArrow_mk___redArg(x_25, x_19);
return x_32;
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_33 = lean_ctor_get(x_25, 0);
lean_inc(x_33);
lean_dec(x_25);
lean_inc(x_5);
x_34 = lean_apply_1(x_21, x_5);
x_35 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_35, 0, x_8);
lean_ctor_set(x_35, 1, x_34);
x_36 = lean_apply_1(x_33, x_5);
x_37 = lean_apply_1(x_24, x_36);
lean_ctor_set(x_19, 1, x_37);
lean_ctor_set(x_19, 0, x_9);
x_38 = lp_mathlib_CategoryTheory_StructuredArrow_mk___redArg(x_35, x_19);
return x_38;
}
}
else
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; 
x_39 = lean_ctor_get(x_19, 0);
lean_inc(x_39);
lean_dec(x_19);
lean_inc(x_18);
x_40 = lean_apply_1(x_10, x_18);
x_41 = lean_ctor_get(x_40, 1);
lean_inc(x_41);
lean_dec_ref(x_40);
lean_inc(x_9);
x_42 = lean_apply_3(x_11, x_3, x_18, x_9);
x_43 = lean_ctor_get(x_42, 0);
lean_inc(x_43);
if (lean_is_exclusive(x_42)) {
 lean_ctor_release(x_42, 0);
 lean_ctor_release(x_42, 1);
 x_44 = x_42;
} else {
 lean_dec_ref(x_42);
 x_44 = lean_box(0);
}
lean_inc(x_5);
x_45 = lean_apply_1(x_39, x_5);
if (lean_is_scalar(x_44)) {
 x_46 = lean_alloc_ctor(0, 2, 0);
} else {
 x_46 = x_44;
}
lean_ctor_set(x_46, 0, x_8);
lean_ctor_set(x_46, 1, x_45);
x_47 = lean_apply_1(x_43, x_5);
x_48 = lean_apply_1(x_41, x_47);
x_49 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_49, 0, x_9);
lean_ctor_set(x_49, 1, x_48);
x_50 = lp_mathlib_CategoryTheory_StructuredArrow_mk___redArg(x_46, x_49);
return x_50;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; 
x_10 = lean_ctor_get(x_1, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_1, 1);
lean_inc(x_11);
lean_dec_ref(x_1);
x_12 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_2);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc(x_3);
x_14 = lean_apply_1(x_13, x_3);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_4, 0);
lean_inc(x_16);
lean_dec_ref(x_4);
x_17 = lean_ctor_get(x_8, 0);
lean_inc(x_17);
x_18 = lean_ctor_get(x_8, 1);
lean_inc(x_18);
x_19 = lean_ctor_get(x_8, 2);
lean_inc(x_19);
lean_dec_ref(x_8);
x_20 = lean_apply_1(x_15, x_17);
lean_inc(x_18);
x_21 = lean_apply_1(x_16, x_18);
lean_inc(x_11);
lean_inc(x_19);
lean_inc(x_21);
x_22 = lean_apply_3(x_11, x_20, x_21, x_19);
x_23 = !lean_is_exclusive(x_22);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; 
x_24 = lean_ctor_get(x_22, 0);
x_25 = lean_ctor_get(x_22, 1);
lean_dec(x_25);
lean_inc(x_21);
x_26 = lean_apply_1(x_10, x_21);
x_27 = lean_ctor_get(x_26, 1);
lean_inc(x_27);
lean_dec_ref(x_26);
lean_inc(x_19);
x_28 = lean_apply_3(x_11, x_3, x_21, x_19);
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; uint8_t x_39; 
x_30 = lean_ctor_get(x_28, 0);
x_31 = lean_ctor_get(x_28, 1);
lean_dec(x_31);
lean_inc(x_5);
x_32 = lean_apply_1(x_24, x_5);
lean_ctor_set(x_28, 1, x_32);
lean_ctor_set(x_28, 0, x_18);
x_33 = lean_apply_1(x_30, x_5);
x_34 = lean_apply_1(x_27, x_33);
lean_ctor_set(x_22, 1, x_34);
lean_ctor_set(x_22, 0, x_19);
x_35 = lp_mathlib_CategoryTheory_StructuredArrow_mk___redArg(x_28, x_22);
x_36 = lean_ctor_get(x_35, 1);
lean_inc(x_36);
lean_dec_ref(x_35);
x_37 = lean_ctor_get(x_9, 1);
x_38 = lean_ctor_get(x_6, 0);
lean_inc(x_38);
lean_dec_ref(x_6);
x_39 = !lean_is_exclusive(x_36);
if (x_39 == 0)
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; 
x_40 = lean_ctor_get(x_36, 0);
x_41 = lean_ctor_get(x_36, 1);
x_42 = lean_apply_1(x_38, x_40);
x_43 = lp_mathlib_CategoryTheory_eqToHom___redArg(x_42, x_41);
lean_inc(x_37);
lean_ctor_set(x_36, 1, x_43);
lean_ctor_set(x_36, 0, x_37);
x_44 = lp_mathlib_CategoryTheory_StructuredArrow_homMk___redArg(x_36);
return x_44;
}
else
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; 
x_45 = lean_ctor_get(x_36, 0);
x_46 = lean_ctor_get(x_36, 1);
lean_inc(x_46);
lean_inc(x_45);
lean_dec(x_36);
x_47 = lean_apply_1(x_38, x_45);
x_48 = lp_mathlib_CategoryTheory_eqToHom___redArg(x_47, x_46);
lean_inc(x_37);
x_49 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_49, 0, x_37);
lean_ctor_set(x_49, 1, x_48);
x_50 = lp_mathlib_CategoryTheory_StructuredArrow_homMk___redArg(x_49);
return x_50;
}
}
else
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; 
x_51 = lean_ctor_get(x_28, 0);
lean_inc(x_51);
lean_dec(x_28);
lean_inc(x_5);
x_52 = lean_apply_1(x_24, x_5);
x_53 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_53, 0, x_18);
lean_ctor_set(x_53, 1, x_52);
x_54 = lean_apply_1(x_51, x_5);
x_55 = lean_apply_1(x_27, x_54);
lean_ctor_set(x_22, 1, x_55);
lean_ctor_set(x_22, 0, x_19);
x_56 = lp_mathlib_CategoryTheory_StructuredArrow_mk___redArg(x_53, x_22);
x_57 = lean_ctor_get(x_56, 1);
lean_inc(x_57);
lean_dec_ref(x_56);
x_58 = lean_ctor_get(x_9, 1);
x_59 = lean_ctor_get(x_6, 0);
lean_inc(x_59);
lean_dec_ref(x_6);
x_60 = lean_ctor_get(x_57, 0);
lean_inc(x_60);
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
x_63 = lean_apply_1(x_59, x_60);
x_64 = lp_mathlib_CategoryTheory_eqToHom___redArg(x_63, x_61);
lean_inc(x_58);
if (lean_is_scalar(x_62)) {
 x_65 = lean_alloc_ctor(0, 2, 0);
} else {
 x_65 = x_62;
}
lean_ctor_set(x_65, 0, x_58);
lean_ctor_set(x_65, 1, x_64);
x_66 = lp_mathlib_CategoryTheory_StructuredArrow_homMk___redArg(x_65);
return x_66;
}
}
else
{
lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; 
x_67 = lean_ctor_get(x_22, 0);
lean_inc(x_67);
lean_dec(x_22);
lean_inc(x_21);
x_68 = lean_apply_1(x_10, x_21);
x_69 = lean_ctor_get(x_68, 1);
lean_inc(x_69);
lean_dec_ref(x_68);
lean_inc(x_19);
x_70 = lean_apply_3(x_11, x_3, x_21, x_19);
x_71 = lean_ctor_get(x_70, 0);
lean_inc(x_71);
if (lean_is_exclusive(x_70)) {
 lean_ctor_release(x_70, 0);
 lean_ctor_release(x_70, 1);
 x_72 = x_70;
} else {
 lean_dec_ref(x_70);
 x_72 = lean_box(0);
}
lean_inc(x_5);
x_73 = lean_apply_1(x_67, x_5);
if (lean_is_scalar(x_72)) {
 x_74 = lean_alloc_ctor(0, 2, 0);
} else {
 x_74 = x_72;
}
lean_ctor_set(x_74, 0, x_18);
lean_ctor_set(x_74, 1, x_73);
x_75 = lean_apply_1(x_71, x_5);
x_76 = lean_apply_1(x_69, x_75);
x_77 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_77, 0, x_19);
lean_ctor_set(x_77, 1, x_76);
x_78 = lp_mathlib_CategoryTheory_StructuredArrow_mk___redArg(x_74, x_77);
x_79 = lean_ctor_get(x_78, 1);
lean_inc(x_79);
lean_dec_ref(x_78);
x_80 = lean_ctor_get(x_9, 1);
x_81 = lean_ctor_get(x_6, 0);
lean_inc(x_81);
lean_dec_ref(x_6);
x_82 = lean_ctor_get(x_79, 0);
lean_inc(x_82);
x_83 = lean_ctor_get(x_79, 1);
lean_inc(x_83);
if (lean_is_exclusive(x_79)) {
 lean_ctor_release(x_79, 0);
 lean_ctor_release(x_79, 1);
 x_84 = x_79;
} else {
 lean_dec_ref(x_79);
 x_84 = lean_box(0);
}
x_85 = lean_apply_1(x_81, x_82);
x_86 = lp_mathlib_CategoryTheory_eqToHom___redArg(x_85, x_83);
lean_inc(x_80);
if (lean_is_scalar(x_84)) {
 x_87 = lean_alloc_ctor(0, 2, 0);
} else {
 x_87 = x_84;
}
lean_ctor_set(x_87, 0, x_80);
lean_ctor_set(x_87, 1, x_86);
x_88 = lp_mathlib_CategoryTheory_StructuredArrow_homMk___redArg(x_87);
return x_88;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_9);
lean_dec_ref(x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_5);
lean_inc_ref(x_3);
lean_inc(x_4);
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg___lam__0), 6, 5);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, x_4);
lean_closure_set(x_6, 3, x_3);
lean_closure_set(x_6, 4, x_5);
lean_inc_ref(x_2);
lean_inc_ref(x_3);
x_7 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_3, x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg___lam__1___boxed), 9, 6);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_1);
lean_closure_set(x_8, 2, x_4);
lean_closure_set(x_8, 3, x_3);
lean_closure_set(x_8, 4, x_5);
lean_closure_set(x_8, 5, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Grothendieck_structuredArrowToStructuredArrowPre(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_2);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Category_Cat_AsSmall(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Comma_StructuredArrow_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_IsConnected(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_Terminal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Types_Products(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Grothendieck(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Filtered_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Yoneda(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_PUnit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Grothendieck(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Final(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Category_Cat_AsSmall(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Comma_StructuredArrow_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_IsConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_Terminal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Types_Products(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Grothendieck(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Filtered_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Yoneda(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Grothendieck(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
