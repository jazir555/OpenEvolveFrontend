// Lean compiler output
// Module: Mathlib.CategoryTheory.Adjunction.Mates
// Imports: public import Init public import Mathlib.CategoryTheory.Adjunction.Basic public import Mathlib.CategoryTheory.Functor.TwoSquare public import Mathlib.CategoryTheory.HomCongr public import Mathlib.Tactic.ApplyFun
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
lean_object* lp_mathlib_CategoryTheory_Functor_leftUnitor___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_TwoSquare_equivNatTrans___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateIsoEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_associator___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_rightUnitor___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_homCongr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_category___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_whiskerRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; 
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_13 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_1, x_2);
lean_inc_ref(x_13);
lean_inc_ref(x_3);
x_14 = lp_mathlib_CategoryTheory_Functor_rightUnitor___redArg(x_3, x_13);
x_15 = lean_ctor_get(x_14, 1);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_4, 0);
lean_inc(x_16);
lean_dec_ref(x_4);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
x_17 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_5, x_6);
lean_inc_ref(x_17);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc_ref(x_3);
x_18 = lp_mathlib_CategoryTheory_Functor_associator___redArg(x_3, x_1, x_2, x_17);
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_2);
lean_inc_ref(x_3);
x_20 = lp_mathlib_CategoryTheory_Functor_associator___redArg(x_3, x_2, x_5, x_6);
x_21 = lean_ctor_get(x_20, 1);
lean_inc(x_21);
lean_dec_ref(x_20);
lean_inc_ref(x_6);
lean_inc_ref(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_3);
x_22 = lp_mathlib_CategoryTheory_Functor_associator___redArg(x_3, x_7, x_8, x_6);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec_ref(x_22);
lean_inc_ref(x_6);
lean_inc_ref(x_8);
x_24 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_8, x_6);
lean_inc_ref(x_24);
lean_inc_ref(x_7);
lean_inc_ref(x_1);
lean_inc_ref(x_3);
x_25 = lp_mathlib_CategoryTheory_Functor_associator___redArg(x_3, x_1, x_7, x_24);
x_26 = lean_ctor_get(x_25, 1);
lean_inc(x_26);
lean_dec_ref(x_25);
x_27 = lean_ctor_get(x_9, 1);
lean_inc(x_27);
lean_dec_ref(x_9);
lean_inc_ref(x_24);
lean_inc_ref(x_3);
x_28 = lp_mathlib_CategoryTheory_Functor_leftUnitor___redArg(x_3, x_24);
x_29 = lean_ctor_get(x_28, 0);
lean_inc(x_29);
lean_dec_ref(x_28);
x_30 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_3);
lean_inc_ref(x_13);
x_31 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_13, x_30);
lean_inc_ref(x_17);
lean_inc_ref(x_13);
x_32 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_13, x_17);
lean_inc_ref(x_13);
x_33 = lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(x_13, x_16);
lean_inc_ref(x_2);
x_34 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_2, x_17);
lean_inc_ref(x_1);
x_35 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_1, x_34);
x_36 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_2, x_5);
lean_inc_ref(x_6);
lean_inc_ref(x_36);
x_37 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_36, x_6);
lean_inc_ref(x_1);
x_38 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_1, x_37);
lean_inc_ref(x_1);
x_39 = lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(x_1, x_21);
lean_inc_ref(x_7);
x_40 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_7, x_8);
lean_inc_ref(x_6);
lean_inc_ref(x_40);
x_41 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_40, x_6);
lean_inc_ref(x_1);
x_42 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_1, x_41);
x_43 = lp_mathlib_CategoryTheory_Functor_whiskerRight___redArg(x_36, x_40, x_11, x_6);
lean_inc_ref(x_1);
x_44 = lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(x_1, x_43);
lean_inc_ref(x_24);
lean_inc_ref(x_7);
x_45 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_7, x_24);
lean_inc_ref(x_1);
x_46 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_1, x_45);
lean_inc_ref(x_1);
x_47 = lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(x_1, x_23);
x_48 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_1, x_7);
lean_inc_ref(x_24);
lean_inc_ref(x_48);
x_49 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_48, x_24);
x_50 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_10);
lean_inc_ref(x_24);
lean_inc_ref(x_50);
x_51 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_50, x_24);
lean_inc_ref(x_24);
x_52 = lp_mathlib_CategoryTheory_Functor_whiskerRight___redArg(x_48, x_50, x_27, x_24);
lean_inc_ref(x_24);
lean_inc_ref(x_49);
lean_inc_ref(x_3);
x_53 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_49, x_51, x_24, x_52, x_29);
lean_inc_ref(x_24);
lean_inc_ref(x_46);
lean_inc_ref(x_3);
x_54 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_46, x_49, x_24, x_26, x_53);
lean_inc_ref(x_24);
lean_inc_ref(x_42);
lean_inc_ref(x_3);
x_55 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_42, x_46, x_24, x_47, x_54);
lean_inc_ref(x_24);
lean_inc_ref(x_38);
lean_inc_ref(x_3);
x_56 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_38, x_42, x_24, x_44, x_55);
lean_inc_ref(x_24);
lean_inc_ref(x_35);
lean_inc_ref(x_3);
x_57 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_35, x_38, x_24, x_39, x_56);
lean_inc_ref(x_24);
lean_inc_ref(x_32);
lean_inc_ref(x_3);
x_58 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_32, x_35, x_24, x_19, x_57);
lean_inc_ref(x_24);
lean_inc_ref(x_31);
lean_inc_ref(x_3);
x_59 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_31, x_32, x_24, x_33, x_58);
x_60 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_13, x_31, x_24, x_15, x_59);
x_61 = lean_apply_1(x_60, x_12);
return x_61;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; 
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_14 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_1, x_2);
lean_inc_ref(x_14);
lean_inc_ref(x_3);
x_15 = lp_mathlib_CategoryTheory_Functor_leftUnitor___redArg(x_3, x_14);
x_16 = lean_ctor_get(x_15, 1);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_ctor_get(x_4, 0);
lean_inc(x_17);
lean_dec_ref(x_4);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
x_18 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_5, x_6);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc_ref(x_18);
lean_inc_ref(x_3);
x_19 = lp_mathlib_CategoryTheory_Functor_associator___redArg(x_3, x_18, x_1, x_2);
x_20 = lean_ctor_get(x_19, 1);
lean_inc(x_20);
lean_dec_ref(x_19);
lean_inc_ref(x_1);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_7);
x_21 = lp_mathlib_CategoryTheory_Functor_associator___redArg(x_7, x_5, x_6, x_1);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
lean_dec_ref(x_21);
lean_inc_ref(x_9);
lean_inc_ref(x_8);
lean_inc_ref(x_5);
x_23 = lp_mathlib_CategoryTheory_Functor_associator___redArg(x_7, x_5, x_8, x_9);
x_24 = lean_ctor_get(x_23, 1);
lean_inc(x_24);
lean_dec_ref(x_23);
lean_inc_ref(x_8);
lean_inc_ref(x_5);
x_25 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_5, x_8);
lean_inc_ref(x_2);
lean_inc_ref(x_9);
lean_inc_ref(x_25);
lean_inc_ref(x_3);
x_26 = lp_mathlib_CategoryTheory_Functor_associator___redArg(x_3, x_25, x_9, x_2);
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
lean_dec_ref(x_26);
x_28 = lean_ctor_get(x_10, 1);
lean_inc(x_28);
lean_dec_ref(x_10);
lean_inc_ref(x_25);
lean_inc_ref(x_3);
x_29 = lp_mathlib_CategoryTheory_Functor_rightUnitor___redArg(x_3, x_25);
x_30 = lean_ctor_get(x_29, 0);
lean_inc(x_30);
lean_dec_ref(x_29);
x_31 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_11);
lean_inc_ref(x_14);
lean_inc_ref(x_31);
x_32 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_31, x_14);
lean_inc_ref(x_14);
lean_inc_ref(x_18);
x_33 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_18, x_14);
lean_inc_ref(x_14);
lean_inc_ref(x_18);
x_34 = lp_mathlib_CategoryTheory_Functor_whiskerRight___redArg(x_31, x_18, x_17, x_14);
lean_inc_ref(x_1);
x_35 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_18, x_1);
lean_inc_ref(x_2);
lean_inc_ref(x_35);
x_36 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_35, x_2);
x_37 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_6, x_1);
lean_inc_ref(x_5);
x_38 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_5, x_37);
lean_inc_ref(x_2);
lean_inc_ref(x_38);
x_39 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_38, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_38);
x_40 = lp_mathlib_CategoryTheory_Functor_whiskerRight___redArg(x_35, x_38, x_22, x_2);
lean_inc_ref(x_9);
x_41 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_8, x_9);
lean_inc_ref(x_5);
x_42 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_5, x_41);
lean_inc_ref(x_2);
lean_inc_ref(x_42);
x_43 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_42, x_2);
x_44 = lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(x_5, x_12);
lean_inc_ref(x_2);
lean_inc_ref(x_42);
x_45 = lp_mathlib_CategoryTheory_Functor_whiskerRight___redArg(x_38, x_42, x_44, x_2);
lean_inc_ref(x_9);
lean_inc_ref(x_25);
x_46 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_25, x_9);
lean_inc_ref(x_2);
lean_inc_ref(x_46);
x_47 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_46, x_2);
lean_inc_ref(x_2);
x_48 = lp_mathlib_CategoryTheory_Functor_whiskerRight___redArg(x_42, x_46, x_24, x_2);
x_49 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_9, x_2);
lean_inc_ref(x_25);
x_50 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_25, x_49);
x_51 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_3);
lean_inc_ref(x_25);
x_52 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_25, x_51);
lean_inc_ref(x_25);
x_53 = lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(x_25, x_28);
lean_inc_ref(x_25);
lean_inc_ref(x_50);
lean_inc_ref(x_3);
x_54 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_50, x_52, x_25, x_53, x_30);
lean_inc_ref(x_25);
lean_inc_ref(x_47);
lean_inc_ref(x_3);
x_55 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_47, x_50, x_25, x_27, x_54);
lean_inc_ref(x_25);
lean_inc_ref(x_43);
lean_inc_ref(x_3);
x_56 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_43, x_47, x_25, x_48, x_55);
lean_inc_ref(x_25);
lean_inc_ref(x_39);
lean_inc_ref(x_3);
x_57 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_39, x_43, x_25, x_45, x_56);
lean_inc_ref(x_25);
lean_inc_ref(x_36);
lean_inc_ref(x_3);
x_58 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_36, x_39, x_25, x_40, x_57);
lean_inc_ref(x_25);
lean_inc_ref(x_33);
lean_inc_ref(x_3);
x_59 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_33, x_36, x_25, x_20, x_58);
lean_inc_ref(x_25);
lean_inc_ref(x_32);
lean_inc_ref(x_3);
x_60 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_32, x_33, x_25, x_34, x_59);
x_61 = lp_mathlib_CategoryTheory_NatTrans_vcomp___redArg(x_3, x_14, x_32, x_25, x_16, x_60);
x_62 = lean_apply_1(x_61, x_13);
return x_62;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_11);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_inc_ref(x_11);
lean_inc_ref(x_6);
lean_inc_ref(x_7);
lean_inc_ref(x_10);
lean_inc_ref(x_9);
lean_inc_ref(x_12);
lean_inc_ref(x_3);
lean_inc_ref(x_5);
lean_inc_ref(x_8);
x_13 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__0___boxed), 12, 10);
lean_closure_set(x_13, 0, x_8);
lean_closure_set(x_13, 1, x_5);
lean_closure_set(x_13, 2, x_3);
lean_closure_set(x_13, 3, x_12);
lean_closure_set(x_13, 4, x_9);
lean_closure_set(x_13, 5, x_10);
lean_closure_set(x_13, 6, x_7);
lean_closure_set(x_13, 7, x_6);
lean_closure_set(x_13, 8, x_11);
lean_closure_set(x_13, 9, x_2);
x_14 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_mateEquiv___redArg___lam__1___boxed), 13, 11);
lean_closure_set(x_14, 0, x_5);
lean_closure_set(x_14, 1, x_9);
lean_closure_set(x_14, 2, x_4);
lean_closure_set(x_14, 3, x_11);
lean_closure_set(x_14, 4, x_7);
lean_closure_set(x_14, 5, x_8);
lean_closure_set(x_14, 6, x_3);
lean_closure_set(x_14, 7, x_6);
lean_closure_set(x_14, 8, x_10);
lean_closure_set(x_14, 9, x_12);
lean_closure_set(x_14, 10, x_1);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_mateEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_CategoryTheory_mateEquiv___redArg(x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
lean_inc_ref(x_2);
x_9 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_2);
x_10 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_1);
lean_inc_ref(x_4);
lean_inc_ref(x_10);
x_11 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_10, x_4);
x_12 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_2);
lean_inc_ref(x_12);
lean_inc_ref(x_3);
x_13 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_3, x_12);
lean_inc_ref(x_4);
lean_inc_ref(x_2);
x_14 = lp_mathlib_CategoryTheory_Functor_leftUnitor___redArg(x_2, x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_15 = lp_mathlib_CategoryTheory_Functor_rightUnitor___redArg(x_2, x_3);
lean_inc_ref(x_3);
lean_inc_ref(x_4);
x_16 = lp_mathlib_CategoryTheory_Iso_homCongr___redArg(x_9, x_11, x_13, x_4, x_3, x_14, x_15);
x_17 = lp_mathlib_Equiv_symm___redArg(x_16);
lean_inc_ref(x_12);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_10);
lean_inc_ref_n(x_2, 2);
lean_inc_ref_n(x_1, 2);
x_18 = lp_mathlib_CategoryTheory_TwoSquare_equivNatTrans___redArg(x_1, x_1, x_2, x_2, x_10, x_3, x_4, x_12);
x_19 = lp_mathlib_Equiv_symm___redArg(x_18);
x_20 = lp_mathlib_Equiv_trans___redArg(x_17, x_19);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_12);
lean_inc_ref(x_10);
lean_inc_ref_n(x_2, 2);
lean_inc_ref_n(x_1, 2);
x_21 = lp_mathlib_CategoryTheory_mateEquiv___redArg(x_1, x_2, x_1, x_2, x_10, x_12, x_3, x_5, x_4, x_6, x_7, x_8);
x_22 = lp_mathlib_Equiv_trans___redArg(x_20, x_21);
lean_inc_ref(x_6);
lean_inc_ref(x_10);
lean_inc_ref(x_12);
lean_inc_ref(x_5);
lean_inc_ref_n(x_1, 2);
lean_inc_ref(x_2);
x_23 = lp_mathlib_CategoryTheory_TwoSquare_equivNatTrans___redArg(x_2, x_1, x_2, x_1, x_5, x_12, x_10, x_6);
x_24 = lp_mathlib_Equiv_trans___redArg(x_22, x_23);
lean_inc_ref(x_1);
x_25 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_1);
lean_inc_ref(x_5);
x_26 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_5, x_10);
lean_inc_ref(x_6);
x_27 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_12, x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_1);
x_28 = lp_mathlib_CategoryTheory_Functor_rightUnitor___redArg(x_1, x_5);
lean_inc_ref(x_6);
x_29 = lp_mathlib_CategoryTheory_Functor_leftUnitor___redArg(x_1, x_6);
x_30 = lp_mathlib_CategoryTheory_Iso_homCongr___redArg(x_25, x_26, x_27, x_5, x_6, x_28, x_29);
x_31 = lp_mathlib_Equiv_trans___redArg(x_24, x_30);
return x_31;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_conjugateEquiv___redArg(x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; 
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_11 = lean_ctor_get(x_9, 0);
x_12 = lean_ctor_get(x_9, 1);
lean_inc_ref(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_13 = lp_mathlib_CategoryTheory_conjugateEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lp_mathlib_CategoryTheory_conjugateEquiv___redArg(x_1, x_2, x_4, x_3, x_6, x_5, x_8, x_7);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_apply_1(x_14, x_11);
x_18 = lean_apply_1(x_16, x_12);
lean_ctor_set(x_9, 1, x_18);
lean_ctor_set(x_9, 0, x_17);
return x_9;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_19 = lean_ctor_get(x_9, 0);
x_20 = lean_ctor_get(x_9, 1);
lean_inc(x_20);
lean_inc(x_19);
lean_dec(x_9);
lean_inc_ref(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_21 = lp_mathlib_CategoryTheory_conjugateEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
lean_dec_ref(x_21);
x_23 = lp_mathlib_CategoryTheory_conjugateEquiv___redArg(x_1, x_2, x_4, x_3, x_6, x_5, x_8, x_7);
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
lean_dec_ref(x_23);
x_25 = lean_apply_1(x_22, x_19);
x_26 = lean_apply_1(x_24, x_20);
x_27 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_27, 0, x_25);
lean_ctor_set(x_27, 1, x_26);
return x_27;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; 
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_11 = lean_ctor_get(x_9, 0);
x_12 = lean_ctor_get(x_9, 1);
lean_inc_ref(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_13 = lp_mathlib_CategoryTheory_conjugateEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
x_14 = lp_mathlib_Equiv_symm___redArg(x_13);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lp_mathlib_CategoryTheory_conjugateEquiv___redArg(x_1, x_2, x_4, x_3, x_6, x_5, x_8, x_7);
x_17 = lp_mathlib_Equiv_symm___redArg(x_16);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lean_apply_1(x_15, x_11);
x_20 = lean_apply_1(x_18, x_12);
lean_ctor_set(x_9, 1, x_20);
lean_ctor_set(x_9, 0, x_19);
return x_9;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_21 = lean_ctor_get(x_9, 0);
x_22 = lean_ctor_get(x_9, 1);
lean_inc(x_22);
lean_inc(x_21);
lean_dec(x_9);
lean_inc_ref(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_23 = lp_mathlib_CategoryTheory_conjugateEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
x_24 = lp_mathlib_Equiv_symm___redArg(x_23);
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
lean_dec_ref(x_24);
x_26 = lp_mathlib_CategoryTheory_conjugateEquiv___redArg(x_1, x_2, x_4, x_3, x_6, x_5, x_8, x_7);
x_27 = lp_mathlib_Equiv_symm___redArg(x_26);
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
lean_dec_ref(x_27);
x_29 = lean_apply_1(x_25, x_21);
x_30 = lean_apply_1(x_28, x_22);
x_31 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_31, 0, x_29);
lean_ctor_set(x_31, 1, x_30);
return x_31;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg___lam__0), 9, 8);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_2);
lean_closure_set(x_9, 2, x_3);
lean_closure_set(x_9, 3, x_4);
lean_closure_set(x_9, 4, x_5);
lean_closure_set(x_9, 5, x_6);
lean_closure_set(x_9, 6, x_7);
lean_closure_set(x_9, 7, x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg___lam__1), 9, 8);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_2);
lean_closure_set(x_10, 2, x_3);
lean_closure_set(x_10, 3, x_4);
lean_closure_set(x_10, 4, x_5);
lean_closure_set(x_10, 5, x_6);
lean_closure_set(x_10, 6, x_7);
lean_closure_set(x_10, 7, x_8);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_conjugateIsoEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg(x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Functor_TwoSquare(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_HomCongr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Mates(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Functor_TwoSquare(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_HomCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
