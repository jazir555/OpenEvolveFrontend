// Lean compiler output
// Module: Mathlib.Algebra.Field.MinimalAxioms
// Imports: public import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.Ring.MinimalAxioms
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
lean_object* lp_mathlib_Rat_castRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_unaryCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddGroup_ofLeftAxioms___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Int_castDef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___boxed(lean_object**);
lean_object* lp_mathlib_NNRat_castRec___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, x_2);
x_7 = lp_mathlib_zsmulRec___redArg(x_3, x_6, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_Rat_castRec___redArg(x_1, x_2, x_3, x_5);
x_8 = lean_apply_2(x_4, x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_NNRat_castRec___redArg(x_1, x_2, x_4);
x_7 = lean_apply_2(x_3, x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Field_ofMinimalAxioms___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; uint8_t x_19; 
lean_inc(x_6);
lean_inc(x_4);
lean_inc(x_2);
x_18 = lp_mathlib_AddGroup_ofLeftAxioms___redArg(x_2, x_4, x_6);
x_19 = !lean_is_exclusive(x_18);
if (x_19 == 0)
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; 
x_20 = lean_ctor_get(x_18, 0);
x_21 = lean_ctor_get(x_18, 2);
x_22 = lean_ctor_get(x_18, 3);
lean_dec(x_22);
x_23 = lean_ctor_get(x_18, 1);
lean_dec(x_23);
lean_inc(x_3);
lean_inc_ref(x_20);
x_24 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_24, 0, x_20);
lean_ctor_set(x_24, 1, x_3);
x_25 = !lean_is_exclusive(x_20);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_26 = lean_ctor_get(x_20, 0);
x_27 = lean_ctor_get(x_20, 1);
x_28 = lean_ctor_get(x_20, 2);
lean_dec(x_28);
lean_inc(x_4);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_29, 0, x_6);
lean_closure_set(x_29, 1, x_2);
lean_closure_set(x_29, 2, x_4);
lean_inc(x_7);
x_30 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, x_7);
lean_closure_set(x_30, 2, x_27);
lean_closure_set(x_30, 3, x_26);
lean_inc(x_7);
lean_inc(x_3);
x_31 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_31, 0, lean_box(0));
lean_closure_set(x_31, 1, x_3);
lean_closure_set(x_31, 2, x_7);
lean_inc_ref(x_31);
lean_inc_ref(x_30);
lean_inc(x_7);
lean_ctor_set(x_18, 3, x_31);
lean_ctor_set(x_18, 2, x_30);
lean_ctor_set(x_18, 1, x_7);
lean_ctor_set(x_18, 0, x_24);
lean_inc(x_4);
lean_inc_ref(x_30);
x_32 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_32, 0, lean_box(0));
lean_closure_set(x_32, 1, x_30);
lean_closure_set(x_32, 2, x_4);
lean_inc_ref(x_32);
x_33 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_33, 0, x_18);
lean_ctor_set(x_33, 1, x_4);
lean_ctor_set(x_33, 2, x_21);
lean_ctor_set(x_33, 3, x_29);
lean_ctor_set(x_33, 4, x_32);
lean_inc(x_7);
lean_inc(x_3);
lean_ctor_set(x_20, 2, x_31);
lean_ctor_set(x_20, 1, x_7);
lean_ctor_set(x_20, 0, x_3);
lean_inc(x_5);
x_34 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_34, 0, lean_box(0));
lean_closure_set(x_34, 1, x_20);
lean_closure_set(x_34, 2, x_5);
lean_inc(x_3);
lean_inc_ref(x_34);
lean_inc_ref(x_32);
lean_inc_ref(x_30);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__1), 6, 4);
lean_closure_set(x_35, 0, x_30);
lean_closure_set(x_35, 1, x_32);
lean_closure_set(x_35, 2, x_34);
lean_closure_set(x_35, 3, x_3);
lean_inc(x_3);
lean_inc_ref(x_34);
lean_inc_ref(x_30);
x_36 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__2), 5, 3);
lean_closure_set(x_36, 0, x_30);
lean_closure_set(x_36, 1, x_34);
lean_closure_set(x_36, 2, x_3);
lean_inc(x_3);
lean_inc(x_7);
x_37 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_37, 0, lean_box(0));
lean_closure_set(x_37, 1, x_7);
lean_closure_set(x_37, 2, x_3);
lean_inc(x_5);
x_38 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_38, 0, lean_box(0));
lean_closure_set(x_38, 1, x_7);
lean_closure_set(x_38, 2, x_3);
lean_closure_set(x_38, 3, x_5);
lean_closure_set(x_38, 4, x_37);
lean_inc_ref(x_34);
lean_inc_ref(x_30);
x_39 = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(x_39, 0, lean_box(0));
lean_closure_set(x_39, 1, x_30);
lean_closure_set(x_39, 2, x_34);
lean_inc_ref(x_34);
x_40 = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(x_40, 0, lean_box(0));
lean_closure_set(x_40, 1, x_30);
lean_closure_set(x_40, 2, x_32);
lean_closure_set(x_40, 3, x_34);
x_41 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_41, 0, x_33);
lean_ctor_set(x_41, 1, x_5);
lean_ctor_set(x_41, 2, x_34);
lean_ctor_set(x_41, 3, x_38);
lean_ctor_set(x_41, 4, x_39);
lean_ctor_set(x_41, 5, x_40);
lean_ctor_set(x_41, 6, x_36);
lean_ctor_set(x_41, 7, x_35);
return x_41;
}
else
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; 
x_42 = lean_ctor_get(x_20, 0);
x_43 = lean_ctor_get(x_20, 1);
lean_inc(x_43);
lean_inc(x_42);
lean_dec(x_20);
lean_inc(x_4);
x_44 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_44, 0, x_6);
lean_closure_set(x_44, 1, x_2);
lean_closure_set(x_44, 2, x_4);
lean_inc(x_7);
x_45 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_45, 0, lean_box(0));
lean_closure_set(x_45, 1, x_7);
lean_closure_set(x_45, 2, x_43);
lean_closure_set(x_45, 3, x_42);
lean_inc(x_7);
lean_inc(x_3);
x_46 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_46, 0, lean_box(0));
lean_closure_set(x_46, 1, x_3);
lean_closure_set(x_46, 2, x_7);
lean_inc_ref(x_46);
lean_inc_ref(x_45);
lean_inc(x_7);
lean_ctor_set(x_18, 3, x_46);
lean_ctor_set(x_18, 2, x_45);
lean_ctor_set(x_18, 1, x_7);
lean_ctor_set(x_18, 0, x_24);
lean_inc(x_4);
lean_inc_ref(x_45);
x_47 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_47, 0, lean_box(0));
lean_closure_set(x_47, 1, x_45);
lean_closure_set(x_47, 2, x_4);
lean_inc_ref(x_47);
x_48 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_48, 0, x_18);
lean_ctor_set(x_48, 1, x_4);
lean_ctor_set(x_48, 2, x_21);
lean_ctor_set(x_48, 3, x_44);
lean_ctor_set(x_48, 4, x_47);
lean_inc(x_7);
lean_inc(x_3);
x_49 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_49, 0, x_3);
lean_ctor_set(x_49, 1, x_7);
lean_ctor_set(x_49, 2, x_46);
lean_inc(x_5);
x_50 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_50, 0, lean_box(0));
lean_closure_set(x_50, 1, x_49);
lean_closure_set(x_50, 2, x_5);
lean_inc(x_3);
lean_inc_ref(x_50);
lean_inc_ref(x_47);
lean_inc_ref(x_45);
x_51 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__1), 6, 4);
lean_closure_set(x_51, 0, x_45);
lean_closure_set(x_51, 1, x_47);
lean_closure_set(x_51, 2, x_50);
lean_closure_set(x_51, 3, x_3);
lean_inc(x_3);
lean_inc_ref(x_50);
lean_inc_ref(x_45);
x_52 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__2), 5, 3);
lean_closure_set(x_52, 0, x_45);
lean_closure_set(x_52, 1, x_50);
lean_closure_set(x_52, 2, x_3);
lean_inc(x_3);
lean_inc(x_7);
x_53 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_53, 0, lean_box(0));
lean_closure_set(x_53, 1, x_7);
lean_closure_set(x_53, 2, x_3);
lean_inc(x_5);
x_54 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_54, 0, lean_box(0));
lean_closure_set(x_54, 1, x_7);
lean_closure_set(x_54, 2, x_3);
lean_closure_set(x_54, 3, x_5);
lean_closure_set(x_54, 4, x_53);
lean_inc_ref(x_50);
lean_inc_ref(x_45);
x_55 = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(x_55, 0, lean_box(0));
lean_closure_set(x_55, 1, x_45);
lean_closure_set(x_55, 2, x_50);
lean_inc_ref(x_50);
x_56 = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(x_56, 0, lean_box(0));
lean_closure_set(x_56, 1, x_45);
lean_closure_set(x_56, 2, x_47);
lean_closure_set(x_56, 3, x_50);
x_57 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_57, 0, x_48);
lean_ctor_set(x_57, 1, x_5);
lean_ctor_set(x_57, 2, x_50);
lean_ctor_set(x_57, 3, x_54);
lean_ctor_set(x_57, 4, x_55);
lean_ctor_set(x_57, 5, x_56);
lean_ctor_set(x_57, 6, x_52);
lean_ctor_set(x_57, 7, x_51);
return x_57;
}
}
else
{
lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; 
x_58 = lean_ctor_get(x_18, 0);
x_59 = lean_ctor_get(x_18, 2);
lean_inc(x_59);
lean_inc(x_58);
lean_dec(x_18);
lean_inc(x_3);
lean_inc_ref(x_58);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_58);
lean_ctor_set(x_60, 1, x_3);
x_61 = lean_ctor_get(x_58, 0);
lean_inc(x_61);
x_62 = lean_ctor_get(x_58, 1);
lean_inc(x_62);
if (lean_is_exclusive(x_58)) {
 lean_ctor_release(x_58, 0);
 lean_ctor_release(x_58, 1);
 lean_ctor_release(x_58, 2);
 x_63 = x_58;
} else {
 lean_dec_ref(x_58);
 x_63 = lean_box(0);
}
lean_inc(x_4);
x_64 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_64, 0, x_6);
lean_closure_set(x_64, 1, x_2);
lean_closure_set(x_64, 2, x_4);
lean_inc(x_7);
x_65 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_65, 0, lean_box(0));
lean_closure_set(x_65, 1, x_7);
lean_closure_set(x_65, 2, x_62);
lean_closure_set(x_65, 3, x_61);
lean_inc(x_7);
lean_inc(x_3);
x_66 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_66, 0, lean_box(0));
lean_closure_set(x_66, 1, x_3);
lean_closure_set(x_66, 2, x_7);
lean_inc_ref(x_66);
lean_inc_ref(x_65);
lean_inc(x_7);
x_67 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_67, 0, x_60);
lean_ctor_set(x_67, 1, x_7);
lean_ctor_set(x_67, 2, x_65);
lean_ctor_set(x_67, 3, x_66);
lean_inc(x_4);
lean_inc_ref(x_65);
x_68 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_68, 0, lean_box(0));
lean_closure_set(x_68, 1, x_65);
lean_closure_set(x_68, 2, x_4);
lean_inc_ref(x_68);
x_69 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_69, 0, x_67);
lean_ctor_set(x_69, 1, x_4);
lean_ctor_set(x_69, 2, x_59);
lean_ctor_set(x_69, 3, x_64);
lean_ctor_set(x_69, 4, x_68);
lean_inc(x_7);
lean_inc(x_3);
if (lean_is_scalar(x_63)) {
 x_70 = lean_alloc_ctor(0, 3, 0);
} else {
 x_70 = x_63;
}
lean_ctor_set(x_70, 0, x_3);
lean_ctor_set(x_70, 1, x_7);
lean_ctor_set(x_70, 2, x_66);
lean_inc(x_5);
x_71 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_71, 0, lean_box(0));
lean_closure_set(x_71, 1, x_70);
lean_closure_set(x_71, 2, x_5);
lean_inc(x_3);
lean_inc_ref(x_71);
lean_inc_ref(x_68);
lean_inc_ref(x_65);
x_72 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__1), 6, 4);
lean_closure_set(x_72, 0, x_65);
lean_closure_set(x_72, 1, x_68);
lean_closure_set(x_72, 2, x_71);
lean_closure_set(x_72, 3, x_3);
lean_inc(x_3);
lean_inc_ref(x_71);
lean_inc_ref(x_65);
x_73 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__2), 5, 3);
lean_closure_set(x_73, 0, x_65);
lean_closure_set(x_73, 1, x_71);
lean_closure_set(x_73, 2, x_3);
lean_inc(x_3);
lean_inc(x_7);
x_74 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_74, 0, lean_box(0));
lean_closure_set(x_74, 1, x_7);
lean_closure_set(x_74, 2, x_3);
lean_inc(x_5);
x_75 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_75, 0, lean_box(0));
lean_closure_set(x_75, 1, x_7);
lean_closure_set(x_75, 2, x_3);
lean_closure_set(x_75, 3, x_5);
lean_closure_set(x_75, 4, x_74);
lean_inc_ref(x_71);
lean_inc_ref(x_65);
x_76 = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(x_76, 0, lean_box(0));
lean_closure_set(x_76, 1, x_65);
lean_closure_set(x_76, 2, x_71);
lean_inc_ref(x_71);
x_77 = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(x_77, 0, lean_box(0));
lean_closure_set(x_77, 1, x_65);
lean_closure_set(x_77, 2, x_68);
lean_closure_set(x_77, 3, x_71);
x_78 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_78, 0, x_69);
lean_ctor_set(x_78, 1, x_5);
lean_ctor_set(x_78, 2, x_71);
lean_ctor_set(x_78, 3, x_75);
lean_ctor_set(x_78, 4, x_76);
lean_ctor_set(x_78, 5, x_77);
lean_ctor_set(x_78, 6, x_73);
lean_ctor_set(x_78, 7, x_72);
return x_78;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; uint8_t x_8; 
lean_inc(x_5);
lean_inc(x_3);
lean_inc(x_1);
x_7 = lp_mathlib_AddGroup_ofLeftAxioms___redArg(x_1, x_3, x_5);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_7, 2);
x_11 = lean_ctor_get(x_7, 3);
lean_dec(x_11);
x_12 = lean_ctor_get(x_7, 1);
lean_dec(x_12);
lean_inc(x_2);
lean_inc_ref(x_9);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_9);
lean_ctor_set(x_13, 1, x_2);
x_14 = !lean_is_exclusive(x_9);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_15 = lean_ctor_get(x_9, 0);
x_16 = lean_ctor_get(x_9, 1);
x_17 = lean_ctor_get(x_9, 2);
lean_dec(x_17);
lean_inc(x_3);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_18, 0, x_5);
lean_closure_set(x_18, 1, x_1);
lean_closure_set(x_18, 2, x_3);
lean_inc(x_6);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_19, 0, lean_box(0));
lean_closure_set(x_19, 1, x_6);
lean_closure_set(x_19, 2, x_16);
lean_closure_set(x_19, 3, x_15);
lean_inc(x_6);
lean_inc(x_2);
x_20 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, x_2);
lean_closure_set(x_20, 2, x_6);
lean_inc_ref(x_20);
lean_inc_ref(x_19);
lean_inc(x_6);
lean_ctor_set(x_7, 3, x_20);
lean_ctor_set(x_7, 2, x_19);
lean_ctor_set(x_7, 1, x_6);
lean_ctor_set(x_7, 0, x_13);
lean_inc(x_3);
lean_inc_ref(x_19);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_19);
lean_closure_set(x_21, 2, x_3);
lean_inc_ref(x_21);
x_22 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_22, 0, x_7);
lean_ctor_set(x_22, 1, x_3);
lean_ctor_set(x_22, 2, x_10);
lean_ctor_set(x_22, 3, x_18);
lean_ctor_set(x_22, 4, x_21);
lean_inc(x_6);
lean_inc(x_2);
lean_ctor_set(x_9, 2, x_20);
lean_ctor_set(x_9, 1, x_6);
lean_ctor_set(x_9, 0, x_2);
lean_inc(x_4);
x_23 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_23, 0, lean_box(0));
lean_closure_set(x_23, 1, x_9);
lean_closure_set(x_23, 2, x_4);
lean_inc(x_2);
lean_inc_ref(x_23);
lean_inc_ref(x_21);
lean_inc_ref(x_19);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__1), 6, 4);
lean_closure_set(x_24, 0, x_19);
lean_closure_set(x_24, 1, x_21);
lean_closure_set(x_24, 2, x_23);
lean_closure_set(x_24, 3, x_2);
lean_inc(x_2);
lean_inc_ref(x_23);
lean_inc_ref(x_19);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__2), 5, 3);
lean_closure_set(x_25, 0, x_19);
lean_closure_set(x_25, 1, x_23);
lean_closure_set(x_25, 2, x_2);
lean_inc(x_2);
lean_inc(x_6);
x_26 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_26, 0, lean_box(0));
lean_closure_set(x_26, 1, x_6);
lean_closure_set(x_26, 2, x_2);
lean_inc(x_4);
x_27 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_27, 0, lean_box(0));
lean_closure_set(x_27, 1, x_6);
lean_closure_set(x_27, 2, x_2);
lean_closure_set(x_27, 3, x_4);
lean_closure_set(x_27, 4, x_26);
lean_inc_ref(x_23);
lean_inc_ref(x_19);
x_28 = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(x_28, 0, lean_box(0));
lean_closure_set(x_28, 1, x_19);
lean_closure_set(x_28, 2, x_23);
lean_inc_ref(x_23);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, x_19);
lean_closure_set(x_29, 2, x_21);
lean_closure_set(x_29, 3, x_23);
x_30 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_30, 0, x_22);
lean_ctor_set(x_30, 1, x_4);
lean_ctor_set(x_30, 2, x_23);
lean_ctor_set(x_30, 3, x_27);
lean_ctor_set(x_30, 4, x_28);
lean_ctor_set(x_30, 5, x_29);
lean_ctor_set(x_30, 6, x_25);
lean_ctor_set(x_30, 7, x_24);
return x_30;
}
else
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_31 = lean_ctor_get(x_9, 0);
x_32 = lean_ctor_get(x_9, 1);
lean_inc(x_32);
lean_inc(x_31);
lean_dec(x_9);
lean_inc(x_3);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_33, 0, x_5);
lean_closure_set(x_33, 1, x_1);
lean_closure_set(x_33, 2, x_3);
lean_inc(x_6);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_34, 0, lean_box(0));
lean_closure_set(x_34, 1, x_6);
lean_closure_set(x_34, 2, x_32);
lean_closure_set(x_34, 3, x_31);
lean_inc(x_6);
lean_inc(x_2);
x_35 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_35, 0, lean_box(0));
lean_closure_set(x_35, 1, x_2);
lean_closure_set(x_35, 2, x_6);
lean_inc_ref(x_35);
lean_inc_ref(x_34);
lean_inc(x_6);
lean_ctor_set(x_7, 3, x_35);
lean_ctor_set(x_7, 2, x_34);
lean_ctor_set(x_7, 1, x_6);
lean_ctor_set(x_7, 0, x_13);
lean_inc(x_3);
lean_inc_ref(x_34);
x_36 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_36, 0, lean_box(0));
lean_closure_set(x_36, 1, x_34);
lean_closure_set(x_36, 2, x_3);
lean_inc_ref(x_36);
x_37 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_37, 0, x_7);
lean_ctor_set(x_37, 1, x_3);
lean_ctor_set(x_37, 2, x_10);
lean_ctor_set(x_37, 3, x_33);
lean_ctor_set(x_37, 4, x_36);
lean_inc(x_6);
lean_inc(x_2);
x_38 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_38, 0, x_2);
lean_ctor_set(x_38, 1, x_6);
lean_ctor_set(x_38, 2, x_35);
lean_inc(x_4);
x_39 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_39, 0, lean_box(0));
lean_closure_set(x_39, 1, x_38);
lean_closure_set(x_39, 2, x_4);
lean_inc(x_2);
lean_inc_ref(x_39);
lean_inc_ref(x_36);
lean_inc_ref(x_34);
x_40 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__1), 6, 4);
lean_closure_set(x_40, 0, x_34);
lean_closure_set(x_40, 1, x_36);
lean_closure_set(x_40, 2, x_39);
lean_closure_set(x_40, 3, x_2);
lean_inc(x_2);
lean_inc_ref(x_39);
lean_inc_ref(x_34);
x_41 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__2), 5, 3);
lean_closure_set(x_41, 0, x_34);
lean_closure_set(x_41, 1, x_39);
lean_closure_set(x_41, 2, x_2);
lean_inc(x_2);
lean_inc(x_6);
x_42 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_42, 0, lean_box(0));
lean_closure_set(x_42, 1, x_6);
lean_closure_set(x_42, 2, x_2);
lean_inc(x_4);
x_43 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_43, 0, lean_box(0));
lean_closure_set(x_43, 1, x_6);
lean_closure_set(x_43, 2, x_2);
lean_closure_set(x_43, 3, x_4);
lean_closure_set(x_43, 4, x_42);
lean_inc_ref(x_39);
lean_inc_ref(x_34);
x_44 = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(x_44, 0, lean_box(0));
lean_closure_set(x_44, 1, x_34);
lean_closure_set(x_44, 2, x_39);
lean_inc_ref(x_39);
x_45 = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(x_45, 0, lean_box(0));
lean_closure_set(x_45, 1, x_34);
lean_closure_set(x_45, 2, x_36);
lean_closure_set(x_45, 3, x_39);
x_46 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_46, 0, x_37);
lean_ctor_set(x_46, 1, x_4);
lean_ctor_set(x_46, 2, x_39);
lean_ctor_set(x_46, 3, x_43);
lean_ctor_set(x_46, 4, x_44);
lean_ctor_set(x_46, 5, x_45);
lean_ctor_set(x_46, 6, x_41);
lean_ctor_set(x_46, 7, x_40);
return x_46;
}
}
else
{
lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; 
x_47 = lean_ctor_get(x_7, 0);
x_48 = lean_ctor_get(x_7, 2);
lean_inc(x_48);
lean_inc(x_47);
lean_dec(x_7);
lean_inc(x_2);
lean_inc_ref(x_47);
x_49 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_49, 0, x_47);
lean_ctor_set(x_49, 1, x_2);
x_50 = lean_ctor_get(x_47, 0);
lean_inc(x_50);
x_51 = lean_ctor_get(x_47, 1);
lean_inc(x_51);
if (lean_is_exclusive(x_47)) {
 lean_ctor_release(x_47, 0);
 lean_ctor_release(x_47, 1);
 lean_ctor_release(x_47, 2);
 x_52 = x_47;
} else {
 lean_dec_ref(x_47);
 x_52 = lean_box(0);
}
lean_inc(x_3);
x_53 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_53, 0, x_5);
lean_closure_set(x_53, 1, x_1);
lean_closure_set(x_53, 2, x_3);
lean_inc(x_6);
x_54 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_54, 0, lean_box(0));
lean_closure_set(x_54, 1, x_6);
lean_closure_set(x_54, 2, x_51);
lean_closure_set(x_54, 3, x_50);
lean_inc(x_6);
lean_inc(x_2);
x_55 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_55, 0, lean_box(0));
lean_closure_set(x_55, 1, x_2);
lean_closure_set(x_55, 2, x_6);
lean_inc_ref(x_55);
lean_inc_ref(x_54);
lean_inc(x_6);
x_56 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_56, 0, x_49);
lean_ctor_set(x_56, 1, x_6);
lean_ctor_set(x_56, 2, x_54);
lean_ctor_set(x_56, 3, x_55);
lean_inc(x_3);
lean_inc_ref(x_54);
x_57 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_57, 0, lean_box(0));
lean_closure_set(x_57, 1, x_54);
lean_closure_set(x_57, 2, x_3);
lean_inc_ref(x_57);
x_58 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_58, 0, x_56);
lean_ctor_set(x_58, 1, x_3);
lean_ctor_set(x_58, 2, x_48);
lean_ctor_set(x_58, 3, x_53);
lean_ctor_set(x_58, 4, x_57);
lean_inc(x_6);
lean_inc(x_2);
if (lean_is_scalar(x_52)) {
 x_59 = lean_alloc_ctor(0, 3, 0);
} else {
 x_59 = x_52;
}
lean_ctor_set(x_59, 0, x_2);
lean_ctor_set(x_59, 1, x_6);
lean_ctor_set(x_59, 2, x_55);
lean_inc(x_4);
x_60 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_60, 0, lean_box(0));
lean_closure_set(x_60, 1, x_59);
lean_closure_set(x_60, 2, x_4);
lean_inc(x_2);
lean_inc_ref(x_60);
lean_inc_ref(x_57);
lean_inc_ref(x_54);
x_61 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__1), 6, 4);
lean_closure_set(x_61, 0, x_54);
lean_closure_set(x_61, 1, x_57);
lean_closure_set(x_61, 2, x_60);
lean_closure_set(x_61, 3, x_2);
lean_inc(x_2);
lean_inc_ref(x_60);
lean_inc_ref(x_54);
x_62 = lean_alloc_closure((void*)(lp_mathlib_Field_ofMinimalAxioms___redArg___lam__2), 5, 3);
lean_closure_set(x_62, 0, x_54);
lean_closure_set(x_62, 1, x_60);
lean_closure_set(x_62, 2, x_2);
lean_inc(x_2);
lean_inc(x_6);
x_63 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_63, 0, lean_box(0));
lean_closure_set(x_63, 1, x_6);
lean_closure_set(x_63, 2, x_2);
lean_inc(x_4);
x_64 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_64, 0, lean_box(0));
lean_closure_set(x_64, 1, x_6);
lean_closure_set(x_64, 2, x_2);
lean_closure_set(x_64, 3, x_4);
lean_closure_set(x_64, 4, x_63);
lean_inc_ref(x_60);
lean_inc_ref(x_54);
x_65 = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(x_65, 0, lean_box(0));
lean_closure_set(x_65, 1, x_54);
lean_closure_set(x_65, 2, x_60);
lean_inc_ref(x_60);
x_66 = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(x_66, 0, lean_box(0));
lean_closure_set(x_66, 1, x_54);
lean_closure_set(x_66, 2, x_57);
lean_closure_set(x_66, 3, x_60);
x_67 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_67, 0, x_58);
lean_ctor_set(x_67, 1, x_4);
lean_ctor_set(x_67, 2, x_60);
lean_ctor_set(x_67, 3, x_64);
lean_ctor_set(x_67, 4, x_65);
lean_ctor_set(x_67, 5, x_66);
lean_ctor_set(x_67, 6, x_62);
lean_ctor_set(x_67, 7, x_61);
return x_67;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_ofMinimalAxioms___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_Field_ofMinimalAxioms(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
return x_18;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_MinimalAxioms(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_MinimalAxioms(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_MinimalAxioms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
