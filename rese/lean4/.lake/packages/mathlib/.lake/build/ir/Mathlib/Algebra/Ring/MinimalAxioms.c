// Lean compiler output
// Module: Mathlib.Algebra.Ring.MinimalAxioms
// Imports: public import Init public import Mathlib.Algebra.Ring.Defs public import Mathlib.Algebra.Group.Basic public import Mathlib.Algebra.Group.MinimalAxioms
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
LEAN_EXPORT lean_object* lp_mathlib_Ring_ofMinimalAxioms___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_ofMinimalAxioms(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_unaryCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddGroup_ofLeftAxioms___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Int_castDef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_ofMinimalAxioms(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_ofMinimalAxioms___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
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
LEAN_EXPORT lean_object* lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_ofMinimalAxioms(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; uint8_t x_16; 
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_2);
x_15 = lp_mathlib_AddGroup_ofLeftAxioms___redArg(x_2, x_4, x_5);
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_17 = lean_ctor_get(x_15, 0);
x_18 = lean_ctor_get(x_15, 2);
x_19 = lean_ctor_get(x_15, 3);
lean_dec(x_19);
x_20 = lean_ctor_get(x_15, 1);
lean_dec(x_20);
lean_inc(x_3);
lean_inc_ref(x_17);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_17);
lean_ctor_set(x_21, 1, x_3);
x_22 = lean_ctor_get(x_17, 0);
lean_inc(x_22);
x_23 = lean_ctor_get(x_17, 1);
lean_inc(x_23);
lean_dec_ref(x_17);
lean_inc(x_4);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_24, 0, x_5);
lean_closure_set(x_24, 1, x_2);
lean_closure_set(x_24, 2, x_4);
lean_inc(x_6);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, x_6);
lean_closure_set(x_25, 2, x_23);
lean_closure_set(x_25, 3, x_22);
lean_inc(x_6);
x_26 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_26, 0, lean_box(0));
lean_closure_set(x_26, 1, x_3);
lean_closure_set(x_26, 2, x_6);
lean_inc_ref(x_25);
lean_ctor_set(x_15, 3, x_26);
lean_ctor_set(x_15, 2, x_25);
lean_ctor_set(x_15, 1, x_6);
lean_ctor_set(x_15, 0, x_21);
lean_inc(x_4);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_27, 0, lean_box(0));
lean_closure_set(x_27, 1, x_25);
lean_closure_set(x_27, 2, x_4);
x_28 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_28, 0, x_15);
lean_ctor_set(x_28, 1, x_4);
lean_ctor_set(x_28, 2, x_18);
lean_ctor_set(x_28, 3, x_24);
lean_ctor_set(x_28, 4, x_27);
return x_28;
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_29 = lean_ctor_get(x_15, 0);
x_30 = lean_ctor_get(x_15, 2);
lean_inc(x_30);
lean_inc(x_29);
lean_dec(x_15);
lean_inc(x_3);
lean_inc_ref(x_29);
x_31 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_31, 0, x_29);
lean_ctor_set(x_31, 1, x_3);
x_32 = lean_ctor_get(x_29, 0);
lean_inc(x_32);
x_33 = lean_ctor_get(x_29, 1);
lean_inc(x_33);
lean_dec_ref(x_29);
lean_inc(x_4);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_34, 0, x_5);
lean_closure_set(x_34, 1, x_2);
lean_closure_set(x_34, 2, x_4);
lean_inc(x_6);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_35, 0, lean_box(0));
lean_closure_set(x_35, 1, x_6);
lean_closure_set(x_35, 2, x_33);
lean_closure_set(x_35, 3, x_32);
lean_inc(x_6);
x_36 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_36, 0, lean_box(0));
lean_closure_set(x_36, 1, x_3);
lean_closure_set(x_36, 2, x_6);
lean_inc_ref(x_35);
x_37 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_37, 0, x_31);
lean_ctor_set(x_37, 1, x_6);
lean_ctor_set(x_37, 2, x_35);
lean_ctor_set(x_37, 3, x_36);
lean_inc(x_4);
x_38 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_38, 0, lean_box(0));
lean_closure_set(x_38, 1, x_35);
lean_closure_set(x_38, 2, x_4);
x_39 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_39, 0, x_37);
lean_ctor_set(x_39, 1, x_4);
lean_ctor_set(x_39, 2, x_30);
lean_ctor_set(x_39, 3, x_34);
lean_ctor_set(x_39, 4, x_38);
return x_39;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_ofMinimalAxioms___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
lean_inc(x_4);
lean_inc(x_3);
lean_inc(x_1);
x_6 = lp_mathlib_AddGroup_ofLeftAxioms___redArg(x_1, x_3, x_4);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 2);
x_10 = lean_ctor_get(x_6, 3);
lean_dec(x_10);
x_11 = lean_ctor_get(x_6, 1);
lean_dec(x_11);
lean_inc(x_2);
lean_inc_ref(x_8);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_8);
lean_ctor_set(x_12, 1, x_2);
x_13 = lean_ctor_get(x_8, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_8, 1);
lean_inc(x_14);
lean_dec_ref(x_8);
lean_inc(x_3);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_15, 0, x_4);
lean_closure_set(x_15, 1, x_1);
lean_closure_set(x_15, 2, x_3);
lean_inc(x_5);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_16, 0, lean_box(0));
lean_closure_set(x_16, 1, x_5);
lean_closure_set(x_16, 2, x_14);
lean_closure_set(x_16, 3, x_13);
lean_inc(x_5);
x_17 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, x_2);
lean_closure_set(x_17, 2, x_5);
lean_inc_ref(x_16);
lean_ctor_set(x_6, 3, x_17);
lean_ctor_set(x_6, 2, x_16);
lean_ctor_set(x_6, 1, x_5);
lean_ctor_set(x_6, 0, x_12);
lean_inc(x_3);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, x_16);
lean_closure_set(x_18, 2, x_3);
x_19 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_19, 0, x_6);
lean_ctor_set(x_19, 1, x_3);
lean_ctor_set(x_19, 2, x_9);
lean_ctor_set(x_19, 3, x_15);
lean_ctor_set(x_19, 4, x_18);
return x_19;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_20 = lean_ctor_get(x_6, 0);
x_21 = lean_ctor_get(x_6, 2);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_6);
lean_inc(x_2);
lean_inc_ref(x_20);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_20);
lean_ctor_set(x_22, 1, x_2);
x_23 = lean_ctor_get(x_20, 0);
lean_inc(x_23);
x_24 = lean_ctor_get(x_20, 1);
lean_inc(x_24);
lean_dec_ref(x_20);
lean_inc(x_3);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_25, 0, x_4);
lean_closure_set(x_25, 1, x_1);
lean_closure_set(x_25, 2, x_3);
lean_inc(x_5);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_26, 0, lean_box(0));
lean_closure_set(x_26, 1, x_5);
lean_closure_set(x_26, 2, x_24);
lean_closure_set(x_26, 3, x_23);
lean_inc(x_5);
x_27 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_27, 0, lean_box(0));
lean_closure_set(x_27, 1, x_2);
lean_closure_set(x_27, 2, x_5);
lean_inc_ref(x_26);
x_28 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_28, 0, x_22);
lean_ctor_set(x_28, 1, x_5);
lean_ctor_set(x_28, 2, x_26);
lean_ctor_set(x_28, 3, x_27);
lean_inc(x_3);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, x_26);
lean_closure_set(x_29, 2, x_3);
x_30 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_30, 0, x_28);
lean_ctor_set(x_30, 1, x_3);
lean_ctor_set(x_30, 2, x_21);
lean_ctor_set(x_30, 3, x_25);
lean_ctor_set(x_30, 4, x_29);
return x_30;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_ofMinimalAxioms(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; uint8_t x_15; 
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_2);
x_14 = lp_mathlib_AddGroup_ofLeftAxioms___redArg(x_2, x_4, x_5);
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_16 = lean_ctor_get(x_14, 0);
x_17 = lean_ctor_get(x_14, 2);
x_18 = lean_ctor_get(x_14, 3);
lean_dec(x_18);
x_19 = lean_ctor_get(x_14, 1);
lean_dec(x_19);
lean_inc(x_3);
lean_inc_ref(x_16);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_16);
lean_ctor_set(x_20, 1, x_3);
x_21 = lean_ctor_get(x_16, 0);
lean_inc(x_21);
x_22 = lean_ctor_get(x_16, 1);
lean_inc(x_22);
lean_dec_ref(x_16);
lean_inc(x_4);
x_23 = lean_alloc_closure((void*)(lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_23, 0, x_5);
lean_closure_set(x_23, 1, x_2);
lean_closure_set(x_23, 2, x_4);
lean_inc(x_6);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_24, 0, lean_box(0));
lean_closure_set(x_24, 1, x_6);
lean_closure_set(x_24, 2, x_22);
lean_closure_set(x_24, 3, x_21);
lean_inc(x_6);
x_25 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, x_3);
lean_closure_set(x_25, 2, x_6);
lean_inc_ref(x_24);
lean_ctor_set(x_14, 3, x_25);
lean_ctor_set(x_14, 2, x_24);
lean_ctor_set(x_14, 1, x_6);
lean_ctor_set(x_14, 0, x_20);
lean_inc(x_4);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_26, 0, lean_box(0));
lean_closure_set(x_26, 1, x_24);
lean_closure_set(x_26, 2, x_4);
x_27 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_27, 0, x_14);
lean_ctor_set(x_27, 1, x_4);
lean_ctor_set(x_27, 2, x_17);
lean_ctor_set(x_27, 3, x_23);
lean_ctor_set(x_27, 4, x_26);
return x_27;
}
else
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_28 = lean_ctor_get(x_14, 0);
x_29 = lean_ctor_get(x_14, 2);
lean_inc(x_29);
lean_inc(x_28);
lean_dec(x_14);
lean_inc(x_3);
lean_inc_ref(x_28);
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_28);
lean_ctor_set(x_30, 1, x_3);
x_31 = lean_ctor_get(x_28, 0);
lean_inc(x_31);
x_32 = lean_ctor_get(x_28, 1);
lean_inc(x_32);
lean_dec_ref(x_28);
lean_inc(x_4);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_33, 0, x_5);
lean_closure_set(x_33, 1, x_2);
lean_closure_set(x_33, 2, x_4);
lean_inc(x_6);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_34, 0, lean_box(0));
lean_closure_set(x_34, 1, x_6);
lean_closure_set(x_34, 2, x_32);
lean_closure_set(x_34, 3, x_31);
lean_inc(x_6);
x_35 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_35, 0, lean_box(0));
lean_closure_set(x_35, 1, x_3);
lean_closure_set(x_35, 2, x_6);
lean_inc_ref(x_34);
x_36 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_36, 0, x_30);
lean_ctor_set(x_36, 1, x_6);
lean_ctor_set(x_36, 2, x_34);
lean_ctor_set(x_36, 3, x_35);
lean_inc(x_4);
x_37 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_37, 0, lean_box(0));
lean_closure_set(x_37, 1, x_34);
lean_closure_set(x_37, 2, x_4);
x_38 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_38, 0, x_36);
lean_ctor_set(x_38, 1, x_4);
lean_ctor_set(x_38, 2, x_29);
lean_ctor_set(x_38, 3, x_33);
lean_ctor_set(x_38, 4, x_37);
return x_38;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_ofMinimalAxioms___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
lean_inc(x_4);
lean_inc(x_3);
lean_inc(x_1);
x_6 = lp_mathlib_AddGroup_ofLeftAxioms___redArg(x_1, x_3, x_4);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 2);
x_10 = lean_ctor_get(x_6, 3);
lean_dec(x_10);
x_11 = lean_ctor_get(x_6, 1);
lean_dec(x_11);
lean_inc(x_2);
lean_inc_ref(x_8);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_8);
lean_ctor_set(x_12, 1, x_2);
x_13 = lean_ctor_get(x_8, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_8, 1);
lean_inc(x_14);
lean_dec_ref(x_8);
lean_inc(x_3);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_15, 0, x_4);
lean_closure_set(x_15, 1, x_1);
lean_closure_set(x_15, 2, x_3);
lean_inc(x_5);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_16, 0, lean_box(0));
lean_closure_set(x_16, 1, x_5);
lean_closure_set(x_16, 2, x_14);
lean_closure_set(x_16, 3, x_13);
lean_inc(x_5);
x_17 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, x_2);
lean_closure_set(x_17, 2, x_5);
lean_inc_ref(x_16);
lean_ctor_set(x_6, 3, x_17);
lean_ctor_set(x_6, 2, x_16);
lean_ctor_set(x_6, 1, x_5);
lean_ctor_set(x_6, 0, x_12);
lean_inc(x_3);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, x_16);
lean_closure_set(x_18, 2, x_3);
x_19 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_19, 0, x_6);
lean_ctor_set(x_19, 1, x_3);
lean_ctor_set(x_19, 2, x_9);
lean_ctor_set(x_19, 3, x_15);
lean_ctor_set(x_19, 4, x_18);
return x_19;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_20 = lean_ctor_get(x_6, 0);
x_21 = lean_ctor_get(x_6, 2);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_6);
lean_inc(x_2);
lean_inc_ref(x_20);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_20);
lean_ctor_set(x_22, 1, x_2);
x_23 = lean_ctor_get(x_20, 0);
lean_inc(x_23);
x_24 = lean_ctor_get(x_20, 1);
lean_inc(x_24);
lean_dec_ref(x_20);
lean_inc(x_3);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Ring_ofMinimalAxioms___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_25, 0, x_4);
lean_closure_set(x_25, 1, x_1);
lean_closure_set(x_25, 2, x_3);
lean_inc(x_5);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_26, 0, lean_box(0));
lean_closure_set(x_26, 1, x_5);
lean_closure_set(x_26, 2, x_24);
lean_closure_set(x_26, 3, x_23);
lean_inc(x_5);
x_27 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_27, 0, lean_box(0));
lean_closure_set(x_27, 1, x_2);
lean_closure_set(x_27, 2, x_5);
lean_inc_ref(x_26);
x_28 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_28, 0, x_22);
lean_ctor_set(x_28, 1, x_5);
lean_ctor_set(x_28, 2, x_26);
lean_ctor_set(x_28, 3, x_27);
lean_inc(x_3);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, x_26);
lean_closure_set(x_29, 2, x_3);
x_30 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_30, 0, x_28);
lean_ctor_set(x_30, 1, x_3);
lean_ctor_set(x_30, 2, x_21);
lean_ctor_set(x_30, 3, x_25);
lean_ctor_set(x_30, 4, x_29);
return x_30;
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_MinimalAxioms(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_MinimalAxioms(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_MinimalAxioms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
