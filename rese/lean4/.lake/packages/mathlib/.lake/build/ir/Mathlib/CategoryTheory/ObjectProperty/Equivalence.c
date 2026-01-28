// Lean compiler output
// Module: Mathlib.CategoryTheory.ObjectProperty.Equivalence
// Imports: public import Init public import Mathlib.CategoryTheory.ObjectProperty.CompleteLattice public import Mathlib.CategoryTheory.ObjectProperty.FullSubcategory public import Mathlib.CategoryTheory.Equivalence
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_topEquivalence___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_isoWhiskerLeft___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_ObjectProperty_00_u03b9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_id(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_FullyFaithful_preimageIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_category___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_ObjectProperty_lift___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_ObjectProperty_FullSubcategory_category___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_topEquivalence(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_FullyFaithful_whiskeringRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_topEquivalence___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_CategoryTheory_ObjectProperty_FullSubcategory_category___redArg(x_1);
x_3 = lp_mathlib_CategoryTheory_ObjectProperty_00_u03b9(lean_box(0), x_1, lean_box(0));
x_4 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_1);
x_5 = lp_mathlib_CategoryTheory_ObjectProperty_lift___redArg(x_4);
lean_inc_ref(x_2);
x_6 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_2);
x_7 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_2);
lean_dec_ref(x_2);
x_8 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_6, x_7);
x_9 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_1);
lean_inc_ref(x_3);
lean_inc_ref(x_5);
x_10 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_5, x_3);
x_11 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_9, x_10);
x_12 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_12, 0, x_3);
lean_ctor_set(x_12, 1, x_5);
lean_ctor_set(x_12, 2, x_8);
lean_ctor_set(x_12, 3, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_topEquivalence(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_ObjectProperty_topEquivalence___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
lean_inc_ref(x_1);
x_4 = lp_mathlib_CategoryTheory_ObjectProperty_FullSubcategory_category___redArg(x_1);
lean_inc_ref(x_2);
x_5 = lp_mathlib_CategoryTheory_ObjectProperty_FullSubcategory_category___redArg(x_2);
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_7 = lean_ctor_get(x_3, 0);
x_8 = lean_ctor_get(x_3, 1);
x_9 = lean_ctor_get(x_3, 2);
x_10 = lean_ctor_get(x_3, 3);
x_11 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory___redArg___lam__0___boxed), 3, 0);
x_12 = lp_mathlib_CategoryTheory_ObjectProperty_00_u03b9(lean_box(0), x_1, lean_box(0));
lean_inc_ref(x_7);
lean_inc_ref(x_12);
x_13 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_12, x_7);
x_14 = lp_mathlib_CategoryTheory_ObjectProperty_lift___redArg(x_13);
x_15 = lp_mathlib_CategoryTheory_ObjectProperty_00_u03b9(lean_box(0), x_2, lean_box(0));
lean_inc_ref(x_8);
lean_inc_ref(x_15);
x_16 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_15, x_8);
x_17 = lp_mathlib_CategoryTheory_ObjectProperty_lift___redArg(x_16);
x_18 = lp_mathlib_CategoryTheory_Functor_FullyFaithful_whiskeringRight___redArg(x_11);
x_19 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_4);
lean_inc_ref(x_17);
lean_inc_ref(x_14);
x_20 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_14, x_17);
x_21 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_1);
lean_inc_ref(x_8);
lean_inc_ref(x_7);
x_22 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_7, x_8);
x_23 = lp_mathlib_CategoryTheory_Functor_isoWhiskerLeft___redArg(x_4, x_1, x_1, x_12, x_21, x_22, x_9);
lean_dec_ref(x_1);
lean_dec_ref(x_4);
lean_inc(x_18);
x_24 = lp_mathlib_CategoryTheory_Functor_FullyFaithful_preimageIso___redArg(x_18, x_19, x_20, x_23);
lean_inc_ref(x_14);
lean_inc_ref(x_17);
x_25 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_17, x_14);
x_26 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_5);
x_27 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_8, x_7);
x_28 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_2);
x_29 = lp_mathlib_CategoryTheory_Functor_isoWhiskerLeft___redArg(x_5, x_2, x_2, x_15, x_27, x_28, x_10);
lean_dec_ref(x_2);
lean_dec_ref(x_5);
x_30 = lp_mathlib_CategoryTheory_Functor_FullyFaithful_preimageIso___redArg(x_18, x_25, x_26, x_29);
lean_ctor_set(x_3, 3, x_30);
lean_ctor_set(x_3, 2, x_24);
lean_ctor_set(x_3, 1, x_17);
lean_ctor_set(x_3, 0, x_14);
return x_3;
}
else
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; 
x_31 = lean_ctor_get(x_3, 0);
x_32 = lean_ctor_get(x_3, 1);
x_33 = lean_ctor_get(x_3, 2);
x_34 = lean_ctor_get(x_3, 3);
lean_inc(x_34);
lean_inc(x_33);
lean_inc(x_32);
lean_inc(x_31);
lean_dec(x_3);
x_35 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory___redArg___lam__0___boxed), 3, 0);
x_36 = lp_mathlib_CategoryTheory_ObjectProperty_00_u03b9(lean_box(0), x_1, lean_box(0));
lean_inc_ref(x_31);
lean_inc_ref(x_36);
x_37 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_36, x_31);
x_38 = lp_mathlib_CategoryTheory_ObjectProperty_lift___redArg(x_37);
x_39 = lp_mathlib_CategoryTheory_ObjectProperty_00_u03b9(lean_box(0), x_2, lean_box(0));
lean_inc_ref(x_32);
lean_inc_ref(x_39);
x_40 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_39, x_32);
x_41 = lp_mathlib_CategoryTheory_ObjectProperty_lift___redArg(x_40);
x_42 = lp_mathlib_CategoryTheory_Functor_FullyFaithful_whiskeringRight___redArg(x_35);
x_43 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_4);
lean_inc_ref(x_41);
lean_inc_ref(x_38);
x_44 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_38, x_41);
x_45 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_1);
lean_inc_ref(x_32);
lean_inc_ref(x_31);
x_46 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_31, x_32);
x_47 = lp_mathlib_CategoryTheory_Functor_isoWhiskerLeft___redArg(x_4, x_1, x_1, x_36, x_45, x_46, x_33);
lean_dec_ref(x_1);
lean_dec_ref(x_4);
lean_inc(x_42);
x_48 = lp_mathlib_CategoryTheory_Functor_FullyFaithful_preimageIso___redArg(x_42, x_43, x_44, x_47);
lean_inc_ref(x_38);
lean_inc_ref(x_41);
x_49 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_41, x_38);
x_50 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_5);
x_51 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_32, x_31);
x_52 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_2);
x_53 = lp_mathlib_CategoryTheory_Functor_isoWhiskerLeft___redArg(x_5, x_2, x_2, x_39, x_51, x_52, x_34);
lean_dec_ref(x_2);
lean_dec_ref(x_5);
x_54 = lp_mathlib_CategoryTheory_Functor_FullyFaithful_preimageIso___redArg(x_42, x_49, x_50, x_53);
x_55 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_55, 0, x_38);
lean_ctor_set(x_55, 1, x_41);
lean_ctor_set(x_55, 2, x_48);
lean_ctor_set(x_55, 3, x_54);
return x_55;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Equivalence_congrFullSubcategory___redArg(x_2, x_4, x_7);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_ObjectProperty_CompleteLattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_ObjectProperty_FullSubcategory(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Equivalence(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_ObjectProperty_Equivalence(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_ObjectProperty_CompleteLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_ObjectProperty_FullSubcategory(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Equivalence(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
