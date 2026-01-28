// Lean compiler output
// Module: Mathlib.CategoryTheory.Sites.PreservesSheafification
// Imports: public import Init public import Mathlib.CategoryTheory.Sites.Localization public import Mathlib.CategoryTheory.Sites.CompatibleSheafification public import Mathlib.CategoryTheory.Sites.Whiskering public import Mathlib.CategoryTheory.Sites.Sheafification
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
lean_object* lp_mathlib_CategoryTheory_Adjunction_homEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_sheafComposeNatTrans___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_whiskeringRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_sheafComposeNatTrans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Sheaf_instCategorySheaf___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_sheafCompose___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_category___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_sheafComposeNatTrans___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_whiskerRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_CategoryStruct_opposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_sheafComposeNatTrans___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_sheafToPresheaf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_sheafComposeNatTrans___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_17 = lean_ctor_get(x_1, 0);
lean_inc(x_17);
lean_dec_ref(x_1);
x_18 = lean_ctor_get(x_2, 0);
lean_inc(x_18);
lean_dec_ref(x_2);
x_19 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_3);
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lp_mathlib_CategoryTheory_sheafToPresheaf(lean_box(0), x_4, x_5, lean_box(0), x_6);
x_22 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_7, x_21);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec_ref(x_22);
x_24 = lean_ctor_get(x_8, 0);
lean_inc(x_24);
lean_dec_ref(x_8);
x_25 = lp_mathlib_CategoryTheory_sheafToPresheaf(lean_box(0), x_4, x_5, lean_box(0), x_9);
lean_inc_ref(x_15);
x_26 = lean_apply_1(x_17, x_15);
lean_inc_ref(x_15);
x_27 = lean_apply_1(x_18, x_15);
x_28 = lp_mathlib_CategoryTheory_Adjunction_homEquiv___redArg(x_10, x_11, x_12, x_25, x_13, x_26, x_27);
x_29 = lp_mathlib_Equiv_symm___redArg(x_28);
x_30 = lean_ctor_get(x_29, 0);
lean_inc(x_30);
lean_dec_ref(x_29);
lean_inc_ref(x_15);
x_31 = lean_apply_1(x_20, x_15);
lean_inc_ref(x_15);
x_32 = lean_apply_1(x_23, x_15);
x_33 = lean_apply_1(x_24, x_15);
x_34 = lp_mathlib_CategoryTheory_Functor_whiskerRight___redArg(x_31, x_32, x_33, x_14);
x_35 = lean_apply_2(x_30, x_34, x_16);
return x_35;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_sheafComposeNatTrans___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_CategoryTheory_sheafComposeNatTrans___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
lean_dec_ref(x_9);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_sheafComposeNatTrans___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
lean_inc_ref(x_1);
x_10 = lp_mathlib_CategoryTheory_CategoryStruct_opposite___redArg(x_1);
lean_inc_ref(x_3);
x_11 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_3);
lean_inc_ref(x_4);
x_12 = lp_mathlib_CategoryTheory_Sheaf_instCategorySheaf___redArg(x_4);
lean_inc_ref(x_4);
x_13 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_4);
x_14 = lp_mathlib_CategoryTheory_Functor_whiskeringRight(lean_box(0), x_10, lean_box(0), x_3, lean_box(0), x_4);
lean_dec_ref(x_10);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc_ref(x_5);
x_16 = lean_apply_1(x_15, x_5);
lean_inc_ref(x_5);
x_17 = lp_mathlib_CategoryTheory_sheafCompose___redArg(x_5);
lean_inc_ref(x_6);
x_18 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_6, x_17);
x_19 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_sheafComposeNatTrans___redArg___lam__0___boxed), 16, 14);
lean_closure_set(x_19, 0, x_16);
lean_closure_set(x_19, 1, x_18);
lean_closure_set(x_19, 2, x_11);
lean_closure_set(x_19, 3, x_1);
lean_closure_set(x_19, 4, x_2);
lean_closure_set(x_19, 5, x_3);
lean_closure_set(x_19, 6, x_6);
lean_closure_set(x_19, 7, x_7);
lean_closure_set(x_19, 8, x_4);
lean_closure_set(x_19, 9, x_13);
lean_closure_set(x_19, 10, x_12);
lean_closure_set(x_19, 11, x_8);
lean_closure_set(x_19, 12, x_9);
lean_closure_set(x_19, 13, x_5);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_sheafComposeNatTrans(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_CategoryTheory_sheafComposeNatTrans___redArg(x_2, x_3, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_14;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_Localization(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_CompatibleSheafification(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_Whiskering(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_Sheafification(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_PreservesSheafification(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Sites_Localization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Sites_CompatibleSheafification(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Sites_Whiskering(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Sites_Sheafification(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
