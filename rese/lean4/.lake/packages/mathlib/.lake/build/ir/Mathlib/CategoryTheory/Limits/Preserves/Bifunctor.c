// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Preserves.Bifunctor
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Fubini public import Mathlib.CategoryTheory.Functor.Currying public import Mathlib.CategoryTheory.Limits.HasLimits public import Mathlib.CategoryTheory.Limits.Preserves.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCone_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_const___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_whiskeringLeft_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCone_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_uncurry___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCocone_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCone_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCocone_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCocone_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCocone_u2082___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_13, 1);
lean_inc(x_15);
x_16 = lean_ctor_get(x_1, 2);
lean_inc(x_16);
lean_dec_ref(x_1);
x_17 = lean_ctor_get(x_2, 0);
lean_inc(x_17);
lean_dec_ref(x_2);
x_18 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_3);
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lean_apply_1(x_19, x_4);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
lean_inc(x_14);
x_22 = lean_apply_1(x_21, x_14);
lean_inc(x_22);
x_23 = lean_apply_1(x_5, x_22);
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
x_25 = lean_ctor_get(x_23, 1);
lean_inc(x_25);
lean_dec_ref(x_23);
x_26 = lean_ctor_get(x_6, 0);
lean_inc(x_26);
lean_dec_ref(x_6);
x_27 = lean_ctor_get(x_7, 0);
lean_inc(x_27);
lean_dec_ref(x_7);
x_28 = lean_ctor_get(x_8, 0);
lean_inc(x_28);
lean_dec_ref(x_8);
lean_inc_ref(x_13);
x_29 = lean_apply_1(x_17, x_13);
lean_inc(x_15);
x_30 = lean_apply_1(x_26, x_15);
lean_inc(x_30);
x_31 = lean_apply_1(x_24, x_30);
x_32 = lean_apply_1(x_27, x_13);
lean_inc(x_14);
x_33 = lean_apply_1(x_28, x_14);
x_34 = lean_apply_1(x_9, x_14);
lean_inc(x_30);
x_35 = lean_apply_4(x_10, x_33, x_22, x_34, x_30);
x_36 = lean_apply_1(x_11, x_15);
x_37 = lean_apply_3(x_25, x_30, x_12, x_36);
x_38 = lean_apply_5(x_16, x_29, x_31, x_32, x_35, x_37);
return x_38;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCocone_u2082___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; 
lean_inc_ref(x_5);
x_11 = lp_mathlib_CategoryTheory_Functor_uncurry___redArg(x_5);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_5);
lean_inc_ref(x_3);
x_13 = lp_mathlib_CategoryTheory_Functor_whiskeringLeft_u2082___redArg(x_1, x_2, x_3, x_4, x_5);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
lean_inc_ref(x_7);
x_15 = lean_apply_1(x_14, x_7);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
lean_inc_ref(x_8);
x_17 = lean_apply_1(x_16, x_8);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lean_ctor_get(x_6, 0);
lean_inc(x_19);
x_20 = lean_ctor_get(x_6, 1);
lean_inc(x_20);
x_21 = lean_ctor_get(x_9, 0);
lean_inc(x_21);
x_22 = lean_ctor_get(x_9, 1);
lean_inc(x_22);
lean_dec_ref(x_9);
lean_inc(x_19);
lean_inc(x_21);
x_23 = lean_apply_1(x_19, x_21);
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
lean_dec_ref(x_23);
x_25 = !lean_is_exclusive(x_10);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_26 = lean_ctor_get(x_10, 0);
x_27 = lean_ctor_get(x_10, 1);
lean_inc_ref(x_5);
x_28 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_5);
x_29 = lean_ctor_get(x_28, 0);
lean_inc(x_29);
lean_dec_ref(x_28);
x_30 = lean_apply_1(x_18, x_6);
x_31 = lean_apply_1(x_12, x_30);
lean_inc(x_26);
x_32 = lean_apply_1(x_24, x_26);
lean_inc(x_32);
x_33 = lean_apply_1(x_29, x_32);
x_34 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_mapCocone_u2082___redArg___lam__0), 13, 12);
lean_closure_set(x_34, 0, x_5);
lean_closure_set(x_34, 1, x_31);
lean_closure_set(x_34, 2, x_3);
lean_closure_set(x_34, 3, x_21);
lean_closure_set(x_34, 4, x_19);
lean_closure_set(x_34, 5, x_8);
lean_closure_set(x_34, 6, x_33);
lean_closure_set(x_34, 7, x_7);
lean_closure_set(x_34, 8, x_22);
lean_closure_set(x_34, 9, x_20);
lean_closure_set(x_34, 10, x_27);
lean_closure_set(x_34, 11, x_26);
lean_ctor_set(x_10, 1, x_34);
lean_ctor_set(x_10, 0, x_32);
return x_10;
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; 
x_35 = lean_ctor_get(x_10, 0);
x_36 = lean_ctor_get(x_10, 1);
lean_inc(x_36);
lean_inc(x_35);
lean_dec(x_10);
lean_inc_ref(x_5);
x_37 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_5);
x_38 = lean_ctor_get(x_37, 0);
lean_inc(x_38);
lean_dec_ref(x_37);
x_39 = lean_apply_1(x_18, x_6);
x_40 = lean_apply_1(x_12, x_39);
lean_inc(x_35);
x_41 = lean_apply_1(x_24, x_35);
lean_inc(x_41);
x_42 = lean_apply_1(x_38, x_41);
x_43 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_mapCocone_u2082___redArg___lam__0), 13, 12);
lean_closure_set(x_43, 0, x_5);
lean_closure_set(x_43, 1, x_40);
lean_closure_set(x_43, 2, x_3);
lean_closure_set(x_43, 3, x_21);
lean_closure_set(x_43, 4, x_19);
lean_closure_set(x_43, 5, x_8);
lean_closure_set(x_43, 6, x_42);
lean_closure_set(x_43, 7, x_7);
lean_closure_set(x_43, 8, x_22);
lean_closure_set(x_43, 9, x_20);
lean_closure_set(x_43, 10, x_36);
lean_closure_set(x_43, 11, x_35);
x_44 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_44, 0, x_41);
lean_ctor_set(x_44, 1, x_43);
return x_44;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCocone_u2082(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CategoryTheory_Functor_mapCocone_u2082___redArg(x_3, x_4, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCone_u2082___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_13, 1);
lean_inc(x_15);
x_16 = lean_ctor_get(x_1, 2);
lean_inc(x_16);
lean_dec_ref(x_1);
x_17 = lean_ctor_get(x_2, 0);
lean_inc(x_17);
lean_dec_ref(x_2);
x_18 = lean_ctor_get(x_3, 0);
lean_inc(x_18);
lean_dec_ref(x_3);
lean_inc(x_14);
x_19 = lean_apply_1(x_18, x_14);
lean_inc(x_19);
x_20 = lean_apply_1(x_4, x_19);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
x_22 = lean_ctor_get(x_20, 1);
lean_inc(x_22);
lean_dec_ref(x_20);
x_23 = lean_ctor_get(x_5, 0);
lean_inc(x_23);
lean_dec_ref(x_5);
x_24 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_6);
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
lean_dec_ref(x_24);
x_26 = lean_apply_1(x_25, x_7);
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
lean_dec_ref(x_26);
x_28 = lean_ctor_get(x_8, 0);
lean_inc(x_28);
lean_dec_ref(x_8);
lean_inc_ref(x_13);
x_29 = lean_apply_1(x_17, x_13);
lean_inc(x_9);
x_30 = lean_apply_1(x_21, x_9);
x_31 = lean_apply_1(x_23, x_13);
lean_inc(x_14);
x_32 = lean_apply_1(x_27, x_14);
x_33 = lean_apply_1(x_10, x_14);
lean_inc(x_9);
x_34 = lean_apply_4(x_11, x_32, x_19, x_33, x_9);
lean_inc(x_15);
x_35 = lean_apply_1(x_28, x_15);
x_36 = lean_apply_1(x_12, x_15);
x_37 = lean_apply_3(x_22, x_9, x_35, x_36);
x_38 = lean_apply_5(x_16, x_29, x_30, x_31, x_34, x_37);
return x_38;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCone_u2082___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; 
lean_inc_ref(x_5);
x_11 = lp_mathlib_CategoryTheory_Functor_uncurry___redArg(x_5);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_5);
lean_inc_ref(x_3);
x_13 = lp_mathlib_CategoryTheory_Functor_whiskeringLeft_u2082___redArg(x_1, x_2, x_3, x_4, x_5);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
lean_inc_ref(x_7);
x_15 = lean_apply_1(x_14, x_7);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
lean_inc_ref(x_8);
x_17 = lean_apply_1(x_16, x_8);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lean_ctor_get(x_6, 0);
lean_inc(x_19);
x_20 = lean_ctor_get(x_6, 1);
lean_inc(x_20);
x_21 = lean_ctor_get(x_9, 0);
lean_inc(x_21);
x_22 = lean_ctor_get(x_9, 1);
lean_inc(x_22);
lean_dec_ref(x_9);
lean_inc(x_19);
lean_inc(x_21);
x_23 = lean_apply_1(x_19, x_21);
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
lean_dec_ref(x_23);
x_25 = !lean_is_exclusive(x_10);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_26 = lean_ctor_get(x_10, 0);
x_27 = lean_ctor_get(x_10, 1);
lean_inc_ref(x_5);
x_28 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_5);
x_29 = lean_ctor_get(x_28, 0);
lean_inc(x_29);
lean_dec_ref(x_28);
x_30 = lean_apply_1(x_18, x_6);
x_31 = lean_apply_1(x_12, x_30);
lean_inc(x_26);
x_32 = lean_apply_1(x_24, x_26);
lean_inc(x_32);
x_33 = lean_apply_1(x_29, x_32);
x_34 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_mapCone_u2082___redArg___lam__0), 13, 12);
lean_closure_set(x_34, 0, x_5);
lean_closure_set(x_34, 1, x_33);
lean_closure_set(x_34, 2, x_7);
lean_closure_set(x_34, 3, x_19);
lean_closure_set(x_34, 4, x_31);
lean_closure_set(x_34, 5, x_3);
lean_closure_set(x_34, 6, x_21);
lean_closure_set(x_34, 7, x_8);
lean_closure_set(x_34, 8, x_26);
lean_closure_set(x_34, 9, x_22);
lean_closure_set(x_34, 10, x_20);
lean_closure_set(x_34, 11, x_27);
lean_ctor_set(x_10, 1, x_34);
lean_ctor_set(x_10, 0, x_32);
return x_10;
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; 
x_35 = lean_ctor_get(x_10, 0);
x_36 = lean_ctor_get(x_10, 1);
lean_inc(x_36);
lean_inc(x_35);
lean_dec(x_10);
lean_inc_ref(x_5);
x_37 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_5);
x_38 = lean_ctor_get(x_37, 0);
lean_inc(x_38);
lean_dec_ref(x_37);
x_39 = lean_apply_1(x_18, x_6);
x_40 = lean_apply_1(x_12, x_39);
lean_inc(x_35);
x_41 = lean_apply_1(x_24, x_35);
lean_inc(x_41);
x_42 = lean_apply_1(x_38, x_41);
x_43 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_mapCone_u2082___redArg___lam__0), 13, 12);
lean_closure_set(x_43, 0, x_5);
lean_closure_set(x_43, 1, x_42);
lean_closure_set(x_43, 2, x_7);
lean_closure_set(x_43, 3, x_19);
lean_closure_set(x_43, 4, x_40);
lean_closure_set(x_43, 5, x_3);
lean_closure_set(x_43, 6, x_21);
lean_closure_set(x_43, 7, x_8);
lean_closure_set(x_43, 8, x_35);
lean_closure_set(x_43, 9, x_22);
lean_closure_set(x_43, 10, x_20);
lean_closure_set(x_43, 11, x_36);
x_44 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_44, 0, x_41);
lean_ctor_set(x_44, 1, x_43);
return x_44;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_mapCone_u2082(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CategoryTheory_Functor_mapCone_u2082___redArg(x_3, x_4, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_16;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Fubini(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Functor_Currying(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_HasLimits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Bifunctor(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Fubini(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Functor_Currying(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_HasLimits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
