// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Preserves.Shapes.Equalizers
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Shapes.SplitCoequalizer public import Mathlib.CategoryTheory.Limits.Shapes.SplitEqualizer public import Mathlib.CategoryTheory.Limits.Preserves.Basic
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
lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cofork_of_u03c0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeForkEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCoforkEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
extern lean_object* lp_mathlib_CategoryTheory_Limits_walkingParallelPairHomCategory;
lean_object* lp_mathlib_CategoryTheory_Limits_parallelPair___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeForkEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Fork_of_u03b9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_diagramIsoParallelPair___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Fork_ext___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeInvEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cofork_ext___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCocone___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCoforkEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCone___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeForkEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; uint8_t x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; 
x_10 = lp_mathlib_CategoryTheory_Limits_walkingParallelPairHomCategory;
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_1);
x_11 = lp_mathlib_CategoryTheory_Limits_parallelPair___redArg(x_1, x_4, x_5, x_7, x_8);
lean_inc_ref(x_3);
lean_inc_ref(x_11);
x_12 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_11, x_3);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_12, 1);
lean_inc(x_14);
lean_inc_ref(x_12);
lean_inc_ref(x_2);
x_15 = lp_mathlib_CategoryTheory_Limits_diagramIsoParallelPair___redArg(x_2, x_12);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
x_17 = 0;
x_18 = lean_box(x_17);
lean_inc(x_13);
x_19 = lean_apply_1(x_13, x_18);
x_20 = 1;
x_21 = lean_box(x_20);
x_22 = lean_apply_1(x_13, x_21);
x_23 = lean_box(0);
x_24 = lean_box(x_17);
x_25 = lean_box(x_20);
lean_inc(x_14);
x_26 = lean_apply_3(x_14, x_24, x_25, x_23);
x_27 = lean_box(1);
x_28 = lean_box(x_17);
x_29 = lean_box(x_20);
x_30 = lean_apply_3(x_14, x_28, x_29, x_27);
lean_inc_ref(x_2);
x_31 = lp_mathlib_CategoryTheory_Limits_parallelPair___redArg(x_2, x_19, x_22, x_26, x_30);
lean_inc_ref(x_31);
lean_inc_ref(x_12);
lean_inc_ref(x_2);
x_32 = lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(x_2, x_12, x_31, x_16);
x_33 = lean_ctor_get(x_32, 0);
lean_inc(x_33);
lean_dec_ref(x_32);
x_34 = lean_ctor_get(x_3, 0);
lean_inc(x_34);
x_35 = lean_ctor_get(x_3, 1);
lean_inc(x_35);
lean_inc(x_9);
lean_inc(x_6);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_5);
lean_inc(x_4);
x_36 = lp_mathlib_CategoryTheory_Limits_Fork_of_u03b9___redArg(x_1, x_4, x_5, x_7, x_8, x_6, x_9);
x_37 = lp_mathlib_CategoryTheory_Functor_mapCone___redArg(x_3, x_11, x_36);
lean_inc_ref(x_37);
x_38 = lean_apply_1(x_33, x_37);
x_39 = lean_ctor_get(x_38, 0);
lean_inc(x_39);
lean_inc_ref(x_2);
x_40 = lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(x_10, x_2, x_12, x_31, x_15, x_37);
x_41 = lp_mathlib_Equiv_symm___redArg(x_40);
lean_inc(x_34);
lean_inc(x_4);
x_42 = lean_apply_1(x_34, x_4);
lean_inc(x_34);
lean_inc(x_5);
x_43 = lean_apply_1(x_34, x_5);
lean_inc(x_35);
lean_inc(x_5);
lean_inc(x_4);
x_44 = lean_apply_3(x_35, x_4, x_5, x_7);
lean_inc(x_35);
lean_inc(x_4);
x_45 = lean_apply_3(x_35, x_4, x_5, x_8);
lean_inc(x_6);
x_46 = lean_apply_1(x_34, x_6);
x_47 = lean_apply_3(x_35, x_6, x_4, x_9);
lean_inc_ref(x_2);
x_48 = lp_mathlib_CategoryTheory_Limits_Fork_of_u03b9___redArg(x_2, x_42, x_43, x_44, x_45, x_46, x_47);
lean_inc_ref(x_2);
x_49 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_39);
x_50 = lp_mathlib_CategoryTheory_Limits_Fork_ext___redArg(x_49);
x_51 = lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg(x_2, x_38, x_48, x_50);
x_52 = lp_mathlib_Equiv_trans___redArg(x_41, x_51);
return x_52;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeForkEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Limits_isLimitMapConeForkEquiv___redArg(x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCoforkEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; uint8_t x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; 
x_10 = lp_mathlib_CategoryTheory_Limits_walkingParallelPairHomCategory;
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_1);
x_11 = lp_mathlib_CategoryTheory_Limits_parallelPair___redArg(x_1, x_4, x_5, x_7, x_8);
lean_inc_ref(x_3);
lean_inc_ref(x_11);
x_12 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_11, x_3);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_12, 1);
lean_inc(x_14);
lean_inc_ref(x_12);
lean_inc_ref(x_2);
x_15 = lp_mathlib_CategoryTheory_Limits_diagramIsoParallelPair___redArg(x_2, x_12);
x_16 = lean_ctor_get(x_15, 1);
lean_inc(x_16);
x_17 = 0;
x_18 = lean_box(x_17);
lean_inc(x_13);
x_19 = lean_apply_1(x_13, x_18);
x_20 = 1;
x_21 = lean_box(x_20);
x_22 = lean_apply_1(x_13, x_21);
x_23 = lean_box(0);
x_24 = lean_box(x_17);
x_25 = lean_box(x_20);
lean_inc(x_14);
x_26 = lean_apply_3(x_14, x_24, x_25, x_23);
x_27 = lean_box(1);
x_28 = lean_box(x_17);
x_29 = lean_box(x_20);
x_30 = lean_apply_3(x_14, x_28, x_29, x_27);
lean_inc_ref(x_2);
x_31 = lp_mathlib_CategoryTheory_Limits_parallelPair___redArg(x_2, x_19, x_22, x_26, x_30);
lean_inc_ref(x_31);
lean_inc_ref(x_12);
lean_inc_ref(x_2);
x_32 = lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(x_2, x_12, x_31, x_16);
x_33 = lean_ctor_get(x_32, 0);
lean_inc(x_33);
lean_dec_ref(x_32);
x_34 = lean_ctor_get(x_3, 0);
lean_inc(x_34);
x_35 = lean_ctor_get(x_3, 1);
lean_inc(x_35);
lean_inc(x_9);
lean_inc(x_6);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_5);
lean_inc(x_4);
x_36 = lp_mathlib_CategoryTheory_Limits_Cofork_of_u03c0___redArg(x_1, x_4, x_5, x_7, x_8, x_6, x_9);
x_37 = lp_mathlib_CategoryTheory_Functor_mapCocone___redArg(x_3, x_11, x_36);
lean_inc_ref(x_37);
x_38 = lean_apply_1(x_33, x_37);
x_39 = lean_ctor_get(x_38, 0);
lean_inc(x_39);
lean_inc_ref(x_2);
x_40 = lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeInvEquiv___redArg(x_10, x_2, x_12, x_31, x_15, x_37);
x_41 = lp_mathlib_Equiv_symm___redArg(x_40);
lean_inc(x_34);
lean_inc(x_4);
x_42 = lean_apply_1(x_34, x_4);
lean_inc(x_34);
lean_inc(x_5);
x_43 = lean_apply_1(x_34, x_5);
lean_inc(x_35);
lean_inc(x_5);
lean_inc(x_4);
x_44 = lean_apply_3(x_35, x_4, x_5, x_7);
lean_inc(x_35);
lean_inc(x_5);
x_45 = lean_apply_3(x_35, x_4, x_5, x_8);
lean_inc(x_6);
x_46 = lean_apply_1(x_34, x_6);
x_47 = lean_apply_3(x_35, x_5, x_6, x_9);
lean_inc_ref(x_2);
x_48 = lp_mathlib_CategoryTheory_Limits_Cofork_of_u03c0___redArg(x_2, x_42, x_43, x_44, x_45, x_46, x_47);
lean_inc_ref(x_2);
x_49 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_39);
x_50 = lp_mathlib_CategoryTheory_Limits_Cofork_ext___redArg(x_49);
x_51 = lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg(x_2, x_38, x_48, x_50);
x_52 = lp_mathlib_Equiv_trans___redArg(x_41, x_51);
return x_52;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCoforkEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCoforkEquiv___redArg(x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_SplitCoequalizer(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_SplitEqualizer(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_Equalizers(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_SplitCoequalizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_SplitEqualizer(builtin);
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
