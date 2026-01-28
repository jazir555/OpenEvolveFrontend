// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Preserves.Shapes.BinaryProducts
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Shapes.BinaryProducts public import Mathlib.CategoryTheory.Limits.Preserves.Basic
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
static lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg___closed__0;
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_diagramIsoPair___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_BinaryFan_mk___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeBinaryCofanEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_discreteCategory(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_pair___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_ext___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_BinaryCofan_mk___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeBinaryCofanEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_ext___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCocone___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCone___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_CategoryTheory_discreteCategory(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; lean_object* x_16; lean_object* x_17; uint8_t x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_9 = lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg___closed__0;
lean_inc(x_6);
lean_inc(x_5);
x_10 = lp_mathlib_CategoryTheory_Limits_pair___redArg(x_1, x_5, x_6);
lean_inc_ref(x_3);
lean_inc_ref(x_10);
x_11 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_10, x_3);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc_ref(x_2);
x_13 = lp_mathlib_CategoryTheory_Limits_diagramIsoPair___redArg(x_2, x_11);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = 0;
x_16 = lean_box(x_15);
lean_inc(x_12);
x_17 = lean_apply_1(x_12, x_16);
x_18 = 1;
x_19 = lean_box(x_18);
x_20 = lean_apply_1(x_12, x_19);
lean_inc_ref(x_2);
x_21 = lp_mathlib_CategoryTheory_Limits_pair___redArg(x_2, x_17, x_20);
lean_inc_ref(x_21);
lean_inc_ref(x_11);
lean_inc_ref(x_2);
x_22 = lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(x_2, x_11, x_21, x_14);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec_ref(x_22);
x_24 = lean_ctor_get(x_3, 0);
lean_inc(x_24);
x_25 = lean_ctor_get(x_3, 1);
lean_inc(x_25);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_4);
x_26 = lp_mathlib_CategoryTheory_Limits_BinaryFan_mk___redArg(x_4, x_7, x_8);
x_27 = lp_mathlib_CategoryTheory_Functor_mapCone___redArg(x_3, x_10, x_26);
lean_inc_ref(x_27);
x_28 = lean_apply_1(x_23, x_27);
x_29 = lean_ctor_get(x_28, 0);
lean_inc(x_29);
lean_inc_ref(x_2);
x_30 = lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(x_9, x_2, x_11, x_21, x_13, x_27);
x_31 = lp_mathlib_Equiv_symm___redArg(x_30);
lean_inc(x_4);
x_32 = lean_apply_1(x_24, x_4);
lean_inc(x_25);
lean_inc(x_4);
x_33 = lean_apply_3(x_25, x_4, x_5, x_7);
x_34 = lean_apply_3(x_25, x_4, x_6, x_8);
x_35 = lp_mathlib_CategoryTheory_Limits_BinaryFan_mk___redArg(x_32, x_33, x_34);
lean_inc_ref(x_2);
x_36 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_29);
x_37 = lp_mathlib_CategoryTheory_Limits_Cones_ext___redArg(x_36);
x_38 = lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg(x_2, x_28, x_35, x_37);
x_39 = lp_mathlib_Equiv_trans___redArg(x_31, x_38);
return x_39;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg(x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeBinaryCofanEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; uint8_t x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_9 = lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg___closed__0;
lean_inc(x_6);
lean_inc(x_5);
x_10 = lp_mathlib_CategoryTheory_Limits_pair___redArg(x_1, x_5, x_6);
lean_inc_ref(x_3);
lean_inc_ref(x_10);
x_11 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_10, x_3);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc_ref(x_2);
x_13 = lp_mathlib_CategoryTheory_Limits_diagramIsoPair___redArg(x_2, x_11);
x_14 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_13);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
x_16 = 0;
x_17 = lean_box(x_16);
lean_inc(x_12);
x_18 = lean_apply_1(x_12, x_17);
x_19 = 1;
x_20 = lean_box(x_19);
x_21 = lean_apply_1(x_12, x_20);
lean_inc_ref(x_2);
x_22 = lp_mathlib_CategoryTheory_Limits_pair___redArg(x_2, x_18, x_21);
lean_inc_ref(x_22);
lean_inc_ref(x_11);
lean_inc_ref(x_2);
x_23 = lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(x_2, x_11, x_22, x_15);
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
lean_dec_ref(x_23);
x_25 = lean_ctor_get(x_3, 0);
lean_inc(x_25);
x_26 = lean_ctor_get(x_3, 1);
lean_inc(x_26);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_4);
x_27 = lp_mathlib_CategoryTheory_Limits_BinaryCofan_mk___redArg(x_4, x_7, x_8);
x_28 = lp_mathlib_CategoryTheory_Functor_mapCocone___redArg(x_3, x_10, x_27);
lean_inc_ref(x_28);
x_29 = lean_apply_1(x_24, x_28);
x_30 = lean_ctor_get(x_29, 0);
lean_inc(x_30);
lean_inc_ref(x_2);
x_31 = lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(x_9, x_2, x_22, x_11, x_14, x_28);
x_32 = lp_mathlib_Equiv_symm___redArg(x_31);
lean_inc(x_4);
x_33 = lean_apply_1(x_25, x_4);
lean_inc(x_26);
lean_inc(x_4);
x_34 = lean_apply_3(x_26, x_5, x_4, x_7);
x_35 = lean_apply_3(x_26, x_6, x_4, x_8);
x_36 = lp_mathlib_CategoryTheory_Limits_BinaryCofan_mk___redArg(x_33, x_34, x_35);
lean_inc_ref(x_2);
x_37 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_30);
x_38 = lp_mathlib_CategoryTheory_Limits_Cocones_ext___redArg(x_37);
x_39 = lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg(x_2, x_29, x_36, x_38);
x_40 = lp_mathlib_Equiv_trans___redArg(x_32, x_39);
return x_40;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeBinaryCofanEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeBinaryCofanEquiv___redArg(x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryProducts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_BinaryProducts(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryProducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg___closed__0 = _init_lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_isLimitMapConeBinaryFanEquiv___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
