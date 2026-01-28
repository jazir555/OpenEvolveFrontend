// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Preserves.Shapes.Products
// Imports: public import Init public import Mathlib.CategoryTheory.Limits.Shapes.Products public import Mathlib.CategoryTheory.Limits.Preserves.Basic
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
lean_object* lp_mathlib_CategoryTheory_Limits_Fan_mk___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cofan_mk___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCofanMkEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_discreteCategory(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCofanMkEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_ext___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Discrete_functor___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_ext___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCofanMkEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCocone___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCone___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_2, x_4);
x_7 = lean_apply_1(x_5, x_6);
x_8 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_3, x_7);
return x_8;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_CategoryTheory_discreteCategory(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_1(x_2, x_3);
x_6 = lean_apply_1(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc(x_5);
x_6 = lean_apply_1(x_1, x_5);
x_7 = lean_apply_1(x_2, x_5);
x_8 = lean_apply_3(x_3, x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
lean_inc_ref(x_2);
lean_inc(x_4);
lean_inc_ref(x_3);
x_7 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___lam__0), 4, 3);
lean_closure_set(x_7, 0, x_3);
lean_closure_set(x_7, 1, x_4);
lean_closure_set(x_7, 2, x_2);
x_8 = lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___closed__0;
x_9 = lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(x_7);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_inc(x_4);
lean_inc_ref(x_3);
x_11 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___lam__1), 3, 2);
lean_closure_set(x_11, 0, x_3);
lean_closure_set(x_11, 1, x_4);
lean_inc(x_4);
x_12 = lp_mathlib_CategoryTheory_Discrete_functor___redArg(x_1, x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_12);
x_13 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_12, x_3);
lean_inc_ref(x_2);
x_14 = lp_mathlib_CategoryTheory_Discrete_functor___redArg(x_2, x_11);
lean_inc_ref(x_14);
lean_inc_ref(x_13);
lean_inc_ref(x_2);
x_15 = lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(x_2, x_13, x_14, x_10);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_ctor_get(x_3, 0);
lean_inc(x_17);
x_18 = lean_ctor_get(x_3, 1);
lean_inc(x_18);
lean_inc(x_6);
lean_inc(x_5);
x_19 = lp_mathlib_CategoryTheory_Limits_Fan_mk___redArg(x_5, x_6);
x_20 = lp_mathlib_CategoryTheory_Functor_mapCone___redArg(x_3, x_12, x_19);
lean_inc_ref(x_20);
x_21 = lean_apply_1(x_16, x_20);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
lean_inc(x_5);
x_23 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___lam__2), 5, 4);
lean_closure_set(x_23, 0, x_4);
lean_closure_set(x_23, 1, x_6);
lean_closure_set(x_23, 2, x_18);
lean_closure_set(x_23, 3, x_5);
lean_inc_ref(x_2);
x_24 = lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(x_8, x_2, x_13, x_14, x_9, x_20);
x_25 = lp_mathlib_Equiv_symm___redArg(x_24);
x_26 = lean_apply_1(x_17, x_5);
x_27 = lp_mathlib_CategoryTheory_Limits_Fan_mk___redArg(x_26, x_23);
lean_inc_ref(x_2);
x_28 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_22);
x_29 = lp_mathlib_CategoryTheory_Limits_Cones_ext___redArg(x_28);
x_30 = lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg(x_2, x_21, x_27, x_29);
x_31 = lp_mathlib_Equiv_trans___redArg(x_25, x_30);
return x_31;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg(x_2, x_4, x_5, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCofanMkEquiv___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc(x_5);
x_6 = lean_apply_1(x_1, x_5);
x_7 = lean_apply_1(x_2, x_5);
x_8 = lean_apply_3(x_3, x_6, x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCofanMkEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
lean_inc_ref(x_2);
lean_inc(x_4);
lean_inc_ref(x_3);
x_7 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___lam__0), 4, 3);
lean_closure_set(x_7, 0, x_3);
lean_closure_set(x_7, 1, x_4);
lean_closure_set(x_7, 2, x_2);
x_8 = lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___closed__0;
x_9 = lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(x_7);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_inc(x_4);
lean_inc_ref(x_3);
x_11 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___lam__1), 3, 2);
lean_closure_set(x_11, 0, x_3);
lean_closure_set(x_11, 1, x_4);
lean_inc_ref(x_2);
x_12 = lp_mathlib_CategoryTheory_Discrete_functor___redArg(x_2, x_11);
lean_inc(x_4);
x_13 = lp_mathlib_CategoryTheory_Discrete_functor___redArg(x_1, x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_13);
x_14 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_13, x_3);
lean_inc_ref(x_12);
lean_inc_ref(x_14);
lean_inc_ref(x_2);
x_15 = lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(x_2, x_14, x_12, x_10);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_ctor_get(x_3, 0);
lean_inc(x_17);
x_18 = lean_ctor_get(x_3, 1);
lean_inc(x_18);
lean_inc(x_6);
lean_inc(x_5);
x_19 = lp_mathlib_CategoryTheory_Limits_Cofan_mk___redArg(x_5, x_6);
x_20 = lp_mathlib_CategoryTheory_Functor_mapCocone___redArg(x_3, x_13, x_19);
lean_inc_ref(x_20);
x_21 = lean_apply_1(x_16, x_20);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
lean_inc(x_5);
x_23 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCofanMkEquiv___redArg___lam__2), 5, 4);
lean_closure_set(x_23, 0, x_4);
lean_closure_set(x_23, 1, x_6);
lean_closure_set(x_23, 2, x_18);
lean_closure_set(x_23, 3, x_5);
lean_inc_ref(x_2);
x_24 = lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(x_8, x_2, x_12, x_14, x_9, x_20);
x_25 = lp_mathlib_Equiv_symm___redArg(x_24);
x_26 = lean_apply_1(x_17, x_5);
x_27 = lp_mathlib_CategoryTheory_Limits_Cofan_mk___redArg(x_26, x_23);
lean_inc_ref(x_2);
x_28 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_22);
x_29 = lp_mathlib_CategoryTheory_Limits_Cocones_ext___redArg(x_28);
x_30 = lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg(x_2, x_21, x_27, x_29);
x_31 = lp_mathlib_Equiv_trans___redArg(x_25, x_30);
return x_31;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCofanMkEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_isColimitMapCoconeCofanMkEquiv___redArg(x_2, x_4, x_5, x_7, x_8, x_9);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Products(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Shapes_Products(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Products(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___closed__0 = _init_lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_isLimitMapConeFanMkEquiv___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
