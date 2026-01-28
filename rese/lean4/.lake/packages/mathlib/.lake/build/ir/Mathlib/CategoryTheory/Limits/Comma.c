// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.Comma
// Imports: public import Init public import Mathlib.CategoryTheory.Comma.Arrow public import Mathlib.CategoryTheory.Comma.Over.Basic public import Mathlib.CategoryTheory.Limits.Constructions.EpiMono public import Mathlib.CategoryTheory.Limits.Creates public import Mathlib.CategoryTheory.Limits.Unit
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg___closed__0;
lean_object* lp_mathlib_CategoryTheory_Comma_fst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_colimitAuxiliaryCocone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Comma_natTrans___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_colimitAuxiliaryCocone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_colimitAuxiliaryCocone___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Comma_snd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_colimitAuxiliaryCocone___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCocone___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCone___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Comma_natTrans___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_8 = lp_mathlib_CategoryTheory_Comma_fst(lean_box(0), x_1, lean_box(0), x_2, lean_box(0), x_3, x_4, x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_8);
x_9 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_8, x_4);
lean_inc_ref(x_6);
x_10 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_6, x_9);
x_11 = lp_mathlib_CategoryTheory_Comma_snd(lean_box(0), x_1, lean_box(0), x_2, lean_box(0), x_3, x_4, x_5);
x_12 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_11, x_5);
lean_inc_ref(x_6);
x_13 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_6, x_12);
x_14 = lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg___closed__0;
lean_inc_ref(x_6);
x_15 = lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(x_6, x_14);
x_16 = lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(x_3, x_10, x_13, x_15);
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_6, x_8);
x_19 = lp_mathlib_CategoryTheory_Functor_mapCone___redArg(x_4, x_18, x_7);
x_20 = lean_apply_1(x_17, x_19);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg(x_4, x_6, x_8, x_9, x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_colimitAuxiliaryCocone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_8 = lp_mathlib_CategoryTheory_Comma_snd(lean_box(0), x_1, lean_box(0), x_2, lean_box(0), x_3, x_4, x_5);
lean_inc_ref(x_5);
lean_inc_ref(x_8);
x_9 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_8, x_5);
lean_inc_ref(x_6);
x_10 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_6, x_9);
x_11 = lp_mathlib_CategoryTheory_Comma_fst(lean_box(0), x_1, lean_box(0), x_2, lean_box(0), x_3, x_4, x_5);
x_12 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_11, x_4);
lean_inc_ref(x_6);
x_13 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_6, x_12);
x_14 = lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg___closed__0;
lean_inc_ref(x_6);
x_15 = lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(x_6, x_14);
x_16 = lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(x_3, x_10, x_13, x_15);
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_6, x_8);
x_19 = lp_mathlib_CategoryTheory_Functor_mapCocone___redArg(x_5, x_18, x_7);
x_20 = lean_apply_1(x_17, x_19);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_colimitAuxiliaryCocone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Comma_colimitAuxiliaryCocone___redArg(x_4, x_6, x_8, x_9, x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_colimitAuxiliaryCocone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Comma_colimitAuxiliaryCocone(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Comma_colimitAuxiliaryCocone___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Comma_colimitAuxiliaryCocone___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Comma_Arrow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Comma_Over_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_EpiMono(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Creates(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Unit(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Comma(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Comma_Arrow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Comma_Over_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_EpiMono(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Creates(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Unit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg___closed__0 = _init_lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_Comma_limitAuxiliaryCone___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
