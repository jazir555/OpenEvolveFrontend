// Lean compiler output
// Module: Mathlib.Algebra.Homology.TotalComplexShift
// Imports: public import Init public import Mathlib.Algebra.Homology.HomotopyCategory.Shift public import Mathlib.Algebra.Homology.TotalComplex
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
lean_object* lp_mathlib_HomologicalComplex_instCategory___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_shiftFunctor___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2082___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapHomologicalComplex___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081XXIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081XXIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081_u2082CommIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2082XXIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081_u2082CommIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2082XXIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_HomologicalComplex_instZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2082(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CochainComplex_instHasShiftInt___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_category___redArg(lean_object*);
lean_object* lp_mathlib_HomologicalComplex_instAddCommGroupHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_eqToIso___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_3);
x_5 = lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(x_3);
x_6 = lean_box(0);
lean_inc_ref(x_2);
x_7 = lp_mathlib_HomologicalComplex_instCategory___redArg(x_2, x_5, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_HomologicalComplex_instAddCommGroupHom___boxed), 7, 5);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, lean_box(0));
lean_closure_set(x_8, 2, x_2);
lean_closure_set(x_8, 3, x_3);
lean_closure_set(x_8, 4, x_6);
x_9 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_7, x_8);
x_10 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_9, x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_2);
x_4 = lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(x_2);
x_5 = lean_box(0);
lean_inc_ref(x_1);
x_6 = lp_mathlib_HomologicalComplex_instCategory___redArg(x_1, x_4, x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_HomologicalComplex_instAddCommGroupHom___boxed), 7, 5);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, x_1);
lean_closure_set(x_7, 3, x_2);
lean_closure_set(x_7, 4, x_5);
x_8 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_6, x_7);
x_9 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_8, x_3);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2082(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_2, x_3);
x_6 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_5, x_4);
x_7 = lp_mathlib_CategoryTheory_Functor_mapHomologicalComplex___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2082___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_1, x_2);
x_5 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_4, x_3);
x_6 = lp_mathlib_CategoryTheory_Functor_mapHomologicalComplex___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081XXIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
lean_inc_ref(x_2);
x_8 = lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(x_2);
x_9 = lean_box(0);
lean_inc_ref(x_1);
x_10 = lp_mathlib_HomologicalComplex_instCategory___redArg(x_1, x_8, x_9);
lean_inc_ref(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_HomologicalComplex_instAddCommGroupHom___boxed), 7, 5);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, lean_box(0));
lean_closure_set(x_11, 2, x_1);
lean_closure_set(x_11, 3, x_2);
lean_closure_set(x_11, 4, x_9);
x_12 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_10, x_11);
x_13 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_12, x_5);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
lean_inc_ref(x_3);
x_15 = lean_apply_1(x_14, x_3);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_apply_1(x_16, x_4);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lean_ctor_get(x_3, 0);
lean_inc(x_19);
lean_dec_ref(x_3);
x_20 = lean_apply_1(x_19, x_6);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
lean_inc(x_7);
x_22 = lean_apply_1(x_18, x_7);
x_23 = lean_apply_1(x_21, x_7);
x_24 = lp_mathlib_CategoryTheory_eqToIso___redArg(x_1, x_22, x_23);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081XXIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081XXIso___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2082XXIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
lean_inc_ref(x_1);
x_8 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_1, x_2);
x_9 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_8, x_6);
x_10 = lp_mathlib_CategoryTheory_Functor_mapHomologicalComplex___redArg(x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
lean_inc_ref(x_3);
x_12 = lean_apply_1(x_11, x_3);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc(x_4);
x_14 = lean_apply_1(x_13, x_4);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_3, 0);
lean_inc(x_16);
lean_dec_ref(x_3);
x_17 = lean_apply_1(x_16, x_4);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lean_apply_1(x_15, x_5);
x_20 = lean_apply_1(x_18, x_7);
x_21 = lp_mathlib_CategoryTheory_eqToIso___redArg(x_1, x_19, x_20);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2082XXIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2082XXIso___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081_u2082CommIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_inc_ref(x_2);
x_5 = lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(x_2);
x_6 = lean_box(0);
lean_inc(x_5);
lean_inc_ref(x_1);
x_7 = lp_mathlib_HomologicalComplex_instCategory___redArg(x_1, x_5, x_6);
lean_inc_ref(x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_HomologicalComplex_instZeroHom___boxed), 7, 5);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, lean_box(0));
lean_closure_set(x_8, 2, x_1);
lean_closure_set(x_8, 3, x_5);
lean_closure_set(x_8, 4, x_6);
lean_inc_ref(x_7);
x_9 = lp_mathlib_HomologicalComplex_instCategory___redArg(x_7, x_8, x_6);
x_10 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_9);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_11 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_1, x_2);
x_12 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_11, x_4);
x_13 = lp_mathlib_CategoryTheory_Functor_mapHomologicalComplex___redArg(x_12);
x_14 = lean_alloc_closure((void*)(lp_mathlib_HomologicalComplex_instAddCommGroupHom___boxed), 7, 5);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, lean_box(0));
lean_closure_set(x_14, 2, x_1);
lean_closure_set(x_14, 3, x_2);
lean_closure_set(x_14, 4, x_6);
x_15 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_7, x_14);
x_16 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_15, x_3);
x_17 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_13, x_16);
x_18 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_10, x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081_u2082CommIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_HomologicalComplex_u2082_shiftFunctor_u2081_u2082CommIso___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_Shift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_TotalComplex(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Homology_TotalComplexShift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_HomotopyCategory_Shift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_TotalComplex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
