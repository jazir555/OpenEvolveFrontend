// Lean compiler output
// Module: Mathlib.Algebra.Homology.BifunctorShift
// Imports: public import Init public import Mathlib.Algebra.Homology.Bifunctor public import Mathlib.Algebra.Homology.TotalComplexShift public import Mathlib.CategoryTheory.Shift.CommShiftTwo
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
lean_object* lp_mathlib_CategoryTheory_Functor_mapBifunctorHomologicalComplex___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2081Iso___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_shiftFunctor___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_HomologicalComplex_Hom_isoOfComponents___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2082Iso___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2082Iso___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2082Iso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2082Iso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CochainComplex_instHasShiftInt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2081Iso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2081Iso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2081Iso___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_1(x_4, x_3);
x_6 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2081Iso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_11 = lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(x_6);
x_12 = lean_box(0);
lean_inc(x_11);
lean_inc_ref(x_3);
x_13 = lp_mathlib_HomologicalComplex_instCategory___redArg(x_3, x_11, x_12);
lean_inc_ref(x_4);
x_14 = lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(x_4);
lean_inc_ref(x_1);
x_15 = lp_mathlib_CategoryTheory_Functor_mapBifunctorHomologicalComplex___redArg(x_1, x_2, x_3, x_14, x_5, x_11, x_9, x_12, x_12);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_1, x_4);
x_18 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_17, x_10);
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lean_apply_1(x_19, x_7);
x_21 = lean_apply_1(x_16, x_20);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
lean_dec_ref(x_21);
x_23 = lean_apply_1(x_22, x_8);
x_24 = lean_alloc_closure((void*)(lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2081Iso___redArg___lam__0), 3, 2);
lean_closure_set(x_24, 0, x_23);
lean_closure_set(x_24, 1, x_13);
x_25 = lp_mathlib_HomologicalComplex_Hom_isoOfComponents___redArg(x_24);
return x_25;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2081Iso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2081Iso___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2082Iso___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_1(x_4, x_3);
x_6 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2082Iso___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_1(x_4, x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2082Iso___redArg___lam__0), 3, 2);
lean_closure_set(x_6, 0, x_5);
lean_closure_set(x_6, 1, x_2);
x_7 = lp_mathlib_HomologicalComplex_Hom_isoOfComponents___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2082Iso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_11 = lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(x_6);
x_12 = lean_box(0);
lean_inc_ref(x_5);
x_13 = lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(x_5);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_14 = lp_mathlib_CategoryTheory_Functor_mapBifunctorHomologicalComplex___redArg(x_1, x_2, x_3, x_4, x_13, x_11, x_9, x_12, x_12);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_apply_1(x_15, x_7);
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lp_mathlib_CochainComplex_instHasShiftInt___redArg(x_2, x_5);
x_19 = lp_mathlib_CategoryTheory_shiftFunctor___redArg(x_18, x_10);
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lean_apply_1(x_20, x_8);
x_22 = lean_apply_1(x_17, x_21);
x_23 = lean_alloc_closure((void*)(lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2082Iso___redArg___lam__1), 3, 2);
lean_closure_set(x_23, 0, x_22);
lean_closure_set(x_23, 1, x_3);
x_24 = lp_mathlib_HomologicalComplex_Hom_isoOfComponents___redArg(x_23);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2082Iso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CochainComplex_mapBifunctorHomologicalComplexShift_u2082Iso___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_15);
return x_16;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_Bifunctor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_TotalComplexShift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Shift_CommShiftTwo(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Homology_BifunctorShift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_Bifunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_TotalComplexShift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Shift_CommShiftTwo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
