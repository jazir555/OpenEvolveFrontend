// Lean compiler output
// Module: Mathlib.CategoryTheory.Preadditive.Projective.Basic
// Imports: public import Init public import Mathlib.CategoryTheory.Adjunction.FullyFaithful public import Mathlib.CategoryTheory.Adjunction.Limits public import Mathlib.CategoryTheory.Limits.Constructions.EpiMono public import Mathlib.CategoryTheory.Limits.Preserves.Finite public import Mathlib.CategoryTheory.Limits.Shapes.BinaryBiproducts
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_projectivePresentationOfMapProjectivePresentation___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_mapProjectivePresentation___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_mapProjectivePresentation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_mapProjectivePresentation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_projectivePresentationOfMapProjectivePresentation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_projectivePresentationOfMapProjectivePresentation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_mapProjectivePresentation___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_3, 0);
x_8 = lean_ctor_get(x_3, 1);
lean_inc(x_7);
x_9 = lean_apply_1(x_4, x_7);
x_10 = lean_apply_3(x_5, x_7, x_2, x_8);
lean_ctor_set(x_3, 1, x_10);
lean_ctor_set(x_3, 0, x_9);
return x_3;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_ctor_get(x_3, 0);
x_12 = lean_ctor_get(x_3, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_3);
lean_inc(x_11);
x_13 = lean_apply_1(x_4, x_11);
x_14 = lean_apply_3(x_5, x_11, x_2, x_12);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_mapProjectivePresentation(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Adjunction_mapProjectivePresentation___redArg(x_5, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_mapProjectivePresentation___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Adjunction_mapProjectivePresentation(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_projectivePresentationOfMapProjectivePresentation___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_5 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_7);
lean_dec_ref(x_2);
x_8 = lean_ctor_get(x_5, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_5, 1);
lean_inc(x_9);
lean_dec_ref(x_5);
x_10 = !lean_is_exclusive(x_4);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_11 = lean_ctor_get(x_4, 0);
x_12 = lean_ctor_get(x_4, 1);
x_13 = lean_ctor_get(x_1, 2);
lean_inc(x_13);
lean_dec_ref(x_1);
x_14 = lean_ctor_get(x_6, 0);
lean_inc(x_14);
lean_dec_ref(x_6);
x_15 = lean_ctor_get(x_7, 1);
lean_inc(x_15);
lean_dec_ref(x_7);
lean_inc(x_8);
lean_inc(x_11);
x_16 = lean_apply_1(x_8, x_11);
lean_inc(x_3);
x_17 = lean_apply_1(x_14, x_3);
lean_inc(x_17);
x_18 = lean_apply_1(x_8, x_17);
x_19 = lean_apply_3(x_9, x_11, x_17, x_12);
lean_inc(x_3);
x_20 = lean_apply_1(x_15, x_3);
lean_inc(x_16);
x_21 = lean_apply_5(x_13, x_16, x_18, x_3, x_19, x_20);
lean_ctor_set(x_4, 1, x_21);
lean_ctor_set(x_4, 0, x_16);
return x_4;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_22 = lean_ctor_get(x_4, 0);
x_23 = lean_ctor_get(x_4, 1);
lean_inc(x_23);
lean_inc(x_22);
lean_dec(x_4);
x_24 = lean_ctor_get(x_1, 2);
lean_inc(x_24);
lean_dec_ref(x_1);
x_25 = lean_ctor_get(x_6, 0);
lean_inc(x_25);
lean_dec_ref(x_6);
x_26 = lean_ctor_get(x_7, 1);
lean_inc(x_26);
lean_dec_ref(x_7);
lean_inc(x_8);
lean_inc(x_22);
x_27 = lean_apply_1(x_8, x_22);
lean_inc(x_3);
x_28 = lean_apply_1(x_25, x_3);
lean_inc(x_28);
x_29 = lean_apply_1(x_8, x_28);
x_30 = lean_apply_3(x_9, x_22, x_28, x_23);
lean_inc(x_3);
x_31 = lean_apply_1(x_26, x_3);
lean_inc(x_27);
x_32 = lean_apply_5(x_24, x_27, x_29, x_3, x_30, x_31);
x_33 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_33, 0, x_27);
lean_ctor_set(x_33, 1, x_32);
return x_33;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_projectivePresentationOfMapProjectivePresentation(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Equivalence_projectivePresentationOfMapProjectivePresentation___redArg(x_2, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Equivalence_projectivePresentationOfMapProjectivePresentation___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Equivalence_projectivePresentationOfMapProjectivePresentation(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Adjunction_FullyFaithful(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Limits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_EpiMono(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryBiproducts(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Preadditive_Projective_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Adjunction_FullyFaithful(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Limits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Constructions_EpiMono(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryBiproducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
