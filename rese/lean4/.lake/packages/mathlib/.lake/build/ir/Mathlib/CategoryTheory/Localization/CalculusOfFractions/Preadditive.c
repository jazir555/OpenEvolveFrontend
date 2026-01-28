// Lean compiler output
// Module: Mathlib.CategoryTheory.Localization.CalculusOfFractions.Preadditive
// Imports: public import Init public import Mathlib.Algebra.Group.TransferInstance public import Mathlib.CategoryTheory.Localization.CalculusOfFractions.Fractions public import Mathlib.CategoryTheory.Localization.HasLocalization public import Mathlib.CategoryTheory.Preadditive.AdditiveFunctor
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_u2082_add(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_u2082_add___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_u2082_add___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_neg___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_neg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_neg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; 
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_7, 1);
lean_inc(x_9);
x_11 = lean_apply_2(x_3, x_5, x_9);
x_12 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_11);
lean_dec_ref(x_11);
x_13 = lean_ctor_get(x_12, 1);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lean_apply_1(x_13, x_10);
lean_ctor_set(x_7, 1, x_14);
return x_7;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_15 = lean_ctor_get(x_7, 0);
x_16 = lean_ctor_get(x_7, 1);
x_17 = lean_ctor_get(x_7, 2);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_15);
lean_dec(x_7);
lean_inc(x_15);
x_18 = lean_apply_2(x_3, x_5, x_15);
x_19 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_18);
lean_dec_ref(x_18);
x_20 = lean_ctor_get(x_19, 1);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lean_apply_1(x_20, x_16);
x_22 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_22, 0, x_15);
lean_ctor_set(x_22, 1, x_21);
lean_ctor_set(x_22, 2, x_17);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_neg___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
x_7 = lean_apply_2(x_1, x_2, x_5);
x_8 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_7);
lean_dec_ref(x_7);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_apply_1(x_9, x_6);
lean_ctor_set(x_3, 1, x_10);
return x_3;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_11 = lean_ctor_get(x_3, 0);
x_12 = lean_ctor_get(x_3, 1);
x_13 = lean_ctor_get(x_3, 2);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_3);
lean_inc(x_11);
x_14 = lean_apply_2(x_1, x_2, x_11);
x_15 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_14);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_15, 1);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_apply_1(x_16, x_12);
x_18 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_18, 0, x_11);
lean_ctor_set(x_18, 1, x_17);
lean_ctor_set(x_18, 2, x_13);
return x_18;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_neg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_neg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_u2082_add(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 1);
lean_inc(x_9);
x_10 = lean_ctor_get(x_7, 2);
lean_inc(x_10);
x_11 = lean_ctor_get(x_7, 3);
lean_inc(x_11);
lean_dec_ref(x_7);
lean_inc(x_8);
x_12 = lean_apply_2(x_3, x_5, x_8);
x_13 = lean_ctor_get(x_12, 0);
lean_inc_ref(x_13);
lean_dec_ref(x_12);
x_14 = !lean_is_exclusive(x_13);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_15 = lean_ctor_get(x_13, 0);
x_16 = lean_ctor_get(x_13, 2);
lean_dec(x_16);
x_17 = lean_ctor_get(x_13, 1);
lean_dec(x_17);
x_18 = lean_apply_2(x_15, x_9, x_10);
lean_ctor_set(x_13, 2, x_11);
lean_ctor_set(x_13, 1, x_18);
lean_ctor_set(x_13, 0, x_8);
return x_13;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_19 = lean_ctor_get(x_13, 0);
lean_inc(x_19);
lean_dec(x_13);
x_20 = lean_apply_2(x_19, x_9, x_10);
x_21 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_21, 0, x_8);
lean_ctor_set(x_21, 1, x_20);
lean_ctor_set(x_21, 2, x_11);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_u2082_add___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 2);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 3);
lean_inc(x_7);
lean_dec_ref(x_3);
lean_inc(x_4);
x_8 = lean_apply_2(x_1, x_2, x_4);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_9, 0);
x_12 = lean_ctor_get(x_9, 2);
lean_dec(x_12);
x_13 = lean_ctor_get(x_9, 1);
lean_dec(x_13);
x_14 = lean_apply_2(x_11, x_5, x_6);
lean_ctor_set(x_9, 2, x_7);
lean_ctor_set(x_9, 1, x_14);
lean_ctor_set(x_9, 0, x_4);
return x_9;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_15 = lean_ctor_get(x_9, 0);
lean_inc(x_15);
lean_dec(x_9);
x_16 = lean_apply_2(x_15, x_5, x_6);
x_17 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_17, 0, x_4);
lean_ctor_set(x_17, 1, x_16);
lean_ctor_set(x_17, 2, x_7);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_u2082_add___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_MorphismProperty_LeftFraction_u2082_add(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_10 = lean_ctor_get(x_1, 2);
lean_inc(x_10);
lean_dec_ref(x_1);
x_11 = lean_ctor_get(x_2, 0);
lean_inc(x_11);
lean_dec_ref(x_2);
x_12 = lean_ctor_get(x_3, 1);
lean_inc(x_12);
lean_dec_ref(x_3);
x_13 = lean_ctor_get(x_4, 0);
lean_inc(x_13);
lean_dec_ref(x_4);
lean_inc(x_11);
x_14 = lean_apply_1(x_11, x_5);
x_15 = lean_apply_1(x_11, x_6);
lean_inc(x_10);
lean_inc(x_7);
lean_inc(x_14);
x_16 = lean_apply_5(x_10, x_14, x_15, x_7, x_9, x_13);
x_17 = lean_apply_5(x_10, x_8, x_14, x_7, x_12, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_10 = lean_ctor_get(x_1, 2);
lean_inc(x_10);
lean_dec_ref(x_1);
x_11 = lean_ctor_get(x_2, 0);
lean_inc(x_11);
lean_dec_ref(x_2);
x_12 = lean_ctor_get(x_3, 0);
lean_inc(x_12);
lean_dec_ref(x_3);
x_13 = lean_ctor_get(x_4, 1);
lean_inc(x_13);
lean_dec_ref(x_4);
lean_inc(x_11);
x_14 = lean_apply_1(x_11, x_5);
x_15 = lean_apply_1(x_11, x_6);
lean_inc(x_10);
lean_inc(x_15);
lean_inc(x_7);
x_16 = lean_apply_5(x_10, x_7, x_8, x_15, x_9, x_13);
x_17 = lean_apply_5(x_10, x_14, x_7, x_15, x_12, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc(x_5);
lean_inc(x_6);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv___redArg___lam__0), 9, 8);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_2);
lean_closure_set(x_9, 2, x_7);
lean_closure_set(x_9, 3, x_8);
lean_closure_set(x_9, 4, x_3);
lean_closure_set(x_9, 5, x_4);
lean_closure_set(x_9, 6, x_6);
lean_closure_set(x_9, 7, x_5);
x_10 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv___redArg___lam__1), 9, 8);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_2);
lean_closure_set(x_10, 2, x_7);
lean_closure_set(x_10, 3, x_8);
lean_closure_set(x_10, 4, x_3);
lean_closure_set(x_10, 5, x_4);
lean_closure_set(x_10, 6, x_5);
lean_closure_set(x_10, 7, x_6);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Localization_Preadditive_homEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_3);
return x_12;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Localization_CalculusOfFractions_Fractions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Localization_HasLocalization(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Preadditive_AdditiveFunctor(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Localization_CalculusOfFractions_Preadditive(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Localization_CalculusOfFractions_Fractions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Localization_HasLocalization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Preadditive_AdditiveFunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
