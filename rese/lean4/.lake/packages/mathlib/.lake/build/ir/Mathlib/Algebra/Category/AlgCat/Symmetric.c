// Lean compiler output
// Module: Mathlib.Algebra.Category.AlgCat.Symmetric
// Imports: public import Init public import Mathlib.Algebra.Category.AlgCat.Monoidal public import Mathlib.Algebra.Category.ModuleCat.Monoidal.Symmetric
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
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instSymmetricCategory___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedModuleCatForget_u2082(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Monoidal_fromInducedMonoidal___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedCategory___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedModuleCatForget_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_Algebra_TensorProduct_instRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedCategory(lean_object*, lean_object*);
lean_object* lp_mathlib_ModuleCat_moduleCategory(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instSymmetricCategory(lean_object*, lean_object*);
lean_object* lp_mathlib_Algebra_TensorProduct_leftAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Algebra_TensorProduct_comm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedModuleCatForget_u2082___redArg(lean_object*);
lean_object* lp_mathlib_AlgEquiv_toAlgebraIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedCategory___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedCategory___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedCategory___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_7);
lean_dec_ref(x_3);
x_8 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_8);
lean_dec_ref(x_4);
x_9 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_9);
lean_inc_ref(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_1);
x_10 = lp_mathlib_Algebra_TensorProduct_instRing___redArg(x_1, x_6, x_7, x_9, x_8);
x_11 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_11);
lean_dec_ref(x_6);
lean_inc_ref(x_7);
lean_inc_ref(x_8);
lean_inc_ref(x_1);
x_12 = lp_mathlib_Algebra_TensorProduct_instRing___redArg(x_1, x_5, x_8, x_11, x_7);
lean_inc_ref(x_8);
lean_inc_ref_n(x_7, 2);
lean_inc_ref(x_1);
x_13 = lp_mathlib_Algebra_TensorProduct_leftAlgebra___redArg(x_1, x_11, x_7, x_9, x_8, x_7);
lean_inc_ref(x_7);
lean_inc_ref_n(x_8, 2);
lean_inc_ref(x_1);
x_14 = lp_mathlib_Algebra_TensorProduct_leftAlgebra___redArg(x_1, x_9, x_8, x_11, x_7, x_8);
x_15 = lp_mathlib_Algebra_TensorProduct_comm___redArg(x_1, x_11, x_7, x_9, x_8);
lean_dec_ref(x_9);
lean_dec_ref(x_11);
x_16 = lp_mathlib_AlgEquiv_toAlgebraIso___redArg(x_2, x_10, x_12, x_13, x_14, x_15);
lean_dec_ref(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_10);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedCategory___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AlgCat_instBraidedCategory___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedCategory___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_AlgCat_instBraidedCategory___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedCategory(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AlgCat_instBraidedCategory___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedModuleCatForget_u2082___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_6);
lean_dec_ref(x_3);
x_7 = lp_mathlib_Ring_toAddCommGroup___redArg(x_5);
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_10);
lean_dec_ref(x_4);
x_11 = lp_mathlib_Ring_toAddCommGroup___redArg(x_9);
x_12 = !lean_is_exclusive(x_10);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_13 = lean_ctor_get(x_10, 0);
x_14 = lean_ctor_get(x_10, 1);
lean_dec(x_14);
x_15 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_15);
lean_dec_ref(x_11);
lean_inc(x_13);
lean_inc(x_8);
lean_inc_ref(x_15);
lean_inc_ref(x_7);
lean_inc_ref(x_1);
x_16 = lp_mathlib_TensorProduct_addCommGroup___redArg(x_1, x_7, x_15, x_8, x_13);
x_17 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_17);
lean_dec_ref(x_7);
lean_inc(x_8);
x_18 = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(x_18, 0, x_1);
lean_closure_set(x_18, 1, x_17);
lean_closure_set(x_18, 2, x_15);
lean_closure_set(x_18, 3, x_8);
lean_closure_set(x_18, 4, x_13);
lean_closure_set(x_18, 5, x_8);
lean_ctor_set(x_10, 1, x_18);
lean_ctor_set(x_10, 0, x_16);
x_19 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_10);
return x_19;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_20 = lean_ctor_get(x_10, 0);
lean_inc(x_20);
lean_dec(x_10);
x_21 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_21);
lean_dec_ref(x_11);
lean_inc(x_20);
lean_inc(x_8);
lean_inc_ref(x_21);
lean_inc_ref(x_7);
lean_inc_ref(x_1);
x_22 = lp_mathlib_TensorProduct_addCommGroup___redArg(x_1, x_7, x_21, x_8, x_20);
x_23 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_23);
lean_dec_ref(x_7);
lean_inc(x_8);
x_24 = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(x_24, 0, x_1);
lean_closure_set(x_24, 1, x_23);
lean_closure_set(x_24, 2, x_21);
lean_closure_set(x_24, 3, x_8);
lean_closure_set(x_24, 4, x_20);
lean_closure_set(x_24, 5, x_8);
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_22);
lean_ctor_set(x_25, 1, x_24);
x_26 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_25);
return x_26;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedModuleCatForget_u2082___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_2 = lp_mathlib_ModuleCat_moduleCategory(lean_box(0), x_1);
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_3);
x_4 = lean_alloc_closure((void*)(lp_mathlib_AlgCat_instBraidedModuleCatForget_u2082___redArg___lam__0), 4, 2);
lean_closure_set(x_4, 0, x_3);
lean_closure_set(x_4, 1, x_2);
x_5 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_6 = lp_mathlib_Semiring_toModule___redArg(x_3);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
x_8 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_4);
lean_ctor_set(x_9, 1, x_8);
x_10 = lp_mathlib_CategoryTheory_Monoidal_fromInducedMonoidal___redArg(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instBraidedModuleCatForget_u2082(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AlgCat_instBraidedModuleCatForget_u2082___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instSymmetricCategory(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AlgCat_instBraidedCategory___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgCat_instSymmetricCategory___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AlgCat_instBraidedCategory___redArg(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_AlgCat_Monoidal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Monoidal_Symmetric(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_AlgCat_Symmetric(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_AlgCat_Monoidal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Monoidal_Symmetric(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
