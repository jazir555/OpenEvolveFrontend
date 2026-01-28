// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Monoidal.Symmetric
// Imports: public import Init public import Mathlib.CategoryTheory.Monoidal.Braided.Basic public import Mathlib.Algebra.Category.ModuleCat.Monoidal.Basic
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
lean_object* lp_mathlib_SemimoduleCat_moduleCategory(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_symmetricCategory___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedSemimoduleCatFunctorEquivalenceSemimoduleCat___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemimoduleCat_MonoidalCategory_symmetricCategory(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedSemimoduleCatFunctorEquivalenceSemimoduleCat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_symmetricCategory(lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_comm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_toModuleIso_u209b___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemimoduleCat_MonoidalCategory_symmetricCategory___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Monoidal_fromInducedMonoidal___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_SemimoduleCat_MonoidalCategory_tensorObj___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedCategory(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedCategory___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemimoduleCat_braiding___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_toModuleIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemimoduleCat_braiding(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ModuleCat_equivalenceSemimoduleCat___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedCategory___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedSemimoduleCatFunctorEquivalenceSemimoduleCat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemimoduleCat_braiding___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_dec_ref(x_2);
x_6 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_3, 1);
lean_inc(x_7);
lean_dec_ref(x_3);
x_8 = lp_mathlib_TensorProduct_comm___redArg(x_1, x_4, x_6, x_5, x_7);
x_9 = lp_mathlib_LinearEquiv_toModuleIso_u209b___redArg(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemimoduleCat_braiding(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SemimoduleCat_braiding___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemimoduleCat_MonoidalCategory_symmetricCategory(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_SemimoduleCat_braiding), 4, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemimoduleCat_MonoidalCategory_symmetricCategory___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_SemimoduleCat_braiding), 4, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedCategory___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_2, 1);
lean_inc(x_6);
lean_dec_ref(x_2);
x_7 = lean_ctor_get(x_3, 1);
lean_inc(x_7);
lean_dec_ref(x_3);
x_8 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_4);
x_9 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_5);
x_10 = lp_mathlib_TensorProduct_comm___redArg(x_1, x_9, x_8, x_6, x_7);
x_11 = lp_mathlib_LinearEquiv_toModuleIso___redArg(x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedCategory___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_MonoidalCategory_instBraidedCategory___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedCategory(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ModuleCat_MonoidalCategory_instBraidedCategory___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedSemimoduleCatFunctorEquivalenceSemimoduleCat___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
lean_inc(x_6);
x_7 = lean_apply_1(x_6, x_4);
x_8 = lean_apply_1(x_6, x_5);
x_9 = lp_mathlib_SemimoduleCat_MonoidalCategory_tensorObj___redArg(x_2, x_7, x_8);
x_10 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_3, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedSemimoduleCatFunctorEquivalenceSemimoduleCat___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_SemimoduleCat_moduleCategory(lean_box(0), x_2);
x_4 = lp_mathlib_ModuleCat_equivalenceSemimoduleCat___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lean_ctor_get(x_7, 1);
lean_dec(x_9);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_MonoidalCategory_instBraidedSemimoduleCatFunctorEquivalenceSemimoduleCat___redArg___lam__0), 5, 3);
lean_closure_set(x_10, 0, x_5);
lean_closure_set(x_10, 1, x_2);
lean_closure_set(x_10, 2, x_3);
x_11 = lp_mathlib_Semiring_toModule___redArg(x_2);
lean_ctor_set(x_7, 1, x_11);
x_12 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_3, x_7);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_10);
lean_ctor_set(x_13, 1, x_12);
x_14 = lp_mathlib_CategoryTheory_Monoidal_fromInducedMonoidal___redArg(x_13);
return x_14;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_15 = lean_ctor_get(x_7, 0);
lean_inc(x_15);
lean_dec(x_7);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_MonoidalCategory_instBraidedSemimoduleCatFunctorEquivalenceSemimoduleCat___redArg___lam__0), 5, 3);
lean_closure_set(x_16, 0, x_5);
lean_closure_set(x_16, 1, x_2);
lean_closure_set(x_16, 2, x_3);
x_17 = lp_mathlib_Semiring_toModule___redArg(x_2);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_15);
lean_ctor_set(x_18, 1, x_17);
x_19 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_3, x_18);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_16);
lean_ctor_set(x_20, 1, x_19);
x_21 = lp_mathlib_CategoryTheory_Monoidal_fromInducedMonoidal___redArg(x_20);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_instBraidedSemimoduleCatFunctorEquivalenceSemimoduleCat(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ModuleCat_MonoidalCategory_instBraidedSemimoduleCatFunctorEquivalenceSemimoduleCat___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_symmetricCategory(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ModuleCat_MonoidalCategory_instBraidedCategory___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_MonoidalCategory_symmetricCategory___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ModuleCat_MonoidalCategory_instBraidedCategory___redArg(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Braided_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Monoidal_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Monoidal_Symmetric(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Braided_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Monoidal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
