// Lean compiler output
// Module: Mathlib.CategoryTheory.Adjunction.Unique
// Imports: public import Init public import Mathlib.CategoryTheory.Adjunction.Mates
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
lean_object* lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointUniq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointUniq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_rightAdjointUniq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_rightAdjointUniq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_category___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointUniq___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_inc_ref_n(x_5, 2);
lean_inc_ref(x_1);
x_8 = lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_5, x_6, x_7);
x_9 = lp_mathlib_Equiv_symm___redArg(x_8);
x_10 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_1);
x_11 = lean_ctor_get(x_9, 0);
lean_inc(x_11);
lean_dec_ref(x_9);
x_12 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_10, x_5);
x_13 = lean_apply_1(x_11, x_12);
x_14 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointUniq(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Adjunction_leftAdjointUniq___redArg(x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_rightAdjointUniq___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref_n(x_3, 2);
lean_inc_ref(x_2);
x_8 = lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg(x_1, x_2, x_3, x_3, x_4, x_5, x_6, x_7);
x_9 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_2);
x_10 = lean_ctor_get(x_8, 0);
lean_inc(x_10);
lean_dec_ref(x_8);
x_11 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_9, x_3);
x_12 = lean_apply_1(x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_rightAdjointUniq(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Adjunction_rightAdjointUniq___redArg(x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Mates(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Unique(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Mates(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
