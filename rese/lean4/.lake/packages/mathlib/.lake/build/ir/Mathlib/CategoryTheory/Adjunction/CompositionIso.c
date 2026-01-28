// Lean compiler output
// Module: Mathlib.CategoryTheory.Adjunction.CompositionIso
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointCompNatTrans___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_id(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Adjunction_comp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointCompIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointIdIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_symm___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_conjugateEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointCompNatTrans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointIdIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointCompIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Adjunction_id___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointIdIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_1);
lean_inc_ref(x_1);
x_7 = lp_mathlib_CategoryTheory_Adjunction_id___redArg(x_1);
lean_inc_ref(x_6);
lean_inc_ref(x_1);
x_8 = lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg(x_1, x_1, x_6, x_2, x_6, x_3, x_7, x_4);
x_9 = lp_mathlib_Equiv_symm___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_5);
x_12 = lean_apply_1(x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointIdIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Adjunction_leftAdjointIdIso___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointCompNatTrans___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_14 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_4, x_5);
lean_inc_ref(x_7);
lean_inc_ref(x_8);
x_15 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_8, x_7);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_16 = lp_mathlib_CategoryTheory_Adjunction_comp___redArg(x_1, x_2, x_4, x_7, x_3, x_5, x_8, x_10, x_11);
x_17 = lp_mathlib_CategoryTheory_conjugateEquiv___redArg(x_1, x_3, x_6, x_14, x_9, x_15, x_12, x_16);
x_18 = lp_mathlib_Equiv_symm___redArg(x_17);
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lean_apply_1(x_19, x_13);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointCompNatTrans(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_CategoryTheory_Adjunction_leftAdjointCompNatTrans___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointCompIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_14 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_4, x_5);
lean_inc_ref(x_7);
lean_inc_ref(x_8);
x_15 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_8, x_7);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_16 = lp_mathlib_CategoryTheory_Adjunction_comp___redArg(x_1, x_2, x_4, x_7, x_3, x_5, x_8, x_10, x_11);
x_17 = lp_mathlib_CategoryTheory_conjugateIsoEquiv___redArg(x_1, x_3, x_6, x_14, x_9, x_15, x_12, x_16);
x_18 = lp_mathlib_Equiv_symm___redArg(x_17);
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_13);
x_21 = lean_apply_1(x_19, x_20);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Adjunction_leftAdjointCompIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_CategoryTheory_Adjunction_leftAdjointCompIso___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
return x_17;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Mates(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Adjunction_CompositionIso(uint8_t builtin) {
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
