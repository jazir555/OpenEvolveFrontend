// Lean compiler output
// Module: Mathlib.CategoryTheory.Abelian.Ext
// Imports: public import Init public import Mathlib.Algebra.Category.ModuleCat.Abelian public import Mathlib.Algebra.Homology.Opposite public import Mathlib.CategoryTheory.Abelian.LeftDerived public import Mathlib.CategoryTheory.Abelian.Opposite public import Mathlib.CategoryTheory.Abelian.Projective.Resolution public import Mathlib.CategoryTheory.Linear.Yoneda
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
LEAN_EXPORT lean_object* lp_mathlib_ChainComplex_linearYonedaObj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_rightOp___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapHomologicalComplex___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ChainComplex_linearYonedaObj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ChainComplex_linearYonedaObj___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_linearYoneda___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_HomologicalComplex_unop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ChainComplex_linearYonedaObj___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_6 = lp_mathlib_CategoryTheory_linearYoneda___redArg(x_1, x_2, x_4);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_7, x_5);
x_9 = lp_mathlib_CategoryTheory_Functor_rightOp___redArg(x_8);
x_10 = lp_mathlib_CategoryTheory_Functor_mapHomologicalComplex___redArg(x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lean_apply_1(x_11, x_3);
x_13 = lp_mathlib_HomologicalComplex_unop___redArg(x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ChainComplex_linearYonedaObj(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_ChainComplex_linearYonedaObj___redArg(x_2, x_3, x_7, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ChainComplex_linearYonedaObj___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_ChainComplex_linearYonedaObj(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_9);
lean_dec(x_6);
lean_dec(x_5);
return x_12;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Abelian(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Abelian_LeftDerived(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Abelian_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Abelian_Projective_Resolution(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Linear_Yoneda(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Abelian_Ext(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Abelian(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Abelian_LeftDerived(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Abelian_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Abelian_Projective_Resolution(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Linear_Yoneda(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
