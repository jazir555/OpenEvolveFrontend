// Lean compiler output
// Module: Mathlib.CategoryTheory.Monoidal.Internal.Types.Grp_
// Imports: public import Init public import Mathlib.CategoryTheory.Monoidal.Internal.Types.Basic public import Mathlib.CategoryTheory.Monoidal.Grp_ public import Mathlib.Algebra.Category.Grp.Basic
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
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpTypeEquivalenceGrp_grpGroup(lean_object*, lean_object*);
lean_object* lp_mathlib_MonTypeEquivalenceMon_monMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpTypeEquivalenceGrp_grpGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpTypeEquivalenceGrp_grpGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lp_mathlib_MonTypeEquivalenceMon_monMonoid___redArg(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_inc(x_3);
lean_inc_ref(x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_4);
lean_closure_set(x_7, 2, x_3);
lean_inc(x_5);
lean_inc(x_6);
x_8 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_6);
lean_closure_set(x_8, 2, x_5);
lean_inc(x_3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_6);
lean_closure_set(x_9, 2, x_5);
lean_closure_set(x_9, 3, x_3);
lean_closure_set(x_9, 4, x_8);
x_10 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_10, 0, x_4);
lean_ctor_set(x_10, 1, x_3);
lean_ctor_set(x_10, 2, x_7);
lean_ctor_set(x_10, 3, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpTypeEquivalenceGrp_grpGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_GrpTypeEquivalenceGrp_grpGroup___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Internal_Types_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Grp__(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Grp_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Internal_Types_Grp__(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Internal_Types_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Grp__(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Grp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
