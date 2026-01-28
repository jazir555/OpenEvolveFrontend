// Lean compiler output
// Module: Mathlib.Algebra.Group.MinimalAxioms
// Imports: public import Init public import Mathlib.Algebra.Group.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_ofLeftAxioms(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_ofRightAxioms___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_ofLeftAxioms___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Group_ofRightAxioms___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_nsmulBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Group_ofLeftAxioms(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Group_ofLeftAxioms___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_ofRightAxioms(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Group_ofRightAxioms(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Group_ofLeftAxioms(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc(x_4);
lean_inc(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, x_4);
lean_inc(x_4);
lean_inc(x_2);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_2);
lean_ctor_set(x_9, 1, x_4);
lean_ctor_set(x_9, 2, x_8);
lean_inc(x_3);
lean_inc_ref(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_9);
lean_closure_set(x_10, 2, x_3);
lean_inc(x_2);
lean_inc(x_4);
x_11 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, x_4);
lean_closure_set(x_11, 2, x_2);
lean_inc(x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_4);
lean_closure_set(x_12, 2, x_2);
lean_closure_set(x_12, 3, x_3);
lean_closure_set(x_12, 4, x_11);
x_13 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_13, 0, x_9);
lean_ctor_set(x_13, 1, x_3);
lean_ctor_set(x_13, 2, x_10);
lean_ctor_set(x_13, 3, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Group_ofLeftAxioms___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_3);
lean_inc(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_1);
lean_closure_set(x_4, 2, x_3);
lean_inc(x_3);
lean_inc(x_1);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
lean_inc(x_2);
lean_inc_ref(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, x_5);
lean_closure_set(x_6, 2, x_2);
lean_inc(x_1);
lean_inc(x_3);
x_7 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_3);
lean_closure_set(x_7, 2, x_1);
lean_inc(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_3);
lean_closure_set(x_8, 2, x_1);
lean_closure_set(x_8, 3, x_2);
lean_closure_set(x_8, 4, x_7);
x_9 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_9, 0, x_5);
lean_ctor_set(x_9, 1, x_2);
lean_ctor_set(x_9, 2, x_6);
lean_ctor_set(x_9, 3, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_ofLeftAxioms___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_3);
lean_inc(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_nsmulBinRecAuto___boxed), 5, 3);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_1);
lean_closure_set(x_4, 2, x_3);
lean_inc(x_3);
lean_inc(x_1);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
lean_inc(x_2);
lean_inc_ref(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, x_5);
lean_closure_set(x_6, 2, x_2);
lean_inc(x_1);
lean_inc(x_3);
x_7 = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_3);
lean_closure_set(x_7, 2, x_1);
lean_inc(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_zsmulRec___boxed), 7, 5);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_3);
lean_closure_set(x_8, 2, x_1);
lean_closure_set(x_8, 3, x_2);
lean_closure_set(x_8, 4, x_7);
x_9 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_9, 0, x_5);
lean_ctor_set(x_9, 1, x_2);
lean_ctor_set(x_9, 2, x_6);
lean_ctor_set(x_9, 3, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_ofLeftAxioms(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_AddGroup_ofLeftAxioms___redArg(x_2, x_3, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Group_ofRightAxioms(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc(x_4);
lean_inc(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, x_4);
lean_inc(x_4);
lean_inc(x_2);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_2);
lean_ctor_set(x_9, 1, x_4);
lean_ctor_set(x_9, 2, x_8);
lean_inc(x_3);
lean_inc_ref(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_9);
lean_closure_set(x_10, 2, x_3);
lean_inc(x_2);
lean_inc(x_4);
x_11 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, x_4);
lean_closure_set(x_11, 2, x_2);
lean_inc(x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_4);
lean_closure_set(x_12, 2, x_2);
lean_closure_set(x_12, 3, x_3);
lean_closure_set(x_12, 4, x_11);
x_13 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_13, 0, x_9);
lean_ctor_set(x_13, 1, x_3);
lean_ctor_set(x_13, 2, x_10);
lean_ctor_set(x_13, 3, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Group_ofRightAxioms___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_3);
lean_inc(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_1);
lean_closure_set(x_4, 2, x_3);
lean_inc(x_3);
lean_inc(x_1);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
lean_inc(x_2);
lean_inc_ref(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, x_5);
lean_closure_set(x_6, 2, x_2);
lean_inc(x_1);
lean_inc(x_3);
x_7 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_3);
lean_closure_set(x_7, 2, x_1);
lean_inc(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_3);
lean_closure_set(x_8, 2, x_1);
lean_closure_set(x_8, 3, x_2);
lean_closure_set(x_8, 4, x_7);
x_9 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_9, 0, x_5);
lean_ctor_set(x_9, 1, x_2);
lean_ctor_set(x_9, 2, x_6);
lean_ctor_set(x_9, 3, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_ofRightAxioms___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_3);
lean_inc(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_nsmulBinRecAuto___boxed), 5, 3);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_1);
lean_closure_set(x_4, 2, x_3);
lean_inc(x_3);
lean_inc(x_1);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
lean_inc(x_2);
lean_inc_ref(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, x_5);
lean_closure_set(x_6, 2, x_2);
lean_inc(x_1);
lean_inc(x_3);
x_7 = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_3);
lean_closure_set(x_7, 2, x_1);
lean_inc(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_zsmulRec___boxed), 7, 5);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_3);
lean_closure_set(x_8, 2, x_1);
lean_closure_set(x_8, 3, x_2);
lean_closure_set(x_8, 4, x_7);
x_9 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_9, 0, x_5);
lean_ctor_set(x_9, 1, x_2);
lean_ctor_set(x_9, 2, x_6);
lean_ctor_set(x_9, 3, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_ofRightAxioms(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_AddGroup_ofRightAxioms___redArg(x_2, x_3, x_4);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_MinimalAxioms(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
