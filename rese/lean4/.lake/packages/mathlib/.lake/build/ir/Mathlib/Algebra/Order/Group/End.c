// Lean compiler output
// Module: Mathlib.Algebra.Order.Group.End
// Imports: public import Init public import Mathlib.Algebra.Group.Defs public import Mathlib.Order.RelIso.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_instMonoid(lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_instGroup___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_RelHom_instMonoid___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_instMonoid___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RelHom_id___lam__0___boxed(lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RelIso_symm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_RelIso_instGroup___closed__1;
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_RelEmbedding_instMonoid___closed__0;
static lean_object* lp_mathlib_RelHom_instMonoid___closed__3;
lean_object* lp_mathlib_Function_Embedding_trans___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_RelHom_instMonoid___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_RelIso_instGroup(lean_object*, lean_object*);
lean_object* lp_mathlib_RelHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instMonoid(lean_object*, lean_object*);
static lean_object* lp_mathlib_RelIso_instGroup___closed__0;
static lean_object* lp_mathlib_RelHom_instMonoid___closed__0;
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Function_Embedding_refl___lam__0___boxed(lean_object*);
static lean_object* _init_lp_mathlib_RelHom_instMonoid___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_RelHom_comp), 8, 6);
lean_closure_set(x_1, 0, lean_box(0));
lean_closure_set(x_1, 1, lean_box(0));
lean_closure_set(x_1, 2, lean_box(0));
lean_closure_set(x_1, 3, lean_box(0));
lean_closure_set(x_1, 4, lean_box(0));
lean_closure_set(x_1, 5, lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_RelHom_instMonoid___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_RelHom_id___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_RelHom_instMonoid___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_RelHom_instMonoid___closed__1;
x_2 = lp_mathlib_RelHom_instMonoid___closed__0;
x_3 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_2);
lean_closure_set(x_3, 2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_RelHom_instMonoid___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_RelHom_instMonoid___closed__2;
x_2 = lp_mathlib_RelHom_instMonoid___closed__1;
x_3 = lp_mathlib_RelHom_instMonoid___closed__0;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RelHom_instMonoid___closed__3;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_instMonoid___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Function_Embedding_trans___redArg(x_2, x_1);
x_5 = lean_apply_1(x_4, x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_RelEmbedding_instMonoid___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_refl___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_instMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_RelEmbedding_instMonoid___lam__0), 3, 0);
x_4 = lp_mathlib_RelEmbedding_instMonoid___closed__0;
lean_inc_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, x_3);
lean_closure_set(x_5, 2, x_4);
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_3);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_instGroup___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_trans___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_RelIso_instGroup___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_RelIso_instGroup___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_RelIso_symm), 5, 4);
lean_closure_set(x_1, 0, lean_box(0));
lean_closure_set(x_1, 1, lean_box(0));
lean_closure_set(x_1, 2, lean_box(0));
lean_closure_set(x_1, 3, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_instGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_RelIso_instGroup___lam__0), 2, 0);
x_4 = lp_mathlib_RelIso_instGroup___closed__0;
lean_inc_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, x_3);
lean_closure_set(x_5, 2, x_4);
lean_inc_ref(x_3);
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_3);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_5);
x_7 = lp_mathlib_RelIso_instGroup___closed__1;
lean_inc_ref(x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_6);
lean_closure_set(x_8, 2, x_7);
lean_inc_ref(x_3);
x_9 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_4);
lean_closure_set(x_9, 2, x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_4);
lean_closure_set(x_10, 2, x_3);
lean_closure_set(x_10, 3, x_7);
lean_closure_set(x_10, 4, x_9);
x_11 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_11, 0, x_6);
lean_ctor_set(x_11, 1, x_7);
lean_ctor_set(x_11, 2, x_8);
lean_ctor_set(x_11, 3, x_10);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_End(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_RelHom_instMonoid___closed__0 = _init_lp_mathlib_RelHom_instMonoid___closed__0();
lean_mark_persistent(lp_mathlib_RelHom_instMonoid___closed__0);
lp_mathlib_RelHom_instMonoid___closed__1 = _init_lp_mathlib_RelHom_instMonoid___closed__1();
lean_mark_persistent(lp_mathlib_RelHom_instMonoid___closed__1);
lp_mathlib_RelHom_instMonoid___closed__2 = _init_lp_mathlib_RelHom_instMonoid___closed__2();
lean_mark_persistent(lp_mathlib_RelHom_instMonoid___closed__2);
lp_mathlib_RelHom_instMonoid___closed__3 = _init_lp_mathlib_RelHom_instMonoid___closed__3();
lean_mark_persistent(lp_mathlib_RelHom_instMonoid___closed__3);
lp_mathlib_RelEmbedding_instMonoid___closed__0 = _init_lp_mathlib_RelEmbedding_instMonoid___closed__0();
lean_mark_persistent(lp_mathlib_RelEmbedding_instMonoid___closed__0);
lp_mathlib_RelIso_instGroup___closed__0 = _init_lp_mathlib_RelIso_instGroup___closed__0();
lean_mark_persistent(lp_mathlib_RelIso_instGroup___closed__0);
lp_mathlib_RelIso_instGroup___closed__1 = _init_lp_mathlib_RelIso_instGroup___closed__1();
lean_mark_persistent(lp_mathlib_RelIso_instGroup___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
