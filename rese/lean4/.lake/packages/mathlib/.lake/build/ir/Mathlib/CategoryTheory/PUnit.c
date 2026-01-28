// Lean compiler output
// Module: Mathlib.CategoryTheory.PUnit
// Imports: public import Init public import Mathlib.CategoryTheory.Functor.Const public import Mathlib.CategoryTheory.Discrete.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_punitExt___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_punitExt___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Functor_star___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_id(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_const___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_refl(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_punitExt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_punitExt(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_star___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_discreteCategory(lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Functor_star___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_fromPUnit___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_category___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_fromPUnit(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_star(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_eqToIso___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_CategoryTheory_Functor_star___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_CategoryTheory_discreteCategory(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Functor_star___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CategoryTheory_Functor_star___closed__0;
x_2 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_star(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_CategoryTheory_Functor_star___closed__1;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_box(0);
x_6 = lean_apply_1(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_star___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Functor_star(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_punitExt___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
lean_dec_ref(x_2);
lean_inc(x_4);
x_7 = lean_apply_1(x_5, x_4);
x_8 = lean_apply_1(x_6, x_4);
x_9 = lp_mathlib_CategoryTheory_eqToIso___redArg(x_3, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_punitExt___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_CategoryTheory_Functor_star___closed__0;
x_4 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_punitExt___redArg___lam__0), 4, 3);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
lean_closure_set(x_4, 2, x_3);
x_5 = lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_punitExt(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Functor_punitExt___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_punitExt___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Functor_punitExt(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_fromPUnit(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_apply_1(x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_fromPUnit___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lean_apply_1(x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = lean_box(0);
x_4 = lean_apply_1(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_box(0);
x_5 = lean_apply_1(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__1(x_1, x_2, x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_1(x_4, x_3);
x_6 = lp_mathlib_CategoryTheory_Iso_refl___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_1(x_4, x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__2), 3, 2);
lean_closure_set(x_6, 0, x_5);
lean_closure_set(x_6, 1, x_2);
x_7 = lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__0), 1, 0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__1___boxed), 3, 0);
lean_inc_ref(x_1);
x_4 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_3);
lean_inc_ref(x_1);
x_6 = lp_mathlib_CategoryTheory_Functor_const___redArg(x_1);
x_7 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_4);
lean_dec_ref(x_4);
lean_inc_ref(x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_equiv___redArg___lam__3), 3, 2);
lean_closure_set(x_8, 0, x_7);
lean_closure_set(x_8, 1, x_1);
x_9 = lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Iso_refl), 3, 2);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_1);
x_11 = lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(x_10);
x_12 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_12, 0, x_5);
lean_ctor_set(x_12, 1, x_6);
lean_ctor_set(x_12, 2, x_9);
lean_ctor_set(x_12, 3, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_equiv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Functor_equiv___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Functor_Const(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Discrete_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_PUnit(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Functor_Const(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Discrete_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CategoryTheory_Functor_star___closed__0 = _init_lp_mathlib_CategoryTheory_Functor_star___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_Functor_star___closed__0);
lp_mathlib_CategoryTheory_Functor_star___closed__1 = _init_lp_mathlib_CategoryTheory_Functor_star___closed__1();
lean_mark_persistent(lp_mathlib_CategoryTheory_Functor_star___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
