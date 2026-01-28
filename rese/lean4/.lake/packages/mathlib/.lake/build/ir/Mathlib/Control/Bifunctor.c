// Lean compiler output
// Module: Mathlib.Control.Bifunctor
// Imports: public import Init public import Mathlib.Control.Functor public import Mathlib.Tactic.Common
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
static lean_object* lp_mathlib_Prod_bifunctor___closed__0;
lean_object* l_Sum_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_functor___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_fst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompl_bifunctor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Bifunctor_fst___closed__0;
lean_object* l_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_snd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_fst___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_bifunctor;
lean_object* l_Prod_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_functor(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_const___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_flip___redArg(lean_object*);
static lean_object* lp_mathlib_Sum_bifunctor___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompl_bifunctor___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_functor___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_snd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompr_bifunctor___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompr_bifunctor___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_functor___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_const___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompr_bifunctor(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_flip___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_bifunctor;
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompl_bifunctor___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_const;
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_flip(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Bifunctor_fst___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_fst(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_Bifunctor_fst___closed__0;
x_9 = lean_apply_7(x_2, lean_box(0), lean_box(0), lean_box(0), lean_box(0), x_6, x_8, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_fst___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Bifunctor_fst___closed__0;
x_5 = lean_apply_7(x_1, lean_box(0), lean_box(0), lean_box(0), lean_box(0), x_2, x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_snd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_Bifunctor_fst___closed__0;
x_9 = lean_apply_7(x_2, lean_box(0), lean_box(0), lean_box(0), lean_box(0), x_8, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_snd___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Bifunctor_fst___closed__0;
x_5 = lean_apply_7(x_1, lean_box(0), lean_box(0), lean_box(0), lean_box(0), x_4, x_2, x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Prod_bifunctor___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Prod_map), 7, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Prod_bifunctor() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Prod_bifunctor___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_const___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_apply_1(x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_const___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Bifunctor_const___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
return x_8;
}
}
static lean_object* _init_lp_mathlib_Bifunctor_const() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Bifunctor_const___lam__0___boxed), 7, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_flip___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lean_apply_7(x_1, lean_box(0), lean_box(0), lean_box(0), lean_box(0), x_7, x_6, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_flip___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Bifunctor_flip___redArg___lam__0), 8, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_flip(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Bifunctor_flip___redArg(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Sum_bifunctor___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Sum_map), 7, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Sum_bifunctor() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Sum_bifunctor___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_functor___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_Bifunctor_fst___closed__0;
x_7 = lean_apply_7(x_1, lean_box(0), lean_box(0), lean_box(0), lean_box(0), x_6, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_functor___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_4);
x_7 = lean_apply_4(x_1, lean_box(0), lean_box(0), x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_functor___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Bifunctor_functor___redArg___lam__0), 5, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Bifunctor_functor___redArg___lam__1), 5, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bifunctor_functor(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Bifunctor_functor___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompl_bifunctor___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_ctor_get(x_1, 0);
lean_inc(x_11);
lean_dec_ref(x_1);
x_12 = lean_ctor_get(x_2, 0);
lean_inc(x_12);
lean_dec_ref(x_2);
x_13 = lean_apply_3(x_11, lean_box(0), lean_box(0), x_8);
x_14 = lean_apply_3(x_12, lean_box(0), lean_box(0), x_9);
x_15 = lean_apply_7(x_3, lean_box(0), lean_box(0), lean_box(0), lean_box(0), x_13, x_14, x_10);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompl_bifunctor___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Function_bicompl_bifunctor___redArg___lam__0), 10, 3);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompl_bifunctor(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Function_bicompl_bifunctor___redArg(x_2, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompr_bifunctor___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_10 = lean_ctor_get(x_1, 0);
lean_inc(x_10);
lean_dec_ref(x_1);
x_11 = lean_apply_6(x_2, lean_box(0), lean_box(0), lean_box(0), lean_box(0), x_7, x_8);
x_12 = lean_apply_4(x_10, lean_box(0), lean_box(0), x_11, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompr_bifunctor___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Function_bicompr_bifunctor___redArg___lam__0), 9, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompr_bifunctor(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Function_bicompr_bifunctor___redArg(x_2, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Control_Bifunctor(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Functor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Bifunctor_fst___closed__0 = _init_lp_mathlib_Bifunctor_fst___closed__0();
lean_mark_persistent(lp_mathlib_Bifunctor_fst___closed__0);
lp_mathlib_Prod_bifunctor___closed__0 = _init_lp_mathlib_Prod_bifunctor___closed__0();
lean_mark_persistent(lp_mathlib_Prod_bifunctor___closed__0);
lp_mathlib_Prod_bifunctor = _init_lp_mathlib_Prod_bifunctor();
lean_mark_persistent(lp_mathlib_Prod_bifunctor);
lp_mathlib_Bifunctor_const = _init_lp_mathlib_Bifunctor_const();
lean_mark_persistent(lp_mathlib_Bifunctor_const);
lp_mathlib_Sum_bifunctor___closed__0 = _init_lp_mathlib_Sum_bifunctor___closed__0();
lean_mark_persistent(lp_mathlib_Sum_bifunctor___closed__0);
lp_mathlib_Sum_bifunctor = _init_lp_mathlib_Sum_bifunctor();
lean_mark_persistent(lp_mathlib_Sum_bifunctor);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
