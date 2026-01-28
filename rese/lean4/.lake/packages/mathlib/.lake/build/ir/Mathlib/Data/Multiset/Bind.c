// Lean compiler output
// Module: Mathlib.Data.Multiset.Bind
// Imports: public import Init public import Mathlib.Algebra.BigOperators.Group.Multiset.Basic
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
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSProd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_join___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_bind___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg___lam__0(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Multiset_instSProd___closed__0;
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_join(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_bind(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_List_appendTR___redArg), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg___closed__0;
x_3 = lean_box(0);
x_4 = l_List_foldrTR___redArg(x_2, x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_join(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_join___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_bind___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Multiset_map___redArg(x_2, x_1);
x_4 = lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_bind(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Multiset_bind___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Multiset_product___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_Multiset_map___redArg(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Multiset_product___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_Multiset_bind___redArg(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Multiset_product___redArg(x_3, x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Multiset_instSProd___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Multiset_product), 4, 2);
lean_closure_set(x_1, 0, lean_box(0));
lean_closure_set(x_1, 1, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSProd(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_instSProd___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Multiset_sigma___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lean_apply_1(x_1, x_2);
x_5 = lp_mathlib_Multiset_map___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Multiset_sigma___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_Multiset_bind___redArg(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Multiset_sigma___redArg(x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg___closed__0 = _init_lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg___closed__0);
lp_mathlib_Multiset_instSProd___closed__0 = _init_lp_mathlib_Multiset_instSProd___closed__0();
lean_mark_persistent(lp_mathlib_Multiset_instSProd___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
