// Lean compiler output
// Module: Mathlib.Data.Finset.Option
// Imports: public import Init public import Mathlib.Data.Finset.Card public import Mathlib.Data.Finset.Union
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
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1___redArg(lean_object*);
static lean_object* lp_mathlib_Finset_insertNone___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Finset_eraseNone(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom___at___00Finset_eraseNone_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1___redArg(lean_object*);
lean_object* lp_mathlib_Finset_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1_spec__1___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom___at___00Finset_eraseNone_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_optionIsSomeEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_toFinset(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___at___00Finset_eraseNone_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom___at___00Finset_eraseNone_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertNone___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_toFinset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_toFinset___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___at___00Finset_eraseNone_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertNone(lean_object*);
static lean_object* lp_mathlib_Finset_eraseNone___closed__2;
lean_object* lp_mathlib_Equiv_toEmbedding___redArg(lean_object*);
static lean_object* lp_mathlib_Finset_eraseNone___closed__0;
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
lean_object* lp_mathlib_Function_Embedding_some___lam__0(lean_object*);
static lean_object* lp_mathlib_Finset_eraseNone___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_toFinset___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___at___00Finset_eraseNone_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_toFinset___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_box(0);
lean_inc(x_3);
x_5 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_toFinset(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Option_toFinset___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_toFinset___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Option_toFinset(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_toFinset___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Option_toFinset___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_insertNone___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_some___lam__0), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertNone___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_box(0);
x_3 = lp_mathlib_Finset_insertNone___lam__0___closed__0;
x_4 = lp_mathlib_Finset_map___redArg(x_3, x_1);
x_5 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_insertNone(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Finset_insertNone___lam__0), 1, 0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1_spec__1___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_1, 0);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; 
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec_ref(x_1);
x_1 = x_5;
goto _start;
}
else
{
uint8_t x_7; 
lean_inc_ref(x_4);
x_7 = !lean_is_exclusive(x_1);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_1, 1);
x_9 = lean_ctor_get(x_1, 0);
lean_dec(x_9);
lean_ctor_set(x_1, 1, x_2);
{
lean_object* _tmp_0 = x_8;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_ctor_get(x_1, 1);
lean_inc(x_11);
lean_dec(x_1);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_4);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_11;
x_2 = x_12;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_box(0);
x_3 = lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1_spec__1___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1___redArg___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1___redArg___lam__0___boxed), 1, 0);
x_3 = lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1___redArg(x_1);
x_4 = lp_mathlib_Multiset_attach___redArg(x_3);
x_5 = lp_mathlib_Finset_map___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1_spec__1___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Finset_subtype___at___00Finset_eraseNone_spec__1_spec__1_spec__1___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom___at___00Finset_eraseNone_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom___at___00Finset_eraseNone_spec__0___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_OrderEmbedding_toOrderHom___at___00Finset_eraseNone_spec__0___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom___at___00Finset_eraseNone_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderEmbedding_toOrderHom___at___00Finset_eraseNone_spec__0___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___at___00Finset_eraseNone_spec__5___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___at___00Finset_eraseNone_spec__5___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_OrderHom_comp___at___00Finset_eraseNone_spec__5___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___at___00Finset_eraseNone_spec__5(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderHom_comp___at___00Finset_eraseNone_spec__5___redArg(x_2, x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_eraseNone___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_optionIsSomeEquiv(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_eraseNone___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_eraseNone___closed__0;
x_2 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_eraseNone___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_eraseNone___closed__1;
x_2 = lean_alloc_closure((void*)(lp_mathlib_Finset_map), 4, 3);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_eraseNone(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Finset_subtype___at___00Finset_eraseNone_spec__1___redArg), 1, 0);
x_3 = lp_mathlib_Finset_eraseNone___closed__2;
x_4 = lp_mathlib_OrderEmbedding_toOrderHom___at___00Finset_eraseNone_spec__0___redArg(x_3);
x_5 = lp_mathlib_OrderHom_comp___at___00Finset_eraseNone_spec__5___redArg(x_4, x_2);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Union(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Option(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Finset_insertNone___lam__0___closed__0 = _init_lp_mathlib_Finset_insertNone___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Finset_insertNone___lam__0___closed__0);
lp_mathlib_Finset_eraseNone___closed__0 = _init_lp_mathlib_Finset_eraseNone___closed__0();
lean_mark_persistent(lp_mathlib_Finset_eraseNone___closed__0);
lp_mathlib_Finset_eraseNone___closed__1 = _init_lp_mathlib_Finset_eraseNone___closed__1();
lean_mark_persistent(lp_mathlib_Finset_eraseNone___closed__1);
lp_mathlib_Finset_eraseNone___closed__2 = _init_lp_mathlib_Finset_eraseNone___closed__2();
lean_mark_persistent(lp_mathlib_Finset_eraseNone___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
