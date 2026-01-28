// Lean compiler output
// Module: Mathlib.Data.Multiset.AddSub
// Imports: public import Init public import Mathlib.Data.Multiset.Count public import Mathlib.Data.List.Count
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
LEAN_EXPORT lean_object* lp_mathlib_Multiset_add(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static lean_object* lp_mathlib_Multiset_instAdd___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSub___redArg(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_eraseTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Multiset_erase___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sub___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_List_diff___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_add___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_erase(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sub(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instAdd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_erase___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_add(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = l_List_appendTR___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_add___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_List_appendTR___redArg(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Multiset_instAdd___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Multiset_add), 3, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instAdd(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiset_instAdd___closed__0;
return x_2;
}
}
static lean_object* _init_lp_mathlib_Multiset_erase___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_erase___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = l_instBEqOfDecidableEq___redArg(x_1);
x_5 = lp_mathlib_Multiset_erase___redArg___closed__0;
lean_inc(x_2);
x_6 = l___private_Init_Data_List_Impl_0__List_eraseTR_go(lean_box(0), x_4, x_2, x_3, x_2, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_erase(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Multiset_erase___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sub___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = l_instBEqOfDecidableEq___redArg(x_1);
x_5 = lp_batteries_List_diff___redArg(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sub(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Multiset_sub___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSub(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Multiset_sub), 4, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSub___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiset_sub), 4, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Count(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Count(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_AddSub(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Multiset_instAdd___closed__0 = _init_lp_mathlib_Multiset_instAdd___closed__0();
lean_mark_persistent(lp_mathlib_Multiset_instAdd___closed__0);
lp_mathlib_Multiset_erase___redArg___closed__0 = _init_lp_mathlib_Multiset_erase___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Multiset_erase___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
