// Lean compiler output
// Module: Mathlib.Data.Multiset.Powerset
// Imports: public import Init public import Mathlib.Data.List.Sublists public import Mathlib.Data.List.Zip public import Mathlib.Data.Multiset.Bind public import Mathlib.Data.Multiset.Range
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
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux_x27(lean_object*, lean_object*);
lean_object* lp_batteries_List_sublistsFast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powerset(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_List_sublists_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCard(lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCard___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_List_sublistsLenAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powerset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
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
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; 
x_5 = lean_ctor_get(x_1, 1);
lean_ctor_set(x_1, 1, x_2);
{
lean_object* _tmp_0 = x_5;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_1);
x_9 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_2);
x_1 = x_8;
x_2 = x_9;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_batteries_List_sublistsFast___redArg(x_1);
x_3 = lean_box(0);
x_4 = lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_powersetAux___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux_x27___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_batteries_List_sublists_x27___redArg(x_1);
x_3 = lean_box(0);
x_4 = lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux_x27(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_powersetAux_x27___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powerset(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_powersetAux___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powerset___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiset_powersetAux___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiset_powersetCardAux___redArg___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Multiset_powersetCardAux___redArg___lam__0___boxed), 1, 0);
x_4 = lean_box(0);
x_5 = lp_mathlib_List_sublistsLenAux___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Multiset_powersetCardAux___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCard(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Multiset_powersetCardAux___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCard___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_powersetCardAux___redArg(x_1, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Sublists(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Zip(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Range(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Powerset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Sublists(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Zip(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
