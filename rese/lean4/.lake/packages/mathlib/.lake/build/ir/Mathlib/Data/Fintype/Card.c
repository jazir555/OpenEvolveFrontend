// Lean compiler output
// Module: Mathlib.Data.Fintype.Card
// Imports: public import Init public import Mathlib.Data.Finset.Card public import Mathlib.Data.Fintype.Basic
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
lean_object* l_List_lengthTR___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_equivEmptyEquiv(lean_object*);
static lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___redArg___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
static lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1;
static lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_List_lengthTR___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_List_lengthTR___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fintype_card(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fintype_card___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_equivEmptyEquiv(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, lean_box(0));
lean_ctor_set(x_2, 1, lean_box(0));
x_3 = lp_mathlib_Equiv_trans___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_truncOfCardPos(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_truncOfCardPos___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Card(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0 = _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0();
lean_mark_persistent(lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0);
lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1 = _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1();
lean_mark_persistent(lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1);
lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2 = _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2();
lean_mark_persistent(lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
