// Lean compiler output
// Module: Mathlib.Data.Finset.Max
// Imports: public import Init public import Mathlib.Data.Finset.Card public import Mathlib.Data.Finset.Lattice.Fold
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
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_inf_x27___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_some(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___redArg___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Finset_min_x27___redArg___closed__0;
static lean_object* lp_mathlib_Finset_max___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_some(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup_x27___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_inf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Finset_min___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Finset_max___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_WithBot_some), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_WithBot_semilatticeSup___redArg(x_4);
x_6 = lean_box(0);
x_7 = lp_mathlib_Finset_max___redArg___closed__0;
x_8 = lp_mathlib_Finset_sup___redArg(x_5, x_6, x_2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_max___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_max(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_max___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_min___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_WithTop_some), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
x_4 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_3);
x_5 = lp_mathlib_WithTop_semilatticeInf___redArg(x_4);
x_6 = lean_box(0);
x_7 = lp_mathlib_Finset_min___redArg___closed__0;
x_8 = lp_mathlib_Finset_inf___redArg(x_5, x_6, x_2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_min___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_min(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_min___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_min_x27___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
x_4 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_3);
x_5 = lp_mathlib_Finset_min_x27___redArg___closed__0;
x_6 = lp_mathlib_Finset_inf_x27___redArg(x_4, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_min_x27___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_min_x27(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_min_x27___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_Finset_min_x27___redArg___closed__0;
x_6 = lp_mathlib_Finset_sup_x27___redArg(x_4, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_max_x27___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_max_x27(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_max_x27___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Finset_max___redArg___closed__0 = _init_lp_mathlib_Finset_max___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Finset_max___redArg___closed__0);
lp_mathlib_Finset_min___redArg___closed__0 = _init_lp_mathlib_Finset_min___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Finset_min___redArg___closed__0);
lp_mathlib_Finset_min_x27___redArg___closed__0 = _init_lp_mathlib_Finset_min_x27___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Finset_min_x27___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
