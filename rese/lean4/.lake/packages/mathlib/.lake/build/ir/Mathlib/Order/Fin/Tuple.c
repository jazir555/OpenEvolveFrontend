// Lean compiler output
// Module: Mathlib.Order.Fin.Tuple
// Imports: public import Init public import Mathlib.Data.Fin.VecNotation public import Mathlib.Logic.Equiv.Fin.Basic public import Mathlib.Order.Fin.Basic public import Mathlib.Order.PiLex public import Mathlib.Order.Interval.Set.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_piFinTwoIso___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consOrderIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocOrderIso(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocOrderIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_finTwoArrowIso___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthOrderIso(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthOrderIso___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_OrderIso_finTwoArrowIso___closed__0;
static lean_object* lp_mathlib_OrderIso_piFinTwoIso___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderIso___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_finTwoArrowIso(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consOrderIso___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_finSuccAboveEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_insertNthEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocOrderIso___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_snocEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthOrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_piFinTwoIso(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_consEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderIso___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderIso(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consOrderIso(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderIso___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_piFinTwoEquiv(lean_object*);
lean_object* lp_mathlib_finTwoArrowEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSuccAboveOrderIso(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_OrderIso_piFinTwoIso___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_piFinTwoEquiv(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_piFinTwoIso(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderIso_piFinTwoIso___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_piFinTwoIso___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderIso_piFinTwoIso(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_OrderIso_finTwoArrowIso___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_finTwoArrowEquiv(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_finTwoArrowIso(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderIso_finTwoArrowIso___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_finTwoArrowIso___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderIso_finTwoArrowIso(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consOrderIso(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Fin_consEquiv___redArg(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consOrderIso___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_consEquiv___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consOrderIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Fin_consOrderIso(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocOrderIso(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Fin_snocEquiv___redArg(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocOrderIso___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_snocEquiv___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocOrderIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Fin_snocOrderIso(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthOrderIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Fin_insertNthEquiv___redArg(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthOrderIso___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_insertNthEquiv___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthOrderIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Fin_insertNthOrderIso(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSuccAboveOrderIso(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_finSuccAboveEquiv(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderIso___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderIso___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_castLEOrderIso___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderIso(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Fin_castLEOrderIso___lam__0___boxed), 1, 0);
lean_inc_ref(x_4);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Fin_castLEOrderIso(x_1, x_2, x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Fin_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_PiLex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Fin_Tuple(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_PiLex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_OrderIso_piFinTwoIso___closed__0 = _init_lp_mathlib_OrderIso_piFinTwoIso___closed__0();
lean_mark_persistent(lp_mathlib_OrderIso_piFinTwoIso___closed__0);
lp_mathlib_OrderIso_finTwoArrowIso___closed__0 = _init_lp_mathlib_OrderIso_finTwoArrowIso___closed__0();
lean_mark_persistent(lp_mathlib_OrderIso_finTwoArrowIso___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
