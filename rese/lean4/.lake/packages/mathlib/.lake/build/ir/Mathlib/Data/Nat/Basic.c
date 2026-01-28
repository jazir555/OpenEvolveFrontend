// Lean compiler output
// Module: Mathlib.Data.Nat.Basic
// Imports: public import Init public import Mathlib.Data.Nat.Init public import Mathlib.Logic.Basic public import Mathlib.Logic.Nontrivial.Defs public import Mathlib.Order.Defs.LinearOrder public import Mathlib.Tactic.GCongr.Core public import Mathlib.Util.AssertExists
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
static lean_object* lp_mathlib_Nat_instLinearOrder___closed__6;
lean_object* l_Nat_decLt___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Nat_instLinearOrder___closed__3;
static lean_object* lp_mathlib_Nat_instLinearOrder___closed__5;
static lean_object* lp_mathlib_Nat_instLinearOrder___closed__1;
static lean_object* lp_mathlib_Nat_instLinearOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instPreorder;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instPartialOrder;
lean_object* l_Nat_decLe___boxed(lean_object*, lean_object*);
lean_object* l_instMinNat___lam__0___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Nat_instLinearOrder___closed__4;
lean_object* l_instDecidableEqNat___boxed(lean_object*, lean_object*);
lean_object* l_instOrdNat___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Nat_instMax___lam__0___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Nat_instLinearOrder___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLinearOrder;
static lean_object* _init_lp_mathlib_Nat_instLinearOrder___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instMinNat___lam__0___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instLinearOrder___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_instMax___lam__0___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instLinearOrder___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instOrdNat___lam__0___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instLinearOrder___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Nat_instLinearOrder___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_decLe___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instLinearOrder___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_decLt___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instLinearOrder___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_1 = lp_mathlib_Nat_instLinearOrder___closed__5;
x_2 = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
x_3 = lp_mathlib_Nat_instLinearOrder___closed__4;
x_4 = lp_mathlib_Nat_instLinearOrder___closed__2;
x_5 = lp_mathlib_Nat_instLinearOrder___closed__1;
x_6 = lp_mathlib_Nat_instLinearOrder___closed__0;
x_7 = lp_mathlib_Nat_instLinearOrder___closed__3;
x_8 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_5);
lean_ctor_set(x_8, 3, x_4);
lean_ctor_set(x_8, 4, x_3);
lean_ctor_set(x_8, 5, x_2);
lean_ctor_set(x_8, 6, x_1);
return x_8;
}
}
static lean_object* _init_lp_mathlib_Nat_instLinearOrder() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_instLinearOrder___closed__6;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_instPreorder() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instLinearOrder;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_instPartialOrder() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instLinearOrder;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Nontrivial_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Defs_LinearOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GCongr_Core(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AssertExists(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Nontrivial_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Defs_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GCongr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AssertExists(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instLinearOrder___closed__0 = _init_lp_mathlib_Nat_instLinearOrder___closed__0();
lean_mark_persistent(lp_mathlib_Nat_instLinearOrder___closed__0);
lp_mathlib_Nat_instLinearOrder___closed__1 = _init_lp_mathlib_Nat_instLinearOrder___closed__1();
lean_mark_persistent(lp_mathlib_Nat_instLinearOrder___closed__1);
lp_mathlib_Nat_instLinearOrder___closed__2 = _init_lp_mathlib_Nat_instLinearOrder___closed__2();
lean_mark_persistent(lp_mathlib_Nat_instLinearOrder___closed__2);
lp_mathlib_Nat_instLinearOrder___closed__3 = _init_lp_mathlib_Nat_instLinearOrder___closed__3();
lean_mark_persistent(lp_mathlib_Nat_instLinearOrder___closed__3);
lp_mathlib_Nat_instLinearOrder___closed__4 = _init_lp_mathlib_Nat_instLinearOrder___closed__4();
lean_mark_persistent(lp_mathlib_Nat_instLinearOrder___closed__4);
lp_mathlib_Nat_instLinearOrder___closed__5 = _init_lp_mathlib_Nat_instLinearOrder___closed__5();
lean_mark_persistent(lp_mathlib_Nat_instLinearOrder___closed__5);
lp_mathlib_Nat_instLinearOrder___closed__6 = _init_lp_mathlib_Nat_instLinearOrder___closed__6();
lean_mark_persistent(lp_mathlib_Nat_instLinearOrder___closed__6);
lp_mathlib_Nat_instLinearOrder = _init_lp_mathlib_Nat_instLinearOrder();
lean_mark_persistent(lp_mathlib_Nat_instLinearOrder);
lp_mathlib_Nat_instPreorder = _init_lp_mathlib_Nat_instPreorder();
lean_mark_persistent(lp_mathlib_Nat_instPreorder);
lp_mathlib_Nat_instPartialOrder = _init_lp_mathlib_Nat_instPartialOrder();
lean_mark_persistent(lp_mathlib_Nat_instPartialOrder);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
