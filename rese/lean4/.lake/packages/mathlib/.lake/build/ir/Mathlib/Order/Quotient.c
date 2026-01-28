// Lean compiler output
// Module: Mathlib.Order.Quotient
// Imports: public import Init public import Mathlib.Order.Interval.Set.OrdConnected
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
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instPreorder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableLTOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Quotient_instPreorder___closed__0;
LEAN_EXPORT uint8_t lp_mathlib_Quotient_instLinearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_decidableLTOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_decidableEqOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLE__mathlib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Quotient_instLinearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLE__mathlib(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Quotient_instPreorder___closed__0() {
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
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instPreorder(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Quotient_instPreorder___closed__0;
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Quotient_instLinearOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_1, 4);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
lean_inc(x_4);
lean_inc(x_3);
x_6 = lean_apply_2(x_5, x_3, x_4);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
lean_object* x_8; uint8_t x_9; 
x_8 = lean_apply_2(x_2, x_3, x_4);
x_9 = lean_unbox(x_8);
return x_9;
}
else
{
uint8_t x_10; 
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_10 = lean_unbox(x_6);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Quotient_instLinearOrder___redArg___lam__0(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_1, 4);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
lean_inc(x_4);
lean_inc(x_3);
x_6 = lean_apply_2(x_5, x_3, x_4);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
lean_object* x_8; uint8_t x_9; 
lean_inc(x_4);
lean_inc(x_3);
x_8 = lean_apply_2(x_2, x_3, x_4);
x_9 = lean_unbox(x_8);
if (x_9 == 0)
{
lean_dec(x_4);
return x_3;
}
else
{
lean_dec(x_3);
return x_4;
}
}
else
{
lean_dec(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_1, 4);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
lean_inc(x_4);
lean_inc(x_3);
x_6 = lean_apply_2(x_5, x_3, x_4);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
lean_object* x_8; uint8_t x_9; 
lean_inc(x_4);
lean_inc(x_3);
x_8 = lean_apply_2(x_2, x_3, x_4);
x_9 = lean_unbox(x_8);
if (x_9 == 0)
{
lean_dec(x_3);
return x_4;
}
else
{
lean_dec(x_4);
return x_3;
}
}
else
{
lean_dec(x_4);
lean_dec_ref(x_2);
return x_3;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Quotient_instLinearOrder___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
lean_inc(x_3);
lean_inc(x_2);
lean_inc_ref(x_1);
x_4 = lp_mathlib_decidableLTOfDecidableLE___redArg(x_1, x_2, x_3);
if (x_4 == 0)
{
uint8_t x_5; 
x_5 = lp_mathlib_decidableEqOfDecidableLE___redArg(x_1, x_2, x_3);
if (x_5 == 0)
{
uint8_t x_6; 
x_6 = 2;
return x_6;
}
else
{
uint8_t x_7; 
x_7 = 1;
return x_7;
}
}
else
{
uint8_t x_8; 
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_8 = 0;
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder___redArg___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_Quotient_instLinearOrder___redArg___lam__3(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_4 = lp_mathlib_LinearOrder_toLattice___redArg(x_2);
x_5 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Quotient_instLinearOrder___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_3);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Quotient_instLinearOrder___redArg___lam__2), 4, 2);
lean_closure_set(x_9, 0, x_2);
lean_closure_set(x_9, 1, x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Quotient_instLinearOrder___redArg___lam__1), 4, 2);
lean_closure_set(x_10, 0, x_2);
lean_closure_set(x_10, 1, x_3);
lean_inc_ref(x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Quotient_instLinearOrder___redArg___lam__3___boxed), 3, 1);
lean_closure_set(x_11, 0, x_8);
x_12 = lp_mathlib_Quotient_instPreorder(lean_box(0), x_1, x_7);
lean_inc_ref(x_8);
lean_inc_ref(x_12);
x_13 = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(x_13, 0, lean_box(0));
lean_closure_set(x_13, 1, x_12);
lean_closure_set(x_13, 2, x_8);
lean_inc_ref(x_8);
lean_inc_ref(x_12);
x_14 = lean_alloc_closure((void*)(lp_mathlib_decidableLTOfDecidableLE___boxed), 5, 3);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, x_12);
lean_closure_set(x_14, 2, x_8);
x_15 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_15, 0, x_12);
lean_ctor_set(x_15, 1, x_10);
lean_ctor_set(x_15, 2, x_9);
lean_ctor_set(x_15, 3, x_11);
lean_ctor_set(x_15, 4, x_8);
lean_ctor_set(x_15, 5, x_13);
lean_ctor_set(x_15, 6, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_instLinearOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Quotient_instLinearOrder___redArg(x_2, x_3, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Quotient(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Quotient_instPreorder___closed__0 = _init_lp_mathlib_Quotient_instPreorder___closed__0();
lean_mark_persistent(lp_mathlib_Quotient_instPreorder___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
