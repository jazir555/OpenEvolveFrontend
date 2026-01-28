// Lean compiler output
// Module: Mathlib.Order.Interval.Finset.Fin
// Imports: public import Init public import Mathlib.Data.Finset.Fin public import Mathlib.Order.Interval.Finset.Nat public import Mathlib.Order.Interval.Set.Fin
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
lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot___boxed(lean_object*);
lean_object* lp_mathlib_Fin_instHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder(lean_object*);
extern lean_object* lp_mathlib_Nat_instLocallyFiniteOrder;
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___boxed(lean_object*);
lean_object* lp_mathlib_Finset_Ioc___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___boxed(lean_object*);
static lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__3(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_instPartialOrder(lean_object*);
lean_object* lp_mathlib_Finset_Ico___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop(lean_object*);
lean_object* lp_mathlib_Finset_Icc___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot___lam__0(lean_object*);
lean_object* lp_mathlib_Finset_attachFin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__2(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_instCoheytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Finset_Ioo___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Icc___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_Finset_attachFin___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Ico___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_Finset_attachFin___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Ioc___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_Finset_attachFin___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Ioo___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_Finset_attachFin___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_Nat_instLocallyFiniteOrder;
x_3 = lean_alloc_closure((void*)(lp_mathlib_Fin_instLocallyFiniteOrder___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Fin_instLocallyFiniteOrder___lam__1), 3, 1);
lean_closure_set(x_4, 0, x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Fin_instLocallyFiniteOrder___lam__2), 3, 1);
lean_closure_set(x_5, 0, x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Fin_instLocallyFiniteOrder___lam__3), 3, 1);
lean_closure_set(x_6, 0, x_2);
x_7 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_7, 0, x_3);
lean_ctor_set(x_7, 1, x_4);
lean_ctor_set(x_7, 2, x_5);
lean_ctor_set(x_7, 3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_instLocallyFiniteOrder(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot___lam__0(lean_object* x_1) {
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_instLocallyFiniteOrderBot___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_nat_dec_eq(x_1, x_2);
if (x_3 == 1)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Fin_instLocallyFiniteOrderBot___lam__0___boxed), 1, 0);
lean_inc_ref(x_4);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_6 = lean_unsigned_to_nat(1u);
x_7 = lean_nat_sub(x_1, x_6);
x_8 = lean_nat_add(x_7, x_6);
lean_dec(x_7);
lean_inc(x_8);
x_9 = lp_mathlib_Fin_instPartialOrder(x_8);
lean_inc(x_8);
x_10 = lp_mathlib_Fin_instHeytingAlgebra___redArg(x_8);
x_11 = lean_ctor_get(x_10, 1);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lp_mathlib_Fin_instLocallyFiniteOrder(x_8);
lean_dec(x_8);
x_13 = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(x_9, x_11, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_instLocallyFiniteOrderBot(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Fin_instLocallyFiniteOrderBot___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__0;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_nat_dec_eq(x_1, x_2);
if (x_3 == 1)
{
lean_object* x_4; 
x_4 = lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__1;
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_unsigned_to_nat(1u);
x_6 = lean_nat_sub(x_1, x_5);
x_7 = lean_nat_add(x_6, x_5);
lean_dec(x_6);
x_8 = lp_mathlib_Fin_instLocallyFiniteOrder(x_7);
x_9 = lp_mathlib_Fin_instCoheytingAlgebra___redArg(x_7);
x_10 = lean_ctor_get(x_9, 1);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(x_8, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_instLocallyFiniteOrderTop(x_1);
lean_dec(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Fin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Fin(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Fin(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__0 = _init_lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__0();
lean_mark_persistent(lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__0);
lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__1 = _init_lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__1();
lean_mark_persistent(lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
