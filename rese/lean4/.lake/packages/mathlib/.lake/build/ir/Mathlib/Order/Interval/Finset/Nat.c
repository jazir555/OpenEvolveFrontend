// Lean compiler output
// Module: Mathlib.Order.Interval.Finset.Nat
// Imports: public import Init public import Mathlib.Algebra.Group.Embedding public import Mathlib.Order.Interval.Finset.SuccPred public import Mathlib.Order.Interval.Multiset
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
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__2(lean_object*, lean_object*);
lean_object* l_List_range_x27TR_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instUniqueSubtypeMemFinsetIicOfNat;
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__3___boxed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_unsigned_to_nat(1u);
x_4 = lean_nat_add(x_2, x_3);
x_5 = lean_nat_sub(x_4, x_1);
lean_dec(x_4);
x_6 = lean_nat_add(x_1, x_5);
x_7 = lean_box(0);
x_8 = l_List_range_x27TR_go(x_3, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_instLocallyFiniteOrder___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_nat_sub(x_2, x_1);
x_4 = lean_unsigned_to_nat(1u);
x_5 = lean_nat_add(x_1, x_3);
x_6 = lean_box(0);
x_7 = l_List_range_x27TR_go(x_4, x_3, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_instLocallyFiniteOrder___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_unsigned_to_nat(1u);
x_4 = lean_nat_add(x_1, x_3);
x_5 = lean_nat_sub(x_2, x_1);
x_6 = lean_nat_add(x_4, x_5);
lean_dec(x_4);
x_7 = lean_box(0);
x_8 = l_List_range_x27TR_go(x_3, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_instLocallyFiniteOrder___lam__2(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__3(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lean_unsigned_to_nat(1u);
x_4 = lean_nat_add(x_1, x_3);
x_5 = lean_nat_sub(x_2, x_1);
x_6 = lean_nat_sub(x_5, x_3);
lean_dec(x_5);
x_7 = lean_nat_add(x_4, x_6);
lean_dec(x_4);
x_8 = lean_box(0);
x_9 = l_List_range_x27TR_go(x_3, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__3___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_instLocallyFiniteOrder___lam__3(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Nat_instLocallyFiniteOrder() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_instLocallyFiniteOrder___lam__0___boxed), 2, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Nat_instLocallyFiniteOrder___lam__1___boxed), 2, 0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Nat_instLocallyFiniteOrder___lam__2___boxed), 2, 0);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Nat_instLocallyFiniteOrder___lam__3___boxed), 2, 0);
x_5 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_2);
lean_ctor_set(x_5, 2, x_3);
lean_ctor_set(x_5, 3, x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Nat_instUniqueSubtypeMemFinsetIicOfNat() {
_start:
{
lean_object* x_1; 
x_1 = lean_unsigned_to_nat(0u);
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Embedding(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Multiset(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Embedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instLocallyFiniteOrder = _init_lp_mathlib_Nat_instLocallyFiniteOrder();
lean_mark_persistent(lp_mathlib_Nat_instLocallyFiniteOrder);
lp_mathlib_Nat_instUniqueSubtypeMemFinsetIicOfNat = _init_lp_mathlib_Nat_instUniqueSubtypeMemFinsetIicOfNat();
lean_mark_persistent(lp_mathlib_Nat_instUniqueSubtypeMemFinsetIicOfNat);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
