// Lean compiler output
// Module: Mathlib.Data.Int.Interval
// Imports: public import Init public import Mathlib.Algebra.Group.Embedding public import Mathlib.Algebra.Ring.CharZero public import Mathlib.Algebra.Ring.Int.Defs public import Mathlib.Algebra.Order.Group.Unbundled.Int public import Mathlib.Order.Interval.Finset.Basic
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
lean_object* l_Int_add___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Int_instLocallyFiniteOrder___closed__1;
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* l_List_range(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder;
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_addLeftEmbedding___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Int_instCommRing;
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Int_instLocallyFiniteOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Int_toNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_castEmbedding___redArg(lean_object*);
static lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lp_mathlib_Nat_castEmbedding___redArg(x_1);
lean_inc(x_3);
x_6 = lp_mathlib_addLeftEmbedding___redArg(x_2, x_3);
x_7 = lp_mathlib_Function_Embedding_trans___redArg(x_5, x_6);
x_8 = lean_int_sub(x_4, x_3);
lean_dec(x_3);
x_9 = l_Int_toNat(x_8);
lean_dec(x_8);
x_10 = l_List_range(x_9);
x_11 = lp_mathlib_Finset_map___redArg(x_7, x_10);
return x_11;
}
}
static lean_object* _init_lp_mathlib_Int_instLocallyFiniteOrder___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Int_instCommRing;
x_2 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_instLocallyFiniteOrder___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Int_add___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_instLocallyFiniteOrder___lam__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_5 = lp_mathlib_Nat_castEmbedding___redArg(x_1);
lean_inc(x_3);
x_6 = lp_mathlib_addLeftEmbedding___redArg(x_2, x_3);
x_7 = lp_mathlib_Function_Embedding_trans___redArg(x_5, x_6);
x_8 = lp_mathlib_Int_instLocallyFiniteOrder___lam__0___closed__0;
x_9 = lean_int_add(x_4, x_8);
x_10 = lean_int_sub(x_9, x_3);
lean_dec(x_3);
lean_dec(x_9);
x_11 = l_Int_toNat(x_10);
lean_dec(x_10);
x_12 = l_List_range(x_11);
x_13 = lp_mathlib_Finset_map___redArg(x_7, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Int_instLocallyFiniteOrder___lam__0(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Int_instLocallyFiniteOrder___lam__1(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_5 = lp_mathlib_Nat_castEmbedding___redArg(x_1);
x_6 = lp_mathlib_Int_instLocallyFiniteOrder___lam__0___closed__0;
x_7 = lean_int_add(x_3, x_6);
x_8 = lp_mathlib_addLeftEmbedding___redArg(x_2, x_7);
x_9 = lp_mathlib_Function_Embedding_trans___redArg(x_5, x_8);
x_10 = lean_int_sub(x_4, x_3);
x_11 = l_Int_toNat(x_10);
lean_dec(x_10);
x_12 = l_List_range(x_11);
x_13 = lp_mathlib_Finset_map___redArg(x_9, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Int_instLocallyFiniteOrder___lam__2(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_5 = lp_mathlib_Nat_castEmbedding___redArg(x_1);
x_6 = lp_mathlib_Int_instLocallyFiniteOrder___lam__0___closed__0;
x_7 = lean_int_add(x_3, x_6);
x_8 = lp_mathlib_addLeftEmbedding___redArg(x_2, x_7);
x_9 = lp_mathlib_Function_Embedding_trans___redArg(x_5, x_8);
x_10 = lean_int_sub(x_4, x_3);
x_11 = lean_int_sub(x_10, x_6);
lean_dec(x_10);
x_12 = l_Int_toNat(x_11);
lean_dec(x_11);
x_13 = l_List_range(x_12);
x_14 = lp_mathlib_Finset_map___redArg(x_9, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instLocallyFiniteOrder___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Int_instLocallyFiniteOrder___lam__3(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Int_instLocallyFiniteOrder() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_1 = lp_mathlib_Int_instLocallyFiniteOrder___closed__0;
x_2 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_2);
x_3 = lp_mathlib_Int_instLocallyFiniteOrder___closed__1;
lean_inc_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Int_instLocallyFiniteOrder___lam__0___boxed), 4, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
lean_inc_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Int_instLocallyFiniteOrder___lam__1___boxed), 4, 2);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_3);
lean_inc_ref(x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Int_instLocallyFiniteOrder___lam__2___boxed), 4, 2);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_3);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Int_instLocallyFiniteOrder___lam__3___boxed), 4, 2);
lean_closure_set(x_7, 0, x_2);
lean_closure_set(x_7, 1, x_3);
x_8 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_8, 0, x_4);
lean_ctor_set(x_8, 1, x_5);
lean_ctor_set(x_8, 2, x_6);
lean_ctor_set(x_8, 3, x_7);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Embedding(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_CharZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_Interval(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Embedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_instLocallyFiniteOrder___closed__0 = _init_lp_mathlib_Int_instLocallyFiniteOrder___closed__0();
lean_mark_persistent(lp_mathlib_Int_instLocallyFiniteOrder___closed__0);
lp_mathlib_Int_instLocallyFiniteOrder___closed__1 = _init_lp_mathlib_Int_instLocallyFiniteOrder___closed__1();
lean_mark_persistent(lp_mathlib_Int_instLocallyFiniteOrder___closed__1);
lp_mathlib_Int_instLocallyFiniteOrder___lam__0___closed__0 = _init_lp_mathlib_Int_instLocallyFiniteOrder___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Int_instLocallyFiniteOrder___lam__0___closed__0);
lp_mathlib_Int_instLocallyFiniteOrder = _init_lp_mathlib_Int_instLocallyFiniteOrder();
lean_mark_persistent(lp_mathlib_Int_instLocallyFiniteOrder);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
