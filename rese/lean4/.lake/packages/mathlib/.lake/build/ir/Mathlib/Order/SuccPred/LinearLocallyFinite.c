// Lean compiler output
// Module: Mathlib.Order.SuccPred.LinearLocallyFinite
// Imports: public import Init public import Mathlib.Algebra.Order.Group.Nat public import Mathlib.Data.Countable.Basic public import Mathlib.Data.Finset.Max public import Mathlib.Data.Fintype.Pigeonhole public import Mathlib.Logic.Encodable.Basic public import Mathlib.Order.Interval.Finset.Defs public import Mathlib.Order.SuccPred.Archimedean
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
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Order_succ___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toZ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instNatCastInt___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Order_pred___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_iterate___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_toZ___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Int_toNat(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_toZ___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lean_int_neg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Nat_findX___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_toZ___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_7 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
x_8 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Order_pred___boxed), 4, 3);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_9);
lean_closure_set(x_10, 2, x_2);
x_11 = lp_mathlib_Nat_iterate___redArg(x_10, x_6, x_3);
x_12 = lean_apply_2(x_4, x_11, x_5);
x_13 = lean_unbox(x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; lean_object* x_8; 
x_7 = lp_mathlib_toZ___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_1);
x_8 = lean_box(x_7);
return x_8;
}
}
LEAN_EXPORT uint8_t lp_mathlib_toZ___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_7 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
x_8 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Order_succ___boxed), 4, 3);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_9);
lean_closure_set(x_10, 2, x_2);
x_11 = lp_mathlib_Nat_iterate___redArg(x_10, x_6, x_3);
x_12 = lean_apply_2(x_4, x_11, x_5);
x_13 = lean_unbox(x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; lean_object* x_8; 
x_7 = lp_mathlib_toZ___redArg___lam__1(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_1);
x_8 = lean_box(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_6 = lean_ctor_get(x_1, 4);
x_7 = lean_ctor_get(x_1, 5);
lean_inc_ref(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
x_8 = lean_apply_2(x_6, x_4, x_5);
x_9 = lean_unbox(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_dec(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_toZ___redArg___lam__0___boxed), 6, 5);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_3);
lean_closure_set(x_10, 2, x_4);
lean_closure_set(x_10, 3, x_7);
lean_closure_set(x_10, 4, x_5);
x_11 = lp_mathlib_Nat_findX___redArg(x_10);
x_12 = l_instNatCastInt___lam__0(x_11);
x_13 = lean_int_neg(x_12);
lean_dec(x_12);
return x_13;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
lean_dec(x_3);
x_14 = lean_alloc_closure((void*)(lp_mathlib_toZ___redArg___lam__1___boxed), 6, 5);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_4);
lean_closure_set(x_14, 3, x_7);
lean_closure_set(x_14, 4, x_5);
x_15 = lp_mathlib_Nat_findX___redArg(x_14);
x_16 = l_instNatCastInt___lam__0(x_15);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_toZ(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_toZ___redArg(x_2, x_3, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_toZ___redArg(x_1, x_2, x_3, x_4, x_5);
x_7 = l_Int_toNat(x_6);
lean_dec(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Order_succ___boxed), 4, 3);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, x_1);
lean_closure_set(x_5, 2, x_2);
x_6 = lp_mathlib_Nat_iterate___redArg(x_5, x_4, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
x_6 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_5);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 1);
lean_dec(x_9);
lean_inc(x_4);
lean_inc(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__0), 5, 4);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_2);
lean_closure_set(x_10, 2, x_3);
lean_closure_set(x_10, 3, x_4);
x_11 = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__1), 4, 3);
lean_closure_set(x_11, 0, x_8);
lean_closure_set(x_11, 1, x_2);
lean_closure_set(x_11, 2, x_4);
lean_ctor_set(x_6, 1, x_11);
lean_ctor_set(x_6, 0, x_10);
return x_6;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_ctor_get(x_6, 0);
lean_inc(x_12);
lean_dec(x_6);
lean_inc(x_4);
lean_inc(x_2);
x_13 = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__0), 5, 4);
lean_closure_set(x_13, 0, x_1);
lean_closure_set(x_13, 1, x_2);
lean_closure_set(x_13, 2, x_3);
lean_closure_set(x_13, 3, x_4);
x_14 = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__1), 4, 3);
lean_closure_set(x_14, 0, x_12);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_4);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg(x_2, x_3, x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
x_6 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_5);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 1);
lean_dec(x_9);
lean_inc(x_4);
lean_inc(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__0), 5, 4);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_2);
lean_closure_set(x_10, 2, x_3);
lean_closure_set(x_10, 3, x_4);
x_11 = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__1), 4, 3);
lean_closure_set(x_11, 0, x_8);
lean_closure_set(x_11, 1, x_2);
lean_closure_set(x_11, 2, x_4);
lean_ctor_set(x_6, 1, x_11);
lean_ctor_set(x_6, 0, x_10);
return x_6;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_ctor_get(x_6, 0);
lean_inc(x_12);
lean_dec(x_6);
lean_inc(x_4);
lean_inc(x_2);
x_13 = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__0), 5, 4);
lean_closure_set(x_13, 0, x_1);
lean_closure_set(x_13, 1, x_2);
lean_closure_set(x_13, 2, x_3);
lean_closure_set(x_13, 3, x_4);
x_14 = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__1), 4, 3);
lean_closure_set(x_14, 0, x_12);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_4);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_orderIsoRangeOfLinearSuccPredArch___redArg(x_2, x_3, x_4, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_orderIsoRangeOfLinearSuccPredArch(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Countable_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Encodable_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_SuccPred_LinearLocallyFinite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Countable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Encodable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
