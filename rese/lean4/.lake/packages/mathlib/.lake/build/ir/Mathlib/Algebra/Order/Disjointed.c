// Lean compiler output
// Module: Mathlib.Algebra.Order.Disjointed
// Imports: public import Init public import Mathlib.Algebra.Order.SuccPred.PartialSups public import Mathlib.Data.Nat.SuccPred public import Mathlib.Order.Disjointed
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
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_instLocallyFiniteOrder;
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_partialSups___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_instPreorder;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Nat_disjointedRec___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Nat_disjointedRec___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Nat_instLocallyFiniteOrder;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_mathlib_Nat_instPreorder;
x_4 = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(x_3, x_2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lp_mathlib_partialSups___redArg(x_1, x_2, x_3);
lean_inc(x_8);
x_11 = lean_apply_1(x_10, x_8);
x_12 = lean_apply_2(x_4, x_5, x_11);
x_13 = lean_nat_add(x_8, x_6);
lean_dec(x_8);
x_14 = lean_apply_3(x_7, x_12, x_13, x_9);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Nat_disjointedRec___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_6);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_6 = lean_unsigned_to_nat(0u);
x_7 = lp_mathlib_Nat_disjointedRec___redArg___closed__0;
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
x_9 = lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg(x_1);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_11);
lean_dec_ref(x_10);
x_12 = lean_nat_dec_eq(x_4, x_6);
if (x_12 == 1)
{
lean_dec_ref(x_11);
lean_dec(x_8);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_13 = lean_unsigned_to_nat(1u);
x_14 = lean_nat_sub(x_4, x_13);
x_15 = lean_nat_add(x_14, x_13);
lean_inc(x_2);
x_16 = lean_apply_1(x_2, x_15);
lean_inc(x_3);
lean_inc(x_16);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Nat_disjointedRec___redArg___lam__0___boxed), 9, 7);
lean_closure_set(x_17, 0, x_11);
lean_closure_set(x_17, 1, x_7);
lean_closure_set(x_17, 2, x_2);
lean_closure_set(x_17, 3, x_8);
lean_closure_set(x_17, 4, x_16);
lean_closure_set(x_17, 5, x_13);
lean_closure_set(x_17, 6, x_3);
x_18 = lean_apply_3(x_3, x_16, x_6, x_5);
x_19 = l_Nat_recCompiled___redArg(x_18, x_17, x_14);
lean_dec(x_14);
lean_dec(x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Nat_disjointedRec___redArg(x_2, x_3, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Nat_disjointedRec(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_disjointedRec___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Nat_disjointedRec___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_SuccPred_PartialSups(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Disjointed(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Disjointed(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_SuccPred_PartialSups(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Disjointed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_disjointedRec___redArg___closed__0 = _init_lp_mathlib_Nat_disjointedRec___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Nat_disjointedRec___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
