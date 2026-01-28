// Lean compiler output
// Module: Mathlib.RingTheory.ChainOfDivisors
// Imports: public import Init public import Mathlib.Algebra.GCDMonoid.Basic public import Mathlib.Algebra.IsPrimePow public import Mathlib.RingTheory.UniqueFactorizationDomain.Multiplicity public import Mathlib.Order.Atoms public import Mathlib.Order.Hom.Bounded
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
LEAN_EXPORT lean_object* lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_associatesEquivOfUniqueUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_associatesEquivOfUniqueUnits___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
lean_dec_ref(x_2);
x_7 = lean_apply_1(x_5, x_3);
x_8 = lean_apply_1(x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lp_mathlib_associatesEquivOfUniqueUnits___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_Equiv_symm___redArg(x_2);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_5, x_3);
x_9 = lean_apply_1(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_3);
x_4 = lean_alloc_closure((void*)(lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv___redArg___lam__0), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv___redArg___lam__1), 3, 2);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv___redArg(x_2, x_4, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_mkFactorOrderIsoOfFactorDvdEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_8);
lean_dec(x_7);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_IsPrimePow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_UniqueFactorizationDomain_Multiplicity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Atoms(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Bounded(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_ChainOfDivisors(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_IsPrimePow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_UniqueFactorizationDomain_Multiplicity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Atoms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
