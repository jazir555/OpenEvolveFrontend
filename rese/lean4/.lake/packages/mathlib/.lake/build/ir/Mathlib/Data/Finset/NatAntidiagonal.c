// Lean compiler output
// Module: Mathlib.Data.Finset.NatAntidiagonal
// Imports: public import Init public import Mathlib.Algebra.Order.Antidiag.Prod public import Mathlib.Algebra.Order.Group.Nat public import Mathlib.Data.Multiset.NatAntidiagonal
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
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin(lean_object*);
lean_object* lp_mathlib_List_Nat_antidiagonal___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_instHasAntidiagonal;
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0___boxed(lean_object*);
static lean_object* lp_mathlib_Finset_Nat_instHasAntidiagonal___closed__0;
static lean_object* _init_lp_mathlib_Finset_Nat_instHasAntidiagonal___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_List_Nat_antidiagonal___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_Nat_instHasAntidiagonal() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Finset_Nat_instHasAntidiagonal___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_nat_sub(x_1, x_2);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0___boxed), 1, 0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1___boxed), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Antidiag_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_NatAntidiagonal(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_NatAntidiagonal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Antidiag_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_NatAntidiagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Finset_Nat_instHasAntidiagonal___closed__0 = _init_lp_mathlib_Finset_Nat_instHasAntidiagonal___closed__0();
lean_mark_persistent(lp_mathlib_Finset_Nat_instHasAntidiagonal___closed__0);
lp_mathlib_Finset_Nat_instHasAntidiagonal = _init_lp_mathlib_Finset_Nat_instHasAntidiagonal();
lean_mark_persistent(lp_mathlib_Finset_Nat_instHasAntidiagonal);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
