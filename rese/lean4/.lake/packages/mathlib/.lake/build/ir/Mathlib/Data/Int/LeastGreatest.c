// Lean compiler output
// Module: Mathlib.Data.Int.LeastGreatest
// Imports: public import Init public import Mathlib.Algebra.Order.Group.OrderIso public import Mathlib.Algebra.Ring.Int.Defs public import Mathlib.Data.Nat.Find public import Mathlib.Order.Bounds.Defs
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
LEAN_EXPORT uint8_t lp_mathlib_Int_leastOfBdd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_greatestOfBdd___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_greatestOfBdd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leastOfBdd___redArg(lean_object*, lean_object*);
lean_object* l_instNatCastInt___lam__0(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Int_greatestOfBdd___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_greatestOfBdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leastOfBdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leastOfBdd___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_greatestOfBdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_greatestOfBdd___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* lp_mathlib_Nat_findX___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Int_leastOfBdd___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_4 = l_instNatCastInt___lam__0(x_3);
x_5 = lean_int_add(x_1, x_4);
lean_dec(x_4);
x_6 = lean_apply_1(x_2, x_5);
x_7 = lean_unbox(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leastOfBdd___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_Int_leastOfBdd___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_1);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leastOfBdd___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Int_leastOfBdd___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
x_4 = lp_mathlib_Nat_findX___redArg(x_3);
x_5 = l_instNatCastInt___lam__0(x_4);
x_6 = lean_int_add(x_2, x_5);
lean_dec(x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leastOfBdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Int_leastOfBdd___redArg(x_2, x_3);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Int_greatestOfBdd___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_int_neg(x_2);
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_unbox(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_greatestOfBdd___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_Int_greatestOfBdd___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_greatestOfBdd___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Int_greatestOfBdd___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_int_neg(x_2);
x_5 = lp_mathlib_Int_leastOfBdd___redArg(x_3, x_4);
x_6 = lean_int_neg(x_5);
lean_dec(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_greatestOfBdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Int_greatestOfBdd___redArg(x_2, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_greatestOfBdd___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Int_greatestOfBdd(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_greatestOfBdd___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Int_greatestOfBdd___redArg(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_OrderIso(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Find(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_LeastGreatest(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_OrderIso(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
