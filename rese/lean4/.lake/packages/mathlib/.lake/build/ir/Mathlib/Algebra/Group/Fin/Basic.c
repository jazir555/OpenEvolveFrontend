// Lean compiler output
// Module: Mathlib.Algebra.Group.Fin.Basic
// Imports: public import Init public import Mathlib.Algebra.Group.Basic public import Mathlib.Algebra.NeZero public import Mathlib.Data.Nat.Cast.Defs public import Mathlib.Data.Fin.Rev
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
lean_object* l_Fin_neg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddRightCancelSemigroup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddLeftCancelSemigroup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instInvolutiveNeg(lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne(lean_object*, lean_object*);
lean_object* l_Fin_sub___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommGroup(lean_object*, lean_object*);
lean_object* l_Fin_add___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommSemigroup(lean_object*);
lean_object* lp_mathlib_zsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommSemigroup(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_1);
x_2 = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_nat_mod(x_3, x_1);
lean_dec(x_1);
lean_inc_ref(x_2);
lean_inc(x_4);
x_5 = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, x_4);
lean_closure_set(x_5, 2, x_2);
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_2);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_addCommMonoid___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_nat_mod(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc(x_1);
x_3 = lp_mathlib_Fin_addCommMonoid___redArg(x_1);
x_4 = lean_unsigned_to_nat(1u);
x_5 = lean_nat_mod(x_4, x_1);
lean_dec(x_1);
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_2);
lean_ctor_set(x_6, 1, x_3);
lean_ctor_set(x_6, 2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_instAddMonoidWithOne___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc(x_1);
x_2 = lp_mathlib_Fin_addCommMonoid___redArg(x_1);
x_3 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
lean_inc(x_1);
x_5 = lean_alloc_closure((void*)(l_Fin_neg___lam__0___boxed), 2, 1);
lean_closure_set(x_5, 0, x_1);
lean_inc(x_1);
x_6 = lean_alloc_closure((void*)(l_Fin_sub___boxed), 3, 1);
lean_closure_set(x_6, 0, x_1);
x_7 = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(x_7, 0, x_1);
lean_inc_ref(x_7);
lean_inc(x_4);
x_8 = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_4);
lean_closure_set(x_8, 2, x_7);
lean_inc_ref(x_5);
x_9 = lean_alloc_closure((void*)(lp_mathlib_zsmulRec___boxed), 7, 5);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_4);
lean_closure_set(x_9, 2, x_7);
lean_closure_set(x_9, 3, x_5);
lean_closure_set(x_9, 4, x_8);
x_10 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_10, 0, x_2);
lean_ctor_set(x_10, 1, x_5);
lean_ctor_set(x_10, 2, x_6);
lean_ctor_set(x_10, 3, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_addCommGroup___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instInvolutiveNeg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(l_Fin_neg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddLeftCancelSemigroup(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddRightCancelSemigroup(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_NeZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_Rev(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Rev(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
