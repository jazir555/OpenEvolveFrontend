// Lean compiler output
// Module: Mathlib.Algebra.Group.ULift
// Imports: public import Init public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Group.InjSurj public import Mathlib.Logic.Nontrivial.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_semigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_subNegAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_ulift(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_group___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addLeftCancelMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_one(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoid___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_group(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoid(lean_object*, lean_object*);
static lean_object* lp_mathlib_MulEquiv_ulift___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ULift_subNegAddMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoid(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_addCancelCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_addRightCancelMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_semigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_div___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_neg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_neg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_vadd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_sub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_vadd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_one(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_one(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ULift_one___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_zero(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ULift_zero___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_mul___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_add___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_add(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_add___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_div___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_div(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_div___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_sub___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_sub(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_sub___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_inv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_neg___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_neg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_neg___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_ULift_smul___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_vadd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_vadd(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_ULift_vadd___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_pow___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_ULift_pow___redArg(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_ulift___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_ulift(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulEquiv_ulift___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulEquiv_ulift(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulEquiv_ulift___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddEquiv_ulift(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_semigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_semigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulOneClass___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_4, 0, x_3);
lean_ctor_set(x_1, 1, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_6);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulOneClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_mulOneClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addZeroClass___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_4, 0, x_3);
lean_ctor_set(x_1, 1, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_6);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addZeroClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_addZeroClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
lean_ctor_set(x_7, 2, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_monoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
x_8 = lp_mathlib_Function_Injective_addMonoid___redArg(x_6, x_3, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_addMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
lean_ctor_set(x_7, 2, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_commMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
x_8 = lp_mathlib_Function_Injective_addMonoid___redArg(x_6, x_3, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_addCommMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
x_6 = lean_ctor_get(x_1, 3);
x_7 = lp_mathlib_Monoid_toMulOneClass___redArg(x_3);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 1);
lean_inc(x_9);
lean_dec_ref(x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_6);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(x_11, 0, x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_12, 0, x_9);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_13, 0, x_4);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_5);
x_15 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_15, 0, x_12);
lean_ctor_set(x_15, 1, x_8);
lean_ctor_set(x_15, 2, x_11);
lean_ctor_set(x_1, 3, x_10);
lean_ctor_set(x_1, 2, x_14);
lean_ctor_set(x_1, 1, x_13);
lean_ctor_set(x_1, 0, x_15);
return x_1;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_16 = lean_ctor_get(x_1, 0);
x_17 = lean_ctor_get(x_1, 1);
x_18 = lean_ctor_get(x_1, 2);
x_19 = lean_ctor_get(x_1, 3);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_1);
x_20 = lp_mathlib_Monoid_toMulOneClass___redArg(x_16);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
x_22 = lean_ctor_get(x_20, 1);
lean_inc(x_22);
lean_dec_ref(x_20);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(x_23, 0, x_19);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(x_24, 0, x_16);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_25, 0, x_22);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_26, 0, x_17);
x_27 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_27, 0, x_18);
x_28 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_28, 0, x_25);
lean_ctor_set(x_28, 1, x_21);
lean_ctor_set(x_28, 2, x_24);
x_29 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_26);
lean_ctor_set(x_29, 2, x_27);
lean_ctor_set(x_29, 3, x_23);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_divInvMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_subNegAddMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_2);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
lean_dec_ref(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_11, 0, x_9);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_12, 0, x_3);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_13, 0, x_4);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_5);
x_15 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_10, x_7, x_11, x_12, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_subNegAddMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_subNegAddMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_group___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
x_6 = lean_ctor_get(x_1, 3);
x_7 = lp_mathlib_Monoid_toMulOneClass___redArg(x_3);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 1);
lean_inc(x_9);
lean_dec_ref(x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_6);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(x_11, 0, x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_12, 0, x_9);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_13, 0, x_4);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_5);
x_15 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_15, 0, x_12);
lean_ctor_set(x_15, 1, x_8);
lean_ctor_set(x_15, 2, x_11);
lean_ctor_set(x_1, 3, x_10);
lean_ctor_set(x_1, 2, x_14);
lean_ctor_set(x_1, 1, x_13);
lean_ctor_set(x_1, 0, x_15);
return x_1;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_16 = lean_ctor_get(x_1, 0);
x_17 = lean_ctor_get(x_1, 1);
x_18 = lean_ctor_get(x_1, 2);
x_19 = lean_ctor_get(x_1, 3);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_1);
x_20 = lp_mathlib_Monoid_toMulOneClass___redArg(x_16);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
x_22 = lean_ctor_get(x_20, 1);
lean_inc(x_22);
lean_dec_ref(x_20);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(x_23, 0, x_19);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(x_24, 0, x_16);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_25, 0, x_22);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_26, 0, x_17);
x_27 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_27, 0, x_18);
x_28 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_28, 0, x_25);
lean_ctor_set(x_28, 1, x_21);
lean_ctor_set(x_28, 2, x_24);
x_29 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_26);
lean_ctor_set(x_29, 2, x_27);
lean_ctor_set(x_29, 3, x_23);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_group(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_group___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_2);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
lean_dec_ref(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_11, 0, x_9);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_12, 0, x_3);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_13, 0, x_4);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_5);
x_15 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_10, x_7, x_11, x_12, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_addGroup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroup___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
x_6 = lean_ctor_get(x_1, 3);
x_7 = lp_mathlib_Monoid_toMulOneClass___redArg(x_3);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 1);
lean_inc(x_9);
lean_dec_ref(x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_6);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(x_11, 0, x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_12, 0, x_9);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_13, 0, x_4);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_5);
x_15 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_15, 0, x_12);
lean_ctor_set(x_15, 1, x_8);
lean_ctor_set(x_15, 2, x_11);
lean_ctor_set(x_1, 3, x_10);
lean_ctor_set(x_1, 2, x_14);
lean_ctor_set(x_1, 1, x_13);
lean_ctor_set(x_1, 0, x_15);
return x_1;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_16 = lean_ctor_get(x_1, 0);
x_17 = lean_ctor_get(x_1, 1);
x_18 = lean_ctor_get(x_1, 2);
x_19 = lean_ctor_get(x_1, 3);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_1);
x_20 = lp_mathlib_Monoid_toMulOneClass___redArg(x_16);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
x_22 = lean_ctor_get(x_20, 1);
lean_inc(x_22);
lean_dec_ref(x_20);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(x_23, 0, x_19);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(x_24, 0, x_16);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_25, 0, x_22);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_26, 0, x_17);
x_27 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_27, 0, x_18);
x_28 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_28, 0, x_25);
lean_ctor_set(x_28, 1, x_21);
lean_ctor_set(x_28, 2, x_24);
x_29 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_26);
lean_ctor_set(x_29, 2, x_27);
lean_ctor_set(x_29, 3, x_23);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_commGroup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_2);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
lean_dec_ref(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_11, 0, x_9);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_12, 0, x_3);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_13, 0, x_4);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_5);
x_15 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_10, x_7, x_11, x_12, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_addCommGroup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
lean_ctor_set(x_7, 2, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_leftCancelMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
x_8 = lp_mathlib_Function_Injective_addLeftCancelMonoid___redArg(x_6, x_3, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_addLeftCancelMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
lean_ctor_set(x_7, 2, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_rightCancelMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
x_8 = lp_mathlib_Function_Injective_addRightCancelMonoid___redArg(x_6, x_3, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_addRightCancelMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
lean_ctor_set(x_7, 2, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_cancelMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
x_8 = lp_mathlib_Function_Injective_addLeftCancelMonoid___redArg(x_6, x_3, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_addCancelMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
lean_ctor_set(x_7, 2, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_cancelCommMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
x_8 = lp_mathlib_Function_Injective_addCancelCommMonoid___redArg(x_6, x_3, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_addCancelCommMonoid___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Nontrivial_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_ULift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Nontrivial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_MulEquiv_ulift___closed__0 = _init_lp_mathlib_MulEquiv_ulift___closed__0();
lean_mark_persistent(lp_mathlib_MulEquiv_ulift___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
