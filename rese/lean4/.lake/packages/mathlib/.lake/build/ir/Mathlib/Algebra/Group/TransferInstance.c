// Lean compiler output
// Module: Mathlib.Algebra.Group.TransferInstance
// Imports: public import Init public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Group.InjSurj public import Mathlib.Data.Fintype.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_div___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vadd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_one___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_add___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Neg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addZeroClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___redArg___boxed(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vadd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_zero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Neg___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_add(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_div(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_zero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_one(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sub___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sub(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_one(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lp_mathlib_Equiv_symm___redArg(x_3);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_1(x_6, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_one___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Equiv_symm___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lean_apply_1(x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_zero___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Equiv_symm___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lean_apply_1(x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_zero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_zero___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
lean_inc(x_5);
x_8 = lean_apply_1(x_5, x_3);
x_9 = lean_apply_1(x_5, x_4);
x_10 = lean_apply_2(x_2, x_8, x_9);
x_11 = lean_apply_1(x_7, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_add___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_add(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_add___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_div(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_div___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sub___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sub(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_sub___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
x_5 = lp_mathlib_Equiv_symm___redArg(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_1(x_4, x_3);
x_8 = lean_apply_1(x_2, x_7);
x_9 = lean_apply_1(x_6, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Inv___redArg___lam__0), 3, 2);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Inv___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Neg___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Inv___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Neg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_Neg___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_5, x_4);
x_9 = lean_apply_2(x_2, x_3, x_8);
x_10 = lean_apply_1(x_7, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vadd___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vadd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_vadd___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_5, x_3);
x_9 = lean_apply_2(x_2, x_8, x_4);
x_10 = lean_apply_1(x_7, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_Equiv_pow___redArg___lam__0), 4, 2);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_pow___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_mulEquiv(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_mulEquiv___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_addEquiv(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_addEquiv___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addSemigroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addSemigroup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commSemigroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commSemigroup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommSemigroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommSemigroup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
lean_inc(x_6);
x_7 = lean_apply_1(x_6, x_4);
x_8 = lean_apply_1(x_6, x_5);
x_9 = lean_apply_2(x_2, x_7, x_8);
x_10 = lean_apply_1(x_3, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_3);
x_8 = lp_mathlib_Equiv_symm___redArg(x_3);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
lean_inc(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_10, 0, x_3);
lean_closure_set(x_10, 1, x_7);
lean_closure_set(x_10, 2, x_9);
x_11 = lean_apply_1(x_9, x_6);
lean_ctor_set(x_4, 1, x_10);
lean_ctor_set(x_4, 0, x_11);
return x_4;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_12 = lean_ctor_get(x_4, 0);
x_13 = lean_ctor_get(x_4, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_4);
lean_inc_ref(x_3);
x_14 = lp_mathlib_Equiv_symm___redArg(x_3);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc(x_15);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_16, 0, x_3);
lean_closure_set(x_16, 1, x_13);
lean_closure_set(x_16, 2, x_15);
x_17 = lean_apply_1(x_15, x_12);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_16);
return x_18;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_1);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
lean_inc(x_7);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_5);
lean_closure_set(x_8, 2, x_7);
x_9 = lean_apply_1(x_7, x_4);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_9);
return x_2;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_2);
lean_inc_ref(x_1);
x_12 = lp_mathlib_Equiv_symm___redArg(x_1);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc(x_13);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_11);
lean_closure_set(x_14, 2, x_13);
x_15 = lean_apply_1(x_13, x_10);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_14);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addZeroClass___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_1);
x_6 = lp_mathlib_Equiv_zero___redArg(x_1, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_5);
lean_ctor_set(x_2, 1, x_7);
lean_ctor_set(x_2, 0, x_6);
return x_2;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_2, 0);
x_9 = lean_ctor_get(x_2, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_2);
lean_inc_ref(x_1);
x_10 = lp_mathlib_Equiv_zero___redArg(x_1, x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_9);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addZeroClass(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_addZeroClass___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 2);
lean_inc(x_6);
lean_dec_ref(x_2);
x_7 = lp_mathlib_Equiv_symm___redArg(x_1);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_5, x_4);
x_10 = lean_apply_2(x_6, x_3, x_9);
x_11 = lean_apply_1(x_8, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_5 = lp_mathlib_Monoid_toMulOneClass___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec_ref(x_5);
lean_inc_ref(x_3);
x_8 = lp_mathlib_Equiv_symm___redArg(x_3);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
lean_inc_ref(x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__0), 4, 2);
lean_closure_set(x_10, 0, x_3);
lean_closure_set(x_10, 1, x_4);
lean_inc(x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_11, 0, x_3);
lean_closure_set(x_11, 1, x_7);
lean_closure_set(x_11, 2, x_9);
x_12 = lean_apply_1(x_9, x_6);
x_13 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
lean_ctor_set(x_13, 2, x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_3 = lp_mathlib_Monoid_toMulOneClass___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
lean_inc_ref(x_1);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
lean_inc_ref(x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__0), 4, 2);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_2);
lean_inc(x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_5);
lean_closure_set(x_9, 2, x_7);
x_10 = lean_apply_1(x_7, x_4);
x_11 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
lean_ctor_set(x_11, 2, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addMonoid___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lp_mathlib_Equiv_symm___redArg(x_2);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_6, x_4);
x_10 = lean_apply_2(x_5, x_3, x_9);
x_11 = lean_apply_1(x_8, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addMonoid___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Equiv_addMonoid___redArg___lam__0), 4, 2);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_1);
lean_inc_ref(x_1);
x_7 = lp_mathlib_Equiv_zero___redArg(x_1, x_4);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_5);
x_9 = lp_mathlib_Function_Injective_addMonoid___redArg(x_8, x_7, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_addMonoid___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_5 = lp_mathlib_Monoid_toMulOneClass___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec_ref(x_5);
lean_inc_ref(x_3);
x_8 = lp_mathlib_Equiv_symm___redArg(x_3);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
lean_inc_ref(x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__0), 4, 2);
lean_closure_set(x_10, 0, x_3);
lean_closure_set(x_10, 1, x_4);
lean_inc(x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_11, 0, x_3);
lean_closure_set(x_11, 1, x_7);
lean_closure_set(x_11, 2, x_9);
x_12 = lean_apply_1(x_9, x_6);
x_13 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
lean_ctor_set(x_13, 2, x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoid___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_3 = lp_mathlib_Monoid_toMulOneClass___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
lean_inc_ref(x_1);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
lean_inc_ref(x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__0), 4, 2);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_2);
lean_inc(x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_5);
lean_closure_set(x_9, 2, x_7);
x_10 = lean_apply_1(x_7, x_4);
x_11 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
lean_ctor_set(x_11, 2, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommMonoid___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Equiv_addMonoid___redArg___lam__0), 4, 2);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_1);
lean_inc_ref(x_1);
x_7 = lp_mathlib_Equiv_zero___redArg(x_1, x_4);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_5);
x_9 = lp_mathlib_Function_Injective_addMonoid___redArg(x_8, x_7, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_addCommMonoid___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_5, x_4);
x_7 = lean_apply_1(x_2, x_6);
x_8 = lean_apply_1(x_3, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
lean_inc(x_6);
x_7 = lean_apply_1(x_6, x_4);
x_8 = lean_apply_1(x_6, x_5);
x_9 = lean_apply_2(x_2, x_7, x_8);
x_10 = lean_apply_1(x_3, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_5, x_4);
x_9 = lean_apply_2(x_2, x_3, x_8);
x_10 = lean_apply_1(x_7, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 2);
lean_inc(x_6);
lean_dec_ref(x_2);
x_7 = lp_mathlib_Equiv_symm___redArg(x_1);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_5, x_4);
x_10 = lean_apply_2(x_6, x_3, x_9);
x_11 = lean_apply_1(x_8, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
x_8 = lean_ctor_get(x_4, 2);
x_9 = lean_ctor_get(x_4, 3);
x_10 = lp_mathlib_Monoid_toMulOneClass___redArg(x_6);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_10, 1);
lean_inc(x_12);
lean_dec_ref(x_10);
lean_inc_ref(x_3);
x_13 = lp_mathlib_Equiv_symm___redArg(x_3);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
lean_inc_ref(x_3);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 4, 2);
lean_closure_set(x_15, 0, x_3);
lean_closure_set(x_15, 1, x_9);
lean_inc_ref(x_3);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 2);
lean_closure_set(x_16, 0, x_3);
lean_closure_set(x_16, 1, x_6);
lean_inc(x_14);
lean_inc_ref(x_3);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(x_17, 0, x_3);
lean_closure_set(x_17, 1, x_8);
lean_closure_set(x_17, 2, x_14);
lean_inc(x_14);
lean_inc_ref(x_3);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__3), 4, 3);
lean_closure_set(x_18, 0, x_3);
lean_closure_set(x_18, 1, x_7);
lean_closure_set(x_18, 2, x_14);
lean_inc(x_14);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_19, 0, x_3);
lean_closure_set(x_19, 1, x_12);
lean_closure_set(x_19, 2, x_14);
x_20 = lean_apply_1(x_14, x_11);
x_21 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_20);
lean_ctor_set(x_21, 2, x_16);
lean_ctor_set(x_4, 3, x_15);
lean_ctor_set(x_4, 2, x_17);
lean_ctor_set(x_4, 1, x_18);
lean_ctor_set(x_4, 0, x_21);
return x_4;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_22 = lean_ctor_get(x_4, 0);
x_23 = lean_ctor_get(x_4, 1);
x_24 = lean_ctor_get(x_4, 2);
x_25 = lean_ctor_get(x_4, 3);
lean_inc(x_25);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_22);
lean_dec(x_4);
x_26 = lp_mathlib_Monoid_toMulOneClass___redArg(x_22);
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
x_28 = lean_ctor_get(x_26, 1);
lean_inc(x_28);
lean_dec_ref(x_26);
lean_inc_ref(x_3);
x_29 = lp_mathlib_Equiv_symm___redArg(x_3);
x_30 = lean_ctor_get(x_29, 0);
lean_inc(x_30);
lean_dec_ref(x_29);
lean_inc_ref(x_3);
x_31 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 4, 2);
lean_closure_set(x_31, 0, x_3);
lean_closure_set(x_31, 1, x_25);
lean_inc_ref(x_3);
x_32 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 2);
lean_closure_set(x_32, 0, x_3);
lean_closure_set(x_32, 1, x_22);
lean_inc(x_30);
lean_inc_ref(x_3);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(x_33, 0, x_3);
lean_closure_set(x_33, 1, x_24);
lean_closure_set(x_33, 2, x_30);
lean_inc(x_30);
lean_inc_ref(x_3);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__3), 4, 3);
lean_closure_set(x_34, 0, x_3);
lean_closure_set(x_34, 1, x_23);
lean_closure_set(x_34, 2, x_30);
lean_inc(x_30);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_35, 0, x_3);
lean_closure_set(x_35, 1, x_28);
lean_closure_set(x_35, 2, x_30);
x_36 = lean_apply_1(x_30, x_27);
x_37 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_37, 0, x_35);
lean_ctor_set(x_37, 1, x_36);
lean_ctor_set(x_37, 2, x_32);
x_38 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_34);
lean_ctor_set(x_38, 2, x_33);
lean_ctor_set(x_38, 3, x_31);
return x_38;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_2, 2);
x_7 = lean_ctor_get(x_2, 3);
x_8 = lp_mathlib_Monoid_toMulOneClass___redArg(x_4);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_8, 1);
lean_inc(x_10);
lean_dec_ref(x_8);
lean_inc_ref(x_1);
x_11 = lp_mathlib_Equiv_symm___redArg(x_1);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_1);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 4, 2);
lean_closure_set(x_13, 0, x_1);
lean_closure_set(x_13, 1, x_7);
lean_inc_ref(x_1);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 2);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_4);
lean_inc(x_12);
lean_inc_ref(x_1);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(x_15, 0, x_1);
lean_closure_set(x_15, 1, x_6);
lean_closure_set(x_15, 2, x_12);
lean_inc(x_12);
lean_inc_ref(x_1);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__3), 4, 3);
lean_closure_set(x_16, 0, x_1);
lean_closure_set(x_16, 1, x_5);
lean_closure_set(x_16, 2, x_12);
lean_inc(x_12);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_17, 0, x_1);
lean_closure_set(x_17, 1, x_10);
lean_closure_set(x_17, 2, x_12);
x_18 = lean_apply_1(x_12, x_9);
x_19 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
lean_ctor_set(x_19, 2, x_14);
lean_ctor_set(x_2, 3, x_13);
lean_ctor_set(x_2, 2, x_15);
lean_ctor_set(x_2, 1, x_16);
lean_ctor_set(x_2, 0, x_19);
return x_2;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_20 = lean_ctor_get(x_2, 0);
x_21 = lean_ctor_get(x_2, 1);
x_22 = lean_ctor_get(x_2, 2);
x_23 = lean_ctor_get(x_2, 3);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_2);
x_24 = lp_mathlib_Monoid_toMulOneClass___redArg(x_20);
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
x_26 = lean_ctor_get(x_24, 1);
lean_inc(x_26);
lean_dec_ref(x_24);
lean_inc_ref(x_1);
x_27 = lp_mathlib_Equiv_symm___redArg(x_1);
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
lean_dec_ref(x_27);
lean_inc_ref(x_1);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 4, 2);
lean_closure_set(x_29, 0, x_1);
lean_closure_set(x_29, 1, x_23);
lean_inc_ref(x_1);
x_30 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 2);
lean_closure_set(x_30, 0, x_1);
lean_closure_set(x_30, 1, x_20);
lean_inc(x_28);
lean_inc_ref(x_1);
x_31 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(x_31, 0, x_1);
lean_closure_set(x_31, 1, x_22);
lean_closure_set(x_31, 2, x_28);
lean_inc(x_28);
lean_inc_ref(x_1);
x_32 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__3), 4, 3);
lean_closure_set(x_32, 0, x_1);
lean_closure_set(x_32, 1, x_21);
lean_closure_set(x_32, 2, x_28);
lean_inc(x_28);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_33, 0, x_1);
lean_closure_set(x_33, 1, x_26);
lean_closure_set(x_33, 2, x_28);
x_34 = lean_apply_1(x_28, x_25);
x_35 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_35, 0, x_33);
lean_ctor_set(x_35, 1, x_34);
lean_ctor_set(x_35, 2, x_30);
x_36 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_36, 0, x_35);
lean_ctor_set(x_36, 1, x_32);
lean_ctor_set(x_36, 2, x_31);
lean_ctor_set(x_36, 3, x_29);
return x_36;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_5, x_4);
x_9 = lean_apply_2(x_2, x_3, x_8);
x_10 = lean_apply_1(x_7, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lp_mathlib_Equiv_symm___redArg(x_2);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_6, x_4);
x_10 = lean_apply_2(x_5, x_3, x_9);
x_11 = lean_apply_1(x_8, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 2);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 3);
lean_inc(x_6);
lean_dec_ref(x_2);
x_7 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_3);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 1);
lean_inc(x_9);
lean_dec_ref(x_7);
lean_inc_ref(x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Equiv_addGroup___redArg___lam__0), 4, 2);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_6);
lean_inc_ref(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Equiv_addGroup___redArg___lam__1), 4, 2);
lean_closure_set(x_11, 0, x_3);
lean_closure_set(x_11, 1, x_1);
lean_inc_ref(x_1);
x_12 = lp_mathlib_Equiv_zero___redArg(x_1, x_8);
lean_inc_ref(x_1);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_13, 0, x_1);
lean_closure_set(x_13, 1, x_9);
lean_inc_ref(x_1);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Inv___redArg___lam__0), 3, 2);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_4);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_15, 0, x_1);
lean_closure_set(x_15, 1, x_5);
x_16 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_13, x_12, x_11, x_14, x_15, x_10);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_addGroup___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
x_8 = lean_ctor_get(x_4, 2);
x_9 = lean_ctor_get(x_4, 3);
x_10 = lp_mathlib_Monoid_toMulOneClass___redArg(x_6);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_10, 1);
lean_inc(x_12);
lean_dec_ref(x_10);
lean_inc_ref(x_3);
x_13 = lp_mathlib_Equiv_symm___redArg(x_3);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
lean_inc_ref(x_3);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 4, 2);
lean_closure_set(x_15, 0, x_3);
lean_closure_set(x_15, 1, x_9);
lean_inc_ref(x_3);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 2);
lean_closure_set(x_16, 0, x_3);
lean_closure_set(x_16, 1, x_6);
lean_inc(x_14);
lean_inc_ref(x_3);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(x_17, 0, x_3);
lean_closure_set(x_17, 1, x_8);
lean_closure_set(x_17, 2, x_14);
lean_inc(x_14);
lean_inc_ref(x_3);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__3), 4, 3);
lean_closure_set(x_18, 0, x_3);
lean_closure_set(x_18, 1, x_7);
lean_closure_set(x_18, 2, x_14);
lean_inc(x_14);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_19, 0, x_3);
lean_closure_set(x_19, 1, x_12);
lean_closure_set(x_19, 2, x_14);
x_20 = lean_apply_1(x_14, x_11);
x_21 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_20);
lean_ctor_set(x_21, 2, x_16);
lean_ctor_set(x_4, 3, x_15);
lean_ctor_set(x_4, 2, x_17);
lean_ctor_set(x_4, 1, x_18);
lean_ctor_set(x_4, 0, x_21);
return x_4;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_22 = lean_ctor_get(x_4, 0);
x_23 = lean_ctor_get(x_4, 1);
x_24 = lean_ctor_get(x_4, 2);
x_25 = lean_ctor_get(x_4, 3);
lean_inc(x_25);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_22);
lean_dec(x_4);
x_26 = lp_mathlib_Monoid_toMulOneClass___redArg(x_22);
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
x_28 = lean_ctor_get(x_26, 1);
lean_inc(x_28);
lean_dec_ref(x_26);
lean_inc_ref(x_3);
x_29 = lp_mathlib_Equiv_symm___redArg(x_3);
x_30 = lean_ctor_get(x_29, 0);
lean_inc(x_30);
lean_dec_ref(x_29);
lean_inc_ref(x_3);
x_31 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 4, 2);
lean_closure_set(x_31, 0, x_3);
lean_closure_set(x_31, 1, x_25);
lean_inc_ref(x_3);
x_32 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 2);
lean_closure_set(x_32, 0, x_3);
lean_closure_set(x_32, 1, x_22);
lean_inc(x_30);
lean_inc_ref(x_3);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(x_33, 0, x_3);
lean_closure_set(x_33, 1, x_24);
lean_closure_set(x_33, 2, x_30);
lean_inc(x_30);
lean_inc_ref(x_3);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__3), 4, 3);
lean_closure_set(x_34, 0, x_3);
lean_closure_set(x_34, 1, x_23);
lean_closure_set(x_34, 2, x_30);
lean_inc(x_30);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_35, 0, x_3);
lean_closure_set(x_35, 1, x_28);
lean_closure_set(x_35, 2, x_30);
x_36 = lean_apply_1(x_30, x_27);
x_37 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_37, 0, x_35);
lean_ctor_set(x_37, 1, x_36);
lean_ctor_set(x_37, 2, x_32);
x_38 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_34);
lean_ctor_set(x_38, 2, x_33);
lean_ctor_set(x_38, 3, x_31);
return x_38;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commGroup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_2, 2);
x_7 = lean_ctor_get(x_2, 3);
x_8 = lp_mathlib_Monoid_toMulOneClass___redArg(x_4);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_8, 1);
lean_inc(x_10);
lean_dec_ref(x_8);
lean_inc_ref(x_1);
x_11 = lp_mathlib_Equiv_symm___redArg(x_1);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_1);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 4, 2);
lean_closure_set(x_13, 0, x_1);
lean_closure_set(x_13, 1, x_7);
lean_inc_ref(x_1);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 2);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_4);
lean_inc(x_12);
lean_inc_ref(x_1);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(x_15, 0, x_1);
lean_closure_set(x_15, 1, x_6);
lean_closure_set(x_15, 2, x_12);
lean_inc(x_12);
lean_inc_ref(x_1);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__3), 4, 3);
lean_closure_set(x_16, 0, x_1);
lean_closure_set(x_16, 1, x_5);
lean_closure_set(x_16, 2, x_12);
lean_inc(x_12);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_17, 0, x_1);
lean_closure_set(x_17, 1, x_10);
lean_closure_set(x_17, 2, x_12);
x_18 = lean_apply_1(x_12, x_9);
x_19 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
lean_ctor_set(x_19, 2, x_14);
lean_ctor_set(x_2, 3, x_13);
lean_ctor_set(x_2, 2, x_15);
lean_ctor_set(x_2, 1, x_16);
lean_ctor_set(x_2, 0, x_19);
return x_2;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_20 = lean_ctor_get(x_2, 0);
x_21 = lean_ctor_get(x_2, 1);
x_22 = lean_ctor_get(x_2, 2);
x_23 = lean_ctor_get(x_2, 3);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_2);
x_24 = lp_mathlib_Monoid_toMulOneClass___redArg(x_20);
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
x_26 = lean_ctor_get(x_24, 1);
lean_inc(x_26);
lean_dec_ref(x_24);
lean_inc_ref(x_1);
x_27 = lp_mathlib_Equiv_symm___redArg(x_1);
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
lean_dec_ref(x_27);
lean_inc_ref(x_1);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 4, 2);
lean_closure_set(x_29, 0, x_1);
lean_closure_set(x_29, 1, x_23);
lean_inc_ref(x_1);
x_30 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 2);
lean_closure_set(x_30, 0, x_1);
lean_closure_set(x_30, 1, x_20);
lean_inc(x_28);
lean_inc_ref(x_1);
x_31 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(x_31, 0, x_1);
lean_closure_set(x_31, 1, x_22);
lean_closure_set(x_31, 2, x_28);
lean_inc(x_28);
lean_inc_ref(x_1);
x_32 = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__3), 4, 3);
lean_closure_set(x_32, 0, x_1);
lean_closure_set(x_32, 1, x_21);
lean_closure_set(x_32, 2, x_28);
lean_inc(x_28);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(x_33, 0, x_1);
lean_closure_set(x_33, 1, x_26);
lean_closure_set(x_33, 2, x_28);
x_34 = lean_apply_1(x_28, x_25);
x_35 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_35, 0, x_33);
lean_ctor_set(x_35, 1, x_34);
lean_ctor_set(x_35, 2, x_30);
x_36 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_36, 0, x_35);
lean_ctor_set(x_36, 1, x_32);
lean_ctor_set(x_36, 2, x_31);
lean_ctor_set(x_36, 3, x_29);
return x_36;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommGroup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 2);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 3);
lean_inc(x_6);
lean_dec_ref(x_2);
x_7 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_3);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 1);
lean_inc(x_9);
lean_dec_ref(x_7);
lean_inc_ref(x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Equiv_addGroup___redArg___lam__0), 4, 2);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_6);
lean_inc_ref(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Equiv_addGroup___redArg___lam__1), 4, 2);
lean_closure_set(x_11, 0, x_3);
lean_closure_set(x_11, 1, x_1);
lean_inc_ref(x_1);
x_12 = lp_mathlib_Equiv_zero___redArg(x_1, x_8);
lean_inc_ref(x_1);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_13, 0, x_1);
lean_closure_set(x_13, 1, x_9);
lean_inc_ref(x_1);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Inv___redArg___lam__0), 3, 2);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_4);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(x_15, 0, x_1);
lean_closure_set(x_15, 1, x_5);
x_16 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_13, x_12, x_11, x_14, x_15, x_10);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_addCommGroup___redArg(x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
