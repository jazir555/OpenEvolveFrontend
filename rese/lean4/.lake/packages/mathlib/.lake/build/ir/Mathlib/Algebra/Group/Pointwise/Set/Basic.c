// Lean compiler output
// Module: Mathlib.Algebra.Group.Pointwise.Set.Basic
// Imports: public import Init public import Mathlib.Algebra.Group.Equiv.Basic public import Mathlib.Algebra.Group.Prod public import Mathlib.Algebra.Order.Monoid.Unbundled.Pow public import Mathlib.Data.Set.NAry
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
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_zero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_NPow___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_ZPow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_mulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_one(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addZeroClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sub___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonOneHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMonoidHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_monoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_ZPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_add___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_div___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonOneHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___redArg(lean_object*);
static lean_object* lp_mathlib_Set_monoid___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Set_semigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMulHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonZeroHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_semigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_zero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_inv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_pointwise_x20nat_x20action;
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_ZSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Set_addMonoid___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Set_NPow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_inv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_neg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_monoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_mulOneClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddMonoidHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_one___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_neg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_NSMul___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveInv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_NSMul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMulHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveNeg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonZeroHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_ZSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_mul___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addMonoid___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_LibraryNote_pointwise_x20nat_x20action() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_one(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_one___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_one(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_zero(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_zero___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_zero(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonOneHom(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonOneHom___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_singletonOneHom(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonZeroHom(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonZeroHom___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_singletonZeroHom(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_inv(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_inv___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_inv(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_neg(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_neg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_neg(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveInv(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveInv___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_involutiveInv(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveNeg(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveNeg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_involutiveNeg(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_mul(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_mul___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_mul(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_add(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_add___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_add(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMulHom(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMulHom___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_singletonMulHom(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddHom(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddHom___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_singletonAddHom(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_div(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_div___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_div(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sub(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sub___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_sub(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_NSMul(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_NSMul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Set_NSMul(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_NPow(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_NPow___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Set_NPow(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_ZSMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_ZSMul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Set_ZSMul(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_ZPow(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_ZPow___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Set_ZPow(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_semigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_semigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_semigroup(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_addSemigroup(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_commSemigroup(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_addCommSemigroup(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_mulOneClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, lean_box(0));
lean_ctor_set(x_3, 1, lean_box(0));
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_mulOneClass___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_mulOneClass(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addZeroClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, lean_box(0));
lean_ctor_set(x_3, 1, lean_box(0));
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addZeroClass___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_addZeroClass(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMonoidHom(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMonoidHom___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_singletonMonoidHom(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddMonoidHom(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddMonoidHom___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_singletonAddMonoidHom(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Set_monoid___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_1, 0, lean_box(0));
lean_ctor_set(x_1, 1, lean_box(0));
lean_ctor_set(x_1, 2, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_monoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_monoid___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_monoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_monoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Set_addMonoid___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_1, 0, lean_box(0));
lean_ctor_set(x_1, 1, lean_box(0));
lean_ctor_set(x_1, 2, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_addMonoid___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_addMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_monoid(lean_box(0), x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_monoid(lean_box(0), x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_commMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_commMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_addMonoid(lean_box(0), x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_addMonoid(lean_box(0), x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_addCommMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_addCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionMonoid___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 3);
lean_dec(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_dec(x_5);
x_6 = lean_ctor_get(x_1, 1);
lean_dec(x_6);
x_7 = lp_mathlib_Set_monoid(lean_box(0), x_3);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 3, lean_box(0));
lean_ctor_set(x_1, 2, lean_box(0));
lean_ctor_set(x_1, 1, lean_box(0));
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_1, 0);
lean_inc(x_8);
lean_dec(x_1);
x_9 = lp_mathlib_Set_monoid(lean_box(0), x_8);
lean_dec_ref(x_8);
x_10 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, lean_box(0));
lean_ctor_set(x_10, 2, lean_box(0));
lean_ctor_set(x_10, 3, lean_box(0));
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_divisionMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionMonoid___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 3);
lean_dec(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_dec(x_5);
x_6 = lean_ctor_get(x_1, 1);
lean_dec(x_6);
x_7 = lp_mathlib_Set_addMonoid(lean_box(0), x_3);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 3, lean_box(0));
lean_ctor_set(x_1, 2, lean_box(0));
lean_ctor_set(x_1, 1, lean_box(0));
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_1, 0);
lean_inc(x_8);
lean_dec(x_1);
x_9 = lp_mathlib_Set_addMonoid(lean_box(0), x_8);
lean_dec_ref(x_8);
x_10 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, lean_box(0));
lean_ctor_set(x_10, 2, lean_box(0));
lean_ctor_set(x_10, 3, lean_box(0));
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_subtractionMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_divisionMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_divisionMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_subtractionMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_subtractionMonoid___redArg(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_NAry(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_pointwise_x20nat_x20action = _init_lp_mathlib_LibraryNote_pointwise_x20nat_x20action();
lean_mark_persistent(lp_mathlib_LibraryNote_pointwise_x20nat_x20action);
lp_mathlib_Set_monoid___closed__0 = _init_lp_mathlib_Set_monoid___closed__0();
lean_mark_persistent(lp_mathlib_Set_monoid___closed__0);
lp_mathlib_Set_addMonoid___closed__0 = _init_lp_mathlib_Set_addMonoid___closed__0();
lean_mark_persistent(lp_mathlib_Set_addMonoid___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
