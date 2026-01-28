// Lean compiler output
// Module: Mathlib.Algebra.Group.TypeTags.Basic
// Imports: public import Init public import Mathlib.Algebra.Group.Torsion public import Mathlib.Algebra.Notation.Pi.Basic public import Mathlib.Data.FunLike.Basic public import Mathlib.Logic.Function.Iterate public import Mathlib.Logic.Equiv.Defs public import Mathlib.Tactic.Set public import Mathlib.Util.AssertExists public import Mathlib.Logic.Nontrivial.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_group(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_involutiveNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqAdditive(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_group___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_monoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun___redArg(lean_object*);
static lean_object* lp_mathlib_Multiplicative_toAdd___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelSemigroup(lean_object*, lean_object*);
static lean_object* lp_mathlib_Additive_toMul___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Additive_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subNegMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqMultiplicative___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_add___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqAdditive___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionMonoid(lean_object*, lean_object*);
static lean_object* lp_mathlib_instUniqueAdditive___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelSemigroup(lean_object*, lean_object*);
static lean_object* lp_mathlib_Multiplicative_ofAdd___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_semigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqAdditive___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqMultiplicative___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_involutiveInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_toMul(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addSemigroup(lean_object*, lean_object*);
static lean_object* lp_mathlib_Multiplicative_ofAdd___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commMonoid(lean_object*, lean_object*);
static lean_object* lp_mathlib_Additive_rec___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueMultiplicative(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_semigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_instCancelCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqMultiplicative(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Multiplicative_toAdd___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelMonoid___redArg(lean_object*);
static lean_object* lp_mathlib_Multiplicative_rec___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Additive_instAddCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAdditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAdditive___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_instAddCancelCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqMultiplicative___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divInvMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMultiplicative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAdditive(lean_object*, lean_object*);
static lean_object* lp_mathlib_Additive_toMul___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Additive_sub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMultiplicativeOfZero___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_involutiveInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAdditive(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rec(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subNegMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMultiplicative(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelMonoid(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_add___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqAdditive___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_involutiveNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_div___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_toAdd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAdditiveOfOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_rec(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAdditiveOfOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueMultiplicative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_monoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMultiplicativeOfZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_instCancelCommMonoid___redArg(lean_object*);
static lean_object* lp_mathlib_instUniqueMultiplicative___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Additive_ofMul___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Additive_ofMul___lam__0___boxed), 1, 0);
lean_inc_ref(x_2);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Additive_toMul___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Additive_ofMul(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Additive_toMul___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Additive_toMul___closed__0;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_toMul(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Additive_toMul___closed__1;
return x_2;
}
}
static lean_object* _init_lp_mathlib_Additive_rec___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Additive_toMul(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_rec___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_Additive_rec___redArg___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lean_apply_1(x_1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_rec(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Additive_rec___redArg(x_3, x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Multiplicative_ofAdd___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Additive_ofMul___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Multiplicative_ofAdd___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Multiplicative_ofAdd___closed__0;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiplicative_ofAdd___closed__1;
return x_2;
}
}
static lean_object* _init_lp_mathlib_Multiplicative_toAdd___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Multiplicative_toAdd___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Multiplicative_toAdd___closed__0;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_toAdd(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiplicative_toAdd___closed__1;
return x_2;
}
}
static lean_object* _init_lp_mathlib_Multiplicative_rec___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Multiplicative_toAdd(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rec___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_Multiplicative_rec___redArg___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lean_apply_1(x_1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rec(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Multiplicative_rec___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAdditive___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Additive_toMul___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAdditive(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instInhabitedAdditive___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMultiplicative___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Multiplicative_toAdd___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMultiplicative(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instInhabitedMultiplicative___redArg(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instUniqueAdditive___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Additive_rec___redArg___closed__0;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAdditive___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_instUniqueAdditive___redArg___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAdditive(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instUniqueAdditive___redArg(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instUniqueMultiplicative___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Multiplicative_rec___redArg___closed__0;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueMultiplicative___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_instUniqueMultiplicative___redArg___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueMultiplicative(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instUniqueMultiplicative___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqMultiplicative(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_apply_2(x_2, x_3, x_4);
x_6 = lean_unbox(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqMultiplicative___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqMultiplicative___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_instDecidableEqMultiplicative(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqMultiplicative___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_instDecidableEqMultiplicative___redArg(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqAdditive(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_apply_2(x_2, x_3, x_4);
x_6 = lean_unbox(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqAdditive___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqAdditive___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_instDecidableEqAdditive(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqAdditive___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_instDecidableEqAdditive___redArg(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_add___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_4 = lp_mathlib_Additive_rec___redArg___closed__0;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lp_mathlib_Additive_toMul___closed__0;
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_inc(x_5);
x_8 = lean_apply_1(x_5, x_2);
x_9 = lean_apply_1(x_5, x_3);
x_10 = lean_apply_2(x_1, x_8, x_9);
x_11 = lean_apply_1(x_7, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_add___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_add(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_add___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_4 = lp_mathlib_Multiplicative_rec___redArg___closed__0;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lp_mathlib_Multiplicative_toAdd___closed__0;
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_inc(x_5);
x_8 = lean_apply_1(x_5, x_2);
x_9 = lean_apply_1(x_5, x_3);
x_10 = lean_apply_2(x_1, x_8, x_9);
x_11 = lean_apply_1(x_7, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_mul___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_semigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_semigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAdditiveOfOne___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Additive_toMul___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAdditiveOfOne(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instZeroAdditiveOfOne___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMultiplicativeOfZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Multiplicative_toAdd___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMultiplicativeOfZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instOneMultiplicativeOfZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addZeroClass___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lp_mathlib_instZeroAdditiveOfOne___redArg(x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
lean_ctor_set(x_1, 1, x_6);
lean_ctor_set(x_1, 0, x_5);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_1);
x_9 = lp_mathlib_instZeroAdditiveOfOne___redArg(x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_8);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addZeroClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_addZeroClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulOneClass___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lp_mathlib_instOneMultiplicativeOfZero___redArg(x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
lean_ctor_set(x_1, 1, x_6);
lean_ctor_set(x_1, 0, x_5);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_1);
x_9 = lp_mathlib_instOneMultiplicativeOfZero___redArg(x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_8);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulOneClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_mulOneClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_3 = lp_mathlib_Additive_addZeroClass___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = !lean_is_exclusive(x_1);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_1, 1);
lean_dec(x_7);
x_8 = lean_ctor_get(x_1, 0);
lean_dec(x_8);
lean_ctor_set(x_1, 1, x_4);
lean_ctor_set(x_1, 0, x_5);
return x_1;
}
else
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
lean_dec(x_1);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_4);
lean_ctor_set(x_10, 2, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_addMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_monoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lp_mathlib_Multiplicative_mulOneClass___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = !lean_is_exclusive(x_1);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_1, 1);
lean_dec(x_7);
x_8 = lean_ctor_get(x_1, 0);
lean_dec(x_8);
lean_ctor_set(x_1, 1, x_4);
lean_ctor_set(x_1, 0, x_5);
return x_1;
}
else
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
lean_dec(x_1);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_4);
lean_ctor_set(x_10, 2, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_monoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_monoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_addMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Additive_addMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_monoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiplicative_monoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_addMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Additive_addMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_monoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiplicative_monoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_addMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Additive_addMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_monoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiplicative_monoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_instAddCancelCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_addMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_instAddCancelCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Additive_addMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_instCancelCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_monoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_instCancelCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiplicative_monoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lp_mathlib_Additive_rec___redArg___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lp_mathlib_Multiplicative_toAdd___closed__0;
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_apply_1(x_4, x_2);
x_8 = lean_apply_1(x_1, x_7);
x_9 = lean_apply_1(x_6, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Additive_neg___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_neg___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lp_mathlib_Multiplicative_rec___redArg___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lp_mathlib_Additive_toMul___closed__0;
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_apply_1(x_4, x_2);
x_8 = lean_apply_1(x_1, x_7);
x_9 = lean_apply_1(x_6, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_inv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_sub___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_sub(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_sub___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_div___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_div(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_div___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_involutiveNeg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Additive_neg___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_involutiveNeg___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Additive_neg___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_involutiveInv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_involutiveInv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subNegMonoid___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Additive_neg___redArg___lam__0), 2, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
x_8 = lp_mathlib_Additive_addMonoid___redArg(x_3);
lean_ctor_set(x_1, 2, x_7);
lean_ctor_set(x_1, 1, x_6);
lean_ctor_set(x_1, 0, x_8);
return x_1;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
x_11 = lean_ctor_get(x_1, 2);
x_12 = lean_ctor_get(x_1, 3);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Additive_neg___redArg___lam__0), 2, 1);
lean_closure_set(x_13, 0, x_10);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_11);
x_15 = lp_mathlib_Additive_addMonoid___redArg(x_9);
x_16 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_13);
lean_ctor_set(x_16, 2, x_14);
lean_ctor_set(x_16, 3, x_12);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subNegMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_subNegMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
x_8 = lp_mathlib_Multiplicative_monoid___redArg(x_3);
lean_ctor_set(x_1, 2, x_7);
lean_ctor_set(x_1, 1, x_6);
lean_ctor_set(x_1, 0, x_8);
return x_1;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
x_11 = lean_ctor_get(x_1, 2);
x_12 = lean_ctor_get(x_1, 3);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_13, 0, x_10);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_11);
x_15 = lp_mathlib_Multiplicative_monoid___redArg(x_9);
x_16 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_13);
lean_ctor_set(x_16, 2, x_14);
lean_ctor_set(x_16, 3, x_12);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divInvMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_divInvMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_subNegMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Additive_subNegMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_divInvMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiplicative_divInvMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_subNegMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Additive_subNegMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_divInvMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiplicative_divInvMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_subNegMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Additive_subNegMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_group(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_divInvMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_group___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiplicative_divInvMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Additive_subNegMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Additive_subNegMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiplicative_divInvMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Multiplicative_divInvMonoid___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_Additive_rec___redArg___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lean_apply_1(x_1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Additive_coeToFun___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Additive_coeToFun___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_Multiplicative_rec___redArg___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lean_apply_1(x_1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_coeToFun___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Multiplicative_coeToFun___redArg(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_FunLike_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AssertExists(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Nontrivial_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_FunLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AssertExists(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Nontrivial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Additive_toMul___closed__0 = _init_lp_mathlib_Additive_toMul___closed__0();
lean_mark_persistent(lp_mathlib_Additive_toMul___closed__0);
lp_mathlib_Additive_toMul___closed__1 = _init_lp_mathlib_Additive_toMul___closed__1();
lean_mark_persistent(lp_mathlib_Additive_toMul___closed__1);
lp_mathlib_Additive_rec___redArg___closed__0 = _init_lp_mathlib_Additive_rec___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Additive_rec___redArg___closed__0);
lp_mathlib_Multiplicative_ofAdd___closed__0 = _init_lp_mathlib_Multiplicative_ofAdd___closed__0();
lean_mark_persistent(lp_mathlib_Multiplicative_ofAdd___closed__0);
lp_mathlib_Multiplicative_ofAdd___closed__1 = _init_lp_mathlib_Multiplicative_ofAdd___closed__1();
lean_mark_persistent(lp_mathlib_Multiplicative_ofAdd___closed__1);
lp_mathlib_Multiplicative_toAdd___closed__0 = _init_lp_mathlib_Multiplicative_toAdd___closed__0();
lean_mark_persistent(lp_mathlib_Multiplicative_toAdd___closed__0);
lp_mathlib_Multiplicative_toAdd___closed__1 = _init_lp_mathlib_Multiplicative_toAdd___closed__1();
lean_mark_persistent(lp_mathlib_Multiplicative_toAdd___closed__1);
lp_mathlib_Multiplicative_rec___redArg___closed__0 = _init_lp_mathlib_Multiplicative_rec___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Multiplicative_rec___redArg___closed__0);
lp_mathlib_instUniqueAdditive___redArg___closed__0 = _init_lp_mathlib_instUniqueAdditive___redArg___closed__0();
lean_mark_persistent(lp_mathlib_instUniqueAdditive___redArg___closed__0);
lp_mathlib_instUniqueMultiplicative___redArg___closed__0 = _init_lp_mathlib_instUniqueMultiplicative___redArg___closed__0();
lean_mark_persistent(lp_mathlib_instUniqueMultiplicative___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
