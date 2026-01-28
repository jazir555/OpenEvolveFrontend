// Lean compiler output
// Module: Mathlib.Algebra.Ring.Defs
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Defs public import Mathlib.Data.Int.Cast.Defs public import Mathlib.Tactic.Spread public import Mathlib.Tactic.StacksAttribute
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
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommRing_toNonUnitalNonAssocCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommRing_toNonUnitalNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSemiring_toSemigroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_negZeroClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonAssocCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRing_toNonUnitalSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_negZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSemiring_toSemigroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 1, x_5);
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_inc(x_6);
lean_dec(x_1);
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 1, x_5);
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_inc(x_6);
lean_dec(x_1);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSemiring_toSemigroupWithZero___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 1, x_5);
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_inc(x_6);
lean_dec(x_1);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSemiring_toSemigroupWithZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalSemiring_toSemigroupWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = !lean_is_exclusive(x_2);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_2, 0);
lean_ctor_set(x_2, 0, x_3);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_2);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_2, 0);
x_9 = lean_ctor_get(x_2, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_2);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_3);
lean_ctor_set(x_10, 1, x_9);
x_11 = lean_ctor_get(x_8, 1);
lean_inc(x_11);
lean_dec_ref(x_8);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
x_6 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 2, x_4);
lean_ctor_set(x_1, 1, x_6);
lean_ctor_set(x_1, 0, x_5);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 1);
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_1);
x_10 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_7);
x_11 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
lean_ctor_set(x_11, 2, x_8);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 0, x_4);
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
x_7 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 0, x_4);
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
x_7 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRing_toNonUnitalSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = !lean_is_exclusive(x_2);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_2, 0);
lean_dec(x_7);
x_8 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_3);
lean_ctor_set(x_2, 0, x_8);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_2);
lean_ctor_set(x_9, 1, x_4);
lean_ctor_set(x_9, 2, x_5);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_ctor_get(x_2, 1);
lean_inc(x_10);
lean_dec(x_2);
x_11 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_11);
lean_dec_ref(x_3);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_10);
x_13 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_4);
lean_ctor_set(x_13, 2, x_5);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 3);
x_6 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 3, x_4);
lean_ctor_set(x_1, 1, x_5);
lean_ctor_set(x_1, 0, x_6);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 1);
x_9 = lean_ctor_get(x_1, 2);
x_10 = lean_ctor_get(x_1, 3);
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_1);
x_11 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_11);
lean_dec_ref(x_7);
x_12 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_10);
lean_ctor_set(x_12, 2, x_9);
lean_ctor_set(x_12, 3, x_8);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Semiring_toNonAssocSemiring(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 3);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_3);
lean_ctor_set(x_8, 2, x_4);
x_9 = lean_ctor_get(x_6, 1);
lean_inc(x_9);
lean_dec_ref(x_6);
lean_ctor_set(x_2, 1, x_9);
lean_ctor_set(x_2, 0, x_8);
return x_2;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_2);
x_12 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_3);
lean_ctor_set(x_12, 2, x_4);
x_13 = lean_ctor_get(x_10, 1);
lean_inc(x_13);
lean_dec_ref(x_10);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 3);
lean_dec(x_5);
x_6 = lean_ctor_get(x_2, 2);
lean_dec(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_dec(x_7);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_1, 3);
lean_inc(x_10);
lean_dec_ref(x_1);
x_11 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_11);
lean_dec_ref(x_4);
lean_ctor_set(x_2, 3, x_10);
lean_ctor_set(x_2, 2, x_9);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_11);
return x_2;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lean_ctor_get(x_2, 0);
lean_inc(x_12);
lean_dec(x_2);
x_13 = lean_ctor_get(x_1, 1);
lean_inc(x_13);
x_14 = lean_ctor_get(x_1, 2);
lean_inc(x_14);
x_15 = lean_ctor_get(x_1, 3);
lean_inc(x_15);
lean_dec_ref(x_1);
x_16 = lean_ctor_get(x_12, 0);
lean_inc_ref(x_16);
lean_dec_ref(x_12);
x_17 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_13);
lean_ctor_set(x_17, 2, x_14);
lean_ctor_set(x_17, 3, x_15);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Ring_toAddCommGroup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_1, 2);
x_7 = lean_ctor_get(x_1, 3);
x_8 = lean_ctor_get(x_1, 4);
x_9 = lean_ctor_get(x_1, 0);
lean_dec(x_9);
x_10 = lean_ctor_get(x_2, 1);
lean_inc(x_10);
x_11 = lean_ctor_get(x_2, 2);
lean_inc(x_11);
lean_dec_ref(x_2);
x_12 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_3);
x_13 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
lean_ctor_set(x_13, 2, x_10);
lean_ctor_set(x_1, 4, x_7);
lean_ctor_set(x_1, 3, x_6);
lean_ctor_set(x_1, 2, x_5);
lean_ctor_set(x_1, 1, x_13);
lean_ctor_set(x_1, 0, x_8);
return x_1;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_14 = lean_ctor_get(x_1, 1);
x_15 = lean_ctor_get(x_1, 2);
x_16 = lean_ctor_get(x_1, 3);
x_17 = lean_ctor_get(x_1, 4);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_1);
x_18 = lean_ctor_get(x_2, 1);
lean_inc(x_18);
x_19 = lean_ctor_get(x_2, 2);
lean_inc(x_19);
lean_dec_ref(x_2);
x_20 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_20);
lean_dec_ref(x_3);
x_21 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_20);
lean_ctor_set(x_21, 2, x_18);
x_22 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_22, 0, x_17);
lean_ctor_set(x_22, 1, x_21);
lean_ctor_set(x_22, 2, x_14);
lean_ctor_set(x_22, 3, x_15);
lean_ctor_set(x_22, 4, x_16);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddGroupWithOne(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalCommSemiring_toCommSemigroup(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_1, 3);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_inc(x_3);
lean_inc(x_5);
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_3);
lean_ctor_set(x_6, 2, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommSemiring_toCommMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommSemiring_toNonAssocCommSemiring(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommSemiring_toNonAssocCommSemiring___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_1);
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_3, 1);
lean_dec(x_5);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
lean_dec_ref(x_2);
lean_ctor_set(x_3, 1, x_6);
return x_3;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_3, 0);
lean_inc(x_7);
lean_dec(x_3);
x_8 = lean_ctor_get(x_2, 0);
lean_inc(x_8);
lean_dec_ref(x_2);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommSemiring_toNonUnitalCommSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_1);
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_dec(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec_ref(x_5);
lean_ctor_set(x_3, 1, x_7);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
lean_dec(x_3);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_2);
lean_ctor_set(x_10, 1, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_negZeroClass___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 0);
lean_dec(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_dec(x_1);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_2);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_negZeroClass(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MulZeroClass_negZeroClass___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 3);
lean_dec(x_5);
x_6 = lean_ctor_get(x_2, 2);
lean_dec(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_dec(x_7);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_1, 3);
lean_inc(x_10);
lean_dec_ref(x_1);
x_11 = !lean_is_exclusive(x_4);
if (x_11 == 0)
{
lean_object* x_12; 
x_12 = lean_ctor_get(x_4, 0);
lean_ctor_set(x_2, 3, x_10);
lean_ctor_set(x_2, 2, x_9);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_12);
lean_ctor_set(x_4, 0, x_2);
return x_4;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_ctor_get(x_4, 0);
x_14 = lean_ctor_get(x_4, 1);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_4);
lean_ctor_set(x_2, 3, x_10);
lean_ctor_set(x_2, 2, x_9);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_13);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_2);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_16 = lean_ctor_get(x_2, 0);
lean_inc(x_16);
lean_dec(x_2);
x_17 = lean_ctor_get(x_1, 1);
lean_inc(x_17);
x_18 = lean_ctor_get(x_1, 2);
lean_inc(x_18);
x_19 = lean_ctor_get(x_1, 3);
lean_inc(x_19);
lean_dec_ref(x_1);
x_20 = lean_ctor_get(x_16, 0);
lean_inc_ref(x_20);
x_21 = lean_ctor_get(x_16, 1);
lean_inc(x_21);
if (lean_is_exclusive(x_16)) {
 lean_ctor_release(x_16, 0);
 lean_ctor_release(x_16, 1);
 x_22 = x_16;
} else {
 lean_dec_ref(x_16);
 x_22 = lean_box(0);
}
x_23 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_23, 0, x_20);
lean_ctor_set(x_23, 1, x_17);
lean_ctor_set(x_23, 2, x_18);
lean_ctor_set(x_23, 3, x_19);
if (lean_is_scalar(x_22)) {
 x_24 = lean_alloc_ctor(0, 2, 0);
} else {
 x_24 = x_22;
}
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_21);
return x_24;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Ring_toNonUnitalRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
x_6 = lean_ctor_get(x_1, 3);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 4);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = !lean_is_exclusive(x_2);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_9 = lean_ctor_get(x_2, 1);
x_10 = lean_ctor_get(x_2, 2);
x_11 = lean_ctor_get(x_2, 3);
lean_dec(x_11);
x_12 = lean_ctor_get(x_2, 0);
lean_dec(x_12);
x_13 = !lean_is_exclusive(x_3);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_3, 0);
lean_ctor_set(x_2, 3, x_6);
lean_ctor_set(x_2, 2, x_5);
lean_ctor_set(x_2, 1, x_4);
lean_ctor_set(x_2, 0, x_14);
lean_ctor_set(x_3, 0, x_2);
x_15 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_15, 0, x_3);
lean_ctor_set(x_15, 1, x_9);
lean_ctor_set(x_15, 2, x_10);
lean_ctor_set(x_15, 3, x_7);
return x_15;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_16 = lean_ctor_get(x_3, 0);
x_17 = lean_ctor_get(x_3, 1);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_3);
lean_ctor_set(x_2, 3, x_6);
lean_ctor_set(x_2, 2, x_5);
lean_ctor_set(x_2, 1, x_4);
lean_ctor_set(x_2, 0, x_16);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_2);
lean_ctor_set(x_18, 1, x_17);
x_19 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_9);
lean_ctor_set(x_19, 2, x_10);
lean_ctor_set(x_19, 3, x_7);
return x_19;
}
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_20 = lean_ctor_get(x_2, 1);
x_21 = lean_ctor_get(x_2, 2);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_2);
x_22 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_22);
x_23 = lean_ctor_get(x_3, 1);
lean_inc(x_23);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 x_24 = x_3;
} else {
 lean_dec_ref(x_3);
 x_24 = lean_box(0);
}
x_25 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_25, 0, x_22);
lean_ctor_set(x_25, 1, x_4);
lean_ctor_set(x_25, 2, x_5);
lean_ctor_set(x_25, 3, x_6);
if (lean_is_scalar(x_24)) {
 x_26 = lean_alloc_ctor(0, 2, 0);
} else {
 x_26 = x_24;
}
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_23);
x_27 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_20);
lean_ctor_set(x_27, 2, x_21);
lean_ctor_set(x_27, 3, x_7);
return x_27;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Ring_toNonAssocRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommRing_toNonUnitalNonAssocCommSemiring___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 0, x_4);
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
x_7 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommRing_toNonUnitalNonAssocCommSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalNonAssocCommRing_toNonUnitalNonAssocCommSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonAssocCommSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = !lean_is_exclusive(x_2);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_2, 0);
lean_dec(x_7);
x_8 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_3);
lean_ctor_set(x_2, 0, x_8);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_2);
lean_ctor_set(x_9, 1, x_4);
lean_ctor_set(x_9, 2, x_5);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_ctor_get(x_2, 1);
lean_inc(x_10);
lean_dec(x_2);
x_11 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_11);
lean_dec_ref(x_3);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_10);
x_13 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_4);
lean_ctor_set(x_13, 2, x_5);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonAssocCommSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonAssocCommRing_toNonAssocCommSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 0, x_4);
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
x_7 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = lean_ctor_get(x_2, 3);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_6);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
lean_ctor_set(x_7, 2, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommRing_toCommMonoid___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommRing_toCommMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommRing_toCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Ring_toNonAssocRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Ring_toNonAssocRing___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommRing_toCommSemiring(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommRing_toCommSemiring___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 3);
lean_dec(x_5);
x_6 = lean_ctor_get(x_2, 2);
lean_dec(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_dec(x_7);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_1, 3);
lean_inc(x_10);
lean_dec_ref(x_1);
x_11 = !lean_is_exclusive(x_4);
if (x_11 == 0)
{
lean_object* x_12; 
x_12 = lean_ctor_get(x_4, 0);
lean_ctor_set(x_2, 3, x_10);
lean_ctor_set(x_2, 2, x_9);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_12);
lean_ctor_set(x_4, 0, x_2);
return x_4;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_ctor_get(x_4, 0);
x_14 = lean_ctor_get(x_4, 1);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_4);
lean_ctor_set(x_2, 3, x_10);
lean_ctor_set(x_2, 2, x_9);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_13);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_2);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_16 = lean_ctor_get(x_2, 0);
lean_inc(x_16);
lean_dec(x_2);
x_17 = lean_ctor_get(x_1, 1);
lean_inc(x_17);
x_18 = lean_ctor_get(x_1, 2);
lean_inc(x_18);
x_19 = lean_ctor_get(x_1, 3);
lean_inc(x_19);
lean_dec_ref(x_1);
x_20 = lean_ctor_get(x_16, 0);
lean_inc_ref(x_20);
x_21 = lean_ctor_get(x_16, 1);
lean_inc(x_21);
if (lean_is_exclusive(x_16)) {
 lean_ctor_release(x_16, 0);
 lean_ctor_release(x_16, 1);
 x_22 = x_16;
} else {
 lean_dec_ref(x_16);
 x_22 = lean_box(0);
}
x_23 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_23, 0, x_20);
lean_ctor_set(x_23, 1, x_17);
lean_ctor_set(x_23, 2, x_18);
lean_ctor_set(x_23, 3, x_19);
if (lean_is_scalar(x_22)) {
 x_24 = lean_alloc_ctor(0, 2, 0);
} else {
 x_24 = x_22;
}
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_21);
return x_24;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
x_6 = lean_ctor_get(x_1, 3);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 4);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = !lean_is_exclusive(x_2);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lean_ctor_get(x_2, 1);
x_10 = lean_ctor_get(x_2, 2);
x_11 = lean_ctor_get(x_2, 3);
lean_dec(x_11);
x_12 = lean_ctor_get(x_2, 0);
lean_dec(x_12);
x_13 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_13);
lean_dec_ref(x_3);
lean_ctor_set(x_2, 3, x_6);
lean_ctor_set(x_2, 2, x_5);
lean_ctor_set(x_2, 1, x_4);
lean_ctor_set(x_2, 0, x_13);
x_14 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_14, 0, x_2);
lean_ctor_set(x_14, 1, x_7);
lean_ctor_set(x_14, 2, x_10);
lean_ctor_set(x_14, 3, x_9);
return x_14;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_2, 1);
x_16 = lean_ctor_get(x_2, 2);
lean_inc(x_16);
lean_inc(x_15);
lean_dec(x_2);
x_17 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_17);
lean_dec_ref(x_3);
x_18 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_4);
lean_ctor_set(x_18, 2, x_5);
lean_ctor_set(x_18, 3, x_6);
x_19 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_7);
lean_ctor_set(x_19, 2, x_16);
lean_ctor_set(x_19, 3, x_15);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommRing_toAddCommGroupWithOne___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_StacksAttribute(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_StacksAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
