// Lean compiler output
// Module: Mathlib.Algebra.GCDMonoid.Basic
// Imports: public import Init public import Mathlib.Algebra.Ring.Associated public import Mathlib.Algebra.Ring.Regular
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
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Associates_mk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Units_mk0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalize___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Quotient_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toCancelCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg(lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalize___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_associatesEquivOfUniqueUnits(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_out(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_out___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalize(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueNormalizationMonoidOfUniqueUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_associatesEquivOfUniqueUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueNormalizationMonoidOfUniqueUnits(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalize___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_apply_2(x_2, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalize___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(x_1);
x_4 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_3);
x_5 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_normalize___redArg___lam__0), 3, 2);
lean_closure_set(x_7, 0, x_2);
lean_closure_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalize(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_normalize___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_out(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_normalize___redArg(x_2, x_3);
x_6 = lean_apply_1(x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_out___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_normalize___redArg(x_1, x_2);
x_5 = lean_apply_1(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_dec(x_6);
lean_inc(x_5);
lean_ctor_set(x_3, 1, x_5);
return x_3;
}
else
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_3, 0);
lean_inc(x_7);
lean_dec(x_3);
lean_inc(x_7);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_4, 0, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueNormalizationMonoidOfUniqueUnits(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueNormalizationMonoidOfUniqueUnits___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_associatesEquivOfUniqueUnits___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(x_1);
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
lean_dec(x_5);
lean_inc_ref(x_1);
x_6 = lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Associates_out), 4, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_1);
lean_closure_set(x_7, 2, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Associates_mk___boxed), 3, 2);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_4);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_7);
return x_2;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_2, 0);
lean_inc(x_9);
lean_dec(x_2);
lean_inc_ref(x_1);
x_10 = lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Associates_out), 4, 3);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, x_1);
lean_closure_set(x_11, 2, x_10);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Associates_mk___boxed), 3, 2);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_9);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_associatesEquivOfUniqueUnits(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_associatesEquivOfUniqueUnits___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
lean_inc(x_5);
x_6 = lean_apply_2(x_1, x_5, x_2);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
lean_object* x_8; uint8_t x_9; 
lean_dec_ref(x_4);
x_8 = lp_mathlib_Units_mk0___redArg(x_3, x_5);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_8, 0);
x_11 = lean_ctor_get(x_8, 1);
lean_ctor_set(x_8, 1, x_10);
lean_ctor_set(x_8, 0, x_11);
return x_8;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_8, 0);
x_13 = lean_ctor_get(x_8, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_8);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_12);
return x_14;
}
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; uint8_t x_18; 
lean_dec(x_5);
lean_dec_ref(x_3);
x_15 = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(x_4);
x_16 = lean_ctor_get(x_15, 0);
lean_inc_ref(x_16);
lean_dec_ref(x_15);
x_17 = lp_mathlib_Monoid_toMulOneClass___redArg(x_16);
lean_dec_ref(x_16);
x_18 = !lean_is_exclusive(x_17);
if (x_18 == 0)
{
lean_object* x_19; lean_object* x_20; 
x_19 = lean_ctor_get(x_17, 0);
x_20 = lean_ctor_get(x_17, 1);
lean_dec(x_20);
lean_inc(x_19);
lean_ctor_set(x_17, 1, x_19);
return x_17;
}
else
{
lean_object* x_21; lean_object* x_22; 
x_21 = lean_ctor_get(x_17, 0);
lean_inc(x_21);
lean_dec(x_17);
lean_inc(x_21);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_21);
lean_ctor_set(x_22, 1, x_21);
return x_22;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_10; uint8_t x_11; 
lean_inc_ref(x_2);
lean_inc(x_3);
x_10 = lean_apply_2(x_2, x_4, x_3);
x_11 = lean_unbox(x_10);
if (x_11 == 0)
{
lean_dec(x_5);
lean_dec(x_3);
lean_dec_ref(x_2);
goto block_9;
}
else
{
lean_object* x_12; uint8_t x_13; 
lean_inc(x_3);
x_12 = lean_apply_2(x_2, x_5, x_3);
x_13 = lean_unbox(x_12);
if (x_13 == 0)
{
lean_dec(x_3);
goto block_9;
}
else
{
lean_dec_ref(x_1);
return x_3;
}
}
block_9:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(x_1);
x_7 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_6);
lean_dec_ref(x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
lean_inc_ref(x_1);
lean_inc(x_2);
x_6 = lean_apply_2(x_1, x_4, x_2);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
lean_object* x_8; uint8_t x_9; 
lean_inc(x_2);
x_8 = lean_apply_2(x_1, x_5, x_2);
x_9 = lean_unbox(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_dec(x_2);
x_10 = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(x_3);
x_11 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_10);
lean_dec_ref(x_10);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
return x_12;
}
else
{
lean_dec_ref(x_3);
return x_2;
}
}
else
{
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_CommGroupWithZero_toCancelCommMonoidWithZero___redArg(x_1);
lean_inc_ref(x_1);
x_4 = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
x_6 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_5);
x_7 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_6);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lean_ctor_get(x_7, 1);
x_10 = lean_ctor_get(x_7, 0);
lean_dec(x_10);
lean_inc(x_9);
lean_inc_ref(x_2);
x_11 = lean_alloc_closure((void*)(lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__0), 5, 4);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_9);
lean_closure_set(x_11, 2, x_4);
lean_closure_set(x_11, 3, x_3);
lean_inc(x_9);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_12 = lean_alloc_closure((void*)(lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__1), 5, 3);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_2);
lean_closure_set(x_12, 2, x_9);
x_13 = lean_alloc_closure((void*)(lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__2), 5, 3);
lean_closure_set(x_13, 0, x_2);
lean_closure_set(x_13, 1, x_9);
lean_closure_set(x_13, 2, x_1);
lean_ctor_set(x_7, 1, x_13);
lean_ctor_set(x_7, 0, x_12);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_11);
lean_ctor_set(x_14, 1, x_7);
return x_14;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_15 = lean_ctor_get(x_7, 1);
lean_inc(x_15);
lean_dec(x_7);
lean_inc(x_15);
lean_inc_ref(x_2);
x_16 = lean_alloc_closure((void*)(lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__0), 5, 4);
lean_closure_set(x_16, 0, x_2);
lean_closure_set(x_16, 1, x_15);
lean_closure_set(x_16, 2, x_4);
lean_closure_set(x_16, 3, x_3);
lean_inc(x_15);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_17 = lean_alloc_closure((void*)(lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__1), 5, 3);
lean_closure_set(x_17, 0, x_1);
lean_closure_set(x_17, 1, x_2);
lean_closure_set(x_17, 2, x_15);
x_18 = lean_alloc_closure((void*)(lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg___lam__2), 5, 3);
lean_closure_set(x_18, 0, x_2);
lean_closure_set(x_18, 1, x_15);
lean_closure_set(x_18, 2, x_1);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_16);
lean_ctor_set(x_20, 1, x_19);
return x_20;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CommGroupWithZero_instNormalizedGCDMonoid___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_box(0);
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_u2082), 10, 8);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_2);
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, lean_box(0));
lean_closure_set(x_6, 5, x_2);
lean_closure_set(x_6, 6, x_4);
lean_closure_set(x_6, 7, lean_box(0));
x_7 = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_u2082), 10, 8);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, x_2);
lean_closure_set(x_7, 3, x_2);
lean_closure_set(x_7, 4, lean_box(0));
lean_closure_set(x_7, 5, x_2);
lean_closure_set(x_7, 6, x_5);
lean_closure_set(x_7, 7, lean_box(0));
lean_ctor_set(x_1, 1, x_7);
lean_ctor_set(x_1, 0, x_6);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_u2082), 10, 8);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, x_2);
lean_closure_set(x_10, 3, x_2);
lean_closure_set(x_10, 4, lean_box(0));
lean_closure_set(x_10, 5, x_2);
lean_closure_set(x_10, 6, x_8);
lean_closure_set(x_10, 7, lean_box(0));
x_11 = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_u2082), 10, 8);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, lean_box(0));
lean_closure_set(x_11, 2, x_2);
lean_closure_set(x_11, 3, x_2);
lean_closure_set(x_11, 4, lean_box(0));
lean_closure_set(x_11, 5, x_2);
lean_closure_set(x_11, 6, x_9);
lean_closure_set(x_11, 7, lean_box(0));
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Associates_instGCDMonoid___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Associates_instGCDMonoid(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Associated(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Regular(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Associated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
