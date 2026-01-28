// Lean compiler output
// Module: Mathlib.RingTheory.Ideal.Over
// Imports: public import Init public import Mathlib.Algebra.Algebra.Tower public import Mathlib.Algebra.Group.Subgroup.Actions public import Mathlib.RingTheory.Ideal.Pointwise public import Mathlib.RingTheory.Ideal.Quotient.Operations
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
lean_object* lp_mathlib_SubgroupClass_toGroup___redArg(lean_object*);
lean_object* lp_mathlib_AlgEquiv_instEquivLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraOfLiesOver(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqMap___boxed(lean_object**);
lean_object* lp_mathlib_Ideal_quotientMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_under___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DistribMulAction_toAddEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_primesOver_mk___redArg(lean_object*);
lean_object* l_Quotient_lift_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqComap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_primesOver_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraOfLiesOver___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqComap___boxed(lean_object**);
lean_object* lp_mathlib_Ideal_quotientEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraOfLiesOver___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_primesOver_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_under(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_box(0);
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(l_Quotient_lift_u2082), 9, 7);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, lean_box(0));
lean_closure_set(x_7, 3, x_2);
lean_closure_set(x_7, 4, x_2);
lean_closure_set(x_7, 5, x_6);
lean_closure_set(x_7, 6, lean_box(0));
x_8 = lp_mathlib_Ideal_quotientMap___redArg(x_5);
lean_ctor_set(x_1, 1, x_8);
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient___redArg___lam__0), 3, 1);
lean_closure_set(x_11, 0, x_9);
x_12 = lean_alloc_closure((void*)(l_Quotient_lift_u2082), 9, 7);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, lean_box(0));
lean_closure_set(x_12, 2, lean_box(0));
lean_closure_set(x_12, 3, x_2);
lean_closure_set(x_12, 4, x_2);
lean_closure_set(x_12, 5, x_11);
lean_closure_set(x_12, 6, lean_box(0));
x_13 = lp_mathlib_Ideal_quotientMap___redArg(x_10);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_under(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_under___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Ideal_under(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraOfLiesOver___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_box(0);
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(l_Quotient_lift_u2082), 9, 7);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, lean_box(0));
lean_closure_set(x_7, 3, x_2);
lean_closure_set(x_7, 4, x_2);
lean_closure_set(x_7, 5, x_6);
lean_closure_set(x_7, 6, lean_box(0));
x_8 = lp_mathlib_Ideal_quotientMap___redArg(x_5);
lean_ctor_set(x_1, 1, x_8);
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Ideal_Quotient_algebraQuotientMapQuotient___redArg___lam__0), 3, 1);
lean_closure_set(x_11, 0, x_9);
x_12 = lean_alloc_closure((void*)(l_Quotient_lift_u2082), 9, 7);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, lean_box(0));
lean_closure_set(x_12, 2, lean_box(0));
lean_closure_set(x_12, 3, x_2);
lean_closure_set(x_12, 4, x_2);
lean_closure_set(x_12, 5, x_11);
lean_closure_set(x_12, 6, lean_box(0));
x_13 = lp_mathlib_Ideal_quotientMap___redArg(x_10);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraOfLiesOver(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Ideal_Quotient_algebraOfLiesOver___redArg(x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algebraOfLiesOver___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Ideal_Quotient_algebraOfLiesOver(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqMap___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_EquivLike_toEquiv___redArg(x_3, x_4);
x_6 = lp_mathlib_Ideal_quotientEquiv___redArg(x_1, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18) {
_start:
{
lean_object* x_19; 
x_19 = lp_mathlib_Ideal_Quotient_algEquivOfEqMap___redArg(x_5, x_6, x_15, x_17);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqMap___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
_start:
{
lean_object* x_19; 
x_19 = lp_mathlib_Ideal_Quotient_algEquivOfEqMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_4);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqComap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18) {
_start:
{
lean_object* x_19; 
x_19 = lp_mathlib_Ideal_Quotient_algEquivOfEqMap___redArg(x_5, x_6, x_15, x_17);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqComap___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Ideal_Quotient_algEquivOfEqMap___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_algEquivOfEqComap___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
_start:
{
lean_object* x_19; 
x_19 = lp_mathlib_Ideal_Quotient_algEquivOfEqComap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_4);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_DistribMulAction_toAddEquiv___redArg(x_1, x_2, x_5);
lean_inc_ref(x_3);
x_7 = lp_mathlib_Ideal_Quotient_algEquivOfEqMap___redArg(x_3, x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Ideal_Quotient_stabilizerHom___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lp_mathlib_SubgroupClass_toGroup___redArg(x_4);
x_8 = lean_ctor_get(x_1, 0);
x_9 = lp_mathlib_AlgEquiv_instEquivLike(lean_box(0), lean_box(0), lean_box(0), x_8, x_6, x_6, x_3, x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_5);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Ideal_Quotient_stabilizerHom___redArg___lam__0___boxed), 5, 4);
lean_closure_set(x_11, 0, x_7);
lean_closure_set(x_11, 1, x_10);
lean_closure_set(x_11, 2, x_2);
lean_closure_set(x_11, 3, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Ideal_Quotient_stabilizerHom___redArg(x_3, x_4, x_5, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Ideal_Quotient_stabilizerHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_Quotient_stabilizerHom___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Ideal_Quotient_stabilizerHom___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_primesOver_mk(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_primesOver_mk___redArg(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_primesOver_mk___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Ideal_primesOver_mk(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Actions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Operations(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Over(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Actions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
