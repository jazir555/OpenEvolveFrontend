// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Submonoid.Instances
// Imports: public import Init public import Mathlib.Algebra.Group.Submonoid.Operations public import Mathlib.Algebra.GroupWithZero.Units.Lemmas
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
lean_object* lp_mathlib_SubmonoidClass_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange___redArg(lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubmonoidClass_toMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMulZeroOneClassSubtypeMemSubmonoidMrange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommMonoidWithZeroSubtypeMemSubmonoidMrange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommMonoidWithZeroSubtypeMemSubmonoidMrange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommMonoidWithZeroSubtypeMemSubmonoidMrange___redArg(lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivisionMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMulZeroOneClassSubtypeMemSubmonoidMrange___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommGroupWithZeroSubtypeMemSubmonoidMrange___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommGroupWithZeroSubtypeMemSubmonoidMrange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMulZeroOneClassSubtypeMemSubmonoidMrange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommGroupWithZeroSubtypeMemSubmonoidMrange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMulZeroOneClassSubtypeMemSubmonoidMrange___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_SubmonoidClass_toMulOneClass___redArg(x_2);
x_4 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_1);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; 
x_6 = lean_ctor_get(x_4, 0);
lean_dec(x_6);
lean_ctor_set(x_4, 0, x_3);
return x_4;
}
else
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_4, 1);
lean_inc(x_7);
lean_dec(x_4);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMulZeroOneClassSubtypeMemSubmonoidMrange(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_instMulZeroOneClassSubtypeMemSubmonoidMrange___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMulZeroOneClassSubtypeMemSubmonoidMrange___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_instMulZeroOneClassSubtypeMemSubmonoidMrange(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_1);
x_4 = lp_mathlib_SubmonoidClass_toMonoid___redArg(x_2);
x_5 = lp_mathlib_MonoidWithZeroHom_instMulZeroOneClassSubtypeMemSubmonoidMrange___redArg(x_3);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 0);
lean_dec(x_7);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
else
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec(x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_4);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommMonoidWithZeroSubtypeMemSubmonoidMrange___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(x_1);
x_4 = lp_mathlib_SubmonoidClass_toMonoid___redArg(x_2);
x_5 = lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange___redArg(x_3);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 0);
lean_dec(x_7);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
else
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec(x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_4);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommMonoidWithZeroSubtypeMemSubmonoidMrange(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_instCommMonoidWithZeroSubtypeMemSubmonoidMrange___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommMonoidWithZeroSubtypeMemSubmonoidMrange___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_instCommMonoidWithZeroSubtypeMemSubmonoidMrange(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange___redArg(x_2);
x_4 = lp_mathlib_GroupWithZero_toDivisionMonoid___redArg(x_1);
x_5 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_4);
lean_dec_ref(x_4);
x_6 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec_ref(x_5);
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 1);
lean_inc(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange___redArg___lam__0), 2, 1);
lean_closure_set(x_10, 0, x_7);
lean_inc_ref(x_10);
x_11 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, x_6);
lean_closure_set(x_11, 2, x_10);
lean_inc(x_8);
lean_inc(x_9);
x_12 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_9);
lean_closure_set(x_12, 2, x_8);
lean_inc_ref(x_10);
x_13 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_13, 0, lean_box(0));
lean_closure_set(x_13, 1, x_9);
lean_closure_set(x_13, 2, x_8);
lean_closure_set(x_13, 3, x_10);
lean_closure_set(x_13, 4, x_12);
x_14 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_14, 0, x_3);
lean_ctor_set(x_14, 1, x_10);
lean_ctor_set(x_14, 2, x_11);
lean_ctor_set(x_14, 3, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommGroupWithZeroSubtypeMemSubmonoidMrange___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_MonoidWithZeroHom_instCommMonoidWithZeroSubtypeMemSubmonoidMrange___redArg(x_2);
x_4 = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(x_1);
lean_inc_ref(x_4);
x_5 = lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange___redArg(x_4);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_7 = lean_ctor_get(x_5, 3);
lean_dec(x_7);
x_8 = lean_ctor_get(x_5, 0);
lean_dec(x_8);
x_9 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_9);
x_10 = lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange___redArg(x_9);
x_11 = lp_mathlib_GroupWithZero_toDivisionMonoid___redArg(x_4);
x_12 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_11);
lean_dec_ref(x_11);
x_13 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_13);
lean_dec_ref(x_10);
x_14 = lean_ctor_get(x_12, 1);
lean_inc(x_14);
lean_dec_ref(x_12);
x_15 = lean_ctor_get(x_13, 0);
lean_inc(x_15);
x_16 = lean_ctor_get(x_13, 1);
lean_inc(x_16);
lean_dec_ref(x_13);
x_17 = lean_alloc_closure((void*)(lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange___redArg___lam__0), 2, 1);
lean_closure_set(x_17, 0, x_14);
lean_inc(x_15);
lean_inc(x_16);
x_18 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, x_16);
lean_closure_set(x_18, 2, x_15);
x_19 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_19, 0, lean_box(0));
lean_closure_set(x_19, 1, x_16);
lean_closure_set(x_19, 2, x_15);
lean_closure_set(x_19, 3, x_17);
lean_closure_set(x_19, 4, x_18);
lean_ctor_set(x_5, 3, x_19);
lean_ctor_set(x_5, 0, x_3);
return x_5;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_20 = lean_ctor_get(x_5, 1);
x_21 = lean_ctor_get(x_5, 2);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_5);
x_22 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_22);
x_23 = lp_mathlib_MonoidWithZeroHom_instMonoidWithZeroSubtypeMemSubmonoidMrange___redArg(x_22);
x_24 = lp_mathlib_GroupWithZero_toDivisionMonoid___redArg(x_4);
x_25 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_24);
lean_dec_ref(x_24);
x_26 = lean_ctor_get(x_23, 0);
lean_inc_ref(x_26);
lean_dec_ref(x_23);
x_27 = lean_ctor_get(x_25, 1);
lean_inc(x_27);
lean_dec_ref(x_25);
x_28 = lean_ctor_get(x_26, 0);
lean_inc(x_28);
x_29 = lean_ctor_get(x_26, 1);
lean_inc(x_29);
lean_dec_ref(x_26);
x_30 = lean_alloc_closure((void*)(lp_mathlib_MonoidWithZeroHom_instGroupWithZeroSubtypeMemSubmonoidMrange___redArg___lam__0), 2, 1);
lean_closure_set(x_30, 0, x_27);
lean_inc(x_28);
lean_inc(x_29);
x_31 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_31, 0, lean_box(0));
lean_closure_set(x_31, 1, x_29);
lean_closure_set(x_31, 2, x_28);
x_32 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_32, 0, lean_box(0));
lean_closure_set(x_32, 1, x_29);
lean_closure_set(x_32, 2, x_28);
lean_closure_set(x_32, 3, x_30);
lean_closure_set(x_32, 4, x_31);
x_33 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_33, 0, x_3);
lean_ctor_set(x_33, 1, x_20);
lean_ctor_set(x_33, 2, x_21);
lean_ctor_set(x_33, 3, x_32);
return x_33;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommGroupWithZeroSubtypeMemSubmonoidMrange(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_instCommGroupWithZeroSubtypeMemSubmonoidMrange___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_instCommGroupWithZeroSubtypeMemSubmonoidMrange___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_instCommGroupWithZeroSubtypeMemSubmonoidMrange(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Submonoid_Instances(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
