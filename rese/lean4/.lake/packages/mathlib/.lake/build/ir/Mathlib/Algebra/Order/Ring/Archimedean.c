// Lean compiler output
// Module: Mathlib.Algebra.Order.Ring.Archimedean
// Imports: public import Init public import Mathlib.Algebra.Order.Archimedean.Class public import Mathlib.Algebra.Order.Group.DenselyOrdered public import Mathlib.Algebra.Order.Ring.Basic public import Mathlib.Algebra.Order.Hom.Ring public import Mathlib.RingTheory.Valuation.Basic
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
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulNat(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAddCommMagma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAdd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ArchimedeanClass_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulInt___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulInt(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAddCommMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAdd___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAddCommMagma(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ArchimedeanClass_mk___redArg(lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulInt___redArg(lean_object*);
lean_object* lp_mathlib_ArchimedeanClass_lift_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulInt___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instNeg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAdd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulNat___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instNeg___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulNat___redArg(lean_object*);
lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAddCommMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instNeg___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lean_ctor_get(x_3, 2);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_ArchimedeanClass_mk___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ArchimedeanClass_instZero___redArg(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instZero___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ArchimedeanClass_instZero(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAdd___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lp_mathlib_ArchimedeanClass_mk___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAdd___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_2);
x_3 = lp_mathlib_Ring_toAddCommGroup___redArg(x_2);
x_4 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_2);
x_5 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_ArchimedeanClass_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_ArchimedeanClass_lift_u2082___boxed), 9, 7);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_3);
lean_closure_set(x_9, 2, x_1);
lean_closure_set(x_9, 3, lean_box(0));
lean_closure_set(x_9, 4, lean_box(0));
lean_closure_set(x_9, 5, x_8);
lean_closure_set(x_9, 6, lean_box(0));
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ArchimedeanClass_instAdd___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulNat___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 3);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
x_6 = lp_mathlib_ArchimedeanClass_mk___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulNat___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_ArchimedeanClass_instSMulNat___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulNat(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ArchimedeanClass_instSMulNat___redArg(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulNat___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ArchimedeanClass_instSMulNat(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAddCommMagma(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ArchimedeanClass_instAdd___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAddCommMagma___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ArchimedeanClass_instAdd___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAddCommMonoid___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lean_ctor_get(x_4, 3);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_apply_2(x_5, x_2, x_3);
x_7 = lp_mathlib_ArchimedeanClass_mk___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAddCommMonoid___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_ArchimedeanClass_instAddCommMonoid___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_2);
lean_inc_ref(x_2);
x_4 = lp_mathlib_ArchimedeanClass_instAdd___redArg(x_1, x_2);
x_5 = lp_mathlib_ArchimedeanClass_instZero___redArg(x_2);
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
lean_ctor_set(x_6, 2, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instAddCommMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ArchimedeanClass_instAddCommMonoid___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instNeg___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_ArchimedeanClass_mk___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instNeg___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_2);
x_3 = lp_mathlib_Field_toDivisionRing___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_Ring_toAddCommGroup___redArg(x_4);
x_6 = lp_mathlib_Field_toSemifield___redArg(x_2);
lean_dec_ref(x_2);
x_7 = lp_mathlib_Semifield_toCommGroupWithZero___redArg(x_6);
x_8 = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(x_7);
x_9 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_8);
lean_dec_ref(x_8);
x_10 = lean_ctor_get(x_9, 1);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ArchimedeanClass_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(x_11, 0, x_10);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ArchimedeanClass_lift___boxed), 8, 7);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_5);
lean_closure_set(x_12, 2, x_1);
lean_closure_set(x_12, 3, lean_box(0));
lean_closure_set(x_12, 4, lean_box(0));
lean_closure_set(x_12, 5, x_11);
lean_closure_set(x_12, 6, lean_box(0));
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instNeg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ArchimedeanClass_instNeg___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulInt___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 3);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
x_6 = lp_mathlib_ArchimedeanClass_mk___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulInt___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ArchimedeanClass_instSMulInt___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulInt(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ArchimedeanClass_instSMulInt___redArg(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ArchimedeanClass_instSMulInt___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ArchimedeanClass_instSMulInt(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Archimedean_Class(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_DenselyOrdered(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Valuation_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Archimedean(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Archimedean_Class(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_DenselyOrdered(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Valuation_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
