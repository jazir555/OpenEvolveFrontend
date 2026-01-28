// Lean compiler output
// Module: Mathlib.Algebra.Order.Positive.Field
// Imports: public import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.Order.Positive.Ring
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
LEAN_EXPORT lean_object* lp_mathlib_Positive_instCommGroupSubtypeLtOfNat___redArg___boxed(lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instCommGroupSubtypeLtOfNat(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_Subtype_inv___redArg___boxed(lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instCommGroupSubtypeLtOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Positive_instMonoidSubtypeLtOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_Subtype_inv(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_Subtype_inv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatInt__mathlib___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatInt__mathlib___redArg(lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_Subtype_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_Subtype_inv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instCommGroupSubtypeLtOfNat___redArg(lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatInt__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatInt__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_Subtype_inv___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_Subtype_inv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_Field_toSemifield___redArg(x_1);
x_3 = lp_mathlib_Semifield_toCommGroupWithZero___redArg(x_2);
x_4 = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(x_3);
x_5 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_4);
lean_dec_ref(x_4);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Positive_Subtype_inv___redArg___lam__0), 2, 1);
lean_closure_set(x_7, 0, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_Subtype_inv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Positive_Subtype_inv___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_Subtype_inv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Positive_Subtype_inv(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_Subtype_inv___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Positive_Subtype_inv___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatInt__mathlib___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 3);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_3, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatInt__mathlib___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Positive_instPowSubtypeLtOfNatInt__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatInt__mathlib(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Positive_instPowSubtypeLtOfNatInt__mathlib___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatInt__mathlib___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Positive_instPowSubtypeLtOfNatInt__mathlib(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instCommGroupSubtypeLtOfNat___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_2 = lp_mathlib_Field_toSemifield___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lp_mathlib_Positive_instMonoidSubtypeLtOfNat___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
x_7 = lp_mathlib_Positive_Subtype_inv___redArg(x_1);
lean_inc(x_7);
lean_inc_ref(x_4);
x_8 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_4);
lean_closure_set(x_8, 2, x_7);
lean_inc(x_5);
lean_inc(x_6);
x_9 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_6);
lean_closure_set(x_9, 2, x_5);
lean_inc(x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_6);
lean_closure_set(x_10, 2, x_5);
lean_closure_set(x_10, 3, x_7);
lean_closure_set(x_10, 4, x_9);
x_11 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_11, 0, x_4);
lean_ctor_set(x_11, 1, x_7);
lean_ctor_set(x_11, 2, x_8);
lean_ctor_set(x_11, 3, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instCommGroupSubtypeLtOfNat(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Positive_instCommGroupSubtypeLtOfNat___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instCommGroupSubtypeLtOfNat___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Positive_instCommGroupSubtypeLtOfNat(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instCommGroupSubtypeLtOfNat___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Positive_instCommGroupSubtypeLtOfNat___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Positive_Field(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
