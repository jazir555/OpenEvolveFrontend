// Lean compiler output
// Module: Mathlib.SetTheory.Cardinal.Subfield
// Imports: public import Init public import Mathlib.Algebra.Field.Subfield.Basic public import Mathlib.Data.W.Cardinal public import Mathlib.Tactic.FinCases
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
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_SetTheory_Cardinal_Subfield_0__Subfield_rangeOfWType___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivisionMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_SetTheory_Cardinal_Subfield_0__Subfield_operate___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_SetTheory_Cardinal_Subfield_0__Subfield_operate(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_SetTheory_Cardinal_Subfield_0__Subfield_rangeOfWType(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_SetTheory_Cardinal_Subfield_0__Subfield_operate___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lp_mathlib_Ring_toAddCommGroup___redArg(x_3);
x_5 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_4);
lean_dec_ref(x_4);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
lean_inc_ref(x_3);
x_7 = lp_mathlib_Ring_toNonAssocRing___redArg(x_3);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
x_9 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_8);
x_10 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_9);
x_11 = lean_ctor_get(x_10, 1);
lean_inc(x_11);
lean_dec_ref(x_10);
lean_inc_ref(x_3);
x_12 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_3);
x_13 = lean_ctor_get(x_12, 1);
lean_inc_ref(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_2, 0);
lean_inc(x_14);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_15 = lean_ctor_get(x_13, 2);
lean_inc(x_15);
lean_dec_ref(x_13);
x_16 = lean_ctor_get(x_2, 1);
lean_inc(x_16);
lean_dec_ref(x_2);
x_17 = lean_ctor_get(x_14, 0);
lean_inc(x_17);
lean_dec_ref(x_14);
x_18 = lean_unsigned_to_nat(0u);
x_19 = lean_nat_dec_eq(x_17, x_18);
if (x_19 == 1)
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
lean_dec(x_17);
lean_dec(x_15);
lean_dec(x_11);
lean_dec(x_6);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_20 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_20);
lean_dec_ref(x_3);
x_21 = lean_ctor_get(x_20, 0);
lean_inc_ref(x_21);
lean_dec_ref(x_20);
x_22 = lean_ctor_get(x_21, 0);
lean_inc_ref(x_22);
lean_dec_ref(x_21);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec_ref(x_22);
x_24 = 0;
x_25 = lean_box(x_24);
lean_inc(x_16);
x_26 = lean_apply_1(x_16, x_25);
x_27 = lean_box(x_19);
x_28 = lean_apply_1(x_16, x_27);
x_29 = lean_apply_2(x_23, x_26, x_28);
return x_29;
}
else
{
lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_30 = lean_unsigned_to_nat(1u);
x_31 = lean_nat_sub(x_17, x_30);
lean_dec(x_17);
x_32 = lean_nat_dec_eq(x_31, x_18);
if (x_32 == 1)
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
lean_dec(x_31);
lean_dec(x_15);
lean_dec(x_11);
lean_dec(x_6);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_33 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_33);
lean_dec_ref(x_3);
x_34 = lean_ctor_get(x_33, 0);
lean_inc_ref(x_34);
lean_dec_ref(x_33);
x_35 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_34);
x_36 = lean_ctor_get(x_35, 0);
lean_inc(x_36);
lean_dec_ref(x_35);
x_37 = lean_box(x_19);
lean_inc(x_16);
x_38 = lean_apply_1(x_16, x_37);
x_39 = lean_box(x_32);
x_40 = lean_apply_1(x_16, x_39);
x_41 = lean_apply_2(x_36, x_38, x_40);
return x_41;
}
else
{
lean_object* x_42; uint8_t x_43; 
x_42 = lean_nat_sub(x_31, x_30);
lean_dec(x_31);
x_43 = lean_nat_dec_eq(x_42, x_18);
if (x_43 == 1)
{
lean_object* x_44; lean_object* x_45; lean_object* x_46; 
lean_dec(x_42);
lean_dec(x_15);
lean_dec(x_11);
lean_dec_ref(x_1);
x_44 = lean_box(0);
x_45 = lean_apply_1(x_16, x_44);
x_46 = lean_apply_1(x_6, x_45);
return x_46;
}
else
{
lean_object* x_47; uint8_t x_48; 
lean_dec(x_6);
x_47 = lean_nat_sub(x_42, x_30);
lean_dec(x_42);
x_48 = lean_nat_dec_eq(x_47, x_18);
if (x_48 == 1)
{
lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; 
lean_dec(x_47);
lean_dec(x_15);
lean_dec(x_11);
x_49 = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(x_1);
lean_dec_ref(x_1);
x_50 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_49);
x_51 = lp_mathlib_GroupWithZero_toDivisionMonoid___redArg(x_50);
x_52 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_51);
lean_dec_ref(x_51);
x_53 = lean_ctor_get(x_52, 1);
lean_inc(x_53);
lean_dec_ref(x_52);
x_54 = lean_box(0);
x_55 = lean_apply_1(x_16, x_54);
x_56 = lean_apply_1(x_53, x_55);
return x_56;
}
else
{
lean_object* x_57; uint8_t x_58; 
lean_dec(x_16);
lean_dec_ref(x_1);
x_57 = lean_nat_sub(x_47, x_30);
lean_dec(x_47);
x_58 = lean_nat_dec_eq(x_57, x_18);
if (x_58 == 1)
{
lean_dec(x_57);
lean_dec(x_15);
return x_11;
}
else
{
lean_object* x_59; uint8_t x_60; 
lean_dec(x_11);
x_59 = lean_nat_sub(x_57, x_30);
lean_dec(x_57);
x_60 = lean_nat_dec_eq(x_59, x_18);
lean_dec(x_59);
return x_15;
}
}
}
}
}
}
else
{
lean_object* x_61; 
lean_dec_ref(x_13);
lean_dec(x_11);
lean_dec(x_6);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_61 = lean_ctor_get(x_14, 0);
lean_inc(x_61);
lean_dec_ref(x_14);
return x_61;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_SetTheory_Cardinal_Subfield_0__Subfield_operate(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib___private_Mathlib_SetTheory_Cardinal_Subfield_0__Subfield_operate___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_SetTheory_Cardinal_Subfield_0__Subfield_rangeOfWType(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_SetTheory_Cardinal_Subfield_0__Subfield_rangeOfWType___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib___private_Mathlib_SetTheory_Cardinal_Subfield_0__Subfield_rangeOfWType(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_W_Cardinal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Subfield(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_W_Cardinal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
