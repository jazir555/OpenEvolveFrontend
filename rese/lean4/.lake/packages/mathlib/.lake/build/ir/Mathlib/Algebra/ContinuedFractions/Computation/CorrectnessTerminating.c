// Lean compiler output
// Module: Mathlib.Algebra.ContinuedFractions.Computation.CorrectnessTerminating
// Imports: public import Init public import Mathlib.Algebra.ContinuedFractions.Computation.Translations public import Mathlib.Algebra.ContinuedFractions.TerminatedStable public import Mathlib.Algebra.ContinuedFractions.ContinuantsRecurrence public import Mathlib.Order.Filter.AtTopBot.Basic public import Mathlib.Tactic.FieldSimp public import Mathlib.Tactic.Ring
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
lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GenContFract_compExactValue___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_GenContFract_nextConts___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GenContFract_compExactValue(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GenContFract_compExactValue___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_6 = lean_ctor_get(x_2, 5);
lean_inc_ref(x_6);
lean_dec_ref(x_2);
x_7 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_7);
x_8 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_7);
x_9 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_8);
x_10 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_9);
x_11 = lean_ctor_get(x_10, 1);
lean_inc(x_11);
lean_dec_ref(x_10);
lean_inc(x_5);
x_12 = lean_apply_2(x_6, x_5, x_11);
x_13 = lean_unbox(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
lean_inc_ref(x_1);
x_14 = lp_mathlib_Field_toDivisionRing___redArg(x_1);
x_15 = lean_ctor_get(x_14, 0);
lean_inc_ref(x_15);
x_16 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_15);
x_17 = lean_ctor_get(x_16, 1);
lean_inc_ref(x_17);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_17, 2);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lp_mathlib_Field_toSemifield___redArg(x_1);
lean_dec_ref(x_1);
x_20 = lp_mathlib_Semifield_toCommGroupWithZero___redArg(x_19);
x_21 = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(x_20);
x_22 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_21);
lean_dec_ref(x_21);
x_23 = lean_ctor_get(x_22, 1);
lean_inc(x_23);
lean_dec_ref(x_22);
lean_inc_ref(x_14);
x_24 = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(x_14);
x_25 = lean_ctor_get(x_24, 2);
lean_inc(x_25);
lean_dec_ref(x_24);
x_26 = lean_apply_1(x_23, x_5);
x_27 = lp_mathlib_GenContFract_nextConts___redArg(x_14, x_18, x_26, x_3, x_4);
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
x_29 = lean_ctor_get(x_27, 1);
lean_inc(x_29);
lean_dec_ref(x_27);
x_30 = lean_apply_2(x_25, x_28, x_29);
return x_30;
}
else
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
lean_dec(x_5);
lean_dec_ref(x_3);
x_31 = lp_mathlib_Field_toDivisionRing___redArg(x_1);
x_32 = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(x_31);
x_33 = lean_ctor_get(x_32, 2);
lean_inc(x_33);
lean_dec_ref(x_32);
x_34 = lean_ctor_get(x_4, 0);
lean_inc(x_34);
x_35 = lean_ctor_get(x_4, 1);
lean_inc(x_35);
lean_dec_ref(x_4);
x_36 = lean_apply_2(x_33, x_34, x_35);
return x_36;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GenContFract_compExactValue(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_GenContFract_compExactValue___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_ContinuedFractions_Computation_Translations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_ContinuedFractions_TerminatedStable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_ContinuedFractions_ContinuantsRecurrence(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_AtTopBot_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FieldSimp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_ContinuedFractions_Computation_CorrectnessTerminating(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_ContinuedFractions_Computation_Translations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_ContinuedFractions_TerminatedStable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_ContinuedFractions_ContinuantsRecurrence(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_AtTopBot_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FieldSimp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
