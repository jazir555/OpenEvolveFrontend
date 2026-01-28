// Lean compiler output
// Module: Mathlib.Algebra.Group.Defs
// Imports: public import Init public import Batteries.Logic public import Mathlib.Algebra.Notation.Defs public import Mathlib.Algebra.Regular.Defs public import Mathlib.Data.Int.Notation public import Mathlib.Data.Nat.BinaryRec public import Mathlib.Tactic.MkIffOfInductiveProp public import Mathlib.Tactic.OfNat public import Mathlib.Tactic.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Group_toDivisionMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_npow__zero___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCancelCommMonoid___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toSubtractionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_ofIsAddCommutative___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_ofIsAddCommutative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toNatSMul___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_ofIsAddCommutative___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___redArg(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toSubtractionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_ofIsMulCommutative___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__21;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25;
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toNatPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___redArg(lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toDivisionAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22;
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCancelCommMonoid(lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_toZPow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCancelCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_ofIsMulCommutative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Group_toCancelMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Group_toDivisionMonoid(lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20;
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCommMonoid___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_zpowRec___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__8;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toNatPow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_ofIsMulCommutative___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_npow__succ___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toDivisionAddCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_ofIsAddCommutative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___redArg___boxed(lean_object*);
lean_object* l_Array_empty(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_ofIsAddCommutative___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_toZPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_ofIsAddCommutative(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toDivisionCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toDivisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCancelCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_zsmul__zero_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___boxed(lean_object*, lean_object*);
lean_object* l_nsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__6;
lean_object* lp_mathlib_Nat_binaryRec___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_x27_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid(lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_forgetful_x20inheritance;
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toAddCancelMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toAddCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_toZPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_ofIsMulCommutative___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_x27_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__23;
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCancelCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toSubtractionMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_x27_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26;
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Group_toCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_toZSMul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___redArg___boxed(lean_object*);
lean_object* lean_nat_abs(lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toNatPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_ofIsMulCommutative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_ofIsAddCommutative___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_lower_x20cancel_x20priority;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16;
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toDivisionCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31;
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toNatSMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toNatSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_zsmul__succ_x27___autoParam;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toNatSMul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_ofIsMulCommutative(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toDivisionCommMonoid___redArg___boxed(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_ofIsAddCommutative(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Group_toCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_ofIsMulCommutative___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Group_toDivisionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toDivisionAddCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27;
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCancelCommMonoid___redArg(lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_zpow__neg_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_zpow__succ_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toAddCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCancelCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_nsmul__succ___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub__eq__add__neg___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___redArg___boxed(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCommMonoid___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_x27_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toDivisionAddCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_zsmul__neg_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_Group_toCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_toZSMul___boxed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_ofIsMulCommutative(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_nsmulBinRec_go___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toSubtractionMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__14;
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17;
LEAN_EXPORT lean_object* lp_mathlib_Group_toDivisionMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toAddCancelMonoid___redArg(lean_object*);
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30;
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_zpow__zero_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_toZSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_toZSMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommSemigroup_toCommMagma(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommSemigroup_toCommMagma___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCommSemigroup_toAddCommMagma(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCommSemigroup_toAddCommMagma___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_lower_x20cancel_x20priority() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_nat_dec_eq(x_1, x_5);
if (x_6 == 1)
{
lean_object* x_7; 
lean_dec(x_4);
x_7 = lean_apply_1(x_3, x_2);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_dec(x_3);
x_8 = lean_unsigned_to_nat(1u);
x_9 = lean_nat_sub(x_1, x_8);
x_10 = lean_apply_2(x_4, x_9, x_2);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_match__1_splitter___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_match__1_splitter(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_match__1_splitter___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_match__1_splitter___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_forgetful_x20inheritance() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_npowBinRec_go___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__1(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
if (x_2 == 0)
{
x_7 = x_5;
goto block_10;
}
else
{
lean_object* x_11; 
lean_inc(x_1);
lean_inc(x_6);
x_11 = lean_apply_2(x_1, x_5, x_6);
x_7 = x_11;
goto block_10;
}
block_10:
{
lean_object* x_8; lean_object* x_9; 
lean_inc(x_6);
x_8 = lean_apply_2(x_1, x_6, x_6);
x_9 = lean_apply_2(x_4, x_7, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; lean_object* x_8; 
x_7 = lean_unbox(x_2);
x_8 = lp_mathlib_npowBinRec_go___redArg___lam__1(x_1, x_7, x_3, x_4, x_5, x_6);
lean_dec(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_npowBinRec_go___redArg___lam__0___boxed), 2, 0);
x_6 = lean_alloc_closure((void*)(lp_mathlib_npowBinRec_go___redArg___lam__1___boxed), 6, 1);
lean_closure_set(x_6, 0, x_1);
x_7 = lp_mathlib_Nat_binaryRec___redArg(x_5, x_6, x_2);
lean_dec_ref(x_5);
x_8 = lean_apply_2(x_7, x_3, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_npowBinRec_go___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_npowBinRec_go(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_npowBinRec_go___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
static lean_object* _init_lp_mathlib_nsmulBinRec_go___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_npowBinRec_go___redArg___lam__0___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lp_mathlib_nsmulBinRec_go___redArg___closed__0;
x_6 = lean_alloc_closure((void*)(lp_mathlib_npowBinRec_go___redArg___lam__1___boxed), 6, 1);
lean_closure_set(x_6, 0, x_1);
x_7 = lp_mathlib_Nat_binaryRec___redArg(x_5, x_6, x_2);
x_8 = lean_apply_2(x_7, x_3, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_nsmulBinRec_go___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_nsmulBinRec_go(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_nsmulBinRec_go___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_npowBinRec_go___redArg(x_3, x_4, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_npowBinRec_go___redArg(x_2, x_3, x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_npowBinRec(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_npowBinRec___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_nsmulBinRec_go___redArg(x_3, x_4, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_nsmulBinRec_go___redArg(x_2, x_3, x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_nsmulBinRec(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_nsmulBinRec___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_nat_dec_eq(x_3, x_5);
if (x_6 == 1)
{
lean_dec(x_4);
lean_dec(x_2);
lean_inc(x_1);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_sub(x_3, x_7);
x_9 = lean_nat_dec_eq(x_8, x_5);
if (x_9 == 1)
{
lean_dec(x_8);
lean_dec(x_2);
return x_4;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_nat_sub(x_8, x_7);
lean_dec(x_8);
x_11 = lean_nat_add(x_10, x_7);
lean_dec(x_10);
lean_inc(x_4);
lean_inc(x_2);
x_12 = lp_mathlib_npowRec_x27___redArg(x_1, x_2, x_11, x_4);
lean_dec(x_11);
x_13 = lean_apply_2(x_2, x_12, x_4);
return x_13;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_npowRec_x27___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_npowRec_x27(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_npowRec_x27___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_nat_dec_eq(x_3, x_5);
if (x_6 == 1)
{
lean_dec(x_4);
lean_dec(x_2);
lean_inc(x_1);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_sub(x_3, x_7);
x_9 = lean_nat_dec_eq(x_8, x_5);
if (x_9 == 1)
{
lean_dec(x_8);
lean_dec(x_2);
return x_4;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_nat_sub(x_8, x_7);
lean_dec(x_8);
x_11 = lean_nat_add(x_10, x_7);
lean_dec(x_10);
lean_inc(x_4);
lean_inc(x_2);
x_12 = lp_mathlib_nsmulRec_x27___redArg(x_1, x_2, x_11, x_4);
lean_dec(x_11);
x_13 = lean_apply_2(x_2, x_12, x_4);
return x_13;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_nsmulRec_x27___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_nsmulRec_x27(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_nsmulRec_x27___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_x27_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_nat_dec_eq(x_1, x_6);
if (x_7 == 1)
{
lean_object* x_8; 
lean_dec(x_5);
lean_dec(x_4);
x_8 = lean_apply_1(x_3, x_2);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; uint8_t x_11; 
lean_dec(x_3);
x_9 = lean_unsigned_to_nat(1u);
x_10 = lean_nat_sub(x_1, x_9);
x_11 = lean_nat_dec_eq(x_10, x_6);
if (x_11 == 1)
{
lean_object* x_12; 
lean_dec(x_10);
lean_dec(x_5);
x_12 = lean_apply_1(x_4, x_2);
return x_12;
}
else
{
lean_object* x_13; lean_object* x_14; 
lean_dec(x_4);
x_13 = lean_nat_sub(x_10, x_9);
lean_dec(x_10);
x_14 = lean_apply_2(x_5, x_13, x_2);
return x_14;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_x27_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_x27_match__1_splitter___redArg(x_3, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_x27_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_x27_match__1_splitter(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_x27_match__1_splitter___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib___private_Mathlib_Algebra_Group_Defs_0__npowRec_x27_match__1_splitter___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = l_npowRec___redArg(x_3, x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = l_npowRec___redArg(x_2, x_1, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_npowRecAuto(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
lean_dec(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_npowRecAuto___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = l_nsmulRec___redArg(x_3, x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = l_nsmulRec___redArg(x_2, x_1, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_nsmulRecAuto(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
lean_dec(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_nsmulRecAuto___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_npowBinRec_go___redArg(x_2, x_4, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_npowBinRec_go___redArg(x_1, x_3, x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_npowBinRecAuto(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_npowBinRecAuto___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_nsmulBinRec_go___redArg(x_2, x_4, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_nsmulBinRec_go___redArg(x_1, x_3, x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_nsmulBinRecAuto(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_nsmulBinRecAuto___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lean", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Parser", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticSeq", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__3;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2;
x_3 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1;
x_4 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticSeq1Indented", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__6;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2;
x_3 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1;
x_4 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("null", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__8;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("intros", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2;
x_3 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1;
x_4 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__14;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__18() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(";", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__18;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__21() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticRfl", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__21;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2;
x_3 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1;
x_4 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__23() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("rfl", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__23;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31;
x_2 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__succ___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
lean_inc(x_3);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddMonoid_toAddZeroClass(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Monoid_npow__zero___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Monoid_npow__succ___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
lean_inc(x_3);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Monoid_toMulOneClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Monoid_toMulOneClass(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toNatPow___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_3, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toNatPow___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Monoid_toNatPow___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toNatPow(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Monoid_toNatPow___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toNatSMul(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 2);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toNatSMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 2);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toNatSMul___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddMonoid_toNatSMul(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toNatSMul___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddMonoid_toNatSMul___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCommMonoid_toAddCommSemigroup(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCommMonoid_toAddCommSemigroup___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommMonoid_toCommSemigroup(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommMonoid_toCommSemigroup___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RightCancelMonoid_toRightCancelSemigroup(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CancelMonoid_toRightCancelMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CancelMonoid_toRightCancelMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CancelCommMonoid_toLeftCancelMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CancelCommMonoid_toCancelMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CancelCommMonoid_toCancelMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_zpowRec___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lp_mathlib_zpowRec___redArg___closed__0;
x_6 = lean_int_dec_lt(x_3, x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
lean_dec(x_1);
x_7 = lean_nat_abs(x_3);
x_8 = lean_apply_2(x_2, x_7, x_4);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lean_nat_abs(x_3);
x_10 = lean_unsigned_to_nat(1u);
x_11 = lean_nat_sub(x_9, x_10);
lean_dec(x_9);
x_12 = lean_nat_add(x_11, x_10);
lean_dec(x_11);
x_13 = lean_apply_2(x_2, x_12, x_4);
x_14 = lean_apply_1(x_1, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_zpowRec___redArg(x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_zpowRec(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec(x_3);
lean_dec(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_zpowRec___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lp_mathlib_zpowRec___redArg___closed__0;
x_6 = lean_int_dec_lt(x_3, x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
lean_dec(x_1);
x_7 = lean_nat_abs(x_3);
x_8 = lean_apply_2(x_2, x_7, x_4);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lean_nat_abs(x_3);
x_10 = lean_unsigned_to_nat(1u);
x_11 = lean_nat_sub(x_9, x_10);
lean_dec(x_9);
x_12 = lean_nat_add(x_11, x_10);
lean_dec(x_11);
x_13 = lean_apply_2(x_2, x_12, x_4);
x_14 = lean_apply_1(x_1, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_zsmulRec___redArg(x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_zsmulRec(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec(x_3);
lean_dec(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_zsmulRec___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_1(x_2, x_4);
x_8 = lean_apply_2(x_6, x_3, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DivInvMonoid_div_x27___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DivInvMonoid_div_x27(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DivInvMonoid_div_x27___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_zpow__zero_x27___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_zpow__succ_x27___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_zpow__neg_x27___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_2, x_4);
x_7 = lean_apply_2(x_5, x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_SubNegMonoid_sub_x27___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_SubNegMonoid_sub__eq__add__neg___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
static lean_object* _init_lp_mathlib_SubNegMonoid_zsmul__zero_x27___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
static lean_object* _init_lp_mathlib_SubNegMonoid_zsmul__succ_x27___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
static lean_object* _init_lp_mathlib_SubNegMonoid_zsmul__neg_x27___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_toZPow___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
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
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_toZPow___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_toZPow___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_toZPow(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_DivInvMonoid_toZPow___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_toZSMul(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 3);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_toZSMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 3);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_toZSMul___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SubNegMonoid_toZSMul(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_toZSMul___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SubNegMonoid_toZSMul___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
lean_inc(x_4);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
lean_inc(x_4);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_DivInvOneMonoid_toInvOneClass(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SubtractionMonoid_toInvolutiveNeg(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SubtractionMonoid_toInvolutiveNeg___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_DivisionMonoid_toInvolutiveInv(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DivisionMonoid_toInvolutiveInv___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SubtractionCommMonoid_toAddCommMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_DivisionCommMonoid_toCommMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DivisionCommMonoid_toCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Group_toDivisionMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Group_toDivisionMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Group_toDivisionMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Group_toDivisionMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Group_toDivisionMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Group_toDivisionMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toSubtractionMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toSubtractionMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toSubtractionMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddGroup_toSubtractionMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toSubtractionMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddGroup_toSubtractionMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Group_toCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Group_toCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Group_toCancelMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Group_toCancelMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Group_toCancelMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Group_toCancelMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toAddCancelMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toAddCancelMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toAddCancelMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddGroup_toAddCancelMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_toAddCancelMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddGroup_toAddCancelMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCommGroup_toAddCommMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCommGroup_toAddCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommGroup_toCommMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommGroup_toCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCancelCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCancelCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCancelCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommGroup_toCancelCommMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toCancelCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommGroup_toCancelCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCancelCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCancelCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCancelCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCommGroup_toAddCancelCommMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toAddCancelCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCommGroup_toAddCancelCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toDivisionCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toDivisionCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toDivisionCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CommGroup_toDivisionCommMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_toDivisionCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommGroup_toDivisionCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toDivisionAddCommMonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toDivisionAddCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toDivisionAddCommMonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCommGroup_toDivisionAddCommMonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toDivisionAddCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCommGroup_toDivisionAddCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_ofIsMulCommutative(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_ofIsMulCommutative___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_ofIsMulCommutative___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CommMonoid_ofIsMulCommutative(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_ofIsMulCommutative___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommMonoid_ofIsMulCommutative___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_ofIsAddCommutative(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_ofIsAddCommutative___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_ofIsAddCommutative___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddCommMonoid_ofIsAddCommutative(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_ofIsAddCommutative___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCommMonoid_ofIsAddCommutative___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_ofIsMulCommutative(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_ofIsMulCommutative___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_ofIsMulCommutative___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CommGroup_ofIsMulCommutative(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroup_ofIsMulCommutative___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommGroup_ofIsMulCommutative___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_ofIsAddCommutative(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_ofIsAddCommutative___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_ofIsAddCommutative___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddCommGroup_ofIsAddCommutative(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_ofIsAddCommutative___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCommGroup_ofIsAddCommutative___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Logic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_BinaryRec(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_OfNat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_BinaryRec(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_OfNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_lower_x20cancel_x20priority = _init_lp_mathlib_LibraryNote_lower_x20cancel_x20priority();
lean_mark_persistent(lp_mathlib_LibraryNote_lower_x20cancel_x20priority);
lp_mathlib_LibraryNote_forgetful_x20inheritance = _init_lp_mathlib_LibraryNote_forgetful_x20inheritance();
lean_mark_persistent(lp_mathlib_LibraryNote_forgetful_x20inheritance);
lp_mathlib_nsmulBinRec_go___redArg___closed__0 = _init_lp_mathlib_nsmulBinRec_go___redArg___closed__0();
lean_mark_persistent(lp_mathlib_nsmulBinRec_go___redArg___closed__0);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__3 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__3();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__3);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__6 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__6();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__6);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__8 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__8();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__8);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__14 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__14();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__14);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__18 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__18();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__18);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__21 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__21();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__21);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__23 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__23();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__23);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31);
lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32 = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32);
lp_mathlib_AddMonoid_nsmul__zero___autoParam = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam);
lp_mathlib_AddMonoid_nsmul__succ___autoParam = _init_lp_mathlib_AddMonoid_nsmul__succ___autoParam();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__succ___autoParam);
lp_mathlib_Monoid_npow__zero___autoParam = _init_lp_mathlib_Monoid_npow__zero___autoParam();
lean_mark_persistent(lp_mathlib_Monoid_npow__zero___autoParam);
lp_mathlib_Monoid_npow__succ___autoParam = _init_lp_mathlib_Monoid_npow__succ___autoParam();
lean_mark_persistent(lp_mathlib_Monoid_npow__succ___autoParam);
lp_mathlib_zpowRec___redArg___closed__0 = _init_lp_mathlib_zpowRec___redArg___closed__0();
lean_mark_persistent(lp_mathlib_zpowRec___redArg___closed__0);
lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam = _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam();
lean_mark_persistent(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam);
lp_mathlib_DivInvMonoid_zpow__zero_x27___autoParam = _init_lp_mathlib_DivInvMonoid_zpow__zero_x27___autoParam();
lean_mark_persistent(lp_mathlib_DivInvMonoid_zpow__zero_x27___autoParam);
lp_mathlib_DivInvMonoid_zpow__succ_x27___autoParam = _init_lp_mathlib_DivInvMonoid_zpow__succ_x27___autoParam();
lean_mark_persistent(lp_mathlib_DivInvMonoid_zpow__succ_x27___autoParam);
lp_mathlib_DivInvMonoid_zpow__neg_x27___autoParam = _init_lp_mathlib_DivInvMonoid_zpow__neg_x27___autoParam();
lean_mark_persistent(lp_mathlib_DivInvMonoid_zpow__neg_x27___autoParam);
lp_mathlib_SubNegMonoid_sub__eq__add__neg___autoParam = _init_lp_mathlib_SubNegMonoid_sub__eq__add__neg___autoParam();
lean_mark_persistent(lp_mathlib_SubNegMonoid_sub__eq__add__neg___autoParam);
lp_mathlib_SubNegMonoid_zsmul__zero_x27___autoParam = _init_lp_mathlib_SubNegMonoid_zsmul__zero_x27___autoParam();
lean_mark_persistent(lp_mathlib_SubNegMonoid_zsmul__zero_x27___autoParam);
lp_mathlib_SubNegMonoid_zsmul__succ_x27___autoParam = _init_lp_mathlib_SubNegMonoid_zsmul__succ_x27___autoParam();
lean_mark_persistent(lp_mathlib_SubNegMonoid_zsmul__succ_x27___autoParam);
lp_mathlib_SubNegMonoid_zsmul__neg_x27___autoParam = _init_lp_mathlib_SubNegMonoid_zsmul__neg_x27___autoParam();
lean_mark_persistent(lp_mathlib_SubNegMonoid_zsmul__neg_x27___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
