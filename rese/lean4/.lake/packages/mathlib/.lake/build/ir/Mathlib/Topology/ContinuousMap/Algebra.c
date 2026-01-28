// Lean compiler output
// Module: Mathlib.Topology.ContinuousMap.Algebra
// Imports: public import Init public import Mathlib.Algebra.Algebra.Pi public import Mathlib.Algebra.Algebra.Subalgebra.Basic public import Mathlib.Tactic.FieldSimp public import Mathlib.Topology.Algebra.InfiniteSum.Basic public import Mathlib.Topology.Algebra.Module.LinearMap public import Mathlib.Topology.Algebra.Ring.Basic public import Mathlib.Topology.UniformSpace.CompactConvergence
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
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeftContinuous___redArg(lean_object*);
lean_object* l_Int_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearMap_const(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalCLM___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommMonoidOfContinuousMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_compLeftContinuous___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeftContinuous___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddGroupOfIsTopologicalAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommSemigroupOfContinuousAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compRightAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonAssocSemiringOfIsTopologicalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddMonoidWithOneOfContinuousAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddMonoidOfContinuousAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommGroupContinuousMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommGroupContinuousMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_C___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommGroupContinuousMap___redArg(lean_object*);
lean_object* lp_mathlib_ContinuousMap_const___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemigroupWithZeroOfContinuousMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeftContinuous(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommRingOfIsTopologicalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddSemigroupOfContinuousAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instInvOfContinuousInv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeftContinuous___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulActionOfContinuousConstSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_toAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonAssocSemiringOfIsTopologicalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instDivOfContinuousDiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommMonoidOfContinuousMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeftContinuous(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instDistribMulActionOfContinuousConstSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compAddMonoidHom_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulActionOfContinuousConstSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddMonoidWithOneOfContinuousAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalNonAssocSemiringOfIsTopologicalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommMonoidWithZeroOfContinuousMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommGroupContinuousMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddSemigroupOfContinuousAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNegOfContinuousNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearMap_compLeftContinuous(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonAssocRingOfIsTopologicalRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubgroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemiringOfIsTopologicalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalAlgHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalCLM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compMonoidHom_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_C___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instIntCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalCommRingOfIsTopologicalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemiringOfIsTopologicalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalNonAssocSemiringOfIsTopologicalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instDistribMulActionOfContinuousConstSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddZeroClassOfContinuousAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compAddMonoidHom_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compMonoidHom_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubmonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearMap_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommSemigroupOfContinuousMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compMonoidHom_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousAddSubgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_algebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubsemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
static lean_object* lp_mathlib_ContinuousMap_coeFnAddMonoidHom___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMonoidOfContinuousMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulOneClassOfContinuousMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommMonoidOfContinuousAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeftContinuous___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalSemiringOfIsTopologicalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSubOfContinuousSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMonoidWithZeroOfContinuousMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulOneClassOfContinuousMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_ContinuousMap_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instRing___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNatCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ContinuousLinearMap_const___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeftContinuous___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousAddSubmonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnMonoidHom___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalCLM___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddMonoidOfContinuousAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalCLM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeftContinuous___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearMap_compLeftContinuous___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalSemiringOfIsTopologicalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instDivOfContinuousDiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonAssocRingOfIsTopologicalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeftContinuous___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulZeroClassOfContinuousMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousFunctions_instCoeFunElemForallSetOfContinuous___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommMonoidOfContinuousAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instDistribMulActionOfContinuousConstSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubmonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemigroupOfContinuousMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubmodule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommSemiringOfIsTopologicalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compRightAlgHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalRingOfIsTopologicalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommMonoidWithZeroOfContinuousMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommRingOfIsTopologicalRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousFunctions_instCoeFunElemForallSetOfContinuous(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalCommRingOfIsTopologicalRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instRing___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalCommSemiringOfIsTopologicalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNegOfContinuousNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommSemigroupOfContinuousMul___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalNonAssocRingOfIsTopologicalRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compRightAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZPow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalCommSemiringOfIsTopologicalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instPow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_C(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSubOfContinuousSub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMonoidWithZeroOfContinuousMul___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instVAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommSemiringOfIsTopologicalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulZeroClassOfContinuousMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instVAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_algebra___redArg(lean_object*);
lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubsemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemigroupWithZeroOfContinuousMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalRingOfIsTopologicalRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZPow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compRightAlgHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeftContinuous(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instPow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_compLeftContinuous(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearMap_compLeftContinuous___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instIntCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemigroupOfContinuousMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_algebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommSemigroupOfContinuousAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddGroupOfIsTopologicalAddGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalNonAssocRingOfIsTopologicalRing___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_C___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousAddSubmonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousAddSubgroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulActionOfContinuousConstSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddZeroClassOfContinuousAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_C___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compMonoidHom_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compAddMonoidHom_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeftContinuous___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_continuousSubmodule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_compLeftContinuous___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousFunctions_instCoeFunElemForallSetOfContinuous___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousFunctions_instCoeFunElemForallSetOfContinuous(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_ContinuousFunctions_instCoeFunElemForallSetOfContinuous___lam__0), 2, 0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
x_5 = lean_apply_1(x_2, x_4);
x_6 = lean_apply_1(x_3, x_4);
x_7 = lean_apply_2(x_1, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instMul___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAdd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instAdd___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_6, 0, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instOne___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_6, 0, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_1, x_2);
x_5 = lp_mathlib_ContinuousMap_const___redArg___lam__0(x_4, x_3);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNatCast___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNatCast(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ContinuousMap_instNatCast___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_1, x_2);
x_5 = lp_mathlib_ContinuousMap_const___redArg___lam__0(x_4, x_3);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instIntCast___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instIntCast(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ContinuousMap_instIntCast___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_3, x_4);
x_7 = lean_apply_2(x_5, x_2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNSMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNSMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instNSMul___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instPow___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_2, x_4);
x_7 = lean_apply_2(x_5, x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instPow___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instPow___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instPow(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instPow___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_2, x_3);
x_5 = lean_apply_1(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instInvOfContinuousInv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNegOfContinuousNeg___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNegOfContinuousNeg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instNegOfContinuousNeg___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instDivOfContinuousDiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instDivOfContinuousDiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instDivOfContinuousDiv___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSubOfContinuousSub___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSubOfContinuousSub(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instSubOfContinuousSub___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_3, x_4);
x_7 = lean_apply_2(x_5, x_2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZSMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZSMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instZSMul___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZPow___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_2, x_4);
x_7 = lean_apply_2(x_5, x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZPow___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZPow___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instZPow(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instZPow___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubmonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubmonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_continuousSubmonoid(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousAddSubmonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousAddSubmonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_continuousAddSubmonoid(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubgroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubgroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_continuousSubgroup(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousAddSubgroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousAddSubgroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_continuousAddSubgroup(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemigroupOfContinuousMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemigroupOfContinuousMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddSemigroupOfContinuousAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddSemigroupOfContinuousAdd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommSemigroupOfContinuousMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommSemigroupOfContinuousMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommSemigroupOfContinuousAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommSemigroupOfContinuousAdd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulOneClassOfContinuousMul___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_6, 0, x_3);
lean_ctor_set(x_1, 1, x_5);
lean_ctor_set(x_1, 0, x_6);
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
x_9 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_10, 0, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_9);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulOneClassOfContinuousMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instMulOneClassOfContinuousMul___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddZeroClassOfContinuousAdd___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_6, 0, x_3);
lean_ctor_set(x_1, 1, x_5);
lean_ctor_set(x_1, 0, x_6);
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
x_9 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_10, 0, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_9);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddZeroClassOfContinuousAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instAddZeroClassOfContinuousAdd___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulZeroClassOfContinuousMul___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_6, 0, x_3);
lean_ctor_set(x_1, 1, x_5);
lean_ctor_set(x_1, 0, x_6);
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
x_9 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_10, 0, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_9);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulZeroClassOfContinuousMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instMulZeroClassOfContinuousMul___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemigroupWithZeroOfContinuousMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
lean_dec_ref(x_2);
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
lean_dec(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_7, 0, x_3);
x_8 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_8, 0, x_5);
lean_ctor_set(x_1, 1, x_7);
lean_ctor_set(x_1, 0, x_8);
return x_1;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
lean_inc(x_9);
lean_dec(x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_10, 0, x_3);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_11, 0, x_9);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_10);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemigroupWithZeroOfContinuousMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instSemigroupWithZeroOfContinuousMul___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_1(x_3, x_4);
x_6 = lean_apply_2(x_1, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 2);
lean_inc(x_3);
x_4 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_5 = !lean_is_exclusive(x_1);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lean_ctor_get(x_1, 2);
lean_dec(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_dec(x_7);
x_8 = lean_ctor_get(x_1, 0);
lean_dec(x_8);
x_9 = lean_ctor_get(x_4, 0);
lean_inc(x_9);
lean_dec_ref(x_4);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_10, 0, x_3);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_11, 0, x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_12, 0, x_9);
lean_ctor_set(x_1, 2, x_10);
lean_ctor_set(x_1, 1, x_12);
lean_ctor_set(x_1, 0, x_11);
return x_1;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_dec(x_1);
x_13 = lean_ctor_get(x_4, 0);
lean_inc(x_13);
lean_dec_ref(x_4);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_14, 0, x_3);
x_15 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_15, 0, x_2);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_16, 0, x_13);
x_17 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
lean_ctor_set(x_17, 2, x_14);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMonoidOfContinuousMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddMonoidOfContinuousAdd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
lean_inc(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_5, 0, x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_7, 0, x_1);
x_8 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddMonoidOfContinuousAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instAddMonoidOfContinuousAdd___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMonoidWithZeroOfContinuousMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_1);
x_3 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_2);
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
lean_dec(x_6);
x_7 = lean_ctor_get(x_3, 1);
lean_inc(x_7);
lean_dec_ref(x_3);
x_8 = lean_ctor_get(x_5, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_5, 2);
lean_inc(x_9);
x_10 = lp_mathlib_Monoid_toMulOneClass___redArg(x_5);
x_11 = !lean_is_exclusive(x_5);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_12 = lean_ctor_get(x_5, 2);
lean_dec(x_12);
x_13 = lean_ctor_get(x_5, 1);
lean_dec(x_13);
x_14 = lean_ctor_get(x_5, 0);
lean_dec(x_14);
x_15 = lean_ctor_get(x_10, 0);
lean_inc(x_15);
lean_dec_ref(x_10);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_16, 0, x_7);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_17, 0, x_9);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_18, 0, x_8);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_19, 0, x_15);
lean_ctor_set(x_5, 2, x_17);
lean_ctor_set(x_5, 1, x_19);
lean_ctor_set(x_5, 0, x_18);
lean_ctor_set(x_1, 1, x_16);
return x_1;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
lean_dec(x_5);
x_20 = lean_ctor_get(x_10, 0);
lean_inc(x_20);
lean_dec_ref(x_10);
x_21 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_21, 0, x_7);
x_22 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_22, 0, x_9);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_23, 0, x_8);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_24, 0, x_20);
x_25 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_25, 0, x_23);
lean_ctor_set(x_25, 1, x_24);
lean_ctor_set(x_25, 2, x_22);
lean_ctor_set(x_1, 1, x_21);
lean_ctor_set(x_1, 0, x_25);
return x_1;
}
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_26 = lean_ctor_get(x_1, 0);
lean_inc(x_26);
lean_dec(x_1);
x_27 = lean_ctor_get(x_3, 1);
lean_inc(x_27);
lean_dec_ref(x_3);
x_28 = lean_ctor_get(x_26, 0);
lean_inc(x_28);
x_29 = lean_ctor_get(x_26, 2);
lean_inc(x_29);
x_30 = lp_mathlib_Monoid_toMulOneClass___redArg(x_26);
if (lean_is_exclusive(x_26)) {
 lean_ctor_release(x_26, 0);
 lean_ctor_release(x_26, 1);
 lean_ctor_release(x_26, 2);
 x_31 = x_26;
} else {
 lean_dec_ref(x_26);
 x_31 = lean_box(0);
}
x_32 = lean_ctor_get(x_30, 0);
lean_inc(x_32);
lean_dec_ref(x_30);
x_33 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_33, 0, x_27);
x_34 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_34, 0, x_29);
x_35 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_35, 0, x_28);
x_36 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_36, 0, x_32);
if (lean_is_scalar(x_31)) {
 x_37 = lean_alloc_ctor(0, 3, 0);
} else {
 x_37 = x_31;
}
lean_ctor_set(x_37, 0, x_35);
lean_ctor_set(x_37, 1, x_36);
lean_ctor_set(x_37, 2, x_34);
x_38 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_33);
return x_38;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMonoidWithZeroOfContinuousMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instMonoidWithZeroOfContinuousMul___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommMonoidOfContinuousMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 2);
lean_inc(x_3);
x_4 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_5 = !lean_is_exclusive(x_1);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lean_ctor_get(x_1, 2);
lean_dec(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_dec(x_7);
x_8 = lean_ctor_get(x_1, 0);
lean_dec(x_8);
x_9 = lean_ctor_get(x_4, 0);
lean_inc(x_9);
lean_dec_ref(x_4);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_10, 0, x_3);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_11, 0, x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_12, 0, x_9);
lean_ctor_set(x_1, 2, x_10);
lean_ctor_set(x_1, 1, x_12);
lean_ctor_set(x_1, 0, x_11);
return x_1;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_dec(x_1);
x_13 = lean_ctor_get(x_4, 0);
lean_inc(x_13);
lean_dec_ref(x_4);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_14, 0, x_3);
x_15 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_15, 0, x_2);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_16, 0, x_13);
x_17 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
lean_ctor_set(x_17, 2, x_14);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommMonoidOfContinuousMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instCommMonoidOfContinuousMul___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommMonoidOfContinuousAdd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
lean_inc(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_5, 0, x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_7, 0, x_1);
x_8 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommMonoidOfContinuousAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instAddCommMonoidOfContinuousAdd___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommMonoidWithZeroOfContinuousMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_2 = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(x_1);
lean_inc_ref(x_2);
x_3 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_2);
x_4 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_3);
x_5 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_2);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_dec_ref(x_4);
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_5, 2);
lean_inc(x_8);
x_9 = lp_mathlib_Monoid_toMulOneClass___redArg(x_5);
x_10 = !lean_is_exclusive(x_5);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_11 = lean_ctor_get(x_5, 2);
lean_dec(x_11);
x_12 = lean_ctor_get(x_5, 1);
lean_dec(x_12);
x_13 = lean_ctor_get(x_5, 0);
lean_dec(x_13);
x_14 = !lean_is_exclusive(x_9);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_15 = lean_ctor_get(x_9, 0);
x_16 = lean_ctor_get(x_9, 1);
lean_dec(x_16);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_17, 0, x_6);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_18, 0, x_8);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_19, 0, x_7);
x_20 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_20, 0, x_15);
lean_ctor_set(x_5, 2, x_18);
lean_ctor_set(x_5, 1, x_20);
lean_ctor_set(x_5, 0, x_19);
lean_ctor_set(x_9, 1, x_17);
lean_ctor_set(x_9, 0, x_5);
return x_9;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_21 = lean_ctor_get(x_9, 0);
lean_inc(x_21);
lean_dec(x_9);
x_22 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_22, 0, x_6);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_23, 0, x_8);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_24, 0, x_7);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_25, 0, x_21);
lean_ctor_set(x_5, 2, x_23);
lean_ctor_set(x_5, 1, x_25);
lean_ctor_set(x_5, 0, x_24);
x_26 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_26, 0, x_5);
lean_ctor_set(x_26, 1, x_22);
return x_26;
}
}
else
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
lean_dec(x_5);
x_27 = lean_ctor_get(x_9, 0);
lean_inc(x_27);
if (lean_is_exclusive(x_9)) {
 lean_ctor_release(x_9, 0);
 lean_ctor_release(x_9, 1);
 x_28 = x_9;
} else {
 lean_dec_ref(x_9);
 x_28 = lean_box(0);
}
x_29 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_29, 0, x_6);
x_30 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_30, 0, x_8);
x_31 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_31, 0, x_7);
x_32 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_32, 0, x_27);
x_33 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_33, 0, x_31);
lean_ctor_set(x_33, 1, x_32);
lean_ctor_set(x_33, 2, x_30);
if (lean_is_scalar(x_28)) {
 x_34 = lean_alloc_ctor(0, 2, 0);
} else {
 x_34 = x_28;
}
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_29);
return x_34;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommMonoidWithZeroOfContinuousMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instCommMonoidWithZeroOfContinuousMul___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnMonoidHom___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnMonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_coeFnMonoidHom___lam__0), 2, 0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnMonoidHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_coeFnMonoidHom(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
return x_7;
}
}
static lean_object* _init_lp_mathlib_ContinuousMap_coeFnAddMonoidHom___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_coeFnMonoidHom___lam__0), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnAddMonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_coeFnAddMonoidHom___closed__0;
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnAddMonoidHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_coeFnAddMonoidHom(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeftContinuous___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeftContinuous___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_ContinuousMap_comp___redArg(x_1, x_2);
x_5 = lean_apply_1(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeftContinuous___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_compLeftContinuous___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_compLeftContinuous___redArg___lam__1), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeftContinuous(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MonoidHom_compLeftContinuous___redArg(x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeftContinuous___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MonoidHom_compLeftContinuous(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_9);
lean_dec_ref(x_6);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeftContinuous___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_compLeftContinuous___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_compLeftContinuous___redArg___lam__1), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeftContinuous(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_AddMonoidHom_compLeftContinuous___redArg(x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeftContinuous___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_AddMonoidHom_compLeftContinuous(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_9);
lean_dec_ref(x_6);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compMonoidHom_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_ContinuousMap_comp___redArg(x_2, x_1);
x_5 = lean_apply_1(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compMonoidHom_x27___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_compMonoidHom_x27___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compMonoidHom_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_compMonoidHom_x27___redArg(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compMonoidHom_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_compMonoidHom_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compAddMonoidHom_x27___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_compMonoidHom_x27___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compAddMonoidHom_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_compAddMonoidHom_x27___redArg(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compAddMonoidHom_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_compAddMonoidHom_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_1(x_3, x_4);
x_6 = lean_apply_2(x_1, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
x_6 = !lean_is_exclusive(x_2);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 2);
x_9 = lean_ctor_get(x_2, 1);
lean_dec(x_9);
x_10 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_1);
x_11 = !lean_is_exclusive(x_1);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_12 = lean_ctor_get(x_1, 3);
lean_dec(x_12);
x_13 = lean_ctor_get(x_1, 2);
lean_dec(x_13);
x_14 = lean_ctor_get(x_1, 1);
lean_dec(x_14);
x_15 = lean_ctor_get(x_1, 0);
lean_dec(x_15);
x_16 = lean_ctor_get(x_10, 0);
lean_inc(x_16);
lean_dec_ref(x_10);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup___redArg___lam__0), 4, 1);
lean_closure_set(x_17, 0, x_5);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_18, 0, x_8);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_19, 0, x_7);
x_20 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_20, 0, x_16);
x_21 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_21, 0, x_3);
x_22 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_22, 0, x_4);
lean_ctor_set(x_2, 2, x_18);
lean_ctor_set(x_2, 1, x_20);
lean_ctor_set(x_2, 0, x_19);
lean_ctor_set(x_1, 3, x_17);
lean_ctor_set(x_1, 2, x_22);
lean_ctor_set(x_1, 1, x_21);
return x_1;
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
lean_dec(x_1);
x_23 = lean_ctor_get(x_10, 0);
lean_inc(x_23);
lean_dec_ref(x_10);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup___redArg___lam__0), 4, 1);
lean_closure_set(x_24, 0, x_5);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_25, 0, x_8);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_26, 0, x_7);
x_27 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_27, 0, x_23);
x_28 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_28, 0, x_3);
x_29 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_29, 0, x_4);
lean_ctor_set(x_2, 2, x_25);
lean_ctor_set(x_2, 1, x_27);
lean_ctor_set(x_2, 0, x_26);
x_30 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_30, 0, x_2);
lean_ctor_set(x_30, 1, x_28);
lean_ctor_set(x_30, 2, x_29);
lean_ctor_set(x_30, 3, x_24);
return x_30;
}
}
else
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; 
x_31 = lean_ctor_get(x_2, 0);
x_32 = lean_ctor_get(x_2, 2);
lean_inc(x_32);
lean_inc(x_31);
lean_dec(x_2);
x_33 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_1);
if (lean_is_exclusive(x_1)) {
 lean_ctor_release(x_1, 0);
 lean_ctor_release(x_1, 1);
 lean_ctor_release(x_1, 2);
 lean_ctor_release(x_1, 3);
 x_34 = x_1;
} else {
 lean_dec_ref(x_1);
 x_34 = lean_box(0);
}
x_35 = lean_ctor_get(x_33, 0);
lean_inc(x_35);
lean_dec_ref(x_33);
x_36 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup___redArg___lam__0), 4, 1);
lean_closure_set(x_36, 0, x_5);
x_37 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_37, 0, x_32);
x_38 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_38, 0, x_31);
x_39 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_39, 0, x_35);
x_40 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_40, 0, x_3);
x_41 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_41, 0, x_4);
x_42 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_42, 0, x_38);
lean_ctor_set(x_42, 1, x_39);
lean_ctor_set(x_42, 2, x_37);
if (lean_is_scalar(x_34)) {
 x_43 = lean_alloc_ctor(0, 4, 0);
} else {
 x_43 = x_34;
}
lean_ctor_set(x_43, 0, x_42);
lean_ctor_set(x_43, 1, x_40);
lean_ctor_set(x_43, 2, x_41);
lean_ctor_set(x_43, 3, x_36);
return x_43;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddGroupOfIsTopologicalAddGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_1, 2);
x_5 = lean_ctor_get(x_2, 0);
x_6 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
lean_inc(x_5);
x_8 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_8, 0, x_5);
x_9 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_9, 0, x_7);
lean_inc_ref(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_10, 0, x_2);
lean_inc(x_3);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_11, 0, x_3);
lean_inc(x_4);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_12, 0, x_4);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_13, 0, x_1);
x_14 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_8, x_9, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddGroupOfIsTopologicalAddGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instAddGroupOfIsTopologicalAddGroup___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommGroupContinuousMap___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
x_6 = !lean_is_exclusive(x_2);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 2);
x_9 = lean_ctor_get(x_2, 1);
lean_dec(x_9);
x_10 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_1);
x_11 = !lean_is_exclusive(x_1);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_12 = lean_ctor_get(x_1, 3);
lean_dec(x_12);
x_13 = lean_ctor_get(x_1, 2);
lean_dec(x_13);
x_14 = lean_ctor_get(x_1, 1);
lean_dec(x_14);
x_15 = lean_ctor_get(x_1, 0);
lean_dec(x_15);
x_16 = lean_ctor_get(x_10, 0);
lean_inc(x_16);
lean_dec_ref(x_10);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup___redArg___lam__0), 4, 1);
lean_closure_set(x_17, 0, x_5);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_18, 0, x_8);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_19, 0, x_7);
x_20 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_20, 0, x_16);
x_21 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_21, 0, x_3);
x_22 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_22, 0, x_4);
lean_ctor_set(x_2, 2, x_18);
lean_ctor_set(x_2, 1, x_20);
lean_ctor_set(x_2, 0, x_19);
lean_ctor_set(x_1, 3, x_17);
lean_ctor_set(x_1, 2, x_22);
lean_ctor_set(x_1, 1, x_21);
return x_1;
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
lean_dec(x_1);
x_23 = lean_ctor_get(x_10, 0);
lean_inc(x_23);
lean_dec_ref(x_10);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup___redArg___lam__0), 4, 1);
lean_closure_set(x_24, 0, x_5);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_25, 0, x_8);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_26, 0, x_7);
x_27 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_27, 0, x_23);
x_28 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_28, 0, x_3);
x_29 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_29, 0, x_4);
lean_ctor_set(x_2, 2, x_25);
lean_ctor_set(x_2, 1, x_27);
lean_ctor_set(x_2, 0, x_26);
x_30 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_30, 0, x_2);
lean_ctor_set(x_30, 1, x_28);
lean_ctor_set(x_30, 2, x_29);
lean_ctor_set(x_30, 3, x_24);
return x_30;
}
}
else
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; 
x_31 = lean_ctor_get(x_2, 0);
x_32 = lean_ctor_get(x_2, 2);
lean_inc(x_32);
lean_inc(x_31);
lean_dec(x_2);
x_33 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_1);
if (lean_is_exclusive(x_1)) {
 lean_ctor_release(x_1, 0);
 lean_ctor_release(x_1, 1);
 lean_ctor_release(x_1, 2);
 lean_ctor_release(x_1, 3);
 x_34 = x_1;
} else {
 lean_dec_ref(x_1);
 x_34 = lean_box(0);
}
x_35 = lean_ctor_get(x_33, 0);
lean_inc(x_35);
lean_dec_ref(x_33);
x_36 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instGroupOfIsTopologicalGroup___redArg___lam__0), 4, 1);
lean_closure_set(x_36, 0, x_5);
x_37 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_37, 0, x_32);
x_38 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_38, 0, x_31);
x_39 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_39, 0, x_35);
x_40 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_40, 0, x_3);
x_41 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_41, 0, x_4);
x_42 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_42, 0, x_38);
lean_ctor_set(x_42, 1, x_39);
lean_ctor_set(x_42, 2, x_37);
if (lean_is_scalar(x_34)) {
 x_43 = lean_alloc_ctor(0, 4, 0);
} else {
 x_43 = x_34;
}
lean_ctor_set(x_43, 0, x_42);
lean_ctor_set(x_43, 1, x_40);
lean_ctor_set(x_43, 2, x_41);
lean_ctor_set(x_43, 3, x_36);
return x_43;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommGroupContinuousMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instCommGroupContinuousMap___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommGroupContinuousMap___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_1, 2);
x_5 = lean_ctor_get(x_2, 0);
x_6 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
lean_inc(x_5);
x_8 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_8, 0, x_5);
x_9 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_9, 0, x_7);
lean_inc_ref(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_10, 0, x_2);
lean_inc(x_3);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_11, 0, x_3);
lean_inc(x_4);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_12, 0, x_4);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_13, 0, x_1);
x_14 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_8, x_9, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddCommGroupContinuousMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instAddCommGroupContinuousMap___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubsemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubsemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_continuousSubsemiring(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_continuousSubring(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalNonAssocSemiringOfIsTopologicalSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
x_5 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = !lean_is_exclusive(x_1);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
lean_dec(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_10, 0, x_4);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_11, 0, x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_12, 0, x_6);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_13, 0, x_8);
x_14 = lp_mathlib_Function_Injective_addMonoid___redArg(x_10, x_12, x_13);
lean_ctor_set(x_1, 1, x_11);
lean_ctor_set(x_1, 0, x_14);
return x_1;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_15 = lean_ctor_get(x_1, 0);
lean_inc(x_15);
lean_dec(x_1);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_16, 0, x_4);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_17, 0, x_3);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_18, 0, x_6);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_19, 0, x_15);
x_20 = lp_mathlib_Function_Injective_addMonoid___redArg(x_16, x_18, x_19);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_17);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalNonAssocSemiringOfIsTopologicalSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instNonUnitalNonAssocSemiringOfIsTopologicalSemiring___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalSemiringOfIsTopologicalSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
x_5 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = !lean_is_exclusive(x_1);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
lean_dec(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_10, 0, x_4);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_11, 0, x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_12, 0, x_6);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_13, 0, x_8);
x_14 = lp_mathlib_Function_Injective_addMonoid___redArg(x_10, x_12, x_13);
lean_ctor_set(x_1, 1, x_11);
lean_ctor_set(x_1, 0, x_14);
return x_1;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_15 = lean_ctor_get(x_1, 0);
lean_inc(x_15);
lean_dec(x_1);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_16, 0, x_4);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_17, 0, x_3);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_18, 0, x_6);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_19, 0, x_15);
x_20 = lp_mathlib_Function_Injective_addMonoid___redArg(x_16, x_18, x_19);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_17);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalSemiringOfIsTopologicalSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instNonUnitalSemiringOfIsTopologicalSemiring___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddMonoidWithOneOfContinuousAdd___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
x_6 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_4);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_ctor_get(x_4, 0);
x_9 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_9, 0, x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_10, 0, x_5);
lean_inc(x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_11, 0, x_8);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_12, 0, x_4);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_13, 0, x_3);
x_14 = lp_mathlib_Function_Injective_addMonoid___redArg(x_11, x_9, x_12);
x_15 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_15, 0, lean_box(0));
lean_closure_set(x_15, 1, x_13);
lean_ctor_set(x_1, 2, x_10);
lean_ctor_set(x_1, 1, x_14);
lean_ctor_set(x_1, 0, x_15);
return x_1;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_16 = lean_ctor_get(x_1, 0);
x_17 = lean_ctor_get(x_1, 1);
x_18 = lean_ctor_get(x_1, 2);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_1);
x_19 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_17);
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lean_ctor_get(x_17, 0);
x_22 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_22, 0, x_20);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_23, 0, x_18);
lean_inc(x_21);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_24, 0, x_21);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_25, 0, x_17);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_26, 0, x_16);
x_27 = lp_mathlib_Function_Injective_addMonoid___redArg(x_24, x_22, x_25);
x_28 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_28, 0, lean_box(0));
lean_closure_set(x_28, 1, x_26);
x_29 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_27);
lean_ctor_set(x_29, 2, x_23);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instAddMonoidWithOneOfContinuousAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instAddMonoidWithOneOfContinuousAdd___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonAssocSemiringOfIsTopologicalSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_inc_ref(x_2);
x_3 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
lean_inc_ref(x_2);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_2);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
lean_inc_ref(x_1);
x_8 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_1);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = !lean_is_exclusive(x_2);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_12 = lean_ctor_get(x_2, 0);
x_13 = lean_ctor_get(x_2, 1);
lean_dec(x_13);
x_14 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_1);
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_16 = lean_ctor_get(x_14, 0);
x_17 = lean_ctor_get(x_14, 2);
lean_dec(x_17);
x_18 = lean_ctor_get(x_14, 1);
lean_dec(x_18);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_19, 0, x_5);
x_20 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_20, 0, x_4);
x_21 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_21, 0, x_7);
x_22 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_22, 0, x_10);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_23, 0, x_12);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_24, 0, x_16);
x_25 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, x_24);
x_26 = lp_mathlib_Function_Injective_addMonoid___redArg(x_19, x_21, x_23);
lean_ctor_set(x_2, 1, x_20);
lean_ctor_set(x_2, 0, x_26);
lean_ctor_set(x_14, 2, x_25);
lean_ctor_set(x_14, 1, x_22);
lean_ctor_set(x_14, 0, x_2);
return x_14;
}
else
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_27 = lean_ctor_get(x_14, 0);
lean_inc(x_27);
lean_dec(x_14);
x_28 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_28, 0, x_5);
x_29 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_29, 0, x_4);
x_30 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_30, 0, x_7);
x_31 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_31, 0, x_10);
x_32 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_32, 0, x_12);
x_33 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_33, 0, x_27);
x_34 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_34, 0, lean_box(0));
lean_closure_set(x_34, 1, x_33);
x_35 = lp_mathlib_Function_Injective_addMonoid___redArg(x_28, x_30, x_32);
lean_ctor_set(x_2, 1, x_29);
lean_ctor_set(x_2, 0, x_35);
x_36 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_36, 0, x_2);
lean_ctor_set(x_36, 1, x_31);
lean_ctor_set(x_36, 2, x_34);
return x_36;
}
}
else
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; 
x_37 = lean_ctor_get(x_2, 0);
lean_inc(x_37);
lean_dec(x_2);
x_38 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_1);
x_39 = lean_ctor_get(x_38, 0);
lean_inc(x_39);
if (lean_is_exclusive(x_38)) {
 lean_ctor_release(x_38, 0);
 lean_ctor_release(x_38, 1);
 lean_ctor_release(x_38, 2);
 x_40 = x_38;
} else {
 lean_dec_ref(x_38);
 x_40 = lean_box(0);
}
x_41 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_41, 0, x_5);
x_42 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_42, 0, x_4);
x_43 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_43, 0, x_7);
x_44 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_44, 0, x_10);
x_45 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_45, 0, x_37);
x_46 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_46, 0, x_39);
x_47 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_47, 0, lean_box(0));
lean_closure_set(x_47, 1, x_46);
x_48 = lp_mathlib_Function_Injective_addMonoid___redArg(x_41, x_43, x_45);
x_49 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_49, 0, x_48);
lean_ctor_set(x_49, 1, x_42);
if (lean_is_scalar(x_40)) {
 x_50 = lean_alloc_ctor(0, 3, 0);
} else {
 x_50 = x_40;
}
lean_ctor_set(x_50, 0, x_49);
lean_ctor_set(x_50, 1, x_44);
lean_ctor_set(x_50, 2, x_47);
return x_50;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonAssocSemiringOfIsTopologicalSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instNonAssocSemiringOfIsTopologicalSemiring___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemiringOfIsTopologicalSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_dec_ref(x_4);
x_7 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_8 = !lean_is_exclusive(x_1);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_9 = lean_ctor_get(x_1, 3);
lean_dec(x_9);
x_10 = lean_ctor_get(x_1, 2);
lean_dec(x_10);
x_11 = lean_ctor_get(x_1, 1);
lean_dec(x_11);
x_12 = lean_ctor_get(x_1, 0);
lean_dec(x_12);
x_13 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_13);
x_14 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_13);
x_15 = lean_ctor_get(x_14, 1);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc_ref(x_7);
x_16 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_7);
x_17 = lean_ctor_get(x_16, 0);
lean_inc_ref(x_17);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = !lean_is_exclusive(x_2);
if (x_19 == 0)
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_20 = lean_ctor_get(x_2, 0);
x_21 = lean_ctor_get(x_2, 1);
lean_dec(x_21);
x_22 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_7);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec_ref(x_22);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_24, 0, x_3);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_25, 0, x_6);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_26, 0, x_5);
x_27 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_27, 0, x_15);
x_28 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_28, 0, x_18);
x_29 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_29, 0, x_20);
x_30 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_30, 0, x_23);
x_31 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_31, 0, lean_box(0));
lean_closure_set(x_31, 1, x_30);
x_32 = lp_mathlib_Function_Injective_addMonoid___redArg(x_25, x_27, x_29);
lean_ctor_set(x_2, 1, x_26);
lean_ctor_set(x_2, 0, x_32);
lean_ctor_set(x_1, 3, x_24);
lean_ctor_set(x_1, 2, x_31);
lean_ctor_set(x_1, 1, x_28);
return x_1;
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; 
x_33 = lean_ctor_get(x_2, 0);
lean_inc(x_33);
lean_dec(x_2);
x_34 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_7);
x_35 = lean_ctor_get(x_34, 0);
lean_inc(x_35);
lean_dec_ref(x_34);
x_36 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_36, 0, x_3);
x_37 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_37, 0, x_6);
x_38 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_38, 0, x_5);
x_39 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_39, 0, x_15);
x_40 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_40, 0, x_18);
x_41 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_41, 0, x_33);
x_42 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_42, 0, x_35);
x_43 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_43, 0, lean_box(0));
lean_closure_set(x_43, 1, x_42);
x_44 = lp_mathlib_Function_Injective_addMonoid___redArg(x_37, x_39, x_41);
x_45 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_45, 0, x_44);
lean_ctor_set(x_45, 1, x_38);
lean_ctor_set(x_1, 3, x_36);
lean_ctor_set(x_1, 2, x_43);
lean_ctor_set(x_1, 1, x_40);
lean_ctor_set(x_1, 0, x_45);
return x_1;
}
}
else
{
lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; 
lean_dec(x_1);
x_46 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_46);
x_47 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_46);
x_48 = lean_ctor_get(x_47, 1);
lean_inc(x_48);
lean_dec_ref(x_47);
lean_inc_ref(x_7);
x_49 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_7);
x_50 = lean_ctor_get(x_49, 0);
lean_inc_ref(x_50);
lean_dec_ref(x_49);
x_51 = lean_ctor_get(x_50, 0);
lean_inc(x_51);
lean_dec_ref(x_50);
x_52 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_52);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 x_53 = x_2;
} else {
 lean_dec_ref(x_2);
 x_53 = lean_box(0);
}
x_54 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_7);
x_55 = lean_ctor_get(x_54, 0);
lean_inc(x_55);
lean_dec_ref(x_54);
x_56 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_56, 0, x_3);
x_57 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_57, 0, x_6);
x_58 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_58, 0, x_5);
x_59 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_59, 0, x_48);
x_60 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_60, 0, x_51);
x_61 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_61, 0, x_52);
x_62 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_62, 0, x_55);
x_63 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_63, 0, lean_box(0));
lean_closure_set(x_63, 1, x_62);
x_64 = lp_mathlib_Function_Injective_addMonoid___redArg(x_57, x_59, x_61);
if (lean_is_scalar(x_53)) {
 x_65 = lean_alloc_ctor(0, 2, 0);
} else {
 x_65 = x_53;
}
lean_ctor_set(x_65, 0, x_64);
lean_ctor_set(x_65, 1, x_58);
x_66 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_66, 0, x_65);
lean_ctor_set(x_66, 1, x_60);
lean_ctor_set(x_66, 2, x_63);
lean_ctor_set(x_66, 3, x_56);
return x_66;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSemiringOfIsTopologicalSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instSemiringOfIsTopologicalSemiring___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalNonAssocRingOfIsTopologicalRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = lean_ctor_get(x_2, 2);
x_6 = lean_ctor_get(x_3, 0);
x_7 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_1);
lean_inc_ref(x_7);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_7);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_12 = lean_ctor_get(x_10, 1);
x_13 = lean_ctor_get(x_10, 0);
lean_dec(x_13);
lean_inc(x_6);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_14, 0, x_6);
x_15 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_15, 0, x_9);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_16, 0, x_12);
lean_inc(x_4);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_17, 0, x_4);
lean_inc(x_5);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_18, 0, x_5);
lean_inc_ref(x_3);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_19, 0, x_3);
x_20 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_20, 0, x_2);
x_21 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_14, x_16, x_19, x_17, x_18, x_20);
lean_ctor_set(x_10, 1, x_15);
lean_ctor_set(x_10, 0, x_21);
return x_10;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_22 = lean_ctor_get(x_10, 1);
lean_inc(x_22);
lean_dec(x_10);
lean_inc(x_6);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_23, 0, x_6);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_24, 0, x_9);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_25, 0, x_22);
lean_inc(x_4);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_26, 0, x_4);
lean_inc(x_5);
x_27 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_27, 0, x_5);
lean_inc_ref(x_3);
x_28 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_28, 0, x_3);
x_29 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_29, 0, x_2);
x_30 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_23, x_25, x_28, x_26, x_27, x_29);
x_31 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_31, 0, x_30);
lean_ctor_set(x_31, 1, x_24);
return x_31;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalNonAssocRingOfIsTopologicalRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instNonUnitalNonAssocRingOfIsTopologicalRing___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalRingOfIsTopologicalRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = lean_ctor_get(x_2, 2);
x_6 = lean_ctor_get(x_3, 0);
x_7 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_1);
lean_inc_ref(x_7);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_7);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_12 = lean_ctor_get(x_10, 1);
x_13 = lean_ctor_get(x_10, 0);
lean_dec(x_13);
lean_inc(x_6);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_14, 0, x_6);
x_15 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_15, 0, x_9);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_16, 0, x_12);
lean_inc(x_4);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_17, 0, x_4);
lean_inc(x_5);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_18, 0, x_5);
lean_inc_ref(x_3);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_19, 0, x_3);
x_20 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_20, 0, x_2);
x_21 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_14, x_16, x_19, x_17, x_18, x_20);
lean_ctor_set(x_10, 1, x_15);
lean_ctor_set(x_10, 0, x_21);
return x_10;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_22 = lean_ctor_get(x_10, 1);
lean_inc(x_22);
lean_dec(x_10);
lean_inc(x_6);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_23, 0, x_6);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_24, 0, x_9);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_25, 0, x_22);
lean_inc(x_4);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_26, 0, x_4);
lean_inc(x_5);
x_27 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_27, 0, x_5);
lean_inc_ref(x_3);
x_28 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_28, 0, x_3);
x_29 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_29, 0, x_2);
x_30 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_23, x_25, x_28, x_26, x_27, x_29);
x_31 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_31, 0, x_30);
lean_ctor_set(x_31, 1, x_24);
return x_31;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalRingOfIsTopologicalRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instNonUnitalRingOfIsTopologicalRing___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonAssocRingOfIsTopologicalRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_3, 0);
x_5 = lean_ctor_get(x_3, 1);
x_6 = lean_ctor_get(x_3, 2);
x_7 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_2);
x_8 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_2);
lean_inc_ref(x_8);
x_9 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_8);
x_12 = lean_ctor_get(x_11, 1);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_1);
x_13 = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(x_1);
lean_inc_ref(x_13);
x_14 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_13);
x_15 = lean_ctor_get(x_14, 0);
lean_inc_ref(x_15);
lean_dec_ref(x_14);
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_17 = lean_ctor_get(x_15, 0);
x_18 = lean_ctor_get(x_15, 1);
lean_dec(x_18);
x_19 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_13);
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(x_1);
x_22 = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(x_21);
lean_dec_ref(x_21);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec_ref(x_22);
lean_inc(x_7);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_24, 0, x_7);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_25, 0, x_10);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_26, 0, x_12);
x_27 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_27, 0, x_17);
lean_inc(x_5);
x_28 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_28, 0, x_5);
lean_inc(x_6);
x_29 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_29, 0, x_6);
lean_inc_ref(x_4);
x_30 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_30, 0, x_4);
x_31 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_31, 0, x_3);
x_32 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_32, 0, x_20);
x_33 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_33, 0, x_23);
x_34 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_34, 0, lean_box(0));
lean_closure_set(x_34, 1, x_32);
x_35 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_35, 0, lean_box(0));
lean_closure_set(x_35, 1, x_33);
x_36 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_24, x_26, x_30, x_28, x_29, x_31);
lean_ctor_set(x_15, 1, x_25);
lean_ctor_set(x_15, 0, x_36);
x_37 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_37, 0, x_15);
lean_ctor_set(x_37, 1, x_27);
lean_ctor_set(x_37, 2, x_34);
lean_ctor_set(x_37, 3, x_35);
return x_37;
}
else
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; 
x_38 = lean_ctor_get(x_15, 0);
lean_inc(x_38);
lean_dec(x_15);
x_39 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_13);
x_40 = lean_ctor_get(x_39, 0);
lean_inc(x_40);
lean_dec_ref(x_39);
x_41 = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(x_1);
x_42 = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(x_41);
lean_dec_ref(x_41);
x_43 = lean_ctor_get(x_42, 0);
lean_inc(x_43);
lean_dec_ref(x_42);
lean_inc(x_7);
x_44 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_44, 0, x_7);
x_45 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_45, 0, x_10);
x_46 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_46, 0, x_12);
x_47 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_47, 0, x_38);
lean_inc(x_5);
x_48 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_48, 0, x_5);
lean_inc(x_6);
x_49 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_49, 0, x_6);
lean_inc_ref(x_4);
x_50 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_50, 0, x_4);
x_51 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_51, 0, x_3);
x_52 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_52, 0, x_40);
x_53 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_53, 0, x_43);
x_54 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_54, 0, lean_box(0));
lean_closure_set(x_54, 1, x_52);
x_55 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_55, 0, lean_box(0));
lean_closure_set(x_55, 1, x_53);
x_56 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_44, x_46, x_50, x_48, x_49, x_51);
x_57 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_57, 0, x_56);
lean_ctor_set(x_57, 1, x_45);
x_58 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_58, 0, x_57);
lean_ctor_set(x_58, 1, x_47);
lean_ctor_set(x_58, 2, x_54);
lean_ctor_set(x_58, 3, x_55);
return x_58;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonAssocRingOfIsTopologicalRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instNonAssocRingOfIsTopologicalRing___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instRing___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_1(x_3, x_4);
x_6 = lean_apply_2(x_1, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_2, 3);
lean_inc(x_5);
lean_inc_ref(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_4);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
lean_inc_ref(x_1);
x_9 = lp_mathlib_Ring_toNonAssocRing___redArg(x_1);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_10);
x_12 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_11);
x_13 = lean_ctor_get(x_12, 1);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_15 = !lean_is_exclusive(x_2);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; 
x_16 = lean_ctor_get(x_2, 3);
lean_dec(x_16);
x_17 = lean_ctor_get(x_2, 2);
lean_dec(x_17);
x_18 = lean_ctor_get(x_2, 1);
lean_dec(x_18);
x_19 = lean_ctor_get(x_2, 0);
lean_dec(x_19);
lean_inc_ref(x_14);
x_20 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_14);
x_21 = lean_ctor_get(x_20, 0);
lean_inc_ref(x_21);
lean_dec_ref(x_20);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
lean_dec_ref(x_21);
lean_inc_ref(x_1);
x_23 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_24 = lean_ctor_get(x_23, 1);
lean_inc(x_24);
x_25 = lean_ctor_get(x_23, 2);
lean_inc(x_25);
lean_dec_ref(x_23);
x_26 = !lean_is_exclusive(x_4);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_27 = lean_ctor_get(x_4, 0);
x_28 = lean_ctor_get(x_4, 1);
lean_dec(x_28);
x_29 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_30 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_29);
x_31 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_14);
x_32 = lean_ctor_get(x_31, 0);
lean_inc(x_32);
lean_dec_ref(x_31);
x_33 = !lean_is_exclusive(x_29);
if (x_33 == 0)
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; 
x_34 = lean_ctor_get(x_29, 0);
x_35 = lean_ctor_get(x_29, 4);
lean_dec(x_35);
x_36 = lean_ctor_get(x_29, 3);
lean_dec(x_36);
x_37 = lean_ctor_get(x_29, 2);
lean_dec(x_37);
x_38 = lean_ctor_get(x_29, 1);
lean_dec(x_38);
x_39 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_39, 0, x_8);
x_40 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_40, 0, x_13);
x_41 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_41, 0, x_24);
x_42 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_42, 0, x_25);
x_43 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_43, 0, x_27);
x_44 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_44, 0, x_30);
lean_inc_ref(x_43);
lean_inc_ref(x_40);
lean_inc_ref(x_39);
x_45 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_39, x_40, x_43, x_41, x_42, x_44);
x_46 = lean_ctor_get(x_45, 1);
lean_inc(x_46);
x_47 = lean_ctor_get(x_45, 2);
lean_inc(x_47);
lean_dec_ref(x_45);
x_48 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instRing___redArg___lam__0), 4, 1);
lean_closure_set(x_48, 0, x_3);
x_49 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_49, 0, x_5);
x_50 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_50, 0, x_7);
x_51 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_51, 0, x_22);
x_52 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_52, 0, x_32);
x_53 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_53, 0, x_34);
x_54 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_54, 0, lean_box(0));
lean_closure_set(x_54, 1, x_53);
x_55 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_55, 0, lean_box(0));
lean_closure_set(x_55, 1, x_52);
x_56 = lp_mathlib_Function_Injective_addMonoid___redArg(x_39, x_40, x_43);
lean_ctor_set(x_4, 1, x_50);
lean_ctor_set(x_4, 0, x_56);
lean_ctor_set(x_2, 3, x_49);
lean_ctor_set(x_2, 2, x_55);
lean_ctor_set(x_2, 1, x_51);
lean_ctor_set(x_29, 4, x_54);
lean_ctor_set(x_29, 3, x_48);
lean_ctor_set(x_29, 2, x_47);
lean_ctor_set(x_29, 1, x_46);
lean_ctor_set(x_29, 0, x_2);
return x_29;
}
else
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; 
x_57 = lean_ctor_get(x_29, 0);
lean_inc(x_57);
lean_dec(x_29);
x_58 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_58, 0, x_8);
x_59 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_59, 0, x_13);
x_60 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_60, 0, x_24);
x_61 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_61, 0, x_25);
x_62 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_62, 0, x_27);
x_63 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_63, 0, x_30);
lean_inc_ref(x_62);
lean_inc_ref(x_59);
lean_inc_ref(x_58);
x_64 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_58, x_59, x_62, x_60, x_61, x_63);
x_65 = lean_ctor_get(x_64, 1);
lean_inc(x_65);
x_66 = lean_ctor_get(x_64, 2);
lean_inc(x_66);
lean_dec_ref(x_64);
x_67 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instRing___redArg___lam__0), 4, 1);
lean_closure_set(x_67, 0, x_3);
x_68 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_68, 0, x_5);
x_69 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_69, 0, x_7);
x_70 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_70, 0, x_22);
x_71 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_71, 0, x_32);
x_72 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_72, 0, x_57);
x_73 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_73, 0, lean_box(0));
lean_closure_set(x_73, 1, x_72);
x_74 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_74, 0, lean_box(0));
lean_closure_set(x_74, 1, x_71);
x_75 = lp_mathlib_Function_Injective_addMonoid___redArg(x_58, x_59, x_62);
lean_ctor_set(x_4, 1, x_69);
lean_ctor_set(x_4, 0, x_75);
lean_ctor_set(x_2, 3, x_68);
lean_ctor_set(x_2, 2, x_74);
lean_ctor_set(x_2, 1, x_70);
x_76 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_76, 0, x_2);
lean_ctor_set(x_76, 1, x_65);
lean_ctor_set(x_76, 2, x_66);
lean_ctor_set(x_76, 3, x_67);
lean_ctor_set(x_76, 4, x_73);
return x_76;
}
}
else
{
lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; 
x_77 = lean_ctor_get(x_4, 0);
lean_inc(x_77);
lean_dec(x_4);
x_78 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_79 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_78);
x_80 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_14);
x_81 = lean_ctor_get(x_80, 0);
lean_inc(x_81);
lean_dec_ref(x_80);
x_82 = lean_ctor_get(x_78, 0);
lean_inc(x_82);
if (lean_is_exclusive(x_78)) {
 lean_ctor_release(x_78, 0);
 lean_ctor_release(x_78, 1);
 lean_ctor_release(x_78, 2);
 lean_ctor_release(x_78, 3);
 lean_ctor_release(x_78, 4);
 x_83 = x_78;
} else {
 lean_dec_ref(x_78);
 x_83 = lean_box(0);
}
x_84 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_84, 0, x_8);
x_85 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_85, 0, x_13);
x_86 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_86, 0, x_24);
x_87 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_87, 0, x_25);
x_88 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_88, 0, x_77);
x_89 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_89, 0, x_79);
lean_inc_ref(x_88);
lean_inc_ref(x_85);
lean_inc_ref(x_84);
x_90 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_84, x_85, x_88, x_86, x_87, x_89);
x_91 = lean_ctor_get(x_90, 1);
lean_inc(x_91);
x_92 = lean_ctor_get(x_90, 2);
lean_inc(x_92);
lean_dec_ref(x_90);
x_93 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instRing___redArg___lam__0), 4, 1);
lean_closure_set(x_93, 0, x_3);
x_94 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_94, 0, x_5);
x_95 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_95, 0, x_7);
x_96 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_96, 0, x_22);
x_97 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_97, 0, x_81);
x_98 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_98, 0, x_82);
x_99 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_99, 0, lean_box(0));
lean_closure_set(x_99, 1, x_98);
x_100 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_100, 0, lean_box(0));
lean_closure_set(x_100, 1, x_97);
x_101 = lp_mathlib_Function_Injective_addMonoid___redArg(x_84, x_85, x_88);
x_102 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_102, 0, x_101);
lean_ctor_set(x_102, 1, x_95);
lean_ctor_set(x_2, 3, x_94);
lean_ctor_set(x_2, 2, x_100);
lean_ctor_set(x_2, 1, x_96);
lean_ctor_set(x_2, 0, x_102);
if (lean_is_scalar(x_83)) {
 x_103 = lean_alloc_ctor(0, 5, 0);
} else {
 x_103 = x_83;
}
lean_ctor_set(x_103, 0, x_2);
lean_ctor_set(x_103, 1, x_91);
lean_ctor_set(x_103, 2, x_92);
lean_ctor_set(x_103, 3, x_93);
lean_ctor_set(x_103, 4, x_99);
return x_103;
}
}
else
{
lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; 
lean_dec(x_2);
lean_inc_ref(x_14);
x_104 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_14);
x_105 = lean_ctor_get(x_104, 0);
lean_inc_ref(x_105);
lean_dec_ref(x_104);
x_106 = lean_ctor_get(x_105, 0);
lean_inc(x_106);
lean_dec_ref(x_105);
lean_inc_ref(x_1);
x_107 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_108 = lean_ctor_get(x_107, 1);
lean_inc(x_108);
x_109 = lean_ctor_get(x_107, 2);
lean_inc(x_109);
lean_dec_ref(x_107);
x_110 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_110);
if (lean_is_exclusive(x_4)) {
 lean_ctor_release(x_4, 0);
 lean_ctor_release(x_4, 1);
 x_111 = x_4;
} else {
 lean_dec_ref(x_4);
 x_111 = lean_box(0);
}
x_112 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_113 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_112);
x_114 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_14);
x_115 = lean_ctor_get(x_114, 0);
lean_inc(x_115);
lean_dec_ref(x_114);
x_116 = lean_ctor_get(x_112, 0);
lean_inc(x_116);
if (lean_is_exclusive(x_112)) {
 lean_ctor_release(x_112, 0);
 lean_ctor_release(x_112, 1);
 lean_ctor_release(x_112, 2);
 lean_ctor_release(x_112, 3);
 lean_ctor_release(x_112, 4);
 x_117 = x_112;
} else {
 lean_dec_ref(x_112);
 x_117 = lean_box(0);
}
x_118 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_118, 0, x_8);
x_119 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_119, 0, x_13);
x_120 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_120, 0, x_108);
x_121 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_121, 0, x_109);
x_122 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_122, 0, x_110);
x_123 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_123, 0, x_113);
lean_inc_ref(x_122);
lean_inc_ref(x_119);
lean_inc_ref(x_118);
x_124 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_118, x_119, x_122, x_120, x_121, x_123);
x_125 = lean_ctor_get(x_124, 1);
lean_inc(x_125);
x_126 = lean_ctor_get(x_124, 2);
lean_inc(x_126);
lean_dec_ref(x_124);
x_127 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instRing___redArg___lam__0), 4, 1);
lean_closure_set(x_127, 0, x_3);
x_128 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_128, 0, x_5);
x_129 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_129, 0, x_7);
x_130 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_130, 0, x_106);
x_131 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_131, 0, x_115);
x_132 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_132, 0, x_116);
x_133 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_133, 0, lean_box(0));
lean_closure_set(x_133, 1, x_132);
x_134 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_134, 0, lean_box(0));
lean_closure_set(x_134, 1, x_131);
x_135 = lp_mathlib_Function_Injective_addMonoid___redArg(x_118, x_119, x_122);
if (lean_is_scalar(x_111)) {
 x_136 = lean_alloc_ctor(0, 2, 0);
} else {
 x_136 = x_111;
}
lean_ctor_set(x_136, 0, x_135);
lean_ctor_set(x_136, 1, x_129);
x_137 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_137, 0, x_136);
lean_ctor_set(x_137, 1, x_130);
lean_ctor_set(x_137, 2, x_134);
lean_ctor_set(x_137, 3, x_128);
if (lean_is_scalar(x_117)) {
 x_138 = lean_alloc_ctor(0, 5, 0);
} else {
 x_138 = x_117;
}
lean_ctor_set(x_138, 0, x_137);
lean_ctor_set(x_138, 1, x_125);
lean_ctor_set(x_138, 2, x_126);
lean_ctor_set(x_138, 3, x_127);
lean_ctor_set(x_138, 4, x_133);
return x_138;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instRing___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalCommSemiringOfIsTopologicalSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
x_5 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = !lean_is_exclusive(x_1);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
lean_dec(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_10, 0, x_4);
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_11, 0, x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_12, 0, x_6);
x_13 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_13, 0, x_8);
x_14 = lp_mathlib_Function_Injective_addMonoid___redArg(x_10, x_12, x_13);
lean_ctor_set(x_1, 1, x_11);
lean_ctor_set(x_1, 0, x_14);
return x_1;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_15 = lean_ctor_get(x_1, 0);
lean_inc(x_15);
lean_dec(x_1);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_16, 0, x_4);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_17, 0, x_3);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_18, 0, x_6);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_19, 0, x_15);
x_20 = lp_mathlib_Function_Injective_addMonoid___redArg(x_16, x_18, x_19);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_17);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalCommSemiringOfIsTopologicalSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instNonUnitalCommSemiringOfIsTopologicalSemiring___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommSemiringOfIsTopologicalSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_dec_ref(x_4);
x_7 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_8 = !lean_is_exclusive(x_1);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_9 = lean_ctor_get(x_1, 3);
lean_dec(x_9);
x_10 = lean_ctor_get(x_1, 2);
lean_dec(x_10);
x_11 = lean_ctor_get(x_1, 1);
lean_dec(x_11);
x_12 = lean_ctor_get(x_1, 0);
lean_dec(x_12);
x_13 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_13);
x_14 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_13);
x_15 = lean_ctor_get(x_14, 1);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc_ref(x_7);
x_16 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_7);
x_17 = lean_ctor_get(x_16, 0);
lean_inc_ref(x_17);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = !lean_is_exclusive(x_2);
if (x_19 == 0)
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_20 = lean_ctor_get(x_2, 0);
x_21 = lean_ctor_get(x_2, 1);
lean_dec(x_21);
x_22 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_7);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec_ref(x_22);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_24, 0, x_3);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_25, 0, x_6);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_26, 0, x_5);
x_27 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_27, 0, x_15);
x_28 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_28, 0, x_18);
x_29 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_29, 0, x_20);
x_30 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_30, 0, x_23);
x_31 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_31, 0, lean_box(0));
lean_closure_set(x_31, 1, x_30);
x_32 = lp_mathlib_Function_Injective_addMonoid___redArg(x_25, x_27, x_29);
lean_ctor_set(x_2, 1, x_26);
lean_ctor_set(x_2, 0, x_32);
lean_ctor_set(x_1, 3, x_24);
lean_ctor_set(x_1, 2, x_31);
lean_ctor_set(x_1, 1, x_28);
return x_1;
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; 
x_33 = lean_ctor_get(x_2, 0);
lean_inc(x_33);
lean_dec(x_2);
x_34 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_7);
x_35 = lean_ctor_get(x_34, 0);
lean_inc(x_35);
lean_dec_ref(x_34);
x_36 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_36, 0, x_3);
x_37 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_37, 0, x_6);
x_38 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_38, 0, x_5);
x_39 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_39, 0, x_15);
x_40 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_40, 0, x_18);
x_41 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_41, 0, x_33);
x_42 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_42, 0, x_35);
x_43 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_43, 0, lean_box(0));
lean_closure_set(x_43, 1, x_42);
x_44 = lp_mathlib_Function_Injective_addMonoid___redArg(x_37, x_39, x_41);
x_45 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_45, 0, x_44);
lean_ctor_set(x_45, 1, x_38);
lean_ctor_set(x_1, 3, x_36);
lean_ctor_set(x_1, 2, x_43);
lean_ctor_set(x_1, 1, x_40);
lean_ctor_set(x_1, 0, x_45);
return x_1;
}
}
else
{
lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; 
lean_dec(x_1);
x_46 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_46);
x_47 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_46);
x_48 = lean_ctor_get(x_47, 1);
lean_inc(x_48);
lean_dec_ref(x_47);
lean_inc_ref(x_7);
x_49 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_7);
x_50 = lean_ctor_get(x_49, 0);
lean_inc_ref(x_50);
lean_dec_ref(x_49);
x_51 = lean_ctor_get(x_50, 0);
lean_inc(x_51);
lean_dec_ref(x_50);
x_52 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_52);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 x_53 = x_2;
} else {
 lean_dec_ref(x_2);
 x_53 = lean_box(0);
}
x_54 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_7);
x_55 = lean_ctor_get(x_54, 0);
lean_inc(x_55);
lean_dec_ref(x_54);
x_56 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_56, 0, x_3);
x_57 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_57, 0, x_6);
x_58 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_58, 0, x_5);
x_59 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_59, 0, x_48);
x_60 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_60, 0, x_51);
x_61 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_61, 0, x_52);
x_62 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_62, 0, x_55);
x_63 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_63, 0, lean_box(0));
lean_closure_set(x_63, 1, x_62);
x_64 = lp_mathlib_Function_Injective_addMonoid___redArg(x_57, x_59, x_61);
if (lean_is_scalar(x_53)) {
 x_65 = lean_alloc_ctor(0, 2, 0);
} else {
 x_65 = x_53;
}
lean_ctor_set(x_65, 0, x_64);
lean_ctor_set(x_65, 1, x_58);
x_66 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_66, 0, x_65);
lean_ctor_set(x_66, 1, x_60);
lean_ctor_set(x_66, 2, x_63);
lean_ctor_set(x_66, 3, x_56);
return x_66;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommSemiringOfIsTopologicalSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instCommSemiringOfIsTopologicalSemiring___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalCommRingOfIsTopologicalRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = lean_ctor_get(x_2, 2);
x_6 = lean_ctor_get(x_3, 0);
x_7 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_1);
lean_inc_ref(x_7);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_7);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_12 = lean_ctor_get(x_10, 1);
x_13 = lean_ctor_get(x_10, 0);
lean_dec(x_13);
lean_inc(x_6);
x_14 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_14, 0, x_6);
x_15 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_15, 0, x_9);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_16, 0, x_12);
lean_inc(x_4);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_17, 0, x_4);
lean_inc(x_5);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_18, 0, x_5);
lean_inc_ref(x_3);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_19, 0, x_3);
x_20 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_20, 0, x_2);
x_21 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_14, x_16, x_19, x_17, x_18, x_20);
lean_ctor_set(x_10, 1, x_15);
lean_ctor_set(x_10, 0, x_21);
return x_10;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_22 = lean_ctor_get(x_10, 1);
lean_inc(x_22);
lean_dec(x_10);
lean_inc(x_6);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_23, 0, x_6);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_24, 0, x_9);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_25, 0, x_22);
lean_inc(x_4);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_26, 0, x_4);
lean_inc(x_5);
x_27 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_27, 0, x_5);
lean_inc_ref(x_3);
x_28 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_28, 0, x_3);
x_29 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_29, 0, x_2);
x_30 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_23, x_25, x_28, x_26, x_27, x_29);
x_31 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_31, 0, x_30);
lean_ctor_set(x_31, 1, x_24);
return x_31;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instNonUnitalCommRingOfIsTopologicalRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instNonUnitalCommRingOfIsTopologicalRing___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommRingOfIsTopologicalRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_2, 3);
lean_inc(x_5);
lean_inc_ref(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_4);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
lean_inc_ref(x_1);
x_9 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_1);
x_10 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_9);
x_11 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_10);
x_12 = lean_ctor_get(x_11, 1);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_14 = !lean_is_exclusive(x_2);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; 
x_15 = lean_ctor_get(x_2, 3);
lean_dec(x_15);
x_16 = lean_ctor_get(x_2, 2);
lean_dec(x_16);
x_17 = lean_ctor_get(x_2, 1);
lean_dec(x_17);
x_18 = lean_ctor_get(x_2, 0);
lean_dec(x_18);
lean_inc_ref(x_13);
x_19 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_13);
x_20 = lean_ctor_get(x_19, 0);
lean_inc_ref(x_20);
lean_dec_ref(x_19);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
lean_inc_ref(x_1);
x_22 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_23 = lean_ctor_get(x_22, 1);
lean_inc(x_23);
x_24 = lean_ctor_get(x_22, 2);
lean_inc(x_24);
lean_dec_ref(x_22);
x_25 = !lean_is_exclusive(x_4);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_26 = lean_ctor_get(x_4, 0);
x_27 = lean_ctor_get(x_4, 1);
lean_dec(x_27);
x_28 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_29 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_28);
x_30 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_13);
x_31 = lean_ctor_get(x_30, 0);
lean_inc(x_31);
lean_dec_ref(x_30);
x_32 = !lean_is_exclusive(x_28);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; 
x_33 = lean_ctor_get(x_28, 0);
x_34 = lean_ctor_get(x_28, 4);
lean_dec(x_34);
x_35 = lean_ctor_get(x_28, 3);
lean_dec(x_35);
x_36 = lean_ctor_get(x_28, 2);
lean_dec(x_36);
x_37 = lean_ctor_get(x_28, 1);
lean_dec(x_37);
x_38 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_38, 0, x_8);
x_39 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_39, 0, x_12);
x_40 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_40, 0, x_23);
x_41 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_41, 0, x_24);
x_42 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_42, 0, x_26);
x_43 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_43, 0, x_29);
lean_inc_ref(x_42);
lean_inc_ref(x_39);
lean_inc_ref(x_38);
x_44 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_38, x_39, x_42, x_40, x_41, x_43);
x_45 = lean_ctor_get(x_44, 1);
lean_inc(x_45);
x_46 = lean_ctor_get(x_44, 2);
lean_inc(x_46);
lean_dec_ref(x_44);
x_47 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instRing___redArg___lam__0), 4, 1);
lean_closure_set(x_47, 0, x_3);
x_48 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_48, 0, x_5);
x_49 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_49, 0, x_7);
x_50 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_50, 0, x_21);
x_51 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_51, 0, x_31);
x_52 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_52, 0, x_33);
x_53 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_53, 0, lean_box(0));
lean_closure_set(x_53, 1, x_52);
x_54 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_54, 0, lean_box(0));
lean_closure_set(x_54, 1, x_51);
x_55 = lp_mathlib_Function_Injective_addMonoid___redArg(x_38, x_39, x_42);
lean_ctor_set(x_4, 1, x_49);
lean_ctor_set(x_4, 0, x_55);
lean_ctor_set(x_2, 3, x_48);
lean_ctor_set(x_2, 2, x_54);
lean_ctor_set(x_2, 1, x_50);
lean_ctor_set(x_28, 4, x_53);
lean_ctor_set(x_28, 3, x_47);
lean_ctor_set(x_28, 2, x_46);
lean_ctor_set(x_28, 1, x_45);
lean_ctor_set(x_28, 0, x_2);
return x_28;
}
else
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; 
x_56 = lean_ctor_get(x_28, 0);
lean_inc(x_56);
lean_dec(x_28);
x_57 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_57, 0, x_8);
x_58 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_58, 0, x_12);
x_59 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_59, 0, x_23);
x_60 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_60, 0, x_24);
x_61 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_61, 0, x_26);
x_62 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_62, 0, x_29);
lean_inc_ref(x_61);
lean_inc_ref(x_58);
lean_inc_ref(x_57);
x_63 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_57, x_58, x_61, x_59, x_60, x_62);
x_64 = lean_ctor_get(x_63, 1);
lean_inc(x_64);
x_65 = lean_ctor_get(x_63, 2);
lean_inc(x_65);
lean_dec_ref(x_63);
x_66 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instRing___redArg___lam__0), 4, 1);
lean_closure_set(x_66, 0, x_3);
x_67 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_67, 0, x_5);
x_68 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_68, 0, x_7);
x_69 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_69, 0, x_21);
x_70 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_70, 0, x_31);
x_71 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_71, 0, x_56);
x_72 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_72, 0, lean_box(0));
lean_closure_set(x_72, 1, x_71);
x_73 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_73, 0, lean_box(0));
lean_closure_set(x_73, 1, x_70);
x_74 = lp_mathlib_Function_Injective_addMonoid___redArg(x_57, x_58, x_61);
lean_ctor_set(x_4, 1, x_68);
lean_ctor_set(x_4, 0, x_74);
lean_ctor_set(x_2, 3, x_67);
lean_ctor_set(x_2, 2, x_73);
lean_ctor_set(x_2, 1, x_69);
x_75 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_75, 0, x_2);
lean_ctor_set(x_75, 1, x_64);
lean_ctor_set(x_75, 2, x_65);
lean_ctor_set(x_75, 3, x_66);
lean_ctor_set(x_75, 4, x_72);
return x_75;
}
}
else
{
lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; 
x_76 = lean_ctor_get(x_4, 0);
lean_inc(x_76);
lean_dec(x_4);
x_77 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_78 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_77);
x_79 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_13);
x_80 = lean_ctor_get(x_79, 0);
lean_inc(x_80);
lean_dec_ref(x_79);
x_81 = lean_ctor_get(x_77, 0);
lean_inc(x_81);
if (lean_is_exclusive(x_77)) {
 lean_ctor_release(x_77, 0);
 lean_ctor_release(x_77, 1);
 lean_ctor_release(x_77, 2);
 lean_ctor_release(x_77, 3);
 lean_ctor_release(x_77, 4);
 x_82 = x_77;
} else {
 lean_dec_ref(x_77);
 x_82 = lean_box(0);
}
x_83 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_83, 0, x_8);
x_84 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_84, 0, x_12);
x_85 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_85, 0, x_23);
x_86 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_86, 0, x_24);
x_87 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_87, 0, x_76);
x_88 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_88, 0, x_78);
lean_inc_ref(x_87);
lean_inc_ref(x_84);
lean_inc_ref(x_83);
x_89 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_83, x_84, x_87, x_85, x_86, x_88);
x_90 = lean_ctor_get(x_89, 1);
lean_inc(x_90);
x_91 = lean_ctor_get(x_89, 2);
lean_inc(x_91);
lean_dec_ref(x_89);
x_92 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instRing___redArg___lam__0), 4, 1);
lean_closure_set(x_92, 0, x_3);
x_93 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_93, 0, x_5);
x_94 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_94, 0, x_7);
x_95 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_95, 0, x_21);
x_96 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_96, 0, x_80);
x_97 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_97, 0, x_81);
x_98 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_98, 0, lean_box(0));
lean_closure_set(x_98, 1, x_97);
x_99 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_99, 0, lean_box(0));
lean_closure_set(x_99, 1, x_96);
x_100 = lp_mathlib_Function_Injective_addMonoid___redArg(x_83, x_84, x_87);
x_101 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_101, 0, x_100);
lean_ctor_set(x_101, 1, x_94);
lean_ctor_set(x_2, 3, x_93);
lean_ctor_set(x_2, 2, x_99);
lean_ctor_set(x_2, 1, x_95);
lean_ctor_set(x_2, 0, x_101);
if (lean_is_scalar(x_82)) {
 x_102 = lean_alloc_ctor(0, 5, 0);
} else {
 x_102 = x_82;
}
lean_ctor_set(x_102, 0, x_2);
lean_ctor_set(x_102, 1, x_90);
lean_ctor_set(x_102, 2, x_91);
lean_ctor_set(x_102, 3, x_92);
lean_ctor_set(x_102, 4, x_98);
return x_102;
}
}
else
{
lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; 
lean_dec(x_2);
lean_inc_ref(x_13);
x_103 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_13);
x_104 = lean_ctor_get(x_103, 0);
lean_inc_ref(x_104);
lean_dec_ref(x_103);
x_105 = lean_ctor_get(x_104, 0);
lean_inc(x_105);
lean_dec_ref(x_104);
lean_inc_ref(x_1);
x_106 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_107 = lean_ctor_get(x_106, 1);
lean_inc(x_107);
x_108 = lean_ctor_get(x_106, 2);
lean_inc(x_108);
lean_dec_ref(x_106);
x_109 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_109);
if (lean_is_exclusive(x_4)) {
 lean_ctor_release(x_4, 0);
 lean_ctor_release(x_4, 1);
 x_110 = x_4;
} else {
 lean_dec_ref(x_4);
 x_110 = lean_box(0);
}
x_111 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_112 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_111);
x_113 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_13);
x_114 = lean_ctor_get(x_113, 0);
lean_inc(x_114);
lean_dec_ref(x_113);
x_115 = lean_ctor_get(x_111, 0);
lean_inc(x_115);
if (lean_is_exclusive(x_111)) {
 lean_ctor_release(x_111, 0);
 lean_ctor_release(x_111, 1);
 lean_ctor_release(x_111, 2);
 lean_ctor_release(x_111, 3);
 lean_ctor_release(x_111, 4);
 x_116 = x_111;
} else {
 lean_dec_ref(x_111);
 x_116 = lean_box(0);
}
x_117 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_117, 0, x_8);
x_118 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_118, 0, x_12);
x_119 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instInvOfContinuousInv___redArg___lam__0), 3, 1);
lean_closure_set(x_119, 0, x_107);
x_120 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_120, 0, x_108);
x_121 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_121, 0, x_109);
x_122 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_122, 0, x_112);
lean_inc_ref(x_121);
lean_inc_ref(x_118);
lean_inc_ref(x_117);
x_123 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_117, x_118, x_121, x_119, x_120, x_122);
x_124 = lean_ctor_get(x_123, 1);
lean_inc(x_124);
x_125 = lean_ctor_get(x_123, 2);
lean_inc(x_125);
lean_dec_ref(x_123);
x_126 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instRing___redArg___lam__0), 4, 1);
lean_closure_set(x_126, 0, x_3);
x_127 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMonoidOfContinuousMul___redArg___lam__0), 4, 1);
lean_closure_set(x_127, 0, x_5);
x_128 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_128, 0, x_7);
x_129 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_129, 0, x_105);
x_130 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instNatCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_130, 0, x_114);
x_131 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_131, 0, x_115);
x_132 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_132, 0, lean_box(0));
lean_closure_set(x_132, 1, x_131);
x_133 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_133, 0, lean_box(0));
lean_closure_set(x_133, 1, x_130);
x_134 = lp_mathlib_Function_Injective_addMonoid___redArg(x_117, x_118, x_121);
if (lean_is_scalar(x_110)) {
 x_135 = lean_alloc_ctor(0, 2, 0);
} else {
 x_135 = x_110;
}
lean_ctor_set(x_135, 0, x_134);
lean_ctor_set(x_135, 1, x_128);
x_136 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_136, 0, x_135);
lean_ctor_set(x_136, 1, x_129);
lean_ctor_set(x_136, 2, x_133);
lean_ctor_set(x_136, 3, x_127);
if (lean_is_scalar(x_116)) {
 x_137 = lean_alloc_ctor(0, 5, 0);
} else {
 x_137 = x_116;
}
lean_ctor_set(x_137, 0, x_136);
lean_ctor_set(x_137, 1, x_124);
lean_ctor_set(x_137, 2, x_125);
lean_ctor_set(x_137, 3, x_126);
lean_ctor_set(x_137, 4, x_132);
return x_137;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instCommRingOfIsTopologicalRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_instCommRingOfIsTopologicalRing___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeftContinuous(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MonoidHom_compLeftContinuous___redArg(x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeftContinuous___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MonoidHom_compLeftContinuous___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_compLeftContinuous___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_RingHom_compLeftContinuous(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_9);
lean_dec_ref(x_6);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnRingHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_coeFnAddMonoidHom___closed__0;
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnRingHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_ContinuousMap_coeFnRingHom(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubmodule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lean_box(0);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubmodule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_continuousSubmodule(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_4);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_1(x_3, x_4);
x_6 = lean_apply_2(x_1, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_ContinuousMap_instSMul___redArg(x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instVAdd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instVAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_ContinuousMap_instVAdd___redArg(x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulActionOfContinuousConstSMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_9, 0, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulActionOfContinuousConstSMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instMulActionOfContinuousConstSMul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_ContinuousMap_instMulActionOfContinuousConstSMul(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instDistribMulActionOfContinuousConstSMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_11, 0, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instDistribMulActionOfContinuousConstSMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instDistribMulActionOfContinuousConstSMul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ContinuousMap_instDistribMulActionOfContinuousConstSMul(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_11, 0, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ContinuousMap_module(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearMap_compLeftContinuous___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_LinearMap_toAddMonoidHom___redArg(x_1);
x_3 = lp_mathlib_AddMonoidHom_compLeftContinuous___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearMap_compLeftContinuous(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_ContinuousLinearMap_compLeftContinuous___redArg(x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearMap_compLeftContinuous___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_ContinuousLinearMap_compLeftContinuous(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec(x_13);
lean_dec(x_10);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_18;
}
}
static lean_object* _init_lp_mathlib_ContinuousLinearMap_const___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_const___redArg___lam__0___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearMap_const(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ContinuousLinearMap_const___closed__0;
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearMap_const___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ContinuousLinearMap_const(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnLinearMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ContinuousMap_coeFnAddMonoidHom___closed__0;
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnLinearMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ContinuousMap_coeFnLinearMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalCLM___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalCLM___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_evalCLM___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalCLM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_ContinuousMap_evalCLM___redArg(x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalCLM___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_ContinuousMap_evalCLM(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubalgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lean_box(0);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_continuousSubalgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_continuousSubalgebra(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_C___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_1(x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_C___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_ContinuousMap_C___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_C___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_C___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_C(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_C___redArg(x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_C___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_C(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_algebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_C___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_algebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_algebra___redArg(x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_algebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_algebra(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_compLeftContinuous(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_MonoidHom_compLeftContinuous___redArg(x_15);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_compLeftContinuous___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MonoidHom_compLeftContinuous___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_compLeftContinuous___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_AlgHom_compLeftContinuous(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compRightAlgHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_ContinuousMap_comp___redArg(x_2, x_1);
x_5 = lean_apply_1(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compRightAlgHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_compRightAlgHom___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compRightAlgHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_ContinuousMap_compRightAlgHom___redArg(x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_compRightAlgHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_ContinuousMap_compRightAlgHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnAlgHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_coeFnAddMonoidHom___closed__0;
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_coeFnAlgHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousMap_coeFnAlgHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul_x27___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ContinuousMap_instSMul_x27___redArg(x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_instSMul_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ContinuousMap_instSMul_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_8);
lean_dec_ref(x_4);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_13, 0, x_9);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module_x27___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_instMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_module_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_ContinuousMap_module_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_8);
lean_dec_ref(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalAlgHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ContinuousMap_evalCLM___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalAlgHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ContinuousMap_evalAlgHom___redArg(x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousMap_evalAlgHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ContinuousMap_evalAlgHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FieldSimp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_InfiniteSum_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Module_LinearMap(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Ring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_UniformSpace_CompactConvergence(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_ContinuousMap_Algebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FieldSimp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_InfiniteSum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Module_LinearMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_UniformSpace_CompactConvergence(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ContinuousMap_coeFnAddMonoidHom___closed__0 = _init_lp_mathlib_ContinuousMap_coeFnAddMonoidHom___closed__0();
lean_mark_persistent(lp_mathlib_ContinuousMap_coeFnAddMonoidHom___closed__0);
lp_mathlib_ContinuousLinearMap_const___closed__0 = _init_lp_mathlib_ContinuousLinearMap_const___closed__0();
lean_mark_persistent(lp_mathlib_ContinuousLinearMap_const___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
