// Lean compiler output
// Module: Mathlib.LinearAlgebra.Multilinear.Basic
// Imports: public import Init public import Mathlib.Algebra.BigOperators.Group.Finset.Powerset public import Mathlib.Algebra.NoZeroSMulDivisors.Pi public import Mathlib.Data.Finset.Sort public import Mathlib.Data.Fintype.BigOperators public import Mathlib.Data.Fintype.Powerset public import Mathlib.LinearAlgebra.Pi public import Mathlib.Logic.Equiv.Fintype public import Mathlib.Tactic.Abel
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
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiRing___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piLinearMap___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg___lam__2(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict_u2097___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebraFin___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap_u2097___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_addCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_addCommMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton_u2097___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebraFin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_range___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebraFin___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap_u2097___boxed(lean_object**);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_linearDeriv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_linearDeriv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_List_prod___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_coeAddMonoidHom___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piLinearMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAdd___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap_u2097(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instFunLikeForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_prod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_smulRight___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebraFin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__1(lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__5;
lean_object* lp_mathlib_LinearMap_addCommMonoid___redArg(lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__15;
lean_object* lp_mathlib_Finset_orderIsoOfFin___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_codRestrict___redArg(lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__9;
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__10;
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__20;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_addCommMonoid___redArg(lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__14;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_smulRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instDistribMulActionOfSMulCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_coeAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebraFin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_empty(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_range(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap_u2097___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_piCongrLeft_x27___redArg(lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1;
lean_object* lp_mathlib_Function_eval(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_pi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAddCommGroup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap_u2097(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict_u2097___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restrictScalars___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_smulRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__25;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_prod___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instFunLikeForall___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instDistribMulActionOfSMulCommClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_toLinearMap___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAddCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__21;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_instLinearOrder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restrictScalars___boxed(lean_object**);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_smulRight___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constOfIsEmpty___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___redArg___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__24;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap_u2097___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAdd___redArg(lean_object*);
lean_object* lp_mathlib_CommSemiring_toCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instFunLikeForall(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_apply_u2097_x27___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__19;
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__17;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSub___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__1(lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_linearDeriv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__1___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSub___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrRight___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap_u2097___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instDistribMulActionOfSMulCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableMem___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instNeg___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_toLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_toLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_pi___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__0;
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__16;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton_u2097(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constOfIsEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_update___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongr___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_coeAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restrictScalars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instDecidableEqFin___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict_u2097___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict_u2097(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__22;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMapMultilinear___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_toLinearMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_ofFn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constOfIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_codRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_smulRight___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__23;
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_codRestrict___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_linearDeriv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMapMultilinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton_u2097___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instFunLikeForall___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instFunLikeForall(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instFunLikeForall___lam__0), 2, 0);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instFunLikeForall___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instFunLikeForall(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_10;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lean", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Parser", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticSeq", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__3;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__2;
x_3 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__1;
x_4 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticSeq1Indented", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__6;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__2;
x_3 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__1;
x_4 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("null", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__8;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Aesop", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Frontend", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesopTactic", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__12;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__1;
x_3 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__11;
x_4 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__10;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__14;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__15;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__5;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__9;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__17;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__16;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__18;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__13;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__19;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__21() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__20;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__9;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__22() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__21;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__23() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__22;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__7;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__24() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__23;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__25() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__24;
x_2 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__4;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__25;
return x_1;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_mk_x27___auto__3() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__25;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mk_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_inc(x_11);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mk_x27___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mk_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_MultilinearMap_mk_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mk_x27___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MultilinearMap_mk_x27___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAdd___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
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
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAdd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instAdd___redArg___lam__0), 4, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instAdd___redArg(x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAdd___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instAdd(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MultilinearMap_instZero___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
lean_dec_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_4, 0, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instZero___redArg(x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instZero(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instZero___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MultilinearMap_instZero___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instInhabited___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
lean_dec_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_4, 0, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instInhabited(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instInhabited___redArg(x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instInhabited___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instInhabited(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instInhabited___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MultilinearMap_instInhabited___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSMul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_1(x_3, x_4);
x_6 = lean_apply_2(x_1, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MultilinearMap_instSMul___redArg(x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSMul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MultilinearMap_instSMul(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_addCommMonoid___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
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
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_addCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_addCommMonoid___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc_ref(x_1);
x_3 = lp_mathlib_MultilinearMap_instAdd___redArg(x_1);
x_4 = lp_mathlib_MultilinearMap_instZero___redArg(x_1);
lean_dec_ref(x_1);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
lean_ctor_set(x_5, 2, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_addCommMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_addCommMonoid___redArg(x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_addCommMonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_addCommMonoid(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_10;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_coeAddMonoidHom___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instFunLikeForall___lam__0), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_coeAddMonoidHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_coeAddMonoidHom___closed__0;
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_coeAddMonoidHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_coeAddMonoidHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_toLinearMap___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_Function_update___boxed), 7, 6);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_1);
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, x_3);
lean_closure_set(x_6, 5, x_5);
x_7 = lean_apply_1(x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_toLinearMap___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_toLinearMap___redArg___lam__0), 5, 4);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_3);
lean_closure_set(x_5, 2, x_4);
lean_closure_set(x_5, 3, x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_toLinearMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_MultilinearMap_toLinearMap___redArg(x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_toLinearMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_MultilinearMap_toLinearMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_prod___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_1(x_2, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_prod___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_prod___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_prod(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_MultilinearMap_prod___redArg(x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_prod___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_MultilinearMap_prod(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_pi___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_pi___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_pi___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_pi(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_MultilinearMap_pi___redArg(x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_pi___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_MultilinearMap_pi(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_3, x_1);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_ofSubsingleton___redArg___lam__2), 2, 0);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_MultilinearMap_ofSubsingleton___redArg(x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_MultilinearMap_ofSubsingleton(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constOfIsEmpty(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, lean_box(0));
lean_closure_set(x_12, 2, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constOfIsEmpty___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constOfIsEmpty___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_MultilinearMap_constOfIsEmpty(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; uint8_t x_8; 
lean_inc(x_1);
x_7 = lean_alloc_closure((void*)(l_instDecidableEqFin___boxed), 3, 1);
lean_closure_set(x_7, 0, x_1);
lean_inc(x_2);
lean_inc(x_6);
x_8 = lp_mathlib_Multiset_decidableMem___redArg(x_7, x_6, x_2);
if (x_8 == 0)
{
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_2);
lean_dec(x_1);
lean_inc(x_3);
return x_3;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lp_mathlib_Fin_instLinearOrder(x_1);
x_10 = lp_mathlib_Finset_orderIsoOfFin___redArg(x_9, x_2, x_4);
x_11 = lp_mathlib_Equiv_symm___redArg(x_10);
x_12 = lp_mathlib_Equiv_toEmbedding___redArg(x_11);
x_13 = lean_apply_1(x_12, x_6);
x_14 = lean_apply_1(x_5, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_MultilinearMap_restr___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_4);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_restr___redArg___lam__0___boxed), 6, 5);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_2);
lean_closure_set(x_7, 2, x_3);
lean_closure_set(x_7, 3, x_4);
lean_closure_set(x_7, 4, x_6);
x_8 = lean_apply_1(x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_restr___redArg___lam__1), 6, 5);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_4);
lean_closure_set(x_6, 2, x_5);
lean_closure_set(x_6, 3, x_1);
lean_closure_set(x_6, 4, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_MultilinearMap_restr___redArg(x_9, x_10, x_11, x_12, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_MultilinearMap_restr(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
lean_inc(x_3);
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_2(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_compLinearMap___redArg___lam__0), 3, 2);
lean_closure_set(x_4, 0, x_3);
lean_closure_set(x_4, 1, x_1);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_compLinearMap___redArg___lam__1), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_MultilinearMap_compLinearMap___redArg(x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_MultilinearMap_compLinearMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_codRestrict___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_codRestrict___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_codRestrict(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MultilinearMap_codRestrict___redArg(x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_codRestrict___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MultilinearMap_codRestrict(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restrictScalars___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instFunLikeForall___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restrictScalars(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_MultilinearMap_restrictScalars___redArg(x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_restrictScalars___boxed(lean_object** _args) {
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
x_18 = lp_mathlib_MultilinearMap_restrictScalars(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongr___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_1(x_4, x_3);
x_6 = lean_apply_1(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongr___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_domDomCongr___redArg___lam__0), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongr___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_domDomCongr___redArg___lam__1), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MultilinearMap_domDomCongr___redArg(x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MultilinearMap_domDomCongr(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_domDomCongr___boxed), 12, 11);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, lean_box(0));
lean_closure_set(x_7, 3, x_1);
lean_closure_set(x_7, 4, x_2);
lean_closure_set(x_7, 5, x_3);
lean_closure_set(x_7, 6, x_4);
lean_closure_set(x_7, 7, x_5);
lean_closure_set(x_7, 8, lean_box(0));
lean_closure_set(x_7, 9, lean_box(0));
lean_closure_set(x_7, 10, x_6);
x_8 = lp_mathlib_Equiv_symm___redArg(x_6);
x_9 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_domDomCongr___boxed), 12, 11);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, lean_box(0));
lean_closure_set(x_9, 3, x_1);
lean_closure_set(x_9, 4, x_2);
lean_closure_set(x_9, 5, x_3);
lean_closure_set(x_9, 6, x_4);
lean_closure_set(x_9, 7, x_5);
lean_closure_set(x_9, 8, lean_box(0));
lean_closure_set(x_9, 9, lean_box(0));
lean_closure_set(x_9, 10, x_8);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_7);
lean_ctor_set(x_10, 1, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_MultilinearMap_domDomCongrEquiv___redArg(x_4, x_5, x_6, x_7, x_8, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
lean_inc(x_4);
x_5 = lean_apply_1(x_1, x_4);
x_6 = lean_unbox(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
lean_dec(x_3);
x_7 = lean_apply_1(x_2, x_4);
return x_7;
}
else
{
lean_object* x_8; 
lean_dec(x_2);
x_8 = lean_apply_1(x_3, x_4);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_domDomRestrict___redArg___lam__0), 4, 3);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_2);
lean_closure_set(x_5, 2, x_4);
x_6 = lean_apply_1(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_domDomRestrict___redArg___lam__1), 4, 3);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_MultilinearMap_domDomRestrict___redArg(x_10, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_MultilinearMap_domDomRestrict(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_linearDeriv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_toLinearMap___redArg___lam__0), 5, 4);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_2);
lean_closure_set(x_6, 2, x_4);
lean_closure_set(x_6, 3, x_3);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, x_4);
x_8 = lp_mathlib_LinearMap_comp___redArg(x_6, x_7);
x_9 = lean_apply_1(x_8, x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_linearDeriv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_linearDeriv___redArg___lam__0), 5, 3);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_5);
lean_closure_set(x_6, 2, x_4);
x_7 = lp_mathlib_LinearMap_addCommMonoid___redArg(x_1);
x_8 = lp_mathlib_Finset_sum___redArg(x_7, x_3, x_6);
lean_dec_ref(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_linearDeriv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_MultilinearMap_linearDeriv___redArg(x_7, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_linearDeriv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_MultilinearMap_linearDeriv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_compMultilinearMap___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instFunLikeForall___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_2);
x_5 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, lean_box(0));
lean_closure_set(x_5, 3, x_3);
lean_closure_set(x_5, 4, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_LinearMap_compMultilinearMap___redArg(x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_LinearMap_compMultilinearMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instDistribMulActionOfSMulCommClass(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_14, 0, x_11);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instDistribMulActionOfSMulCommClass___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instDistribMulActionOfSMulCommClass___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_MultilinearMap_instDistribMulActionOfSMulCommClass(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_12);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instModule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_14, 0, x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instModule___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instModule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_MultilinearMap_instModule(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap_u2097(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20) {
_start:
{
lean_object* x_21; 
x_21 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_compMultilinearMap___boxed), 14, 13);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, lean_box(0));
lean_closure_set(x_21, 2, lean_box(0));
lean_closure_set(x_21, 3, lean_box(0));
lean_closure_set(x_21, 4, lean_box(0));
lean_closure_set(x_21, 5, x_7);
lean_closure_set(x_21, 6, x_8);
lean_closure_set(x_21, 7, x_10);
lean_closure_set(x_21, 8, x_15);
lean_closure_set(x_21, 9, x_9);
lean_closure_set(x_21, 10, x_11);
lean_closure_set(x_21, 11, x_17);
lean_closure_set(x_21, 12, x_20);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap_u2097___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_compMultilinearMap___boxed), 14, 13);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, lean_box(0));
lean_closure_set(x_9, 3, lean_box(0));
lean_closure_set(x_9, 4, lean_box(0));
lean_closure_set(x_9, 5, x_1);
lean_closure_set(x_9, 6, x_2);
lean_closure_set(x_9, 7, x_4);
lean_closure_set(x_9, 8, x_6);
lean_closure_set(x_9, 9, x_3);
lean_closure_set(x_9, 10, x_5);
lean_closure_set(x_9, 11, x_7);
lean_closure_set(x_9, 12, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compMultilinearMap_u2097___boxed(lean_object** _args) {
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
lean_object* x_19 = _args[18];
lean_object* x_20 = _args[19];
_start:
{
lean_object* x_21; 
x_21 = lp_mathlib_LinearMap_compMultilinearMap_u2097(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20);
lean_dec(x_16);
lean_dec(x_13);
lean_dec_ref(x_12);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrRight___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_LinearMap_compMultilinearMap___redArg(x_1, x_2);
x_5 = lean_apply_1(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrRight___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
x_10 = lp_mathlib_LinearEquiv_symm___redArg(x_8);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
lean_dec(x_13);
x_14 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_compMultilinearMap___boxed), 14, 13);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, lean_box(0));
lean_closure_set(x_14, 2, lean_box(0));
lean_closure_set(x_14, 3, lean_box(0));
lean_closure_set(x_14, 4, lean_box(0));
lean_closure_set(x_14, 5, x_1);
lean_closure_set(x_14, 6, x_2);
lean_closure_set(x_14, 7, x_4);
lean_closure_set(x_14, 8, x_6);
lean_closure_set(x_14, 9, x_3);
lean_closure_set(x_14, 10, x_5);
lean_closure_set(x_14, 11, x_7);
lean_closure_set(x_14, 12, x_9);
x_15 = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_multilinearMapCongrRight___redArg___lam__0), 3, 1);
lean_closure_set(x_15, 0, x_12);
lean_ctor_set(x_10, 1, x_15);
lean_ctor_set(x_10, 0, x_14);
return x_10;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_16 = lean_ctor_get(x_10, 0);
lean_inc(x_16);
lean_dec(x_10);
x_17 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_compMultilinearMap___boxed), 14, 13);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, lean_box(0));
lean_closure_set(x_17, 2, lean_box(0));
lean_closure_set(x_17, 3, lean_box(0));
lean_closure_set(x_17, 4, lean_box(0));
lean_closure_set(x_17, 5, x_1);
lean_closure_set(x_17, 6, x_2);
lean_closure_set(x_17, 7, x_4);
lean_closure_set(x_17, 8, x_6);
lean_closure_set(x_17, 9, x_3);
lean_closure_set(x_17, 10, x_5);
lean_closure_set(x_17, 11, x_7);
lean_closure_set(x_17, 12, x_9);
x_18 = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_multilinearMapCongrRight___redArg___lam__0), 3, 1);
lean_closure_set(x_18, 0, x_16);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21) {
_start:
{
lean_object* x_22; 
x_22 = lp_mathlib_LinearEquiv_multilinearMapCongrRight___redArg(x_7, x_8, x_9, x_10, x_11, x_15, x_17, x_21);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrRight___boxed(lean_object** _args) {
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
lean_object* x_19 = _args[18];
lean_object* x_20 = _args[19];
lean_object* x_21 = _args[20];
_start:
{
lean_object* x_22; 
x_22 = lp_mathlib_LinearEquiv_multilinearMapCongrRight(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21);
lean_dec(x_16);
lean_dec(x_13);
lean_dec_ref(x_12);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton_u2097___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lp_mathlib_MultilinearMap_ofSubsingleton___redArg(x_1);
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
return x_2;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_inc(x_4);
lean_dec(x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton_u2097(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_MultilinearMap_ofSubsingleton_u2097___redArg(x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_ofSubsingleton_u2097___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_MultilinearMap_ofSubsingleton_u2097(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lp_mathlib_Equiv_piCongrLeft_x27___redArg(x_1);
x_5 = lp_mathlib_Equiv_symm___redArg(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg___lam__0), 3, 2);
lean_closure_set(x_6, 0, x_5);
lean_closure_set(x_6, 1, x_3);
x_7 = lean_apply_1(x_2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_Equiv_piCongrLeft_x27___redArg(x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg___lam__0), 3, 2);
lean_closure_set(x_5, 0, x_4);
lean_closure_set(x_5, 1, x_3);
x_6 = lean_apply_1(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg___lam__1), 3, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg___lam__3), 3, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___redArg(x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_MultilinearMap_domDomCongrLinearEquiv_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_3);
lean_dec_ref(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___redArg___lam__0), 2, 1);
lean_closure_set(x_6, 0, x_2);
x_7 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___redArg___lam__1), 2, 1);
lean_closure_set(x_7, 0, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_constOfIsEmpty___boxed), 11, 10);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, lean_box(0));
lean_closure_set(x_8, 2, lean_box(0));
lean_closure_set(x_8, 3, lean_box(0));
lean_closure_set(x_8, 4, x_1);
lean_closure_set(x_8, 5, x_2);
lean_closure_set(x_8, 6, x_4);
lean_closure_set(x_8, 7, x_3);
lean_closure_set(x_8, 8, x_5);
lean_closure_set(x_8, 9, lean_box(0));
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___redArg(x_6, x_7, x_8, x_9, x_10);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_12);
lean_dec_ref(x_11);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; uint8_t x_8; 
x_7 = lp_mathlib_MultilinearMap_domDomCongrEquiv___redArg(x_1, x_2, x_4, x_3, x_5, x_6);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
return x_7;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_7, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_MultilinearMap_domDomCongrLinearEquiv___redArg(x_5, x_6, x_7, x_9, x_11, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomCongrLinearEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_MultilinearMap_domDomCongrLinearEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_10);
lean_dec_ref(x_8);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict_u2097___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_MultilinearMap_domDomRestrict___redArg___lam__1(x_1, x_3, x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict_u2097___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_domDomRestrict_u2097___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict_u2097(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MultilinearMap_domDomRestrict_u2097___redArg(x_10, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_domDomRestrict_u2097___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MultilinearMap_domDomRestrict_u2097(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap_u2097___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MultilinearMap_compLinearMap___redArg___lam__1(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap_u2097___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_compLinearMap_u2097___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap_u2097(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_MultilinearMap_compLinearMap_u2097___redArg(x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMap_u2097___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_MultilinearMap_compLinearMap_u2097(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_apply_1(x_1, x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_apply_1(x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_apply_1(x_1, x_2);
x_5 = lp_mathlib_LinearEquiv_symm___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_1(x_6, x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MultilinearMap_compLinearMap___redArg___lam__1(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg___lam__1), 3, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg___lam__2), 3, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_compLinearMap_u2097___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_LinearEquiv_multilinearMapCongrLeft___redArg(x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_multilinearMapCongrLeft___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_LinearEquiv_multilinearMapCongrLeft(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMapMultilinear(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_compLinearMap_u2097___boxed), 13, 12);
lean_closure_set(x_13, 0, lean_box(0));
lean_closure_set(x_13, 1, lean_box(0));
lean_closure_set(x_13, 2, lean_box(0));
lean_closure_set(x_13, 3, lean_box(0));
lean_closure_set(x_13, 4, lean_box(0));
lean_closure_set(x_13, 5, x_6);
lean_closure_set(x_13, 6, x_7);
lean_closure_set(x_13, 7, x_8);
lean_closure_set(x_13, 8, x_9);
lean_closure_set(x_13, 9, x_10);
lean_closure_set(x_13, 10, x_11);
lean_closure_set(x_13, 11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_compLinearMapMultilinear___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_compLinearMap_u2097___boxed), 13, 12);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, lean_box(0));
lean_closure_set(x_8, 2, lean_box(0));
lean_closure_set(x_8, 3, lean_box(0));
lean_closure_set(x_8, 4, lean_box(0));
lean_closure_set(x_8, 5, x_1);
lean_closure_set(x_8, 6, x_2);
lean_closure_set(x_8, 7, x_3);
lean_closure_set(x_8, 8, x_4);
lean_closure_set(x_8, 9, x_5);
lean_closure_set(x_8, 10, x_6);
lean_closure_set(x_8, 11, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piLinearMap___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_apply_u2097_x27___lam__0), 2, 1);
lean_closure_set(x_11, 0, x_8);
x_12 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_compLinearMap_u2097___boxed), 13, 12);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, lean_box(0));
lean_closure_set(x_12, 2, lean_box(0));
lean_closure_set(x_12, 3, lean_box(0));
lean_closure_set(x_12, 4, lean_box(0));
lean_closure_set(x_12, 5, x_1);
lean_closure_set(x_12, 6, x_2);
lean_closure_set(x_12, 7, x_3);
lean_closure_set(x_12, 8, x_4);
lean_closure_set(x_12, 9, x_5);
lean_closure_set(x_12, 10, x_6);
lean_closure_set(x_12, 11, x_7);
x_13 = lp_mathlib_LinearMap_compMultilinearMap___redArg(x_11, x_12);
x_14 = lean_apply_2(x_13, x_9, x_10);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piLinearMap___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_piLinearMap___redArg___lam__0), 10, 7);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, x_3);
lean_closure_set(x_8, 3, x_4);
lean_closure_set(x_8, 4, x_5);
lean_closure_set(x_8, 5, x_6);
lean_closure_set(x_8, 6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piLinearMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MultilinearMap_piLinearMap___redArg(x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_mkPiAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lp_mathlib_Finset_prod___redArg(x_1, x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MultilinearMap_mkPiAlgebra___redArg___lam__1(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_mkPiAlgebra___redArg___lam__1___boxed), 3, 2);
lean_closure_set(x_4, 0, x_3);
lean_closure_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_MultilinearMap_mkPiAlgebra___redArg(x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_MultilinearMap_mkPiAlgebra(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebra___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MultilinearMap_mkPiAlgebra___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebraFin___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = l_List_ofFn___redArg(x_1, x_4);
x_6 = lp_batteries_List_prod___redArg(x_2, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebraFin___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
x_5 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_3);
x_8 = lean_ctor_get(x_7, 2);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_mkPiAlgebraFin___redArg___lam__0), 4, 3);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_6);
lean_closure_set(x_9, 2, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebraFin(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_MultilinearMap_mkPiAlgebraFin___redArg(x_2, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebraFin___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_MultilinearMap_mkPiAlgebraFin(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiAlgebraFin___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MultilinearMap_mkPiAlgebraFin___redArg(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_MultilinearMap_smulRight___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_smulRight___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_MultilinearMap_smulRight___redArg___closed__0;
x_5 = lp_mathlib_LinearMap_smulRight___redArg(x_1, x_4, x_3);
x_6 = lp_mathlib_LinearMap_compMultilinearMap___redArg(x_5, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_smulRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_MultilinearMap_smulRight___redArg(x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_smulRight___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_MultilinearMap_smulRight(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_MultilinearMap_mkPiAlgebra___redArg(x_1, x_3);
x_6 = lp_mathlib_MultilinearMap_smulRight___redArg(x_2, x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_MultilinearMap_mkPiRing___redArg(x_4, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiRing___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_MultilinearMap_mkPiRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_mkPiRing___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_MultilinearMap_mkPiRing___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instNeg___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_2, x_3);
x_5 = lean_apply_1(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instNeg___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
lean_dec_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instNeg___redArg___lam__0), 3, 1);
lean_closure_set(x_4, 0, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instNeg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instNeg___redArg(x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instNeg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instNeg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instNeg___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MultilinearMap_instNeg___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSub___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
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
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSub___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 2);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instSub___redArg___lam__0), 4, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSub(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instSub___redArg(x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instSub___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instSub(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAddCommGroup___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_1(x_3, x_4);
x_6 = lean_apply_2(x_1, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAddCommGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
x_4 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_instAddCommGroup___redArg___lam__0), 4, 1);
lean_closure_set(x_4, 0, x_3);
lean_inc_ref(x_2);
x_5 = lp_mathlib_MultilinearMap_addCommMonoid___redArg(x_2);
x_6 = lp_mathlib_MultilinearMap_instNeg___redArg(x_1);
x_7 = lp_mathlib_MultilinearMap_instSub___redArg(x_1);
x_8 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_7);
lean_ctor_set(x_8, 3, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAddCommGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instAddCommGroup___redArg(x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_instAddCommGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MultilinearMap_instAddCommGroup(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_MultilinearMap_mkPiRing___redArg(x_1, x_2, x_3, x_4);
x_7 = lean_apply_1(x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_5 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_4);
x_6 = lean_ctor_get(x_5, 2);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_2);
lean_closure_set(x_7, 2, x_3);
x_8 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_piRingEquiv___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_8, 0, x_6);
x_9 = lean_alloc_closure((void*)(lp_mathlib_MultilinearMap_constLinearEquivOfIsEmpty___redArg___lam__1), 2, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_7);
lean_ctor_set(x_10, 1, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_MultilinearMap_piRingEquiv___redArg(x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_piRingEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_MultilinearMap_piRingEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lean_box(0);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_map___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_MultilinearMap_map(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_range(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lean_box(0);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MultilinearMap_range___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_MultilinearMap_range(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_12;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Powerset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_NoZeroSMulDivisors_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Sort(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Powerset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Fintype(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Multilinear_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_NoZeroSMulDivisors_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Fintype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__0 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__0();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__0);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__1 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__1();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__1);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__2 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__2();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__2);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__3 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__3();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__3);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__4 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__4();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__4);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__5 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__5();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__5);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__6 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__6();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__6);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__7 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__7();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__7);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__8 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__8();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__8);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__9 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__9();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__9);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__10 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__10();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__10);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__11 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__11();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__11);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__12 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__12();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__12);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__13 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__13();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__13);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__14 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__14();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__14);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__15 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__15();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__15);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__16 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__16();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__16);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__17 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__17();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__17);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__18 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__18();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__18);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__19 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__19();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__19);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__20 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__20();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__20);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__21 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__21();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__21);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__22 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__22();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__22);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__23 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__23();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__23);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__24 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__24();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__24);
lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__25 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__25();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1___closed__25);
lp_mathlib_MultilinearMap_mk_x27___auto__1 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__1();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__1);
lp_mathlib_MultilinearMap_mk_x27___auto__3 = _init_lp_mathlib_MultilinearMap_mk_x27___auto__3();
lean_mark_persistent(lp_mathlib_MultilinearMap_mk_x27___auto__3);
lp_mathlib_MultilinearMap_coeAddMonoidHom___closed__0 = _init_lp_mathlib_MultilinearMap_coeAddMonoidHom___closed__0();
lean_mark_persistent(lp_mathlib_MultilinearMap_coeAddMonoidHom___closed__0);
lp_mathlib_MultilinearMap_smulRight___redArg___closed__0 = _init_lp_mathlib_MultilinearMap_smulRight___redArg___closed__0();
lean_mark_persistent(lp_mathlib_MultilinearMap_smulRight___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
