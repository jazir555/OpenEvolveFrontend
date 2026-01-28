// Lean compiler output
// Module: Mathlib.Algebra.Ring.Subring.Basic
// Imports: public import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.Group.Subgroup.Basic public import Mathlib.Algebra.Ring.Subring.Defs public import Mathlib.Algebra.Ring.Subsemiring.Basic public import Mathlib.RingTheory.NonUnitalSubring.Defs public import Mathlib.Data.Set.Finite.Basic
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
lean_object* lp_mathlib_RingHom_codRestrict___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_Rat_castRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Subring_inclusion___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulSemiringActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerCongr___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Set_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_RingEquiv_subringCongr___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_range___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulSemiringActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubringClass_subtype___lam__0(lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg(lean_object*);
lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
lean_object* lp_mathlib_RingEquiv_subsemiringMap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionWithZeroSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingEquiv_instEquivLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_center(lean_object*, lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionWithZeroSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulSemiringActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instModuleSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instDistribMulActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_closure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Subring_instCompleteLattice___closed__0;
lean_object* lp_mathlib_RingHom_restrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulWithZeroSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg(lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulWithZeroSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instModuleSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerToMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv(lean_object*, lean_object*);
static lean_object* lp_mathlib_Subring_prodEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivisionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_SubringClass_subtype___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centralizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionSubtypeMem___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerToMulOpposite___redArg(lean_object*);
static lean_object* lp_mathlib_Subring_instCompleteLattice___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instDistribMulActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Submonoid_instSMulSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureCommRingOfComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCommRingSubtypeMemCenter(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv___redArg___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeProdEquivProd(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subsemiring_center_commSemiring___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Subsemiring_topEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_SetLike_instPartialOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulDistribMulActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_range(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubringClass_toRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulDistribMulActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Subring_inclusion___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subring_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_center___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocus___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCommRingSubtypeMemCenter___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centralizer(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Subring_instCompleteLattice___closed__2;
static lean_object* lp_mathlib_Subring_instCompleteLattice___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureCommRingOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instModuleSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCommRingSubtypeMemCenter___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulDistribMulActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Set_fintypeUniv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instDistribMulActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_closure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulWithZeroSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionWithZeroSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instTop(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instTop___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_instTop(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_4 = lp_mathlib_Subsemiring_topEquiv(lean_box(0), x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_topEquiv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_topEquiv(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Subring_topEquiv___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Set_fintypeUniv___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_fintypeUniv___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subring_instFintypeSubtypeMemTop(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_comap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_comap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subring_comap(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_map___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subring_map(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_range(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_range___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RingHom_range(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_RingHom_fintypeRange___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lp_mathlib_PLift_fintype___redArg(x_1);
x_6 = lp_mathlib_Set_fintypeRange___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_RingHom_fintypeRange___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_RingHom_fintypeRange(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instBot(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instBot___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_instBot(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInhabited___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_instInhabited(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subring_instMin___lam__0), 2, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_instMin(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subring_instInfSet___lam__0), 1, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_instInfSet(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Subring_instCompleteLattice___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lp_mathlib_SetLike_instPartialOrder(lean_box(0), lean_box(0), x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Subring_instCompleteLattice___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Subring_instInfSet___lam__0), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Subring_instCompleteLattice___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Subring_instCompleteLattice___closed__1;
x_2 = lp_mathlib_Subring_instCompleteLattice___closed__0;
x_3 = lp_mathlib_completeLatticeOfInf___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Subring_instCompleteLattice___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lp_mathlib_Subring_instCompleteLattice___closed__2;
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 3);
lean_dec(x_6);
x_7 = !lean_is_exclusive(x_5);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_5, 1);
lean_dec(x_8);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Subring_instCompleteLattice___lam__0), 2, 0);
lean_ctor_set(x_5, 1, x_9);
x_10 = lp_mathlib_Subring_instCompleteLattice___closed__3;
lean_ctor_set(x_3, 3, x_10);
return x_3;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_5, 0);
lean_inc(x_11);
lean_dec(x_5);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Subring_instCompleteLattice___lam__0), 2, 0);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
x_14 = lp_mathlib_Subring_instCompleteLattice___closed__3;
lean_ctor_set(x_3, 3, x_14);
lean_ctor_set(x_3, 0, x_13);
return x_3;
}
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_15 = lean_ctor_get(x_3, 0);
x_16 = lean_ctor_get(x_3, 1);
x_17 = lean_ctor_get(x_3, 2);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_15);
lean_dec(x_3);
x_18 = lean_ctor_get(x_15, 0);
lean_inc_ref(x_18);
if (lean_is_exclusive(x_15)) {
 lean_ctor_release(x_15, 0);
 lean_ctor_release(x_15, 1);
 x_19 = x_15;
} else {
 lean_dec_ref(x_15);
 x_19 = lean_box(0);
}
x_20 = lean_alloc_closure((void*)(lp_mathlib_Subring_instCompleteLattice___lam__0), 2, 0);
if (lean_is_scalar(x_19)) {
 x_21 = lean_alloc_ctor(0, 2, 0);
} else {
 x_21 = x_19;
}
lean_ctor_set(x_21, 0, x_18);
lean_ctor_set(x_21, 1, x_20);
x_22 = lp_mathlib_Subring_instCompleteLattice___closed__3;
x_23 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_16);
lean_ctor_set(x_23, 2, x_17);
lean_ctor_set(x_23, 3, x_22);
return x_23;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_instCompleteLattice(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_center(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_center___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_center(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
lean_inc(x_1);
lean_inc(x_2);
lean_inc(x_4);
x_5 = lean_apply_2(x_1, x_4, x_2);
x_6 = lean_apply_2(x_1, x_2, x_4);
x_7 = lean_apply_2(x_3, x_5, x_6);
x_8 = lean_unbox(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Subring_decidableMemCenter___redArg___lam__0(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_5 = lp_mathlib_Ring_toNonAssocRing___redArg(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_6);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Subring_decidableMemCenter___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_10, 0, x_9);
lean_closure_set(x_10, 1, x_4);
lean_closure_set(x_10, 2, x_2);
x_11 = lp_mathlib_Fintype_decidableForallFintype___redArg(x_10, x_3);
return x_11;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = lp_mathlib_Subring_decidableMemCenter___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_Subring_decidableMemCenter(x_1, x_2, x_3, x_4, x_5);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Subring_decidableMemCenter___redArg(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCommRingSubtypeMemCenter___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCommRingSubtypeMemCenter___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_4 = lp_mathlib_Subsemiring_center_commSemiring___redArg(x_2);
x_5 = lp_mathlib_SubringClass_toRing___redArg(x_1);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_5, 3);
lean_dec(x_7);
x_8 = lean_ctor_get(x_5, 0);
lean_dec(x_8);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Subring_instCommRingSubtypeMemCenter___redArg___lam__0), 3, 1);
lean_closure_set(x_9, 0, x_3);
lean_ctor_set(x_5, 3, x_9);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_5, 1);
x_11 = lean_ctor_get(x_5, 2);
x_12 = lean_ctor_get(x_5, 4);
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_5);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Subring_instCommRingSubtypeMemCenter___redArg___lam__0), 3, 1);
lean_closure_set(x_13, 0, x_3);
x_14 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_14, 0, x_4);
lean_ctor_set(x_14, 1, x_10);
lean_ctor_set(x_14, 2, x_11);
lean_ctor_set(x_14, 3, x_13);
lean_ctor_set(x_14, 4, x_12);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCommRingSubtypeMemCenter(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_instCommRingSubtypeMemCenter___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerCongr___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_4 = lp_mathlib_Ring_toNonAssocRing___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_5);
x_7 = lp_mathlib_Ring_toNonAssocRing___redArg(x_2);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
x_9 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_8);
x_10 = lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(x_6, x_9, x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subring_centerCongr___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerToMulOpposite___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_mathlib_Ring_toNonAssocRing___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_3);
x_5 = lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerToMulOpposite(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_centerToMulOpposite___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_1);
x_8 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_Rat_castRec___redArg(x_2, x_3, x_4, x_5);
x_12 = lean_apply_2(x_10, x_11, x_6);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_1);
x_7 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_NNRat_castRec___redArg(x_2, x_3, x_4);
x_11 = lean_apply_2(x_9, x_10, x_5);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_inc_ref(x_2);
x_3 = lp_mathlib_Subring_instCommRingSubtypeMemCenter___redArg(x_2);
x_4 = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(x_1);
x_5 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_4);
x_6 = lp_mathlib_GroupWithZero_toDivisionMonoid___redArg(x_5);
x_7 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_6);
lean_dec_ref(x_6);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(x_1);
x_10 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_10);
x_11 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_11);
x_12 = lean_ctor_get(x_9, 2);
lean_inc(x_12);
lean_dec_ref(x_9);
x_13 = lean_ctor_get(x_3, 4);
lean_inc(x_13);
x_14 = lean_ctor_get(x_10, 1);
lean_inc(x_14);
x_15 = lean_ctor_get(x_10, 2);
lean_inc(x_15);
lean_dec_ref(x_10);
x_16 = lean_ctor_get(x_11, 1);
lean_inc(x_16);
lean_dec_ref(x_11);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Subring_instField___redArg___lam__0), 2, 1);
lean_closure_set(x_17, 0, x_8);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Subring_instField___redArg___lam__1), 3, 1);
lean_closure_set(x_18, 0, x_12);
lean_inc_ref(x_18);
lean_inc(x_13);
lean_inc(x_15);
lean_inc_ref(x_2);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Subring_instField___redArg___lam__2), 6, 4);
lean_closure_set(x_19, 0, x_2);
lean_closure_set(x_19, 1, x_15);
lean_closure_set(x_19, 2, x_13);
lean_closure_set(x_19, 3, x_18);
lean_inc_ref(x_18);
lean_inc(x_15);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Subring_instField___redArg___lam__3), 5, 3);
lean_closure_set(x_20, 0, x_2);
lean_closure_set(x_20, 1, x_15);
lean_closure_set(x_20, 2, x_18);
lean_inc(x_16);
lean_inc(x_14);
x_21 = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_14);
lean_closure_set(x_21, 2, x_16);
lean_inc_ref(x_17);
x_22 = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(x_22, 0, lean_box(0));
lean_closure_set(x_22, 1, x_14);
lean_closure_set(x_22, 2, x_16);
lean_closure_set(x_22, 3, x_17);
lean_closure_set(x_22, 4, x_21);
lean_inc_ref(x_18);
lean_inc(x_15);
x_23 = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(x_23, 0, lean_box(0));
lean_closure_set(x_23, 1, x_15);
lean_closure_set(x_23, 2, x_18);
lean_inc_ref(x_18);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(x_24, 0, lean_box(0));
lean_closure_set(x_24, 1, x_15);
lean_closure_set(x_24, 2, x_13);
lean_closure_set(x_24, 3, x_18);
x_25 = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(x_25, 0, x_3);
lean_ctor_set(x_25, 1, x_17);
lean_ctor_set(x_25, 2, x_18);
lean_ctor_set(x_25, 3, x_22);
lean_ctor_set(x_25, 4, x_23);
lean_ctor_set(x_25, 5, x_24);
lean_ctor_set(x_25, 6, x_20);
lean_ctor_set(x_25, 7, x_19);
return x_25;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_instField___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centralizer(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centralizer___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subring_centralizer(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_closure(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_closure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subring_closure(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureCommRingOfComm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SubringClass_toRing___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureCommRingOfComm___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SubringClass_toRing___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subring_gi___lam__0), 2, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subring_gi(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_prod(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_prod___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subring_prod(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
static lean_object* _init_lp_mathlib_Subring_prodEquiv___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_subtypeProdEquivProd(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_prodEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subring_prodEquiv___closed__0;
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_prodEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subring_prodEquiv(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RingHom_codRestrict___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RingHom_codRestrict___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RingHom_rangeRestrict(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocus(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocus___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_RingHom_eqLocus(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
static lean_object* _init_lp_mathlib_Subring_inclusion___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_SubringClass_subtype___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Subring_inclusion___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Subring_inclusion___closed__0;
x_2 = lp_mathlib_RingHom_codRestrict___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_inclusion(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subring_inclusion___closed__1;
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_inclusion___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subring_inclusion(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_subringCongr___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RingEquiv_subringCongr___closed__0;
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringCongr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RingEquiv_subringCongr(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_RingHom_codRestrict___redArg(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_SubringClass_subtype___lam__0(x_2);
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_RingEquiv_ofLeftInverse___redArg(x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_RingEquiv_ofLeftInverse(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_4);
x_6 = lean_ctor_get(x_2, 0);
x_7 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_6);
x_8 = lp_mathlib_RingEquiv_subsemiringMap___redArg(x_5, x_7, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_RingEquiv_subringMap___redArg(x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_RingEquiv_subringMap(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RingEquiv_subringMap___redArg(x_1, x_2, x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_RingHom_restrict___redArg(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec_ref(x_5);
x_8 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_2);
x_9 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_9, 1);
lean_inc(x_11);
lean_dec_ref(x_9);
x_12 = lp_mathlib_RingEquiv_instEquivLike(lean_box(0), lean_box(0), x_6, x_10, x_7, x_11);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_RingEquiv_instEquivLike(lean_box(0), lean_box(0), x_10, x_6, x_11, x_7);
lean_dec(x_7);
lean_dec(x_11);
lean_dec(x_6);
lean_dec(x_10);
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_16 = lean_ctor_get(x_14, 0);
x_17 = lean_ctor_get(x_14, 1);
lean_dec(x_17);
lean_inc_ref(x_3);
x_18 = lean_apply_1(x_13, x_3);
x_19 = lp_mathlib_RingHom_restrict___redArg(x_18);
x_20 = lp_mathlib_Equiv_symm___redArg(x_3);
x_21 = lean_apply_1(x_16, x_20);
x_22 = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_restrict___redArg___lam__0), 2, 1);
lean_closure_set(x_22, 0, x_21);
lean_ctor_set(x_14, 1, x_22);
lean_ctor_set(x_14, 0, x_19);
return x_14;
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_23 = lean_ctor_get(x_14, 0);
lean_inc(x_23);
lean_dec(x_14);
lean_inc_ref(x_3);
x_24 = lean_apply_1(x_13, x_3);
x_25 = lp_mathlib_RingHom_restrict___redArg(x_24);
x_26 = lp_mathlib_Equiv_symm___redArg(x_3);
x_27 = lean_apply_1(x_23, x_26);
x_28 = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_restrict___redArg___lam__0), 2, 1);
lean_closure_set(x_28, 0, x_27);
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_25);
lean_ctor_set(x_29, 1, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_RingEquiv_restrict___redArg(x_3, x_4, x_11);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_RingEquiv_restrict(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_13);
lean_dec(x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Submonoid_instSMulSubtypeMem___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Submonoid_instSMulSubtypeMem___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subring_instSMulSubtypeMem(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subring_instMulActionSubtypeMem(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instDistribMulActionSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instDistribMulActionSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instDistribMulActionSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subring_instDistribMulActionSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulDistribMulActionSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulDistribMulActionSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulDistribMulActionSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subring_instMulDistribMulActionSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulWithZeroSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Submonoid_instSMulSubtypeMem___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulWithZeroSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Submonoid_instSMulSubtypeMem___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSMulWithZeroSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subring_instSMulWithZeroSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionWithZeroSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionWithZeroSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulActionWithZeroSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subring_instMulActionWithZeroSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instModuleSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instModuleSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instModuleSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subring_instModuleSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulSemiringActionSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulSemiringActionSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMulSemiringActionSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subring_instMulSemiringActionSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Subring_instCompleteLattice___closed__0 = _init_lp_mathlib_Subring_instCompleteLattice___closed__0();
lean_mark_persistent(lp_mathlib_Subring_instCompleteLattice___closed__0);
lp_mathlib_Subring_instCompleteLattice___closed__1 = _init_lp_mathlib_Subring_instCompleteLattice___closed__1();
lean_mark_persistent(lp_mathlib_Subring_instCompleteLattice___closed__1);
lp_mathlib_Subring_instCompleteLattice___closed__2 = _init_lp_mathlib_Subring_instCompleteLattice___closed__2();
lean_mark_persistent(lp_mathlib_Subring_instCompleteLattice___closed__2);
lp_mathlib_Subring_instCompleteLattice___closed__3 = _init_lp_mathlib_Subring_instCompleteLattice___closed__3();
lean_mark_persistent(lp_mathlib_Subring_instCompleteLattice___closed__3);
lp_mathlib_Subring_prodEquiv___closed__0 = _init_lp_mathlib_Subring_prodEquiv___closed__0();
lean_mark_persistent(lp_mathlib_Subring_prodEquiv___closed__0);
lp_mathlib_Subring_inclusion___closed__0 = _init_lp_mathlib_Subring_inclusion___closed__0();
lean_mark_persistent(lp_mathlib_Subring_inclusion___closed__0);
lp_mathlib_Subring_inclusion___closed__1 = _init_lp_mathlib_Subring_inclusion___closed__1();
lean_mark_persistent(lp_mathlib_Subring_inclusion___closed__1);
lp_mathlib_RingEquiv_subringCongr___closed__0 = _init_lp_mathlib_RingEquiv_subringCongr___closed__0();
lean_mark_persistent(lp_mathlib_RingEquiv_subringCongr___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
