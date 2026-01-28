// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Unitization
// Imports: public import Init public import Mathlib.Algebra.Algebra.Defs public import Mathlib.Algebra.Algebra.NonUnitalHom public import Mathlib.Algebra.Star.Module public import Mathlib.Algebra.Star.NonUnitalSubalgebra public import Mathlib.LinearAlgebra.Prod public import Mathlib.Tactic.Abel
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
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalAlgHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAlgebra___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Unitization_fstHom___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Unitization_lift___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starLift___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_addEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_snd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fst___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instRing___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMonoid___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toAlgHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_sndHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instAddZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddGroup___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Unitization_inrRangeEquiv___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalAlgHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_snd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalAlgHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starLift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrHom___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_unaryCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starMap___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalStarAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fst___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarAddMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrRangeEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instModule___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_subNegMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrRangeEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAdd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulOneClass___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fstHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Int_castDef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommMonoid___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulOneClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrRangeEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulOneClass___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommSemiring_toCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instInhabited___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starLift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instSMul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddZeroClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fst(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_snd___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalStarAlgHom___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_Unitization_addEquiv___closed__0;
lean_object* lp_mathlib_NonUnitalStarAlgHom_instFunLike___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_lift___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instOne(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_StarAlgEquiv_ofLeftInverse_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalStarAlgHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddGroup(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instAddMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inlRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCoeTCOfZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNeg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarAddMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instAdd___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Unitization_sndHom___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_sndHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instNeg___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulOneClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fstHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instDistribMulAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAdd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_lift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStar___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toAlgHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalStarAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarRing___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNonAssocRing___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inlRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStar(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStar___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fst___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCoeTCOfZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_addEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instSMul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNeg___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Algebra_id___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrRangeEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toAlgHom___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starLift___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_snd___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inlRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommGroup___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inl___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inr___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCoeTCOfZero(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Unitization_inr), 4, 3);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, lean_box(0));
lean_closure_set(x_4, 2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCoeTCOfZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Unitization_inr), 4, 3);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fst(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fst___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fst___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Unitization_fst(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fst___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Unitization_fst___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_snd(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_snd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_snd___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Unitization_snd(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_snd___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Unitization_snd___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instInhabited(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instInhabited___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instZero___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instAdd___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAdd___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_instAdd___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNeg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instNeg___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNeg___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_instNeg___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddSemigroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instAdd___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddSemigroup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_instAdd___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddZeroClass(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instAddZeroClass___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddZeroClass___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_instAddZeroClass___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instAddMonoid___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddMonoid___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_instAddMonoid___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_subNegMonoid___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddGroup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_subNegMonoid___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommSemigroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instAdd___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommSemigroup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_instAdd___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instAddMonoid___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommMonoid___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_instAddMonoid___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_subNegMonoid___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAddCommGroup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_subNegMonoid___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instSMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Prod_instSMul___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instSMul___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_instSMul___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulAction(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Prod_instSMul___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulAction___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_instSMul___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulAction___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Unitization_instMulAction(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instDistribMulAction(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Prod_instSMul___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instDistribMulAction___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_instSMul___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instDistribMulAction___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Unitization_instDistribMulAction(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instModule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Prod_instSMul___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instModule___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Prod_instSMul___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instModule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Unitization_instModule(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_9;
}
}
static lean_object* _init_lp_mathlib_Unitization_addEquiv___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_addEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Unitization_addEquiv___closed__0;
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_addEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Unitization_addEquiv(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_3);
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Unitization_inr), 4, 3);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_inrHom___redArg(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_inrHom(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrHom___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Unitization_inrHom___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Unitization_sndHom___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Unitization_snd___boxed), 3, 2);
lean_closure_set(x_1, 0, lean_box(0));
lean_closure_set(x_1, 1, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_sndHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_sndHom___closed__0;
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_sndHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_sndHom(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instOne___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec_ref(x_5);
x_9 = !lean_is_exclusive(x_6);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_10 = lean_ctor_get(x_6, 0);
x_11 = lean_ctor_get(x_6, 1);
lean_inc(x_10);
lean_inc(x_7);
x_12 = lean_apply_2(x_1, x_7, x_10);
lean_inc(x_2);
lean_inc(x_11);
x_13 = lean_apply_2(x_2, x_7, x_11);
lean_inc(x_8);
x_14 = lean_apply_2(x_2, x_10, x_8);
lean_inc(x_3);
x_15 = lean_apply_2(x_3, x_13, x_14);
x_16 = lean_apply_2(x_4, x_8, x_11);
x_17 = lean_apply_2(x_3, x_15, x_16);
lean_ctor_set(x_6, 1, x_17);
lean_ctor_set(x_6, 0, x_12);
return x_6;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_18 = lean_ctor_get(x_6, 0);
x_19 = lean_ctor_get(x_6, 1);
lean_inc(x_19);
lean_inc(x_18);
lean_dec(x_6);
lean_inc(x_18);
lean_inc(x_7);
x_20 = lean_apply_2(x_1, x_7, x_18);
lean_inc(x_2);
lean_inc(x_19);
x_21 = lean_apply_2(x_2, x_7, x_19);
lean_inc(x_8);
x_22 = lean_apply_2(x_2, x_18, x_8);
lean_inc(x_3);
x_23 = lean_apply_2(x_3, x_21, x_22);
x_24 = lean_apply_2(x_4, x_8, x_19);
x_25 = lean_apply_2(x_3, x_23, x_24);
x_26 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_26, 0, x_20);
lean_ctor_set(x_26, 1, x_25);
return x_26;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMul___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instMul___redArg___lam__0), 6, 4);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_4);
lean_closure_set(x_5, 2, x_2);
lean_closure_set(x_5, 3, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Unitization_instMul___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulOneClass___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_2);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_2);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_2);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
lean_ctor_set(x_10, 1, x_9);
lean_ctor_set(x_10, 0, x_6);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instMul___redArg___lam__0), 6, 4);
lean_closure_set(x_14, 0, x_7);
lean_closure_set(x_14, 1, x_3);
lean_closure_set(x_14, 2, x_13);
lean_closure_set(x_14, 3, x_12);
lean_ctor_set(x_4, 1, x_14);
lean_ctor_set(x_4, 0, x_10);
return x_4;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_15 = lean_ctor_get(x_10, 0);
x_16 = lean_ctor_get(x_10, 1);
lean_inc(x_16);
lean_inc(x_15);
lean_dec(x_10);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_6);
lean_ctor_set(x_17, 1, x_9);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instMul___redArg___lam__0), 6, 4);
lean_closure_set(x_18, 0, x_7);
lean_closure_set(x_18, 1, x_3);
lean_closure_set(x_18, 2, x_16);
lean_closure_set(x_18, 3, x_15);
lean_ctor_set(x_4, 1, x_18);
lean_ctor_set(x_4, 0, x_17);
return x_4;
}
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_19 = lean_ctor_get(x_4, 0);
x_20 = lean_ctor_get(x_4, 1);
lean_inc(x_20);
lean_inc(x_19);
lean_dec(x_4);
lean_inc_ref(x_2);
x_21 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_2);
x_22 = lean_ctor_get(x_21, 1);
lean_inc(x_22);
lean_dec_ref(x_21);
x_23 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_2);
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
x_25 = lean_ctor_get(x_23, 1);
lean_inc(x_25);
if (lean_is_exclusive(x_23)) {
 lean_ctor_release(x_23, 0);
 lean_ctor_release(x_23, 1);
 x_26 = x_23;
} else {
 lean_dec_ref(x_23);
 x_26 = lean_box(0);
}
if (lean_is_scalar(x_26)) {
 x_27 = lean_alloc_ctor(0, 2, 0);
} else {
 x_27 = x_26;
}
lean_ctor_set(x_27, 0, x_19);
lean_ctor_set(x_27, 1, x_22);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instMul___redArg___lam__0), 6, 4);
lean_closure_set(x_28, 0, x_20);
lean_closure_set(x_28, 1, x_3);
lean_closure_set(x_28, 2, x_25);
lean_closure_set(x_28, 3, x_24);
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_27);
lean_ctor_set(x_29, 1, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulOneClass(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_instMulOneClass___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulOneClass___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_instMulOneClass(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMulOneClass___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Unitization_instMulOneClass___redArg(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNonAssocSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
lean_inc_ref(x_1);
x_4 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
lean_inc_ref(x_2);
x_6 = lp_mathlib_Unitization_instMulOneClass___redArg(x_5, x_2, x_3);
lean_dec_ref(x_5);
x_7 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
lean_dec_ref(x_1);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_7, 2);
lean_dec(x_10);
x_11 = lean_ctor_get(x_7, 1);
lean_dec(x_11);
x_12 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_9);
x_13 = !lean_is_exclusive(x_2);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_14 = lean_ctor_get(x_2, 0);
x_15 = lean_ctor_get(x_2, 1);
lean_dec(x_15);
x_16 = lp_mathlib_Prod_instAddMonoid___redArg(x_12, x_14);
x_17 = lean_ctor_get(x_6, 0);
lean_inc(x_17);
x_18 = lean_ctor_get(x_6, 1);
lean_inc(x_18);
lean_dec_ref(x_6);
lean_inc_ref(x_16);
lean_ctor_set(x_2, 1, x_18);
lean_ctor_set(x_2, 0, x_16);
x_19 = lean_ctor_get(x_16, 0);
lean_inc(x_19);
x_20 = lean_ctor_get(x_16, 1);
lean_inc(x_20);
lean_dec_ref(x_16);
lean_inc(x_17);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_17);
lean_closure_set(x_21, 2, x_20);
lean_closure_set(x_21, 3, x_19);
lean_ctor_set(x_7, 2, x_21);
lean_ctor_set(x_7, 1, x_17);
lean_ctor_set(x_7, 0, x_2);
return x_7;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_22 = lean_ctor_get(x_2, 0);
lean_inc(x_22);
lean_dec(x_2);
x_23 = lp_mathlib_Prod_instAddMonoid___redArg(x_12, x_22);
x_24 = lean_ctor_get(x_6, 0);
lean_inc(x_24);
x_25 = lean_ctor_get(x_6, 1);
lean_inc(x_25);
lean_dec_ref(x_6);
lean_inc_ref(x_23);
x_26 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_26, 0, x_23);
lean_ctor_set(x_26, 1, x_25);
x_27 = lean_ctor_get(x_23, 0);
lean_inc(x_27);
x_28 = lean_ctor_get(x_23, 1);
lean_inc(x_28);
lean_dec_ref(x_23);
lean_inc(x_24);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, x_24);
lean_closure_set(x_29, 2, x_28);
lean_closure_set(x_29, 3, x_27);
lean_ctor_set(x_7, 2, x_29);
lean_ctor_set(x_7, 1, x_24);
lean_ctor_set(x_7, 0, x_26);
return x_7;
}
}
else
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_30 = lean_ctor_get(x_7, 0);
lean_inc(x_30);
lean_dec(x_7);
x_31 = lean_ctor_get(x_30, 0);
lean_inc_ref(x_31);
lean_dec_ref(x_30);
x_32 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_32);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 x_33 = x_2;
} else {
 lean_dec_ref(x_2);
 x_33 = lean_box(0);
}
x_34 = lp_mathlib_Prod_instAddMonoid___redArg(x_31, x_32);
x_35 = lean_ctor_get(x_6, 0);
lean_inc(x_35);
x_36 = lean_ctor_get(x_6, 1);
lean_inc(x_36);
lean_dec_ref(x_6);
lean_inc_ref(x_34);
if (lean_is_scalar(x_33)) {
 x_37 = lean_alloc_ctor(0, 2, 0);
} else {
 x_37 = x_33;
}
lean_ctor_set(x_37, 0, x_34);
lean_ctor_set(x_37, 1, x_36);
x_38 = lean_ctor_get(x_34, 0);
lean_inc(x_38);
x_39 = lean_ctor_get(x_34, 1);
lean_inc(x_39);
lean_dec_ref(x_34);
lean_inc(x_35);
x_40 = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(x_40, 0, lean_box(0));
lean_closure_set(x_40, 1, x_35);
lean_closure_set(x_40, 2, x_39);
lean_closure_set(x_40, 3, x_38);
x_41 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_41, 0, x_37);
lean_ctor_set(x_41, 1, x_35);
lean_ctor_set(x_41, 2, x_40);
return x_41;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNonAssocSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_instNonAssocSemiring___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMonoid___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_Unitization_instMulOneClass___redArg(x_1, x_2, x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_dec_ref(x_4);
lean_inc(x_5);
lean_inc(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_6);
lean_closure_set(x_7, 2, x_5);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_5);
lean_ctor_set(x_8, 2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_instMonoid___redArg(x_3, x_4, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_instMonoid(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instMonoid___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Unitization_instMonoid___redArg(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_instMonoid___redArg(x_3, x_4, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommMonoid___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Unitization_instMonoid___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommMonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_instCommMonoid(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommMonoid___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Unitization_instCommMonoid___redArg(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_4 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_1);
lean_inc(x_3);
lean_inc_ref(x_2);
x_5 = lp_mathlib_Unitization_instMonoid___redArg(x_4, x_2, x_3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_6 = lp_mathlib_Unitization_instNonAssocSemiring___redArg(x_1, x_2, x_3);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_6, 2);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = !lean_is_exclusive(x_7);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_10 = lean_ctor_get(x_7, 1);
lean_dec(x_10);
x_11 = lean_ctor_get(x_5, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_5, 1);
lean_inc(x_12);
lean_dec_ref(x_5);
lean_ctor_set(x_7, 1, x_11);
x_13 = lp_mathlib_Unitization_instMulOneClass___redArg(x_4, x_2, x_3);
lean_dec_ref(x_4);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_13, 1);
lean_inc(x_15);
lean_dec_ref(x_13);
x_16 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_16, 0, lean_box(0));
lean_closure_set(x_16, 1, x_15);
lean_closure_set(x_16, 2, x_14);
x_17 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_17, 0, x_7);
lean_ctor_set(x_17, 1, x_12);
lean_ctor_set(x_17, 2, x_8);
lean_ctor_set(x_17, 3, x_16);
return x_17;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_18 = lean_ctor_get(x_7, 0);
lean_inc(x_18);
lean_dec(x_7);
x_19 = lean_ctor_get(x_5, 0);
lean_inc(x_19);
x_20 = lean_ctor_get(x_5, 1);
lean_inc(x_20);
lean_dec_ref(x_5);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_18);
lean_ctor_set(x_21, 1, x_19);
x_22 = lp_mathlib_Unitization_instMulOneClass___redArg(x_4, x_2, x_3);
lean_dec_ref(x_4);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
x_24 = lean_ctor_get(x_22, 1);
lean_inc(x_24);
lean_dec_ref(x_22);
x_25 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, x_24);
lean_closure_set(x_25, 2, x_23);
x_26 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_26, 0, x_21);
lean_ctor_set(x_26, 1, x_20);
lean_ctor_set(x_26, 2, x_8);
lean_ctor_set(x_26, 3, x_25);
return x_26;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_instSemiring___redArg(x_3, x_4, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_4 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_1);
lean_inc(x_3);
lean_inc_ref(x_2);
x_5 = lp_mathlib_Unitization_instMonoid___redArg(x_4, x_2, x_3);
lean_inc(x_3);
lean_inc_ref(x_2);
x_6 = lp_mathlib_Unitization_instNonAssocSemiring___redArg(x_1, x_2, x_3);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_6, 2);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = !lean_is_exclusive(x_7);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_10 = lean_ctor_get(x_7, 1);
lean_dec(x_10);
x_11 = lean_ctor_get(x_5, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_5, 1);
lean_inc(x_12);
lean_dec_ref(x_5);
lean_ctor_set(x_7, 1, x_11);
x_13 = lp_mathlib_Unitization_instMulOneClass___redArg(x_4, x_2, x_3);
lean_dec_ref(x_4);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_13, 1);
lean_inc(x_15);
lean_dec_ref(x_13);
x_16 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_16, 0, lean_box(0));
lean_closure_set(x_16, 1, x_15);
lean_closure_set(x_16, 2, x_14);
x_17 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_17, 0, x_7);
lean_ctor_set(x_17, 1, x_12);
lean_ctor_set(x_17, 2, x_8);
lean_ctor_set(x_17, 3, x_16);
return x_17;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_18 = lean_ctor_get(x_7, 0);
lean_inc(x_18);
lean_dec(x_7);
x_19 = lean_ctor_get(x_5, 0);
lean_inc(x_19);
x_20 = lean_ctor_get(x_5, 1);
lean_inc(x_20);
lean_dec_ref(x_5);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_18);
lean_ctor_set(x_21, 1, x_19);
x_22 = lp_mathlib_Unitization_instMulOneClass___redArg(x_4, x_2, x_3);
lean_dec_ref(x_4);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
x_24 = lean_ctor_get(x_22, 1);
lean_inc(x_24);
lean_dec_ref(x_22);
x_25 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, x_24);
lean_closure_set(x_25, 2, x_23);
x_26 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_26, 0, x_21);
lean_ctor_set(x_26, 1, x_20);
lean_ctor_set(x_26, 2, x_8);
lean_ctor_set(x_26, 3, x_25);
return x_26;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_instCommSemiring___redArg(x_3, x_4, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNonAssocRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
lean_inc_ref(x_1);
x_4 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_5 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_5);
x_6 = lp_mathlib_Prod_subNegMonoid___redArg(x_4, x_5);
x_7 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_1);
x_8 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_2);
x_9 = lp_mathlib_Unitization_instNonAssocSemiring___redArg(x_7, x_8, x_3);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
x_11 = lean_ctor_get(x_9, 1);
lean_inc(x_11);
x_12 = lean_ctor_get(x_9, 2);
lean_inc(x_12);
lean_dec_ref(x_9);
x_13 = !lean_is_exclusive(x_10);
if (x_13 == 0)
{
lean_object* x_14; uint8_t x_15; 
x_14 = lean_ctor_get(x_10, 0);
lean_dec(x_14);
lean_inc_ref(x_6);
lean_ctor_set(x_10, 0, x_6);
x_15 = !lean_is_exclusive(x_6);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_16 = lean_ctor_get(x_6, 1);
x_17 = lean_ctor_get(x_6, 3);
lean_dec(x_17);
x_18 = lean_ctor_get(x_6, 2);
lean_dec(x_18);
x_19 = lean_ctor_get(x_6, 0);
lean_dec(x_19);
lean_inc(x_12);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, x_12);
lean_closure_set(x_20, 2, x_16);
lean_ctor_set(x_6, 3, x_20);
lean_ctor_set(x_6, 2, x_12);
lean_ctor_set(x_6, 1, x_11);
lean_ctor_set(x_6, 0, x_10);
return x_6;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_21 = lean_ctor_get(x_6, 1);
lean_inc(x_21);
lean_dec(x_6);
lean_inc(x_12);
x_22 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_22, 0, lean_box(0));
lean_closure_set(x_22, 1, x_12);
lean_closure_set(x_22, 2, x_21);
x_23 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_23, 0, x_10);
lean_ctor_set(x_23, 1, x_11);
lean_ctor_set(x_23, 2, x_12);
lean_ctor_set(x_23, 3, x_22);
return x_23;
}
}
else
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_24 = lean_ctor_get(x_10, 1);
lean_inc(x_24);
lean_dec(x_10);
lean_inc_ref(x_6);
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_6);
lean_ctor_set(x_25, 1, x_24);
x_26 = lean_ctor_get(x_6, 1);
lean_inc(x_26);
if (lean_is_exclusive(x_6)) {
 lean_ctor_release(x_6, 0);
 lean_ctor_release(x_6, 1);
 lean_ctor_release(x_6, 2);
 lean_ctor_release(x_6, 3);
 x_27 = x_6;
} else {
 lean_dec_ref(x_6);
 x_27 = lean_box(0);
}
lean_inc(x_12);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(x_28, 0, lean_box(0));
lean_closure_set(x_28, 1, x_12);
lean_closure_set(x_28, 2, x_26);
if (lean_is_scalar(x_27)) {
 x_29 = lean_alloc_ctor(0, 4, 0);
} else {
 x_29 = x_27;
}
lean_ctor_set(x_29, 0, x_25);
lean_ctor_set(x_29, 1, x_11);
lean_ctor_set(x_29, 2, x_12);
lean_ctor_set(x_29, 3, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instNonAssocRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_instNonAssocRing___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instRing___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
x_8 = lean_ctor_get(x_1, 3);
lean_inc(x_8);
lean_dec_ref(x_1);
lean_inc(x_3);
x_9 = lean_apply_2(x_2, x_3, x_6);
x_10 = lean_apply_2(x_8, x_3, x_7);
lean_ctor_set(x_4, 1, x_10);
lean_ctor_set(x_4, 0, x_9);
return x_4;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_4, 0);
x_12 = lean_ctor_get(x_4, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_4);
x_13 = lean_ctor_get(x_1, 3);
lean_inc(x_13);
lean_dec_ref(x_1);
lean_inc(x_3);
x_14 = lean_apply_2(x_2, x_3, x_11);
x_15 = lean_apply_2(x_13, x_3, x_12);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_14);
lean_ctor_set(x_16, 1, x_15);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
lean_inc_ref(x_1);
x_4 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_5 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_5);
lean_inc_ref(x_5);
x_6 = lp_mathlib_Prod_subNegMonoid___redArg(x_4, x_5);
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 3);
lean_inc(x_8);
lean_inc_ref(x_2);
x_9 = lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(x_2);
lean_inc(x_3);
lean_inc_ref(x_9);
lean_inc_ref(x_7);
x_10 = lp_mathlib_Unitization_instSemiring___redArg(x_7, x_9, x_3);
x_11 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_11);
x_12 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_12);
x_13 = lean_ctor_get(x_6, 1);
lean_inc(x_13);
x_14 = lean_ctor_get(x_6, 2);
lean_inc(x_14);
lean_dec_ref(x_6);
x_15 = !lean_is_exclusive(x_10);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; uint8_t x_18; 
x_16 = lean_ctor_get(x_10, 3);
lean_dec(x_16);
x_17 = lean_ctor_get(x_10, 0);
lean_dec(x_17);
x_18 = !lean_is_exclusive(x_11);
if (x_18 == 0)
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; uint8_t x_28; 
x_19 = lean_ctor_get(x_11, 0);
lean_dec(x_19);
lean_ctor_set(x_11, 0, x_12);
x_20 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_7);
lean_inc(x_3);
x_21 = lp_mathlib_Unitization_instMulOneClass___redArg(x_20, x_9, x_3);
lean_dec_ref(x_20);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
x_23 = lean_ctor_get(x_21, 1);
lean_inc(x_23);
lean_dec_ref(x_21);
x_24 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_24, 0, lean_box(0));
lean_closure_set(x_24, 1, x_23);
lean_closure_set(x_24, 2, x_22);
lean_ctor_set(x_10, 3, x_24);
x_25 = lp_mathlib_Unitization_instNonAssocRing___redArg(x_1, x_2, x_3);
x_26 = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(x_25);
x_27 = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(x_26);
lean_dec_ref(x_26);
x_28 = !lean_is_exclusive(x_27);
if (x_28 == 0)
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_29 = lean_ctor_get(x_27, 0);
x_30 = lean_ctor_get(x_27, 4);
lean_dec(x_30);
x_31 = lean_ctor_get(x_27, 3);
lean_dec(x_31);
x_32 = lean_ctor_get(x_27, 2);
lean_dec(x_32);
x_33 = lean_ctor_get(x_27, 1);
lean_dec(x_33);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instRing___redArg___lam__0), 4, 2);
lean_closure_set(x_34, 0, x_5);
lean_closure_set(x_34, 1, x_8);
lean_ctor_set(x_27, 4, x_29);
lean_ctor_set(x_27, 3, x_34);
lean_ctor_set(x_27, 2, x_14);
lean_ctor_set(x_27, 1, x_13);
lean_ctor_set(x_27, 0, x_10);
return x_27;
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_35 = lean_ctor_get(x_27, 0);
lean_inc(x_35);
lean_dec(x_27);
x_36 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instRing___redArg___lam__0), 4, 2);
lean_closure_set(x_36, 0, x_5);
lean_closure_set(x_36, 1, x_8);
x_37 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_37, 0, x_10);
lean_ctor_set(x_37, 1, x_13);
lean_ctor_set(x_37, 2, x_14);
lean_ctor_set(x_37, 3, x_36);
lean_ctor_set(x_37, 4, x_35);
return x_37;
}
}
else
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; 
x_38 = lean_ctor_get(x_11, 1);
lean_inc(x_38);
lean_dec(x_11);
x_39 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_39, 0, x_12);
lean_ctor_set(x_39, 1, x_38);
x_40 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_7);
lean_inc(x_3);
x_41 = lp_mathlib_Unitization_instMulOneClass___redArg(x_40, x_9, x_3);
lean_dec_ref(x_40);
x_42 = lean_ctor_get(x_41, 0);
lean_inc(x_42);
x_43 = lean_ctor_get(x_41, 1);
lean_inc(x_43);
lean_dec_ref(x_41);
x_44 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_44, 0, lean_box(0));
lean_closure_set(x_44, 1, x_43);
lean_closure_set(x_44, 2, x_42);
lean_ctor_set(x_10, 3, x_44);
lean_ctor_set(x_10, 0, x_39);
x_45 = lp_mathlib_Unitization_instNonAssocRing___redArg(x_1, x_2, x_3);
x_46 = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(x_45);
x_47 = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(x_46);
lean_dec_ref(x_46);
x_48 = lean_ctor_get(x_47, 0);
lean_inc(x_48);
if (lean_is_exclusive(x_47)) {
 lean_ctor_release(x_47, 0);
 lean_ctor_release(x_47, 1);
 lean_ctor_release(x_47, 2);
 lean_ctor_release(x_47, 3);
 lean_ctor_release(x_47, 4);
 x_49 = x_47;
} else {
 lean_dec_ref(x_47);
 x_49 = lean_box(0);
}
x_50 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instRing___redArg___lam__0), 4, 2);
lean_closure_set(x_50, 0, x_5);
lean_closure_set(x_50, 1, x_8);
if (lean_is_scalar(x_49)) {
 x_51 = lean_alloc_ctor(0, 5, 0);
} else {
 x_51 = x_49;
}
lean_ctor_set(x_51, 0, x_10);
lean_ctor_set(x_51, 1, x_13);
lean_ctor_set(x_51, 2, x_14);
lean_ctor_set(x_51, 3, x_50);
lean_ctor_set(x_51, 4, x_48);
return x_51;
}
}
else
{
lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; 
x_52 = lean_ctor_get(x_10, 1);
x_53 = lean_ctor_get(x_10, 2);
lean_inc(x_53);
lean_inc(x_52);
lean_dec(x_10);
x_54 = lean_ctor_get(x_11, 1);
lean_inc(x_54);
if (lean_is_exclusive(x_11)) {
 lean_ctor_release(x_11, 0);
 lean_ctor_release(x_11, 1);
 x_55 = x_11;
} else {
 lean_dec_ref(x_11);
 x_55 = lean_box(0);
}
if (lean_is_scalar(x_55)) {
 x_56 = lean_alloc_ctor(0, 2, 0);
} else {
 x_56 = x_55;
}
lean_ctor_set(x_56, 0, x_12);
lean_ctor_set(x_56, 1, x_54);
x_57 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_7);
lean_inc(x_3);
x_58 = lp_mathlib_Unitization_instMulOneClass___redArg(x_57, x_9, x_3);
lean_dec_ref(x_57);
x_59 = lean_ctor_get(x_58, 0);
lean_inc(x_59);
x_60 = lean_ctor_get(x_58, 1);
lean_inc(x_60);
lean_dec_ref(x_58);
x_61 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_61, 0, lean_box(0));
lean_closure_set(x_61, 1, x_60);
lean_closure_set(x_61, 2, x_59);
x_62 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_62, 0, x_56);
lean_ctor_set(x_62, 1, x_52);
lean_ctor_set(x_62, 2, x_53);
lean_ctor_set(x_62, 3, x_61);
x_63 = lp_mathlib_Unitization_instNonAssocRing___redArg(x_1, x_2, x_3);
x_64 = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(x_63);
x_65 = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(x_64);
lean_dec_ref(x_64);
x_66 = lean_ctor_get(x_65, 0);
lean_inc(x_66);
if (lean_is_exclusive(x_65)) {
 lean_ctor_release(x_65, 0);
 lean_ctor_release(x_65, 1);
 lean_ctor_release(x_65, 2);
 lean_ctor_release(x_65, 3);
 lean_ctor_release(x_65, 4);
 x_67 = x_65;
} else {
 lean_dec_ref(x_65);
 x_67 = lean_box(0);
}
x_68 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instRing___redArg___lam__0), 4, 2);
lean_closure_set(x_68, 0, x_5);
lean_closure_set(x_68, 1, x_8);
if (lean_is_scalar(x_67)) {
 x_69 = lean_alloc_ctor(0, 5, 0);
} else {
 x_69 = x_67;
}
lean_ctor_set(x_69, 0, x_62);
lean_ctor_set(x_69, 1, x_13);
lean_ctor_set(x_69, 2, x_14);
lean_ctor_set(x_69, 3, x_68);
lean_ctor_set(x_69, 4, x_66);
return x_69;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_instRing___redArg(x_3, x_4, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
lean_inc_ref(x_1);
x_4 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_5 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_5);
lean_inc_ref(x_5);
x_6 = lp_mathlib_Prod_subNegMonoid___redArg(x_4, x_5);
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 3);
lean_inc(x_8);
lean_inc_ref(x_2);
x_9 = lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring___redArg(x_2);
lean_inc(x_3);
lean_inc_ref(x_9);
lean_inc_ref(x_7);
x_10 = lp_mathlib_Unitization_instCommSemiring___redArg(x_7, x_9, x_3);
x_11 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_11);
x_12 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_12);
x_13 = lean_ctor_get(x_6, 1);
lean_inc(x_13);
x_14 = lean_ctor_get(x_6, 2);
lean_inc(x_14);
lean_dec_ref(x_6);
x_15 = !lean_is_exclusive(x_10);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; uint8_t x_18; 
x_16 = lean_ctor_get(x_10, 3);
lean_dec(x_16);
x_17 = lean_ctor_get(x_10, 0);
lean_dec(x_17);
x_18 = !lean_is_exclusive(x_11);
if (x_18 == 0)
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; uint8_t x_27; 
x_19 = lean_ctor_get(x_11, 0);
lean_dec(x_19);
lean_ctor_set(x_11, 0, x_12);
x_20 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_7);
lean_inc(x_3);
x_21 = lp_mathlib_Unitization_instMulOneClass___redArg(x_20, x_9, x_3);
lean_dec_ref(x_20);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
x_23 = lean_ctor_get(x_21, 1);
lean_inc(x_23);
lean_dec_ref(x_21);
x_24 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_24, 0, lean_box(0));
lean_closure_set(x_24, 1, x_23);
lean_closure_set(x_24, 2, x_22);
lean_ctor_set(x_10, 3, x_24);
x_25 = lp_mathlib_Unitization_instRing___redArg(x_1, x_2, x_3);
x_26 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_25);
x_27 = !lean_is_exclusive(x_26);
if (x_27 == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_28 = lean_ctor_get(x_26, 0);
x_29 = lean_ctor_get(x_26, 4);
lean_dec(x_29);
x_30 = lean_ctor_get(x_26, 3);
lean_dec(x_30);
x_31 = lean_ctor_get(x_26, 2);
lean_dec(x_31);
x_32 = lean_ctor_get(x_26, 1);
lean_dec(x_32);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instRing___redArg___lam__0), 4, 2);
lean_closure_set(x_33, 0, x_5);
lean_closure_set(x_33, 1, x_8);
lean_ctor_set(x_26, 4, x_28);
lean_ctor_set(x_26, 3, x_33);
lean_ctor_set(x_26, 2, x_14);
lean_ctor_set(x_26, 1, x_13);
lean_ctor_set(x_26, 0, x_10);
return x_26;
}
else
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_34 = lean_ctor_get(x_26, 0);
lean_inc(x_34);
lean_dec(x_26);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instRing___redArg___lam__0), 4, 2);
lean_closure_set(x_35, 0, x_5);
lean_closure_set(x_35, 1, x_8);
x_36 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_36, 0, x_10);
lean_ctor_set(x_36, 1, x_13);
lean_ctor_set(x_36, 2, x_14);
lean_ctor_set(x_36, 3, x_35);
lean_ctor_set(x_36, 4, x_34);
return x_36;
}
}
else
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; 
x_37 = lean_ctor_get(x_11, 1);
lean_inc(x_37);
lean_dec(x_11);
x_38 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_38, 0, x_12);
lean_ctor_set(x_38, 1, x_37);
x_39 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_7);
lean_inc(x_3);
x_40 = lp_mathlib_Unitization_instMulOneClass___redArg(x_39, x_9, x_3);
lean_dec_ref(x_39);
x_41 = lean_ctor_get(x_40, 0);
lean_inc(x_41);
x_42 = lean_ctor_get(x_40, 1);
lean_inc(x_42);
lean_dec_ref(x_40);
x_43 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_43, 0, lean_box(0));
lean_closure_set(x_43, 1, x_42);
lean_closure_set(x_43, 2, x_41);
lean_ctor_set(x_10, 3, x_43);
lean_ctor_set(x_10, 0, x_38);
x_44 = lp_mathlib_Unitization_instRing___redArg(x_1, x_2, x_3);
x_45 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_44);
x_46 = lean_ctor_get(x_45, 0);
lean_inc(x_46);
if (lean_is_exclusive(x_45)) {
 lean_ctor_release(x_45, 0);
 lean_ctor_release(x_45, 1);
 lean_ctor_release(x_45, 2);
 lean_ctor_release(x_45, 3);
 lean_ctor_release(x_45, 4);
 x_47 = x_45;
} else {
 lean_dec_ref(x_45);
 x_47 = lean_box(0);
}
x_48 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instRing___redArg___lam__0), 4, 2);
lean_closure_set(x_48, 0, x_5);
lean_closure_set(x_48, 1, x_8);
if (lean_is_scalar(x_47)) {
 x_49 = lean_alloc_ctor(0, 5, 0);
} else {
 x_49 = x_47;
}
lean_ctor_set(x_49, 0, x_10);
lean_ctor_set(x_49, 1, x_13);
lean_ctor_set(x_49, 2, x_14);
lean_ctor_set(x_49, 3, x_48);
lean_ctor_set(x_49, 4, x_46);
return x_49;
}
}
else
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; 
x_50 = lean_ctor_get(x_10, 1);
x_51 = lean_ctor_get(x_10, 2);
lean_inc(x_51);
lean_inc(x_50);
lean_dec(x_10);
x_52 = lean_ctor_get(x_11, 1);
lean_inc(x_52);
if (lean_is_exclusive(x_11)) {
 lean_ctor_release(x_11, 0);
 lean_ctor_release(x_11, 1);
 x_53 = x_11;
} else {
 lean_dec_ref(x_11);
 x_53 = lean_box(0);
}
if (lean_is_scalar(x_53)) {
 x_54 = lean_alloc_ctor(0, 2, 0);
} else {
 x_54 = x_53;
}
lean_ctor_set(x_54, 0, x_12);
lean_ctor_set(x_54, 1, x_52);
x_55 = lp_mathlib_CommSemiring_toCommMonoid___redArg(x_7);
lean_inc(x_3);
x_56 = lp_mathlib_Unitization_instMulOneClass___redArg(x_55, x_9, x_3);
lean_dec_ref(x_55);
x_57 = lean_ctor_get(x_56, 0);
lean_inc(x_57);
x_58 = lean_ctor_get(x_56, 1);
lean_inc(x_58);
lean_dec_ref(x_56);
x_59 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_59, 0, lean_box(0));
lean_closure_set(x_59, 1, x_58);
lean_closure_set(x_59, 2, x_57);
x_60 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_60, 0, x_54);
lean_ctor_set(x_60, 1, x_50);
lean_ctor_set(x_60, 2, x_51);
lean_ctor_set(x_60, 3, x_59);
x_61 = lp_mathlib_Unitization_instRing___redArg(x_1, x_2, x_3);
x_62 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_61);
x_63 = lean_ctor_get(x_62, 0);
lean_inc(x_63);
if (lean_is_exclusive(x_62)) {
 lean_ctor_release(x_62, 0);
 lean_ctor_release(x_62, 1);
 lean_ctor_release(x_62, 2);
 lean_ctor_release(x_62, 3);
 lean_ctor_release(x_62, 4);
 x_64 = x_62;
} else {
 lean_dec_ref(x_62);
 x_64 = lean_box(0);
}
x_65 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instRing___redArg___lam__0), 4, 2);
lean_closure_set(x_65, 0, x_5);
lean_closure_set(x_65, 1, x_8);
if (lean_is_scalar(x_64)) {
 x_66 = lean_alloc_ctor(0, 5, 0);
} else {
 x_66 = x_64;
}
lean_ctor_set(x_66, 0, x_60);
lean_ctor_set(x_66, 1, x_13);
lean_ctor_set(x_66, 2, x_14);
lean_ctor_set(x_66, 3, x_65);
lean_ctor_set(x_66, 4, x_63);
return x_66;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instCommRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_instCommRing___redArg(x_3, x_4, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inlRingHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
lean_dec_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Unitization_inl), 4, 3);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, lean_box(0));
lean_closure_set(x_4, 2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inlRingHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_inlRingHom___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inlRingHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_inlRingHom(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStar___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
x_7 = lean_apply_1(x_1, x_5);
x_8 = lean_apply_1(x_2, x_6);
lean_ctor_set(x_3, 1, x_8);
lean_ctor_set(x_3, 0, x_7);
return x_3;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_3, 0);
x_10 = lean_ctor_get(x_3, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_3);
x_11 = lean_apply_1(x_1, x_9);
x_12 = lean_apply_1(x_2, x_10);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStar___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instStar___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStar(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Unitization_instStar___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarAddMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instStar___redArg___lam__0), 3, 2);
lean_closure_set(x_7, 0, x_5);
lean_closure_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarAddMonoid___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instStar___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarAddMonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Unitization_instStarAddMonoid(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instStar___redArg___lam__0), 3, 2);
lean_closure_set(x_9, 0, x_4);
lean_closure_set(x_9, 1, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarRing___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Unitization_instStar___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instStarRing___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Unitization_instStarRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAlgebra___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_2);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_2, 1);
x_7 = lp_mathlib_Prod_instSMul___redArg(x_5, x_3);
x_8 = lp_mathlib_Unitization_inlRingHom___redArg(x_1);
x_9 = lp_mathlib_RingHom_comp___redArg(x_8, x_6);
lean_ctor_set(x_2, 1, x_9);
lean_ctor_set(x_2, 0, x_7);
return x_2;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_2);
x_12 = lp_mathlib_Prod_instSMul___redArg(x_10, x_3);
x_13 = lp_mathlib_Unitization_inlRingHom___redArg(x_1);
x_14 = lp_mathlib_RingHom_comp___redArg(x_13, x_11);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_12);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Unitization_instAlgebra___redArg(x_6, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_instAlgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Unitization_instAlgebra(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_13;
}
}
static lean_object* _init_lp_mathlib_Unitization_fstHom___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Unitization_fst___boxed), 3, 2);
lean_closure_set(x_1, 0, lean_box(0));
lean_closure_set(x_1, 1, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fstHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_fstHom___closed__0;
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_fstHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_fstHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalAlgHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_3);
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Unitization_inr), 4, 3);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalAlgHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_inrNonUnitalAlgHom___redArg(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalAlgHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Unitization_inrNonUnitalAlgHom(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalAlgHom___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Unitization_inrNonUnitalAlgHom___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalStarAlgHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_inrNonUnitalAlgHom___redArg(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalStarAlgHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Unitization_inrNonUnitalAlgHom___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalStarAlgHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Unitization_inrNonUnitalStarAlgHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrNonUnitalStarAlgHom___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Unitization_inrNonUnitalStarAlgHom___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Unitization_inrRangeEquiv___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_NonUnitalStarAlgHom_instFunLike___lam__0), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrRangeEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_mathlib_Unitization_inrRangeEquiv___redArg___closed__0;
x_3 = lp_mathlib_Unitization_sndHom___closed__0;
x_4 = lp_mathlib_Unitization_inrNonUnitalAlgHom___redArg(x_1);
x_5 = lp_mathlib_StarAlgEquiv_ofLeftInverse_x27___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrRangeEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Unitization_inrRangeEquiv___redArg(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrRangeEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Unitization_inrRangeEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_inrRangeEquiv___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Unitization_inrRangeEquiv___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toAlgHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_4, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_4, 1);
lean_inc(x_7);
lean_dec_ref(x_4);
x_8 = lean_apply_1(x_5, x_6);
x_9 = lean_apply_1(x_2, x_7);
x_10 = lean_apply_2(x_3, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toAlgHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_toAlgHom___redArg___lam__0), 4, 3);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_3);
lean_closure_set(x_8, 2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toAlgHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_NonUnitalAlgHom_toAlgHom___redArg(x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toAlgHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_NonUnitalAlgHom_toAlgHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_9);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_toAlgHom___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_NonUnitalAlgHom_toAlgHom___redArg(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_lift___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_Unitization_inrNonUnitalAlgHom___redArg(x_1);
x_5 = lp_mathlib_NonUnitalAlgHom_comp___redArg(x_2, x_4);
x_6 = lean_apply_1(x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_lift___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Unitization_lift___redArg___lam__0(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_lift___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Unitization_lift___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_6, 0, x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_toAlgHom___boxed), 11, 10);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, x_1);
lean_closure_set(x_7, 3, x_2);
lean_closure_set(x_7, 4, x_3);
lean_closure_set(x_7, 5, lean_box(0));
lean_closure_set(x_7, 6, lean_box(0));
lean_closure_set(x_7, 7, lean_box(0));
lean_closure_set(x_7, 8, x_4);
lean_closure_set(x_7, 9, x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_lift(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Unitization_lift___redArg(x_3, x_4, x_5, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starLift___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lp_mathlib_Unitization_lift___redArg(x_1, x_2, x_3, x_4, x_5);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_apply_2(x_9, x_6, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starLift___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Unitization_starLift___redArg___lam__0), 7, 5);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_2);
lean_closure_set(x_6, 2, x_3);
lean_closure_set(x_6, 3, x_4);
lean_closure_set(x_6, 4, x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Unitization_lift___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_7, 0, x_1);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starLift(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Unitization_starLift___redArg(x_4, x_6, x_8, x_11, x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starLift___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Unitization_starLift(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_13);
lean_dec(x_7);
lean_dec(x_5);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starMap___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
x_7 = lp_mathlib_Unitization_instSemiring___redArg(x_1, x_4, x_5);
x_8 = lp_mathlib_Algebra_id___redArg(x_1);
x_9 = lp_mathlib_Unitization_instAlgebra___redArg(x_4, x_8, x_5);
lean_inc_ref(x_1);
x_10 = lp_mathlib_Unitization_starLift___redArg(x_1, x_2, x_3, x_7, x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lp_mathlib_Unitization_inrNonUnitalAlgHom___redArg(x_1);
lean_dec_ref(x_1);
x_13 = lp_mathlib_NonUnitalAlgHom_comp___redArg(x_12, x_6);
x_14 = lean_apply_1(x_11, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_Unitization_starMap___redArg(x_4, x_6, x_8, x_11, x_13, x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unitization_starMap___boxed(lean_object** _args) {
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
x_18 = lp_mathlib_Unitization_starMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec(x_12);
lean_dec(x_7);
lean_dec(x_5);
return x_18;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Star_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Star_NonUnitalSubalgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Unitization(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Star_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Star_NonUnitalSubalgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Unitization_addEquiv___closed__0 = _init_lp_mathlib_Unitization_addEquiv___closed__0();
lean_mark_persistent(lp_mathlib_Unitization_addEquiv___closed__0);
lp_mathlib_Unitization_sndHom___closed__0 = _init_lp_mathlib_Unitization_sndHom___closed__0();
lean_mark_persistent(lp_mathlib_Unitization_sndHom___closed__0);
lp_mathlib_Unitization_fstHom___closed__0 = _init_lp_mathlib_Unitization_fstHom___closed__0();
lean_mark_persistent(lp_mathlib_Unitization_fstHom___closed__0);
lp_mathlib_Unitization_inrRangeEquiv___redArg___closed__0 = _init_lp_mathlib_Unitization_inrRangeEquiv___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Unitization_inrRangeEquiv___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
