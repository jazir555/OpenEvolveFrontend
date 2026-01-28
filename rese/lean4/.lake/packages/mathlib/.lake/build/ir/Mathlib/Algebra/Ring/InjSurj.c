// Lean compiler output
// Module: Mathlib.Algebra.Ring.InjSurj
// Imports: public import Init public import Mathlib.Algebra.Ring.Defs public import Mathlib.Algebra.Opposites public import Mathlib.Algebra.GroupWithZero.InjSurj public import Mathlib.Data.Int.Cast.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Int_cast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Surjective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddOpposite_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___boxed(lean_object**);
lean_object* lp_mathlib_Function_Surjective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_cast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Surjective_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Surjective_addCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_6);
lean_ctor_set(x_10, 1, x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Function_Injective_distrib(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_7);
lean_dec(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Function_Injective_hasDistribNeg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Function_Injective_hasDistribNeg___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_3, x_6);
x_17 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, x_7);
x_18 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_16);
lean_ctor_set(x_18, 2, x_4);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lp_mathlib_Function_Injective_addMonoid___redArg(x_3, x_1, x_4);
x_7 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_5);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_Function_Injective_addMonoidWithOne(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_9);
lean_dec_ref(x_8);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_3, x_6);
x_17 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, x_7);
x_18 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_16);
lean_ctor_set(x_18, 2, x_4);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lp_mathlib_Function_Injective_addMonoid___redArg(x_3, x_1, x_4);
x_7 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_5);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_Function_Injective_addCommMonoidWithOne(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_9);
lean_dec_ref(x_8);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23) {
_start:
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
lean_inc(x_9);
x_24 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_5, x_3, x_6, x_7, x_8, x_9);
x_25 = lean_ctor_get(x_24, 0);
lean_inc_ref(x_25);
x_26 = lean_ctor_get(x_24, 1);
lean_inc(x_26);
x_27 = lean_ctor_get(x_24, 2);
lean_inc(x_27);
lean_dec_ref(x_24);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_28, 0, x_9);
x_29 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, x_10);
x_30 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, x_11);
x_31 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_31, 0, x_29);
lean_ctor_set(x_31, 1, x_25);
lean_ctor_set(x_31, 2, x_4);
x_32 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_32, 0, x_30);
lean_ctor_set(x_32, 1, x_31);
lean_ctor_set(x_32, 2, x_26);
lean_ctor_set(x_32, 3, x_27);
lean_ctor_set(x_32, 4, x_28);
return x_32;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_inc(x_7);
x_10 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_3, x_1, x_4, x_5, x_6, x_7);
x_11 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_11);
x_12 = lean_ctor_get(x_10, 1);
lean_inc(x_12);
x_13 = lean_ctor_get(x_10, 2);
lean_inc(x_13);
lean_dec_ref(x_10);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_7);
x_15 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_15, 0, lean_box(0));
lean_closure_set(x_15, 1, x_8);
x_16 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_16, 0, lean_box(0));
lean_closure_set(x_16, 1, x_9);
x_17 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_11);
lean_ctor_set(x_17, 2, x_2);
x_18 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
lean_ctor_set(x_18, 2, x_12);
lean_ctor_set(x_18, 3, x_13);
lean_ctor_set(x_18, 4, x_14);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne___boxed(lean_object** _args) {
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
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
_start:
{
lean_object* x_24; 
x_24 = lp_mathlib_Function_Injective_addGroupWithOne(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23);
lean_dec(x_13);
lean_dec_ref(x_12);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23) {
_start:
{
lean_object* x_24; uint8_t x_25; 
lean_inc(x_9);
x_24 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_5, x_3, x_6, x_7, x_8, x_9);
x_25 = !lean_is_exclusive(x_24);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_26 = lean_ctor_get(x_24, 3);
lean_dec(x_26);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_27, 0, x_9);
x_28 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_28, 0, lean_box(0));
lean_closure_set(x_28, 1, x_10);
x_29 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, x_11);
lean_ctor_set(x_24, 3, x_27);
x_30 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_30, 0, x_24);
lean_ctor_set(x_30, 1, x_29);
lean_ctor_set(x_30, 2, x_28);
lean_ctor_set(x_30, 3, x_4);
return x_30;
}
else
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_31 = lean_ctor_get(x_24, 0);
x_32 = lean_ctor_get(x_24, 1);
x_33 = lean_ctor_get(x_24, 2);
lean_inc(x_33);
lean_inc(x_32);
lean_inc(x_31);
lean_dec(x_24);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_34, 0, x_9);
x_35 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_35, 0, lean_box(0));
lean_closure_set(x_35, 1, x_10);
x_36 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_36, 0, lean_box(0));
lean_closure_set(x_36, 1, x_11);
x_37 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_37, 0, x_31);
lean_ctor_set(x_37, 1, x_32);
lean_ctor_set(x_37, 2, x_33);
lean_ctor_set(x_37, 3, x_34);
x_38 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_36);
lean_ctor_set(x_38, 2, x_35);
lean_ctor_set(x_38, 3, x_4);
return x_38;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; uint8_t x_11; 
lean_inc(x_7);
x_10 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_3, x_1, x_4, x_5, x_6, x_7);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_12 = lean_ctor_get(x_10, 3);
lean_dec(x_12);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_13, 0, x_7);
x_14 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, x_8);
x_15 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_15, 0, lean_box(0));
lean_closure_set(x_15, 1, x_9);
lean_ctor_set(x_10, 3, x_13);
x_16 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_16, 0, x_10);
lean_ctor_set(x_16, 1, x_15);
lean_ctor_set(x_16, 2, x_14);
lean_ctor_set(x_16, 3, x_2);
return x_16;
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_17 = lean_ctor_get(x_10, 0);
x_18 = lean_ctor_get(x_10, 1);
x_19 = lean_ctor_get(x_10, 2);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_dec(x_10);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_20, 0, x_7);
x_21 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_8);
x_22 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_22, 0, lean_box(0));
lean_closure_set(x_22, 1, x_9);
x_23 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_23, 0, x_17);
lean_ctor_set(x_23, 1, x_18);
lean_ctor_set(x_23, 2, x_19);
lean_ctor_set(x_23, 3, x_20);
x_24 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_22);
lean_ctor_set(x_24, 2, x_21);
lean_ctor_set(x_24, 3, x_2);
return x_24;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne___boxed(lean_object** _args) {
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
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
_start:
{
lean_object* x_24; 
x_24 = lp_mathlib_Function_Injective_addCommGroupWithOne(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23);
lean_dec(x_13);
lean_dec_ref(x_12);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; 
x_14 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_7, x_8);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Function_Injective_addMonoid___redArg(x_1, x_3, x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_9);
lean_dec(x_3);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; 
x_14 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_7, x_8);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Function_Injective_addMonoid___redArg(x_1, x_3, x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Function_Injective_nonUnitalSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_9);
lean_dec(x_3);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_18 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, x_10);
x_19 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_7, x_9);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_19);
lean_ctor_set(x_20, 1, x_6);
x_21 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_8);
lean_ctor_set(x_21, 2, x_18);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_6);
x_8 = lp_mathlib_Function_Injective_addMonoid___redArg(x_1, x_3, x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_4);
lean_ctor_set(x_10, 2, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring___boxed(lean_object** _args) {
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
x_18 = lp_mathlib_Function_Injective_nonAssocSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec_ref(x_11);
lean_dec(x_3);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_20 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_20, 0, x_10);
x_21 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_11);
x_22 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_7, x_9);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_6);
x_24 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_8);
lean_ctor_set(x_24, 2, x_21);
lean_ctor_set(x_24, 3, x_20);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_6);
x_9 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_7);
x_10 = lp_mathlib_Function_Injective_addMonoid___redArg(x_1, x_3, x_5);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_2);
x_12 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_4);
lean_ctor_set(x_12, 2, x_9);
lean_ctor_set(x_12, 3, x_8);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Injective_semiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec_ref(x_12);
lean_dec(x_3);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; 
x_20 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_3, x_5, x_8, x_6, x_7, x_9);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_4);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_1, x_3, x_6, x_4, x_5, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Injective_nonUnitalNonAssocRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec(x_11);
lean_dec_ref(x_10);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; 
x_20 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_5, x_7, x_10, x_8, x_9, x_11);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_6);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_1, x_3, x_6, x_4, x_5, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Injective_nonUnitalRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec_ref(x_12);
lean_dec(x_3);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23, lean_object* x_24, lean_object* x_25) {
_start:
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_26 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_26, 0, lean_box(0));
lean_closure_set(x_26, 1, x_13);
x_27 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_27, 0, lean_box(0));
lean_closure_set(x_27, 1, x_14);
x_28 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_5, x_7, x_11, x_9, x_10, x_12);
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_6);
x_30 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_30, 0, x_29);
lean_ctor_set(x_30, 1, x_8);
lean_ctor_set(x_30, 2, x_26);
lean_ctor_set(x_30, 3, x_27);
return x_30;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, x_9);
x_12 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_10);
x_13 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_1, x_3, x_7, x_5, x_6, x_8);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_2);
x_15 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_4);
lean_ctor_set(x_15, 2, x_11);
lean_ctor_set(x_15, 3, x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing___boxed(lean_object** _args) {
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
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
lean_object* x_24 = _args[23];
lean_object* x_25 = _args[24];
_start:
{
lean_object* x_26; 
x_26 = lp_mathlib_Function_Injective_nonAssocRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25);
lean_dec_ref(x_15);
lean_dec(x_3);
return x_26;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23, lean_object* x_24, lean_object* x_25, lean_object* x_26, lean_object* x_27) {
_start:
{
lean_object* x_28; uint8_t x_29; 
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_7);
lean_inc(x_5);
x_28 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_5, x_7, x_11, x_9, x_10, x_12);
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_30 = lean_ctor_get(x_28, 1);
x_31 = lean_ctor_get(x_28, 2);
x_32 = lean_ctor_get(x_28, 3);
lean_dec(x_32);
x_33 = lean_ctor_get(x_28, 0);
lean_dec(x_33);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_34, 0, x_13);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_35, 0, x_12);
x_36 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_36, 0, lean_box(0));
lean_closure_set(x_36, 1, x_15);
x_37 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_37, 0, lean_box(0));
lean_closure_set(x_37, 1, x_14);
x_38 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_7, x_11);
x_39 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_39, 0, x_38);
lean_ctor_set(x_39, 1, x_6);
lean_ctor_set(x_28, 3, x_34);
lean_ctor_set(x_28, 2, x_37);
lean_ctor_set(x_28, 1, x_8);
lean_ctor_set(x_28, 0, x_39);
x_40 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_40, 0, x_28);
lean_ctor_set(x_40, 1, x_30);
lean_ctor_set(x_40, 2, x_31);
lean_ctor_set(x_40, 3, x_35);
lean_ctor_set(x_40, 4, x_36);
return x_40;
}
else
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; 
x_41 = lean_ctor_get(x_28, 1);
x_42 = lean_ctor_get(x_28, 2);
lean_inc(x_42);
lean_inc(x_41);
lean_dec(x_28);
x_43 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_43, 0, x_13);
x_44 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_44, 0, x_12);
x_45 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_45, 0, lean_box(0));
lean_closure_set(x_45, 1, x_15);
x_46 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_46, 0, lean_box(0));
lean_closure_set(x_46, 1, x_14);
x_47 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_7, x_11);
x_48 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_48, 0, x_47);
lean_ctor_set(x_48, 1, x_6);
x_49 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_49, 0, x_48);
lean_ctor_set(x_49, 1, x_8);
lean_ctor_set(x_49, 2, x_46);
lean_ctor_set(x_49, 3, x_43);
x_50 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_50, 0, x_49);
lean_ctor_set(x_50, 1, x_41);
lean_ctor_set(x_50, 2, x_42);
lean_ctor_set(x_50, 3, x_44);
lean_ctor_set(x_50, 4, x_45);
return x_50;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; uint8_t x_13; 
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_3);
lean_inc(x_1);
x_12 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_1, x_3, x_7, x_5, x_6, x_8);
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_14 = lean_ctor_get(x_12, 1);
x_15 = lean_ctor_get(x_12, 2);
x_16 = lean_ctor_get(x_12, 3);
lean_dec(x_16);
x_17 = lean_ctor_get(x_12, 0);
lean_dec(x_17);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_18, 0, x_9);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_19, 0, x_8);
x_20 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, x_11);
x_21 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_10);
x_22 = lp_mathlib_Function_Injective_addMonoid___redArg(x_1, x_3, x_7);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_2);
lean_ctor_set(x_12, 3, x_18);
lean_ctor_set(x_12, 2, x_21);
lean_ctor_set(x_12, 1, x_4);
lean_ctor_set(x_12, 0, x_23);
x_24 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_24, 0, x_12);
lean_ctor_set(x_24, 1, x_14);
lean_ctor_set(x_24, 2, x_15);
lean_ctor_set(x_24, 3, x_19);
lean_ctor_set(x_24, 4, x_20);
return x_24;
}
else
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_25 = lean_ctor_get(x_12, 1);
x_26 = lean_ctor_get(x_12, 2);
lean_inc(x_26);
lean_inc(x_25);
lean_dec(x_12);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_27, 0, x_9);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_28, 0, x_8);
x_29 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, x_11);
x_30 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, x_10);
x_31 = lp_mathlib_Function_Injective_addMonoid___redArg(x_1, x_3, x_7);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_2);
x_33 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_33, 1, x_4);
lean_ctor_set(x_33, 2, x_30);
lean_ctor_set(x_33, 3, x_27);
x_34 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_25);
lean_ctor_set(x_34, 2, x_26);
lean_ctor_set(x_34, 3, x_28);
lean_ctor_set(x_34, 4, x_29);
return x_34;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring___boxed(lean_object** _args) {
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
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
lean_object* x_24 = _args[23];
lean_object* x_25 = _args[24];
lean_object* x_26 = _args[25];
lean_object* x_27 = _args[26];
_start:
{
lean_object* x_28; 
x_28 = lp_mathlib_Function_Injective_ring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27);
lean_dec_ref(x_16);
lean_dec(x_3);
return x_28;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; 
x_14 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_7, x_8);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Function_Injective_addMonoid___redArg(x_1, x_3, x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_9);
lean_dec(x_3);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; 
x_14 = lp_mathlib_Function_Injective_addMonoid___redArg(x_3, x_5, x_6);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_4);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Function_Injective_addMonoid___redArg(x_1, x_3, x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Function_Injective_nonUnitalCommSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_8);
lean_dec_ref(x_7);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_20 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_20, 0, x_10);
x_21 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_11);
x_22 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_7, x_9);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_6);
x_24 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_8);
lean_ctor_set(x_24, 2, x_21);
lean_ctor_set(x_24, 3, x_20);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_6);
x_9 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_7);
x_10 = lp_mathlib_Function_Injective_addMonoid___redArg(x_1, x_3, x_5);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_2);
x_12 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_4);
lean_ctor_set(x_12, 2, x_9);
lean_ctor_set(x_12, 3, x_8);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Injective_commSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec_ref(x_12);
lean_dec(x_3);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; 
x_20 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_3, x_5, x_8, x_6, x_7, x_9);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_4);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_1, x_3, x_6, x_4, x_5, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec(x_11);
lean_dec_ref(x_10);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; 
x_20 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_3, x_5, x_8, x_6, x_7, x_9);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_4);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_1, x_3, x_6, x_4, x_5, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Injective_nonUnitalCommRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec(x_11);
lean_dec_ref(x_10);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23, lean_object* x_24, lean_object* x_25, lean_object* x_26, lean_object* x_27) {
_start:
{
lean_object* x_28; uint8_t x_29; 
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_7);
lean_inc(x_5);
x_28 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_5, x_7, x_11, x_9, x_10, x_12);
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_30 = lean_ctor_get(x_28, 1);
x_31 = lean_ctor_get(x_28, 2);
x_32 = lean_ctor_get(x_28, 3);
lean_dec(x_32);
x_33 = lean_ctor_get(x_28, 0);
lean_dec(x_33);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_34, 0, x_13);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_35, 0, x_12);
x_36 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_36, 0, lean_box(0));
lean_closure_set(x_36, 1, x_15);
x_37 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_37, 0, lean_box(0));
lean_closure_set(x_37, 1, x_14);
x_38 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_7, x_11);
x_39 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_39, 0, x_38);
lean_ctor_set(x_39, 1, x_6);
lean_ctor_set(x_28, 3, x_34);
lean_ctor_set(x_28, 2, x_37);
lean_ctor_set(x_28, 1, x_8);
lean_ctor_set(x_28, 0, x_39);
x_40 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_40, 0, x_28);
lean_ctor_set(x_40, 1, x_30);
lean_ctor_set(x_40, 2, x_31);
lean_ctor_set(x_40, 3, x_35);
lean_ctor_set(x_40, 4, x_36);
return x_40;
}
else
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; 
x_41 = lean_ctor_get(x_28, 1);
x_42 = lean_ctor_get(x_28, 2);
lean_inc(x_42);
lean_inc(x_41);
lean_dec(x_28);
x_43 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_43, 0, x_13);
x_44 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_44, 0, x_12);
x_45 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_45, 0, lean_box(0));
lean_closure_set(x_45, 1, x_15);
x_46 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_46, 0, lean_box(0));
lean_closure_set(x_46, 1, x_14);
x_47 = lp_mathlib_Function_Injective_addMonoid___redArg(x_5, x_7, x_11);
x_48 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_48, 0, x_47);
lean_ctor_set(x_48, 1, x_6);
x_49 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_49, 0, x_48);
lean_ctor_set(x_49, 1, x_8);
lean_ctor_set(x_49, 2, x_46);
lean_ctor_set(x_49, 3, x_43);
x_50 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_50, 0, x_49);
lean_ctor_set(x_50, 1, x_41);
lean_ctor_set(x_50, 2, x_42);
lean_ctor_set(x_50, 3, x_44);
lean_ctor_set(x_50, 4, x_45);
return x_50;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; uint8_t x_13; 
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_3);
lean_inc(x_1);
x_12 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_1, x_3, x_7, x_5, x_6, x_8);
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_14 = lean_ctor_get(x_12, 1);
x_15 = lean_ctor_get(x_12, 2);
x_16 = lean_ctor_get(x_12, 3);
lean_dec(x_16);
x_17 = lean_ctor_get(x_12, 0);
lean_dec(x_17);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_18, 0, x_9);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_19, 0, x_8);
x_20 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, x_11);
x_21 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_10);
x_22 = lp_mathlib_Function_Injective_addMonoid___redArg(x_1, x_3, x_7);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_2);
lean_ctor_set(x_12, 3, x_18);
lean_ctor_set(x_12, 2, x_21);
lean_ctor_set(x_12, 1, x_4);
lean_ctor_set(x_12, 0, x_23);
x_24 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_24, 0, x_12);
lean_ctor_set(x_24, 1, x_14);
lean_ctor_set(x_24, 2, x_15);
lean_ctor_set(x_24, 3, x_19);
lean_ctor_set(x_24, 4, x_20);
return x_24;
}
else
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_25 = lean_ctor_get(x_12, 1);
x_26 = lean_ctor_get(x_12, 2);
lean_inc(x_26);
lean_inc(x_25);
lean_dec(x_12);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_27, 0, x_9);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_28, 0, x_8);
x_29 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, x_11);
x_30 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, x_10);
x_31 = lp_mathlib_Function_Injective_addMonoid___redArg(x_1, x_3, x_7);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_2);
x_33 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_33, 1, x_4);
lean_ctor_set(x_33, 2, x_30);
lean_ctor_set(x_33, 3, x_27);
x_34 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_25);
lean_ctor_set(x_34, 2, x_26);
lean_ctor_set(x_34, 3, x_28);
lean_ctor_set(x_34, 4, x_29);
return x_34;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing___boxed(lean_object** _args) {
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
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
lean_object* x_24 = _args[23];
lean_object* x_25 = _args[24];
lean_object* x_26 = _args[25];
lean_object* x_27 = _args[26];
_start:
{
lean_object* x_28; 
x_28 = lp_mathlib_Function_Injective_commRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27);
lean_dec_ref(x_16);
lean_dec(x_3);
return x_28;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_6);
lean_ctor_set(x_10, 1, x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Function_Surjective_distrib(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_7);
lean_dec(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_inc(x_6);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Function_Surjective_hasDistribNeg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_3);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Function_Surjective_hasDistribNeg___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lp_mathlib_Function_Surjective_addMonoid___redArg(x_5, x_6, x_8);
x_17 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, x_9);
x_18 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_16);
lean_ctor_set(x_18, 2, x_7);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lp_mathlib_Function_Surjective_addMonoid___redArg(x_1, x_2, x_4);
x_7 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_5);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_Function_Surjective_addMonoidWithOne(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec_ref(x_10);
lean_dec(x_3);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lp_mathlib_Function_Surjective_addMonoid___redArg(x_5, x_6, x_8);
x_17 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, x_9);
x_18 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_16);
lean_ctor_set(x_18, 2, x_7);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lp_mathlib_Function_Surjective_addMonoid___redArg(x_1, x_2, x_4);
x_7 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_5);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_Function_Surjective_addCommMonoidWithOne(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec_ref(x_10);
lean_dec(x_3);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23) {
_start:
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
lean_inc(x_10);
lean_inc(x_6);
lean_inc(x_5);
x_24 = lp_mathlib_Function_Surjective_addMonoid___redArg(x_5, x_6, x_10);
x_25 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, x_12);
x_26 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_24);
lean_ctor_set(x_26, 2, x_7);
lean_inc(x_11);
x_27 = lp_mathlib_Function_Surjective_subNegMonoid___redArg(x_5, x_6, x_10, x_8, x_9, x_11);
x_28 = lean_ctor_get(x_27, 1);
lean_inc(x_28);
x_29 = lean_ctor_get(x_27, 2);
lean_inc(x_29);
lean_dec_ref(x_27);
x_30 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_30, 0, x_11);
x_31 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_31, 0, lean_box(0));
lean_closure_set(x_31, 1, x_13);
x_32 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_26);
lean_ctor_set(x_32, 2, x_28);
lean_ctor_set(x_32, 3, x_29);
lean_ctor_set(x_32, 4, x_30);
return x_32;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_inc(x_6);
lean_inc(x_2);
lean_inc(x_1);
x_10 = lp_mathlib_Function_Surjective_addMonoid___redArg(x_1, x_2, x_6);
x_11 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, x_8);
x_12 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_10);
lean_ctor_set(x_12, 2, x_3);
lean_inc(x_7);
x_13 = lp_mathlib_Function_Surjective_subNegMonoid___redArg(x_1, x_2, x_6, x_4, x_5, x_7);
x_14 = lean_ctor_get(x_13, 1);
lean_inc(x_14);
x_15 = lean_ctor_get(x_13, 2);
lean_inc(x_15);
lean_dec_ref(x_13);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_16, 0, x_7);
x_17 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, x_9);
x_18 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_12);
lean_ctor_set(x_18, 2, x_14);
lean_ctor_set(x_18, 3, x_15);
lean_ctor_set(x_18, 4, x_16);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne___boxed(lean_object** _args) {
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
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
_start:
{
lean_object* x_24; 
x_24 = lp_mathlib_Function_Surjective_addGroupWithOne(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23);
lean_dec_ref(x_14);
lean_dec(x_3);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23) {
_start:
{
lean_object* x_24; lean_object* x_25; uint8_t x_26; 
lean_inc(x_10);
lean_inc(x_6);
lean_inc(x_5);
x_24 = lp_mathlib_Function_Surjective_addMonoid___redArg(x_5, x_6, x_10);
lean_inc(x_11);
x_25 = lp_mathlib_Function_Surjective_subNegMonoid___redArg(x_5, x_6, x_10, x_8, x_9, x_11);
x_26 = !lean_is_exclusive(x_25);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_27 = lean_ctor_get(x_25, 3);
lean_dec(x_27);
x_28 = lean_ctor_get(x_25, 0);
lean_dec(x_28);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_29, 0, x_11);
x_30 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, x_12);
x_31 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_31, 0, lean_box(0));
lean_closure_set(x_31, 1, x_13);
lean_ctor_set(x_25, 3, x_29);
lean_ctor_set(x_25, 0, x_24);
x_32 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_32, 0, x_25);
lean_ctor_set(x_32, 1, x_31);
lean_ctor_set(x_32, 2, x_30);
lean_ctor_set(x_32, 3, x_7);
return x_32;
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_33 = lean_ctor_get(x_25, 1);
x_34 = lean_ctor_get(x_25, 2);
lean_inc(x_34);
lean_inc(x_33);
lean_dec(x_25);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_35, 0, x_11);
x_36 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_36, 0, lean_box(0));
lean_closure_set(x_36, 1, x_12);
x_37 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_37, 0, lean_box(0));
lean_closure_set(x_37, 1, x_13);
x_38 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_38, 0, x_24);
lean_ctor_set(x_38, 1, x_33);
lean_ctor_set(x_38, 2, x_34);
lean_ctor_set(x_38, 3, x_35);
x_39 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_39, 0, x_38);
lean_ctor_set(x_39, 1, x_37);
lean_ctor_set(x_39, 2, x_36);
lean_ctor_set(x_39, 3, x_7);
return x_39;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; 
lean_inc(x_6);
lean_inc(x_2);
lean_inc(x_1);
x_10 = lp_mathlib_Function_Surjective_addMonoid___redArg(x_1, x_2, x_6);
lean_inc(x_7);
x_11 = lp_mathlib_Function_Surjective_subNegMonoid___redArg(x_1, x_2, x_6, x_4, x_5, x_7);
x_12 = !lean_is_exclusive(x_11);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_13 = lean_ctor_get(x_11, 3);
lean_dec(x_13);
x_14 = lean_ctor_get(x_11, 0);
lean_dec(x_14);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_15, 0, x_7);
x_16 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_16, 0, lean_box(0));
lean_closure_set(x_16, 1, x_8);
x_17 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, x_9);
lean_ctor_set(x_11, 3, x_15);
lean_ctor_set(x_11, 0, x_10);
x_18 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_18, 0, x_11);
lean_ctor_set(x_18, 1, x_17);
lean_ctor_set(x_18, 2, x_16);
lean_ctor_set(x_18, 3, x_3);
return x_18;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_19 = lean_ctor_get(x_11, 1);
x_20 = lean_ctor_get(x_11, 2);
lean_inc(x_20);
lean_inc(x_19);
lean_dec(x_11);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_21, 0, x_7);
x_22 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_22, 0, lean_box(0));
lean_closure_set(x_22, 1, x_8);
x_23 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_23, 0, lean_box(0));
lean_closure_set(x_23, 1, x_9);
x_24 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_24, 0, x_10);
lean_ctor_set(x_24, 1, x_19);
lean_ctor_set(x_24, 2, x_20);
lean_ctor_set(x_24, 3, x_21);
x_25 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_25, 0, x_24);
lean_ctor_set(x_25, 1, x_23);
lean_ctor_set(x_25, 2, x_22);
lean_ctor_set(x_25, 3, x_3);
return x_25;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne___boxed(lean_object** _args) {
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
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
_start:
{
lean_object* x_24; 
x_24 = lp_mathlib_Function_Surjective_addCommGroupWithOne(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23);
lean_dec_ref(x_14);
lean_dec(x_3);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; 
x_14 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_5, x_7, x_8);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_1, x_3, x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_9);
lean_dec(x_3);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; 
x_14 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_5, x_7, x_8);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_1, x_3, x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Function_Surjective_nonUnitalSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_9);
lean_dec(x_3);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_18 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, x_10);
x_19 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_5, x_7, x_9);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_19);
lean_ctor_set(x_20, 1, x_6);
x_21 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_8);
lean_ctor_set(x_21, 2, x_18);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_6);
x_8 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_1, x_3, x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_4);
lean_ctor_set(x_10, 2, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring___boxed(lean_object** _args) {
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
x_18 = lp_mathlib_Function_Surjective_nonAssocSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec_ref(x_11);
lean_dec(x_3);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_20 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_20, 0, x_10);
x_21 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_11);
x_22 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_5, x_7, x_9);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_6);
x_24 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_8);
lean_ctor_set(x_24, 2, x_21);
lean_ctor_set(x_24, 3, x_20);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_6);
x_9 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_7);
x_10 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_1, x_3, x_5);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_2);
x_12 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_4);
lean_ctor_set(x_12, 2, x_9);
lean_ctor_set(x_12, 3, x_8);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Surjective_semiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec_ref(x_12);
lean_dec(x_3);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; 
x_20 = lp_mathlib_Function_Surjective_addCommGroup___redArg(x_5, x_7, x_10, x_8, x_9, x_11);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_6);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_Function_Surjective_addCommGroup___redArg(x_1, x_3, x_6, x_4, x_5, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Surjective_nonUnitalNonAssocRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec_ref(x_12);
lean_dec(x_3);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; 
x_20 = lp_mathlib_Function_Surjective_addCommGroup___redArg(x_5, x_7, x_10, x_8, x_9, x_11);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_6);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_Function_Surjective_addCommGroup___redArg(x_1, x_3, x_6, x_4, x_5, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Surjective_nonUnitalRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec_ref(x_12);
lean_dec(x_3);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23, lean_object* x_24, lean_object* x_25) {
_start:
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_26 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_26, 0, lean_box(0));
lean_closure_set(x_26, 1, x_13);
x_27 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_27, 0, lean_box(0));
lean_closure_set(x_27, 1, x_14);
x_28 = lp_mathlib_Function_Surjective_addCommGroup___redArg(x_5, x_7, x_11, x_9, x_10, x_12);
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_6);
x_30 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_30, 0, x_29);
lean_ctor_set(x_30, 1, x_8);
lean_ctor_set(x_30, 2, x_26);
lean_ctor_set(x_30, 3, x_27);
return x_30;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, x_9);
x_12 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_10);
x_13 = lp_mathlib_Function_Surjective_addCommGroup___redArg(x_1, x_3, x_7, x_5, x_6, x_8);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_2);
x_15 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_4);
lean_ctor_set(x_15, 2, x_11);
lean_ctor_set(x_15, 3, x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing___boxed(lean_object** _args) {
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
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
lean_object* x_24 = _args[23];
lean_object* x_25 = _args[24];
_start:
{
lean_object* x_26; 
x_26 = lp_mathlib_Function_Surjective_nonAssocRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25);
lean_dec_ref(x_15);
lean_dec(x_3);
return x_26;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23, lean_object* x_24, lean_object* x_25, lean_object* x_26, lean_object* x_27) {
_start:
{
lean_object* x_28; uint8_t x_29; 
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_7);
lean_inc(x_5);
x_28 = lp_mathlib_Function_Surjective_subNegMonoid___redArg(x_5, x_7, x_11, x_9, x_10, x_12);
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_30 = lean_ctor_get(x_28, 1);
x_31 = lean_ctor_get(x_28, 2);
x_32 = lean_ctor_get(x_28, 3);
lean_dec(x_32);
x_33 = lean_ctor_get(x_28, 0);
lean_dec(x_33);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_34, 0, x_12);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_35, 0, x_13);
x_36 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_36, 0, lean_box(0));
lean_closure_set(x_36, 1, x_15);
x_37 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_37, 0, lean_box(0));
lean_closure_set(x_37, 1, x_14);
x_38 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_5, x_7, x_11);
x_39 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_39, 0, x_38);
lean_ctor_set(x_39, 1, x_6);
lean_ctor_set(x_28, 3, x_35);
lean_ctor_set(x_28, 2, x_37);
lean_ctor_set(x_28, 1, x_8);
lean_ctor_set(x_28, 0, x_39);
x_40 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_40, 0, x_28);
lean_ctor_set(x_40, 1, x_30);
lean_ctor_set(x_40, 2, x_31);
lean_ctor_set(x_40, 3, x_34);
lean_ctor_set(x_40, 4, x_36);
return x_40;
}
else
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; 
x_41 = lean_ctor_get(x_28, 1);
x_42 = lean_ctor_get(x_28, 2);
lean_inc(x_42);
lean_inc(x_41);
lean_dec(x_28);
x_43 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_43, 0, x_12);
x_44 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_44, 0, x_13);
x_45 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_45, 0, lean_box(0));
lean_closure_set(x_45, 1, x_15);
x_46 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_46, 0, lean_box(0));
lean_closure_set(x_46, 1, x_14);
x_47 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_5, x_7, x_11);
x_48 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_48, 0, x_47);
lean_ctor_set(x_48, 1, x_6);
x_49 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_49, 0, x_48);
lean_ctor_set(x_49, 1, x_8);
lean_ctor_set(x_49, 2, x_46);
lean_ctor_set(x_49, 3, x_44);
x_50 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_50, 0, x_49);
lean_ctor_set(x_50, 1, x_41);
lean_ctor_set(x_50, 2, x_42);
lean_ctor_set(x_50, 3, x_43);
lean_ctor_set(x_50, 4, x_45);
return x_50;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; uint8_t x_13; 
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_3);
lean_inc(x_1);
x_12 = lp_mathlib_Function_Surjective_subNegMonoid___redArg(x_1, x_3, x_7, x_5, x_6, x_8);
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_14 = lean_ctor_get(x_12, 1);
x_15 = lean_ctor_get(x_12, 2);
x_16 = lean_ctor_get(x_12, 3);
lean_dec(x_16);
x_17 = lean_ctor_get(x_12, 0);
lean_dec(x_17);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_18, 0, x_8);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_19, 0, x_9);
x_20 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, x_11);
x_21 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_10);
x_22 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_1, x_3, x_7);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_2);
lean_ctor_set(x_12, 3, x_19);
lean_ctor_set(x_12, 2, x_21);
lean_ctor_set(x_12, 1, x_4);
lean_ctor_set(x_12, 0, x_23);
x_24 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_24, 0, x_12);
lean_ctor_set(x_24, 1, x_14);
lean_ctor_set(x_24, 2, x_15);
lean_ctor_set(x_24, 3, x_18);
lean_ctor_set(x_24, 4, x_20);
return x_24;
}
else
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_25 = lean_ctor_get(x_12, 1);
x_26 = lean_ctor_get(x_12, 2);
lean_inc(x_26);
lean_inc(x_25);
lean_dec(x_12);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_27, 0, x_8);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_28, 0, x_9);
x_29 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, x_11);
x_30 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, x_10);
x_31 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_1, x_3, x_7);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_2);
x_33 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_33, 1, x_4);
lean_ctor_set(x_33, 2, x_30);
lean_ctor_set(x_33, 3, x_28);
x_34 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_25);
lean_ctor_set(x_34, 2, x_26);
lean_ctor_set(x_34, 3, x_27);
lean_ctor_set(x_34, 4, x_29);
return x_34;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring___boxed(lean_object** _args) {
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
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
lean_object* x_24 = _args[23];
lean_object* x_25 = _args[24];
lean_object* x_26 = _args[25];
lean_object* x_27 = _args[26];
_start:
{
lean_object* x_28; 
x_28 = lp_mathlib_Function_Surjective_ring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27);
lean_dec_ref(x_16);
lean_dec(x_3);
return x_28;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; 
x_14 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_5, x_7, x_8);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_1, x_3, x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_9);
lean_dec(x_3);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; 
x_14 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_5, x_7, x_8);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_1, x_3, x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Function_Surjective_nonUnitalCommSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_9);
lean_dec(x_3);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_20 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_20, 0, x_10);
x_21 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_11);
x_22 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_5, x_7, x_9);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_6);
x_24 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_8);
lean_ctor_set(x_24, 2, x_21);
lean_ctor_set(x_24, 3, x_20);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_6);
x_9 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_7);
x_10 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_1, x_3, x_5);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_2);
x_12 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_4);
lean_ctor_set(x_12, 2, x_9);
lean_ctor_set(x_12, 3, x_8);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Surjective_commSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec_ref(x_12);
lean_dec(x_3);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; 
x_20 = lp_mathlib_Function_Surjective_addCommGroup___redArg(x_5, x_7, x_10, x_8, x_9, x_11);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_6);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_Function_Surjective_addCommGroup___redArg(x_1, x_3, x_6, x_4, x_5, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec_ref(x_12);
lean_dec(x_3);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; lean_object* x_21; 
x_20 = lp_mathlib_Function_Surjective_addCommGroup___redArg(x_5, x_7, x_10, x_8, x_9, x_11);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_6);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_Function_Surjective_addCommGroup___redArg(x_1, x_3, x_6, x_4, x_5, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Function_Surjective_nonUnitalCommRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec_ref(x_12);
lean_dec(x_3);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23, lean_object* x_24, lean_object* x_25, lean_object* x_26, lean_object* x_27) {
_start:
{
lean_object* x_28; uint8_t x_29; 
lean_inc(x_12);
lean_inc(x_11);
lean_inc(x_7);
lean_inc(x_5);
x_28 = lp_mathlib_Function_Surjective_subNegMonoid___redArg(x_5, x_7, x_11, x_9, x_10, x_12);
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_30 = lean_ctor_get(x_28, 1);
x_31 = lean_ctor_get(x_28, 2);
x_32 = lean_ctor_get(x_28, 3);
lean_dec(x_32);
x_33 = lean_ctor_get(x_28, 0);
lean_dec(x_33);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_34, 0, x_12);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_35, 0, x_13);
x_36 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_36, 0, lean_box(0));
lean_closure_set(x_36, 1, x_15);
x_37 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_37, 0, lean_box(0));
lean_closure_set(x_37, 1, x_14);
x_38 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_5, x_7, x_11);
x_39 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_39, 0, x_38);
lean_ctor_set(x_39, 1, x_6);
lean_ctor_set(x_28, 3, x_35);
lean_ctor_set(x_28, 2, x_37);
lean_ctor_set(x_28, 1, x_8);
lean_ctor_set(x_28, 0, x_39);
x_40 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_40, 0, x_28);
lean_ctor_set(x_40, 1, x_30);
lean_ctor_set(x_40, 2, x_31);
lean_ctor_set(x_40, 3, x_34);
lean_ctor_set(x_40, 4, x_36);
return x_40;
}
else
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; 
x_41 = lean_ctor_get(x_28, 1);
x_42 = lean_ctor_get(x_28, 2);
lean_inc(x_42);
lean_inc(x_41);
lean_dec(x_28);
x_43 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_43, 0, x_12);
x_44 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_44, 0, x_13);
x_45 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_45, 0, lean_box(0));
lean_closure_set(x_45, 1, x_15);
x_46 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_46, 0, lean_box(0));
lean_closure_set(x_46, 1, x_14);
x_47 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_5, x_7, x_11);
x_48 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_48, 0, x_47);
lean_ctor_set(x_48, 1, x_6);
x_49 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_49, 0, x_48);
lean_ctor_set(x_49, 1, x_8);
lean_ctor_set(x_49, 2, x_46);
lean_ctor_set(x_49, 3, x_44);
x_50 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_50, 0, x_49);
lean_ctor_set(x_50, 1, x_41);
lean_ctor_set(x_50, 2, x_42);
lean_ctor_set(x_50, 3, x_43);
lean_ctor_set(x_50, 4, x_45);
return x_50;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; uint8_t x_13; 
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_3);
lean_inc(x_1);
x_12 = lp_mathlib_Function_Surjective_subNegMonoid___redArg(x_1, x_3, x_7, x_5, x_6, x_8);
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_14 = lean_ctor_get(x_12, 1);
x_15 = lean_ctor_get(x_12, 2);
x_16 = lean_ctor_get(x_12, 3);
lean_dec(x_16);
x_17 = lean_ctor_get(x_12, 0);
lean_dec(x_17);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_18, 0, x_8);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_19, 0, x_9);
x_20 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, x_11);
x_21 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, x_10);
x_22 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_1, x_3, x_7);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_2);
lean_ctor_set(x_12, 3, x_19);
lean_ctor_set(x_12, 2, x_21);
lean_ctor_set(x_12, 1, x_4);
lean_ctor_set(x_12, 0, x_23);
x_24 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_24, 0, x_12);
lean_ctor_set(x_24, 1, x_14);
lean_ctor_set(x_24, 2, x_15);
lean_ctor_set(x_24, 3, x_18);
lean_ctor_set(x_24, 4, x_20);
return x_24;
}
else
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_25 = lean_ctor_get(x_12, 1);
x_26 = lean_ctor_get(x_12, 2);
lean_inc(x_26);
lean_inc(x_25);
lean_dec(x_12);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addGroupWithOne___redArg___lam__0), 3, 1);
lean_closure_set(x_27, 0, x_8);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(x_28, 0, x_9);
x_29 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, x_11);
x_30 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, x_10);
x_31 = lp_mathlib_Function_Surjective_addCommMonoid___redArg(x_1, x_3, x_7);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_2);
x_33 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_33, 1, x_4);
lean_ctor_set(x_33, 2, x_30);
lean_ctor_set(x_33, 3, x_28);
x_34 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_25);
lean_ctor_set(x_34, 2, x_26);
lean_ctor_set(x_34, 3, x_27);
lean_ctor_set(x_34, 4, x_29);
return x_34;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing___boxed(lean_object** _args) {
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
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
lean_object* x_24 = _args[23];
lean_object* x_25 = _args[24];
lean_object* x_26 = _args[25];
lean_object* x_27 = _args[26];
_start:
{
lean_object* x_28; 
x_28 = lp_mathlib_Function_Surjective_commRing(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27);
lean_dec_ref(x_16);
lean_dec(x_3);
return x_28;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddOpposite_instNeg___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddOpposite_instNeg___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddOpposite_instHasDistribNeg(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
