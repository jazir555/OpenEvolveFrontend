// Lean compiler output
// Module: Mathlib.Algebra.DirectSum.Ring
// Imports: public import Init public import Mathlib.Algebra.GradedMonoid public import Mathlib.Algebra.DirectSum.Basic public import Mathlib.Algebra.Ring.Associator
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
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Int_cast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_semiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_ring___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_semiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_ring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_smulWithZero___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_smulWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_module___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_GradedMonoid_GradeZero_smul___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_smulWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_addCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_flip___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_cast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_semiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_toAddMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_gMonoid___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg___boxed(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_ring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mul_gMul___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_instFunLike___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commRing___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_semiring___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DFinsupp_mapRange___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_semiring___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_semiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_of___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_GradedMonoid_GradeZero_mul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_semiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_module___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___redArg___boxed(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_ring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commRing___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_inc(x_3);
lean_inc(x_2);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DirectSum_GSemiring_toGMonoid(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_inc(x_3);
lean_inc(x_2);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DirectSum_GCommRing_toGCommSemiring(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DirectSum_GCommRing_toGCommSemiring___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_DirectSum_of___redArg(x_6, x_2, x_4);
x_8 = lean_apply_1(x_7, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instOne___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_DirectSum_of___redArg(x_4, x_1, x_2);
x_6 = lean_apply_1(x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_apply_4(x_1, x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_gMulHom___redArg___lam__0), 5, 3);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
lean_closure_set(x_4, 2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_DirectSum_gMulHom___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_DirectSum_gMulHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_inc(x_6);
lean_inc(x_2);
x_9 = lean_apply_2(x_1, x_2, x_6);
x_10 = lp_mathlib_DirectSum_of___redArg(x_3, x_4, x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_comp___redArg), 2, 1);
lean_closure_set(x_11, 0, x_10);
x_12 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_gMulHom___redArg___lam__0), 5, 3);
lean_closure_set(x_12, 0, x_5);
lean_closure_set(x_12, 1, x_2);
lean_closure_set(x_12, 2, x_6);
x_13 = lp_mathlib_AddMonoidHom_comp___redArg(x_11, x_12);
x_14 = lp_mathlib_AddMonoidHom_flip___redArg(x_13);
x_15 = lean_apply_2(x_14, x_7, x_8);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_9 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_mulHom___redArg___lam__0), 8, 5);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_6);
lean_closure_set(x_9, 2, x_2);
lean_closure_set(x_9, 3, x_3);
lean_closure_set(x_9, 4, x_4);
x_10 = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(x_5);
x_11 = lp_mathlib_DirectSum_toAddMonoid___redArg(x_2, x_3, x_10, x_9);
x_12 = lp_mathlib_AddMonoidHom_flip___redArg(x_11);
x_13 = lean_apply_2(x_12, x_7, x_8);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_3);
x_5 = lp_mathlib_DFinsupp_addCommMonoid___redArg(x_3);
lean_inc_ref(x_5);
lean_inc_ref(x_1);
lean_inc_ref(x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_mulHom___redArg___lam__1), 8, 5);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_3);
lean_closure_set(x_6, 2, x_1);
lean_closure_set(x_6, 3, x_4);
lean_closure_set(x_6, 4, x_5);
x_7 = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(x_5);
x_8 = lp_mathlib_DirectSum_toAddMonoid___redArg(x_3, x_1, x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_mulHom___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_DirectSum_mulHom___redArg(x_1, x_2, x_3, x_4);
x_8 = lean_apply_2(x_7, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instMul___redArg___lam__0), 6, 4);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_2);
lean_closure_set(x_5, 2, x_3);
lean_closure_set(x_5, 3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_instMul___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_3);
x_5 = lp_mathlib_DFinsupp_addCommMonoid___redArg(x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instMul___redArg___lam__0), 6, 4);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_2);
lean_closure_set(x_6, 2, x_3);
lean_closure_set(x_6, 3, x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_1, 3);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_1(x_6, x_5);
x_8 = lp_mathlib_DirectSum_of___redArg(x_2, x_3, x_4);
x_9 = lean_apply_1(x_8, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_3);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instNatCast___redArg___lam__0), 5, 4);
lean_closure_set(x_7, 0, x_4);
lean_closure_set(x_7, 1, x_2);
lean_closure_set(x_7, 2, x_1);
lean_closure_set(x_7, 3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_instNatCast___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_instNatCast(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DirectSum_instNatCast___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_semiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 0);
lean_inc(x_6);
lean_inc(x_6);
lean_inc_ref(x_2);
lean_inc(x_5);
lean_inc_ref(x_1);
x_7 = lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring___redArg(x_1, x_5, x_2, x_6);
x_8 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_3);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_4);
x_11 = lean_ctor_get(x_10, 1);
lean_inc(x_11);
lean_dec_ref(x_10);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_12 = lp_mathlib_DirectSum_instNatCast___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_13 = lp_mathlib_DirectSum_of___redArg(x_2, x_1, x_9);
x_14 = lean_apply_1(x_13, x_11);
x_15 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instMul___redArg___lam__0), 6, 4);
lean_closure_set(x_15, 0, x_1);
lean_closure_set(x_15, 1, x_5);
lean_closure_set(x_15, 2, x_2);
lean_closure_set(x_15, 3, x_6);
lean_inc_ref(x_14);
x_16 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_16, 0, lean_box(0));
lean_closure_set(x_16, 1, x_15);
lean_closure_set(x_16, 2, x_14);
x_17 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_17, 0, x_7);
lean_ctor_set(x_17, 1, x_14);
lean_ctor_set(x_17, 2, x_12);
lean_ctor_set(x_17, 3, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_semiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_semiring___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_semiring___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DirectSum_semiring___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_1, x_3);
x_6 = lean_ctor_get(x_5, 3);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, x_2, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__1), 4, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
x_5 = lp_mathlib_DFinsupp_mapRange___redArg(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
lean_inc_ref(x_2);
x_5 = lp_mathlib_DFinsupp_addCommGroup___redArg(x_2);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_7 = lean_ctor_get(x_5, 3);
lean_dec(x_7);
x_8 = lean_ctor_get(x_5, 0);
lean_dec(x_8);
lean_inc_ref(x_2);
x_9 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0), 2, 1);
lean_closure_set(x_9, 0, x_2);
lean_inc(x_4);
lean_inc_ref(x_9);
lean_inc(x_3);
lean_inc_ref(x_1);
x_10 = lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring___redArg(x_1, x_3, x_9, x_4);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
lean_dec(x_13);
x_14 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__2), 3, 1);
lean_closure_set(x_14, 0, x_2);
lean_ctor_set(x_5, 3, x_14);
lean_ctor_set(x_5, 0, x_12);
x_15 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instMul___redArg___lam__0), 6, 4);
lean_closure_set(x_15, 0, x_1);
lean_closure_set(x_15, 1, x_3);
lean_closure_set(x_15, 2, x_9);
lean_closure_set(x_15, 3, x_4);
lean_ctor_set(x_10, 1, x_15);
lean_ctor_set(x_10, 0, x_5);
return x_10;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_16 = lean_ctor_get(x_10, 0);
lean_inc(x_16);
lean_dec(x_10);
x_17 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__2), 3, 1);
lean_closure_set(x_17, 0, x_2);
lean_ctor_set(x_5, 3, x_17);
lean_ctor_set(x_5, 0, x_16);
x_18 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instMul___redArg___lam__0), 6, 4);
lean_closure_set(x_18, 0, x_1);
lean_closure_set(x_18, 1, x_3);
lean_closure_set(x_18, 2, x_9);
lean_closure_set(x_18, 3, x_4);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_5);
lean_ctor_set(x_19, 1, x_18);
return x_19;
}
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_20 = lean_ctor_get(x_5, 1);
x_21 = lean_ctor_get(x_5, 2);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_5);
lean_inc_ref(x_2);
x_22 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0), 2, 1);
lean_closure_set(x_22, 0, x_2);
lean_inc(x_4);
lean_inc_ref(x_22);
lean_inc(x_3);
lean_inc_ref(x_1);
x_23 = lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring___redArg(x_1, x_3, x_22, x_4);
x_24 = lean_ctor_get(x_23, 0);
lean_inc_ref(x_24);
if (lean_is_exclusive(x_23)) {
 lean_ctor_release(x_23, 0);
 lean_ctor_release(x_23, 1);
 x_25 = x_23;
} else {
 lean_dec_ref(x_23);
 x_25 = lean_box(0);
}
x_26 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__2), 3, 1);
lean_closure_set(x_26, 0, x_2);
x_27 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_27, 0, x_24);
lean_ctor_set(x_27, 1, x_20);
lean_ctor_set(x_27, 2, x_21);
lean_ctor_set(x_27, 3, x_26);
x_28 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instMul___redArg___lam__0), 6, 4);
lean_closure_set(x_28, 0, x_1);
lean_closure_set(x_28, 1, x_3);
lean_closure_set(x_28, 2, x_22);
lean_closure_set(x_28, 3, x_4);
if (lean_is_scalar(x_25)) {
 x_29 = lean_alloc_ctor(0, 2, 0);
} else {
 x_29 = x_25;
}
lean_ctor_set(x_29, 0, x_27);
lean_ctor_set(x_29, 1, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_nonAssocRing___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_apply_1(x_1, x_5);
x_7 = lp_mathlib_DirectSum_of___redArg(x_2, x_3, x_4);
x_8 = lean_apply_1(x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_dec_ref(x_4);
lean_inc_ref(x_2);
x_7 = lp_mathlib_DFinsupp_addCommGroup___redArg(x_2);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 2);
lean_inc(x_9);
lean_dec_ref(x_7);
x_10 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_3);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
lean_inc_ref(x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0), 2, 1);
lean_closure_set(x_12, 0, x_2);
x_13 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__2), 3, 1);
lean_closure_set(x_13, 0, x_2);
lean_inc_ref(x_1);
lean_inc_ref(x_12);
x_14 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_ring___redArg___lam__3), 5, 4);
lean_closure_set(x_14, 0, x_6);
lean_closure_set(x_14, 1, x_12);
lean_closure_set(x_14, 2, x_1);
lean_closure_set(x_14, 3, x_11);
x_15 = lp_mathlib_DirectSum_semiring___redArg(x_1, x_12, x_3, x_5);
x_16 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_8);
lean_ctor_set(x_16, 2, x_9);
lean_ctor_set(x_16, 3, x_13);
lean_ctor_set(x_16, 4, x_14);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_ring___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_ring___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DirectSum_ring___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_2, x_4);
x_6 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_5);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 1);
x_10 = lean_ctor_get(x_5, 2);
lean_inc(x_10);
lean_dec_ref(x_5);
x_11 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_1, x_3);
x_12 = lp_mathlib_Function_Injective_addMonoid___redArg(x_9, x_8, x_10);
lean_ctor_set(x_6, 1, x_11);
lean_ctor_set(x_6, 0, x_12);
return x_6;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_13 = lean_ctor_get(x_6, 0);
x_14 = lean_ctor_get(x_6, 1);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_6);
x_15 = lean_ctor_get(x_5, 2);
lean_inc(x_15);
lean_dec_ref(x_5);
x_16 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_1, x_3);
x_17 = lp_mathlib_Function_Injective_addMonoid___redArg(x_14, x_13, x_15);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_16);
return x_18;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocSemiring___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocSemiring(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_smulWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_GradedMonoid_GradeZero_smul___redArg(x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_smulWithZero___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_GradedMonoid_GradeZero_smul___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_smulWithZero___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_DirectSum_GradeZero_smulWithZero(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_ctor_get(x_5, 3);
lean_inc(x_6);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 3);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DirectSum_instNatCastOfNat(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DirectSum_instNatCastOfNat___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_semiring___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_3(x_2, x_3, x_6, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_semiring___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DirectSum_GradeZero_semiring___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_semiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_4 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_apply_1(x_1, x_5);
x_7 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_6);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_7, 1);
x_11 = lean_ctor_get(x_3, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_3, 2);
lean_inc(x_12);
x_13 = lean_ctor_get(x_3, 3);
lean_inc(x_13);
x_14 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_3);
x_15 = !lean_is_exclusive(x_3);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_16 = lean_ctor_get(x_3, 3);
lean_dec(x_16);
x_17 = lean_ctor_get(x_3, 2);
lean_dec(x_17);
x_18 = lean_ctor_get(x_3, 1);
lean_dec(x_18);
x_19 = lean_ctor_get(x_3, 0);
lean_dec(x_19);
x_20 = lean_ctor_get(x_14, 1);
lean_inc(x_20);
lean_dec_ref(x_14);
x_21 = lean_ctor_get(x_6, 2);
lean_inc(x_21);
lean_dec_ref(x_6);
x_22 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_semiring___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_22, 0, x_2);
lean_closure_set(x_22, 1, x_12);
x_23 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_11);
x_24 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_24, 0, lean_box(0));
lean_closure_set(x_24, 1, x_13);
x_25 = lp_mathlib_Function_Injective_addMonoid___redArg(x_10, x_9, x_21);
lean_ctor_set(x_7, 1, x_23);
lean_ctor_set(x_7, 0, x_25);
lean_ctor_set(x_3, 3, x_22);
lean_ctor_set(x_3, 2, x_24);
lean_ctor_set(x_3, 1, x_20);
lean_ctor_set(x_3, 0, x_7);
return x_3;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
lean_dec(x_3);
x_26 = lean_ctor_get(x_14, 1);
lean_inc(x_26);
lean_dec_ref(x_14);
x_27 = lean_ctor_get(x_6, 2);
lean_inc(x_27);
lean_dec_ref(x_6);
x_28 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_semiring___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_28, 0, x_2);
lean_closure_set(x_28, 1, x_12);
x_29 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_11);
x_30 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, x_13);
x_31 = lp_mathlib_Function_Injective_addMonoid___redArg(x_10, x_9, x_27);
lean_ctor_set(x_7, 1, x_29);
lean_ctor_set(x_7, 0, x_31);
x_32 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_32, 0, x_7);
lean_ctor_set(x_32, 1, x_26);
lean_ctor_set(x_32, 2, x_30);
lean_ctor_set(x_32, 3, x_28);
return x_32;
}
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; 
x_33 = lean_ctor_get(x_7, 0);
x_34 = lean_ctor_get(x_7, 1);
lean_inc(x_34);
lean_inc(x_33);
lean_dec(x_7);
x_35 = lean_ctor_get(x_3, 0);
lean_inc(x_35);
x_36 = lean_ctor_get(x_3, 2);
lean_inc(x_36);
x_37 = lean_ctor_get(x_3, 3);
lean_inc(x_37);
x_38 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_3);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 lean_ctor_release(x_3, 2);
 lean_ctor_release(x_3, 3);
 x_39 = x_3;
} else {
 lean_dec_ref(x_3);
 x_39 = lean_box(0);
}
x_40 = lean_ctor_get(x_38, 1);
lean_inc(x_40);
lean_dec_ref(x_38);
x_41 = lean_ctor_get(x_6, 2);
lean_inc(x_41);
lean_dec_ref(x_6);
x_42 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_semiring___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_42, 0, x_2);
lean_closure_set(x_42, 1, x_36);
x_43 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_35);
x_44 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_44, 0, lean_box(0));
lean_closure_set(x_44, 1, x_37);
x_45 = lp_mathlib_Function_Injective_addMonoid___redArg(x_34, x_33, x_41);
x_46 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_46, 0, x_45);
lean_ctor_set(x_46, 1, x_43);
if (lean_is_scalar(x_39)) {
 x_47 = lean_alloc_ctor(0, 4, 0);
} else {
 x_47 = x_39;
}
lean_ctor_set(x_47, 0, x_46);
lean_ctor_set(x_47, 1, x_40);
lean_ctor_set(x_47, 2, x_44);
lean_ctor_set(x_47, 3, x_42);
return x_47;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_semiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_semiring___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_semiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_semiring(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_DirectSum_of___redArg(x_2, x_1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_ofZeroRingHom___redArg(x_2, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_ofZeroRingHom(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_DirectSum_ofZeroRingHom___redArg(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_module___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_5 = lean_ctor_get(x_2, 0);
lean_inc(x_5);
lean_dec_ref(x_2);
x_6 = lp_mathlib_GradedMonoid_GradeZero_smul___redArg(x_4, x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_module(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_DirectSum_GradeZero_module___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_module___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_DirectSum_GradeZero_module(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_module___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_DirectSum_GradeZero_module___redArg(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commSemiring___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_3(x_1, x_3, x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_4 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_inc_ref(x_1);
lean_inc(x_5);
x_6 = lean_apply_1(x_1, x_5);
x_7 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_6);
lean_dec_ref(x_6);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_7, 1);
x_11 = lean_ctor_get(x_3, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_3, 2);
lean_inc(x_12);
x_13 = lean_ctor_get(x_3, 3);
lean_inc(x_13);
x_14 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_3);
x_15 = lean_ctor_get(x_14, 1);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lp_mathlib_DirectSum_GradeZero_semiring___redArg(x_1, x_2, x_3);
x_17 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_16);
lean_dec_ref(x_16);
x_18 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_17);
x_19 = lean_ctor_get(x_18, 1);
lean_inc_ref(x_19);
lean_dec_ref(x_18);
x_20 = lean_ctor_get(x_19, 2);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commSemiring___redArg___lam__0), 4, 2);
lean_closure_set(x_21, 0, x_12);
lean_closure_set(x_21, 1, x_5);
x_22 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_11);
x_23 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_23, 0, lean_box(0));
lean_closure_set(x_23, 1, x_13);
x_24 = lp_mathlib_Function_Injective_addMonoid___redArg(x_10, x_9, x_20);
lean_ctor_set(x_7, 1, x_22);
lean_ctor_set(x_7, 0, x_24);
x_25 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_25, 0, x_7);
lean_ctor_set(x_25, 1, x_15);
lean_ctor_set(x_25, 2, x_23);
lean_ctor_set(x_25, 3, x_21);
return x_25;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; 
x_26 = lean_ctor_get(x_7, 0);
x_27 = lean_ctor_get(x_7, 1);
lean_inc(x_27);
lean_inc(x_26);
lean_dec(x_7);
x_28 = lean_ctor_get(x_3, 0);
lean_inc(x_28);
x_29 = lean_ctor_get(x_3, 2);
lean_inc(x_29);
x_30 = lean_ctor_get(x_3, 3);
lean_inc(x_30);
x_31 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_3);
x_32 = lean_ctor_get(x_31, 1);
lean_inc(x_32);
lean_dec_ref(x_31);
x_33 = lp_mathlib_DirectSum_GradeZero_semiring___redArg(x_1, x_2, x_3);
x_34 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_33);
lean_dec_ref(x_33);
x_35 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_34);
x_36 = lean_ctor_get(x_35, 1);
lean_inc_ref(x_36);
lean_dec_ref(x_35);
x_37 = lean_ctor_get(x_36, 2);
lean_inc(x_37);
lean_dec_ref(x_36);
x_38 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commSemiring___redArg___lam__0), 4, 2);
lean_closure_set(x_38, 0, x_29);
lean_closure_set(x_38, 1, x_5);
x_39 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_28);
x_40 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_40, 0, lean_box(0));
lean_closure_set(x_40, 1, x_30);
x_41 = lp_mathlib_Function_Injective_addMonoid___redArg(x_27, x_26, x_37);
x_42 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_42, 0, x_41);
lean_ctor_set(x_42, 1, x_39);
x_43 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_43, 0, x_42);
lean_ctor_set(x_43, 1, x_32);
lean_ctor_set(x_43, 2, x_40);
lean_ctor_set(x_43, 3, x_38);
return x_43;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_commSemiring___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_commSemiring(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_1, x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_5, 2);
lean_inc(x_8);
x_9 = lean_ctor_get(x_5, 3);
lean_inc(x_9);
lean_dec_ref(x_5);
x_10 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_6);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
x_14 = lean_ctor_get(x_6, 2);
lean_inc(x_14);
lean_dec_ref(x_6);
x_15 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_2, x_3);
x_16 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_13, x_12, x_14, x_7, x_8, x_9);
lean_ctor_set(x_10, 1, x_15);
lean_ctor_set(x_10, 0, x_16);
return x_10;
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_17 = lean_ctor_get(x_10, 0);
x_18 = lean_ctor_get(x_10, 1);
lean_inc(x_18);
lean_inc(x_17);
lean_dec(x_10);
x_19 = lean_ctor_get(x_6, 2);
lean_inc(x_19);
lean_dec_ref(x_6);
x_20 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_2, x_3);
x_21 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_18, x_17, x_19, x_7, x_8, x_9);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_21);
lean_ctor_set(x_22, 1, x_20);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocRing___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocRing___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_nonUnitalNonAssocRing(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DirectSum_instIntCastOfNat(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_DirectSum_instIntCastOfNat___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_ring___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_ring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_4 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_inc_ref(x_1);
lean_inc(x_5);
x_6 = lean_apply_1(x_1, x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_6, 3);
lean_inc(x_10);
lean_dec_ref(x_6);
x_11 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_7);
lean_dec_ref(x_7);
x_12 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_12);
x_13 = lean_ctor_get(x_11, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_11, 1);
lean_inc(x_14);
lean_dec_ref(x_11);
x_15 = !lean_is_exclusive(x_3);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; uint8_t x_30; 
x_16 = lean_ctor_get(x_3, 1);
x_17 = lean_ctor_get(x_3, 0);
lean_dec(x_17);
x_18 = lean_ctor_get(x_12, 0);
lean_inc(x_18);
x_19 = lean_ctor_get(x_12, 2);
lean_inc(x_19);
x_20 = lean_ctor_get(x_12, 3);
lean_inc(x_20);
x_21 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_12);
x_22 = lean_ctor_get(x_21, 1);
lean_inc(x_22);
lean_dec_ref(x_21);
x_23 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0), 2, 1);
lean_closure_set(x_23, 0, x_1);
x_24 = lp_mathlib_DirectSum_GradeZero_semiring___redArg(x_23, x_2, x_12);
x_25 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_24);
lean_dec_ref(x_24);
x_26 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_25);
x_27 = lean_ctor_get(x_26, 1);
lean_inc_ref(x_27);
lean_dec_ref(x_26);
x_28 = lean_ctor_get(x_27, 2);
lean_inc(x_28);
lean_dec_ref(x_27);
lean_inc(x_10);
lean_inc(x_28);
lean_inc(x_13);
lean_inc(x_14);
x_29 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_14, x_13, x_28, x_8, x_9, x_10);
x_30 = !lean_is_exclusive(x_29);
if (x_30 == 0)
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_31 = lean_ctor_get(x_29, 1);
x_32 = lean_ctor_get(x_29, 2);
x_33 = lean_ctor_get(x_29, 3);
lean_dec(x_33);
x_34 = lean_ctor_get(x_29, 0);
lean_dec(x_34);
x_35 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_ring___redArg___lam__1), 3, 1);
lean_closure_set(x_35, 0, x_10);
x_36 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commSemiring___redArg___lam__0), 4, 2);
lean_closure_set(x_36, 0, x_19);
lean_closure_set(x_36, 1, x_5);
x_37 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_18);
x_38 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_38, 0, lean_box(0));
lean_closure_set(x_38, 1, x_16);
x_39 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_39, 0, lean_box(0));
lean_closure_set(x_39, 1, x_20);
x_40 = lp_mathlib_Function_Injective_addMonoid___redArg(x_14, x_13, x_28);
lean_ctor_set(x_3, 1, x_37);
lean_ctor_set(x_3, 0, x_40);
lean_ctor_set(x_29, 3, x_36);
lean_ctor_set(x_29, 2, x_39);
lean_ctor_set(x_29, 1, x_22);
lean_ctor_set(x_29, 0, x_3);
x_41 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_41, 0, x_29);
lean_ctor_set(x_41, 1, x_31);
lean_ctor_set(x_41, 2, x_32);
lean_ctor_set(x_41, 3, x_35);
lean_ctor_set(x_41, 4, x_38);
return x_41;
}
else
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; 
x_42 = lean_ctor_get(x_29, 1);
x_43 = lean_ctor_get(x_29, 2);
lean_inc(x_43);
lean_inc(x_42);
lean_dec(x_29);
x_44 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_ring___redArg___lam__1), 3, 1);
lean_closure_set(x_44, 0, x_10);
x_45 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commSemiring___redArg___lam__0), 4, 2);
lean_closure_set(x_45, 0, x_19);
lean_closure_set(x_45, 1, x_5);
x_46 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_18);
x_47 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_47, 0, lean_box(0));
lean_closure_set(x_47, 1, x_16);
x_48 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_48, 0, lean_box(0));
lean_closure_set(x_48, 1, x_20);
x_49 = lp_mathlib_Function_Injective_addMonoid___redArg(x_14, x_13, x_28);
lean_ctor_set(x_3, 1, x_46);
lean_ctor_set(x_3, 0, x_49);
x_50 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_50, 0, x_3);
lean_ctor_set(x_50, 1, x_22);
lean_ctor_set(x_50, 2, x_48);
lean_ctor_set(x_50, 3, x_45);
x_51 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_51, 0, x_50);
lean_ctor_set(x_51, 1, x_42);
lean_ctor_set(x_51, 2, x_43);
lean_ctor_set(x_51, 3, x_44);
lean_ctor_set(x_51, 4, x_47);
return x_51;
}
}
else
{
lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; 
x_52 = lean_ctor_get(x_3, 1);
lean_inc(x_52);
lean_dec(x_3);
x_53 = lean_ctor_get(x_12, 0);
lean_inc(x_53);
x_54 = lean_ctor_get(x_12, 2);
lean_inc(x_54);
x_55 = lean_ctor_get(x_12, 3);
lean_inc(x_55);
x_56 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_12);
x_57 = lean_ctor_get(x_56, 1);
lean_inc(x_57);
lean_dec_ref(x_56);
x_58 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0), 2, 1);
lean_closure_set(x_58, 0, x_1);
x_59 = lp_mathlib_DirectSum_GradeZero_semiring___redArg(x_58, x_2, x_12);
x_60 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_59);
lean_dec_ref(x_59);
x_61 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_60);
x_62 = lean_ctor_get(x_61, 1);
lean_inc_ref(x_62);
lean_dec_ref(x_61);
x_63 = lean_ctor_get(x_62, 2);
lean_inc(x_63);
lean_dec_ref(x_62);
lean_inc(x_10);
lean_inc(x_63);
lean_inc(x_13);
lean_inc(x_14);
x_64 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_14, x_13, x_63, x_8, x_9, x_10);
x_65 = lean_ctor_get(x_64, 1);
lean_inc(x_65);
x_66 = lean_ctor_get(x_64, 2);
lean_inc(x_66);
if (lean_is_exclusive(x_64)) {
 lean_ctor_release(x_64, 0);
 lean_ctor_release(x_64, 1);
 lean_ctor_release(x_64, 2);
 lean_ctor_release(x_64, 3);
 x_67 = x_64;
} else {
 lean_dec_ref(x_64);
 x_67 = lean_box(0);
}
x_68 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_ring___redArg___lam__1), 3, 1);
lean_closure_set(x_68, 0, x_10);
x_69 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commSemiring___redArg___lam__0), 4, 2);
lean_closure_set(x_69, 0, x_54);
lean_closure_set(x_69, 1, x_5);
x_70 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_53);
x_71 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_71, 0, lean_box(0));
lean_closure_set(x_71, 1, x_52);
x_72 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_72, 0, lean_box(0));
lean_closure_set(x_72, 1, x_55);
x_73 = lp_mathlib_Function_Injective_addMonoid___redArg(x_14, x_13, x_63);
x_74 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_74, 0, x_73);
lean_ctor_set(x_74, 1, x_70);
if (lean_is_scalar(x_67)) {
 x_75 = lean_alloc_ctor(0, 4, 0);
} else {
 x_75 = x_67;
}
lean_ctor_set(x_75, 0, x_74);
lean_ctor_set(x_75, 1, x_57);
lean_ctor_set(x_75, 2, x_72);
lean_ctor_set(x_75, 3, x_69);
x_76 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_76, 0, x_75);
lean_ctor_set(x_76, 1, x_65);
lean_ctor_set(x_76, 2, x_66);
lean_ctor_set(x_76, 3, x_68);
lean_ctor_set(x_76, 4, x_71);
return x_76;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_ring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_ring___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_ring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_ring(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commRing___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_1(x_2, x_6);
x_8 = lean_ctor_get(x_7, 3);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_2(x_8, x_3, x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commRing___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DirectSum_GradeZero_commRing___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_4 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_inc_ref(x_1);
lean_inc(x_5);
x_6 = lean_apply_1(x_1, x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_7);
lean_dec_ref(x_7);
x_9 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_9);
x_10 = !lean_is_exclusive(x_8);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; uint8_t x_20; 
x_11 = lean_ctor_get(x_8, 0);
x_12 = lean_ctor_get(x_8, 1);
x_13 = lean_ctor_get(x_3, 1);
lean_inc(x_13);
x_14 = lean_ctor_get(x_9, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_9, 2);
lean_inc(x_15);
x_16 = lean_ctor_get(x_9, 3);
lean_inc(x_16);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_17 = lp_mathlib_DirectSum_GradeZero_ring___redArg(x_1, x_2, x_3);
x_18 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_17);
x_19 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_18);
x_20 = !lean_is_exclusive(x_18);
if (x_20 == 0)
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; uint8_t x_34; 
x_21 = lean_ctor_get(x_18, 1);
x_22 = lean_ctor_get(x_18, 4);
lean_dec(x_22);
x_23 = lean_ctor_get(x_18, 3);
lean_dec(x_23);
x_24 = lean_ctor_get(x_18, 2);
lean_dec(x_24);
x_25 = lean_ctor_get(x_18, 0);
lean_dec(x_25);
x_26 = lean_ctor_get(x_21, 1);
lean_inc_ref(x_26);
lean_dec_ref(x_21);
x_27 = lean_ctor_get(x_19, 1);
lean_inc(x_27);
x_28 = lean_ctor_get(x_19, 2);
lean_inc(x_28);
x_29 = lean_ctor_get(x_19, 3);
lean_inc(x_29);
lean_dec_ref(x_19);
x_30 = lean_ctor_get(x_26, 2);
lean_inc(x_30);
lean_dec_ref(x_26);
x_31 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_9);
lean_dec_ref(x_9);
x_32 = lean_ctor_get(x_31, 1);
lean_inc(x_32);
lean_dec_ref(x_31);
lean_inc(x_30);
lean_inc(x_11);
lean_inc(x_12);
x_33 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_12, x_11, x_30, x_27, x_28, x_29);
x_34 = !lean_is_exclusive(x_33);
if (x_34 == 0)
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; 
x_35 = lean_ctor_get(x_33, 1);
x_36 = lean_ctor_get(x_33, 2);
x_37 = lean_ctor_get(x_33, 3);
lean_dec(x_37);
x_38 = lean_ctor_get(x_33, 0);
lean_dec(x_38);
x_39 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commRing___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_39, 0, x_2);
lean_closure_set(x_39, 1, x_1);
x_40 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commSemiring___redArg___lam__0), 4, 2);
lean_closure_set(x_40, 0, x_15);
lean_closure_set(x_40, 1, x_5);
x_41 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_14);
x_42 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_42, 0, lean_box(0));
lean_closure_set(x_42, 1, x_13);
x_43 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_43, 0, lean_box(0));
lean_closure_set(x_43, 1, x_16);
x_44 = lp_mathlib_Function_Injective_addMonoid___redArg(x_12, x_11, x_30);
lean_ctor_set(x_8, 1, x_41);
lean_ctor_set(x_8, 0, x_44);
lean_ctor_set(x_33, 3, x_40);
lean_ctor_set(x_33, 2, x_43);
lean_ctor_set(x_33, 1, x_32);
lean_ctor_set(x_33, 0, x_8);
lean_ctor_set(x_18, 4, x_42);
lean_ctor_set(x_18, 3, x_39);
lean_ctor_set(x_18, 2, x_36);
lean_ctor_set(x_18, 1, x_35);
lean_ctor_set(x_18, 0, x_33);
return x_18;
}
else
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; 
x_45 = lean_ctor_get(x_33, 1);
x_46 = lean_ctor_get(x_33, 2);
lean_inc(x_46);
lean_inc(x_45);
lean_dec(x_33);
x_47 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commRing___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_47, 0, x_2);
lean_closure_set(x_47, 1, x_1);
x_48 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commSemiring___redArg___lam__0), 4, 2);
lean_closure_set(x_48, 0, x_15);
lean_closure_set(x_48, 1, x_5);
x_49 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_14);
x_50 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_50, 0, lean_box(0));
lean_closure_set(x_50, 1, x_13);
x_51 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_51, 0, lean_box(0));
lean_closure_set(x_51, 1, x_16);
x_52 = lp_mathlib_Function_Injective_addMonoid___redArg(x_12, x_11, x_30);
lean_ctor_set(x_8, 1, x_49);
lean_ctor_set(x_8, 0, x_52);
x_53 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_53, 0, x_8);
lean_ctor_set(x_53, 1, x_32);
lean_ctor_set(x_53, 2, x_51);
lean_ctor_set(x_53, 3, x_48);
lean_ctor_set(x_18, 4, x_50);
lean_ctor_set(x_18, 3, x_47);
lean_ctor_set(x_18, 2, x_46);
lean_ctor_set(x_18, 1, x_45);
lean_ctor_set(x_18, 0, x_53);
return x_18;
}
}
else
{
lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; 
x_54 = lean_ctor_get(x_18, 1);
lean_inc(x_54);
lean_dec(x_18);
x_55 = lean_ctor_get(x_54, 1);
lean_inc_ref(x_55);
lean_dec_ref(x_54);
x_56 = lean_ctor_get(x_19, 1);
lean_inc(x_56);
x_57 = lean_ctor_get(x_19, 2);
lean_inc(x_57);
x_58 = lean_ctor_get(x_19, 3);
lean_inc(x_58);
lean_dec_ref(x_19);
x_59 = lean_ctor_get(x_55, 2);
lean_inc(x_59);
lean_dec_ref(x_55);
x_60 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_9);
lean_dec_ref(x_9);
x_61 = lean_ctor_get(x_60, 1);
lean_inc(x_61);
lean_dec_ref(x_60);
lean_inc(x_59);
lean_inc(x_11);
lean_inc(x_12);
x_62 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_12, x_11, x_59, x_56, x_57, x_58);
x_63 = lean_ctor_get(x_62, 1);
lean_inc(x_63);
x_64 = lean_ctor_get(x_62, 2);
lean_inc(x_64);
if (lean_is_exclusive(x_62)) {
 lean_ctor_release(x_62, 0);
 lean_ctor_release(x_62, 1);
 lean_ctor_release(x_62, 2);
 lean_ctor_release(x_62, 3);
 x_65 = x_62;
} else {
 lean_dec_ref(x_62);
 x_65 = lean_box(0);
}
x_66 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commRing___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_66, 0, x_2);
lean_closure_set(x_66, 1, x_1);
x_67 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commSemiring___redArg___lam__0), 4, 2);
lean_closure_set(x_67, 0, x_15);
lean_closure_set(x_67, 1, x_5);
x_68 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_14);
x_69 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_69, 0, lean_box(0));
lean_closure_set(x_69, 1, x_13);
x_70 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_70, 0, lean_box(0));
lean_closure_set(x_70, 1, x_16);
x_71 = lp_mathlib_Function_Injective_addMonoid___redArg(x_12, x_11, x_59);
lean_ctor_set(x_8, 1, x_68);
lean_ctor_set(x_8, 0, x_71);
if (lean_is_scalar(x_65)) {
 x_72 = lean_alloc_ctor(0, 4, 0);
} else {
 x_72 = x_65;
}
lean_ctor_set(x_72, 0, x_8);
lean_ctor_set(x_72, 1, x_61);
lean_ctor_set(x_72, 2, x_70);
lean_ctor_set(x_72, 3, x_67);
x_73 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_73, 0, x_72);
lean_ctor_set(x_73, 1, x_63);
lean_ctor_set(x_73, 2, x_64);
lean_ctor_set(x_73, 3, x_66);
lean_ctor_set(x_73, 4, x_69);
return x_73;
}
}
else
{
lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; 
x_74 = lean_ctor_get(x_8, 0);
x_75 = lean_ctor_get(x_8, 1);
lean_inc(x_75);
lean_inc(x_74);
lean_dec(x_8);
x_76 = lean_ctor_get(x_3, 1);
lean_inc(x_76);
x_77 = lean_ctor_get(x_9, 0);
lean_inc(x_77);
x_78 = lean_ctor_get(x_9, 2);
lean_inc(x_78);
x_79 = lean_ctor_get(x_9, 3);
lean_inc(x_79);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_80 = lp_mathlib_DirectSum_GradeZero_ring___redArg(x_1, x_2, x_3);
x_81 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_80);
x_82 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_81);
x_83 = lean_ctor_get(x_81, 1);
lean_inc_ref(x_83);
if (lean_is_exclusive(x_81)) {
 lean_ctor_release(x_81, 0);
 lean_ctor_release(x_81, 1);
 lean_ctor_release(x_81, 2);
 lean_ctor_release(x_81, 3);
 lean_ctor_release(x_81, 4);
 x_84 = x_81;
} else {
 lean_dec_ref(x_81);
 x_84 = lean_box(0);
}
x_85 = lean_ctor_get(x_83, 1);
lean_inc_ref(x_85);
lean_dec_ref(x_83);
x_86 = lean_ctor_get(x_82, 1);
lean_inc(x_86);
x_87 = lean_ctor_get(x_82, 2);
lean_inc(x_87);
x_88 = lean_ctor_get(x_82, 3);
lean_inc(x_88);
lean_dec_ref(x_82);
x_89 = lean_ctor_get(x_85, 2);
lean_inc(x_89);
lean_dec_ref(x_85);
x_90 = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(x_9);
lean_dec_ref(x_9);
x_91 = lean_ctor_get(x_90, 1);
lean_inc(x_91);
lean_dec_ref(x_90);
lean_inc(x_89);
lean_inc(x_74);
lean_inc(x_75);
x_92 = lp_mathlib_Function_Injective_subNegMonoid___redArg(x_75, x_74, x_89, x_86, x_87, x_88);
x_93 = lean_ctor_get(x_92, 1);
lean_inc(x_93);
x_94 = lean_ctor_get(x_92, 2);
lean_inc(x_94);
if (lean_is_exclusive(x_92)) {
 lean_ctor_release(x_92, 0);
 lean_ctor_release(x_92, 1);
 lean_ctor_release(x_92, 2);
 lean_ctor_release(x_92, 3);
 x_95 = x_92;
} else {
 lean_dec_ref(x_92);
 x_95 = lean_box(0);
}
x_96 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commRing___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_96, 0, x_2);
lean_closure_set(x_96, 1, x_1);
x_97 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_GradeZero_commSemiring___redArg___lam__0), 4, 2);
lean_closure_set(x_97, 0, x_78);
lean_closure_set(x_97, 1, x_5);
x_98 = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(x_4, x_77);
x_99 = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(x_99, 0, lean_box(0));
lean_closure_set(x_99, 1, x_76);
x_100 = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(x_100, 0, lean_box(0));
lean_closure_set(x_100, 1, x_79);
x_101 = lp_mathlib_Function_Injective_addMonoid___redArg(x_75, x_74, x_89);
x_102 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_102, 0, x_101);
lean_ctor_set(x_102, 1, x_98);
if (lean_is_scalar(x_95)) {
 x_103 = lean_alloc_ctor(0, 4, 0);
} else {
 x_103 = x_95;
}
lean_ctor_set(x_103, 0, x_102);
lean_ctor_set(x_103, 1, x_91);
lean_ctor_set(x_103, 2, x_100);
lean_ctor_set(x_103, 3, x_97);
if (lean_is_scalar(x_84)) {
 x_104 = lean_alloc_ctor(0, 5, 0);
} else {
 x_104 = x_84;
}
lean_ctor_set(x_104, 0, x_103);
lean_ctor_set(x_104, 1, x_93);
lean_ctor_set(x_104, 2, x_94);
lean_ctor_set(x_104, 3, x_96);
lean_ctor_set(x_104, 4, x_99);
return x_104;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_commRing___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GradeZero_commRing___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_DirectSum_GradeZero_commRing(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_mathlib_DirectSum_toAddMonoid___redArg(x_1, x_2, x_3, x_4);
x_7 = lean_apply_1(x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_3);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_toSemiring___redArg___lam__0), 5, 4);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_1);
lean_closure_set(x_8, 2, x_7);
lean_closure_set(x_8, 3, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_DirectSum_toSemiring___redArg(x_2, x_5, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_DirectSum_toSemiring(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DirectSum_toSemiring___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_liftRingHom___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
x_7 = lp_mathlib_DirectSum_toSemiring___redArg(x_1, x_2, x_3, x_6);
x_8 = lean_apply_1(x_7, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DirectSum_liftRingHom___redArg___lam__1(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_instFunLike___lam__0), 2, 1);
lean_closure_set(x_6, 0, x_3);
x_7 = lp_mathlib_DirectSum_of___redArg(x_1, x_2, x_4);
x_8 = lp_mathlib_AddMonoidHom_comp___redArg(x_6, x_7);
x_9 = lean_apply_1(x_8, x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_liftRingHom___redArg___lam__1___boxed), 5, 3);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
lean_closure_set(x_4, 2, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_DirectSum_liftRingHom___redArg___lam__2), 5, 2);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_1);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_DirectSum_liftRingHom___redArg(x_2, x_5, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_DirectSum_liftRingHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
lean_dec_ref(x_2);
x_4 = lp_mathlib_Mul_gMul___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_2(x_5, x_2, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Semiring_directSumGSemiring___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_2 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_4 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_Monoid_gMonoid___redArg(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_2);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Semiring_directSumGSemiring___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_10, 0, x_1);
x_11 = lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___redArg(x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Semiring_directSumGSemiring___redArg___lam__1), 2, 1);
lean_closure_set(x_12, 0, x_9);
x_13 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_7);
lean_ctor_set(x_13, 2, x_10);
lean_ctor_set(x_13, 3, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Semiring_directSumGSemiring___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Semiring_directSumGSemiring(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_mathlib_Semiring_directSumGSemiring___redArg(x_2);
x_4 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Ring_directSumGRing___redArg___lam__0), 2, 1);
lean_closure_set(x_6, 0, x_5);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_3);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Ring_directSumGRing___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Ring_directSumGRing(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Semiring_directSumGSemiring___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Semiring_directSumGSemiring___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CommSemiring_directSumGCommSemiring(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Ring_directSumGRing___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Ring_directSumGRing___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CommRing_directSumGCommRing(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GradedMonoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Associator(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GradedMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Associator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
