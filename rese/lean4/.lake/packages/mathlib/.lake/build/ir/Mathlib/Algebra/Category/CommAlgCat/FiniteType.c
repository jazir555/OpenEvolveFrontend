// Lean compiler output
// Module: Mathlib.Algebra.Category.CommAlgCat.FiniteType
// Imports: public import Init public import Mathlib.Algebra.Category.CommAlgCat.Basic public import Mathlib.CategoryTheory.MorphismProperty.Comma public import Mathlib.RingTheory.FinitePresentation public import Mathlib.RingTheory.RingHomProperties
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
lean_object* lp_mathlib_ULift_ring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_fullyFaithfulUliftFunctor___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_homMk___at___00FGAlgCat_equivUnder_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_homMk___at___00FGAlgCat_equivUnder_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__11___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_uliftFunctor___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__10___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__9___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_fullyFaithfulUliftFunctor(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__9___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_uliftFunctor(lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_fullyFaithfulUliftFunctor___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_homMk___at___00FGAlgCat_equivUnder_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__10___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ULift_algebra___redArg(lean_object*);
lean_object* lp_mathlib_ULift_algEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_uliftFunctor___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_uliftFunctor___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_uliftFunctor___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__11___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_homMk___at___00FGAlgCat_equivUnder_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__14___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1___boxed(lean_object*);
lean_object* lp_mathlib_commAlgCatEquivUnder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_fullyFaithfulUliftFunctor___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_uliftFunctor___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_5, 0);
x_8 = lp_mathlib_ULift_algEquiv___redArg(x_7);
x_9 = lp_mathlib_Equiv_symm___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_ULift_algEquiv___redArg(x_6);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lp_mathlib_AlgHom_comp___redArg(x_3, x_12);
x_14 = lp_mathlib_AlgHom_comp___redArg(x_10, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_uliftFunctor___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FGAlgCat_uliftFunctor___lam__0(x_1, x_2, x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_uliftFunctor___lam__1(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lp_mathlib_ULift_ring___redArg(x_3);
x_6 = lp_mathlib_ULift_algebra___redArg(x_4);
lean_ctor_set(x_1, 1, x_6);
lean_ctor_set(x_1, 0, x_5);
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
x_9 = lp_mathlib_ULift_ring___redArg(x_7);
x_10 = lp_mathlib_ULift_algebra___redArg(x_8);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_uliftFunctor(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_uliftFunctor___lam__0___boxed), 3, 0);
x_4 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_uliftFunctor___lam__1), 1, 0);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_uliftFunctor___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_FGAlgCat_uliftFunctor(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_fullyFaithfulUliftFunctor___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_5, 0);
x_8 = lp_mathlib_ULift_algEquiv___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_ULift_algEquiv___redArg(x_6);
x_11 = lp_mathlib_Equiv_symm___redArg(x_10);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lp_mathlib_AlgHom_comp___redArg(x_3, x_12);
x_14 = lp_mathlib_AlgHom_comp___redArg(x_9, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_fullyFaithfulUliftFunctor___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FGAlgCat_fullyFaithfulUliftFunctor___lam__0(x_1, x_2, x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_fullyFaithfulUliftFunctor(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_fullyFaithfulUliftFunctor___lam__0___boxed), 3, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_fullyFaithfulUliftFunctor___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_FGAlgCat_fullyFaithfulUliftFunctor(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__9(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc_ref(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__9___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__0(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__0(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_homMk___at___00FGAlgCat_equivUnder_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_homMk___at___00FGAlgCat_equivUnder_spec__0___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__10(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__10___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_symm___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10(x_2, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__14___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__14(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__14___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_ctor_get(x_2, 0);
lean_inc(x_5);
lean_dec_ref(x_2);
x_6 = lean_apply_1(x_5, x_3);
x_7 = lean_apply_1(x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
lean_dec_ref(x_2);
lean_inc(x_7);
lean_inc_ref(x_3);
x_9 = lean_apply_1(x_7, x_3);
lean_inc_ref(x_4);
x_10 = lean_apply_1(x_7, x_4);
x_11 = lean_apply_3(x_8, x_3, x_4, x_5);
x_12 = lean_apply_3(x_6, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___redArg___lam__1), 5, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_ctor_get(x_2, 0);
lean_inc(x_5);
lean_dec_ref(x_2);
x_6 = lean_apply_1(x_5, x_3);
x_7 = lean_apply_1(x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
lean_dec_ref(x_2);
lean_inc(x_7);
lean_inc_ref(x_3);
x_9 = lean_apply_1(x_7, x_3);
lean_inc_ref(x_4);
x_10 = lean_apply_1(x_7, x_4);
x_11 = lean_apply_3(x_8, x_3, x_4, x_5);
x_12 = lean_apply_3(x_6, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___redArg___lam__1), 5, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1___lam__0___boxed), 1, 0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 1);
x_3 = lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1(x_2);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, lean_box(0));
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19___redArg(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
x_5 = lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19___redArg(x_4);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
return x_4;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_inc(x_7);
lean_inc(x_6);
lean_dec(x_4);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16___redArg(x_1, x_3, x_4, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__11(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_symm___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__11___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19(x_1, x_2);
lean_inc(x_3);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__0(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__1(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__2(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__3(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__1___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__2___boxed), 4, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__0___boxed), 4, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1___lam__3), 1, 0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__1(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__0___boxed), 1, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3___lam__1___boxed), 3, 0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__1(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__0___boxed), 1, 0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___lam__1___boxed), 3, 0);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_5);
x_6 = lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8___redArg(x_5);
x_7 = lp_mathlib_AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9(x_1, x_2, x_3, x_4, x_5);
x_8 = lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__14___redArg(x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lp_mathlib_commAlgCatEquivUnder(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_3(x_7, x_2, x_3, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_commAlgCatEquivUnder(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_apply_1(x_5, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_commAlgCatEquivUnder(x_1);
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_apply_1(x_5, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lp_mathlib_commAlgCatEquivUnder(x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_3(x_7, x_2, x_3, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
lean_dec_ref(x_2);
lean_inc_ref(x_5);
x_8 = lean_apply_1(x_3, x_5);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
lean_inc_ref(x_5);
x_10 = lean_apply_1(x_6, x_5);
lean_inc_ref(x_5);
x_11 = lean_apply_1(x_7, x_5);
x_12 = lp_mathlib_RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6(x_5);
lean_dec_ref(x_5);
x_13 = lp_mathlib_CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8(x_4, x_10, x_9, x_8, x_12);
lean_dec_ref(x_8);
lean_dec(x_9);
x_14 = lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16___redArg(x_4, x_10, x_11, x_13);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder___lam__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_FGAlgCat_equivUnder___lam__4(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FGAlgCat_equivUnder(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1;
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
lean_dec(x_5);
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_equivUnder___lam__0), 4, 1);
lean_closure_set(x_6, 0, x_1);
lean_inc_ref(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_equivUnder___lam__1), 2, 1);
lean_closure_set(x_7, 0, x_1);
lean_inc_ref(x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_equivUnder___lam__2), 2, 1);
lean_closure_set(x_8, 0, x_1);
lean_inc_ref(x_1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_equivUnder___lam__3), 4, 1);
lean_closure_set(x_9, 0, x_1);
lean_inc_ref(x_1);
x_10 = lean_apply_1(x_4, x_1);
lean_inc_ref(x_7);
lean_ctor_set(x_2, 1, x_6);
lean_ctor_set(x_2, 0, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_8);
lean_ctor_set(x_11, 1, x_9);
x_12 = lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4(x_1);
lean_inc_ref(x_11);
lean_inc_ref(x_2);
x_13 = lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___redArg(x_2, x_11);
x_14 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_equivUnder___lam__4___boxed), 5, 4);
lean_closure_set(x_14, 0, x_12);
lean_closure_set(x_14, 1, x_13);
lean_closure_set(x_14, 2, x_7);
lean_closure_set(x_14, 3, x_1);
x_15 = lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___redArg(x_14);
lean_inc_ref(x_2);
lean_inc_ref(x_11);
x_16 = lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___redArg(x_11, x_2);
x_17 = lp_mathlib_CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19(x_10, x_16);
lean_dec_ref(x_10);
x_18 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_18, 0, x_2);
lean_ctor_set(x_18, 1, x_11);
lean_ctor_set(x_18, 2, x_15);
lean_ctor_set(x_18, 3, x_17);
return x_18;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_19 = lean_ctor_get(x_2, 0);
lean_inc(x_19);
lean_dec(x_2);
lean_inc_ref(x_1);
x_20 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_equivUnder___lam__0), 4, 1);
lean_closure_set(x_20, 0, x_1);
lean_inc_ref(x_1);
x_21 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_equivUnder___lam__1), 2, 1);
lean_closure_set(x_21, 0, x_1);
lean_inc_ref(x_1);
x_22 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_equivUnder___lam__2), 2, 1);
lean_closure_set(x_22, 0, x_1);
lean_inc_ref(x_1);
x_23 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_equivUnder___lam__3), 4, 1);
lean_closure_set(x_23, 0, x_1);
lean_inc_ref(x_1);
x_24 = lean_apply_1(x_19, x_1);
lean_inc_ref(x_21);
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_21);
lean_ctor_set(x_25, 1, x_20);
x_26 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_26, 0, x_22);
lean_ctor_set(x_26, 1, x_23);
x_27 = lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4(x_1);
lean_inc_ref(x_26);
lean_inc_ref(x_25);
x_28 = lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___redArg(x_25, x_26);
x_29 = lean_alloc_closure((void*)(lp_mathlib_FGAlgCat_equivUnder___lam__4___boxed), 5, 4);
lean_closure_set(x_29, 0, x_27);
lean_closure_set(x_29, 1, x_28);
lean_closure_set(x_29, 2, x_21);
lean_closure_set(x_29, 3, x_1);
x_30 = lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___redArg(x_29);
lean_inc_ref(x_25);
lean_inc_ref(x_26);
x_31 = lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___redArg(x_26, x_25);
x_32 = lp_mathlib_CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19(x_24, x_31);
lean_dec_ref(x_24);
x_33 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_33, 0, x_25);
lean_ctor_set(x_33, 1, x_26);
lean_ctor_set(x_33, 2, x_30);
lean_ctor_set(x_33, 3, x_32);
return x_33;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__9___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AlgEquiv_toRingEquiv___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__9(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__9___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AlgEquiv_toRingEquiv___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__9___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_homMk___at___00FGAlgCat_equivUnder_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_ObjectProperty_homMk___at___00FGAlgCat_equivUnder_spec__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_homMk___at___00FGAlgCat_equivUnder_spec__0___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_ObjectProperty_homMk___at___00FGAlgCat_equivUnder_spec__0___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__10___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RingEquiv_toMulEquiv___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__10(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__10___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RingEquiv_toMulEquiv___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__10___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__14___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__14(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_AlgHomClass_toAlgHom___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__8(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__18(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Functor_comp___at___00FGAlgCat_equivUnder_spec__5(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_NatIso_ofComponents___at___00FGAlgCat_equivUnder_spec__17(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__11___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MulEquiv_symm___at___00RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10_spec__11(x_1, x_2, x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RingEquiv_symm___at___00AlgEquiv_symm___at___00CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8_spec__9_spec__10(x_1, x_2, x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RingHom_id___at___00CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1_spec__1(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__4(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_ObjectProperty_isoMk___at___00FGAlgCat_equivUnder_spec__16___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CommAlgCat_isoMk___at___00FGAlgCat_equivUnder_spec__8(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_MorphismProperty_Comma_id___at___00CategoryTheory_NatTrans_id___at___00CategoryTheory_Iso_refl___at___00FGAlgCat_equivUnder_spec__19_spec__19_spec__19___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_CommAlgCat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_MorphismProperty_Comma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_FinitePresentation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_RingHomProperties(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_CommAlgCat_FiniteType(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_CommAlgCat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_MorphismProperty_Comma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_FinitePresentation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_RingHomProperties(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6___closed__0 = _init_lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6___closed__0();
lean_mark_persistent(lp_mathlib_MulEquiv_refl___at___00RingEquiv_refl___at___00FGAlgCat_equivUnder_spec__6_spec__6___closed__0);
lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1 = _init_lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1();
lean_mark_persistent(lp_mathlib_CategoryTheory_Functor_const___at___00FGAlgCat_equivUnder_spec__1);
lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3 = _init_lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3();
lean_mark_persistent(lp_mathlib_CategoryTheory_Functor_id___at___00FGAlgCat_equivUnder_spec__3);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
