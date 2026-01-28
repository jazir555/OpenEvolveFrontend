// Lean compiler output
// Module: Mathlib.CategoryTheory.Limits.IsLimit
// Imports: public import Init public import Mathlib.CategoryTheory.Adjunction.Basic public import Mathlib.CategoryTheory.Limits.Cones public import Batteries.Tactic.Congr
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_coneOfHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_fac___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCorepresentableBy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofFaithful(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfEquivalence___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_natIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_homOfCocone___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeInvEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mapConeEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_trans___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCorepresentableBy___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_postcomposeWhiskerLeftMapCone___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_coconeOfHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfEquivalence___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_isoWhiskerLeft___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCorepresentableBy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCoconeEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointUniqueUpToIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_uniqueUpToIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_homOfCocone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_corepresentableBy___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_colimitCocone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mapCoconeEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mapCoconeEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfEquivalence(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_uniqueUpToIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivOfNatIsoOfIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivOfNatIsoOfIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__20;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointUniqueUpToIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointUniqueUpToIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_natIso___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_equivalenceOfReindexing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_uniqueUpToIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofFaithful___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfNatIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfEquivalence(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCoconeEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCoconeEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCorepresentableBy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_forget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivOfNatIsoOfIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Adjunction_homEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_descCoconeMorphism___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_ulift(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mapCoconeEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfNatIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam;
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_natIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRepresentableBy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeInvEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mapCoconeEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_descCoconeMorphism___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_empty(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_postcomposeEquivalence___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofFaithful(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_uniqueUpToIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_liftConeMorphism___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_liftConeMorphism___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRepresentableBy___redArg(lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointUniqueUpToIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_limitCone___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalenceEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfEquivalence___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalenceEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_equivalenceOfReindexing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocone_category___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Iso_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_natIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_colimitCocone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofFaithful___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_limitCone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalenceEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfNatIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_forget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfNatIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mapConeEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalenceEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofConeEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfEquivalence___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocone_extend___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_corepresentableBy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Equivalence_toAdjunction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_representableBy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeInvEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_coneOfHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfEquivalence___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg___closed__0;
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__21;
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofFaithful___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__14;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_representableBy___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_colimitCocone___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_precomposeEquivalence___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cone_category___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofConeEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_coconeOfHom___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_CategoryTheory_types;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointUniqueUpToIso___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfNatIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_homOfCone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_homOfCocone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRepresentableBy___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_coneOfHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_natIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofFaithful___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointUniqueUpToIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cone_extend___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Equivalence_invFunIdAssoc___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_coconeOfHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__16;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_precomposeWhiskerLeftMapCocone___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_liftConeMorphism(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Equivalence_symm___redArg(lean_object*);
lean_object* lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofConeEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalenceEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_uniq___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofFaithful___redArg___boxed(lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_category___redArg(lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_uniq___autoParam;
lean_object* lp_mathlib_CategoryTheory_Iso_app___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__7;
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__5;
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__9;
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_homOfCone___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_uniqueUpToIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfEquivalence___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointUniqueUpToIso___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cones_whiskeringEquivalence___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mapConeEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalenceEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_natIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfNatIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_limitCone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeInvEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRepresentableBy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_uniqueUpToIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivOfNatIsoOfIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_descCoconeMorphism(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCocone___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__17;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalenceEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cocones_whiskeringEquivalence___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mapConeEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_homOfCone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointUniqueUpToIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalenceEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Functor_mapCone___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofFaithful___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lean", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Parser", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticSeq", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__3;
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__2;
x_3 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__1;
x_4 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticSeq1Indented", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__6;
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__2;
x_3 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__1;
x_4 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("null", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__8;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("CategoryTheory", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("cat_disch", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__11;
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__10;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__11;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__13;
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__14;
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__12;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__15;
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__16;
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__9;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__17;
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__18;
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__7;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__19;
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__21() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__20;
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__4;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__21;
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_uniq___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__21;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_map___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(x_1, x_2, x_3, x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_8, x_4);
x_10 = lean_apply_1(x_5, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsLimit_map___redArg(x_4, x_5, x_6, x_7, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_map___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsLimit_map(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_liftConeMorphism(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lean_apply_1(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_liftConeMorphism___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_liftConeMorphism___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_liftConeMorphism(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism___redArg(x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_uniqueUpToIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_4, x_1);
x_6 = lean_apply_1(x_3, x_2);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_uniqueUpToIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsLimit_uniqueUpToIso___redArg(x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_uniqueUpToIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsLimit_uniqueUpToIso(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointUniqueUpToIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lp_mathlib_CategoryTheory_Limits_Cones_forget(lean_box(0), x_1, lean_box(0), x_2, x_3);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_uniqueUpToIso___redArg(x_4, x_5, x_6, x_7);
x_10 = lp_mathlib_CategoryTheory_Functor_mapIso___redArg(x_8, x_4, x_5, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointUniqueUpToIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsLimit_conePointUniqueUpToIso___redArg(x_2, x_4, x_5, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointUniqueUpToIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsLimit_conePointUniqueUpToIso(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointUniqueUpToIso___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_conePointUniqueUpToIso___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_ctor_get(x_2, 2);
lean_inc(x_8);
lean_dec_ref(x_2);
x_9 = lean_ctor_get(x_6, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_3, 0);
lean_inc(x_10);
lean_dec_ref(x_3);
x_11 = lean_ctor_get(x_4, 0);
lean_inc(x_11);
lean_dec_ref(x_4);
x_12 = lean_apply_1(x_5, x_6);
x_13 = lean_apply_5(x_8, x_9, x_10, x_11, x_12, x_7);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg___lam__0), 6, 5);
lean_closure_set(x_6, 0, x_5);
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, x_2);
lean_closure_set(x_6, 3, x_3);
lean_closure_set(x_6, 4, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism___redArg___lam__0), 2, 1);
lean_closure_set(x_7, 0, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg(x_4, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg(x_1, x_2, x_3, x_5, x_4);
x_8 = lean_apply_1(x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_1);
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg(x_2, x_3, x_4, x_5, x_7);
x_9 = lean_apply_1(x_8, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg___lam__0), 6, 4);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_2);
lean_closure_set(x_5, 2, x_3);
lean_closure_set(x_5, 3, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg___lam__1), 6, 4);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, x_3);
lean_closure_set(x_6, 3, x_2);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg(x_4, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lean_ctor_get(x_1, 0);
lean_inc(x_9);
lean_inc_ref(x_8);
x_10 = lp_mathlib_CategoryTheory_Adjunction_homEquiv___redArg(x_2, x_3, x_1, x_4, x_5, x_8, x_6);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lean_apply_1(x_9, x_8);
x_13 = lean_apply_1(x_7, x_12);
x_14 = lean_apply_1(x_11, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lp_mathlib_CategoryTheory_Limits_Cone_category___redArg(x_1);
x_9 = lp_mathlib_CategoryTheory_Limits_Cone_category___redArg(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___redArg___lam__0), 8, 7);
lean_closure_set(x_10, 0, x_3);
lean_closure_set(x_10, 1, x_8);
lean_closure_set(x_10, 2, x_9);
lean_closure_set(x_10, 3, x_4);
lean_closure_set(x_10, 4, x_5);
lean_closure_set(x_10, 5, x_6);
lean_closure_set(x_10, 6, x_7);
x_11 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_mkConeMorphism___redArg___lam__0), 2, 1);
lean_closure_set(x_11, 0, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___redArg(x_6, x_9, x_11, x_12, x_13, x_14, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec_ref(x_10);
lean_dec_ref(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofConeEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_7 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_8);
x_9 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_7, 0);
x_11 = lean_ctor_get(x_8, 0);
lean_inc(x_11);
lean_inc_ref(x_2);
x_12 = lean_apply_1(x_11, x_2);
lean_inc(x_10);
lean_inc_ref(x_12);
x_13 = lean_apply_1(x_10, x_12);
x_14 = lp_mathlib_CategoryTheory_Equivalence_toAdjunction___redArg(x_1);
lean_inc_ref(x_3);
x_15 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___redArg(x_3, x_4, x_8, x_7, x_14, x_12, x_5);
x_16 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_9);
lean_inc_ref(x_2);
x_17 = lp_mathlib_CategoryTheory_Iso_app___redArg(x_16, x_2);
x_18 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg(x_3, x_13, x_2, x_15, x_17);
x_19 = lean_apply_1(x_18, x_6);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofConeEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_inc_ref(x_7);
x_9 = lp_mathlib_CategoryTheory_Equivalence_symm___redArg(x_7);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
x_11 = lean_ctor_get(x_9, 1);
lean_inc_ref(x_11);
lean_inc_ref(x_3);
lean_inc_ref(x_5);
lean_inc_ref(x_8);
x_12 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_ofConeEquiv___redArg___lam__0), 6, 4);
lean_closure_set(x_12, 0, x_7);
lean_closure_set(x_12, 1, x_8);
lean_closure_set(x_12, 2, x_5);
lean_closure_set(x_12, 3, x_3);
x_13 = lp_mathlib_CategoryTheory_Equivalence_toAdjunction___redArg(x_9);
x_14 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___boxed), 15, 14);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, x_1);
lean_closure_set(x_14, 2, lean_box(0));
lean_closure_set(x_14, 3, x_2);
lean_closure_set(x_14, 4, lean_box(0));
lean_closure_set(x_14, 5, x_3);
lean_closure_set(x_14, 6, x_4);
lean_closure_set(x_14, 7, lean_box(0));
lean_closure_set(x_14, 8, x_5);
lean_closure_set(x_14, 9, x_6);
lean_closure_set(x_14, 10, x_10);
lean_closure_set(x_14, 11, x_11);
lean_closure_set(x_14, 12, x_13);
lean_closure_set(x_14, 13, x_8);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_12);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofConeEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofConeEquiv___redArg(x_2, x_4, x_6, x_7, x_9, x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_7 = lp_mathlib_CategoryTheory_Limits_Cones_postcomposeEquivalence___redArg(x_2, x_3, x_4, x_5);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofConeEquiv___redArg(x_1, x_1, x_2, x_4, x_2, x_3, x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(x_2, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeInvEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_5);
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(x_1, x_2, x_4, x_3, x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeInvEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeInvEquiv___redArg(x_2, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivOfNatIsoOfIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_9 = lean_ctor_get(x_5, 0);
lean_inc(x_9);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_10 = lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(x_2, x_3, x_4, x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
lean_inc_ref(x_6);
lean_inc_ref(x_2);
x_12 = lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeHomEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
x_13 = lp_mathlib_Equiv_symm___redArg(x_12);
x_14 = lean_apply_1(x_11, x_6);
x_15 = lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg(x_2, x_14, x_7, x_8);
x_16 = lp_mathlib_Equiv_trans___redArg(x_13, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_equivOfNatIsoOfIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsLimit_equivOfNatIsoOfIso___redArg(x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfNatIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; 
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_ctor_get(x_8, 0);
x_11 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_12 = lp_mathlib_CategoryTheory_Limits_IsLimit_map___redArg(x_1, x_2, x_3, x_4, x_7, x_10);
x_13 = lp_mathlib_CategoryTheory_Limits_IsLimit_map___redArg(x_1, x_3, x_2, x_5, x_6, x_11);
lean_ctor_set(x_8, 1, x_13);
lean_ctor_set(x_8, 0, x_12);
return x_8;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_14 = lean_ctor_get(x_8, 0);
x_15 = lean_ctor_get(x_8, 1);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_8);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_16 = lp_mathlib_CategoryTheory_Limits_IsLimit_map___redArg(x_1, x_2, x_3, x_4, x_7, x_14);
x_17 = lp_mathlib_CategoryTheory_Limits_IsLimit_map___redArg(x_1, x_3, x_2, x_5, x_6, x_15);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
return x_18;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfNatIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfNatIso___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfNatIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfNatIso(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_2);
x_7 = lp_mathlib_CategoryTheory_Limits_Cones_whiskeringEquivalence___redArg(x_1, x_2, x_3, x_6);
x_8 = lp_mathlib_CategoryTheory_Equivalence_symm___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_10);
x_11 = lp_mathlib_CategoryTheory_Equivalence_toAdjunction___redArg(x_8);
lean_inc_ref(x_2);
x_12 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___redArg(x_2, x_2, x_9, x_10, x_11, x_4, x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence___redArg(x_2, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
lean_inc_ref(x_2);
x_7 = lp_mathlib_CategoryTheory_Limits_Cone_category___redArg(x_2);
lean_inc_ref(x_2);
x_8 = lp_mathlib_CategoryTheory_Limits_Cones_whiskeringEquivalence___redArg(x_1, x_2, x_3, x_5);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_10);
x_11 = lean_ctor_get(x_8, 2);
lean_inc_ref(x_11);
lean_inc_ref(x_10);
lean_inc_ref(x_9);
x_12 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_9, x_10);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_7);
lean_dec_ref(x_7);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_4);
x_17 = lean_apply_1(x_13, x_4);
lean_inc_ref(x_4);
x_18 = lean_apply_1(x_15, x_4);
lean_inc_ref(x_4);
x_19 = lp_mathlib_CategoryTheory_Iso_app___redArg(x_11, x_4);
x_20 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_19);
lean_inc_ref(x_2);
x_21 = lp_mathlib_CategoryTheory_Limits_IsLimit_equivIsoLimit___redArg(x_2, x_17, x_18, x_20);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
lean_dec_ref(x_21);
x_23 = lp_mathlib_CategoryTheory_Equivalence_toAdjunction___redArg(x_8);
lean_inc(x_16);
x_24 = lean_apply_1(x_16, x_4);
lean_inc_ref(x_2);
x_25 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofRightAdjoint___redArg(x_2, x_2, x_9, x_10, x_23, x_24, x_6);
x_26 = lean_apply_1(x_22, x_25);
return x_26;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence___redArg(x_2, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalenceEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalence___redArg(x_1, x_2, x_3, x_4, x_6, x_5);
x_9 = lean_apply_1(x_8, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalenceEquiv___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalenceEquiv___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalenceEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalenceEquiv___redArg___lam__0___boxed), 7, 5);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_3);
lean_closure_set(x_7, 2, x_4);
lean_closure_set(x_7, 3, x_5);
lean_closure_set(x_7, 4, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_ofWhiskerEquivalence___boxed), 10, 9);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_1);
lean_closure_set(x_8, 2, lean_box(0));
lean_closure_set(x_8, 3, x_2);
lean_closure_set(x_8, 4, lean_box(0));
lean_closure_set(x_8, 5, x_3);
lean_closure_set(x_8, 6, x_4);
lean_closure_set(x_8, 7, x_5);
lean_closure_set(x_8, 8, x_6);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalenceEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsLimit_whiskerEquivalenceEquiv___redArg(x_2, x_4, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfEquivalence___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; uint8_t x_28; 
lean_inc_ref(x_3);
x_12 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_3);
x_13 = lean_ctor_get(x_10, 0);
x_14 = lean_ctor_get(x_10, 1);
lean_inc_ref(x_4);
lean_inc_ref(x_14);
x_15 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_14, x_4);
lean_inc_ref(x_6);
lean_inc_ref(x_13);
x_16 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_13, x_6);
lean_inc_ref(x_16);
lean_inc_ref(x_14);
x_17 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_14, x_16);
lean_inc_ref(x_11);
lean_inc_ref(x_4);
lean_inc_ref(x_14);
x_18 = lp_mathlib_CategoryTheory_Functor_isoWhiskerLeft___redArg(x_2, x_1, x_3, x_14, x_16, x_4, x_11);
x_19 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_18);
lean_inc_ref(x_6);
lean_inc_ref(x_10);
lean_inc_ref(x_3);
x_20 = lp_mathlib_CategoryTheory_Equivalence_invFunIdAssoc___redArg(x_2, x_3, x_10, x_6);
lean_inc_ref(x_6);
x_21 = lp_mathlib_CategoryTheory_Iso_trans___redArg(x_12, x_15, x_17, x_6, x_19, x_20);
lean_inc_ref(x_10);
x_22 = lp_mathlib_CategoryTheory_Equivalence_symm___redArg(x_10);
lean_inc_ref(x_6);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
x_23 = lp_mathlib_CategoryTheory_Limits_Cones_equivalenceOfReindexing___redArg(x_1, x_3, x_4, x_6, x_22, x_21);
x_24 = lean_ctor_get(x_23, 0);
lean_inc_ref(x_24);
lean_dec_ref(x_23);
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
lean_dec_ref(x_24);
x_26 = lp_mathlib_CategoryTheory_Limits_Cones_equivalenceOfReindexing___redArg(x_2, x_3, x_6, x_4, x_10, x_11);
x_27 = lean_ctor_get(x_26, 0);
lean_inc_ref(x_27);
lean_dec_ref(x_26);
x_28 = !lean_is_exclusive(x_27);
if (x_28 == 0)
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_29 = lean_ctor_get(x_27, 0);
x_30 = lean_ctor_get(x_27, 1);
lean_dec(x_30);
x_31 = lean_apply_1(x_25, x_5);
x_32 = lean_apply_1(x_9, x_31);
x_33 = lean_apply_1(x_29, x_7);
x_34 = lean_apply_1(x_8, x_33);
lean_ctor_set(x_27, 1, x_34);
lean_ctor_set(x_27, 0, x_32);
return x_27;
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_35 = lean_ctor_get(x_27, 0);
lean_inc(x_35);
lean_dec(x_27);
x_36 = lean_apply_1(x_25, x_5);
x_37 = lean_apply_1(x_9, x_36);
x_38 = lean_apply_1(x_35, x_7);
x_39 = lean_apply_1(x_8, x_38);
x_40 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_40, 0, x_37);
lean_ctor_set(x_40, 1, x_39);
return x_40;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfEquivalence(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfEquivalence___redArg(x_2, x_4, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfEquivalence___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfEquivalence(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfEquivalence___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Limits_IsLimit_conePointsIsoOfEquivalence___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_mathlib_CategoryTheory_Limits_Cone_extend___redArg(x_1, x_2, x_3, x_4, x_5);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_8, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_1);
lean_ctor_set(x_4, 1, x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___redArg___lam__0), 6, 4);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_2);
lean_closure_set(x_6, 2, x_3);
lean_closure_set(x_6, 3, x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___redArg___lam__1), 3, 2);
lean_closure_set(x_7, 0, x_5);
lean_closure_set(x_7, 1, x_4);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_2);
return x_9;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_ulift(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg___closed__0;
x_7 = lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___redArg(x_1, x_2, x_3, x_4, x_5);
x_8 = lp_mathlib_Equiv_trans___redArg(x_6, x_7);
x_9 = lp_mathlib_Equiv_toIso___redArg(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_homIso(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_natIso___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg(x_1, x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_natIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_natIso___redArg___lam__0), 5, 4);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_2);
lean_closure_set(x_5, 2, x_3);
lean_closure_set(x_5, 3, x_4);
x_6 = lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_natIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_natIso___redArg(x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_natIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_natIso(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___redArg___lam__0), 2, 0);
x_7 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___redArg___lam__1), 2, 0);
x_8 = lp_mathlib_CategoryTheory_types;
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg(x_1, x_2, x_3, x_4, x_5);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_6);
lean_ctor_set(x_10, 1, x_7);
x_11 = lp_mathlib_CategoryTheory_Iso_trans___redArg(x_8, lean_box(0), lean_box(0), lean_box(0), x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofFaithful(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_inc(x_12);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofFaithful___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofFaithful___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofFaithful(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofFaithful___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofFaithful___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mapConeEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
lean_inc_ref(x_7);
x_10 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_7);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_12 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_4, x_5);
lean_inc_ref(x_6);
lean_inc_ref(x_4);
x_13 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_4, x_6);
lean_inc_ref(x_4);
x_14 = lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(x_4, x_11);
lean_inc_ref(x_12);
lean_inc_ref(x_13);
lean_inc_ref(x_3);
x_15 = lp_mathlib_CategoryTheory_Limits_Cones_postcompose___redArg(x_3, x_13, x_12, x_14);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_17 = lp_mathlib_CategoryTheory_Functor_isoWhiskerLeft___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_inc_ref(x_8);
lean_inc_ref(x_4);
x_18 = lp_mathlib_CategoryTheory_Functor_mapCone___redArg(x_6, x_4, x_8);
lean_inc_ref(x_18);
lean_inc_ref(x_3);
x_19 = lp_mathlib_CategoryTheory_Limits_IsLimit_postcomposeInvEquiv___redArg(x_1, x_3, x_12, x_13, x_17, x_18);
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
lean_inc_ref(x_8);
x_21 = lp_mathlib_CategoryTheory_Functor_mapCone___redArg(x_5, x_4, x_8);
x_22 = lean_apply_1(x_16, x_18);
x_23 = lp_mathlib_CategoryTheory_Functor_postcomposeWhiskerLeftMapCone___redArg(x_10, x_8);
x_24 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_23);
x_25 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofIsoLimit___redArg(x_3, x_21, x_22, x_9, x_24);
x_26 = lean_apply_1(x_20, x_25);
return x_26;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mapConeEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Limits_IsLimit_mapConeEquiv___redArg(x_2, x_4, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mapConeEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Limits_IsLimit_mapConeEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_mapConeEquiv___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsLimit_mapConeEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0___redArg___lam__0), 2, 0);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___elam__0___boxed), 7, 6);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_1);
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, x_3);
lean_closure_set(x_6, 5, x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Limits_IsLimit_isoUniqueConeMorphism___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_coneOfHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
lean_inc(x_2);
x_4 = lean_apply_1(x_1, x_2);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_dec(x_7);
x_8 = lean_apply_1(x_6, x_3);
lean_ctor_set(x_4, 1, x_8);
lean_ctor_set(x_4, 0, x_2);
return x_4;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_4, 0);
lean_inc(x_9);
lean_dec(x_4);
x_10 = lean_apply_1(x_9, x_3);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_2);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_coneOfHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_coneOfHom___redArg(x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_coneOfHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_coneOfHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_homOfCone___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_apply_1(x_1, x_3);
x_6 = lp_mathlib_Equiv_symm___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_7, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_homOfCone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_homOfCone___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_homOfCone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_homOfCone(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_limitCone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
lean_inc(x_2);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_coneOfHom___redArg(x_3, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_limitCone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_limitCone___redArg(x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_limitCone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_limitCone(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRepresentableBy___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Limits_IsLimit_OfNatIso_homOfCone___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRepresentableBy___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_ofRepresentableBy___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRepresentableBy(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofRepresentableBy___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_ofRepresentableBy___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsLimit_ofRepresentableBy(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_representableBy(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___boxed), 8, 7);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, lean_box(0));
lean_closure_set(x_8, 3, x_4);
lean_closure_set(x_8, 4, x_5);
lean_closure_set(x_8, 5, x_6);
lean_closure_set(x_8, 6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsLimit_representableBy___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_homEquiv___boxed), 8, 7);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, lean_box(0));
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, x_3);
lean_closure_set(x_6, 5, x_4);
lean_closure_set(x_6, 6, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsColimit_fac___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__21;
return x_1;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsColimit_uniq___autoParam() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__21;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_map___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(x_1, x_3, x_2, x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_8, x_5);
x_10 = lean_apply_1(x_4, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsColimit_map___redArg(x_4, x_5, x_6, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_map___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsColimit_map(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_7);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_descCoconeMorphism(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lean_apply_1(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_descCoconeMorphism___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_descCoconeMorphism___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_descCoconeMorphism(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism___redArg(x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_uniqueUpToIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_3, x_2);
x_6 = lean_apply_1(x_4, x_1);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_uniqueUpToIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsColimit_uniqueUpToIso___redArg(x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_uniqueUpToIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsColimit_uniqueUpToIso(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointUniqueUpToIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lp_mathlib_CategoryTheory_Limits_Cocones_forget(lean_box(0), x_1, lean_box(0), x_2, x_3);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_uniqueUpToIso___redArg(x_4, x_5, x_6, x_7);
x_10 = lp_mathlib_CategoryTheory_Functor_mapIso___redArg(x_8, x_4, x_5, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointUniqueUpToIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointUniqueUpToIso___redArg(x_2, x_4, x_5, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointUniqueUpToIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointUniqueUpToIso(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointUniqueUpToIso___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointUniqueUpToIso___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_ctor_get(x_2, 2);
lean_inc(x_8);
lean_dec_ref(x_2);
x_9 = lean_ctor_get(x_3, 0);
lean_inc(x_9);
lean_dec_ref(x_3);
x_10 = lean_ctor_get(x_4, 0);
lean_inc(x_10);
lean_dec_ref(x_4);
x_11 = lean_ctor_get(x_6, 0);
lean_inc(x_11);
x_12 = lean_apply_1(x_5, x_6);
x_13 = lean_apply_5(x_8, x_9, x_10, x_11, x_7, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg___lam__0), 6, 5);
lean_closure_set(x_6, 0, x_5);
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, x_3);
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism___redArg___lam__0), 2, 1);
lean_closure_set(x_7, 0, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg(x_4, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg(x_1, x_2, x_3, x_5, x_4);
x_8 = lean_apply_1(x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_1);
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg(x_2, x_3, x_4, x_5, x_7);
x_9 = lean_apply_1(x_8, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg___lam__0), 6, 4);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_2);
lean_closure_set(x_5, 2, x_3);
lean_closure_set(x_5, 3, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg___lam__1), 6, 4);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, x_3);
lean_closure_set(x_6, 3, x_2);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg(x_4, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_9 = lean_ctor_get(x_1, 0);
lean_inc(x_9);
lean_inc_ref(x_8);
x_10 = lp_mathlib_CategoryTheory_Adjunction_homEquiv___redArg(x_2, x_3, x_4, x_1, x_5, x_6, x_8);
x_11 = lp_mathlib_Equiv_symm___redArg(x_10);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lean_apply_1(x_9, x_8);
x_14 = lean_apply_1(x_7, x_13);
x_15 = lean_apply_1(x_12, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lp_mathlib_CategoryTheory_Limits_Cocone_category___redArg(x_2);
x_9 = lp_mathlib_CategoryTheory_Limits_Cocone_category___redArg(x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___redArg___lam__0), 8, 7);
lean_closure_set(x_10, 0, x_4);
lean_closure_set(x_10, 1, x_8);
lean_closure_set(x_10, 2, x_9);
lean_closure_set(x_10, 3, x_3);
lean_closure_set(x_10, 4, x_5);
lean_closure_set(x_10, 5, x_6);
lean_closure_set(x_10, 6, x_7);
x_11 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_mkCoconeMorphism___redArg___lam__0), 2, 1);
lean_closure_set(x_11, 0, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___redArg(x_6, x_9, x_11, x_12, x_13, x_14, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec_ref(x_10);
lean_dec_ref(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCoconeEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_9 = lp_mathlib_CategoryTheory_Equivalence_symm___redArg(x_1);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
x_11 = lean_ctor_get(x_9, 1);
lean_inc_ref(x_11);
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_2, 0);
lean_inc(x_13);
lean_dec_ref(x_2);
lean_inc_ref(x_3);
x_14 = lean_apply_1(x_13, x_3);
lean_inc(x_12);
lean_inc_ref(x_14);
x_15 = lean_apply_1(x_12, x_14);
x_16 = lp_mathlib_CategoryTheory_Equivalence_toAdjunction___redArg(x_9);
lean_inc_ref(x_4);
x_17 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___redArg(x_4, x_5, x_10, x_11, x_16, x_14, x_7);
x_18 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_6);
lean_inc_ref(x_3);
x_19 = lp_mathlib_CategoryTheory_Iso_app___redArg(x_18, x_3);
x_20 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg(x_4, x_15, x_3, x_17, x_19);
x_21 = lean_apply_1(x_20, x_8);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCoconeEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_9 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_7, 1);
lean_inc_ref(x_10);
x_11 = lean_ctor_get(x_7, 2);
lean_inc_ref(x_11);
lean_inc_ref(x_3);
lean_inc_ref(x_5);
lean_inc_ref(x_8);
lean_inc_ref(x_9);
lean_inc_ref(x_7);
x_12 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_ofCoconeEquiv___redArg___lam__0), 8, 6);
lean_closure_set(x_12, 0, x_7);
lean_closure_set(x_12, 1, x_9);
lean_closure_set(x_12, 2, x_8);
lean_closure_set(x_12, 3, x_5);
lean_closure_set(x_12, 4, x_3);
lean_closure_set(x_12, 5, x_11);
x_13 = lp_mathlib_CategoryTheory_Equivalence_toAdjunction___redArg(x_7);
x_14 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___boxed), 15, 14);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, x_1);
lean_closure_set(x_14, 2, lean_box(0));
lean_closure_set(x_14, 3, x_2);
lean_closure_set(x_14, 4, lean_box(0));
lean_closure_set(x_14, 5, x_3);
lean_closure_set(x_14, 6, x_4);
lean_closure_set(x_14, 7, lean_box(0));
lean_closure_set(x_14, 8, x_5);
lean_closure_set(x_14, 9, x_6);
lean_closure_set(x_14, 10, x_9);
lean_closure_set(x_14, 11, x_10);
lean_closure_set(x_14, 12, x_13);
lean_closure_set(x_14, 13, x_8);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_12);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCoconeEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofCoconeEquiv___redArg(x_2, x_4, x_6, x_7, x_9, x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_3);
lean_inc_ref(x_4);
lean_inc_ref(x_2);
x_7 = lp_mathlib_CategoryTheory_Limits_Cocones_precomposeEquivalence___redArg(x_2, x_4, x_3, x_5);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofCoconeEquiv___redArg(x_1, x_1, x_2, x_3, x_2, x_4, x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(x_2, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeInvEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_5);
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeHomEquiv___redArg(x_1, x_2, x_4, x_3, x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeInvEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeInvEquiv___redArg(x_2, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivOfNatIsoOfIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_9 = lean_ctor_get(x_5, 1);
lean_inc(x_9);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_10 = lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(x_2, x_3, x_4, x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
lean_inc_ref(x_6);
lean_inc_ref(x_2);
x_12 = lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeInvEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
x_13 = lp_mathlib_Equiv_symm___redArg(x_12);
x_14 = lean_apply_1(x_11, x_6);
x_15 = lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg(x_2, x_14, x_7, x_8);
x_16 = lp_mathlib_Equiv_trans___redArg(x_13, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_equivOfNatIsoOfIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsColimit_equivOfNatIsoOfIso___redArg(x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfNatIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; 
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_ctor_get(x_8, 0);
x_11 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_12 = lp_mathlib_CategoryTheory_Limits_IsColimit_map___redArg(x_1, x_2, x_3, x_6, x_5, x_10);
x_13 = lp_mathlib_CategoryTheory_Limits_IsColimit_map___redArg(x_1, x_3, x_2, x_7, x_4, x_11);
lean_ctor_set(x_8, 1, x_13);
lean_ctor_set(x_8, 0, x_12);
return x_8;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_14 = lean_ctor_get(x_8, 0);
x_15 = lean_ctor_get(x_8, 1);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_8);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_16 = lp_mathlib_CategoryTheory_Limits_IsColimit_map___redArg(x_1, x_2, x_3, x_6, x_5, x_14);
x_17 = lp_mathlib_CategoryTheory_Limits_IsColimit_map___redArg(x_1, x_3, x_2, x_7, x_4, x_15);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
return x_18;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfNatIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfNatIso___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfNatIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfNatIso(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_2);
x_7 = lp_mathlib_CategoryTheory_Limits_Cocones_whiskeringEquivalence___redArg(x_1, x_2, x_3, x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
x_9 = lean_ctor_get(x_7, 1);
lean_inc_ref(x_9);
x_10 = lp_mathlib_CategoryTheory_Equivalence_toAdjunction___redArg(x_7);
lean_inc_ref(x_2);
x_11 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___redArg(x_2, x_2, x_8, x_9, x_10, x_4, x_5);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence___redArg(x_2, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
lean_inc_ref(x_2);
x_7 = lp_mathlib_CategoryTheory_Limits_Cocone_category___redArg(x_2);
lean_inc_ref(x_2);
x_8 = lp_mathlib_CategoryTheory_Limits_Cocones_whiskeringEquivalence___redArg(x_1, x_2, x_3, x_5);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_10);
x_11 = lean_ctor_get(x_8, 2);
lean_inc_ref(x_11);
lean_inc_ref(x_9);
x_12 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_9, x_10);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_CategoryTheory_Functor_id(lean_box(0), x_7);
lean_dec_ref(x_7);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lp_mathlib_CategoryTheory_Equivalence_symm___redArg(x_8);
x_17 = lean_ctor_get(x_16, 0);
lean_inc_ref(x_17);
x_18 = lean_ctor_get(x_16, 1);
lean_inc_ref(x_18);
x_19 = lean_ctor_get(x_9, 0);
lean_inc(x_19);
lean_dec_ref(x_9);
lean_inc_ref(x_4);
x_20 = lean_apply_1(x_13, x_4);
lean_inc_ref(x_4);
x_21 = lean_apply_1(x_15, x_4);
lean_inc_ref(x_4);
x_22 = lp_mathlib_CategoryTheory_Iso_app___redArg(x_11, x_4);
x_23 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_22);
lean_inc_ref(x_2);
x_24 = lp_mathlib_CategoryTheory_Limits_IsColimit_equivIsoColimit___redArg(x_2, x_20, x_21, x_23);
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
lean_dec_ref(x_24);
x_26 = lp_mathlib_CategoryTheory_Equivalence_toAdjunction___redArg(x_16);
x_27 = lean_apply_1(x_19, x_4);
lean_inc_ref(x_2);
x_28 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofLeftAdjoint___redArg(x_2, x_2, x_17, x_18, x_26, x_27, x_6);
x_29 = lean_apply_1(x_25, x_28);
return x_29;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence___redArg(x_2, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalenceEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalence___redArg(x_1, x_2, x_3, x_4, x_6, x_5);
x_9 = lean_apply_1(x_8, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalenceEquiv___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalenceEquiv___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalenceEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalenceEquiv___redArg___lam__0___boxed), 7, 5);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_3);
lean_closure_set(x_7, 2, x_4);
lean_closure_set(x_7, 3, x_5);
lean_closure_set(x_7, 4, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_ofWhiskerEquivalence___boxed), 10, 9);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_1);
lean_closure_set(x_8, 2, lean_box(0));
lean_closure_set(x_8, 3, x_2);
lean_closure_set(x_8, 4, lean_box(0));
lean_closure_set(x_8, 5, x_3);
lean_closure_set(x_8, 6, x_4);
lean_closure_set(x_8, 7, x_5);
lean_closure_set(x_8, 8, x_6);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalenceEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsColimit_whiskerEquivalenceEquiv___redArg(x_2, x_4, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfEquivalence___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; uint8_t x_28; 
lean_inc_ref(x_3);
x_12 = lp_mathlib_CategoryTheory_Functor_category___redArg(x_3);
x_13 = lean_ctor_get(x_10, 0);
x_14 = lean_ctor_get(x_10, 1);
lean_inc_ref(x_11);
lean_inc_ref(x_10);
lean_inc_ref(x_4);
lean_inc_ref(x_6);
lean_inc_ref(x_3);
x_15 = lp_mathlib_CategoryTheory_Limits_Cocones_equivalenceOfReindexing___redArg(x_2, x_3, x_6, x_4, x_10, x_11);
x_16 = lean_ctor_get(x_15, 0);
lean_inc_ref(x_16);
lean_dec_ref(x_15);
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
lean_inc_ref(x_4);
lean_inc_ref(x_14);
x_18 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_14, x_4);
lean_inc_ref(x_6);
lean_inc_ref(x_13);
x_19 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_13, x_6);
lean_inc_ref(x_19);
lean_inc_ref(x_14);
x_20 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_14, x_19);
lean_inc_ref(x_4);
lean_inc_ref(x_14);
x_21 = lp_mathlib_CategoryTheory_Functor_isoWhiskerLeft___redArg(x_2, x_1, x_3, x_14, x_19, x_4, x_11);
x_22 = lp_mathlib_CategoryTheory_Iso_symm___redArg(x_21);
lean_inc_ref(x_6);
lean_inc_ref(x_10);
lean_inc_ref(x_3);
x_23 = lp_mathlib_CategoryTheory_Equivalence_invFunIdAssoc___redArg(x_2, x_3, x_10, x_6);
lean_inc_ref(x_6);
x_24 = lp_mathlib_CategoryTheory_Iso_trans___redArg(x_12, x_18, x_20, x_6, x_22, x_23);
x_25 = lp_mathlib_CategoryTheory_Equivalence_symm___redArg(x_10);
x_26 = lp_mathlib_CategoryTheory_Limits_Cocones_equivalenceOfReindexing___redArg(x_1, x_3, x_4, x_6, x_25, x_24);
x_27 = lean_ctor_get(x_26, 0);
lean_inc_ref(x_27);
lean_dec_ref(x_26);
x_28 = !lean_is_exclusive(x_27);
if (x_28 == 0)
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_29 = lean_ctor_get(x_27, 0);
x_30 = lean_ctor_get(x_27, 1);
lean_dec(x_30);
x_31 = lean_apply_1(x_17, x_7);
x_32 = lean_apply_1(x_8, x_31);
x_33 = lean_apply_1(x_29, x_5);
x_34 = lean_apply_1(x_9, x_33);
lean_ctor_set(x_27, 1, x_34);
lean_ctor_set(x_27, 0, x_32);
return x_27;
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_35 = lean_ctor_get(x_27, 0);
lean_inc(x_35);
lean_dec(x_27);
x_36 = lean_apply_1(x_17, x_7);
x_37 = lean_apply_1(x_8, x_36);
x_38 = lean_apply_1(x_35, x_5);
x_39 = lean_apply_1(x_9, x_38);
x_40 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_40, 0, x_37);
lean_ctor_set(x_40, 1, x_39);
return x_40;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfEquivalence(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfEquivalence___redArg(x_2, x_4, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfEquivalence___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfEquivalence(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfEquivalence___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Limits_IsColimit_coconePointsIsoOfEquivalence___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_mathlib_CategoryTheory_Limits_Cocone_extend___redArg(x_1, x_2, x_3, x_4, x_5);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_8, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_1);
lean_ctor_set(x_4, 1, x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___redArg___lam__0), 6, 4);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_2);
lean_closure_set(x_6, 2, x_3);
lean_closure_set(x_6, 3, x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___redArg___lam__1), 3, 2);
lean_closure_set(x_7, 0, x_5);
lean_closure_set(x_7, 1, x_4);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg___closed__0;
x_7 = lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___redArg(x_1, x_2, x_3, x_4, x_5);
x_8 = lp_mathlib_Equiv_trans___redArg(x_6, x_7);
x_9 = lp_mathlib_Equiv_toIso___redArg(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_homIso___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_homIso(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_natIso___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_homIso___boxed), 8, 7);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, lean_box(0));
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, x_3);
lean_closure_set(x_6, 5, x_4);
lean_closure_set(x_6, 6, x_5);
x_7 = lp_mathlib_CategoryTheory_NatIso_ofComponents___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_natIso(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_natIso___redArg(x_2, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsLimit_homIso_x27___redArg___lam__1), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg___lam__0), 2, 0);
x_7 = lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg___closed__0;
x_8 = lp_mathlib_CategoryTheory_types;
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_homIso___redArg(x_1, x_2, x_3, x_4, x_5);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_6);
lean_ctor_set(x_10, 1, x_7);
x_11 = lp_mathlib_CategoryTheory_Iso_trans___redArg(x_8, lean_box(0), lean_box(0), lean_box(0), x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofFaithful(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_inc(x_12);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofFaithful___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofFaithful___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofFaithful(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofFaithful___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofFaithful___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mapCoconeEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_10 = lean_ctor_get(x_7, 1);
lean_inc_ref(x_6);
lean_inc_ref(x_4);
x_11 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_4, x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_12 = lp_mathlib_CategoryTheory_Functor_comp___redArg(x_4, x_5);
lean_inc(x_10);
lean_inc_ref(x_4);
x_13 = lp_mathlib_CategoryTheory_Functor_whiskerLeft___redArg(x_4, x_10);
lean_inc_ref(x_11);
lean_inc_ref(x_12);
lean_inc_ref(x_3);
x_14 = lp_mathlib_CategoryTheory_Limits_Cocones_precompose___redArg(x_3, x_12, x_11, x_13);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc_ref(x_8);
lean_inc_ref(x_4);
lean_inc_ref(x_5);
x_16 = lp_mathlib_CategoryTheory_Functor_mapCocone___redArg(x_5, x_4, x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_4);
x_17 = lp_mathlib_CategoryTheory_Functor_isoWhiskerLeft___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_inc_ref(x_16);
lean_inc_ref(x_3);
x_18 = lp_mathlib_CategoryTheory_Limits_IsColimit_precomposeInvEquiv___redArg(x_1, x_3, x_12, x_11, x_17, x_16);
x_19 = lp_mathlib_Equiv_symm___redArg(x_18);
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lean_apply_1(x_15, x_16);
lean_inc_ref(x_8);
x_22 = lp_mathlib_CategoryTheory_Functor_mapCocone___redArg(x_6, x_4, x_8);
x_23 = lean_apply_1(x_20, x_9);
x_24 = lp_mathlib_CategoryTheory_Functor_precomposeWhiskerLeftMapCocone___redArg(x_7, x_8);
x_25 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofIsoColimit___redArg(x_3, x_21, x_22, x_23, x_24);
return x_25;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mapCoconeEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Limits_IsColimit_mapCoconeEquiv___redArg(x_2, x_4, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mapCoconeEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Limits_IsColimit_mapCoconeEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_mapCoconeEquiv___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsColimit_mapCoconeEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0___redArg___lam__0), 2, 0);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___elam__0___boxed), 7, 6);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_1);
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, x_3);
lean_closure_set(x_6, 5, x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Limits_IsColimit_isoUniqueCoconeMorphism___redArg(x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_coconeOfHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
lean_inc(x_2);
x_4 = lean_apply_1(x_1, x_2);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_dec(x_7);
x_8 = lean_apply_1(x_6, x_3);
lean_ctor_set(x_4, 1, x_8);
lean_ctor_set(x_4, 0, x_2);
return x_4;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_4, 0);
lean_inc(x_9);
lean_dec(x_4);
x_10 = lean_apply_1(x_9, x_3);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_2);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_coconeOfHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_coconeOfHom___redArg(x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_coconeOfHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_coconeOfHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_homOfCocone___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_apply_1(x_1, x_3);
x_6 = lp_mathlib_Equiv_symm___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_7, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_homOfCocone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_homOfCocone___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_homOfCocone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_homOfCocone(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_colimitCocone___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
lean_inc(x_2);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_coconeOfHom___redArg(x_3, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_colimitCocone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_colimitCocone___redArg(x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_colimitCocone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_colimitCocone(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCorepresentableBy___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CategoryTheory_Limits_IsColimit_OfNatIso_homOfCocone___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCorepresentableBy___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_ofCorepresentableBy___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCorepresentableBy(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofCorepresentableBy___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_ofCorepresentableBy___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Limits_IsColimit_ofCorepresentableBy(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_corepresentableBy(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___boxed), 8, 7);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, lean_box(0));
lean_closure_set(x_8, 3, x_4);
lean_closure_set(x_8, 4, x_5);
lean_closure_set(x_8, 5, x_6);
lean_closure_set(x_8, 6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Limits_IsColimit_corepresentableBy___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Limits_IsColimit_homEquiv___boxed), 8, 7);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, lean_box(0));
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, x_3);
lean_closure_set(x_6, 5, x_4);
lean_closure_set(x_6, 6, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Cones(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Congr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_IsLimit(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Adjunction_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Cones(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Congr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__0 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__0);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__1 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__1();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__1);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__2 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__2();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__2);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__3 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__3();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__3);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__4 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__4();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__4);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__5 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__5();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__5);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__6 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__6();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__6);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__7 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__7();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__7);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__8 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__8();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__8);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__9 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__9();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__9);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__10 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__10();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__10);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__11 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__11();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__11);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__12 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__12();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__12);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__13 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__13();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__13);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__14 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__14();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__14);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__15 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__15();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__15);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__16 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__16();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__16);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__17 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__17();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__17);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__18 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__18();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__18);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__19 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__19();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__19);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__20 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__20();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__20);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__21 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__21();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam___closed__21);
lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_fac___autoParam);
lp_mathlib_CategoryTheory_Limits_IsLimit_uniq___autoParam = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_uniq___autoParam();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_uniq___autoParam);
lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg___closed__0 = _init_lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsLimit_homIso___redArg___closed__0);
lp_mathlib_CategoryTheory_Limits_IsColimit_fac___autoParam = _init_lp_mathlib_CategoryTheory_Limits_IsColimit_fac___autoParam();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsColimit_fac___autoParam);
lp_mathlib_CategoryTheory_Limits_IsColimit_uniq___autoParam = _init_lp_mathlib_CategoryTheory_Limits_IsColimit_uniq___autoParam();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsColimit_uniq___autoParam);
lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg___closed__0 = _init_lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_Limits_IsColimit_homIso_x27___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
