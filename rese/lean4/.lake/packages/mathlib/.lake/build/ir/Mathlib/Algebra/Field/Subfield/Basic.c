// Lean compiler output
// Module: Mathlib.Algebra.Field.Subfield.Basic
// Imports: public import Init public import Mathlib.Algebra.Algebra.Defs public import Mathlib.Algebra.Field.Subfield.Defs public import Mathlib.Algebra.GroupWithZero.Units.Lemmas public import Mathlib.Algebra.Ring.Subring.Basic public import Mathlib.RingTheory.SimpleRing.Basic
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
static lean_object* lp_mathlib_Subfield_instCompleteLattice___closed__1;
lean_object* lp_mathlib_RingHom_codRestrict___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_Set_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet___lam__0(lean_object*);
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocusField___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subfieldCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_closure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin(lean_object*, lean_object*);
static lean_object* lp_mathlib_Subfield_instCompleteLattice___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocusField(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv(lean_object*, lean_object*);
static lean_object* lp_mathlib_RingEquiv_subfieldCongr___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Subfield_inclusion___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subfieldCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_closure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subfield_subtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Field_Subfield_Basic_0__Subfield_commClosure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submonoid_instSMulSubtypeMem___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subsemiring_topEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_SetLike_instPartialOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subfield_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem___redArg(lean_object*);
static lean_object* lp_mathlib_Subfield_instCompleteLattice___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fieldRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fieldRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Field_Subfield_Basic_0__Subfield_commClosure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Subfield_inclusion___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instTop(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instTop___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subfield_instTop(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInhabited___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subfield_instInhabited(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_2, 0);
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_3);
x_5 = lp_mathlib_Subsemiring_topEquiv(lean_box(0), x_4);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subfield_topEquiv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subfield_topEquiv(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Subfield_topEquiv___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_comap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_comap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subfield_comap(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_map___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subfield_map(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fieldRange(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fieldRange___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RingHom_fieldRange(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_RingHom_fintypeFieldRange___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lp_mathlib_PLift_fintype___redArg(x_1);
x_6 = lp_mathlib_Set_fintypeRange___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_RingHom_fintypeFieldRange___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_RingHom_fintypeFieldRange(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subfield_instMin___lam__0), 2, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subfield_instMin(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subfield_instInfSet___lam__0), 1, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subfield_instInfSet(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Subfield_instCompleteLattice___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lp_mathlib_SetLike_instPartialOrder(lean_box(0), lean_box(0), x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Subfield_instCompleteLattice___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Subfield_instInfSet___lam__0), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Subfield_instCompleteLattice___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Subfield_instCompleteLattice___closed__1;
x_2 = lp_mathlib_Subfield_instCompleteLattice___closed__0;
x_3 = lp_mathlib_completeLatticeOfInf___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lp_mathlib_Subfield_instCompleteLattice___closed__2;
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_ctor_get(x_3, 3);
x_8 = lean_ctor_get(x_5, 1);
lean_dec(x_8);
x_9 = !lean_is_exclusive(x_7);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_10 = lean_ctor_get(x_7, 0);
lean_dec(x_10);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Subfield_instCompleteLattice___lam__0), 2, 0);
x_12 = lean_box(0);
lean_ctor_set(x_5, 1, x_11);
lean_ctor_set(x_7, 0, x_12);
return x_3;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_13 = lean_ctor_get(x_7, 1);
lean_inc(x_13);
lean_dec(x_7);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Subfield_instCompleteLattice___lam__0), 2, 0);
x_15 = lean_box(0);
lean_ctor_set(x_5, 1, x_14);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_13);
lean_ctor_set(x_3, 3, x_16);
return x_3;
}
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_17 = lean_ctor_get(x_3, 3);
x_18 = lean_ctor_get(x_5, 0);
lean_inc(x_18);
lean_dec(x_5);
x_19 = lean_ctor_get(x_17, 1);
lean_inc(x_19);
if (lean_is_exclusive(x_17)) {
 lean_ctor_release(x_17, 0);
 lean_ctor_release(x_17, 1);
 x_20 = x_17;
} else {
 lean_dec_ref(x_17);
 x_20 = lean_box(0);
}
x_21 = lean_alloc_closure((void*)(lp_mathlib_Subfield_instCompleteLattice___lam__0), 2, 0);
x_22 = lean_box(0);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_18);
lean_ctor_set(x_23, 1, x_21);
if (lean_is_scalar(x_20)) {
 x_24 = lean_alloc_ctor(0, 2, 0);
} else {
 x_24 = x_20;
}
lean_ctor_set(x_24, 0, x_22);
lean_ctor_set(x_24, 1, x_19);
lean_ctor_set(x_3, 3, x_24);
lean_ctor_set(x_3, 0, x_23);
return x_3;
}
}
else
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_25 = lean_ctor_get(x_3, 0);
x_26 = lean_ctor_get(x_3, 3);
x_27 = lean_ctor_get(x_3, 1);
x_28 = lean_ctor_get(x_3, 2);
lean_inc(x_26);
lean_inc(x_28);
lean_inc(x_27);
lean_inc(x_25);
lean_dec(x_3);
x_29 = lean_ctor_get(x_25, 0);
lean_inc_ref(x_29);
if (lean_is_exclusive(x_25)) {
 lean_ctor_release(x_25, 0);
 lean_ctor_release(x_25, 1);
 x_30 = x_25;
} else {
 lean_dec_ref(x_25);
 x_30 = lean_box(0);
}
x_31 = lean_ctor_get(x_26, 1);
lean_inc(x_31);
if (lean_is_exclusive(x_26)) {
 lean_ctor_release(x_26, 0);
 lean_ctor_release(x_26, 1);
 x_32 = x_26;
} else {
 lean_dec_ref(x_26);
 x_32 = lean_box(0);
}
x_33 = lean_alloc_closure((void*)(lp_mathlib_Subfield_instCompleteLattice___lam__0), 2, 0);
x_34 = lean_box(0);
if (lean_is_scalar(x_30)) {
 x_35 = lean_alloc_ctor(0, 2, 0);
} else {
 x_35 = x_30;
}
lean_ctor_set(x_35, 0, x_29);
lean_ctor_set(x_35, 1, x_33);
if (lean_is_scalar(x_32)) {
 x_36 = lean_alloc_ctor(0, 2, 0);
} else {
 x_36 = x_32;
}
lean_ctor_set(x_36, 0, x_34);
lean_ctor_set(x_36, 1, x_31);
x_37 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_37, 0, x_35);
lean_ctor_set(x_37, 1, x_27);
lean_ctor_set(x_37, 2, x_28);
lean_ctor_set(x_37, 3, x_36);
return x_37;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subfield_instCompleteLattice(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_closure(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_closure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subfield_closure(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subfield_gi___lam__0), 2, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subfield_gi(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(x_6, 0, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RingHom_rangeRestrictField(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocusField(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocusField___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_RingHom_eqLocusField(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
static lean_object* _init_lp_mathlib_Subfield_inclusion___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Subfield_subtype___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Subfield_inclusion___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Subfield_inclusion___closed__0;
x_2 = lp_mathlib_RingHom_codRestrict___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_inclusion(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subfield_inclusion___closed__1;
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_inclusion___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subfield_inclusion(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_subfieldCongr___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subfieldCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RingEquiv_subfieldCongr___closed__0;
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subfieldCongr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RingEquiv_subfieldCongr(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Subfield_subtype___lam__0(x_2);
x_5 = lean_apply_2(x_1, x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subfield_toAlgebra___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_2 = lp_mathlib_Field_toSemifield___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_3);
lean_dec_ref(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_5);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 1);
lean_dec(x_9);
x_10 = lp_mathlib_Subfield_inclusion___closed__0;
x_11 = lean_alloc_closure((void*)(lp_mathlib_Subfield_toAlgebra___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_11, 0, x_8);
lean_ctor_set(x_6, 1, x_10);
lean_ctor_set(x_6, 0, x_11);
return x_6;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_ctor_get(x_6, 0);
lean_inc(x_12);
lean_dec(x_6);
x_13 = lp_mathlib_Subfield_inclusion___closed__0;
x_14 = lean_alloc_closure((void*)(lp_mathlib_Subfield_toAlgebra___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_14, 0, x_12);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_13);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subfield_toAlgebra___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subfield_toAlgebra(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Subfield_toAlgebra___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Field_Subfield_Basic_0__Subfield_commClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Field_Subfield_Basic_0__Subfield_commClosure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib___private_Mathlib_Algebra_Field_Subfield_Basic_0__Subfield_commClosure(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Submonoid_instSMulSubtypeMem___redArg(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Submonoid_instSMulSubtypeMem___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subfield_instSMulSubtypeMem(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subfield_instMulActionSubtypeMem(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subfield_instDistribMulActionSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Submonoid_instSMulSubtypeMem___redArg(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Submonoid_instSMulSubtypeMem___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subfield_instSMulWithZeroSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subfield_instModuleSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Subfield_instMulSemiringActionSubtypeMem(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Subfield_instCompleteLattice___closed__0 = _init_lp_mathlib_Subfield_instCompleteLattice___closed__0();
lean_mark_persistent(lp_mathlib_Subfield_instCompleteLattice___closed__0);
lp_mathlib_Subfield_instCompleteLattice___closed__1 = _init_lp_mathlib_Subfield_instCompleteLattice___closed__1();
lean_mark_persistent(lp_mathlib_Subfield_instCompleteLattice___closed__1);
lp_mathlib_Subfield_instCompleteLattice___closed__2 = _init_lp_mathlib_Subfield_instCompleteLattice___closed__2();
lean_mark_persistent(lp_mathlib_Subfield_instCompleteLattice___closed__2);
lp_mathlib_Subfield_inclusion___closed__0 = _init_lp_mathlib_Subfield_inclusion___closed__0();
lean_mark_persistent(lp_mathlib_Subfield_inclusion___closed__0);
lp_mathlib_Subfield_inclusion___closed__1 = _init_lp_mathlib_Subfield_inclusion___closed__1();
lean_mark_persistent(lp_mathlib_Subfield_inclusion___closed__1);
lp_mathlib_RingEquiv_subfieldCongr___closed__0 = _init_lp_mathlib_RingEquiv_subfieldCongr___closed__0();
lean_mark_persistent(lp_mathlib_RingEquiv_subfieldCongr___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
