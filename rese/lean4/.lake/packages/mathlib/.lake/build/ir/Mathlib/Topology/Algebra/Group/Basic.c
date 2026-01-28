// Lean compiler output
// Module: Mathlib.Topology.Algebra.Group.Basic
// Imports: public import Init public import Mathlib.Algebra.Group.Subgroup.Pointwise public import Mathlib.Algebra.Group.Submonoid.Units public import Mathlib.Algebra.Group.Submonoid.MulOpposite public import Mathlib.Algebra.Order.Archimedean.Basic public import Mathlib.Order.Filter.Bases.Finite public import Mathlib.Topology.Algebra.Group.Defs public import Mathlib.Topology.Algebra.Monoid public import Mathlib.Topology.Homeomorph.Lemmas
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
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodUnits(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_toAddUnits___redArg(lean_object*);
lean_object* lp_mathlib_SubgroupClass_toGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nhdsMulHom___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_divLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subRight___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_addLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addRight___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toUnits__homeomorph___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits__homeomorph___redArg(lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addCommGroupTopologicalClosure___redArg(lean_object*);
lean_object* lp_mathlib_toUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nhdsAddHom___redArg(lean_object*);
static lean_object* lp_mathlib_Homeomorph_shearMulRight___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_divRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodUnits___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_addRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_divRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_connectedComponentOfZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addRight___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_nhds___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toUnits__homeomorph(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nhdsAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_shearAddRight___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_mulRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_shearMulRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_divLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_neg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodAddUnits(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_connectedComponentOfZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nhdsAddHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_subRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toUnits__homeomorph___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_inv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodUnits___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_connectedComponentOfOne(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodShear___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AddEquiv_prodAddUnits(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits__homeomorph(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toUnits__homeomorph___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_subLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits__homeomorph___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topologicalClosure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodAddUnits___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_shearAddRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_subRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_connectedComponentOfOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_commGroupTopologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addLeft___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_mulLeft___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_neg___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_mulLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_commGroupTopologicalClosure___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_addLeft___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodAddUnits___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_subLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits__homeomorph___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nhdsMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulEquiv_prodUnits(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topologicalClosure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_shearMulRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nhdsMulHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodAddUnits___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addCommGroupTopologicalClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodUnits___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_divRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_divLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_mulLeft___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_mulLeft___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Homeomorph_mulLeft(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulLeft___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Homeomorph_mulLeft___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addLeft(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_addLeft___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addLeft___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_addLeft___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addLeft___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Homeomorph_addLeft(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addLeft___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Homeomorph_addLeft___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_mulRight___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_mulRight___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Homeomorph_mulRight(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_mulRight___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Homeomorph_mulRight___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_addRight___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addRight___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_addRight___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addRight___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Homeomorph_addRight(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_addRight___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Homeomorph_addRight___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_inv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
lean_inc(x_3);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_inv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
lean_inc(x_1);
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_neg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
lean_inc(x_3);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_neg___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
lean_inc(x_1);
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_shearMulRight___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_shearMulRight___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Homeomorph_shearMulRight___redArg___closed__0;
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulLeft___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lp_mathlib_Equiv_prodShear___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_shearMulRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Homeomorph_shearMulRight___redArg(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_shearAddRight___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Homeomorph_shearMulRight___redArg___closed__0;
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_addLeft___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lp_mathlib_Equiv_prodShear___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_shearAddRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Homeomorph_shearAddRight___redArg(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topologicalClosure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subgroup_topologicalClosure(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topologicalClosure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddSubgroup_topologicalClosure(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_connectedComponentOfOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_connectedComponentOfOne___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Subgroup_connectedComponentOfOne(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_connectedComponentOfZero(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_connectedComponentOfZero___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddSubgroup_connectedComponentOfZero(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_commGroupTopologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_SubgroupClass_toGroup___redArg(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_commGroupTopologicalClosure___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SubgroupClass_toGroup___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addCommGroupTopologicalClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addCommGroupTopologicalClosure___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_divLeft(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_divLeft___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_divLeft___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_divLeft___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_subLeft(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_subLeft___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_subLeft___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_subLeft___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_divRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_divRight___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_divRight___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_divRight___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_subRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_subRight___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_subRight___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_subRight___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nhdsMulHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_nhds___boxed), 3, 2);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nhdsMulHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_nhds___boxed), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nhdsMulHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_nhdsMulHom(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nhdsAddHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_nhds___boxed), 3, 2);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nhdsAddHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_nhds___boxed), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nhdsAddHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_nhdsAddHom(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toUnits__homeomorph(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_toUnits___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toUnits__homeomorph___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_toUnits___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toUnits__homeomorph___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_toUnits__homeomorph(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toUnits__homeomorph___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_toUnits__homeomorph___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits__homeomorph(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_toAddUnits___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits__homeomorph___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_toAddUnits___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits__homeomorph___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_toAddUnits__homeomorph(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits__homeomorph___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_toAddUnits__homeomorph___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodUnits(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_MulEquiv_prodUnits(lean_box(0), lean_box(0), x_3, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodUnits___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulEquiv_prodUnits(lean_box(0), lean_box(0), x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodUnits___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Homeomorph_prodUnits(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodUnits___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Homeomorph_prodUnits___redArg(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodAddUnits(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_AddEquiv_prodAddUnits(lean_box(0), lean_box(0), x_3, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodAddUnits___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddEquiv_prodAddUnits(lean_box(0), lean_box(0), x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodAddUnits___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Homeomorph_prodAddUnits(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodAddUnits___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Homeomorph_prodAddUnits___redArg(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Archimedean_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Bases_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Monoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Homeomorph_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Group_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Archimedean_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Bases_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Homeomorph_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Homeomorph_shearMulRight___redArg___closed__0 = _init_lp_mathlib_Homeomorph_shearMulRight___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_shearMulRight___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
