// Lean compiler output
// Module: Mathlib.GroupTheory.Perm.Cycle.Type
// Imports: public import Init public import Mathlib.Algebra.GCDMonoid.Multiset public import Mathlib.Algebra.GCDMonoid.Nat public import Mathlib.Algebra.Group.TypeTags.Finite public import Mathlib.Combinatorics.Enumerative.Partition.Basic public import Mathlib.Data.List.Rotate public import Mathlib.GroupTheory.Perm.Closure public import Mathlib.GroupTheory.Perm.Cycle.Factors public import Mathlib.Tactic.NormNum.GCD
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
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_Perm_support(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleType(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_partition___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Vector_fintype___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_rotate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_instFintypeElemVectorVectorsProdEqOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector___redArg(lean_object*, lean_object*);
lean_object* lp_batteries_List_prod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique___redArg(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_Perm_cycleType___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_instFintypeElemVectorVectorsProdEqOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_zeroUnique(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique(lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_partition(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_zeroUnique___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_rotate___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___lam__0(lean_object*);
lean_object* lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_instFintypeElemVectorVectorsProdEqOne___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_instFintypeElemVectorVectorsProdEqOne___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique___redArg___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_Perm_support___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_ofUnique___redArg(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector___redArg___boxed(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_rotate___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_rotate___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleType___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_card___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___boxed(lean_object*);
lean_object* l_List_splitAt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Equiv_Perm_cycleType___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Finset_card___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleType___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_Equiv_Perm_cycleType___redArg___closed__0;
lean_inc(x_1);
lean_inc_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_support), 4, 3);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, x_2);
lean_closure_set(x_5, 2, x_1);
x_6 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, lean_box(0));
lean_closure_set(x_6, 3, x_4);
lean_closure_set(x_6, 4, x_5);
lean_inc_ref(x_3);
lean_inc(x_1);
x_7 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(x_2, x_1, x_3, x_1, x_3);
x_8 = lp_mathlib_Multiset_map___redArg(x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleType(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_Perm_cycleType___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_zeroUnique(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_zeroUnique___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_zeroUnique(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_1);
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_2, 1);
lean_dec(x_4);
x_5 = lean_box(0);
lean_ctor_set_tag(x_2, 1);
lean_ctor_set(x_2, 1, x_5);
return x_2;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
lean_dec(x_2);
x_7 = lean_box(0);
x_8 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___lam__0(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
return x_1;
}
else
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
x_5 = lp_batteries_List_prod___redArg(x_1, x_2, x_4);
x_6 = lean_apply_1(x_3, x_5);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_2 = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_ctor_get(x_1, 0);
x_6 = lp_mathlib_Monoid_toMulOneClass___redArg(x_5);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_ctor_get(x_6, 1);
x_9 = lean_ctor_get(x_6, 0);
lean_dec(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___lam__0___boxed), 1, 0);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___lam__1), 4, 3);
lean_closure_set(x_11, 0, x_8);
lean_closure_set(x_11, 1, x_3);
lean_closure_set(x_11, 2, x_4);
lean_ctor_set(x_6, 1, x_10);
lean_ctor_set(x_6, 0, x_11);
return x_6;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_ctor_get(x_6, 1);
lean_inc(x_12);
lean_dec(x_6);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___lam__0___boxed), 1, 0);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___lam__1), 4, 3);
lean_closure_set(x_14, 0, x_12);
lean_closure_set(x_14, 1, x_3);
lean_closure_set(x_14, 2, x_4);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_13);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_nat_dec_eq(x_2, x_3);
if (x_4 == 1)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_box(0);
x_6 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_oneUnique___redArg(x_1);
x_7 = lp_mathlib_Equiv_ofUnique___redArg(x_5, x_6);
x_8 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg(x_1);
x_9 = lp_mathlib_Equiv_symm___redArg(x_8);
x_10 = lp_mathlib_Equiv_trans___redArg(x_7, x_9);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; 
x_11 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_vectorEquiv___redArg(x_1);
x_12 = lp_mathlib_Equiv_symm___redArg(x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_instFintypeElemVectorVectorsProdEqOne___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_unsigned_to_nat(1u);
x_5 = lean_nat_sub(x_2, x_4);
x_6 = lp_mathlib_Vector_fintype___redArg(x_3, x_5);
x_7 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_equivVector___redArg(x_1, x_2);
x_8 = lp_mathlib_Equiv_symm___redArg(x_7);
x_9 = lp_mathlib_Fintype_ofEquiv___redArg(x_6, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_instFintypeElemVectorVectorsProdEqOne(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_instFintypeElemVectorVectorsProdEqOne___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_instFintypeElemVectorVectorsProdEqOne___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_instFintypeElemVectorVectorsProdEqOne(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_instFintypeElemVectorVectorsProdEqOne___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_instFintypeElemVectorVectorsProdEqOne___redArg(x_1, x_2, x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_rotate___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = l_List_lengthTR___redArg(x_1);
x_4 = lean_nat_mod(x_2, x_3);
lean_dec(x_3);
x_5 = l_List_splitAt___redArg(x_4, x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec_ref(x_5);
x_8 = l_List_appendTR___redArg(x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_rotate(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_rotate___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_rotate___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_rotate(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_VectorsProdEqOne_rotate___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_Perm_VectorsProdEqOne_rotate___redArg(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_partition___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_4 = l_List_lengthTR___redArg(x_1);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc(x_1);
x_5 = lp_mathlib_Equiv_Perm_cycleType___redArg(x_1, x_2, x_3);
x_6 = lp_mathlib_Equiv_Perm_support___redArg(x_2, x_1, x_3);
x_7 = l_List_lengthTR___redArg(x_6);
lean_dec(x_6);
x_8 = lean_nat_sub(x_4, x_7);
lean_dec(x_7);
lean_dec(x_4);
x_9 = lean_unsigned_to_nat(1u);
x_10 = l_List_replicateTR___redArg(x_8, x_9);
x_11 = l_List_appendTR___redArg(x_5, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_partition(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_Perm_partition___redArg(x_2, x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Combinatorics_Enumerative_Partition_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Rotate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Perm_Closure(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Perm_Cycle_Factors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_GCD(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Perm_Cycle_Type(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Combinatorics_Enumerative_Partition_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Rotate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Perm_Closure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Perm_Cycle_Factors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_GCD(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Equiv_Perm_cycleType___redArg___closed__0 = _init_lp_mathlib_Equiv_Perm_cycleType___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_Perm_cycleType___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
