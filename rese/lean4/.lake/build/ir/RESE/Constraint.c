// Lean compiler output
// Module: RESE.Constraint
// Imports: public import Init public import RESE.Basic
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
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_hard_elim(lean_object*, uint8_t, lean_object*, lean_object*);
static lean_object* lp_rese_RESE_Constraint_DependencyGraph_addNode___closed__0;
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_preference_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_getDeps(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_countHard(lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_countDependencies(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_toCtorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr(uint8_t, lean_object*);
LEAN_EXPORT uint64_t lp_rese_RESE_Constraint_instHashableConstraintType_hash(uint8_t);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_instHashableConstraintType;
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_Constraint_isSoft(lean_object*);
LEAN_EXPORT lean_object* lp_rese_List_foldl___at___00RESE_Constraint_countDependencies_spec__0(lean_object*, lean_object*);
static lean_object* lp_rese_RESE_Constraint_instReprConstraintType___closed__0;
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorElim___redArg(lean_object*);
static lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__7;
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_Constraint_isPref___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_complexityScore(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_List_filterTR_loop___at___00RESE_Constraint_countPref_spec__0(lean_object*, lean_object*);
static lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__2;
lean_object* lean_nat_to_int(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_Constraint_isHard___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_hasCycle___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_countDependencies___boxed(lean_object*);
static lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__3;
static lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__4;
LEAN_EXPORT lean_object* lp_rese_List_filterTR_loop___at___00RESE_Constraint_countSoft_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_List_filterTR_loop___at___00RESE_Constraint_DependencyGraph_getDeps_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_DependencyGraph_hasCycle___lam__0(lean_object*);
static lean_object* lp_rese_RESE_Constraint_instHashableConstraintType___closed__0;
LEAN_EXPORT lean_object* lp_rese_List_foldl___at___00RESE_Constraint_countDependencies_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_hard_elim___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_Constraint_isHard(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorIdx___boxed(lean_object*);
static lean_object* lp_rese_RESE_Constraint_instBEqConstraintType___closed__0;
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_soft_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_instBEqConstraintType_beq___boxed(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_soft_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_countPref(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_soft_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorIdx(uint8_t);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_instReprConstraintType;
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_hard_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_hard_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_DependencyGraph_hasCycle(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_instHashableConstraintType_hash___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_instBEqConstraintType;
static lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__0;
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_countSoft(lean_object*);
static lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__1;
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_Constraint_isPref(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_preference_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_soft_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_rese_List_mapTR_loop___at___00RESE_Constraint_DependencyGraph_getDeps_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_toCtorIdx___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_instBEqConstraintType_beq(uint8_t, uint8_t);
static lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__5;
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t l_List_any___redArg(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_List_filterTR_loop___at___00RESE_Constraint_DependencyGraph_getDeps_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_Constraint_isSoft___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_getDeps___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_addNode(lean_object*, lean_object*);
static lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__6;
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_addEdge(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_preference_elim___redArg(lean_object*);
lean_object* l_instDecidableEqString___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_preference_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_List_filterTR_loop___at___00RESE_Constraint_countHard_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_hasCycle___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorIdx(uint8_t x_1) {
_start:
{
switch (x_1) {
case 0:
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
case 1:
{
lean_object* x_3; 
x_3 = lean_unsigned_to_nat(1u);
return x_3;
}
default: 
{
lean_object* x_4; 
x_4 = lean_unsigned_to_nat(2u);
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorIdx___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_rese_RESE_Constraint_ConstraintType_ctorIdx(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_toCtorIdx(uint8_t x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_rese_RESE_Constraint_ConstraintType_ctorIdx(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_toCtorIdx___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_rese_RESE_Constraint_ConstraintType_toCtorIdx(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorElim(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorElim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lean_unbox(x_3);
x_7 = lp_rese_RESE_Constraint_ConstraintType_ctorElim(x_1, x_2, x_6, x_4, x_5);
lean_dec(x_5);
lean_dec(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_ctorElim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_rese_RESE_Constraint_ConstraintType_ctorElim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_hard_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_hard_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_hard_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_rese_RESE_Constraint_ConstraintType_hard_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_hard_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_rese_RESE_Constraint_ConstraintType_hard_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_soft_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_soft_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_soft_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_rese_RESE_Constraint_ConstraintType_soft_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_soft_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_rese_RESE_Constraint_ConstraintType_soft_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_preference_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_preference_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_preference_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_rese_RESE_Constraint_ConstraintType_preference_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_ConstraintType_preference_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_rese_RESE_Constraint_ConstraintType_preference_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("RESE.Constraint.ConstraintType.hard", 35, 35);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__0;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("RESE.Constraint.ConstraintType.soft", 35, 35);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__2;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("RESE.Constraint.ConstraintType.preference", 41, 41);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__4;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr(uint8_t x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_10; lean_object* x_17; 
switch (x_1) {
case 0:
{
lean_object* x_24; uint8_t x_25; 
x_24 = lean_unsigned_to_nat(1024u);
x_25 = lean_nat_dec_le(x_24, x_2);
if (x_25 == 0)
{
lean_object* x_26; 
x_26 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__6;
x_3 = x_26;
goto block_9;
}
else
{
lean_object* x_27; 
x_27 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__7;
x_3 = x_27;
goto block_9;
}
}
case 1:
{
lean_object* x_28; uint8_t x_29; 
x_28 = lean_unsigned_to_nat(1024u);
x_29 = lean_nat_dec_le(x_28, x_2);
if (x_29 == 0)
{
lean_object* x_30; 
x_30 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__6;
x_10 = x_30;
goto block_16;
}
else
{
lean_object* x_31; 
x_31 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__7;
x_10 = x_31;
goto block_16;
}
}
default: 
{
lean_object* x_32; uint8_t x_33; 
x_32 = lean_unsigned_to_nat(1024u);
x_33 = lean_nat_dec_le(x_32, x_2);
if (x_33 == 0)
{
lean_object* x_34; 
x_34 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__6;
x_17 = x_34;
goto block_23;
}
else
{
lean_object* x_35; 
x_35 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__7;
x_17 = x_35;
goto block_23;
}
}
}
block_9:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__1;
x_5 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
x_6 = 0;
x_7 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set_uint8(x_7, sizeof(void*)*1, x_6);
x_8 = l_Repr_addAppParen(x_7, x_2);
return x_8;
}
block_16:
{
lean_object* x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__3;
x_12 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set(x_12, 1, x_11);
x_13 = 0;
x_14 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set_uint8(x_14, sizeof(void*)*1, x_13);
x_15 = l_Repr_addAppParen(x_14, x_2);
return x_15;
}
block_23:
{
lean_object* x_18; lean_object* x_19; uint8_t x_20; lean_object* x_21; lean_object* x_22; 
x_18 = lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__5;
x_19 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
x_20 = 0;
x_21 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set_uint8(x_21, sizeof(void*)*1, x_20);
x_22 = l_Repr_addAppParen(x_21, x_2);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_instReprConstraintType_repr___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lean_unbox(x_1);
x_4 = lp_rese_RESE_Constraint_instReprConstraintType_repr(x_3, x_2);
lean_dec(x_2);
return x_4;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instReprConstraintType___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_rese_RESE_Constraint_instReprConstraintType_repr___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instReprConstraintType() {
_start:
{
lean_object* x_1; 
x_1 = lp_rese_RESE_Constraint_instReprConstraintType___closed__0;
return x_1;
}
}
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_instBEqConstraintType_beq(uint8_t x_1, uint8_t x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lp_rese_RESE_Constraint_ConstraintType_ctorIdx(x_1);
x_4 = lp_rese_RESE_Constraint_ConstraintType_ctorIdx(x_2);
x_5 = lean_nat_dec_eq(x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_instBEqConstraintType_beq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_rese_RESE_Constraint_instBEqConstraintType_beq(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instBEqConstraintType___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_rese_RESE_Constraint_instBEqConstraintType_beq___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instBEqConstraintType() {
_start:
{
lean_object* x_1; 
x_1 = lp_rese_RESE_Constraint_instBEqConstraintType___closed__0;
return x_1;
}
}
LEAN_EXPORT uint64_t lp_rese_RESE_Constraint_instHashableConstraintType_hash(uint8_t x_1) {
_start:
{
switch (x_1) {
case 0:
{
uint64_t x_2; 
x_2 = 0;
return x_2;
}
case 1:
{
uint64_t x_3; 
x_3 = 1;
return x_3;
}
default: 
{
uint64_t x_4; 
x_4 = 2;
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_instHashableConstraintType_hash___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint64_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_rese_RESE_Constraint_instHashableConstraintType_hash(x_2);
x_4 = lean_box_uint64(x_3);
return x_4;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instHashableConstraintType___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_rese_RESE_Constraint_instHashableConstraintType_hash___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_instHashableConstraintType() {
_start:
{
lean_object* x_1; 
x_1 = lp_rese_RESE_Constraint_instHashableConstraintType___closed__0;
return x_1;
}
}
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_Constraint_isHard(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; uint8_t x_4; 
x_2 = lean_ctor_get_uint8(x_1, sizeof(void*)*4);
x_3 = 0;
x_4 = lp_rese_RESE_Constraint_instBEqConstraintType_beq(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_Constraint_isHard___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_rese_RESE_Constraint_Constraint_isHard(x_1);
lean_dec_ref(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_Constraint_isSoft(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; uint8_t x_4; 
x_2 = lean_ctor_get_uint8(x_1, sizeof(void*)*4);
x_3 = 1;
x_4 = lp_rese_RESE_Constraint_instBEqConstraintType_beq(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_Constraint_isSoft___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_rese_RESE_Constraint_Constraint_isSoft(x_1);
lean_dec_ref(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_Constraint_isPref(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; uint8_t x_4; 
x_2 = lean_ctor_get_uint8(x_1, sizeof(void*)*4);
x_3 = 2;
x_4 = lp_rese_RESE_Constraint_instBEqConstraintType_beq(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_Constraint_isPref___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_rese_RESE_Constraint_Constraint_isPref(x_1);
lean_dec_ref(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
static lean_object* _init_lp_rese_RESE_Constraint_DependencyGraph_addNode___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_alloc_closure((void*)(l_instDecidableEqString___boxed), 2, 0);
x_2 = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_addNode(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lp_rese_RESE_Constraint_DependencyGraph_addNode___closed__0;
lean_inc(x_3);
lean_inc_ref(x_2);
x_6 = l_List_elem___redArg(x_5, x_2, x_3);
if (x_6 == 0)
{
uint8_t x_7; 
lean_inc(x_4);
lean_inc(x_3);
x_7 = !lean_is_exclusive(x_1);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_1, 1);
lean_dec(x_8);
x_9 = lean_ctor_get(x_1, 0);
lean_dec(x_9);
x_10 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_10, 0, x_2);
lean_ctor_set(x_10, 1, x_3);
lean_ctor_set(x_1, 0, x_10);
return x_1;
}
else
{
lean_object* x_11; lean_object* x_12; 
lean_dec(x_1);
x_11 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_11, 0, x_2);
lean_ctor_set(x_11, 1, x_3);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_4);
return x_12;
}
}
else
{
lean_dec_ref(x_2);
return x_1;
}
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_addEdge(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_2);
lean_ctor_set(x_6, 1, x_3);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_5);
lean_ctor_set(x_1, 1, x_7);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_1);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_2);
lean_ctor_set(x_10, 1, x_3);
x_11 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_9);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_8);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_rese_List_filterTR_loop___at___00RESE_Constraint_DependencyGraph_getDeps_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_string_dec_eq(x_8, x_1);
if (x_9 == 0)
{
lean_free_object(x_2);
lean_dec(x_6);
x_2 = x_7;
goto _start;
}
else
{
lean_ctor_set(x_2, 1, x_3);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_12 = lean_ctor_get(x_2, 0);
x_13 = lean_ctor_get(x_2, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_2);
x_14 = lean_ctor_get(x_12, 0);
x_15 = lean_string_dec_eq(x_14, x_1);
if (x_15 == 0)
{
lean_dec(x_12);
x_2 = x_13;
goto _start;
}
else
{
lean_object* x_17; 
x_17 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_17, 0, x_12);
lean_ctor_set(x_17, 1, x_3);
x_2 = x_13;
x_3 = x_17;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_rese_List_mapTR_loop___at___00RESE_Constraint_DependencyGraph_getDeps_spec__1(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = lean_ctor_get(x_9, 1);
lean_inc(x_11);
lean_dec(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_getDeps(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_box(0);
x_5 = lp_rese_List_filterTR_loop___at___00RESE_Constraint_DependencyGraph_getDeps_spec__0(x_2, x_3, x_4);
x_6 = lp_rese_List_mapTR_loop___at___00RESE_Constraint_DependencyGraph_getDeps_spec__1(x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_getDeps___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_rese_RESE_Constraint_DependencyGraph_getDeps(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_rese_List_filterTR_loop___at___00RESE_Constraint_DependencyGraph_getDeps_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_rese_List_filterTR_loop___at___00RESE_Constraint_DependencyGraph_getDeps_spec__0(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_DependencyGraph_hasCycle___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_string_dec_eq(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_hasCycle___lam__0___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_rese_RESE_Constraint_DependencyGraph_hasCycle___lam__0(x_1);
lean_dec_ref(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_rese_RESE_Constraint_DependencyGraph_hasCycle(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = l_List_lengthTR___redArg(x_2);
lean_dec(x_2);
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_nat_dec_eq(x_4, x_5);
lean_dec(x_4);
if (x_6 == 0)
{
lean_object* x_7; uint8_t x_8; 
x_7 = lean_alloc_closure((void*)(lp_rese_RESE_Constraint_DependencyGraph_hasCycle___lam__0___boxed), 1, 0);
x_8 = l_List_any___redArg(x_3, x_7);
return x_8;
}
else
{
uint8_t x_9; 
lean_dec(x_3);
x_9 = 0;
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_DependencyGraph_hasCycle___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_rese_RESE_Constraint_DependencyGraph_hasCycle(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_rese_List_filterTR_loop___at___00RESE_Constraint_countHard_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lp_rese_RESE_Constraint_Constraint_isHard(x_5);
if (x_7 == 0)
{
lean_free_object(x_1);
lean_dec(x_5);
x_1 = x_6;
goto _start;
}
else
{
lean_ctor_set(x_1, 1, x_2);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
}
else
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_10 = lean_ctor_get(x_1, 0);
x_11 = lean_ctor_get(x_1, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_1);
x_12 = lp_rese_RESE_Constraint_Constraint_isHard(x_10);
if (x_12 == 0)
{
lean_dec(x_10);
x_1 = x_11;
goto _start;
}
else
{
lean_object* x_14; 
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_10);
lean_ctor_set(x_14, 1, x_2);
x_1 = x_11;
x_2 = x_14;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_countHard(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_box(0);
x_3 = lp_rese_List_filterTR_loop___at___00RESE_Constraint_countHard_spec__0(x_1, x_2);
x_4 = l_List_lengthTR___redArg(x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_rese_List_foldl___at___00RESE_Constraint_countDependencies_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = lean_ctor_get(x_3, 2);
x_6 = l_List_lengthTR___redArg(x_5);
x_7 = lean_nat_add(x_1, x_6);
lean_dec(x_6);
lean_dec(x_1);
x_1 = x_7;
x_2 = x_4;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_countDependencies(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_rese_List_foldl___at___00RESE_Constraint_countDependencies_spec__0(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_countDependencies___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_rese_RESE_Constraint_countDependencies(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_rese_List_foldl___at___00RESE_Constraint_countDependencies_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_rese_List_foldl___at___00RESE_Constraint_countDependencies_spec__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_rese_List_filterTR_loop___at___00RESE_Constraint_countSoft_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lp_rese_RESE_Constraint_Constraint_isSoft(x_5);
if (x_7 == 0)
{
lean_free_object(x_1);
lean_dec(x_5);
x_1 = x_6;
goto _start;
}
else
{
lean_ctor_set(x_1, 1, x_2);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
}
else
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_10 = lean_ctor_get(x_1, 0);
x_11 = lean_ctor_get(x_1, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_1);
x_12 = lp_rese_RESE_Constraint_Constraint_isSoft(x_10);
if (x_12 == 0)
{
lean_dec(x_10);
x_1 = x_11;
goto _start;
}
else
{
lean_object* x_14; 
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_10);
lean_ctor_set(x_14, 1, x_2);
x_1 = x_11;
x_2 = x_14;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_countSoft(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_box(0);
x_3 = lp_rese_List_filterTR_loop___at___00RESE_Constraint_countSoft_spec__0(x_1, x_2);
x_4 = l_List_lengthTR___redArg(x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_rese_List_filterTR_loop___at___00RESE_Constraint_countPref_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lp_rese_RESE_Constraint_Constraint_isPref(x_5);
if (x_7 == 0)
{
lean_free_object(x_1);
lean_dec(x_5);
x_1 = x_6;
goto _start;
}
else
{
lean_ctor_set(x_1, 1, x_2);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
}
else
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_10 = lean_ctor_get(x_1, 0);
x_11 = lean_ctor_get(x_1, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_1);
x_12 = lp_rese_RESE_Constraint_Constraint_isPref(x_10);
if (x_12 == 0)
{
lean_dec(x_10);
x_1 = x_11;
goto _start;
}
else
{
lean_object* x_14; 
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_10);
lean_ctor_set(x_14, 1, x_2);
x_1 = x_11;
x_2 = x_14;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_countPref(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_box(0);
x_3 = lp_rese_List_filterTR_loop___at___00RESE_Constraint_countPref_spec__0(x_1, x_2);
x_4 = l_List_lengthTR___redArg(x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_rese_RESE_Constraint_complexityScore(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc(x_1);
x_2 = lp_rese_RESE_Constraint_countHard(x_1);
x_3 = lean_unsigned_to_nat(3u);
x_4 = lean_nat_mul(x_2, x_3);
lean_dec(x_2);
lean_inc(x_1);
x_5 = lp_rese_RESE_Constraint_countSoft(x_1);
x_6 = lean_unsigned_to_nat(2u);
x_7 = lean_nat_mul(x_5, x_6);
lean_dec(x_5);
x_8 = lean_nat_add(x_4, x_7);
lean_dec(x_7);
lean_dec(x_4);
x_9 = lp_rese_RESE_Constraint_countPref(x_1);
x_10 = lean_nat_add(x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_rese_RESE_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_rese_RESE_Constraint(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_rese_RESE_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__0 = _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__0();
lean_mark_persistent(lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__0);
lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__1 = _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__1();
lean_mark_persistent(lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__1);
lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__2 = _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__2();
lean_mark_persistent(lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__2);
lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__3 = _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__3();
lean_mark_persistent(lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__3);
lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__4 = _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__4();
lean_mark_persistent(lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__4);
lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__5 = _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__5();
lean_mark_persistent(lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__5);
lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__6 = _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__6();
lean_mark_persistent(lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__6);
lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__7 = _init_lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__7();
lean_mark_persistent(lp_rese_RESE_Constraint_instReprConstraintType_repr___closed__7);
lp_rese_RESE_Constraint_instReprConstraintType___closed__0 = _init_lp_rese_RESE_Constraint_instReprConstraintType___closed__0();
lean_mark_persistent(lp_rese_RESE_Constraint_instReprConstraintType___closed__0);
lp_rese_RESE_Constraint_instReprConstraintType = _init_lp_rese_RESE_Constraint_instReprConstraintType();
lean_mark_persistent(lp_rese_RESE_Constraint_instReprConstraintType);
lp_rese_RESE_Constraint_instBEqConstraintType___closed__0 = _init_lp_rese_RESE_Constraint_instBEqConstraintType___closed__0();
lean_mark_persistent(lp_rese_RESE_Constraint_instBEqConstraintType___closed__0);
lp_rese_RESE_Constraint_instBEqConstraintType = _init_lp_rese_RESE_Constraint_instBEqConstraintType();
lean_mark_persistent(lp_rese_RESE_Constraint_instBEqConstraintType);
lp_rese_RESE_Constraint_instHashableConstraintType___closed__0 = _init_lp_rese_RESE_Constraint_instHashableConstraintType___closed__0();
lean_mark_persistent(lp_rese_RESE_Constraint_instHashableConstraintType___closed__0);
lp_rese_RESE_Constraint_instHashableConstraintType = _init_lp_rese_RESE_Constraint_instHashableConstraintType();
lean_mark_persistent(lp_rese_RESE_Constraint_instHashableConstraintType);
lp_rese_RESE_Constraint_DependencyGraph_addNode___closed__0 = _init_lp_rese_RESE_Constraint_DependencyGraph_addNode___closed__0();
lean_mark_persistent(lp_rese_RESE_Constraint_DependencyGraph_addNode___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
