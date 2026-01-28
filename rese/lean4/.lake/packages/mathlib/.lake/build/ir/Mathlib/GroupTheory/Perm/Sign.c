// Lean compiler output
// Module: Mathlib.GroupTheory.Perm.Sign
// Imports: public import Init public import Mathlib.Algebra.Group.Conj public import Mathlib.Algebra.Group.Subgroup.Lattice public import Mathlib.Algebra.Group.Submonoid.BigOperators public import Mathlib.Data.Finset.Fin public import Mathlib.Data.Finset.Sort public import Mathlib.Data.Fintype.Perm public import Mathlib.Data.Fintype.Prod public import Mathlib.Data.Fintype.Sum public import Mathlib.Data.Int.Order.Units public import Mathlib.GroupTheory.Perm.Support public import Mathlib.Logic.Equiv.Fintype public import Mathlib.Tactic.NormNum.Ineq public import Mathlib.Data.Finset.Sigma
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sign___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_sort___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_Units_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_swapFactorsAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signBijAux___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_modSwap(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sign(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSign___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_modSwap___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_Perm_Sign_0__Equiv_Perm_signAux2_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sigma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_instFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_Perm_signAux___lam__0___closed__0;
static lean_object* lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_swapFactors___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_Perm_signAux___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___at___00Equiv_Perm_signAux_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_truncSwapFactors(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_Perm_Sign_0__Equiv_Perm_signAux2_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* l_Int_instDecidableEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_swapFactors(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_finPairsLT___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_swapFactorsAux___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___at___00Equiv_Perm_signAux_spec__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_swap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signBijAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signBijAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSign___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_finPairsLT(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_truncSwapFactors___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSign(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Perm_ofSign___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_attachFin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sign___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
lean_object* lean_int_neg(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_modSwap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_modSwap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_Perm_modSwap(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_swapFactorsAux___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_4 = lean_box(0);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
lean_inc(x_6);
x_9 = lean_apply_1(x_8, x_6);
lean_inc_ref(x_1);
lean_inc(x_9);
lean_inc(x_6);
x_10 = lean_apply_2(x_1, x_6, x_9);
x_11 = lean_unbox(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_inc_ref(x_1);
x_12 = lp_mathlib_Equiv_swap___redArg(x_1, x_6, x_9);
lean_inc_ref(x_12);
x_13 = lp_mathlib_Equiv_trans___redArg(x_3, x_12);
x_14 = lp_mathlib_Equiv_Perm_swapFactorsAux___redArg(x_1, x_7, x_13);
lean_ctor_set(x_2, 1, x_14);
lean_ctor_set(x_2, 0, x_12);
return x_2;
}
else
{
lean_dec(x_9);
lean_free_object(x_2);
lean_dec(x_6);
x_2 = x_7;
goto _start;
}
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_16 = lean_ctor_get(x_2, 0);
x_17 = lean_ctor_get(x_2, 1);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_2);
x_18 = lean_ctor_get(x_3, 0);
lean_inc(x_18);
lean_inc(x_16);
x_19 = lean_apply_1(x_18, x_16);
lean_inc_ref(x_1);
lean_inc(x_19);
lean_inc(x_16);
x_20 = lean_apply_2(x_1, x_16, x_19);
x_21 = lean_unbox(x_20);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
lean_inc_ref(x_1);
x_22 = lp_mathlib_Equiv_swap___redArg(x_1, x_16, x_19);
lean_inc_ref(x_22);
x_23 = lp_mathlib_Equiv_trans___redArg(x_3, x_22);
x_24 = lp_mathlib_Equiv_Perm_swapFactorsAux___redArg(x_1, x_17, x_23);
x_25 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_25, 0, x_22);
lean_ctor_set(x_25, 1, x_24);
return x_25;
}
else
{
lean_dec(x_19);
lean_dec(x_16);
x_2 = x_17;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_swapFactorsAux(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_Perm_swapFactorsAux___redArg(x_2, x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_swapFactors___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_3, 4);
lean_inc_ref(x_5);
lean_dec_ref(x_3);
x_6 = lp_mathlib_Multiset_sort___redArg(x_2, x_5);
x_7 = lp_mathlib_Equiv_Perm_swapFactorsAux___redArg(x_1, x_6, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_swapFactors(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_Perm_swapFactors___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_truncSwapFactors(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_Perm_swapFactorsAux___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_truncSwapFactors___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_Perm_swapFactorsAux___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_finPairsLT___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = l_List_range(x_1);
x_3 = lp_mathlib_Finset_attachFin___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_finPairsLT(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_finPairsLT___lam__0), 1, 0);
x_3 = l_List_finRange(x_1);
x_4 = lp_mathlib_Finset_sigma___redArg(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_int_mul(x_4, x_6);
lean_dec(x_6);
x_9 = lean_int_mul(x_7, x_5);
lean_dec(x_7);
lean_ctor_set(x_2, 1, x_9);
lean_ctor_set(x_2, 0, x_8);
return x_2;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_10 = lean_ctor_get(x_1, 0);
x_11 = lean_ctor_get(x_1, 1);
x_12 = lean_ctor_get(x_2, 0);
x_13 = lean_ctor_get(x_2, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_2);
x_14 = lean_int_mul(x_10, x_12);
lean_dec(x_12);
x_15 = lean_int_mul(x_13, x_11);
lean_dec(x_13);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_14);
lean_ctor_set(x_16, 1, x_15);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___lam__0(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__0;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___lam__0___boxed), 2, 0);
x_3 = lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__1;
x_4 = l_List_foldrTR___redArg(x_2, x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___at___00Equiv_Perm_signAux_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Multiset_map___redArg(x_2, x_1);
x_4 = lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___at___00Equiv_Perm_signAux_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_prod___at___00Equiv_Perm_signAux_spec__0___redArg(x_2, x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Equiv_Perm_signAux___lam__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__0;
x_2 = lean_int_neg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Equiv_Perm_signAux___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Equiv_Perm_signAux___lam__0___closed__0;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, x_3);
x_7 = lean_apply_1(x_5, x_4);
x_8 = lean_nat_dec_le(x_6, x_7);
lean_dec(x_7);
lean_dec(x_6);
if (x_8 == 0)
{
lean_object* x_9; 
x_9 = lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__1;
return x_9;
}
else
{
lean_object* x_10; 
x_10 = lp_mathlib_Equiv_Perm_signAux___lam__0___closed__1;
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_signAux___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_Equiv_Perm_finPairsLT(x_1);
x_5 = lp_mathlib_Finset_prod___at___00Equiv_Perm_signAux_spec__0___redArg(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signBijAux___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
lean_inc(x_6);
x_7 = lean_apply_1(x_6, x_5);
x_8 = lean_apply_1(x_6, x_4);
x_9 = lean_nat_dec_lt(x_7, x_8);
if (x_9 == 0)
{
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_7);
return x_2;
}
else
{
lean_ctor_set(x_2, 1, x_7);
lean_ctor_set(x_2, 0, x_8);
return x_2;
}
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_2);
x_12 = lean_ctor_get(x_1, 0);
lean_inc(x_12);
lean_dec_ref(x_1);
lean_inc(x_12);
x_13 = lean_apply_1(x_12, x_11);
x_14 = lean_apply_1(x_12, x_10);
x_15 = lean_nat_dec_lt(x_13, x_14);
if (x_15 == 0)
{
lean_object* x_16; 
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_13);
lean_ctor_set(x_16, 1, x_14);
return x_16;
}
else
{
lean_object* x_17; 
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_14);
lean_ctor_set(x_17, 1, x_13);
return x_17;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signBijAux(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_Perm_signBijAux___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signBijAux___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_Perm_signBijAux(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux2___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_4 = lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__1;
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_5 = lean_ctor_get(x_2, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 1);
lean_inc(x_6);
lean_dec_ref(x_2);
x_7 = lean_ctor_get(x_3, 0);
lean_inc(x_7);
lean_inc(x_5);
x_8 = lean_apply_1(x_7, x_5);
lean_inc_ref(x_1);
lean_inc(x_8);
lean_inc(x_5);
x_9 = lean_apply_2(x_1, x_5, x_8);
x_10 = lean_unbox(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
lean_inc_ref(x_1);
x_11 = lp_mathlib_Equiv_swap___redArg(x_1, x_5, x_8);
x_12 = lp_mathlib_Equiv_trans___redArg(x_3, x_11);
x_13 = lp_mathlib_Equiv_Perm_signAux2___redArg(x_1, x_6, x_12);
x_14 = !lean_is_exclusive(x_13);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_15 = lean_ctor_get(x_13, 0);
x_16 = lean_ctor_get(x_13, 1);
x_17 = lean_int_neg(x_15);
lean_dec(x_15);
x_18 = lean_int_neg(x_16);
lean_dec(x_16);
lean_ctor_set(x_13, 1, x_18);
lean_ctor_set(x_13, 0, x_17);
return x_13;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_19 = lean_ctor_get(x_13, 0);
x_20 = lean_ctor_get(x_13, 1);
lean_inc(x_20);
lean_inc(x_19);
lean_dec(x_13);
x_21 = lean_int_neg(x_19);
lean_dec(x_19);
x_22 = lean_int_neg(x_20);
lean_dec(x_20);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_22);
return x_23;
}
}
else
{
lean_dec(x_8);
lean_dec(x_5);
x_2 = x_6;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_Perm_signAux2___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_Perm_Sign_0__Equiv_Perm_signAux2_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_5; 
lean_dec(x_4);
x_5 = lean_apply_1(x_3, x_2);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_dec(x_3);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_apply_3(x_4, x_6, x_7, x_2);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_Perm_Sign_0__Equiv_Perm_signAux2_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_GroupTheory_Perm_Sign_0__Equiv_Perm_signAux2_match__1_splitter___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Equiv_Perm_signAux2___redArg(x_2, x_5, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_signAux3___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_Perm_signAux2___redArg(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sign___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_Perm_signAux2___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sign___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_sign___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sign(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_Perm_sign___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Perm_ofSign___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_alloc_closure((void*)(l_Int_instDecidableEq___boxed), 2, 0);
x_6 = lp_mathlib_Equiv_Perm_signAux2___redArg(x_1, x_2, x_4);
x_7 = lp_mathlib_Units_instDecidableEq___redArg(x_5, x_6, x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSign___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Equiv_Perm_ofSign___redArg___lam__0(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSign___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_2);
lean_inc_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_ofSign___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
lean_closure_set(x_4, 2, x_3);
lean_inc(x_2);
lean_inc_ref(x_1);
x_5 = lp_mathlib_Equiv_instFintype___redArg(x_1, x_1, x_2, x_2);
x_6 = lp_mathlib_Multiset_filter___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSign(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_Perm_ofSign___redArg(x_2, x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Conj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Fin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Sort(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Perm(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Order_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Perm_Support(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Fintype(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Sigma(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Perm_Sign(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Conj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Perm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Order_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Perm_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Fintype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__0 = _init_lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__0();
lean_mark_persistent(lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__0);
lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__1 = _init_lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__1();
lean_mark_persistent(lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Equiv_Perm_signAux_spec__0_spec__0___closed__1);
lp_mathlib_Equiv_Perm_signAux___lam__0___closed__0 = _init_lp_mathlib_Equiv_Perm_signAux___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_Perm_signAux___lam__0___closed__0);
lp_mathlib_Equiv_Perm_signAux___lam__0___closed__1 = _init_lp_mathlib_Equiv_Perm_signAux___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_Perm_signAux___lam__0___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
