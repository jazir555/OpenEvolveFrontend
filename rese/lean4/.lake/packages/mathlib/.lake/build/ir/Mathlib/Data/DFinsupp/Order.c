// Lean compiler output
// Module: Mathlib.Data.DFinsupp.Order
// Imports: public import Init public import Mathlib.Algebra.Order.Module.Defs public import Mathlib.Algebra.Order.Sub.Basic public import Mathlib.Data.DFinsupp.Module
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
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__0(lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPreorder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBot___redArg(lean_object*);
static lean_object* lp_mathlib_DFinsupp_instPreorder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPreorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_support___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBot___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_zipWith___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLE(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLE___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DFinsupp_instLE(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun___lam__0(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_orderEmbeddingToFun___lam__0), 2, 0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DFinsupp_orderEmbeddingToFun(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_DFinsupp_instPreorder___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPreorder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DFinsupp_instPreorder___closed__0;
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPreorder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DFinsupp_instPreorder(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instPartialOrder___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_DFinsupp_instPreorder(lean_box(0), lean_box(0), x_1, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DFinsupp_instPartialOrder___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DFinsupp_instPartialOrder(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_DFinsupp_instPartialOrder___redArg(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_1, x_2);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, x_3, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__1), 4, 1);
lean_closure_set(x_4, 0, x_2);
x_5 = lp_mathlib_DFinsupp_instPartialOrder___redArg(x_1, x_3);
lean_inc_n(x_1, 2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, lean_box(0));
lean_closure_set(x_6, 3, lean_box(0));
lean_closure_set(x_6, 4, x_1);
lean_closure_set(x_6, 5, x_1);
lean_closure_set(x_6, 6, x_1);
lean_closure_set(x_6, 7, x_4);
lean_closure_set(x_6, 8, lean_box(0));
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DFinsupp_instSemilatticeInf___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_1, x_2);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, x_3, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__1), 4, 1);
lean_closure_set(x_4, 0, x_2);
x_5 = lp_mathlib_DFinsupp_instPartialOrder___redArg(x_1, x_3);
lean_inc_n(x_1, 2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, lean_box(0));
lean_closure_set(x_6, 3, lean_box(0));
lean_closure_set(x_6, 4, x_1);
lean_closure_set(x_6, 5, x_1);
lean_closure_set(x_6, 6, x_1);
lean_closure_set(x_6, 7, x_4);
lean_closure_set(x_6, 8, lean_box(0));
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DFinsupp_instSemilatticeSup___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_apply_1(x_1, x_2);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_2(x_7, x_3, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_1, x_2);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, x_3, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lattice___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
lean_inc_ref(x_3);
lean_inc(x_1);
x_4 = lp_mathlib_DFinsupp_instSemilatticeInf___redArg(x_1, x_3);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_4, 1);
lean_dec(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lattice___redArg___lam__1), 4, 1);
lean_closure_set(x_7, 0, x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lattice___redArg___lam__2), 4, 1);
lean_closure_set(x_8, 0, x_3);
lean_inc_n(x_1, 3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, lean_box(0));
lean_closure_set(x_9, 3, lean_box(0));
lean_closure_set(x_9, 4, x_1);
lean_closure_set(x_9, 5, x_1);
lean_closure_set(x_9, 6, x_1);
lean_closure_set(x_9, 7, x_7);
lean_closure_set(x_9, 8, lean_box(0));
lean_ctor_set(x_4, 1, x_9);
lean_inc_n(x_1, 2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, lean_box(0));
lean_closure_set(x_10, 3, lean_box(0));
lean_closure_set(x_10, 4, x_1);
lean_closure_set(x_10, 5, x_1);
lean_closure_set(x_10, 6, x_1);
lean_closure_set(x_10, 7, x_8);
lean_closure_set(x_10, 8, lean_box(0));
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_4);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_12 = lean_ctor_get(x_4, 0);
lean_inc(x_12);
lean_dec(x_4);
x_13 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lattice___redArg___lam__1), 4, 1);
lean_closure_set(x_13, 0, x_2);
x_14 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lattice___redArg___lam__2), 4, 1);
lean_closure_set(x_14, 0, x_3);
lean_inc_n(x_1, 3);
x_15 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(x_15, 0, lean_box(0));
lean_closure_set(x_15, 1, lean_box(0));
lean_closure_set(x_15, 2, lean_box(0));
lean_closure_set(x_15, 3, lean_box(0));
lean_closure_set(x_15, 4, x_1);
lean_closure_set(x_15, 5, x_1);
lean_closure_set(x_15, 6, x_1);
lean_closure_set(x_15, 7, x_13);
lean_closure_set(x_15, 8, lean_box(0));
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_16, 1, x_15);
lean_inc_n(x_1, 2);
x_17 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(x_17, 0, lean_box(0));
lean_closure_set(x_17, 1, lean_box(0));
lean_closure_set(x_17, 2, lean_box(0));
lean_closure_set(x_17, 3, lean_box(0));
lean_closure_set(x_17, 4, x_1);
lean_closure_set(x_17, 5, x_1);
lean_closure_set(x_17, 6, x_1);
lean_closure_set(x_17, 7, x_14);
lean_closure_set(x_17, 8, lean_box(0));
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
return x_18;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_DFinsupp_lattice___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBot___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_3);
lean_dec_ref(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBot___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instOrderBot___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_box(0);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBot(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DFinsupp_instOrderBot___redArg(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBot___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_DFinsupp_instOrderBot(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
lean_dec_ref(x_2);
lean_inc(x_4);
x_8 = lean_apply_1(x_6, x_4);
lean_inc(x_4);
x_9 = lean_apply_1(x_7, x_4);
x_10 = lean_apply_3(x_3, x_4, x_8, x_9);
x_11 = lean_unbox(x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_DFinsupp_decidableLE___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; uint8_t x_8; 
lean_inc_ref(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_decidableLE___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_5);
lean_closure_set(x_6, 2, x_3);
x_7 = lp_mathlib_DFinsupp_support___redArg(x_1, x_2, x_4);
x_8 = lp_mathlib_Multiset_decidableDforallMultiset___redArg(x_7, x_6);
return x_8;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; 
x_11 = lp_mathlib_DFinsupp_decidableLE___redArg(x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lp_mathlib_DFinsupp_decidableLE(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_12 = lean_box(x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_DFinsupp_decidableLE___redArg(x_1, x_2, x_3, x_4, x_5);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_3);
lean_dec_ref(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_3(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_tsub___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_tsub___redArg___lam__1), 4, 1);
lean_closure_set(x_4, 0, x_2);
lean_inc_ref_n(x_3, 2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, lean_box(0));
lean_closure_set(x_5, 3, lean_box(0));
lean_closure_set(x_5, 4, x_3);
lean_closure_set(x_5, 5, x_3);
lean_closure_set(x_5, 6, x_3);
lean_closure_set(x_5, 7, x_4);
lean_closure_set(x_5, 8, lean_box(0));
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_DFinsupp_tsub___redArg(x_3, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_DFinsupp_tsub(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_DFinsupp_instPreorder___closed__0 = _init_lp_mathlib_DFinsupp_instPreorder___closed__0();
lean_mark_persistent(lp_mathlib_DFinsupp_instPreorder___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
