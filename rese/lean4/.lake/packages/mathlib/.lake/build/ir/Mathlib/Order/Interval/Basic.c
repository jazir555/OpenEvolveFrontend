// Lean compiler output
// Module: Mathlib.Order.Interval.Basic
// Imports: public import Init public import Mathlib.Order.Interval.Set.Basic public import Mathlib.Data.Set.Lattice.Image public import Mathlib.Data.SetLike.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Interval_boundedOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_WithBot_instPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_lattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_instLE(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instOrderTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_pure___redArg(lean_object*);
lean_object* lp_mathlib_WithBot_some(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_instUniqueOfIsEmpty(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_pure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instPreorder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_setLike___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_coeHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_coeHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_dual(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_recBotCoe___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableLE__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_pure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_recBotCoe___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_dual___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_le(lean_object*, lean_object*);
lean_object* l_Prod_map___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqProd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instMax(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProd___redArg(lean_object*);
lean_object* lp_mathlib_WithBot_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_boundedOrder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_partialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instMax___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_instPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_dual(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_instPreorder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instCoeSet(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableLE__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_NonemptyInterval_dual___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProdHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_pure___boxed(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Function_Injective_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_setLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_pure___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_instOrderBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instMax___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_semilatticeSup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_recBotCoe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_instPreorder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProd___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instMembership___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_lattice___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_lattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_setLike___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instSemilatticeSup(lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_partialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_coeHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_instPreorder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_pure___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_NonemptyInterval_dual___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instSemilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instOrderTop(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_withBotCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_partialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instMembership(lean_object*, lean_object*);
static lean_object* lp_mathlib_Interval_instCoeNonemptyInterval___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Interval_partialOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instInhabited___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProdHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_recBotCoe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instCoeSet___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_NonemptyInterval_instPreorder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableLE__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_setLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_instCoeNonemptyInterval(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Interval_boundedOrder___redArg(lean_object*);
lean_object* l_Prod_swap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProd___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableLE__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_recBotCoe___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_coeHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProd(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProd___redArg(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProd___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_NonemptyInterval_toDualProd(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProd___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_NonemptyInterval_toDualProd___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableEq___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
lean_inc_ref(x_1);
x_4 = l_instDecidableEqProd___redArg(x_1, x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableEq___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_NonemptyInterval_instDecidableEq___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableEq___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_NonemptyInterval_instDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_5, 0, x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_NonemptyInterval_toDualProd___boxed), 3, 2);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, x_1);
x_7 = lp_mathlib_Function_Injective_decidableEq___redArg(x_6, x_5, x_3, x_4);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableEq(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = lp_mathlib_NonemptyInterval_instDecidableEq___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableEq___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_NonemptyInterval_instDecidableEq(x_1, x_2, x_3, x_4, x_5);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableEq___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_NonemptyInterval_instDecidableEq___redArg(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_le(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableLE___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_7);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
x_8 = lean_apply_2(x_1, x_4, x_6);
x_9 = lean_unbox(x_8);
if (x_9 == 0)
{
uint8_t x_10; 
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_1);
x_10 = lean_unbox(x_8);
return x_10;
}
else
{
lean_object* x_11; uint8_t x_12; 
x_11 = lean_apply_2(x_1, x_7, x_5);
x_12 = lean_unbox(x_11);
return x_12;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableLE(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = lp_mathlib_NonemptyInterval_instDecidableLE___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableLE___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_NonemptyInterval_instDecidableLE(x_1, x_2, x_3, x_4, x_5);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableLE___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_NonemptyInterval_instDecidableLE___redArg(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProdHom(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_NonemptyInterval_toDualProd___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_toDualProdHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_NonemptyInterval_toDualProd___boxed), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_NonemptyInterval_dual___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Prod_swap___redArg), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_NonemptyInterval_dual___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_NonemptyInterval_dual___closed__0;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_dual(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonemptyInterval_dual___closed__1;
return x_3;
}
}
static lean_object* _init_lp_mathlib_NonemptyInterval_instPreorder___closed__0() {
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
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instPreorder(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonemptyInterval_instPreorder___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instPreorder___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonemptyInterval_instPreorder(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instCoeSet(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instCoeSet___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonemptyInterval_instCoeSet(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instMembership(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instMembership___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonemptyInterval_instMembership(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_pure(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
lean_inc(x_3);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_pure___redArg(lean_object* x_1) {
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
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_pure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_NonemptyInterval_pure(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instInhabited(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
lean_inc(x_3);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instInhabited___redArg(lean_object* x_1) {
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
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instInhabited___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_NonemptyInterval_instInhabited(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_NonemptyInterval_map___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_1);
lean_inc_ref(x_3);
x_4 = l_Prod_map___redArg(x_3, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_NonemptyInterval_map___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_NonemptyInterval_map(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map_u2082___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_dec_ref(x_2);
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_3, 0);
x_8 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
x_9 = lean_apply_2(x_1, x_4, x_7);
x_10 = lean_apply_2(x_1, x_5, x_8);
lean_ctor_set(x_3, 1, x_10);
lean_ctor_set(x_3, 0, x_9);
return x_3;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_ctor_get(x_3, 0);
x_12 = lean_ctor_get(x_3, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_3);
lean_inc(x_1);
x_13 = lean_apply_2(x_1, x_4, x_11);
x_14 = lean_apply_2(x_1, x_5, x_12);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map_u2082(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_NonemptyInterval_map_u2082___redArg(x_7, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_map_u2082___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_NonemptyInterval_map_u2082(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instOrderTop___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
lean_ctor_set(x_1, 1, x_3);
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_1);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_5);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instOrderTop(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_NonemptyInterval_instOrderTop___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instOrderTop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_NonemptyInterval_instOrderTop(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instPartialOrder(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonemptyInterval_instPreorder___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instPartialOrder___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonemptyInterval_instPartialOrder(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableLE__1___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_7);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
x_8 = lean_apply_2(x_1, x_4, x_6);
x_9 = lean_unbox(x_8);
if (x_9 == 0)
{
uint8_t x_10; 
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_1);
x_10 = lean_unbox(x_8);
return x_10;
}
else
{
lean_object* x_11; uint8_t x_12; 
x_11 = lean_apply_2(x_1, x_7, x_5);
x_12 = lean_unbox(x_11);
return x_12;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonemptyInterval_instDecidableLE__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = lp_mathlib_NonemptyInterval_instDecidableLE__1___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableLE__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_NonemptyInterval_instDecidableLE__1(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instDecidableLE__1___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_NonemptyInterval_instDecidableLE__1___redArg(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_coeHom(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_coeHom___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonemptyInterval_coeHom(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_setLike(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_setLike___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonemptyInterval_setLike(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instMax___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = !lean_is_exclusive(x_4);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_4, 0);
x_9 = lean_ctor_get(x_4, 1);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_dec_ref(x_1);
x_11 = lean_apply_2(x_2, x_5, x_8);
x_12 = lean_apply_2(x_10, x_6, x_9);
lean_ctor_set(x_4, 1, x_12);
lean_ctor_set(x_4, 0, x_11);
return x_4;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_13 = lean_ctor_get(x_4, 0);
x_14 = lean_ctor_get(x_4, 1);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_4);
x_15 = lean_ctor_get(x_1, 1);
lean_inc(x_15);
lean_dec_ref(x_1);
x_16 = lean_apply_2(x_2, x_5, x_13);
x_17 = lean_apply_2(x_15, x_6, x_14);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
return x_18;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instMax___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_NonemptyInterval_instMax___redArg___lam__0), 4, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instMax(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonemptyInterval_instMax___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instSemilatticeSup___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = !lean_is_exclusive(x_3);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_3, 0);
x_10 = lean_ctor_get(x_3, 1);
x_11 = lean_ctor_get(x_4, 1);
lean_inc(x_11);
lean_dec_ref(x_4);
x_12 = lean_apply_2(x_5, x_6, x_9);
x_13 = lean_apply_2(x_11, x_7, x_10);
lean_ctor_set(x_3, 1, x_13);
lean_ctor_set(x_3, 0, x_12);
return x_3;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_14 = lean_ctor_get(x_3, 0);
x_15 = lean_ctor_get(x_3, 1);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_3);
x_16 = lean_ctor_get(x_4, 1);
lean_inc(x_16);
lean_dec_ref(x_4);
x_17 = lean_apply_2(x_5, x_6, x_14);
x_18 = lean_apply_2(x_16, x_7, x_15);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instSemilatticeSup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_NonemptyInterval_instSemilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_NonemptyInterval_instPreorder___closed__0;
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonemptyInterval_instSemilatticeSup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_NonemptyInterval_instSemilatticeSup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_instLE(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_instOrderBot(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Interval_instCoeNonemptyInterval___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_WithBot_some), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_instCoeNonemptyInterval(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Interval_instCoeNonemptyInterval___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_recBotCoe(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_WithBot_recBotCoe___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_recBotCoe___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_WithBot_recBotCoe___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_recBotCoe___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Interval_recBotCoe(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_recBotCoe___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Interval_recBotCoe___redArg(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_instUniqueOfIsEmpty(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_dual___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_NonemptyInterval_dual(lean_box(0), x_1);
x_3 = lp_mathlib_Equiv_withBotCongr___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_dual(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Interval_dual___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_instPreorder___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_NonemptyInterval_instPreorder(lean_box(0), x_1);
x_3 = lp_mathlib_WithBot_instPreorder(lean_box(0), x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_instPreorder(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Interval_instPreorder___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_instPreorder___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Interval_instPreorder(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_instPreorder___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Interval_instPreorder___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_pure___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
lean_inc(x_1);
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
x_3 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_pure(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Interval_pure___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_pure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Interval_pure(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_map___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_NonemptyInterval_map___boxed), 6, 5);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, x_1);
lean_closure_set(x_5, 3, x_2);
lean_closure_set(x_5, 4, x_3);
x_6 = lp_mathlib_WithBot_map___redArg(x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Interval_map___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_boundedOrder___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_NonemptyInterval_instOrderTop___redArg(x_1);
x_3 = lp_mathlib_WithBot_instBoundedOrder___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_boundedOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Interval_boundedOrder___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_boundedOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Interval_boundedOrder(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_partialOrder___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_NonemptyInterval_instPartialOrder(lean_box(0), x_1);
x_3 = lp_mathlib_WithBot_instPreorder(lean_box(0), x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_partialOrder(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Interval_partialOrder___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_partialOrder___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Interval_partialOrder(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_partialOrder___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Interval_partialOrder___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_coeHom(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_coeHom___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Interval_coeHom(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_setLike(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_setLike___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Interval_setLike(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_semilatticeSup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_NonemptyInterval_instSemilatticeSup___redArg(x_1);
x_3 = lp_mathlib_WithBot_semilatticeSup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_semilatticeSup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Interval_semilatticeSup___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_lattice___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_3;
}
else
{
if (lean_obj_tag(x_4) == 0)
{
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
else
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = !lean_is_exclusive(x_4);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_7 = lean_ctor_get(x_4, 0);
x_8 = lean_ctor_get(x_5, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_5, 1);
lean_inc(x_9);
lean_dec(x_5);
x_10 = !lean_is_exclusive(x_7);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_11 = lean_ctor_get(x_7, 0);
x_12 = lean_ctor_get(x_7, 1);
lean_inc_ref(x_1);
lean_inc(x_12);
lean_inc(x_8);
x_13 = lean_apply_2(x_1, x_8, x_12);
x_14 = lean_unbox(x_13);
if (x_14 == 0)
{
lean_object* x_15; 
lean_free_object(x_7);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_9);
lean_dec(x_8);
lean_free_object(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_15 = lean_box(0);
return x_15;
}
else
{
lean_object* x_16; uint8_t x_17; 
lean_inc(x_9);
lean_inc(x_11);
x_16 = lean_apply_2(x_1, x_11, x_9);
x_17 = lean_unbox(x_16);
if (x_17 == 0)
{
lean_object* x_18; 
lean_free_object(x_7);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_9);
lean_dec(x_8);
lean_free_object(x_4);
lean_dec_ref(x_2);
x_18 = lean_box(0);
return x_18;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_19 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_19);
x_20 = lean_ctor_get(x_2, 1);
lean_inc(x_20);
lean_dec_ref(x_2);
x_21 = lean_ctor_get(x_19, 1);
lean_inc(x_21);
lean_dec_ref(x_19);
x_22 = lean_apply_2(x_21, x_8, x_11);
x_23 = lean_apply_2(x_20, x_9, x_12);
lean_ctor_set(x_7, 1, x_23);
lean_ctor_set(x_7, 0, x_22);
return x_4;
}
}
}
else
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; uint8_t x_27; 
x_24 = lean_ctor_get(x_7, 0);
x_25 = lean_ctor_get(x_7, 1);
lean_inc(x_25);
lean_inc(x_24);
lean_dec(x_7);
lean_inc_ref(x_1);
lean_inc(x_25);
lean_inc(x_8);
x_26 = lean_apply_2(x_1, x_8, x_25);
x_27 = lean_unbox(x_26);
if (x_27 == 0)
{
lean_object* x_28; 
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_9);
lean_dec(x_8);
lean_free_object(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_28 = lean_box(0);
return x_28;
}
else
{
lean_object* x_29; uint8_t x_30; 
lean_inc(x_9);
lean_inc(x_24);
x_29 = lean_apply_2(x_1, x_24, x_9);
x_30 = lean_unbox(x_29);
if (x_30 == 0)
{
lean_object* x_31; 
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_9);
lean_dec(x_8);
lean_free_object(x_4);
lean_dec_ref(x_2);
x_31 = lean_box(0);
return x_31;
}
else
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_32 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_32);
x_33 = lean_ctor_get(x_2, 1);
lean_inc(x_33);
lean_dec_ref(x_2);
x_34 = lean_ctor_get(x_32, 1);
lean_inc(x_34);
lean_dec_ref(x_32);
x_35 = lean_apply_2(x_34, x_8, x_24);
x_36 = lean_apply_2(x_33, x_9, x_25);
x_37 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_37, 0, x_35);
lean_ctor_set(x_37, 1, x_36);
lean_ctor_set(x_4, 0, x_37);
return x_4;
}
}
}
}
else
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; uint8_t x_45; 
x_38 = lean_ctor_get(x_4, 0);
lean_inc(x_38);
lean_dec(x_4);
x_39 = lean_ctor_get(x_5, 0);
lean_inc(x_39);
x_40 = lean_ctor_get(x_5, 1);
lean_inc(x_40);
lean_dec(x_5);
x_41 = lean_ctor_get(x_38, 0);
lean_inc(x_41);
x_42 = lean_ctor_get(x_38, 1);
lean_inc(x_42);
if (lean_is_exclusive(x_38)) {
 lean_ctor_release(x_38, 0);
 lean_ctor_release(x_38, 1);
 x_43 = x_38;
} else {
 lean_dec_ref(x_38);
 x_43 = lean_box(0);
}
lean_inc_ref(x_1);
lean_inc(x_42);
lean_inc(x_39);
x_44 = lean_apply_2(x_1, x_39, x_42);
x_45 = lean_unbox(x_44);
if (x_45 == 0)
{
lean_object* x_46; 
lean_dec(x_43);
lean_dec(x_42);
lean_dec(x_41);
lean_dec(x_40);
lean_dec(x_39);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_46 = lean_box(0);
return x_46;
}
else
{
lean_object* x_47; uint8_t x_48; 
lean_inc(x_40);
lean_inc(x_41);
x_47 = lean_apply_2(x_1, x_41, x_40);
x_48 = lean_unbox(x_47);
if (x_48 == 0)
{
lean_object* x_49; 
lean_dec(x_43);
lean_dec(x_42);
lean_dec(x_41);
lean_dec(x_40);
lean_dec(x_39);
lean_dec_ref(x_2);
x_49 = lean_box(0);
return x_49;
}
else
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; 
x_50 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_50);
x_51 = lean_ctor_get(x_2, 1);
lean_inc(x_51);
lean_dec_ref(x_2);
x_52 = lean_ctor_get(x_50, 1);
lean_inc(x_52);
lean_dec_ref(x_50);
x_53 = lean_apply_2(x_52, x_39, x_41);
x_54 = lean_apply_2(x_51, x_40, x_42);
if (lean_is_scalar(x_43)) {
 x_55 = lean_alloc_ctor(0, 2, 0);
} else {
 x_55 = x_43;
}
lean_ctor_set(x_55, 0, x_53);
lean_ctor_set(x_55, 1, x_54);
x_56 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_56, 0, x_55);
return x_56;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_lattice___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Interval_lattice___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
x_4 = lp_mathlib_Interval_semilatticeSup___redArg(x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Interval_lattice(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Interval_lattice___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Interval_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_NonemptyInterval_dual___closed__0 = _init_lp_mathlib_NonemptyInterval_dual___closed__0();
lean_mark_persistent(lp_mathlib_NonemptyInterval_dual___closed__0);
lp_mathlib_NonemptyInterval_dual___closed__1 = _init_lp_mathlib_NonemptyInterval_dual___closed__1();
lean_mark_persistent(lp_mathlib_NonemptyInterval_dual___closed__1);
lp_mathlib_NonemptyInterval_instPreorder___closed__0 = _init_lp_mathlib_NonemptyInterval_instPreorder___closed__0();
lean_mark_persistent(lp_mathlib_NonemptyInterval_instPreorder___closed__0);
lp_mathlib_Interval_instCoeNonemptyInterval___closed__0 = _init_lp_mathlib_Interval_instCoeNonemptyInterval___closed__0();
lean_mark_persistent(lp_mathlib_Interval_instCoeNonemptyInterval___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
