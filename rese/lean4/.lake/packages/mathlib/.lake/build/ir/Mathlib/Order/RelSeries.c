// Lean compiler output
// Module: Mathlib.Order.RelSeries
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Nat public import Mathlib.Algebra.Order.Group.Nat public import Mathlib.Algebra.Order.Monoid.NatCast public import Mathlib.Data.Fin.VecNotation public import Mathlib.Data.Fintype.Pi public import Mathlib.Data.Fintype.Pigeonhole public import Mathlib.Data.Fintype.Sigma public import Mathlib.Data.Rel public import Mathlib.Order.OrderIsoNat
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
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg___lam__0(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_last(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_ofLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_tail___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3(lean_object*, lean_object*);
static lean_object* lp_mathlib_RelSeries_Equiv___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0(lean_object*, lean_object*);
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_ofLE___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Sigma_instFintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_toList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_last___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_snoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_Equiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_head___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat(lean_object*, lean_object*);
static lean_object* lp_mathlib_RelSeries_Equiv___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListChain_x27___redArg(lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_snoc___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2(lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_fintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instFintype___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_succ___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_cons(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_cons___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_head(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range(lean_object*);
lean_object* l_instDecidableEqFin___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListChain_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton(lean_object*, lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
lean_object* lp_mathlib_Fin_succAboveCases___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_addCases___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2___boxed(lean_object*, lean_object*);
uint8_t l_Nat_decidableForallFin___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_toList___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_membership(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_ofFn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_RelSeries_Equiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat___lam__0), 2, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RelSeries_singleton___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_singleton___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RelSeries_singleton___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instInhabited(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RelSeries_singleton___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instInhabited___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RelSeries_singleton___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_ofLE___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_inc(x_3);
lean_dec(x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_ofLE(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RelSeries_ofLE___redArg(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_toList___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_unsigned_to_nat(1u);
x_5 = lean_nat_add(x_2, x_4);
lean_dec(x_2);
x_6 = l_List_ofFn___redArg(x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_toList(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RelSeries_toList___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_List_get___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = l_List_lengthTR___redArg(x_1);
lean_dec(x_1);
x_4 = lean_unsigned_to_nat(1u);
x_5 = lean_nat_sub(x_3, x_4);
lean_dec(x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RelSeries_fromListIsChain___redArg(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListChain_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RelSeries_fromListIsChain___redArg(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListChain_x27___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RelSeries_fromListIsChain___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_RelSeries_Equiv___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_toList___redArg), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_RelSeries_Equiv___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_fromListIsChain___redArg), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_RelSeries_Equiv___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_RelSeries_Equiv___closed__1;
x_2 = lp_mathlib_RelSeries_Equiv___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_Equiv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RelSeries_Equiv___closed__2;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_membership(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_head___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_unsigned_to_nat(1u);
x_5 = lean_nat_add(x_2, x_4);
lean_dec(x_2);
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_nat_mod(x_6, x_5);
lean_dec(x_5);
x_8 = lean_apply_1(x_3, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_head(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RelSeries_head___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_last___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_last(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RelSeries_last___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = l_Fin_addCases___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_RelSeries_append___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_nat_add(x_3, x_6);
lean_dec(x_6);
x_9 = lean_unsigned_to_nat(1u);
x_10 = lean_nat_add(x_8, x_9);
lean_dec(x_8);
x_11 = lean_nat_add(x_3, x_9);
lean_dec(x_3);
x_12 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_append___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_12, 0, x_11);
lean_closure_set(x_12, 1, x_4);
lean_closure_set(x_12, 2, x_7);
lean_ctor_set(x_2, 1, x_12);
lean_ctor_set(x_2, 0, x_10);
return x_2;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_13 = lean_ctor_get(x_2, 0);
x_14 = lean_ctor_get(x_2, 1);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_2);
x_15 = lean_nat_add(x_3, x_13);
lean_dec(x_13);
x_16 = lean_unsigned_to_nat(1u);
x_17 = lean_nat_add(x_15, x_16);
lean_dec(x_15);
x_18 = lean_nat_add(x_3, x_16);
lean_dec(x_3);
x_19 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_append___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_19, 0, x_18);
lean_closure_set(x_19, 1, x_4);
lean_closure_set(x_19, 2, x_14);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_17);
lean_ctor_set(x_20, 1, x_19);
return x_20;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RelSeries_append___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_map___redArg___lam__0), 3, 2);
lean_closure_set(x_5, 0, x_4);
lean_closure_set(x_5, 1, x_2);
lean_ctor_set(x_1, 1, x_5);
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_inc(x_6);
lean_dec(x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_map___redArg___lam__0), 3, 2);
lean_closure_set(x_8, 0, x_7);
lean_closure_set(x_8, 1, x_2);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_RelSeries_map___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Fin_succAboveCases___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_RelSeries_insertNth___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec(x_2);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_add(x_5, x_7);
lean_dec(x_5);
x_9 = l_Fin_succ___redArg(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_insertNth___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_10, 0, x_9);
lean_closure_set(x_10, 1, x_3);
lean_closure_set(x_10, 2, x_6);
lean_ctor_set(x_1, 1, x_10);
lean_ctor_set(x_1, 0, x_8);
return x_1;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_11 = lean_ctor_get(x_1, 0);
x_12 = lean_ctor_get(x_1, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_1);
x_13 = lean_unsigned_to_nat(1u);
x_14 = lean_nat_add(x_11, x_13);
lean_dec(x_11);
x_15 = l_Fin_succ___redArg(x_2);
x_16 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_insertNth___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_16, 0, x_15);
lean_closure_set(x_16, 1, x_3);
lean_closure_set(x_16, 2, x_12);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_14);
lean_ctor_set(x_17, 1, x_16);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_RelSeries_insertNth___redArg(x_3, x_4, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_RelSeries_insertNth(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RelSeries_insertNth___redArg(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_nat_add(x_4, x_1);
x_6 = lean_nat_sub(x_2, x_5);
lean_dec(x_5);
x_7 = lean_apply_1(x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_RelSeries_reverse___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec(x_2);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_unsigned_to_nat(1u);
x_6 = lean_nat_add(x_3, x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_reverse___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_7, 0, x_5);
lean_closure_set(x_7, 1, x_6);
lean_closure_set(x_7, 2, x_4);
lean_ctor_set(x_1, 1, x_7);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_1);
x_10 = lean_unsigned_to_nat(1u);
x_11 = lean_nat_add(x_8, x_10);
x_12 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_reverse___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_12, 0, x_10);
lean_closure_set(x_12, 1, x_11);
lean_closure_set(x_12, 2, x_9);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_8);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RelSeries_reverse___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_cons___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_RelSeries_singleton___redArg(x_2);
x_4 = lp_mathlib_RelSeries_append___redArg(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_cons(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RelSeries_cons___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_snoc___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_RelSeries_singleton___redArg(x_2);
x_4 = lp_mathlib_RelSeries_append___redArg(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_snoc(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RelSeries_snoc___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_tail___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RelSeries_tail___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_tail___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lean_unsigned_to_nat(1u);
x_7 = lean_nat_sub(x_3, x_6);
lean_dec(x_3);
lean_ctor_set(x_1, 1, x_5);
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_tail___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_10, 0, x_9);
x_11 = lean_unsigned_to_nat(1u);
x_12 = lean_nat_sub(x_8, x_11);
lean_dec(x_8);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_10);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_RelSeries_tail___redArg(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_RelSeries_head___redArg(x_2);
x_5 = lean_apply_1(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_4);
x_6 = lp_mathlib_RelSeries_tail___redArg(x_4);
x_7 = lp_mathlib_RelSeries_head___redArg(x_4);
lean_inc_ref(x_6);
x_8 = lean_apply_2(x_3, x_6, lean_box(0));
x_9 = lean_apply_4(x_1, x_6, x_7, lean_box(0), x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RelSeries_inductionOn___redArg___lam__1(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_3, 0);
x_5 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_inductionOn___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_inductionOn___redArg___lam__1___boxed), 5, 1);
lean_closure_set(x_6, 0, x_2);
x_7 = l_Nat_recCompiled___redArg(x_5, x_6, x_4);
lean_dec_ref(x_5);
x_8 = lean_apply_2(x_7, x_3, lean_box(0));
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_RelSeries_inductionOn___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_eraseLast___redArg___lam__0), 2, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lean_unsigned_to_nat(1u);
x_7 = lean_nat_sub(x_3, x_6);
lean_dec(x_3);
lean_ctor_set(x_1, 1, x_5);
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_eraseLast___redArg___lam__0), 2, 1);
lean_closure_set(x_10, 0, x_9);
x_11 = lean_unsigned_to_nat(1u);
x_12 = lean_nat_sub(x_8, x_11);
lean_dec(x_8);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_10);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RelSeries_eraseLast___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_4);
x_6 = lp_mathlib_RelSeries_eraseLast___redArg(x_4);
x_7 = lp_mathlib_RelSeries_last___redArg(x_4);
lean_inc_ref(x_6);
x_8 = lean_apply_2(x_3, x_6, lean_box(0));
x_9 = lean_apply_4(x_1, x_6, x_7, lean_box(0), x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_3, 0);
x_5 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_inductionOn___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1___boxed), 5, 1);
lean_closure_set(x_6, 0, x_2);
x_7 = l_Nat_recCompiled___redArg(x_5, x_6, x_4);
lean_dec_ref(x_5);
x_8 = lean_apply_2(x_7, x_3, lean_box(0));
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_RelSeries_inductionOn_x27___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = l_Fin_addCases___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_RelSeries_smash___redArg___lam__1(x_1, x_2, x_3, x_4);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_smash___redArg___lam__0), 2, 1);
lean_closure_set(x_8, 0, x_4);
lean_inc(x_3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_smash___redArg___lam__1___boxed), 4, 3);
lean_closure_set(x_9, 0, x_3);
lean_closure_set(x_9, 1, x_8);
lean_closure_set(x_9, 2, x_7);
x_10 = lean_nat_add(x_3, x_6);
lean_dec(x_6);
lean_dec(x_3);
lean_ctor_set(x_2, 1, x_9);
lean_ctor_set(x_2, 0, x_10);
return x_2;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_2, 0);
x_12 = lean_ctor_get(x_2, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_2);
x_13 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_smash___redArg___lam__0), 2, 1);
lean_closure_set(x_13, 0, x_4);
lean_inc(x_3);
x_14 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_smash___redArg___lam__1___boxed), 4, 3);
lean_closure_set(x_14, 0, x_3);
lean_closure_set(x_14, 1, x_13);
lean_closure_set(x_14, 2, x_12);
x_15 = lean_nat_add(x_3, x_11);
lean_dec(x_11);
lean_dec(x_3);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_14);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RelSeries_smash___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_take___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_RelSeries_take___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_nat_add(x_3, x_1);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RelSeries_drop___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_drop___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_5);
x_7 = lean_nat_sub(x_4, x_2);
lean_dec(x_2);
lean_dec(x_4);
lean_ctor_set(x_1, 1, x_6);
lean_ctor_set(x_1, 0, x_7);
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
lean_inc(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_RelSeries_drop___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_10, 0, x_2);
lean_closure_set(x_10, 1, x_9);
x_11 = lean_nat_sub(x_8, x_2);
lean_dec(x_2);
lean_dec(x_8);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_10);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_RelSeries_drop___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_3);
lean_ctor_set(x_6, 1, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LTSeries_mk(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono___lam__0(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_inc(x_3);
lean_dec(x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_LTSeries_injStrictMono___lam__0), 1, 0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_LTSeries_injStrictMono(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, lean_box(0));
lean_closure_set(x_5, 3, x_2);
lean_closure_set(x_5, 4, x_4);
lean_ctor_set(x_1, 1, x_5);
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_inc(x_6);
lean_dec(x_1);
x_8 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, lean_box(0));
lean_closure_set(x_8, 2, lean_box(0));
lean_closure_set(x_8, 3, x_2);
lean_closure_set(x_8, 4, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_LTSeries_map___redArg(x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_LTSeries_map(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_LTSeries_range___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_LTSeries_range___lam__0___boxed), 1, 0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = lean_nat_dec_lt(x_1, x_4);
if (x_5 == 0)
{
uint8_t x_6; 
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_6 = 1;
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
lean_inc(x_2);
x_7 = lean_apply_1(x_2, x_1);
x_8 = lean_apply_1(x_2, x_4);
x_9 = lean_apply_2(x_3, x_7, x_8);
x_10 = lean_unbox(x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_5, 0, x_4);
lean_closure_set(x_5, 1, x_1);
lean_closure_set(x_5, 2, x_2);
x_6 = l_Nat_decidableForallFin___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1(x_1, x_2, x_3, x_4);
lean_dec(x_3);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_unsigned_to_nat(1u);
x_6 = lean_nat_add(x_3, x_5);
lean_dec(x_3);
lean_inc(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1___boxed), 4, 3);
lean_closure_set(x_7, 0, x_4);
lean_closure_set(x_7, 1, x_1);
lean_closure_set(x_7, 2, x_6);
x_8 = l_Nat_decidableForallFin___redArg(x_6, x_7);
lean_dec(x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_unsigned_to_nat(1u);
x_4 = lean_nat_add(x_2, x_3);
lean_inc(x_4);
x_5 = lean_alloc_closure((void*)(l_instDecidableEqFin___boxed), 3, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = l_List_finRange(x_4);
x_7 = lp_mathlib_Pi_instFintype___redArg(x_5, x_6, x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_LTSeries_injStrictMono___lam__0), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2___boxed), 2, 1);
lean_closure_set(x_3, 0, x_2);
lean_inc(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3___boxed), 2, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4___boxed), 2, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = l_List_lengthTR___redArg(x_1);
lean_dec(x_1);
x_7 = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___closed__0;
x_8 = l_List_finRange(x_6);
x_9 = lp_mathlib_Sigma_instFintype___redArg(x_5, x_8);
x_10 = lp_mathlib_Subtype_fintype___redArg(x_3, x_9);
x_11 = lp_mathlib_Finset_map___redArg(x_7, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LTSeries_instFintypeOfDecidableLT(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_NatCast(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rel(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_OrderIsoNat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_RelSeries(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_NatCast(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_OrderIsoNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_RelSeries_Equiv___closed__0 = _init_lp_mathlib_RelSeries_Equiv___closed__0();
lean_mark_persistent(lp_mathlib_RelSeries_Equiv___closed__0);
lp_mathlib_RelSeries_Equiv___closed__1 = _init_lp_mathlib_RelSeries_Equiv___closed__1();
lean_mark_persistent(lp_mathlib_RelSeries_Equiv___closed__1);
lp_mathlib_RelSeries_Equiv___closed__2 = _init_lp_mathlib_RelSeries_Equiv___closed__2();
lean_mark_persistent(lp_mathlib_RelSeries_Equiv___closed__2);
lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___closed__0 = _init_lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___closed__0();
lean_mark_persistent(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
