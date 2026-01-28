// Lean compiler output
// Module: Mathlib.GroupTheory.Abelianization.Defs
// Imports: public import Init public import Mathlib.GroupTheory.Commutator.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_abelianizationCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Con_lift___redArg(lean_object*);
static lean_object* lp_mathlib_Abelianization_lift___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_abelianizationCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_equivOfComm___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_id___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAbelianization___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_QuotientGroup_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_of___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_equivOfComm(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_of(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAbelianization___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAbelianization(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAbelianization___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_equivOfComm___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_instInhabited___redArg___boxed(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_QuotientGroup_Quotient_group___redArg(lean_object*);
static lean_object* lp_mathlib_Abelianization_equivOfComm___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_equivOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_abelianizationCongr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_commGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_commGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_commGroup(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_QuotientGroup_Quotient_group___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_commGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_QuotientGroup_Quotient_group___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_instInhabited___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lp_mathlib_Monoid_toMulOneClass___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Abelianization_instInhabited___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_instInhabited___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Abelianization_instInhabited(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_instInhabited___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Abelianization_instInhabited___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_of___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_box(0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
lean_closure_set(x_3, 2, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_of(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Abelianization_of___redArg(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Abelianization_lift___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_Abelianization_of___redArg(x_1);
x_5 = lp_mathlib_MonoidHom_comp___redArg(x_2, x_4);
x_6 = lean_apply_1(x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Abelianization_lift___redArg___closed__0;
x_3 = lean_alloc_closure((void*)(lp_mathlib_Abelianization_lift___redArg___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Abelianization_lift___redArg(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Abelianization_lift(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_map___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_Abelianization_lift___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_Abelianization_of___redArg(x_2);
x_7 = lp_mathlib_MonoidHom_comp___redArg(x_6, x_3);
x_8 = lean_apply_1(x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Abelianization_map___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_abelianizationCongr___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Abelianization_map___redArg(x_1, x_2, x_3);
x_6 = lean_apply_1(x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_abelianizationCongr___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lp_mathlib_Equiv_symm___redArg(x_3);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_5, 0);
x_8 = lean_ctor_get(x_5, 1);
lean_dec(x_8);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_abelianizationCongr___redArg___lam__0), 4, 3);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_2);
lean_closure_set(x_9, 2, x_4);
x_10 = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_abelianizationCongr___redArg___lam__0), 4, 3);
lean_closure_set(x_10, 0, x_2);
lean_closure_set(x_10, 1, x_1);
lean_closure_set(x_10, 2, x_7);
lean_ctor_set(x_5, 1, x_10);
lean_ctor_set(x_5, 0, x_9);
return x_5;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_5, 0);
lean_inc(x_11);
lean_dec(x_5);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_12 = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_abelianizationCongr___redArg___lam__0), 4, 3);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_2);
lean_closure_set(x_12, 2, x_4);
x_13 = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_abelianizationCongr___redArg___lam__0), 4, 3);
lean_closure_set(x_13, 0, x_2);
lean_closure_set(x_13, 1, x_1);
lean_closure_set(x_13, 2, x_11);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_abelianizationCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MulEquiv_abelianizationCongr___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_equivOfComm___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Abelianization_of___redArg(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Abelianization_equivOfComm___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_OneHom_id___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_equivOfComm___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_equivOfComm___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_Abelianization_lift___redArg(x_1);
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
lean_dec(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Abelianization_equivOfComm___redArg___lam__0), 2, 1);
lean_closure_set(x_6, 0, x_1);
x_7 = lp_mathlib_Abelianization_equivOfComm___redArg___closed__0;
x_8 = lean_alloc_closure((void*)(lp_mathlib_Abelianization_equivOfComm___redArg___lam__1), 3, 2);
lean_closure_set(x_8, 0, x_4);
lean_closure_set(x_8, 1, x_7);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_6);
return x_2;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_2, 0);
lean_inc(x_9);
lean_dec(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Abelianization_equivOfComm___redArg___lam__0), 2, 1);
lean_closure_set(x_10, 0, x_1);
x_11 = lp_mathlib_Abelianization_equivOfComm___redArg___closed__0;
x_12 = lean_alloc_closure((void*)(lp_mathlib_Abelianization_equivOfComm___redArg___lam__1), 3, 2);
lean_closure_set(x_12, 0, x_9);
lean_closure_set(x_12, 1, x_11);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_10);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_equivOfComm(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Abelianization_equivOfComm___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAbelianization(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAbelianization___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAbelianization___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instUniqueAbelianization(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAbelianization___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_instUniqueAbelianization___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Commutator_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Abelianization_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Commutator_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Abelianization_lift___redArg___closed__0 = _init_lp_mathlib_Abelianization_lift___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Abelianization_lift___redArg___closed__0);
lp_mathlib_Abelianization_equivOfComm___redArg___closed__0 = _init_lp_mathlib_Abelianization_equivOfComm___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Abelianization_equivOfComm___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
