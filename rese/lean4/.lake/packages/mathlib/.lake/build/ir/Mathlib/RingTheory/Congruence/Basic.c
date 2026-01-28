// Lean compiler output
// Module: Mathlib.RingTheory.Congruence.Basic
// Imports: public import Init public import Mathlib.Algebra.Ring.Action.Basic public import Mathlib.GroupTheory.Congruence.Basic public import Mathlib.RingTheory.Congruence.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPartialOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSMulQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSMulQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instLE___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Setoid_completeLattice(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_RingCon_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instLE(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Con_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Con_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSMulQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSMulQuotient(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Con_instSMul___redArg(x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSMulQuotient___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Con_instSMul___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSMulQuotient___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_RingCon_instSMulQuotient(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instLE(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instLE___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RingCon_instLE(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_RingCon_instInfSet___lam__0), 1, 0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RingCon_instInfSet(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_RingCon_instPartialOrder___closed__0() {
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
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPartialOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RingCon_instPartialOrder___closed__0;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPartialOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RingCon_instPartialOrder(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_RingCon_instInfSet___lam__0), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_RingCon_instCompleteLattice___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Setoid_completeLattice(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_3 = lp_mathlib_RingCon_instPartialOrder(lean_box(0), x_1, x_2);
x_4 = lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0;
x_5 = lp_mathlib_completeLatticeOfInf___redArg(x_3, x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_5, 2);
lean_inc(x_8);
lean_dec_ref(x_5);
x_9 = !lean_is_exclusive(x_6);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_10 = lean_ctor_get(x_6, 1);
lean_dec(x_10);
x_11 = lp_mathlib_RingCon_instCompleteLattice___redArg___closed__1;
x_12 = !lean_is_exclusive(x_11);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_13 = lean_ctor_get(x_11, 3);
x_14 = lean_ctor_get(x_11, 2);
lean_dec(x_14);
x_15 = lean_ctor_get(x_11, 1);
lean_dec(x_15);
x_16 = lean_ctor_get(x_11, 0);
lean_dec(x_16);
x_17 = !lean_is_exclusive(x_13);
if (x_17 == 0)
{
lean_object* x_18; 
x_18 = lean_alloc_closure((void*)(lp_mathlib_RingCon_instCompleteLattice___redArg___lam__0), 2, 0);
lean_ctor_set(x_6, 1, x_18);
lean_ctor_set(x_11, 2, x_8);
lean_ctor_set(x_11, 1, x_7);
lean_ctor_set(x_11, 0, x_6);
return x_11;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_19 = lean_ctor_get(x_13, 0);
x_20 = lean_ctor_get(x_13, 1);
lean_inc(x_20);
lean_inc(x_19);
lean_dec(x_13);
x_21 = lean_alloc_closure((void*)(lp_mathlib_RingCon_instCompleteLattice___redArg___lam__0), 2, 0);
lean_ctor_set(x_6, 1, x_21);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_19);
lean_ctor_set(x_22, 1, x_20);
lean_ctor_set(x_11, 3, x_22);
lean_ctor_set(x_11, 2, x_8);
lean_ctor_set(x_11, 1, x_7);
lean_ctor_set(x_11, 0, x_6);
return x_11;
}
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_23 = lean_ctor_get(x_11, 3);
lean_inc(x_23);
lean_dec(x_11);
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
x_25 = lean_ctor_get(x_23, 1);
lean_inc(x_25);
if (lean_is_exclusive(x_23)) {
 lean_ctor_release(x_23, 0);
 lean_ctor_release(x_23, 1);
 x_26 = x_23;
} else {
 lean_dec_ref(x_23);
 x_26 = lean_box(0);
}
x_27 = lean_alloc_closure((void*)(lp_mathlib_RingCon_instCompleteLattice___redArg___lam__0), 2, 0);
lean_ctor_set(x_6, 1, x_27);
if (lean_is_scalar(x_26)) {
 x_28 = lean_alloc_ctor(0, 2, 0);
} else {
 x_28 = x_26;
}
lean_ctor_set(x_28, 0, x_24);
lean_ctor_set(x_28, 1, x_25);
x_29 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_29, 0, x_6);
lean_ctor_set(x_29, 1, x_7);
lean_ctor_set(x_29, 2, x_8);
lean_ctor_set(x_29, 3, x_28);
return x_29;
}
}
else
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_30 = lean_ctor_get(x_6, 0);
lean_inc(x_30);
lean_dec(x_6);
x_31 = lp_mathlib_RingCon_instCompleteLattice___redArg___closed__1;
x_32 = lean_ctor_get(x_31, 3);
lean_inc_ref(x_32);
if (lean_is_exclusive(x_31)) {
 lean_ctor_release(x_31, 0);
 lean_ctor_release(x_31, 1);
 lean_ctor_release(x_31, 2);
 lean_ctor_release(x_31, 3);
 x_33 = x_31;
} else {
 lean_dec_ref(x_31);
 x_33 = lean_box(0);
}
x_34 = lean_ctor_get(x_32, 0);
lean_inc(x_34);
x_35 = lean_ctor_get(x_32, 1);
lean_inc(x_35);
if (lean_is_exclusive(x_32)) {
 lean_ctor_release(x_32, 0);
 lean_ctor_release(x_32, 1);
 x_36 = x_32;
} else {
 lean_dec_ref(x_32);
 x_36 = lean_box(0);
}
x_37 = lean_alloc_closure((void*)(lp_mathlib_RingCon_instCompleteLattice___redArg___lam__0), 2, 0);
x_38 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_38, 0, x_30);
lean_ctor_set(x_38, 1, x_37);
if (lean_is_scalar(x_36)) {
 x_39 = lean_alloc_ctor(0, 2, 0);
} else {
 x_39 = x_36;
}
lean_ctor_set(x_39, 0, x_34);
lean_ctor_set(x_39, 1, x_35);
if (lean_is_scalar(x_33)) {
 x_40 = lean_alloc_ctor(0, 4, 0);
} else {
 x_40 = x_33;
}
lean_ctor_set(x_40, 0, x_38);
lean_ctor_set(x_40, 1, x_7);
lean_ctor_set(x_40, 2, x_8);
lean_ctor_set(x_40, 3, x_39);
return x_40;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RingCon_instCompleteLattice___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RingCon_instCompleteLattice(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_RingCon_instCompleteLattice___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_RingCon_gi___lam__0), 2, 0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RingCon_gi(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Congruence_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_RingCon_instPartialOrder___closed__0 = _init_lp_mathlib_RingCon_instPartialOrder___closed__0();
lean_mark_persistent(lp_mathlib_RingCon_instPartialOrder___closed__0);
lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0 = _init_lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0();
lean_mark_persistent(lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0);
lp_mathlib_RingCon_instCompleteLattice___redArg___closed__1 = _init_lp_mathlib_RingCon_instCompleteLattice___redArg___closed__1();
lean_mark_persistent(lp_mathlib_RingCon_instCompleteLattice___redArg___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
