// Lean compiler output
// Module: Mathlib.LinearAlgebra.CliffordAlgebra.Grading
// Imports: public import Init public import Mathlib.LinearAlgebra.CliffordAlgebra.Basic public import Mathlib.RingTheory.GradedAlgebra.Basic
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
lean_object* lp_mathlib_DFinsupp_lsingle___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_galgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg(lean_object*);
lean_object* lp_mathlib_ZMod_commRing(lean_object*);
static lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__2;
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_completeLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_evenOdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_evenOdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_evenOdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_semiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_instAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SetLike_gsemiring___redArg(lean_object*);
lean_object* lp_mathlib_instRingCliffordAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_CliffordAlgebra_00_u03b9___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CliffordAlgebra_lift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZMod_decidableEq___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__0;
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
static lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__1;
lean_object* lp_mathlib_LinearMap_codRestrict___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instAlgebraCliffordAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_evenOdd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_3 = lp_mathlib_instRingCliffordAlgebra___redArg(x_1);
x_4 = lp_mathlib_Ring_toNonAssocRing___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = lp_mathlib_instAlgebraCliffordAlgebra___redArg(x_1);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_Submodule_completeLattice(lean_box(0), lean_box(0), x_2, x_7, x_9);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_2);
x_11 = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(x_10);
x_12 = lean_ctor_get(x_11, 1);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lean_apply_1(x_12, lean_box(0));
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_evenOdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CliffordAlgebra_evenOdd___redArg(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_evenOdd___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CliffordAlgebra_evenOdd(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_8;
}
}
static lean_object* _init_lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lp_mathlib_ZMod_commRing(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__0;
x_2 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_alloc_closure((void*)(lp_mathlib_ZMod_decidableEq___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_instRingCliffordAlgebra___redArg(x_1);
x_3 = lp_mathlib_Ring_toNonAssocRing___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__1;
x_8 = lean_ctor_get(x_7, 1);
lean_inc_ref(x_8);
x_9 = lean_ctor_get(x_8, 2);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_10, 0, x_6);
x_11 = lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__2;
x_12 = lp_mathlib_DFinsupp_lsingle___redArg(x_10, x_11, x_9);
x_13 = lp_mathlib_CliffordAlgebra_00_u03b9___redArg(x_1);
x_14 = lp_mathlib_LinearMap_codRestrict___redArg(x_13);
x_15 = lp_mathlib_LinearMap_comp___redArg(x_12, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0(x_1, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg___lam__1(x_1, x_2, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_2 = lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__1;
x_3 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
lean_inc_ref(x_1);
x_5 = lp_mathlib_instRingCliffordAlgebra___redArg(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
lean_inc_ref(x_1);
x_7 = lp_mathlib_instAlgebraCliffordAlgebra___redArg(x_1);
x_8 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_6);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
x_11 = lean_ctor_get(x_7, 0);
lean_inc(x_11);
x_12 = lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__2;
x_13 = lean_alloc_closure((void*)(lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_13, 0, x_10);
x_14 = lean_alloc_closure((void*)(lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg___lam__1___boxed), 4, 1);
lean_closure_set(x_14, 0, x_11);
x_15 = lp_mathlib_SetLike_gsemiring___redArg(x_6);
lean_inc_ref(x_4);
lean_inc_ref(x_13);
x_16 = lp_mathlib_DirectSum_semiring___redArg(x_12, x_13, x_4, x_15);
x_17 = lp_mathlib_Submodule_galgebra___redArg(x_7);
x_18 = lp_mathlib_DirectSum_instAlgebra___redArg(x_13, x_14, x_4, x_17, x_12);
lean_dec_ref(x_4);
lean_inc_ref(x_1);
x_19 = lp_mathlib_CliffordAlgebra_lift___redArg(x_1, x_16, x_18);
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg(x_1);
x_22 = lean_alloc_closure((void*)(lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg___lam__0), 3, 2);
lean_closure_set(x_22, 0, x_20);
lean_closure_set(x_22, 1, x_21);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_gradedAlgebra___redArg(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_gradedAlgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_gradedAlgebra(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_CliffordAlgebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_CliffordAlgebra_Grading(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_CliffordAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__0 = _init_lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__0);
lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__1 = _init_lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__1();
lean_mark_persistent(lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__1);
lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__2 = _init_lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__2();
lean_mark_persistent(lp_mathlib_CliffordAlgebra_GradedAlgebra_00_u03b9___redArg___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
