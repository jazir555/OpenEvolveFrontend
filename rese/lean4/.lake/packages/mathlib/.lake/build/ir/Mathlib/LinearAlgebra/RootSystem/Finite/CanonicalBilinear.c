// Lean compiler output
// Module: Mathlib.LinearAlgebra.RootSystem.Finite.CanonicalBilinear
// Imports: public import Init public import Mathlib.Algebra.Ring.SumsOfSquares public import Mathlib.LinearAlgebra.RootSystem.RootPositive
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
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_RootForm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CorootForm___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_Polarization___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_RootForm___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CoPolarization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_smulRight___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CoPolarization___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CorootForm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CoPolarization___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_posRootForm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_RootForm___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_Polarization___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_posRootForm___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RootPairing_flip___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_Polarization___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_RootForm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CorootForm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_posRootForm___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_Polarization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_toSpanSingleton___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_flip___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_Polarization___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_1, 2);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_1(x_6, x_3);
lean_inc(x_7);
x_8 = lp_mathlib_LinearMap_toSpanSingleton___redArg(x_2, x_7);
x_9 = lp_mathlib_LinearMap_flip___redArg(x_5);
x_10 = lean_apply_1(x_9, x_7);
x_11 = lp_mathlib_LinearMap_comp___redArg(x_8, x_10);
x_12 = lean_apply_1(x_11, x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_Polarization___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_RootPairing_Polarization___redArg___lam__0), 4, 2);
lean_closure_set(x_6, 0, x_3);
lean_closure_set(x_6, 1, x_2);
x_7 = lp_mathlib_LinearMap_addCommMonoid___redArg(x_5);
x_8 = lp_mathlib_Finset_sum___redArg(x_7, x_4, x_6);
lean_dec_ref(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_Polarization(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_RootPairing_Polarization___redArg(x_8, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_Polarization___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_RootPairing_Polarization(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CoPolarization___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_RootPairing_flip___redArg(x_3);
x_6 = lp_mathlib_RootPairing_Polarization___redArg(x_1, x_2, x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CoPolarization(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_RootPairing_CoPolarization___redArg(x_6, x_7, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CoPolarization___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_RootPairing_CoPolarization(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_RootForm___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 2);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_apply_1(x_7, x_3);
x_9 = lp_mathlib_LinearMap_flip___redArg(x_6);
x_10 = lean_apply_1(x_9, x_8);
lean_inc(x_10);
x_11 = lp_mathlib_LinearMap_smulRight___redArg(x_2, x_10, x_10);
x_12 = lean_apply_2(x_11, x_4, x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_RootForm___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = lp_mathlib_Semiring_toModule___redArg(x_4);
x_9 = lp_mathlib_LinearMap_addCommMonoid___redArg(x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_10, 0, x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_RootPairing_RootForm___redArg___lam__0), 5, 2);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_10);
x_12 = lp_mathlib_LinearMap_addCommMonoid___redArg(x_9);
x_13 = lp_mathlib_Finset_sum___redArg(x_12, x_3, x_11);
lean_dec_ref(x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_RootForm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_RootPairing_RootForm___redArg(x_5, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_RootForm___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_RootPairing_RootForm(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CorootForm___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_RootPairing_flip___redArg(x_2);
x_5 = lp_mathlib_RootPairing_RootForm___redArg(x_1, x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CorootForm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_RootPairing_CorootForm___redArg(x_5, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_CorootForm___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_RootPairing_CorootForm(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_posRootForm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18) {
_start:
{
lean_object* x_19; 
x_19 = lp_mathlib_RootPairing_RootForm___redArg(x_5, x_10, x_18);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_posRootForm___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_RootPairing_RootForm___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_posRootForm___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
_start:
{
lean_object* x_19; 
x_19 = lp_mathlib_RootPairing_posRootForm(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18);
lean_dec_ref(x_15);
lean_dec_ref(x_13);
lean_dec_ref(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_19;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_SumsOfSquares(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_RootPositive(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_Finite_CanonicalBilinear(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_SumsOfSquares(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_RootPositive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
