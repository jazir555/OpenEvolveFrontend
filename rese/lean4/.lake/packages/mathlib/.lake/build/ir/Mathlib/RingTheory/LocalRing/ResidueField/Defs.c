// Lean compiler output
// Module: Mathlib.RingTheory.LocalRing.ResidueField.Defs
// Imports: public import Init public import Mathlib.RingTheory.Ideal.Quotient.Basic public import Mathlib.RingTheory.LocalRing.MaximalIdeal.Basic
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
lean_object* lp_mathlib_Submodule_Quotient_instInhabitedQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_instInhabitedResidueField(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_IsLocalRing_residue___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_residue___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ideal_Quotient_mk___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_residue(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_instCommRingResidueField(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_instInhabitedResidueField___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_instCommRingResidueField___redArg(lean_object*);
lean_object* lp_mathlib_Ideal_Quotient_ring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_instCommRingResidueField(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Ideal_Quotient_ring___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_instCommRingResidueField___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Ideal_Quotient_ring___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_instInhabitedResidueField___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_3 = lp_mathlib_Submodule_Quotient_instInhabitedQuotient___redArg(x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_instInhabitedResidueField(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_IsLocalRing_instInhabitedResidueField___redArg(x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_IsLocalRing_residue___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Ideal_Quotient_mk___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_residue(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_IsLocalRing_residue___closed__0;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLocalRing_residue___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_IsLocalRing_residue(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_LocalRing_MaximalIdeal_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_LocalRing_ResidueField_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_LocalRing_MaximalIdeal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_IsLocalRing_residue___closed__0 = _init_lp_mathlib_IsLocalRing_residue___closed__0();
lean_mark_persistent(lp_mathlib_IsLocalRing_residue___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
