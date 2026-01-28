// Lean compiler output
// Module: Mathlib.LinearAlgebra.InvariantBasisNumber
// Imports: public import Init public import Mathlib.RingTheory.Ideal.Quotient.Basic public import Mathlib.RingTheory.Noetherian.Orzech public import Mathlib.RingTheory.OrzechProperty public import Mathlib.RingTheory.PrincipalIdealDomain public import Mathlib.LinearAlgebra.Finsupp.Pi
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__inducedEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__inducedEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ideal_Quotient_mk___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_Ideal_Quotient_mk___lam__0(x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__inducedEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lp_mathlib_LinearEquiv_symm___redArg(x_5);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_7, 1);
lean_dec(x_10);
lean_inc_ref(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map___boxed), 8, 7);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, x_1);
lean_closure_set(x_11, 2, lean_box(0));
lean_closure_set(x_11, 3, x_2);
lean_closure_set(x_11, 4, lean_box(0));
lean_closure_set(x_11, 5, x_4);
lean_closure_set(x_11, 6, x_6);
x_12 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map___boxed), 8, 7);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, x_1);
lean_closure_set(x_12, 2, lean_box(0));
lean_closure_set(x_12, 3, x_3);
lean_closure_set(x_12, 4, lean_box(0));
lean_closure_set(x_12, 5, x_4);
lean_closure_set(x_12, 6, x_9);
lean_ctor_set(x_7, 1, x_12);
lean_ctor_set(x_7, 0, x_11);
return x_7;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_13 = lean_ctor_get(x_7, 0);
lean_inc(x_13);
lean_dec(x_7);
lean_inc_ref(x_1);
x_14 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map___boxed), 8, 7);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, x_1);
lean_closure_set(x_14, 2, lean_box(0));
lean_closure_set(x_14, 3, x_2);
lean_closure_set(x_14, 4, lean_box(0));
lean_closure_set(x_14, 5, x_4);
lean_closure_set(x_14, 6, x_6);
x_15 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__induced__map___boxed), 8, 7);
lean_closure_set(x_15, 0, lean_box(0));
lean_closure_set(x_15, 1, x_1);
lean_closure_set(x_15, 2, lean_box(0));
lean_closure_set(x_15, 3, x_3);
lean_closure_set(x_15, 4, lean_box(0));
lean_closure_set(x_15, 5, x_4);
lean_closure_set(x_15, 6, x_13);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_14);
lean_ctor_set(x_16, 1, x_15);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__inducedEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib___private_Mathlib_LinearAlgebra_InvariantBasisNumber_0__inducedEquiv___redArg(x_2, x_4, x_6, x_7, x_8);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Noetherian_Orzech(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_OrzechProperty(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_PrincipalIdealDomain(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Pi(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_InvariantBasisNumber(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Noetherian_Orzech(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_OrzechProperty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_PrincipalIdealDomain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
