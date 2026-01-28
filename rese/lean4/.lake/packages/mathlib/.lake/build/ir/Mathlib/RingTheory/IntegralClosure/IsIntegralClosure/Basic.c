// Lean compiler output
// Module: Mathlib.RingTheory.IntegralClosure.IsIntegralClosure.Basic
// Imports: public import Init public import Mathlib.Algebra.Polynomial.Roots public import Mathlib.RingTheory.FiniteType public import Mathlib.RingTheory.IntegralClosure.Algebra.Basic public import Mathlib.RingTheory.IntegralClosure.IsIntegralClosure.Defs public import Mathlib.RingTheory.Polynomial.IntegralNormalization public import Mathlib.RingTheory.Polynomial.ScaleRoots public import Mathlib.RingTheory.TensorProduct.MvPolynomial
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
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapIntegralClosure___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapIntegralClosure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapIntegralClosure___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_AlgEquiv_ofAlgHom___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Algebra_ofSubsemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapIntegralClosure___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapIntegralClosure___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_restrictDomain___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapIntegralClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_instFunLike___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapIntegralClosure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapIntegralClosure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Algebra_id___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapIntegralClosure___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lp_mathlib_Algebra_id___redArg(x_3);
x_5 = lp_mathlib_Algebra_ofSubsemiring___redArg(x_4);
x_6 = lp_mathlib_AlgHom_restrictDomain___redArg(x_5, x_2);
x_7 = lp_mathlib_AlgHom_codRestrict___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapIntegralClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_AlgHom_mapIntegralClosure___redArg(x_5, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapIntegralClosure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_AlgHom_mapIntegralClosure(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapIntegralClosure___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AlgHom_mapIntegralClosure___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapIntegralClosure___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_2, 0);
x_9 = lean_ctor_get(x_3, 0);
x_10 = lp_mathlib_AlgEquiv_instFunLike___redArg(x_7, x_8, x_9, x_4, x_5);
lean_inc_ref(x_6);
x_11 = lean_apply_1(x_10, x_6);
x_12 = lp_mathlib_AlgHom_mapIntegralClosure___redArg(x_2, x_11);
x_13 = lp_mathlib_Equiv_symm___redArg(x_6);
x_14 = lp_mathlib_AlgEquiv_instFunLike___redArg(x_7, x_9, x_8, x_5, x_4);
x_15 = lean_apply_1(x_14, x_13);
x_16 = lp_mathlib_AlgHom_mapIntegralClosure___redArg(x_3, x_15);
x_17 = lp_mathlib_AlgEquiv_ofAlgHom___redArg(x_12, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapIntegralClosure(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_AlgEquiv_mapIntegralClosure___redArg(x_4, x_5, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapIntegralClosure___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_AlgEquiv_mapIntegralClosure(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapIntegralClosure___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_AlgEquiv_mapIntegralClosure___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_FiniteType(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_IntegralClosure_Algebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_IntegralClosure_IsIntegralClosure_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Polynomial_IntegralNormalization(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Polynomial_ScaleRoots(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_MvPolynomial(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_IntegralClosure_IsIntegralClosure_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_FiniteType(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_IntegralClosure_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_IntegralClosure_IsIntegralClosure_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Polynomial_IntegralNormalization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Polynomial_ScaleRoots(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TensorProduct_MvPolynomial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
