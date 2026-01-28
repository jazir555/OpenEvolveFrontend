// Lean compiler output
// Module: Mathlib.RingTheory.AdjoinRoot
// Imports: public import Init public import Mathlib.Algebra.Algebra.Defs public import Mathlib.Algebra.Polynomial.FieldDivision public import Mathlib.FieldTheory.Minpoly.Basic public import Mathlib.RingTheory.Adjoin.Basic public import Mathlib.RingTheory.FinitePresentation public import Mathlib.RingTheory.FiniteType public import Mathlib.RingTheory.Ideal.Quotient.Noetherian public import Mathlib.RingTheory.PowerBasis public import Mathlib.RingTheory.PrincipalIdealDomain public import Mathlib.RingTheory.Polynomial.Quotient
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
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_mk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_lift___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ideal_Quotient_mk___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddCon_lift___redArg(lean_object*);
static lean_object* lp_mathlib_AdjoinRoot_mk___closed__0;
static lean_object* lp_mathlib_AdjoinRoot_mk_u2090___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftAlgHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ideal_Quotient_mk_u2090___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_mk_u2090(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Polynomial_eval_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_mk(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_mk_u2090___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_AdjoinRoot_mk___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Ideal_Quotient_mk___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_mk(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AdjoinRoot_mk___closed__0;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_mk___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AdjoinRoot_mk(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_AdjoinRoot_mk_u2090___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Ideal_Quotient_mk_u2090___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_mk_u2090(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AdjoinRoot_mk_u2090___closed__0;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_mk_u2090___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AdjoinRoot_mk_u2090(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_lift___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_2);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Polynomial_eval_u2082___boxed), 7, 6);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, x_5);
lean_closure_set(x_7, 3, x_6);
lean_closure_set(x_7, 4, x_3);
lean_closure_set(x_7, 5, x_4);
x_8 = lp_mathlib_AddCon_lift___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_lift(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_AdjoinRoot_lift___redArg(x_3, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_lift___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_AdjoinRoot_lift(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftAlgHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_AdjoinRoot_lift___redArg(x_4, x_6, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftAlgHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AdjoinRoot_lift___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftAlgHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_AdjoinRoot_liftAlgHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lp_mathlib_AdjoinRoot_lift___redArg(x_1, x_2, x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_AdjoinRoot_liftHom___redArg(x_3, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AdjoinRoot_liftHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_AdjoinRoot_liftHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_4);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_FieldDivision(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_FieldTheory_Minpoly_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Adjoin_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_FinitePresentation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_FiniteType(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Noetherian(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_PowerBasis(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_PrincipalIdealDomain(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Polynomial_Quotient(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_AdjoinRoot(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_FieldDivision(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_FieldTheory_Minpoly_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Adjoin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_FinitePresentation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_FiniteType(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Noetherian(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_PowerBasis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_PrincipalIdealDomain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Polynomial_Quotient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_AdjoinRoot_mk___closed__0 = _init_lp_mathlib_AdjoinRoot_mk___closed__0();
lean_mark_persistent(lp_mathlib_AdjoinRoot_mk___closed__0);
lp_mathlib_AdjoinRoot_mk_u2090___closed__0 = _init_lp_mathlib_AdjoinRoot_mk_u2090___closed__0();
lean_mark_persistent(lp_mathlib_AdjoinRoot_mk_u2090___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
