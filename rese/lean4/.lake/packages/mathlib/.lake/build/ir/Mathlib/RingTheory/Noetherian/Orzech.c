// Lean compiler output
// Module: Mathlib.RingTheory.Noetherian.Orzech
// Imports: public import Init public import Mathlib.Algebra.Module.Submodule.IterateMapComap public import Mathlib.Order.PartialSups public import Mathlib.RingTheory.Noetherian.Basic public import Mathlib.RingTheory.OrzechProperty
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
LEAN_EXPORT lean_object* lp_mathlib_IsNoetherian_equivPUnitOfProdInjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___boxed(lean_object*);
extern lean_object* lp_mathlib_PUnit_commRing;
LEAN_EXPORT lean_object* lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_ofSubsingleton___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
static lean_object* lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg(lean_object*);
static lean_object* lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__0;
static lean_object* _init_lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_PUnit_commRing;
x_2 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__0;
x_2 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__1;
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 0);
x_5 = lp_mathlib_LinearEquiv_ofSubsingleton___redArg(x_4, x_3);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsNoetherian_equivPUnitOfProdInjective(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg(x_7);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_IsNoetherian_equivPUnitOfProdInjective(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_10);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_IterateMapComap(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_PartialSups(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Noetherian_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_OrzechProperty(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Noetherian_Orzech(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_IterateMapComap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_PartialSups(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Noetherian_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_OrzechProperty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__0 = _init_lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__0();
lean_mark_persistent(lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__0);
lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__1 = _init_lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__1();
lean_mark_persistent(lp_mathlib_IsNoetherian_equivPUnitOfProdInjective___redArg___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
