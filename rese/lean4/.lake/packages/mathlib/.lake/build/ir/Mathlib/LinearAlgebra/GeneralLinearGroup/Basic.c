// Lean compiler output
// Module: Mathlib.LinearAlgebra.GeneralLinearGroup.Basic
// Imports: public import Init public import Mathlib.Algebra.Module.Equiv.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Units_mapEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___redArg(lean_object* x_1) {
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
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lp_mathlib_LinearEquiv_symm___redArg(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_dec(x_6);
lean_ctor_set(x_3, 1, x_5);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
else
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_3, 0);
lean_inc(x_7);
lean_dec(x_3);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_2);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_GeneralLinearGroup_toLinearEquiv___boxed), 6, 5);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, lean_box(0));
lean_closure_set(x_4, 2, x_1);
lean_closure_set(x_4, 3, x_2);
lean_closure_set(x_4, 4, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_GeneralLinearGroup_ofLinearEquiv___boxed), 6, 5);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, x_1);
lean_closure_set(x_5, 3, x_2);
lean_closure_set(x_5, 4, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LinearMap_GeneralLinearGroup_generalLinearEquiv___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(x_1, x_1);
x_3 = lp_mathlib_Units_mapEquiv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv___redArg(x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_LinearMap_GeneralLinearGroup_congrLinearEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_16;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
