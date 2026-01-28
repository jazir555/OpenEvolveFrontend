// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Products
// Imports: public import Init public import Mathlib.Algebra.Category.ModuleCat.Basic public import Mathlib.LinearAlgebra.Pi public import Mathlib.Algebra.DirectSum.Module public import Mathlib.Tactic.CategoryTheory.Elementwise
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
lean_object* lp_mathlib_CategoryTheory_Limits_Fan_mk___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone___redArg(lean_object*);
lean_object* lp_mathlib_Pi_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCoconeIsColimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Limits_Cofan_mk___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCoconeIsColimit___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCocone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCocone___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCocone___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCoconeIsColimit___redArg___lam__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_toModule___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productConeIsLimit___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCoconeIsColimit___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productConeIsLimit___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCocone___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productConeIsLimit(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCocone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_pi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productConeIsLimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCoconeIsColimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_module___redArg(lean_object*);
lean_object* lp_mathlib_Pi_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_1, x_2);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, x_3, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone___redArg___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_productCone___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_productCone___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_productCone___redArg___lam__2), 2, 0);
x_5 = lp_mathlib_Pi_addCommGroup___redArg(x_3);
x_6 = lp_mathlib_Pi_module___redArg(x_2);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
x_8 = lp_mathlib_CategoryTheory_Limits_Fan_mk___redArg(x_7, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ModuleCat_productCone___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productCone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ModuleCat_productCone(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productConeIsLimit___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productConeIsLimit___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_productConeIsLimit___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lp_mathlib_LinearMap_pi___redArg(x_3);
x_5 = lean_apply_1(x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productConeIsLimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_productConeIsLimit___lam__1), 2, 0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_productConeIsLimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ModuleCat_productConeIsLimit(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCocone___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCocone___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_DFinsupp_lsingle___redArg(x_1, x_2, x_3);
x_6 = lean_apply_1(x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCocone___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_productCone___redArg___lam__0), 4, 1);
lean_closure_set(x_3, 0, x_1);
lean_inc_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_coproductCocone___redArg___lam__1), 2, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_productCone___redArg___lam__1), 2, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_coproductCocone___redArg___lam__2), 4, 2);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_2);
x_7 = lp_mathlib_DFinsupp_addCommGroup___redArg(x_5);
x_8 = lp_mathlib_DFinsupp_module___redArg(x_3);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
x_10 = lp_mathlib_CategoryTheory_Limits_Cofan_mk___redArg(x_9, x_6);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCocone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_coproductCocone___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCocone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_coproductCocone(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCoconeIsColimit___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCoconeIsColimit___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_3, 1);
lean_inc(x_7);
lean_dec_ref(x_3);
x_8 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_6);
x_9 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_coproductCoconeIsColimit___redArg___lam__1), 3, 1);
lean_closure_set(x_9, 0, x_7);
x_10 = lp_mathlib_DirectSum_toModule___redArg(x_1, x_2, x_8, x_9);
x_11 = lean_apply_1(x_10, x_4);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCoconeIsColimit___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_coproductCocone___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_coproductCoconeIsColimit___redArg___lam__0), 4, 2);
lean_closure_set(x_4, 0, x_3);
lean_closure_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCoconeIsColimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_coproductCoconeIsColimit___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_coproductCoconeIsColimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_coproductCoconeIsColimit(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CategoryTheory_Elementwise(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Products(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CategoryTheory_Elementwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
