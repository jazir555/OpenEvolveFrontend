// Lean compiler output
// Module: Mathlib.Algebra.Category.ModuleCat.Biproducts
// Imports: public import Init public import Mathlib.Algebra.Group.Pi.Lemmas public import Mathlib.CategoryTheory.Limits.Shapes.BinaryBiproducts public import Mathlib.Algebra.Category.ModuleCat.Abelian public import Mathlib.Algebra.Homology.ShortComplex.ModuleCat
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
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg___lam__2(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__1(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_subNegMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_prod___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_fst___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_productLimitCone(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_lift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instSMul___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_snd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = 0;
x_5 = lean_box(x_4);
lean_inc(x_3);
x_6 = lean_apply_1(x_3, x_5);
x_7 = 1;
x_8 = lean_box(x_7);
x_9 = lean_apply_1(x_3, x_8);
x_10 = lp_mathlib_LinearMap_prod___redArg(x_6, x_9);
x_11 = lean_apply_1(x_10, x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__1(uint8_t x_1, lean_object* x_2) {
_start:
{
if (x_1 == 0)
{
lean_object* x_3; 
x_3 = lp_mathlib_LinearMap_fst___lam__0(x_2);
return x_3;
}
else
{
lean_object* x_4; 
x_4 = lp_mathlib_LinearMap_snd___lam__0(x_2);
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lean_unbox(x_1);
x_4 = lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__1(x_3, x_2);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_2);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__0), 2, 0);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__1___boxed), 2, 0);
x_11 = lp_mathlib_Prod_subNegMonoid___redArg(x_5, x_7);
x_12 = lp_mathlib_Prod_instSMul___redArg(x_6, x_8);
lean_ctor_set(x_2, 1, x_12);
lean_ctor_set(x_2, 0, x_11);
lean_ctor_set(x_1, 1, x_10);
lean_ctor_set(x_1, 0, x_2);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_1);
lean_ctor_set(x_13, 1, x_9);
return x_13;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_14 = lean_ctor_get(x_1, 0);
x_15 = lean_ctor_get(x_1, 1);
x_16 = lean_ctor_get(x_2, 0);
x_17 = lean_ctor_get(x_2, 1);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_2);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__0), 2, 0);
x_19 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__1___boxed), 2, 0);
x_20 = lp_mathlib_Prod_subNegMonoid___redArg(x_14, x_16);
x_21 = lp_mathlib_Prod_instSMul___redArg(x_15, x_17);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_20);
lean_ctor_set(x_22, 1, x_21);
lean_ctor_set(x_1, 1, x_19);
lean_ctor_set(x_1, 0, x_22);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_1);
lean_ctor_set(x_23, 1, x_18);
return x_23;
}
}
else
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_24 = lean_ctor_get(x_1, 0);
x_25 = lean_ctor_get(x_1, 1);
lean_inc(x_25);
lean_inc(x_24);
lean_dec(x_1);
x_26 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_26);
x_27 = lean_ctor_get(x_2, 1);
lean_inc(x_27);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 x_28 = x_2;
} else {
 lean_dec_ref(x_2);
 x_28 = lean_box(0);
}
x_29 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__0), 2, 0);
x_30 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_binaryProductLimitCone___redArg___lam__1___boxed), 2, 0);
x_31 = lp_mathlib_Prod_subNegMonoid___redArg(x_24, x_26);
x_32 = lp_mathlib_Prod_instSMul___redArg(x_25, x_27);
if (lean_is_scalar(x_28)) {
 x_33 = lean_alloc_ctor(0, 2, 0);
} else {
 x_33 = x_28;
}
lean_ctor_set(x_33, 0, x_31);
lean_ctor_set(x_33, 1, x_32);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_30);
x_35 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_35, 0, x_34);
lean_ctor_set(x_35, 1, x_29);
return x_35;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ModuleCat_binaryProductLimitCone___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_binaryProductLimitCone___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ModuleCat_binaryProductLimitCone(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_lift___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_3, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_lift___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_HasLimit_lift___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_lift(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_HasLimit_lift___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_lift___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ModuleCat_HasLimit_lift(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
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
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg___lam__0), 4, 1);
lean_closure_set(x_3, 0, x_2);
lean_inc_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg___lam__1), 2, 1);
lean_closure_set(x_4, 0, x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg___lam__2), 2, 0);
x_6 = lp_mathlib_Pi_addCommGroup___redArg(x_4);
x_7 = lp_mathlib_Pi_module___redArg(x_3);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_5);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ModuleCat_HasLimit_lift___boxed), 5, 4);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, x_1);
lean_closure_set(x_10, 2, lean_box(0));
lean_closure_set(x_10, 3, x_2);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ModuleCat_HasLimit_productLimitCone(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ModuleCat_HasLimit_productLimitCone___redArg(x_2, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryBiproducts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Abelian(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_ShortComplex_ModuleCat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Biproducts(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_BinaryBiproducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_ModuleCat_Abelian(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_ShortComplex_ModuleCat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
