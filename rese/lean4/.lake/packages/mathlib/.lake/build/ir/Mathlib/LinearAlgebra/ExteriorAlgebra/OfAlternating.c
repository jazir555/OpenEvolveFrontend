// Lean compiler output
// Module: Mathlib.LinearAlgebra.ExteriorAlgebra.OfAlternating
// Imports: public import Init public import Mathlib.LinearAlgebra.CliffordAlgebra.Fold public import Mathlib.LinearAlgebra.ExteriorAlgebra.Basic
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
lean_object* lp_mathlib_CliffordAlgebra_foldl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_module___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_compMultilinearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternatingEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlternatingMap_instModuleAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
static lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___closed__0;
lean_object* lp_mathlib_ExteriorAlgebra_00_u03b9Multi___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_eval(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlternatingMap_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlternatingMap_instModuleAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_AlternatingMap_instAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_mk_u2082_x27_u209b_u2097___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlternatingMap_curryLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_AlternatingMap_constLinearEquivOfIsEmpty___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlternatingMap_instModuleAddCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternatingEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternatingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternatingEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_compr_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlternatingMap_instModuleAddCommGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_AlternatingMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_10, 0, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlternatingMap_instModuleAddCommGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_AlternatingMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlternatingMap_instModuleAddCommGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_AlternatingMap_instModuleAddCommGroup(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AlternatingMap_instSMul___redArg___lam__0(x_1, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__1(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_unsigned_to_nat(1u);
x_6 = lean_nat_add(x_3, x_5);
x_7 = lean_apply_1(x_2, x_6);
x_8 = lp_mathlib_AlternatingMap_curryLeft___redArg(x_3, x_7);
x_9 = lean_apply_2(x_8, x_1, x_4);
return x_9;
}
}
static lean_object* _init_lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_2);
lean_inc_ref(x_3);
x_8 = lp_mathlib_AlternatingMap_instAddCommGroup___redArg(x_3);
x_9 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_3);
x_10 = lean_alloc_closure((void*)(lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_10, 0, x_8);
x_11 = lp_mathlib_Pi_addCommGroup___redArg(x_10);
lean_inc(x_5);
lean_inc_ref(x_6);
x_12 = lp_mathlib_AlternatingMap_constLinearEquivOfIsEmpty___redArg(x_6, x_7, x_9, x_4, x_5);
x_13 = lp_mathlib_LinearEquiv_symm___redArg(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lean_alloc_closure((void*)(lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__1___boxed), 5, 1);
lean_closure_set(x_15, 0, x_5);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___lam__2), 4, 0);
x_17 = lp_mathlib_Pi_module___redArg(x_15);
x_18 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_mk_u2082_x27_u209b_u2097___redArg___lam__0), 3, 1);
lean_closure_set(x_18, 0, x_16);
x_19 = lp_mathlib_CliffordAlgebra_foldl___redArg(x_1, x_11, x_17, x_18);
x_20 = lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___closed__0;
x_21 = lp_mathlib_LinearMap_comp___redArg(x_14, x_20);
x_22 = lp_mathlib_LinearMap_compr_u2082___redArg(x_19, x_21);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternating(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_ExteriorAlgebra_liftAlternating___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternatingEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lp_mathlib_ExteriorAlgebra_00_u03b9Multi___redArg(x_1, x_3);
x_6 = lp_mathlib_LinearMap_compMultilinearMap___redArg(x_2, x_5);
x_7 = lean_apply_1(x_6, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternatingEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_ExteriorAlgebra_liftAlternating___redArg(x_1, x_2, x_3, x_4, x_5);
x_9 = lean_apply_2(x_8, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternatingEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_ExteriorAlgebra_liftAlternatingEquiv___redArg___lam__0), 4, 1);
lean_closure_set(x_6, 0, x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_ExteriorAlgebra_liftAlternatingEquiv___redArg___lam__1), 7, 5);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_2);
lean_closure_set(x_7, 2, x_3);
lean_closure_set(x_7, 3, x_4);
lean_closure_set(x_7, 4, x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExteriorAlgebra_liftAlternatingEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_ExteriorAlgebra_liftAlternatingEquiv___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_CliffordAlgebra_Fold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_ExteriorAlgebra_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_ExteriorAlgebra_OfAlternating(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_CliffordAlgebra_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_ExteriorAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___closed__0 = _init_lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___closed__0();
lean_mark_persistent(lp_mathlib_ExteriorAlgebra_liftAlternating___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
