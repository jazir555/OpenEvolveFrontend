// Lean compiler output
// Module: Mathlib.LinearAlgebra.TensorProduct.Prod
// Imports: public import Init public import Mathlib.LinearAlgebra.Prod public import Mathlib.LinearAlgebra.TensorProduct.Tower
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
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_toAddEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_AlgebraTensorModule_lift___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_comm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_inr___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_prod___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_addMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddEquiv_toLinearEquiv___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_prodMapLinear___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instSMul___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instAddMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_inl___redArg(lean_object*);
static lean_object* lp_mathlib_TensorProduct_prodRight___redArg___closed__0;
lean_object* lp_mathlib_TensorProduct_AlgebraTensorModule_lTensor___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_coprod___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_ofLinear___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_prodCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodLeft___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodLeft___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_TensorProduct_prodRight___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_prodMapLinear___lam__0), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodRight___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
lean_inc_ref(x_4);
lean_inc_ref(x_3);
x_8 = lp_mathlib_Prod_instAddMonoid___redArg(x_3, x_4);
lean_inc(x_7);
lean_inc(x_6);
x_9 = lp_mathlib_Prod_instSMul___redArg(x_6, x_7);
lean_inc_ref(x_9);
lean_inc(x_5);
lean_inc_ref(x_8);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_10 = lp_mathlib_TensorProduct_addMonoid___redArg(x_1, x_2, x_8, x_5, x_9);
lean_inc(x_6);
lean_inc(x_5);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_11 = lp_mathlib_TensorProduct_addMonoid___redArg(x_1, x_2, x_3, x_5, x_6);
lean_inc(x_7);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_12 = lp_mathlib_TensorProduct_addMonoid___redArg(x_1, x_2, x_4, x_5, x_7);
x_13 = lp_mathlib_Prod_instAddMonoid___redArg(x_11, x_12);
lean_inc(x_6);
lean_inc_n(x_5, 2);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_14 = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_3);
lean_closure_set(x_14, 3, x_5);
lean_closure_set(x_14, 4, x_6);
lean_closure_set(x_14, 5, x_5);
lean_inc(x_7);
lean_inc_n(x_5, 2);
lean_inc_ref(x_4);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_15 = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(x_15, 0, x_1);
lean_closure_set(x_15, 1, x_2);
lean_closure_set(x_15, 2, x_4);
lean_closure_set(x_15, 3, x_5);
lean_closure_set(x_15, 4, x_7);
lean_closure_set(x_15, 5, x_5);
x_16 = lp_mathlib_Prod_instSMul___redArg(x_14, x_15);
x_17 = lp_mathlib_TensorProduct_prodRight___redArg___closed__0;
x_18 = lp_mathlib_TensorProduct_mk(lean_box(0), x_1, lean_box(0), lean_box(0), x_2, x_3, x_5, x_6);
x_19 = lp_mathlib_TensorProduct_mk(lean_box(0), x_1, lean_box(0), lean_box(0), x_2, x_4, x_5, x_7);
x_20 = lp_mathlib_LinearMap_prod___redArg(x_18, x_19);
x_21 = lp_mathlib_LinearMap_comp___redArg(x_17, x_20);
lean_inc_ref(x_9);
lean_inc_ref(x_8);
lean_inc_ref(x_1);
x_22 = lp_mathlib_TensorProduct_AlgebraTensorModule_lift___redArg(x_1, x_8, x_9, x_13, x_16, x_21);
x_23 = lp_mathlib_LinearMap_inl___redArg(x_4);
lean_inc_ref(x_9);
lean_inc_ref(x_8);
lean_inc_ref(x_3);
lean_inc(x_5);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_24 = lp_mathlib_TensorProduct_AlgebraTensorModule_lTensor___redArg(x_1, x_2, x_5, x_3, x_6, x_8, x_9);
x_25 = lean_apply_1(x_24, x_23);
x_26 = lp_mathlib_LinearMap_inr___redArg(x_3);
lean_dec_ref(x_3);
x_27 = lp_mathlib_TensorProduct_AlgebraTensorModule_lTensor___redArg(x_1, x_2, x_5, x_4, x_7, x_8, x_9);
x_28 = lean_apply_1(x_27, x_26);
x_29 = lp_mathlib_LinearMap_coprod___redArg(x_10, x_25, x_28);
x_30 = lp_mathlib_LinearEquiv_ofLinear___redArg(x_22, x_29);
return x_30;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_TensorProduct_prodRight___redArg(x_6, x_8, x_9, x_10, x_12, x_15, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodRight___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_TensorProduct_prodRight(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
lean_dec(x_13);
lean_dec_ref(x_11);
lean_dec_ref(x_7);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodLeft___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_8 = lp_mathlib_Prod_instAddMonoid___redArg(x_2, x_3);
lean_inc(x_6);
lean_inc(x_5);
x_9 = lp_mathlib_Prod_instSMul___redArg(x_5, x_6);
lean_inc(x_7);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
x_10 = lp_mathlib_TensorProduct_comm___redArg(x_1, x_8, x_4, x_9, x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_7);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
x_11 = lp_mathlib_TensorProduct_prodRight___redArg(x_1, x_4, x_2, x_3, x_7, x_5, x_6);
x_12 = lp_mathlib_LinearEquiv_trans___redArg(x_10, x_11);
lean_inc(x_7);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
x_13 = lp_mathlib_TensorProduct_comm___redArg(x_1, x_4, x_2, x_7, x_5);
x_14 = lp_mathlib_TensorProduct_comm___redArg(x_1, x_4, x_3, x_7, x_6);
x_15 = lp_mathlib_LinearEquiv_prodCongr___redArg(x_13, x_14);
x_16 = lp_mathlib_LinearEquiv_trans___redArg(x_12, x_15);
x_17 = lp_mathlib_LinearEquiv_toAddEquiv___redArg(x_16);
x_18 = lp_mathlib_AddEquiv_toLinearEquiv___redArg(x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodLeft(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18) {
_start:
{
lean_object* x_19; 
x_19 = lp_mathlib_TensorProduct_prodLeft___redArg(x_6, x_8, x_9, x_10, x_12, x_15, x_16);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_prodLeft___boxed(lean_object** _args) {
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
x_19 = lp_mathlib_TensorProduct_prodLeft(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18);
lean_dec(x_17);
lean_dec(x_13);
lean_dec_ref(x_11);
lean_dec_ref(x_7);
return x_19;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Tower(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_TensorProduct_prodRight___redArg___closed__0 = _init_lp_mathlib_TensorProduct_prodRight___redArg___closed__0();
lean_mark_persistent(lp_mathlib_TensorProduct_prodRight___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
