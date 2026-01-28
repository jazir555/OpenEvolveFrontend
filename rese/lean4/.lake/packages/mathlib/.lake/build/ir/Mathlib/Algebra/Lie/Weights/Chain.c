// Lean compiler output
// Module: Mathlib.Algebra.Lie.Weights.Chain
// Imports: public import Init public import Mathlib.Algebra.DirectSum.LinearMap public import Mathlib.Algebra.Lie.Weights.Cartan public import Mathlib.Algebra.Order.Group.Pointwise.Interval public import Mathlib.RingTheory.Finiteness.Nilpotent public import Mathlib.Data.Int.Interval public import Mathlib.Order.Filter.Cofinite
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
LEAN_EXPORT lean_object* lp_mathlib_LieModule_genWeightSpaceChain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_LieModule_genWeightSpaceChain___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LieModule_genWeightSpaceChain___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LieSubmodule_instInfSet___lam__0(lean_object*);
static lean_object* _init_lp_mathlib_LieModule_genWeightSpaceChain___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_LieSubmodule_instInfSet___lam__0(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieModule_genWeightSpaceChain(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_LieModule_genWeightSpaceChain___closed__0;
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieModule_genWeightSpaceChain___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_LieModule_genWeightSpaceChain(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_16;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_LinearMap(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Weights_Cartan(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Pointwise_Interval(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Nilpotent(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Interval(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Cofinite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Weights_Chain(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_LinearMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Lie_Weights_Cartan(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Pointwise_Interval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Finiteness_Nilpotent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Interval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Cofinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LieModule_genWeightSpaceChain___closed__0 = _init_lp_mathlib_LieModule_genWeightSpaceChain___closed__0();
lean_mark_persistent(lp_mathlib_LieModule_genWeightSpaceChain___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
