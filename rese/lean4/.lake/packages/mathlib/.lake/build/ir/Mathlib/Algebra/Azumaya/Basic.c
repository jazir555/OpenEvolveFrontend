// Lean compiler output
// Module: Mathlib.Algebra.Azumaya.Basic
// Imports: public import Init public import Mathlib.Algebra.Azumaya.Defs public import Mathlib.Algebra.Central.End public import Mathlib.Algebra.Central.TensorProduct public import Mathlib.LinearAlgebra.Matrix.ToLin public import Mathlib.RingTheory.Finiteness.Basic public import Mathlib.GroupTheory.GroupAction.Hom public import Mathlib.RingTheory.TensorProduct.Maps
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
lean_object* lp_mathlib_RingEquiv_moduleEndSelf___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAzumaya_tensorEquivEnd___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAzumaya_tensorEquivEnd(lean_object*, lean_object*);
lean_object* lp_mathlib_Algebra_TensorProduct_lid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Algebra_id___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAzumaya_tensorEquivEnd(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lp_mathlib_Algebra_id___redArg(x_2);
lean_inc_ref(x_2);
x_4 = lp_mathlib_MulOpposite_instSemiring___redArg(x_2);
x_5 = lp_mathlib_MulOpposite_instAlgebra___redArg(x_3);
lean_inc_ref(x_2);
x_6 = lp_mathlib_Algebra_TensorProduct_lid___redArg(x_2, x_4, x_5);
lean_dec_ref(x_4);
x_7 = lp_mathlib_RingEquiv_moduleEndSelf___redArg(x_2);
x_8 = lp_mathlib_Equiv_trans___redArg(x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAzumaya_tensorEquivEnd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_Algebra_id___redArg(x_1);
lean_inc_ref(x_1);
x_3 = lp_mathlib_MulOpposite_instSemiring___redArg(x_1);
x_4 = lp_mathlib_MulOpposite_instAlgebra___redArg(x_2);
lean_inc_ref(x_1);
x_5 = lp_mathlib_Algebra_TensorProduct_lid___redArg(x_1, x_3, x_4);
lean_dec_ref(x_3);
x_6 = lp_mathlib_RingEquiv_moduleEndSelf___redArg(x_1);
x_7 = lp_mathlib_Equiv_trans___redArg(x_5, x_6);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Azumaya_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Central_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Central_TensorProduct(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ToLin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_Maps(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Azumaya_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Azumaya_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Central_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Central_TensorProduct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ToLin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TensorProduct_Maps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
