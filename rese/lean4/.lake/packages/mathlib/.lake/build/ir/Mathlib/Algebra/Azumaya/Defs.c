// Lean compiler output
// Module: Mathlib.Algebra.Azumaya.Defs
// Imports: public import Init public import Mathlib.Algebra.Module.Projective public import Mathlib.RingTheory.Finiteness.Defs public import Mathlib.RingTheory.TensorProduct.Basic
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
lean_object* lp_mathlib_TensorProduct_Algebra_module___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleTensorProductMop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_Algebra_TensorProduct_instSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRight___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleTensorProductMop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toOppositeModule___redArg(lean_object*);
lean_object* lp_mathlib_Algebra_lsmul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleTensorProductMop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_6 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_4);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
x_9 = lean_ctor_get(x_5, 0);
lean_inc(x_9);
lean_inc_ref(x_4);
x_10 = lp_mathlib_MulOpposite_instSemiring___redArg(x_4);
lean_inc_ref(x_4);
x_11 = lp_mathlib_Semiring_toModule___redArg(x_4);
x_12 = lp_mathlib_Semiring_toOppositeModule___redArg(x_4);
x_13 = lp_mathlib_MulOpposite_instAlgebra___redArg(x_5);
x_14 = lp_mathlib_TensorProduct_Algebra_module___redArg(x_3, x_8, x_9, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instModuleTensorProductMop___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_3, 0);
lean_inc(x_7);
lean_inc_ref(x_2);
x_8 = lp_mathlib_MulOpposite_instSemiring___redArg(x_2);
lean_inc_ref(x_2);
x_9 = lp_mathlib_Semiring_toModule___redArg(x_2);
x_10 = lp_mathlib_Semiring_toOppositeModule___redArg(x_2);
x_11 = lp_mathlib_MulOpposite_instAlgebra___redArg(x_3);
x_12 = lp_mathlib_TensorProduct_Algebra_module___redArg(x_1, x_6, x_7, x_8, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRight___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_inc_ref(x_2);
x_4 = lp_mathlib_MulOpposite_instSemiring___redArg(x_2);
lean_inc_ref(x_3);
x_5 = lp_mathlib_MulOpposite_instAlgebra___redArg(x_3);
lean_inc_ref(x_5);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_6 = lp_mathlib_Algebra_TensorProduct_instSemiring___redArg(x_1, x_2, x_3, x_4, x_5);
x_7 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_ctor_get(x_3, 0);
lean_inc(x_10);
lean_dec_ref(x_3);
lean_inc_ref(x_2);
x_11 = lp_mathlib_Semiring_toModule___redArg(x_2);
x_12 = lp_mathlib_Semiring_toOppositeModule___redArg(x_2);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc_ref(x_1);
x_13 = lp_mathlib_TensorProduct_Algebra_module___redArg(x_1, x_9, x_10, x_4, x_11, x_12, x_5);
x_14 = lp_mathlib_Algebra_lsmul___redArg(x_6, x_1, x_9, x_13, x_10);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mulLeftRight(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AlgHom_mulLeftRight___redArg(x_3, x_4, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Projective(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Azumaya_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Projective(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Finiteness_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TensorProduct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
