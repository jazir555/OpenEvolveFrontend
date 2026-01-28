// Lean compiler output
// Module: Mathlib.RingTheory.MatrixAlgebra
// Imports: public import Init public import Mathlib.Algebra.Star.StarAlgHom public import Mathlib.Data.Matrix.Basis public import Mathlib.Data.Matrix.Composition public import Mathlib.LinearAlgebra.Matrix.Kronecker public import Mathlib.RingTheory.TensorProduct.Maps
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
lean_object* lp_mathlib_LinearMap_mapMatrix___redArg(lean_object*);
static lean_object* lp_mathlib_MatrixEquivTensor_toFunLinear___redArg___closed__0;
lean_object* lp_mathlib_TensorProduct_AlgebraTensorModule_lift___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunBilinear___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_equiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulAlgEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_ofLinearMap___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_liftAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_singleAddMonoidHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_invFun___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_toLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_invFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulStarAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_module___redArg(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqProd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunAlgHom___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_matrixEquivTensor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_kroneckerLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_equiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_invFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerStarAlgEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_tmul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_invFun___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_addMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_kroneckerTMulBilinear___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_kroneckerLinearEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulStarAlgEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunLinear___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulStarAlgEquiv___boxed(lean_object**);
lean_object* lp_mathlib_LinearEquiv_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerAlgEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_lid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulAlgEquiv___boxed(lean_object**);
lean_object* lp_mathlib_Algebra_TensorProduct_lid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerStarAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_liftLinear___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_invFun___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunBilinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulAlgEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_equiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_ofLinear___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulStarAlgEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerStarAlgEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_matrixEquivTensor___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunLinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_AlgebraTensorModule_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Algebra_id___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_compl_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_single(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Algebra_lsmul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = l_instDecidableEqProd___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__0(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_16 = lean_ctor_get(x_13, 0);
lean_inc(x_16);
x_17 = lean_ctor_get(x_13, 1);
lean_inc(x_17);
lean_dec_ref(x_13);
x_18 = lean_ctor_get(x_14, 0);
lean_inc(x_18);
x_19 = lean_ctor_get(x_14, 1);
lean_inc(x_19);
lean_dec_ref(x_14);
x_20 = lp_mathlib_Matrix_singleAddMonoidHom___redArg(x_1, x_2, x_3, x_16, x_18);
x_21 = lp_mathlib_Matrix_singleAddMonoidHom___redArg(x_4, x_5, x_6, x_17, x_19);
x_22 = lp_mathlib_TensorProduct_AlgebraTensorModule_map___redArg(x_7, x_6, x_8, x_9, x_10, x_11, x_12, x_20, x_21);
x_23 = lean_apply_1(x_22, x_15);
return x_23;
}
}
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec_ref(x_3);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
lean_inc_ref(x_14);
lean_inc_ref(x_12);
x_16 = lean_alloc_closure((void*)(lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_16, 0, x_12);
lean_closure_set(x_16, 1, x_14);
lean_inc_ref(x_15);
lean_inc_ref(x_13);
x_17 = lean_alloc_closure((void*)(lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_17, 0, x_13);
lean_closure_set(x_17, 1, x_15);
lean_inc_ref(x_3);
x_18 = lp_mathlib_Matrix_addCommMonoid___redArg(x_3);
lean_inc_ref(x_4);
x_19 = lp_mathlib_Matrix_addCommMonoid___redArg(x_4);
lean_inc(x_5);
x_20 = lp_mathlib_Matrix_module___redArg(x_5);
lean_inc(x_7);
x_21 = lp_mathlib_Matrix_module___redArg(x_7);
lean_inc(x_21);
lean_inc(x_20);
lean_inc_ref(x_19);
lean_inc_ref(x_18);
lean_inc_ref(x_1);
x_22 = lp_mathlib_TensorProduct_addMonoid___redArg(x_1, x_18, x_19, x_20, x_21);
lean_inc(x_7);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_23 = lp_mathlib_TensorProduct_addMonoid___redArg(x_1, x_3, x_4, x_5, x_7);
lean_inc_ref(x_23);
x_24 = lp_mathlib_Matrix_addCommMonoid___redArg(x_23);
lean_inc(x_6);
x_25 = lp_mathlib_Matrix_module___redArg(x_6);
lean_inc(x_21);
lean_inc(x_20);
lean_inc_ref(x_19);
lean_inc_ref(x_18);
lean_inc_ref(x_1);
x_26 = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(x_26, 0, x_1);
lean_closure_set(x_26, 1, x_18);
lean_closure_set(x_26, 2, x_19);
lean_closure_set(x_26, 3, x_20);
lean_closure_set(x_26, 4, x_21);
lean_closure_set(x_26, 5, x_25);
lean_inc(x_7);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_27 = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(x_27, 0, x_1);
lean_closure_set(x_27, 1, x_3);
lean_closure_set(x_27, 2, x_4);
lean_closure_set(x_27, 3, x_5);
lean_closure_set(x_27, 4, x_7);
lean_closure_set(x_27, 5, x_6);
x_28 = lp_mathlib_Multiset_product___redArg(x_8, x_10);
x_29 = lp_mathlib_Multiset_product___redArg(x_9, x_11);
x_30 = lp_mathlib_Matrix_liftLinear___redArg(x_16, x_17, x_28, x_29, x_2, x_23, x_22, x_27, x_26);
lean_dec_ref(x_26);
lean_dec_ref(x_27);
x_31 = lean_ctor_get(x_30, 0);
lean_inc(x_31);
lean_dec_ref(x_30);
lean_inc(x_21);
lean_inc_ref(x_19);
lean_inc(x_7);
lean_inc_ref(x_1);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
x_32 = lean_alloc_closure((void*)(lp_mathlib_kroneckerTMulLinearEquiv___redArg___lam__2___boxed), 15, 12);
lean_closure_set(x_32, 0, x_12);
lean_closure_set(x_32, 1, x_13);
lean_closure_set(x_32, 2, x_3);
lean_closure_set(x_32, 3, x_14);
lean_closure_set(x_32, 4, x_15);
lean_closure_set(x_32, 5, x_4);
lean_closure_set(x_32, 6, x_1);
lean_closure_set(x_32, 7, x_7);
lean_closure_set(x_32, 8, x_18);
lean_closure_set(x_32, 9, x_20);
lean_closure_set(x_32, 10, x_19);
lean_closure_set(x_32, 11, x_21);
lean_inc(x_7);
lean_inc_n(x_5, 2);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_33 = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(x_33, 0, x_1);
lean_closure_set(x_33, 1, x_3);
lean_closure_set(x_33, 2, x_4);
lean_closure_set(x_33, 3, x_5);
lean_closure_set(x_33, 4, x_7);
lean_closure_set(x_33, 5, x_5);
x_34 = lp_mathlib_Matrix_module___redArg(x_33);
x_35 = lp_mathlib_Matrix_kroneckerTMulBilinear___redArg(x_1, x_3, x_4, x_5, x_7);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_36 = lp_mathlib_TensorProduct_AlgebraTensorModule_lift___redArg(x_1, x_19, x_21, x_24, x_34, x_35);
x_37 = lean_apply_1(x_31, x_32);
x_38 = lp_mathlib_LinearEquiv_ofLinear___redArg(x_36, x_37);
return x_38;
}
}
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23, lean_object* x_24, lean_object* x_25) {
_start:
{
lean_object* x_26; 
x_26 = lp_mathlib_kroneckerTMulLinearEquiv___redArg(x_9, x_10, x_11, x_12, x_14, x_15, x_16, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25);
return x_26;
}
}
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___boxed(lean_object** _args) {
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
lean_object* x_19 = _args[18];
lean_object* x_20 = _args[19];
lean_object* x_21 = _args[20];
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
lean_object* x_24 = _args[23];
lean_object* x_25 = _args[24];
_start:
{
lean_object* x_26; 
x_26 = lp_mathlib_kroneckerTMulLinearEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25);
lean_dec_ref(x_13);
lean_dec_ref(x_10);
return x_26;
}
}
LEAN_EXPORT lean_object* lp_mathlib_kroneckerTMulLinearEquiv___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_kroneckerTMulLinearEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec_ref(x_2);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_kroneckerLinearEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_10 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_11 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_11);
lean_dec_ref(x_10);
x_12 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_1);
x_13 = lp_mathlib_Semiring_toModule___redArg(x_1);
lean_inc_n(x_13, 3);
lean_inc_ref_n(x_12, 2);
lean_inc_ref(x_1);
x_14 = lp_mathlib_kroneckerTMulLinearEquiv___redArg(x_1, x_1, x_12, x_12, x_13, x_13, x_13, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
x_15 = lp_mathlib_TensorProduct_lid___redArg(x_1, x_12, x_13);
x_16 = lp_mathlib_LinearEquiv_mapMatrix___redArg(x_15);
x_17 = lp_mathlib_LinearEquiv_trans___redArg(x_14, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_kroneckerLinearEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_kroneckerLinearEquiv___redArg(x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunBilinear___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_Matrix_addCommMonoid___redArg(x_6);
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_3, 1);
lean_inc(x_9);
lean_dec_ref(x_3);
x_10 = lp_mathlib_Matrix_module___redArg(x_8);
lean_inc_ref(x_2);
x_11 = lp_mathlib_Semiring_toModule___redArg(x_2);
x_12 = lp_mathlib_Matrix_module___redArg(x_11);
x_13 = lp_mathlib_Algebra_lsmul___redArg(x_2, x_1, x_7, x_12, x_10);
x_14 = lp_mathlib_AlgHom_toLinearMap___redArg(x_13);
x_15 = lp_mathlib_LinearMap_mapMatrix___redArg(x_9);
x_16 = lp_mathlib_LinearMap_compl_u2082___redArg(x_14, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunBilinear(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_MatrixEquivTensor_toFunBilinear___redArg(x_4, x_5, x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_MatrixEquivTensor_toFunLinear___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunLinear___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_5 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_6 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_4);
x_8 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_6);
x_9 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_7);
x_10 = lp_mathlib_Matrix_addCommMonoid___redArg(x_9);
x_11 = lp_mathlib_Matrix_addCommMonoid___redArg(x_8);
x_12 = lean_ctor_get(x_3, 0);
x_13 = lp_mathlib_MatrixEquivTensor_toFunLinear___redArg___closed__0;
lean_inc_ref(x_1);
x_14 = lp_mathlib_Semiring_toModule___redArg(x_1);
x_15 = lp_mathlib_Matrix_module___redArg(x_14);
lean_inc(x_12);
x_16 = lp_mathlib_Matrix_module___redArg(x_12);
lean_inc_ref(x_1);
x_17 = lp_mathlib_MatrixEquivTensor_toFunBilinear___redArg(x_1, x_2, x_3);
lean_inc_ref(x_1);
x_18 = lp_mathlib_TensorProduct_liftAux___redArg(x_1, x_1, x_13, x_10, x_11, x_15, x_16, x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunLinear(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_MatrixEquivTensor_toFunLinear___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunAlgHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_MatrixEquivTensor_toFunLinear___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_AlgHom_ofLinearMap___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunAlgHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_MatrixEquivTensor_toFunAlgHom___redArg(x_4, x_5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_toFunAlgHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_MatrixEquivTensor_toFunAlgHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_invFun___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec_ref(x_5);
lean_inc(x_7);
lean_inc(x_6);
x_8 = lean_apply_2(x_1, x_6, x_7);
lean_inc_ref(x_2);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_single), 11, 9);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, lean_box(0));
lean_closure_set(x_9, 3, x_2);
lean_closure_set(x_9, 4, x_2);
lean_closure_set(x_9, 5, x_3);
lean_closure_set(x_9, 6, x_6);
lean_closure_set(x_9, 7, x_7);
lean_closure_set(x_9, 8, x_4);
x_10 = lp_mathlib_TensorProduct_tmul___redArg(x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_invFun___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_7 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_11 = lean_ctor_get(x_10, 0);
lean_inc_ref(x_11);
x_12 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_12);
x_13 = lp_mathlib_Matrix_addCommMonoid___redArg(x_12);
x_14 = lean_ctor_get(x_3, 0);
lean_inc(x_14);
lean_dec_ref(x_3);
lean_inc_ref(x_1);
x_15 = lp_mathlib_Semiring_toModule___redArg(x_1);
x_16 = lp_mathlib_Matrix_module___redArg(x_15);
x_17 = lp_mathlib_TensorProduct_addMonoid___redArg(x_1, x_9, x_13, x_14, x_16);
x_18 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_11);
x_19 = lean_ctor_get(x_18, 1);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_10);
x_21 = lean_ctor_get(x_20, 2);
lean_inc(x_21);
lean_dec_ref(x_20);
lean_inc(x_5);
x_22 = lp_mathlib_Multiset_product___redArg(x_5, x_5);
x_23 = lean_alloc_closure((void*)(lp_mathlib_MatrixEquivTensor_invFun___redArg___lam__0), 5, 4);
lean_closure_set(x_23, 0, x_6);
lean_closure_set(x_23, 1, x_4);
lean_closure_set(x_23, 2, x_19);
lean_closure_set(x_23, 3, x_21);
x_24 = lp_mathlib_Finset_sum___redArg(x_17, x_22, x_23);
lean_dec_ref(x_17);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_invFun(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MatrixEquivTensor_invFun___redArg(x_4, x_5, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_invFun___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_MatrixEquivTensor_invFun(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_invFun___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_MatrixEquivTensor_invFun___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_equiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_MatrixEquivTensor_toFunAlgHom___redArg(x_1, x_2, x_3);
x_8 = lean_apply_3(x_7, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_equiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_MatrixEquivTensor_equiv___redArg___lam__0), 6, 3);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_2);
lean_closure_set(x_6, 2, x_3);
x_7 = lean_alloc_closure((void*)(lp_mathlib_MatrixEquivTensor_invFun___boxed), 9, 8);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, lean_box(0));
lean_closure_set(x_7, 3, x_1);
lean_closure_set(x_7, 4, x_2);
lean_closure_set(x_7, 5, x_3);
lean_closure_set(x_7, 6, x_4);
lean_closure_set(x_7, 7, x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MatrixEquivTensor_equiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_MatrixEquivTensor_equiv___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_matrixEquivTensor___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_6 = lp_mathlib_MatrixEquivTensor_equiv___redArg(x_1, x_2, x_3, x_5, x_4);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_6, 0);
lean_dec(x_8);
x_9 = lp_mathlib_MatrixEquivTensor_toFunAlgHom___redArg(x_1, x_2, x_3);
lean_ctor_set(x_6, 0, x_9);
x_10 = lp_mathlib_Equiv_symm___redArg(x_6);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_6, 1);
lean_inc(x_11);
lean_dec(x_6);
x_12 = lp_mathlib_MatrixEquivTensor_toFunAlgHom___redArg(x_1, x_2, x_3);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_11);
x_14 = lp_mathlib_Equiv_symm___redArg(x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_matrixEquivTensor(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_matrixEquivTensor___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulAlgEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_12 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_13 = lean_ctor_get(x_12, 0);
lean_inc_ref(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc_ref(x_14);
lean_dec_ref(x_13);
x_15 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_3);
x_16 = lean_ctor_get(x_15, 0);
lean_inc_ref(x_16);
lean_dec_ref(x_15);
x_17 = lean_ctor_get(x_16, 0);
lean_inc_ref(x_17);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_4, 0);
lean_inc(x_18);
lean_dec_ref(x_4);
x_19 = lean_ctor_get(x_9, 0);
lean_inc(x_19);
lean_dec_ref(x_9);
x_20 = lean_ctor_get(x_5, 0);
lean_inc(x_20);
lean_dec_ref(x_5);
lean_inc_ref(x_7);
lean_inc_ref(x_11);
lean_inc(x_6);
lean_inc(x_10);
x_21 = lp_mathlib_kroneckerTMulLinearEquiv___redArg(x_1, x_8, x_14, x_17, x_18, x_19, x_20, x_10, x_10, x_6, x_6, x_11, x_11, x_7, x_7);
x_22 = lp_mathlib_AlgEquiv_ofLinearEquiv___redArg(x_21);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulAlgEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Matrix_kroneckerTMulAlgEquiv___redArg(x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_16, x_18, x_19);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulAlgEquiv___boxed(lean_object** _args) {
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
lean_object* x_19 = _args[18];
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Matrix_kroneckerTMulAlgEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec_ref(x_15);
lean_dec_ref(x_14);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulAlgEquiv___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_kroneckerTMulAlgEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_8);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulStarAlgEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23, lean_object* x_24) {
_start:
{
lean_object* x_25; 
x_25 = lp_mathlib_Matrix_kroneckerTMulAlgEquiv___redArg(x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_16, x_18, x_19);
return x_25;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulStarAlgEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_kroneckerTMulAlgEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulStarAlgEquiv___boxed(lean_object** _args) {
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
lean_object* x_19 = _args[18];
lean_object* x_20 = _args[19];
lean_object* x_21 = _args[20];
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
lean_object* x_24 = _args[23];
_start:
{
lean_object* x_25; 
x_25 = lp_mathlib_Matrix_kroneckerTMulStarAlgEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24);
lean_dec(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec_ref(x_15);
lean_dec_ref(x_14);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
return x_25;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerTMulStarAlgEquiv___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_kroneckerTMulStarAlgEquiv___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_8);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerAlgEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lp_mathlib_Algebra_id___redArg(x_1);
lean_inc_ref_n(x_6, 3);
lean_inc_ref(x_1);
x_7 = lp_mathlib_Matrix_kroneckerTMulAlgEquiv___redArg(x_1, x_1, x_1, x_6, x_6, x_2, x_3, x_1, x_6, x_4, x_5);
lean_inc_ref(x_1);
x_8 = lp_mathlib_Algebra_TensorProduct_lid___redArg(x_1, x_1, x_6);
lean_dec_ref(x_1);
x_9 = lp_mathlib_AlgEquiv_mapMatrix___redArg(x_8);
x_10 = lp_mathlib_Equiv_trans___redArg(x_7, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerAlgEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Matrix_kroneckerAlgEquiv___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerStarAlgEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_kroneckerAlgEquiv___redArg(x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerStarAlgEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_kroneckerAlgEquiv___redArg(x_1, x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_kroneckerStarAlgEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_kroneckerStarAlgEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Star_StarAlgHom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Basis(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Composition(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Kronecker(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_Maps(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_MatrixAlgebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Star_StarAlgHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Basis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Composition(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Kronecker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TensorProduct_Maps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_MatrixEquivTensor_toFunLinear___redArg___closed__0 = _init_lp_mathlib_MatrixEquivTensor_toFunLinear___redArg___closed__0();
lean_mark_persistent(lp_mathlib_MatrixEquivTensor_toFunLinear___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
