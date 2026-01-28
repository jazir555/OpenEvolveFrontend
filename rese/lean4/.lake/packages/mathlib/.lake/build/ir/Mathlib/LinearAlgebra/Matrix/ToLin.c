// Lean compiler output
// Module: Mathlib.LinearAlgebra.Matrix.ToLin
// Imports: public import Init public import Mathlib.Algebra.Algebra.Subalgebra.Tower public import Mathlib.Algebra.Module.Projective public import Mathlib.Data.Finite.Sum public import Mathlib.Data.Matrix.Block public import Mathlib.LinearAlgebra.Basis.Basic public import Mathlib.LinearAlgebra.Basis.Fin public import Mathlib.LinearAlgebra.Basis.Prod public import Mathlib.LinearAlgebra.Basis.SMul public import Mathlib.LinearAlgebra.Matrix.Notation public import Mathlib.LinearAlgebra.Matrix.StdBasis public import Mathlib.RingTheory.AlgebraTower public import Mathlib.RingTheory.Ideal.Span
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
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecLin___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixRight_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProductBilin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecBilin___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_endVecAlgEquivMatrixEnd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_matrixAlgEquivEndVecMulOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_matrixAlgEquivEndVecMulOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearMapRight_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_endVecAlgEquivMatrixEnd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLin_x27OfInv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_vecMulVec___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_matrixAlgEquivEndVecMulOpposite___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_matrixRingEquivEndVecMulOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingEquiv_moduleEndSelf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_vecMulVecBilin___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_vecMulVecBilin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulBilin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecBilin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLin_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixAlgEquiv_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLin_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixRight_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulBilin___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearMapRight_x27___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_semiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProductBilin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Module_End_instSemiring___redArg(lean_object*);
static lean_object* lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__2___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecBilin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_algEquivMatrix_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProductBilin___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulLinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_matrixRingEquivEndVecMulOpposite___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_vecMulVecBilin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquivRight_x27OfInv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_mulVec___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulBilin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_endVecAlgEquivMatrixEnd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_op___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProductBilin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecLin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_opOp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_dotProduct___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinAlgEquiv_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulLinear___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_vecMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixAlgEquiv_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecLin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinAlgEquiv_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_vecMulVecBilin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_single___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecLin___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLin_x27OfInv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulLinear___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProductBilin___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecBilin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_single___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulLinear___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_flip___redArg(lean_object*);
lean_object* lp_mathlib_AlgEquiv_mopMatrix___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulBilin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_algEquivMatrix_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulBilin___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_vecMul___redArg(x_1, x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulBilin___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecMulBilin___redArg___lam__0), 5, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulBilin(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_vecMulBilin___redArg(x_8, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulBilin___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_vecMulBilin(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecBilin___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_mulVec___redArg(x_1, x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecBilin___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Matrix_mulVecBilin___redArg___lam__0), 5, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecBilin(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_mulVecBilin___redArg(x_8, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecBilin___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_mulVecBilin(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_vecMulVecBilin___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_vecMulVec___redArg(x_1, x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_vecMulVecBilin___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
lean_dec_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_vecMulVecBilin___redArg___lam__0), 5, 1);
lean_closure_set(x_4, 0, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_vecMulVecBilin(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_vecMulVecBilin___redArg(x_8);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_vecMulVecBilin___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_vecMulVecBilin(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProductBilin___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_dotProduct___redArg(x_1, x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProductBilin___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_dotProductBilin___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProductBilin___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_dotProductBilin___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_5);
lean_closure_set(x_6, 2, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProductBilin(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_dotProductBilin___redArg(x_7, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProductBilin___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_dotProductBilin(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulLinear(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_7 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecMulBilin___redArg___lam__0), 5, 2);
lean_closure_set(x_9, 0, x_8);
lean_closure_set(x_9, 1, x_5);
x_10 = lp_mathlib_LinearMap_flip___redArg(x_9);
x_11 = lean_apply_1(x_10, x_6);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulLinear___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecMulBilin___redArg___lam__0), 5, 2);
lean_closure_set(x_6, 0, x_5);
lean_closure_set(x_6, 1, x_2);
x_7 = lp_mathlib_LinearMap_flip___redArg(x_6);
x_8 = lean_apply_1(x_7, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulLinear___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_vecMulLinear(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_vecMulLinear___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Matrix_vecMulLinear___redArg(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_mathlib_LinearMap_single___redArg(x_1, x_2, x_5);
x_8 = lean_apply_1(x_7, x_3);
x_9 = lean_apply_2(x_4, x_8, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixRight_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_7 = lean_ctor_get(x_5, 0);
x_8 = lean_ctor_get(x_5, 1);
lean_dec(x_8);
x_9 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_4);
x_10 = lean_ctor_get(x_9, 2);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_11, 0, x_7);
x_12 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__1), 6, 3);
lean_closure_set(x_12, 0, x_11);
lean_closure_set(x_12, 1, x_3);
lean_closure_set(x_12, 2, x_10);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecMulLinear___boxed), 6, 5);
lean_closure_set(x_13, 0, lean_box(0));
lean_closure_set(x_13, 1, x_1);
lean_closure_set(x_13, 2, lean_box(0));
lean_closure_set(x_13, 3, lean_box(0));
lean_closure_set(x_13, 4, x_2);
lean_ctor_set(x_5, 1, x_13);
lean_ctor_set(x_5, 0, x_12);
return x_5;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_14 = lean_ctor_get(x_5, 0);
lean_inc(x_14);
lean_dec(x_5);
x_15 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_4);
x_16 = lean_ctor_get(x_15, 2);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_17, 0, x_14);
x_18 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__1), 6, 3);
lean_closure_set(x_18, 0, x_17);
lean_closure_set(x_18, 1, x_3);
lean_closure_set(x_18, 2, x_16);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecMulLinear___boxed), 6, 5);
lean_closure_set(x_19, 0, lean_box(0));
lean_closure_set(x_19, 1, x_1);
lean_closure_set(x_19, 2, lean_box(0));
lean_closure_set(x_19, 3, lean_box(0));
lean_closure_set(x_19, 4, x_2);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_18);
lean_ctor_set(x_20, 1, x_19);
return x_20;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixRight_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_LinearMap_toMatrixRight_x27___redArg(x_2, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearMapRight_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_mathlib_LinearMap_toMatrixRight_x27___redArg(x_2, x_5, x_6);
x_8 = lp_mathlib_LinearEquiv_symm___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearMapRight_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_LinearMap_toMatrixRight_x27___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_LinearEquiv_symm___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_3(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_3(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
lean_inc_ref(x_1);
x_8 = lp_mathlib_LinearMap_toMatrixRight_x27___redArg(x_1, x_4, x_5);
x_9 = lp_mathlib_LinearEquiv_symm___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_LinearMap_toMatrixRight_x27___redArg(x_1, x_2, x_3);
x_12 = lp_mathlib_LinearEquiv_symm___redArg(x_11);
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_14 = lean_ctor_get(x_12, 0);
x_15 = lean_ctor_get(x_12, 1);
lean_dec(x_15);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__0), 4, 2);
lean_closure_set(x_16, 0, x_10);
lean_closure_set(x_16, 1, x_7);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__1), 4, 2);
lean_closure_set(x_17, 0, x_14);
lean_closure_set(x_17, 1, x_6);
lean_ctor_set(x_12, 1, x_17);
lean_ctor_set(x_12, 0, x_16);
return x_12;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_18 = lean_ctor_get(x_12, 0);
lean_inc(x_18);
lean_dec(x_12);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__0), 4, 2);
lean_closure_set(x_19, 0, x_10);
lean_closure_set(x_19, 1, x_7);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__1), 4, 2);
lean_closure_set(x_20, 0, x_18);
lean_closure_set(x_20, 1, x_6);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_20);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinearEquivRight_x27OfInv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg(x_2, x_5, x_6, x_7, x_8, x_9, x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecLin(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_2);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_mulVecBilin___redArg___lam__0), 5, 3);
lean_closure_set(x_9, 0, x_8);
lean_closure_set(x_9, 1, x_5);
lean_closure_set(x_9, 2, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecLin___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Matrix_mulVecBilin___redArg___lam__0), 5, 3);
lean_closure_set(x_6, 0, x_5);
lean_closure_set(x_6, 1, x_2);
lean_closure_set(x_6, 2, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecLin___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_mulVecLin(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_mulVecLin___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Matrix_mulVecLin___redArg(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__2___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Pi_single___boxed), 7, 6);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, x_1);
lean_closure_set(x_7, 3, x_2);
lean_closure_set(x_7, 4, x_6);
lean_closure_set(x_7, 5, x_3);
x_8 = lean_apply_2(x_4, x_7, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__2___closed__0;
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__1), 6, 4);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_2);
lean_closure_set(x_9, 2, x_3);
lean_closure_set(x_9, 3, x_4);
x_10 = lean_apply_3(x_8, x_9, x_5, x_6);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
x_6 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_5);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_8 = lean_ctor_get(x_6, 1);
x_9 = lean_ctor_get(x_6, 0);
lean_dec(x_9);
x_10 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_4);
x_11 = lean_ctor_get(x_10, 2);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_12, 0, x_8);
x_13 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__2), 6, 3);
lean_closure_set(x_13, 0, x_12);
lean_closure_set(x_13, 1, x_2);
lean_closure_set(x_13, 2, x_11);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_mulVecLin___boxed), 6, 5);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, x_1);
lean_closure_set(x_14, 2, lean_box(0));
lean_closure_set(x_14, 3, lean_box(0));
lean_closure_set(x_14, 4, x_3);
lean_ctor_set(x_6, 1, x_14);
lean_ctor_set(x_6, 0, x_13);
return x_6;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_15 = lean_ctor_get(x_6, 1);
lean_inc(x_15);
lean_dec(x_6);
x_16 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_4);
x_17 = lean_ctor_get(x_16, 2);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_18, 0, x_15);
x_19 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__2), 6, 3);
lean_closure_set(x_19, 0, x_18);
lean_closure_set(x_19, 1, x_2);
lean_closure_set(x_19, 2, x_17);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_mulVecLin___boxed), 6, 5);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, x_1);
lean_closure_set(x_20, 2, lean_box(0));
lean_closure_set(x_20, 3, lean_box(0));
lean_closure_set(x_20, 4, x_3);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_20);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrix_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_LinearMap_toMatrix_x27___redArg(x_2, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLin_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_LinearMap_toMatrix_x27___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_LinearEquiv_symm___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLin_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_toLin_x27___redArg(x_2, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLin_x27OfInv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
lean_inc_ref(x_1);
x_8 = lp_mathlib_Matrix_toLin_x27___redArg(x_1, x_5, x_4);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_Matrix_toLin_x27___redArg(x_1, x_2, x_3);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
lean_dec(x_13);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__0), 4, 2);
lean_closure_set(x_14, 0, x_9);
lean_closure_set(x_14, 1, x_7);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__1), 4, 2);
lean_closure_set(x_15, 0, x_12);
lean_closure_set(x_15, 1, x_6);
lean_ctor_set(x_10, 1, x_15);
lean_ctor_set(x_10, 0, x_14);
return x_10;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_16 = lean_ctor_get(x_10, 0);
lean_inc(x_16);
lean_dec(x_10);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__0), 4, 2);
lean_closure_set(x_17, 0, x_9);
lean_closure_set(x_17, 1, x_7);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toLinearEquivRight_x27OfInv___redArg___lam__1), 4, 2);
lean_closure_set(x_18, 0, x_16);
lean_closure_set(x_18, 1, x_6);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLin_x27OfInv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_toLin_x27OfInv___redArg(x_2, x_5, x_6, x_7, x_8, x_9, x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixAlgEquiv_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_LinearMap_toMatrix_x27___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_AlgEquiv_ofLinearEquiv___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMatrixAlgEquiv_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LinearMap_toMatrixAlgEquiv_x27___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinAlgEquiv_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_LinearMap_toMatrixAlgEquiv_x27___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_Equiv_symm___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_toLinAlgEquiv_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_toLinAlgEquiv_x27___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_algEquivMatrix_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LinearMap_toMatrixAlgEquiv_x27___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_algEquivMatrix_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_LinearMap_toMatrixAlgEquiv_x27___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
lean_inc(x_4);
x_5 = lean_apply_1(x_1, x_4);
x_6 = lean_apply_3(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__0), 4, 3);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_3);
lean_closure_set(x_6, 2, x_5);
x_7 = lp_mathlib_Finset_sum___redArg(x_1, x_2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__1(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Pi_single___boxed), 7, 6);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, x_1);
lean_closure_set(x_7, 3, x_2);
lean_closure_set(x_7, 4, x_5);
lean_closure_set(x_7, 5, x_6);
x_8 = lean_apply_2(x_3, x_7, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_3);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_dec(x_7);
x_8 = lean_alloc_closure((void*)(lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__1___boxed), 5, 2);
lean_closure_set(x_8, 0, x_3);
lean_closure_set(x_8, 1, x_1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_9, 0, x_6);
x_10 = lean_alloc_closure((void*)(lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__3), 6, 2);
lean_closure_set(x_10, 0, x_9);
lean_closure_set(x_10, 1, x_2);
lean_ctor_set(x_4, 1, x_8);
lean_ctor_set(x_4, 0, x_10);
return x_4;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_ctor_get(x_4, 0);
lean_inc(x_11);
lean_dec(x_4);
x_12 = lean_alloc_closure((void*)(lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__1___boxed), 5, 2);
lean_closure_set(x_12, 0, x_3);
lean_closure_set(x_12, 1, x_1);
x_13 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_13, 0, x_11);
x_14 = lean_alloc_closure((void*)(lp_mathlib_endVecRingEquivMatrixEnd___redArg___lam__3), 6, 2);
lean_closure_set(x_14, 0, x_13);
lean_closure_set(x_14, 1, x_2);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_12);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_endVecRingEquivMatrixEnd___redArg(x_2, x_3, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_endVecRingEquivMatrixEnd___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_endVecRingEquivMatrixEnd(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_endVecAlgEquivMatrixEnd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_endVecRingEquivMatrixEnd___redArg(x_2, x_3, x_10);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_endVecAlgEquivMatrixEnd___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_endVecRingEquivMatrixEnd___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_endVecAlgEquivMatrixEnd___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_endVecAlgEquivMatrixEnd(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_matrixAlgEquivEndVecMulOpposite___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
lean_inc_ref(x_2);
lean_inc(x_1);
lean_inc_ref(x_3);
x_4 = lp_mathlib_Matrix_semiring___redArg(x_3, x_1, x_2);
lean_inc_ref(x_4);
x_5 = lp_mathlib_MulOpposite_instSemiring___redArg(x_4);
x_6 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_3);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_7);
lean_inc_ref(x_8);
x_9 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toMatrixRight_x27___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lp_mathlib_Pi_addCommMonoid___redArg(x_9);
x_11 = lp_mathlib_Module_End_instSemiring___redArg(x_10);
x_12 = lp_mathlib_AlgEquiv_op___redArg(x_5, x_11);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_AlgEquiv_opOp___redArg(x_4);
lean_dec_ref(x_4);
lean_inc_ref(x_3);
x_15 = lp_mathlib_AlgEquiv_mopMatrix___redArg(x_1, x_3);
x_16 = lp_mathlib_Equiv_symm___redArg(x_15);
x_17 = lp_mathlib_RingEquiv_moduleEndSelf___redArg(x_3);
x_18 = lp_mathlib_AlgEquiv_mapMatrix___redArg(x_17);
x_19 = lp_mathlib_endVecRingEquivMatrixEnd___redArg(x_1, x_2, x_8);
x_20 = lp_mathlib_Equiv_symm___redArg(x_19);
x_21 = lp_mathlib_Equiv_trans___redArg(x_18, x_20);
x_22 = lp_mathlib_Equiv_trans___redArg(x_16, x_21);
x_23 = lean_apply_1(x_13, x_22);
x_24 = lp_mathlib_Equiv_trans___redArg(x_14, x_23);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_matrixAlgEquivEndVecMulOpposite(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_matrixAlgEquivEndVecMulOpposite___redArg(x_2, x_3, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_matrixAlgEquivEndVecMulOpposite___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_matrixAlgEquivEndVecMulOpposite(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_8);
lean_dec_ref(x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_matrixRingEquivEndVecMulOpposite(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_matrixAlgEquivEndVecMulOpposite___redArg(x_2, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_matrixRingEquivEndVecMulOpposite___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_matrixAlgEquivEndVecMulOpposite___redArg(x_1, x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Tower(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Projective(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finite_Sum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Block(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Basis_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Basis_Fin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Basis_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Basis_SMul(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_StdBasis(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_AlgebraTower(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Span(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ToLin(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Projective(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finite_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Block(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Basis_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Basis_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Basis_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Basis_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_StdBasis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_AlgebraTower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Span(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__2___closed__0 = _init_lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__2___closed__0();
lean_mark_persistent(lp_mathlib_LinearMap_toMatrix_x27___redArg___lam__2___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
