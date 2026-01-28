// Lean compiler output
// Module: Mathlib.LinearAlgebra.Matrix.NonsingularInverse
// Imports: public import Init public import Mathlib.Data.Matrix.Invertible public import Mathlib.LinearAlgebra.FiniteDimensional.Basic public import Mathlib.LinearAlgebra.Matrix.Adjugate public import Mathlib.LinearAlgebra.Matrix.Kronecker public import Mathlib.LinearAlgebra.Matrix.SemiringInverse public import Mathlib.LinearAlgebra.Matrix.ToLin public import Mathlib.LinearAlgebra.Matrix.Trace
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
lean_object* lp_mathlib_Matrix_detRowAlternating___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_diag(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instInvertibleInv___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_submatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfInvertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instInvertibleInv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDetInvertible___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDiagonalInvertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalInvertibleEquivInvertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalInvertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_diagonalRingHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instInvertibleInv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivDetInvertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalInvertibleEquivInvertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalInvertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDiagonalInvertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instInvertibleInv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfLeftInverse___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_instFunLike___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_adjugate___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfInvertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDetInvertible___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_unitOfDetInvertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfRightInverse___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_unitOfDetInvertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfLeftInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalInvertible___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivDetInvertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfLeftInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDetInvertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDiagonalInvertible___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfInvertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDetInvertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDetInvertible___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_10);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lp_mathlib_Matrix_adjugate___redArg(x_2, x_3, x_4, x_5, x_7, x_8);
x_14 = lean_apply_2(x_12, x_6, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDetInvertible___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Matrix_invertibleOfDetInvertible___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_1);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDetInvertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfDetInvertible___redArg___lam__0___boxed), 8, 6);
lean_closure_set(x_7, 0, x_6);
lean_closure_set(x_7, 1, x_2);
lean_closure_set(x_7, 2, x_1);
lean_closure_set(x_7, 3, x_3);
lean_closure_set(x_7, 4, x_4);
lean_closure_set(x_7, 5, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDetInvertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_invertibleOfDetInvertible___redArg(x_3, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfLeftInverse(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; 
x_9 = lp_mathlib_Matrix_detRowAlternating___redArg(x_4, x_3, x_5);
x_10 = lean_apply_1(x_9, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfLeftInverse___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Matrix_detRowAlternating___redArg(x_2, x_1, x_3);
x_6 = lean_apply_1(x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfLeftInverse___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Matrix_detInvertibleOfLeftInverse(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfRightInverse(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; 
x_9 = lp_mathlib_Matrix_detRowAlternating___redArg(x_4, x_3, x_5);
x_10 = lean_apply_1(x_9, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfRightInverse___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Matrix_detRowAlternating___redArg(x_2, x_1, x_3);
x_6 = lean_apply_1(x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfRightInverse___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Matrix_detInvertibleOfRightInverse(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfInvertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lp_mathlib_Matrix_detRowAlternating___redArg(x_4, x_3, x_5);
x_9 = lean_apply_1(x_8, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfInvertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Matrix_detRowAlternating___redArg(x_2, x_1, x_3);
x_6 = lean_apply_1(x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_detInvertibleOfInvertible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_detInvertibleOfInvertible(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivDetInvertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc(x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Matrix_detInvertibleOfInvertible___boxed), 7, 6);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, x_1);
lean_closure_set(x_5, 3, x_2);
lean_closure_set(x_5, 4, x_3);
lean_closure_set(x_5, 5, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfDetInvertible), 7, 6);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_1);
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, x_3);
lean_closure_set(x_6, 5, x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivDetInvertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_invertibleEquivDetInvertible___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_unitOfDetInvertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
x_6 = lp_mathlib_Matrix_invertibleOfDetInvertible___redArg(x_1, x_2, x_3, x_4, x_5);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_4);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_unitOfDetInvertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_unitOfDetInvertible___redArg(x_3, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instInvertibleInv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_inc(x_6);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instInvertibleInv___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instInvertibleInv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_instInvertibleInv(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instInvertibleInv___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Matrix_instInvertibleInv___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalInvertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Matrix_diagonalRingHom___redArg(x_2, x_1);
x_5 = lp_mathlib_NonUnitalRingHom_instFunLike___lam__0(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalInvertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_diagonalInvertible___redArg(x_3, x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalInvertible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_diagonalInvertible(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDiagonalInvertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Matrix_diag), 4, 3);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, lean_box(0));
lean_closure_set(x_8, 2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDiagonalInvertible___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Matrix_diag), 4, 3);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfDiagonalInvertible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_invertibleOfDiagonalInvertible(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalInvertibleEquivInvertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_2);
lean_inc(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfDiagonalInvertible___boxed), 7, 6);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_1);
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, x_3);
lean_closure_set(x_6, 5, x_4);
x_7 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_5);
lean_dec_ref(x_5);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonalInvertible___boxed), 7, 6);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_1);
lean_closure_set(x_8, 2, x_2);
lean_closure_set(x_8, 3, lean_box(0));
lean_closure_set(x_8, 4, x_7);
lean_closure_set(x_8, 5, x_4);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalInvertibleEquivInvertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_diagonalInvertibleEquivInvertible___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___redArg___lam__1), 2, 1);
lean_closure_set(x_5, 0, x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrix), 10, 8);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, lean_box(0));
lean_closure_set(x_6, 3, lean_box(0));
lean_closure_set(x_6, 4, lean_box(0));
lean_closure_set(x_6, 5, x_3);
lean_closure_set(x_6, 6, x_5);
lean_closure_set(x_6, 7, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_submatrixEquivInvertible___redArg(x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_submatrixEquivInvertible(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_Equiv_symm___redArg(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___redArg___lam__0), 2, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___redArg___lam__0), 2, 1);
lean_closure_set(x_7, 0, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrix), 10, 8);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, lean_box(0));
lean_closure_set(x_8, 2, lean_box(0));
lean_closure_set(x_8, 3, lean_box(0));
lean_closure_set(x_8, 4, lean_box(0));
lean_closure_set(x_8, 5, x_3);
lean_closure_set(x_8, 6, x_5);
lean_closure_set(x_8, 7, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___redArg(x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_inc(x_8);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0_spec__0___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lp_mathlib_Equiv_symm___redArg(x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___redArg___lam__0), 2, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lp_mathlib_Equiv_symm___redArg(x_6);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___redArg___lam__0), 2, 1);
lean_closure_set(x_11, 0, x_10);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrix), 10, 8);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, lean_box(0));
lean_closure_set(x_12, 2, lean_box(0));
lean_closure_set(x_12, 3, lean_box(0));
lean_closure_set(x_12, 4, lean_box(0));
lean_closure_set(x_12, 5, x_4);
lean_closure_set(x_12, 6, x_9);
lean_closure_set(x_12, 7, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0___redArg(x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___redArg___lam__0), 2, 1);
lean_closure_set(x_8, 0, x_6);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___redArg___lam__1), 2, 1);
lean_closure_set(x_9, 0, x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrix), 10, 8);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, lean_box(0));
lean_closure_set(x_10, 3, lean_box(0));
lean_closure_set(x_10, 4, lean_box(0));
lean_closure_set(x_10, 5, x_4);
lean_closure_set(x_10, 6, x_9);
lean_closure_set(x_10, 7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0___redArg(x_4, x_5, x_6, x_9, x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0___redArg(x_6, x_7, x_8, x_12, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0___redArg(x_4, x_5, x_6, x_12, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc(x_1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__0___boxed), 12, 11);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, lean_box(0));
lean_closure_set(x_9, 3, x_1);
lean_closure_set(x_9, 4, x_2);
lean_closure_set(x_9, 5, x_3);
lean_closure_set(x_9, 6, x_4);
lean_closure_set(x_9, 7, x_5);
lean_closure_set(x_9, 8, x_6);
lean_closure_set(x_9, 9, x_7);
lean_closure_set(x_9, 10, x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__1___boxed), 12, 11);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, lean_box(0));
lean_closure_set(x_10, 3, x_1);
lean_closure_set(x_10, 4, x_2);
lean_closure_set(x_10, 5, x_3);
lean_closure_set(x_10, 6, x_4);
lean_closure_set(x_10, 7, x_5);
lean_closure_set(x_10, 8, x_6);
lean_closure_set(x_10, 9, x_7);
lean_closure_set(x_10, 10, x_8);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0___redArg(x_3, x_4, x_5, x_9, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__1___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0___redArg(x_1, x_2, x_3, x_9, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0_spec__0___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0_spec__0___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_10);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_10);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__1___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_submatrixEquivInvertibleEquivInvertible___elam__1___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__1_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_invertibleOfSubmatrixEquivInvertible___at___00Matrix_submatrixEquivInvertibleEquivInvertible___elam__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Invertible(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_FiniteDimensional_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Adjugate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Kronecker(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_SemiringInverse(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ToLin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Trace(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_NonsingularInverse(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_FiniteDimensional_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Adjugate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Kronecker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_SemiringInverse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ToLin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
