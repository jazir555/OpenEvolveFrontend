// Lean compiler output
// Module: Mathlib.LinearAlgebra.Matrix.SchurComplement
// Imports: public import Init public import Mathlib.Data.Matrix.Invertible public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
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
lean_object* lp_mathlib_Invertible_mulRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__3___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_fromBlocks(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Invertible_mulRight___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9_spec__9___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__8___boxed(lean_object**);
lean_object* lp_mathlib_Matrix_submatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__1___boxed(lean_object**);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Invertible_mulLeft___elam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__9(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumComm(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Invertible_mulRight___elam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__2___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_semiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_fromBlocks___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__3___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_toBlocks_u2081_u2081(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_dotProduct___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Invertible_mulLeft___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqSum_decEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__1___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__17___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_toBlocks_u2082_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_disjSum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__0___redArg(lean_object*);
lean_object* lp_mathlib_invertibleOne___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
static lean_object* lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9_spec__9___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__2___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6_spec__6___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_diagonal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__2___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__3(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Invertible_mulLeft___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__17___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4___redArg___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__3), 3, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_6);
x_8 = lp_mathlib_dotProduct___redArg(x_2, x_3, x_4, x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__4(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1), 3, 2);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_10);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_12, 0, x_2);
lean_closure_set(x_12, 1, x_9);
lean_inc_ref(x_6);
lean_inc(x_5);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__4___boxed), 6, 5);
lean_closure_set(x_13, 0, x_3);
lean_closure_set(x_13, 1, x_4);
lean_closure_set(x_13, 2, x_5);
lean_closure_set(x_13, 3, x_6);
lean_closure_set(x_13, 4, x_12);
x_14 = lp_mathlib_dotProduct___redArg(x_7, x_5, x_6, x_13, x_11);
lean_dec_ref(x_6);
x_15 = lean_apply_1(x_8, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
lean_inc_ref(x_3);
x_7 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_3);
x_8 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_7);
lean_inc_ref(x_8);
x_9 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_8);
x_10 = lean_ctor_get(x_9, 1);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_Ring_toAddCommGroup___redArg(x_3);
x_12 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_11);
lean_dec_ref(x_11);
x_13 = lean_ctor_get(x_12, 1);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc_ref(x_8);
x_14 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_8);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_16);
lean_dec_ref(x_8);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_17, 0, x_10);
lean_inc(x_5);
lean_inc(x_6);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__5), 10, 8);
lean_closure_set(x_18, 0, x_6);
lean_closure_set(x_18, 1, x_5);
lean_closure_set(x_18, 2, x_4);
lean_closure_set(x_18, 3, x_1);
lean_closure_set(x_18, 4, x_15);
lean_closure_set(x_18, 5, x_16);
lean_closure_set(x_18, 6, x_2);
lean_closure_set(x_18, 7, x_13);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_19, 0, lean_box(0));
lean_closure_set(x_19, 1, lean_box(0));
lean_closure_set(x_19, 2, lean_box(0));
lean_closure_set(x_19, 3, lean_box(0));
lean_closure_set(x_19, 4, lean_box(0));
lean_closure_set(x_19, 5, x_5);
lean_closure_set(x_19, 6, x_18);
lean_closure_set(x_19, 7, x_17);
lean_closure_set(x_19, 8, x_6);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg(x_4, x_5, x_8, x_10, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_11);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__3), 3, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_6);
x_8 = lp_mathlib_dotProduct___redArg(x_2, x_3, x_4, x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1), 3, 2);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_10);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_12, 0, x_2);
lean_closure_set(x_12, 1, x_9);
lean_inc_ref(x_6);
lean_inc(x_5);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__0___boxed), 6, 5);
lean_closure_set(x_13, 0, x_3);
lean_closure_set(x_13, 1, x_4);
lean_closure_set(x_13, 2, x_5);
lean_closure_set(x_13, 3, x_6);
lean_closure_set(x_13, 4, x_12);
x_14 = lp_mathlib_dotProduct___redArg(x_7, x_5, x_6, x_13, x_11);
lean_dec_ref(x_6);
x_15 = lean_apply_1(x_8, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
lean_inc_ref(x_3);
x_7 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_3);
x_8 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_7);
lean_inc_ref(x_8);
x_9 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_8);
x_10 = lean_ctor_get(x_9, 1);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_Ring_toAddCommGroup___redArg(x_3);
x_12 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_11);
lean_dec_ref(x_11);
x_13 = lean_ctor_get(x_12, 1);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc_ref(x_8);
x_14 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_8);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_16);
lean_dec_ref(x_8);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_17, 0, x_10);
lean_inc(x_6);
lean_inc(x_5);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__1), 10, 8);
lean_closure_set(x_18, 0, x_5);
lean_closure_set(x_18, 1, x_6);
lean_closure_set(x_18, 2, x_4);
lean_closure_set(x_18, 3, x_2);
lean_closure_set(x_18, 4, x_15);
lean_closure_set(x_18, 5, x_16);
lean_closure_set(x_18, 6, x_1);
lean_closure_set(x_18, 7, x_13);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_19, 0, lean_box(0));
lean_closure_set(x_19, 1, lean_box(0));
lean_closure_set(x_19, 2, lean_box(0));
lean_closure_set(x_19, 3, lean_box(0));
lean_closure_set(x_19, 4, lean_box(0));
lean_closure_set(x_19, 5, x_5);
lean_closure_set(x_19, 6, x_17);
lean_closure_set(x_19, 7, x_18);
lean_closure_set(x_19, 8, x_6);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg(x_4, x_5, x_8, x_10, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_11);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2081_u2081), 8, 6);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, lean_box(0));
lean_closure_set(x_2, 3, lean_box(0));
lean_closure_set(x_2, 4, lean_box(0));
lean_closure_set(x_2, 5, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2082_u2082), 8, 6);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, lean_box(0));
lean_closure_set(x_3, 2, lean_box(0));
lean_closure_set(x_3, 3, lean_box(0));
lean_closure_set(x_3, 4, lean_box(0));
lean_closure_set(x_3, 5, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___redArg(x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2081_u2081), 8, 6);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, lean_box(0));
lean_closure_set(x_2, 3, lean_box(0));
lean_closure_set(x_2, 4, lean_box(0));
lean_closure_set(x_2, 5, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2082_u2082), 8, 6);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, lean_box(0));
lean_closure_set(x_3, 2, lean_box(0));
lean_closure_set(x_3, 3, lean_box(0));
lean_closure_set(x_3, 4, lean_box(0));
lean_closure_set(x_3, 5, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___redArg(x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_inc(x_8);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__0___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_inc(x_8);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__1___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_ctor_get(x_5, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_5, 1);
lean_inc(x_9);
lean_dec_ref(x_5);
x_10 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg(x_1, x_2, x_3, x_4, x_8, x_9);
x_11 = lean_apply_2(x_10, x_6, x_7);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc(x_6);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2081_u2081), 8, 6);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, lean_box(0));
lean_closure_set(x_9, 3, lean_box(0));
lean_closure_set(x_9, 4, lean_box(0));
lean_closure_set(x_9, 5, x_6);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2082_u2082), 8, 6);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, lean_box(0));
lean_closure_set(x_10, 3, lean_box(0));
lean_closure_set(x_10, 4, lean_box(0));
lean_closure_set(x_10, 5, x_6);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_12, x_9, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc(x_7);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___redArg___lam__0), 7, 4);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_2);
lean_closure_set(x_9, 2, x_5);
lean_closure_set(x_9, 3, x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0___boxed), 12, 11);
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
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_9, x_6, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_invertibleOfLeftInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__0___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_invertibleOfLeftInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__0___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__1___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_invertibleOfRightInverse___at___00Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0_spec__1___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocksZero_u2082_u2081InvertibleEquiv___elam__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_ctor_get(x_5, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_5, 1);
lean_inc(x_9);
lean_dec_ref(x_5);
x_10 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg(x_1, x_2, x_3, x_4, x_8, x_9);
x_11 = lean_apply_2(x_10, x_6, x_7);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0_spec__0___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2081_u2081), 8, 6);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, lean_box(0));
lean_closure_set(x_2, 3, lean_box(0));
lean_closure_set(x_2, 4, lean_box(0));
lean_closure_set(x_2, 5, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2082_u2082), 8, 6);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, lean_box(0));
lean_closure_set(x_3, 2, lean_box(0));
lean_closure_set(x_3, 3, lean_box(0));
lean_closure_set(x_3, 4, lean_box(0));
lean_closure_set(x_3, 5, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0_spec__0___redArg(x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc(x_7);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_1);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___redArg___lam__0), 7, 4);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_2);
lean_closure_set(x_9, 2, x_5);
lean_closure_set(x_9, 3, x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0___boxed), 12, 11);
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
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0_spec__0___redArg(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0_spec__0___redArg(x_9);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_fromBlocksZero_u2081_u2082InvertibleEquiv___elam__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__4), 3, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_6);
x_8 = lp_mathlib_dotProduct___redArg(x_2, x_3, x_4, x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
lean_inc(x_11);
lean_inc(x_1);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1), 3, 2);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_11);
lean_inc(x_10);
lean_inc(x_1);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_13, 0, x_1);
lean_closure_set(x_13, 1, x_10);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc(x_3);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__0___boxed), 6, 5);
lean_closure_set(x_14, 0, x_2);
lean_closure_set(x_14, 1, x_3);
lean_closure_set(x_14, 2, x_4);
lean_closure_set(x_14, 3, x_5);
lean_closure_set(x_14, 4, x_13);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc(x_7);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__0___boxed), 6, 5);
lean_closure_set(x_15, 0, x_6);
lean_closure_set(x_15, 1, x_7);
lean_closure_set(x_15, 2, x_4);
lean_closure_set(x_15, 3, x_5);
lean_closure_set(x_15, 4, x_14);
lean_inc_ref(x_5);
lean_inc(x_4);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__4___boxed), 6, 5);
lean_closure_set(x_16, 0, x_8);
lean_closure_set(x_16, 1, x_7);
lean_closure_set(x_16, 2, x_4);
lean_closure_set(x_16, 3, x_5);
lean_closure_set(x_16, 4, x_15);
x_17 = lean_apply_2(x_1, x_10, x_11);
x_18 = lp_mathlib_dotProduct___redArg(x_3, x_4, x_5, x_16, x_12);
lean_dec_ref(x_5);
x_19 = lean_apply_2(x_9, x_17, x_18);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
lean_inc_ref(x_3);
x_8 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_3);
x_9 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_8);
lean_inc_ref(x_9);
x_10 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_10, 1);
lean_inc(x_12);
lean_dec_ref(x_10);
x_13 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_13);
lean_dec_ref(x_9);
x_14 = lp_mathlib_Ring_toAddCommGroup___redArg(x_3);
x_15 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_14);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_15, 1);
lean_inc(x_16);
lean_dec_ref(x_15);
lean_inc(x_4);
lean_inc(x_1);
lean_inc(x_7);
lean_inc_ref(x_13);
lean_inc(x_11);
lean_inc(x_2);
lean_inc(x_5);
lean_inc(x_6);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__3), 11, 9);
lean_closure_set(x_17, 0, x_6);
lean_closure_set(x_17, 1, x_5);
lean_closure_set(x_17, 2, x_2);
lean_closure_set(x_17, 3, x_11);
lean_closure_set(x_17, 4, x_13);
lean_closure_set(x_17, 5, x_7);
lean_closure_set(x_17, 6, x_1);
lean_closure_set(x_17, 7, x_4);
lean_closure_set(x_17, 8, x_12);
lean_inc(x_16);
lean_inc(x_2);
lean_inc_ref(x_13);
lean_inc(x_11);
lean_inc(x_1);
lean_inc(x_7);
lean_inc(x_6);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__5), 10, 8);
lean_closure_set(x_18, 0, x_6);
lean_closure_set(x_18, 1, x_7);
lean_closure_set(x_18, 2, x_4);
lean_closure_set(x_18, 3, x_1);
lean_closure_set(x_18, 4, x_11);
lean_closure_set(x_18, 5, x_13);
lean_closure_set(x_18, 6, x_2);
lean_closure_set(x_18, 7, x_16);
lean_inc(x_7);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__1), 10, 8);
lean_closure_set(x_19, 0, x_7);
lean_closure_set(x_19, 1, x_6);
lean_closure_set(x_19, 2, x_5);
lean_closure_set(x_19, 3, x_2);
lean_closure_set(x_19, 4, x_11);
lean_closure_set(x_19, 5, x_13);
lean_closure_set(x_19, 6, x_1);
lean_closure_set(x_19, 7, x_16);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, lean_box(0));
lean_closure_set(x_20, 2, lean_box(0));
lean_closure_set(x_20, 3, lean_box(0));
lean_closure_set(x_20, 4, lean_box(0));
lean_closure_set(x_20, 5, x_7);
lean_closure_set(x_20, 6, x_18);
lean_closure_set(x_20, 7, x_19);
lean_closure_set(x_20, 8, x_17);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg(x_4, x_5, x_8, x_10, x_11, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___redArg___lam__8(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
lean_inc(x_11);
lean_inc(x_1);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1), 3, 2);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_11);
lean_inc(x_10);
lean_inc(x_1);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_13, 0, x_1);
lean_closure_set(x_13, 1, x_10);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc(x_3);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__4___boxed), 6, 5);
lean_closure_set(x_14, 0, x_2);
lean_closure_set(x_14, 1, x_3);
lean_closure_set(x_14, 2, x_4);
lean_closure_set(x_14, 3, x_5);
lean_closure_set(x_14, 4, x_13);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc(x_7);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__0___boxed), 6, 5);
lean_closure_set(x_15, 0, x_6);
lean_closure_set(x_15, 1, x_7);
lean_closure_set(x_15, 2, x_4);
lean_closure_set(x_15, 3, x_5);
lean_closure_set(x_15, 4, x_14);
lean_inc_ref(x_5);
lean_inc(x_4);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__0___boxed), 6, 5);
lean_closure_set(x_16, 0, x_8);
lean_closure_set(x_16, 1, x_7);
lean_closure_set(x_16, 2, x_4);
lean_closure_set(x_16, 3, x_5);
lean_closure_set(x_16, 4, x_15);
x_17 = lean_apply_2(x_1, x_10, x_11);
x_18 = lp_mathlib_dotProduct___redArg(x_3, x_4, x_5, x_16, x_12);
lean_dec_ref(x_5);
x_19 = lean_apply_2(x_9, x_17, x_18);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
lean_inc_ref(x_3);
x_8 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_3);
x_9 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_8);
lean_inc_ref(x_9);
x_10 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_10, 1);
lean_inc(x_12);
lean_dec_ref(x_10);
x_13 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_13);
lean_dec_ref(x_9);
x_14 = lp_mathlib_Ring_toAddCommGroup___redArg(x_3);
x_15 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_14);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_15, 1);
lean_inc(x_16);
lean_dec_ref(x_15);
lean_inc(x_5);
lean_inc(x_2);
lean_inc(x_7);
lean_inc_ref(x_13);
lean_inc(x_11);
lean_inc(x_1);
lean_inc(x_4);
lean_inc(x_6);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___redArg___lam__8), 11, 9);
lean_closure_set(x_17, 0, x_6);
lean_closure_set(x_17, 1, x_4);
lean_closure_set(x_17, 2, x_1);
lean_closure_set(x_17, 3, x_11);
lean_closure_set(x_17, 4, x_13);
lean_closure_set(x_17, 5, x_7);
lean_closure_set(x_17, 6, x_2);
lean_closure_set(x_17, 7, x_5);
lean_closure_set(x_17, 8, x_12);
lean_inc(x_16);
lean_inc(x_2);
lean_inc_ref(x_13);
lean_inc(x_11);
lean_inc(x_1);
lean_inc(x_6);
lean_inc(x_7);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__5), 10, 8);
lean_closure_set(x_18, 0, x_7);
lean_closure_set(x_18, 1, x_6);
lean_closure_set(x_18, 2, x_4);
lean_closure_set(x_18, 3, x_1);
lean_closure_set(x_18, 4, x_11);
lean_closure_set(x_18, 5, x_13);
lean_closure_set(x_18, 6, x_2);
lean_closure_set(x_18, 7, x_16);
lean_inc(x_7);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__1), 10, 8);
lean_closure_set(x_19, 0, x_6);
lean_closure_set(x_19, 1, x_7);
lean_closure_set(x_19, 2, x_5);
lean_closure_set(x_19, 3, x_2);
lean_closure_set(x_19, 4, x_11);
lean_closure_set(x_19, 5, x_13);
lean_closure_set(x_19, 6, x_1);
lean_closure_set(x_19, 7, x_16);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, lean_box(0));
lean_closure_set(x_20, 2, lean_box(0));
lean_closure_set(x_20, 3, lean_box(0));
lean_closure_set(x_20, 4, lean_box(0));
lean_closure_set(x_20, 5, x_17);
lean_closure_set(x_20, 6, x_18);
lean_closure_set(x_20, 7, x_19);
lean_closure_set(x_20, 8, x_7);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___redArg(x_4, x_5, x_8, x_10, x_11, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = l_instDecidableEqSum_decEq___redArg(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__0(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1), 3, 2);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__3), 3, 2);
lean_closure_set(x_9, 0, x_2);
lean_closure_set(x_9, 1, x_6);
x_10 = lp_mathlib_dotProduct___redArg(x_3, x_4, x_5, x_9, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__6(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_9, 0, x_2);
lean_closure_set(x_9, 1, x_6);
x_10 = lp_mathlib_dotProduct___redArg(x_3, x_4, x_5, x_9, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__6___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__6(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__10(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
lean_inc(x_10);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_10);
lean_inc(x_9);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__3), 3, 2);
lean_closure_set(x_12, 0, x_2);
lean_closure_set(x_12, 1, x_9);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__0___boxed), 6, 5);
lean_closure_set(x_13, 0, x_3);
lean_closure_set(x_13, 1, x_4);
lean_closure_set(x_13, 2, x_5);
lean_closure_set(x_13, 3, x_6);
lean_closure_set(x_13, 4, x_12);
x_14 = lean_apply_2(x_7, x_9, x_10);
x_15 = lp_mathlib_dotProduct___redArg(x_4, x_5, x_6, x_13, x_11);
lean_dec_ref(x_6);
x_16 = lean_apply_2(x_8, x_14, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_fromBlocks___redArg(x_1, x_2, x_3, x_4, x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__7(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_fromBlocks___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__8(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__4), 6, 5);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_3);
lean_closure_set(x_14, 3, x_4);
lean_closure_set(x_14, 4, x_13);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__7), 6, 5);
lean_closure_set(x_15, 0, x_5);
lean_closure_set(x_15, 1, x_6);
lean_closure_set(x_15, 2, x_7);
lean_closure_set(x_15, 3, x_8);
lean_closure_set(x_15, 4, x_12);
x_16 = lp_mathlib_dotProduct___redArg(x_9, x_10, x_11, x_15, x_14);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__8___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__8(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_11);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; 
x_12 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_5);
x_13 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_5);
x_14 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_13);
lean_inc_ref(x_14);
x_15 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_14);
x_16 = lean_ctor_get(x_15, 1);
lean_inc(x_16);
lean_dec_ref(x_15);
lean_inc_ref(x_5);
x_17 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_5);
x_18 = lean_ctor_get(x_17, 1);
lean_inc_ref(x_18);
x_19 = lean_ctor_get(x_18, 2);
lean_inc(x_19);
lean_dec_ref(x_18);
lean_inc_ref(x_14);
x_20 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_14);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
x_22 = lean_ctor_get(x_14, 0);
lean_inc_ref(x_22);
lean_dec_ref(x_14);
lean_inc_ref(x_3);
lean_inc(x_1);
lean_inc_ref(x_12);
x_23 = lp_mathlib_Matrix_semiring___redArg(x_12, x_1, x_3);
x_24 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_23);
x_25 = lean_ctor_get(x_24, 0);
lean_inc_ref(x_25);
lean_dec_ref(x_24);
lean_inc_ref(x_4);
lean_inc(x_2);
lean_inc_ref(x_12);
x_26 = lp_mathlib_Matrix_semiring___redArg(x_12, x_2, x_4);
x_27 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_26);
x_28 = lean_ctor_get(x_27, 0);
lean_inc_ref(x_28);
lean_dec_ref(x_27);
x_29 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_17);
lean_dec_ref(x_17);
x_30 = lean_ctor_get(x_29, 2);
lean_inc(x_30);
lean_dec_ref(x_29);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
x_31 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_31, 0, x_3);
lean_closure_set(x_31, 1, x_4);
lean_inc(x_2);
lean_inc(x_1);
x_32 = lp_mathlib_Multiset_disjSum___redArg(x_1, x_2);
lean_inc(x_32);
lean_inc_ref(x_12);
x_33 = lp_mathlib_Matrix_semiring___redArg(x_12, x_32, x_31);
x_34 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_33);
x_35 = lean_ctor_get(x_12, 0);
x_36 = lean_ctor_get(x_34, 0);
lean_inc_ref(x_36);
lean_dec_ref(x_34);
x_37 = lean_ctor_get(x_35, 0);
lean_inc_ref(x_35);
x_38 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_35);
x_39 = lean_ctor_get(x_38, 0);
lean_inc(x_39);
lean_dec_ref(x_38);
lean_inc(x_16);
x_40 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_40, 0, x_16);
x_41 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5___boxed), 2, 1);
lean_closure_set(x_41, 0, x_19);
lean_inc_ref(x_22);
lean_inc(x_21);
lean_inc(x_2);
lean_inc(x_7);
lean_inc(x_10);
x_42 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__1___boxed), 7, 5);
lean_closure_set(x_42, 0, x_10);
lean_closure_set(x_42, 1, x_7);
lean_closure_set(x_42, 2, x_2);
lean_closure_set(x_42, 3, x_21);
lean_closure_set(x_42, 4, x_22);
lean_inc_ref(x_22);
lean_inc(x_21);
lean_inc(x_2);
lean_inc(x_10);
lean_inc(x_8);
x_43 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__6___boxed), 7, 5);
lean_closure_set(x_43, 0, x_8);
lean_closure_set(x_43, 1, x_10);
lean_closure_set(x_43, 2, x_2);
lean_closure_set(x_43, 3, x_21);
lean_closure_set(x_43, 4, x_22);
x_44 = lp_mathlib_invertibleOne___redArg(x_25);
lean_dec_ref(x_25);
x_45 = lp_mathlib_invertibleOne___redArg(x_28);
lean_dec_ref(x_28);
lean_inc(x_2);
x_46 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__10), 10, 8);
lean_closure_set(x_46, 0, x_8);
lean_closure_set(x_46, 1, x_7);
lean_closure_set(x_46, 2, x_10);
lean_closure_set(x_46, 3, x_2);
lean_closure_set(x_46, 4, x_21);
lean_closure_set(x_46, 5, x_22);
lean_closure_set(x_46, 6, x_6);
lean_closure_set(x_46, 7, x_30);
lean_inc_ref(x_41);
lean_inc(x_16);
x_47 = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonal), 7, 5);
lean_closure_set(x_47, 0, lean_box(0));
lean_closure_set(x_47, 1, lean_box(0));
lean_closure_set(x_47, 2, x_3);
lean_closure_set(x_47, 3, x_16);
lean_closure_set(x_47, 4, x_41);
x_48 = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonal), 7, 5);
lean_closure_set(x_48, 0, lean_box(0));
lean_closure_set(x_48, 1, lean_box(0));
lean_closure_set(x_48, 2, x_4);
lean_closure_set(x_48, 3, x_16);
lean_closure_set(x_48, 4, x_41);
lean_inc_ref(x_37);
lean_inc_ref(x_48);
lean_inc_ref(x_42);
lean_inc_ref(x_47);
lean_inc(x_9);
lean_inc_ref_n(x_40, 3);
lean_inc_ref(x_46);
x_49 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__8___boxed), 13, 11);
lean_closure_set(x_49, 0, x_46);
lean_closure_set(x_49, 1, x_40);
lean_closure_set(x_49, 2, x_40);
lean_closure_set(x_49, 3, x_9);
lean_closure_set(x_49, 4, x_47);
lean_closure_set(x_49, 5, x_42);
lean_closure_set(x_49, 6, x_40);
lean_closure_set(x_49, 7, x_48);
lean_closure_set(x_49, 8, x_32);
lean_closure_set(x_49, 9, x_39);
lean_closure_set(x_49, 10, x_37);
lean_inc_ref(x_48);
lean_inc_ref(x_43);
lean_inc_ref(x_40);
lean_inc_ref(x_47);
x_50 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_50, 0, lean_box(0));
lean_closure_set(x_50, 1, lean_box(0));
lean_closure_set(x_50, 2, lean_box(0));
lean_closure_set(x_50, 3, lean_box(0));
lean_closure_set(x_50, 4, lean_box(0));
lean_closure_set(x_50, 5, x_47);
lean_closure_set(x_50, 6, x_40);
lean_closure_set(x_50, 7, x_43);
lean_closure_set(x_50, 8, x_48);
lean_inc(x_45);
lean_inc(x_44);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_1);
x_51 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg(x_1, x_2, x_5, x_43, x_44, x_45);
lean_inc_ref(x_36);
x_52 = lp_mathlib_Invertible_mulRight___redArg(x_36, x_49, x_50, x_51);
x_53 = lp_mathlib_Equiv_symm___redArg(x_52);
x_54 = lean_ctor_get(x_53, 0);
lean_inc(x_54);
lean_dec_ref(x_53);
lean_inc_ref(x_40);
lean_inc_ref(x_42);
x_55 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_55, 0, lean_box(0));
lean_closure_set(x_55, 1, lean_box(0));
lean_closure_set(x_55, 2, lean_box(0));
lean_closure_set(x_55, 3, lean_box(0));
lean_closure_set(x_55, 4, lean_box(0));
lean_closure_set(x_55, 5, x_47);
lean_closure_set(x_55, 6, x_42);
lean_closure_set(x_55, 7, x_40);
lean_closure_set(x_55, 8, x_48);
x_56 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg(x_1, x_2, x_5, x_42, x_44, x_45);
lean_inc_ref(x_40);
x_57 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_57, 0, lean_box(0));
lean_closure_set(x_57, 1, lean_box(0));
lean_closure_set(x_57, 2, lean_box(0));
lean_closure_set(x_57, 3, lean_box(0));
lean_closure_set(x_57, 4, lean_box(0));
lean_closure_set(x_57, 5, x_46);
lean_closure_set(x_57, 6, x_40);
lean_closure_set(x_57, 7, x_40);
lean_closure_set(x_57, 8, x_9);
x_58 = lp_mathlib_Invertible_mulLeft___redArg(x_36, x_55, x_56, x_57);
x_59 = lp_mathlib_Equiv_symm___redArg(x_58);
x_60 = lean_ctor_get(x_59, 0);
lean_inc(x_60);
lean_dec_ref(x_59);
x_61 = lean_apply_1(x_54, x_11);
x_62 = lean_apply_1(x_60, x_61);
x_63 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___redArg(x_62);
x_64 = lean_ctor_get(x_63, 0);
lean_inc(x_64);
lean_dec_ref(x_63);
return x_64;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_15;
}
}
static lean_object* _init_lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumComm(lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___closed__0;
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___lam__0), 2, 1);
lean_closure_set(x_13, 0, x_12);
lean_inc_ref(x_13);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrix), 10, 8);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, lean_box(0));
lean_closure_set(x_14, 2, lean_box(0));
lean_closure_set(x_14, 3, lean_box(0));
lean_closure_set(x_14, 4, lean_box(0));
lean_closure_set(x_14, 5, x_11);
lean_closure_set(x_14, 6, x_13);
lean_closure_set(x_14, 7, x_13);
x_15 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg(x_2, x_1, x_4, x_3, x_5, x_9, x_8, x_7, x_6, x_10, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23) {
_start:
{
lean_inc(x_21);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__3___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__17(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_inc(x_12);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__17___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9_spec__9(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_inc(x_11);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9_spec__9___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3_spec__3___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_3, 0);
x_5 = lean_ctor_get(x_4, 0);
x_6 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3_spec__3___redArg___lam__0), 3, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = l_List_foldrTR___redArg(x_9, x_7, x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Multiset_map___redArg(x_3, x_2);
x_5 = lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12___redArg(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3_spec__3___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_3 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_1);
x_4 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_5);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3_spec__3___redArg___lam__0), 3, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = l_List_foldrTR___redArg(x_9, x_7, x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Multiset_map___redArg(x_3, x_2);
x_5 = lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3_spec__3___redArg(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6_spec__6___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Multiset_map___redArg(x_3, x_2);
x_5 = lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3_spec__3___redArg(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6_spec__6(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6_spec__6___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
lean_inc(x_4);
x_6 = lean_apply_2(x_1, x_4, x_5);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
lean_dec(x_4);
lean_dec(x_3);
lean_inc(x_2);
return x_2;
}
else
{
lean_object* x_8; 
x_8 = lean_apply_1(x_3, x_4);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_6 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_2);
x_7 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_6);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_7);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2___redArg___closed__0;
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_9);
lean_closure_set(x_12, 2, x_3);
x_13 = lean_apply_3(x_11, x_12, x_4, x_5);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2___redArg(x_1, x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
x_5 = lean_apply_1(x_1, x_4);
x_6 = lean_apply_1(x_2, x_4);
x_7 = lean_apply_2(x_3, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_2);
x_5 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_2);
x_6 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_5);
x_7 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg___lam__0), 4, 3);
lean_closure_set(x_9, 0, x_3);
lean_closure_set(x_9, 1, x_4);
lean_closure_set(x_9, 2, x_8);
x_10 = lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3___redArg(x_2, x_1, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__2), 3, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__3), 3, 2);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_5);
x_9 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_3, x_4, x_8, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_6);
x_8 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_7);
x_9 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_9);
lean_dec_ref(x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2___redArg___closed__0;
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_11);
lean_closure_set(x_14, 2, x_3);
x_15 = lean_apply_3(x_13, x_14, x_4, x_5);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg(x_1, x_2, x_3, x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__5(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__3), 3, 2);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_5);
x_7 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_2, x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
lean_inc_ref(x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__5___boxed), 5, 4);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_2);
lean_closure_set(x_12, 2, x_3);
lean_closure_set(x_12, 3, x_11);
lean_inc_ref(x_2);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__0___boxed), 5, 4);
lean_closure_set(x_13, 0, x_4);
lean_closure_set(x_13, 1, x_2);
lean_closure_set(x_13, 2, x_5);
lean_closure_set(x_13, 3, x_10);
lean_inc_ref(x_2);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__2), 5, 4);
lean_closure_set(x_14, 0, x_6);
lean_closure_set(x_14, 1, x_7);
lean_closure_set(x_14, 2, x_2);
lean_closure_set(x_14, 3, x_13);
x_15 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_8, x_2, x_14, x_12);
x_16 = lean_apply_1(x_9, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
lean_inc_ref(x_5);
x_9 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_5);
x_10 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_9);
x_11 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_10);
x_12 = lean_ctor_get(x_11, 1);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_5);
x_13 = lp_mathlib_Ring_toAddCommGroup___redArg(x_5);
x_14 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_13);
lean_dec_ref(x_13);
x_15 = lean_ctor_get(x_14, 1);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_5, 0);
x_17 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_16);
x_18 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_17);
x_19 = lean_ctor_get(x_18, 2);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_20, 0, x_12);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5___boxed), 2, 1);
lean_closure_set(x_21, 0, x_19);
lean_inc_ref(x_21);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_22 = lean_alloc_closure((void*)(lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1___boxed), 5, 3);
lean_closure_set(x_22, 0, x_4);
lean_closure_set(x_22, 1, x_5);
lean_closure_set(x_22, 2, x_21);
lean_inc_ref(x_21);
lean_inc_ref(x_5);
lean_inc_ref(x_3);
x_23 = lean_alloc_closure((void*)(lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1___boxed), 5, 3);
lean_closure_set(x_23, 0, x_3);
lean_closure_set(x_23, 1, x_5);
lean_closure_set(x_23, 2, x_21);
lean_inc_ref(x_21);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__1), 11, 9);
lean_closure_set(x_24, 0, x_4);
lean_closure_set(x_24, 1, x_5);
lean_closure_set(x_24, 2, x_21);
lean_closure_set(x_24, 3, x_3);
lean_closure_set(x_24, 4, x_21);
lean_closure_set(x_24, 5, x_7);
lean_closure_set(x_24, 6, x_1);
lean_closure_set(x_24, 7, x_2);
lean_closure_set(x_24, 8, x_15);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, lean_box(0));
lean_closure_set(x_25, 2, lean_box(0));
lean_closure_set(x_25, 3, lean_box(0));
lean_closure_set(x_25, 4, lean_box(0));
lean_closure_set(x_25, 5, x_23);
lean_closure_set(x_25, 6, x_24);
lean_closure_set(x_25, 7, x_20);
lean_closure_set(x_25, 8, x_22);
return x_25;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_11 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_12, 0, x_4);
lean_closure_set(x_12, 1, x_5);
lean_inc(x_3);
lean_inc(x_2);
x_13 = lp_mathlib_Multiset_disjSum___redArg(x_2, x_3);
lean_inc_ref(x_11);
x_14 = lp_mathlib_Matrix_semiring___redArg(x_11, x_13, x_12);
x_15 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_14);
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_17 = lean_ctor_get(x_15, 0);
x_18 = lean_ctor_get(x_15, 1);
lean_dec(x_18);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_19, 0, x_6);
lean_inc_ref(x_19);
lean_inc_ref(x_1);
lean_inc_ref(x_4);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_20, 0, x_4);
lean_closure_set(x_20, 1, x_1);
lean_closure_set(x_20, 2, x_19);
lean_inc_ref(x_1);
lean_inc(x_3);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__4), 6, 4);
lean_closure_set(x_21, 0, x_7);
lean_closure_set(x_21, 1, x_8);
lean_closure_set(x_21, 2, x_3);
lean_closure_set(x_21, 3, x_1);
lean_inc_ref(x_1);
lean_inc_ref(x_5);
x_22 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_22, 0, x_5);
lean_closure_set(x_22, 1, x_1);
lean_closure_set(x_22, 2, x_19);
x_23 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg(x_2, x_3, x_4, x_5, x_1, x_20, x_21, x_22);
lean_dec_ref(x_22);
lean_dec_ref(x_20);
lean_inc(x_10);
lean_inc(x_9);
lean_inc_ref(x_17);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___elam__0___boxed), 6, 5);
lean_closure_set(x_24, 0, lean_box(0));
lean_closure_set(x_24, 1, x_17);
lean_closure_set(x_24, 2, x_9);
lean_closure_set(x_24, 3, x_10);
lean_closure_set(x_24, 4, x_23);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___elam__1___boxed), 5, 4);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, x_17);
lean_closure_set(x_25, 2, x_9);
lean_closure_set(x_25, 3, x_10);
lean_ctor_set(x_15, 1, x_25);
lean_ctor_set(x_15, 0, x_24);
return x_15;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_26 = lean_ctor_get(x_15, 0);
lean_inc(x_26);
lean_dec(x_15);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_27, 0, x_6);
lean_inc_ref(x_27);
lean_inc_ref(x_1);
lean_inc_ref(x_4);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_28, 0, x_4);
lean_closure_set(x_28, 1, x_1);
lean_closure_set(x_28, 2, x_27);
lean_inc_ref(x_1);
lean_inc(x_3);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__4), 6, 4);
lean_closure_set(x_29, 0, x_7);
lean_closure_set(x_29, 1, x_8);
lean_closure_set(x_29, 2, x_3);
lean_closure_set(x_29, 3, x_1);
lean_inc_ref(x_1);
lean_inc_ref(x_5);
x_30 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_30, 0, x_5);
lean_closure_set(x_30, 1, x_1);
lean_closure_set(x_30, 2, x_27);
x_31 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg(x_2, x_3, x_4, x_5, x_1, x_28, x_29, x_30);
lean_dec_ref(x_30);
lean_dec_ref(x_28);
lean_inc(x_10);
lean_inc(x_9);
lean_inc_ref(x_26);
x_32 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___elam__0___boxed), 6, 5);
lean_closure_set(x_32, 0, lean_box(0));
lean_closure_set(x_32, 1, x_26);
lean_closure_set(x_32, 2, x_9);
lean_closure_set(x_32, 3, x_10);
lean_closure_set(x_32, 4, x_31);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___elam__1___boxed), 5, 4);
lean_closure_set(x_33, 0, lean_box(0));
lean_closure_set(x_33, 1, x_26);
lean_closure_set(x_33, 2, x_9);
lean_closure_set(x_33, 3, x_10);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_32);
lean_ctor_set(x_34, 1, x_33);
return x_34;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg___lam__4), 3, 2);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_5);
x_9 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_3, x_4, x_8, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___lam__8(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__3), 3, 2);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_5);
x_7 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_2, x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
lean_inc_ref(x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__5___boxed), 5, 4);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_2);
lean_closure_set(x_12, 2, x_3);
lean_closure_set(x_12, 3, x_11);
lean_inc_ref(x_2);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__0___boxed), 5, 4);
lean_closure_set(x_13, 0, x_4);
lean_closure_set(x_13, 1, x_2);
lean_closure_set(x_13, 2, x_5);
lean_closure_set(x_13, 3, x_10);
lean_inc_ref(x_2);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___lam__8), 5, 4);
lean_closure_set(x_14, 0, x_6);
lean_closure_set(x_14, 1, x_7);
lean_closure_set(x_14, 2, x_2);
lean_closure_set(x_14, 3, x_13);
x_15 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_8, x_2, x_14, x_12);
x_16 = lean_apply_1(x_9, x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
lean_inc_ref(x_5);
x_9 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_5);
x_10 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_9);
x_11 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_10);
x_12 = lean_ctor_get(x_11, 1);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_5);
x_13 = lp_mathlib_Ring_toAddCommGroup___redArg(x_5);
x_14 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_13);
lean_dec_ref(x_13);
x_15 = lean_ctor_get(x_14, 1);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_5, 0);
x_17 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_16);
x_18 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_17);
x_19 = lean_ctor_get(x_18, 2);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_20, 0, x_12);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5___boxed), 2, 1);
lean_closure_set(x_21, 0, x_19);
lean_inc_ref(x_21);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_22 = lean_alloc_closure((void*)(lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1___boxed), 5, 3);
lean_closure_set(x_22, 0, x_4);
lean_closure_set(x_22, 1, x_5);
lean_closure_set(x_22, 2, x_21);
lean_inc_ref(x_21);
lean_inc_ref(x_5);
lean_inc_ref(x_3);
x_23 = lean_alloc_closure((void*)(lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1___boxed), 5, 3);
lean_closure_set(x_23, 0, x_3);
lean_closure_set(x_23, 1, x_5);
lean_closure_set(x_23, 2, x_21);
lean_inc_ref(x_21);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___lam__0), 11, 9);
lean_closure_set(x_24, 0, x_3);
lean_closure_set(x_24, 1, x_5);
lean_closure_set(x_24, 2, x_21);
lean_closure_set(x_24, 3, x_4);
lean_closure_set(x_24, 4, x_21);
lean_closure_set(x_24, 5, x_7);
lean_closure_set(x_24, 6, x_2);
lean_closure_set(x_24, 7, x_1);
lean_closure_set(x_24, 8, x_15);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, lean_box(0));
lean_closure_set(x_25, 2, lean_box(0));
lean_closure_set(x_25, 3, lean_box(0));
lean_closure_set(x_25, 4, lean_box(0));
lean_closure_set(x_25, 5, x_23);
lean_closure_set(x_25, 6, x_20);
lean_closure_set(x_25, 7, x_24);
lean_closure_set(x_25, 8, x_22);
return x_25;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_11 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_12, 0, x_4);
lean_closure_set(x_12, 1, x_5);
lean_inc(x_3);
lean_inc(x_2);
x_13 = lp_mathlib_Multiset_disjSum___redArg(x_2, x_3);
lean_inc_ref(x_11);
x_14 = lp_mathlib_Matrix_semiring___redArg(x_11, x_13, x_12);
x_15 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_14);
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_17 = lean_ctor_get(x_15, 0);
x_18 = lean_ctor_get(x_15, 1);
lean_dec(x_18);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_19, 0, x_6);
lean_inc_ref(x_19);
lean_inc_ref(x_1);
lean_inc_ref(x_4);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_20, 0, x_4);
lean_closure_set(x_20, 1, x_1);
lean_closure_set(x_20, 2, x_19);
lean_inc_ref(x_1);
lean_inc(x_3);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg___lam__0), 6, 4);
lean_closure_set(x_21, 0, x_7);
lean_closure_set(x_21, 1, x_8);
lean_closure_set(x_21, 2, x_3);
lean_closure_set(x_21, 3, x_1);
lean_inc_ref(x_1);
lean_inc_ref(x_5);
x_22 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_22, 0, x_5);
lean_closure_set(x_22, 1, x_1);
lean_closure_set(x_22, 2, x_19);
x_23 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg(x_2, x_3, x_4, x_5, x_1, x_20, x_21, x_22);
lean_dec_ref(x_22);
lean_dec_ref(x_20);
lean_inc(x_10);
lean_inc(x_9);
lean_inc_ref(x_17);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulRight___elam__0___boxed), 6, 5);
lean_closure_set(x_24, 0, lean_box(0));
lean_closure_set(x_24, 1, x_17);
lean_closure_set(x_24, 2, x_9);
lean_closure_set(x_24, 3, x_10);
lean_closure_set(x_24, 4, x_23);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulRight___elam__1___boxed), 5, 4);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, x_17);
lean_closure_set(x_25, 2, x_10);
lean_closure_set(x_25, 3, x_9);
lean_ctor_set(x_15, 1, x_25);
lean_ctor_set(x_15, 0, x_24);
return x_15;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_26 = lean_ctor_get(x_15, 0);
lean_inc(x_26);
lean_dec(x_15);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_27, 0, x_6);
lean_inc_ref(x_27);
lean_inc_ref(x_1);
lean_inc_ref(x_4);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_28, 0, x_4);
lean_closure_set(x_28, 1, x_1);
lean_closure_set(x_28, 2, x_27);
lean_inc_ref(x_1);
lean_inc(x_3);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg___lam__0), 6, 4);
lean_closure_set(x_29, 0, x_7);
lean_closure_set(x_29, 1, x_8);
lean_closure_set(x_29, 2, x_3);
lean_closure_set(x_29, 3, x_1);
lean_inc_ref(x_1);
lean_inc_ref(x_5);
x_30 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_30, 0, x_5);
lean_closure_set(x_30, 1, x_1);
lean_closure_set(x_30, 2, x_27);
x_31 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg(x_2, x_3, x_4, x_5, x_1, x_28, x_29, x_30);
lean_dec_ref(x_30);
lean_dec_ref(x_28);
lean_inc(x_10);
lean_inc(x_9);
lean_inc_ref(x_26);
x_32 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulRight___elam__0___boxed), 6, 5);
lean_closure_set(x_32, 0, lean_box(0));
lean_closure_set(x_32, 1, x_26);
lean_closure_set(x_32, 2, x_9);
lean_closure_set(x_32, 3, x_10);
lean_closure_set(x_32, 4, x_31);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulRight___elam__1___boxed), 5, 4);
lean_closure_set(x_33, 0, lean_box(0));
lean_closure_set(x_33, 1, x_26);
lean_closure_set(x_33, 2, x_10);
lean_closure_set(x_33, 3, x_9);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_32);
lean_ctor_set(x_34, 1, x_33);
return x_34;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2___redArg(x_3, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg(x_3, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1), 3, 2);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg___lam__2), 3, 2);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_8);
lean_inc_ref(x_5);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__2), 5, 4);
lean_closure_set(x_12, 0, x_3);
lean_closure_set(x_12, 1, x_4);
lean_closure_set(x_12, 2, x_5);
lean_closure_set(x_12, 3, x_11);
x_13 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_6, x_5, x_12, x_10);
x_14 = lean_apply_1(x_7, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
lean_inc_ref(x_5);
x_11 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_5);
x_12 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_11);
x_13 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_12);
x_14 = lean_ctor_get(x_13, 1);
lean_inc(x_14);
lean_dec_ref(x_13);
lean_inc_ref(x_5);
x_15 = lp_mathlib_Ring_toAddCommGroup___redArg(x_5);
x_16 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_15);
lean_dec_ref(x_15);
x_17 = lean_ctor_get(x_16, 1);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_18, 0, x_14);
lean_inc(x_6);
lean_inc(x_7);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg___lam__3), 9, 7);
lean_closure_set(x_19, 0, x_7);
lean_closure_set(x_19, 1, x_6);
lean_closure_set(x_19, 2, x_9);
lean_closure_set(x_19, 3, x_1);
lean_closure_set(x_19, 4, x_5);
lean_closure_set(x_19, 5, x_2);
lean_closure_set(x_19, 6, x_17);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, lean_box(0));
lean_closure_set(x_20, 2, lean_box(0));
lean_closure_set(x_20, 3, lean_box(0));
lean_closure_set(x_20, 4, lean_box(0));
lean_closure_set(x_20, 5, x_6);
lean_closure_set(x_20, 6, x_19);
lean_closure_set(x_20, 7, x_18);
lean_closure_set(x_20, 8, x_7);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__0), 3, 2);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_8);
lean_inc_ref(x_5);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___lam__8), 5, 4);
lean_closure_set(x_12, 0, x_3);
lean_closure_set(x_12, 1, x_4);
lean_closure_set(x_12, 2, x_5);
lean_closure_set(x_12, 3, x_11);
x_13 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_6, x_5, x_12, x_10);
x_14 = lean_apply_1(x_7, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__11(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__11), 3, 2);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_5);
x_7 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_2, x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_inc(x_10);
lean_inc(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1), 3, 2);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_10);
lean_inc(x_9);
lean_inc(x_1);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_9);
lean_inc_ref(x_4);
lean_inc(x_3);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___lam__8), 5, 4);
lean_closure_set(x_13, 0, x_2);
lean_closure_set(x_13, 1, x_3);
lean_closure_set(x_13, 2, x_4);
lean_closure_set(x_13, 3, x_12);
lean_inc_ref(x_4);
lean_inc(x_6);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__1), 5, 4);
lean_closure_set(x_14, 0, x_5);
lean_closure_set(x_14, 1, x_6);
lean_closure_set(x_14, 2, x_4);
lean_closure_set(x_14, 3, x_13);
lean_inc_ref(x_4);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__2), 5, 4);
lean_closure_set(x_15, 0, x_7);
lean_closure_set(x_15, 1, x_6);
lean_closure_set(x_15, 2, x_4);
lean_closure_set(x_15, 3, x_14);
x_16 = lean_apply_2(x_1, x_9, x_10);
x_17 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_3, x_4, x_15, x_11);
x_18 = lean_apply_2(x_8, x_16, x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
lean_inc_ref(x_5);
x_12 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_5);
x_13 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_12);
x_14 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_13);
lean_inc_ref(x_5);
x_15 = lp_mathlib_Ring_toAddCommGroup___redArg(x_5);
x_16 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_15);
lean_dec_ref(x_15);
x_17 = lean_ctor_get(x_16, 1);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_14, 1);
lean_inc(x_18);
lean_dec_ref(x_14);
lean_inc(x_17);
lean_inc(x_1);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_10);
lean_inc(x_6);
lean_inc(x_7);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__4), 9, 7);
lean_closure_set(x_19, 0, x_7);
lean_closure_set(x_19, 1, x_6);
lean_closure_set(x_19, 2, x_10);
lean_closure_set(x_19, 3, x_2);
lean_closure_set(x_19, 4, x_5);
lean_closure_set(x_19, 5, x_1);
lean_closure_set(x_19, 6, x_17);
lean_inc(x_2);
lean_inc_ref(x_5);
lean_inc(x_1);
lean_inc(x_9);
lean_inc(x_7);
lean_inc(x_6);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg___lam__3), 9, 7);
lean_closure_set(x_20, 0, x_6);
lean_closure_set(x_20, 1, x_7);
lean_closure_set(x_20, 2, x_9);
lean_closure_set(x_20, 3, x_1);
lean_closure_set(x_20, 4, x_5);
lean_closure_set(x_20, 5, x_2);
lean_closure_set(x_20, 6, x_17);
lean_inc(x_7);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__5), 10, 8);
lean_closure_set(x_21, 0, x_6);
lean_closure_set(x_21, 1, x_10);
lean_closure_set(x_21, 2, x_2);
lean_closure_set(x_21, 3, x_5);
lean_closure_set(x_21, 4, x_7);
lean_closure_set(x_21, 5, x_1);
lean_closure_set(x_21, 6, x_9);
lean_closure_set(x_21, 7, x_18);
x_22 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_22, 0, lean_box(0));
lean_closure_set(x_22, 1, lean_box(0));
lean_closure_set(x_22, 2, lean_box(0));
lean_closure_set(x_22, 3, lean_box(0));
lean_closure_set(x_22, 4, lean_box(0));
lean_closure_set(x_22, 5, x_7);
lean_closure_set(x_22, 6, x_20);
lean_closure_set(x_22, 7, x_19);
lean_closure_set(x_22, 8, x_21);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_5);
x_9 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_3, x_4, x_8, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1), 3, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__3), 3, 2);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_5);
x_9 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_3, x_4, x_8, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__13(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__4), 3, 2);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_5);
x_7 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_2, x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_inc(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_9);
lean_inc(x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__3), 3, 2);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_8);
lean_inc_ref(x_5);
lean_inc(x_4);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__13), 5, 4);
lean_closure_set(x_12, 0, x_3);
lean_closure_set(x_12, 1, x_4);
lean_closure_set(x_12, 2, x_5);
lean_closure_set(x_12, 3, x_11);
x_13 = lean_apply_2(x_6, x_8, x_9);
x_14 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_4, x_5, x_12, x_10);
x_15 = lean_apply_2(x_7, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_fromBlocks___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_4);
x_5 = lean_apply_1(x_1, x_4);
x_6 = lean_apply_1(x_2, x_4);
x_7 = lean_apply_2(x_3, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lean_ctor_get(x_3, 0);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6___redArg___lam__0), 4, 3);
lean_closure_set(x_10, 0, x_4);
lean_closure_set(x_10, 1, x_5);
lean_closure_set(x_10, 2, x_9);
x_11 = lp_mathlib_Multiset_disjSum___redArg(x_1, x_2);
x_12 = lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12___redArg(x_3, x_11, x_10);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__4), 6, 5);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_3);
lean_closure_set(x_14, 3, x_4);
lean_closure_set(x_14, 4, x_13);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__4), 6, 5);
lean_closure_set(x_15, 0, x_5);
lean_closure_set(x_15, 1, x_6);
lean_closure_set(x_15, 2, x_7);
lean_closure_set(x_15, 3, x_8);
lean_closure_set(x_15, 4, x_12);
x_16 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12___redArg(x_9, x_10, x_11, x_15, x_14);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; uint8_t x_18; 
x_17 = lean_ctor_get(x_6, 0);
lean_inc(x_17);
lean_dec_ref(x_6);
x_18 = !lean_is_exclusive(x_7);
if (x_18 == 0)
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_19 = lean_ctor_get(x_7, 0);
x_20 = lean_ctor_get(x_7, 1);
lean_dec(x_20);
x_21 = lean_apply_1(x_19, x_8);
x_22 = lean_apply_1(x_17, x_21);
lean_inc(x_22);
x_23 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2081_u2081), 8, 6);
lean_closure_set(x_23, 0, lean_box(0));
lean_closure_set(x_23, 1, lean_box(0));
lean_closure_set(x_23, 2, lean_box(0));
lean_closure_set(x_23, 3, lean_box(0));
lean_closure_set(x_23, 4, lean_box(0));
lean_closure_set(x_23, 5, x_22);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2082_u2082), 8, 6);
lean_closure_set(x_24, 0, lean_box(0));
lean_closure_set(x_24, 1, lean_box(0));
lean_closure_set(x_24, 2, lean_box(0));
lean_closure_set(x_24, 3, lean_box(0));
lean_closure_set(x_24, 4, lean_box(0));
lean_closure_set(x_24, 5, x_22);
lean_ctor_set(x_7, 1, x_24);
lean_ctor_set(x_7, 0, x_23);
return x_7;
}
else
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_25 = lean_ctor_get(x_7, 0);
lean_inc(x_25);
lean_dec(x_7);
x_26 = lean_apply_1(x_25, x_8);
x_27 = lean_apply_1(x_17, x_26);
lean_inc(x_27);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2081_u2081), 8, 6);
lean_closure_set(x_28, 0, lean_box(0));
lean_closure_set(x_28, 1, lean_box(0));
lean_closure_set(x_28, 2, lean_box(0));
lean_closure_set(x_28, 3, lean_box(0));
lean_closure_set(x_28, 4, lean_box(0));
lean_closure_set(x_28, 5, x_27);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2082_u2082), 8, 6);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, lean_box(0));
lean_closure_set(x_29, 2, lean_box(0));
lean_closure_set(x_29, 3, lean_box(0));
lean_closure_set(x_29, 4, lean_box(0));
lean_closure_set(x_29, 5, x_27);
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_28);
lean_ctor_set(x_30, 1, x_29);
return x_30;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
lean_inc_ref(x_5);
x_12 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_5);
x_13 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_12);
x_14 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_13);
x_15 = lean_ctor_get(x_14, 1);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc_ref(x_5);
x_16 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_5);
x_17 = lean_ctor_get(x_16, 1);
lean_inc_ref(x_17);
x_18 = lean_ctor_get(x_17, 2);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_16);
lean_dec_ref(x_16);
x_20 = lean_ctor_get(x_19, 2);
lean_inc(x_20);
lean_dec_ref(x_19);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_6);
lean_inc(x_10);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__2), 6, 4);
lean_closure_set(x_21, 0, x_10);
lean_closure_set(x_21, 1, x_6);
lean_closure_set(x_21, 2, x_2);
lean_closure_set(x_21, 3, x_5);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_9);
lean_inc(x_6);
x_22 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__3), 6, 4);
lean_closure_set(x_22, 0, x_6);
lean_closure_set(x_22, 1, x_9);
lean_closure_set(x_22, 2, x_2);
lean_closure_set(x_22, 3, x_5);
lean_inc(x_15);
x_23 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_23, 0, x_15);
lean_inc(x_18);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5___boxed), 2, 1);
lean_closure_set(x_24, 0, x_18);
lean_inc_ref(x_24);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_25, 0, x_4);
lean_closure_set(x_25, 1, x_5);
lean_closure_set(x_25, 2, x_24);
lean_inc_ref(x_5);
lean_inc_ref(x_3);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_26, 0, x_3);
lean_closure_set(x_26, 1, x_5);
lean_closure_set(x_26, 2, x_24);
lean_inc(x_20);
lean_inc(x_8);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_6);
lean_inc(x_9);
lean_inc(x_10);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__0), 9, 7);
lean_closure_set(x_27, 0, x_10);
lean_closure_set(x_27, 1, x_9);
lean_closure_set(x_27, 2, x_6);
lean_closure_set(x_27, 3, x_2);
lean_closure_set(x_27, 4, x_5);
lean_closure_set(x_27, 5, x_8);
lean_closure_set(x_27, 6, x_20);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_1);
lean_inc_ref(x_25);
lean_inc_ref(x_22);
lean_inc_ref(x_26);
lean_inc(x_11);
lean_inc_ref_n(x_23, 3);
lean_inc_ref(x_27);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__1), 13, 11);
lean_closure_set(x_28, 0, x_27);
lean_closure_set(x_28, 1, x_23);
lean_closure_set(x_28, 2, x_23);
lean_closure_set(x_28, 3, x_11);
lean_closure_set(x_28, 4, x_26);
lean_closure_set(x_28, 5, x_22);
lean_closure_set(x_28, 6, x_23);
lean_closure_set(x_28, 7, x_25);
lean_closure_set(x_28, 8, x_1);
lean_closure_set(x_28, 9, x_2);
lean_closure_set(x_28, 10, x_5);
lean_inc_ref(x_25);
lean_inc_ref(x_23);
lean_inc_ref(x_26);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_29, 0, lean_box(0));
lean_closure_set(x_29, 1, lean_box(0));
lean_closure_set(x_29, 2, lean_box(0));
lean_closure_set(x_29, 3, lean_box(0));
lean_closure_set(x_29, 4, lean_box(0));
lean_closure_set(x_29, 5, x_26);
lean_closure_set(x_29, 6, x_22);
lean_closure_set(x_29, 7, x_23);
lean_closure_set(x_29, 8, x_25);
lean_inc(x_11);
lean_inc_ref_n(x_23, 2);
x_30 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, lean_box(0));
lean_closure_set(x_30, 2, lean_box(0));
lean_closure_set(x_30, 3, lean_box(0));
lean_closure_set(x_30, 4, lean_box(0));
lean_closure_set(x_30, 5, x_27);
lean_closure_set(x_30, 6, x_23);
lean_closure_set(x_30, 7, x_23);
lean_closure_set(x_30, 8, x_11);
lean_inc(x_9);
lean_inc(x_6);
lean_inc(x_18);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc(x_2);
lean_inc(x_1);
lean_inc_ref(x_5);
x_31 = lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg(x_5, x_1, x_2, x_3, x_4, x_18, x_6, x_9, x_29, x_30);
x_32 = lp_mathlib_Equiv_symm___redArg(x_31);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_33, 0, lean_box(0));
lean_closure_set(x_33, 1, lean_box(0));
lean_closure_set(x_33, 2, lean_box(0));
lean_closure_set(x_33, 3, lean_box(0));
lean_closure_set(x_33, 4, lean_box(0));
lean_closure_set(x_33, 5, x_26);
lean_closure_set(x_33, 6, x_23);
lean_closure_set(x_33, 7, x_21);
lean_closure_set(x_33, 8, x_25);
lean_inc(x_6);
lean_inc(x_10);
lean_inc(x_18);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc(x_2);
lean_inc(x_1);
lean_inc_ref(x_5);
x_34 = lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg(x_5, x_1, x_2, x_3, x_4, x_18, x_10, x_6, x_28, x_33);
x_35 = lp_mathlib_Equiv_symm___redArg(x_34);
x_36 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18___redArg(x_1, x_2, x_3, x_4, x_5, x_32, x_35, x_7, x_8, x_9, x_10, x_11, x_18, x_15, x_6, x_20);
lean_dec(x_20);
lean_dec(x_6);
lean_dec(x_15);
lean_dec(x_18);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_37 = lean_ctor_get(x_36, 0);
lean_inc(x_37);
lean_dec_ref(x_36);
return x_37;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_13, x_14, x_9, x_10, x_11, x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_13, x_14, x_9, x_10, x_11, x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc(x_2);
lean_inc(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0), 14, 13);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, lean_box(0));
lean_closure_set(x_11, 2, lean_box(0));
lean_closure_set(x_11, 3, x_1);
lean_closure_set(x_11, 4, x_2);
lean_closure_set(x_11, 5, x_3);
lean_closure_set(x_11, 6, x_4);
lean_closure_set(x_11, 7, x_5);
lean_closure_set(x_11, 8, x_6);
lean_closure_set(x_11, 9, x_7);
lean_closure_set(x_11, 10, x_8);
lean_closure_set(x_11, 11, x_9);
lean_closure_set(x_11, 12, x_10);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1___boxed), 14, 13);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, lean_box(0));
lean_closure_set(x_12, 2, lean_box(0));
lean_closure_set(x_12, 3, x_1);
lean_closure_set(x_12, 4, x_2);
lean_closure_set(x_12, 5, x_3);
lean_closure_set(x_12, 6, x_4);
lean_closure_set(x_12, 7, x_5);
lean_closure_set(x_12, 8, x_6);
lean_closure_set(x_12, 9, x_7);
lean_closure_set(x_12, 10, x_8);
lean_closure_set(x_12, 11, x_9);
lean_closure_set(x_12, 12, x_10);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_10, x_11, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_10, x_11, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22) {
_start:
{
lean_object* x_23; 
x_23 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
return x_23;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3_spec__3_spec__3___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_3);
x_6 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_3);
x_7 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_6);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6___redArg___lam__0), 4, 3);
lean_closure_set(x_10, 0, x_4);
lean_closure_set(x_10, 1, x_5);
lean_closure_set(x_10, 2, x_9);
x_11 = lp_mathlib_Multiset_disjSum___redArg(x_1, x_2);
x_12 = lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6_spec__6___redArg(x_3, x_11, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__8___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_3);
x_6 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_3);
x_7 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_6);
x_8 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6___redArg___lam__0), 4, 3);
lean_closure_set(x_10, 0, x_4);
lean_closure_set(x_10, 1, x_5);
lean_closure_set(x_10, 2, x_9);
x_11 = lp_mathlib_Multiset_disjSum___redArg(x_1, x_2);
x_12 = lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__6_spec__6___redArg(x_3, x_11, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__8(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__8___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_1, x_2, x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__5(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1), 3, 2);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg___lam__2), 3, 2);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_8);
lean_inc_ref(x_4);
x_12 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__5___boxed), 5, 4);
lean_closure_set(x_12, 0, x_3);
lean_closure_set(x_12, 1, x_4);
lean_closure_set(x_12, 2, x_11);
lean_closure_set(x_12, 3, x_5);
x_13 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_6, x_4, x_12, x_10);
x_14 = lean_apply_1(x_7, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__9(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__2), 3, 2);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_5);
x_7 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_2, x_3, x_6, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__9), 3, 2);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_7);
lean_inc_ref(x_4);
x_9 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__3), 5, 4);
lean_closure_set(x_9, 0, x_2);
lean_closure_set(x_9, 1, x_3);
lean_closure_set(x_9, 2, x_4);
lean_closure_set(x_9, 3, x_8);
x_10 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_5, x_4, x_6, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__6(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_inc_ref(x_2);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__5___boxed), 5, 4);
lean_closure_set(x_13, 0, x_1);
lean_closure_set(x_13, 1, x_2);
lean_closure_set(x_13, 2, x_3);
lean_closure_set(x_13, 3, x_12);
lean_inc_ref(x_2);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__0___boxed), 5, 4);
lean_closure_set(x_14, 0, x_4);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_5);
lean_closure_set(x_14, 3, x_11);
lean_inc_ref(x_2);
lean_inc(x_8);
x_15 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__4), 7, 6);
lean_closure_set(x_15, 0, x_6);
lean_closure_set(x_15, 1, x_7);
lean_closure_set(x_15, 2, x_8);
lean_closure_set(x_15, 3, x_2);
lean_closure_set(x_15, 4, x_9);
lean_closure_set(x_15, 5, x_14);
x_16 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_8, x_2, x_15, x_13);
x_17 = lean_apply_1(x_10, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__7(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_fromBlocks___redArg(x_1, x_2, x_3, x_4, x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__8(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_fromBlocks___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__10(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__7), 6, 5);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_3);
lean_closure_set(x_14, 3, x_4);
lean_closure_set(x_14, 4, x_13);
x_15 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__8), 6, 5);
lean_closure_set(x_15, 0, x_5);
lean_closure_set(x_15, 1, x_6);
lean_closure_set(x_15, 2, x_7);
lean_closure_set(x_15, 3, x_8);
lean_closure_set(x_15, 4, x_12);
x_16 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12___redArg(x_9, x_10, x_11, x_15, x_14);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_11 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_1);
x_12 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_1);
x_13 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_12);
x_14 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_13);
x_15 = lean_ctor_get(x_14, 1);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc_ref(x_1);
x_16 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_17 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_16);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_17, 1);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_11);
x_20 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_19);
x_21 = lean_ctor_get(x_20, 2);
lean_inc(x_21);
lean_dec_ref(x_20);
x_22 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_22, 0, x_10);
x_23 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_23, 0, x_15);
lean_inc(x_18);
lean_inc(x_3);
lean_inc_ref(x_1);
lean_inc(x_2);
lean_inc(x_8);
lean_inc(x_9);
x_24 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__1), 9, 7);
lean_closure_set(x_24, 0, x_9);
lean_closure_set(x_24, 1, x_8);
lean_closure_set(x_24, 2, x_2);
lean_closure_set(x_24, 3, x_1);
lean_closure_set(x_24, 4, x_22);
lean_closure_set(x_24, 5, x_3);
lean_closure_set(x_24, 6, x_18);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5___boxed), 2, 1);
lean_closure_set(x_25, 0, x_21);
lean_inc_ref(x_25);
lean_inc_ref(x_1);
lean_inc_ref(x_5);
x_26 = lean_alloc_closure((void*)(lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1___boxed), 5, 3);
lean_closure_set(x_26, 0, x_5);
lean_closure_set(x_26, 1, x_1);
lean_closure_set(x_26, 2, x_25);
lean_inc_ref(x_25);
lean_inc_ref(x_1);
lean_inc_ref(x_4);
x_27 = lean_alloc_closure((void*)(lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1___boxed), 5, 3);
lean_closure_set(x_27, 0, x_4);
lean_closure_set(x_27, 1, x_1);
lean_closure_set(x_27, 2, x_25);
lean_inc(x_2);
lean_inc(x_3);
lean_inc_ref(x_25);
lean_inc_ref(x_1);
x_28 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__6), 12, 10);
lean_closure_set(x_28, 0, x_5);
lean_closure_set(x_28, 1, x_1);
lean_closure_set(x_28, 2, x_25);
lean_closure_set(x_28, 3, x_4);
lean_closure_set(x_28, 4, x_25);
lean_closure_set(x_28, 5, x_6);
lean_closure_set(x_28, 6, x_7);
lean_closure_set(x_28, 7, x_3);
lean_closure_set(x_28, 8, x_2);
lean_closure_set(x_28, 9, x_18);
lean_inc_ref(x_23);
x_29 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__10), 13, 11);
lean_closure_set(x_29, 0, x_27);
lean_closure_set(x_29, 1, x_28);
lean_closure_set(x_29, 2, x_23);
lean_closure_set(x_29, 3, x_26);
lean_closure_set(x_29, 4, x_8);
lean_closure_set(x_29, 5, x_24);
lean_closure_set(x_29, 6, x_23);
lean_closure_set(x_29, 7, x_9);
lean_closure_set(x_29, 8, x_2);
lean_closure_set(x_29, 9, x_3);
lean_closure_set(x_29, 10, x_1);
return x_29;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20) {
_start:
{
lean_object* x_21; 
x_21 = lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg(x_4, x_5, x_6, x_7, x_8, x_10, x_11, x_12, x_13, x_17);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__2(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__14(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___redArg___lam__4), 3, 2);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_7);
lean_inc_ref(x_4);
x_9 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__3), 5, 4);
lean_closure_set(x_9, 0, x_2);
lean_closure_set(x_9, 1, x_3);
lean_closure_set(x_9, 2, x_4);
lean_closure_set(x_9, 3, x_8);
x_10 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_5, x_4, x_6, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_inc_ref(x_2);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__5___boxed), 5, 4);
lean_closure_set(x_13, 0, x_1);
lean_closure_set(x_13, 1, x_2);
lean_closure_set(x_13, 2, x_3);
lean_closure_set(x_13, 3, x_12);
lean_inc_ref(x_2);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__0___boxed), 5, 4);
lean_closure_set(x_14, 0, x_4);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_5);
lean_closure_set(x_14, 3, x_11);
lean_inc_ref(x_2);
lean_inc(x_8);
x_15 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__14), 7, 6);
lean_closure_set(x_15, 0, x_6);
lean_closure_set(x_15, 1, x_7);
lean_closure_set(x_15, 2, x_8);
lean_closure_set(x_15, 3, x_2);
lean_closure_set(x_15, 4, x_9);
lean_closure_set(x_15, 5, x_14);
x_16 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_8, x_2, x_15, x_13);
x_17 = lean_apply_1(x_10, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__10(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__10), 3, 2);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_5);
x_7 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_2, x_3, x_6, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___redArg___lam__3), 3, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_6);
lean_inc_ref(x_4);
lean_inc(x_3);
x_8 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__1), 5, 4);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_3);
lean_closure_set(x_8, 2, x_4);
lean_closure_set(x_8, 3, x_7);
x_9 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_3, x_4, x_5, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_inc_ref(x_2);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__5___boxed), 5, 4);
lean_closure_set(x_13, 0, x_1);
lean_closure_set(x_13, 1, x_2);
lean_closure_set(x_13, 2, x_3);
lean_closure_set(x_13, 3, x_12);
lean_inc_ref(x_2);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__0___boxed), 5, 4);
lean_closure_set(x_14, 0, x_4);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_5);
lean_closure_set(x_14, 3, x_11);
lean_inc_ref(x_2);
x_15 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__3), 6, 5);
lean_closure_set(x_15, 0, x_6);
lean_closure_set(x_15, 1, x_7);
lean_closure_set(x_15, 2, x_8);
lean_closure_set(x_15, 3, x_2);
lean_closure_set(x_15, 4, x_14);
x_16 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_9, x_2, x_15, x_13);
x_17 = lean_apply_1(x_10, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__6(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_fromBlocks___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__6), 6, 5);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_2);
lean_closure_set(x_10, 2, x_3);
lean_closure_set(x_10, 3, x_4);
lean_closure_set(x_10, 4, x_9);
x_11 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12___redArg(x_5, x_6, x_7, x_10, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__8(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_18 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__7), 6, 5);
lean_closure_set(x_18, 0, x_1);
lean_closure_set(x_18, 1, x_2);
lean_closure_set(x_18, 2, x_3);
lean_closure_set(x_18, 3, x_4);
lean_closure_set(x_18, 4, x_17);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc(x_9);
x_19 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__5), 9, 8);
lean_closure_set(x_19, 0, x_5);
lean_closure_set(x_19, 1, x_6);
lean_closure_set(x_19, 2, x_7);
lean_closure_set(x_19, 3, x_8);
lean_closure_set(x_19, 4, x_9);
lean_closure_set(x_19, 5, x_10);
lean_closure_set(x_19, 6, x_11);
lean_closure_set(x_19, 7, x_18);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__4), 6, 5);
lean_closure_set(x_20, 0, x_12);
lean_closure_set(x_20, 1, x_13);
lean_closure_set(x_20, 2, x_14);
lean_closure_set(x_20, 3, x_15);
lean_closure_set(x_20, 4, x_16);
x_21 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12___redArg(x_9, x_10, x_11, x_20, x_19);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__8___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__8(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_1);
x_11 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_1);
x_12 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_11);
x_13 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_12);
x_14 = lean_ctor_get(x_13, 1);
lean_inc(x_14);
lean_dec_ref(x_13);
lean_inc_ref(x_1);
x_15 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_16 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_15);
lean_dec_ref(x_15);
x_17 = lean_ctor_get(x_16, 1);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_10);
x_19 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_18);
x_20 = lean_ctor_get(x_19, 2);
lean_inc(x_20);
lean_dec_ref(x_19);
lean_inc(x_14);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_21, 0, x_14);
x_22 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__2___boxed), 2, 1);
lean_closure_set(x_22, 0, x_14);
lean_inc(x_17);
lean_inc(x_3);
lean_inc_ref(x_1);
lean_inc(x_2);
lean_inc(x_8);
lean_inc(x_6);
x_23 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___redArg___lam__1), 9, 7);
lean_closure_set(x_23, 0, x_6);
lean_closure_set(x_23, 1, x_8);
lean_closure_set(x_23, 2, x_2);
lean_closure_set(x_23, 3, x_1);
lean_closure_set(x_23, 4, x_22);
lean_closure_set(x_23, 5, x_3);
lean_closure_set(x_23, 6, x_17);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5___boxed), 2, 1);
lean_closure_set(x_24, 0, x_20);
lean_inc_ref(x_24);
lean_inc_ref(x_1);
lean_inc_ref(x_5);
x_25 = lean_alloc_closure((void*)(lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1___boxed), 5, 3);
lean_closure_set(x_25, 0, x_5);
lean_closure_set(x_25, 1, x_1);
lean_closure_set(x_25, 2, x_24);
lean_inc(x_17);
lean_inc(x_2);
lean_inc(x_3);
lean_inc(x_6);
lean_inc_ref(x_4);
lean_inc_ref_n(x_24, 2);
lean_inc_ref(x_1);
lean_inc_ref(x_5);
x_26 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__0), 12, 10);
lean_closure_set(x_26, 0, x_5);
lean_closure_set(x_26, 1, x_1);
lean_closure_set(x_26, 2, x_24);
lean_closure_set(x_26, 3, x_4);
lean_closure_set(x_26, 4, x_24);
lean_closure_set(x_26, 5, x_6);
lean_closure_set(x_26, 6, x_7);
lean_closure_set(x_26, 7, x_3);
lean_closure_set(x_26, 8, x_2);
lean_closure_set(x_26, 9, x_17);
lean_inc_ref(x_24);
lean_inc_ref(x_1);
lean_inc_ref(x_4);
x_27 = lean_alloc_closure((void*)(lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1___boxed), 5, 3);
lean_closure_set(x_27, 0, x_4);
lean_closure_set(x_27, 1, x_1);
lean_closure_set(x_27, 2, x_24);
lean_inc(x_2);
lean_inc(x_3);
lean_inc(x_6);
lean_inc_ref(x_24);
lean_inc_ref(x_1);
x_28 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__4), 12, 10);
lean_closure_set(x_28, 0, x_4);
lean_closure_set(x_28, 1, x_1);
lean_closure_set(x_28, 2, x_24);
lean_closure_set(x_28, 3, x_5);
lean_closure_set(x_28, 4, x_24);
lean_closure_set(x_28, 5, x_9);
lean_closure_set(x_28, 6, x_6);
lean_closure_set(x_28, 7, x_3);
lean_closure_set(x_28, 8, x_2);
lean_closure_set(x_28, 9, x_17);
lean_inc_ref(x_25);
lean_inc_ref_n(x_21, 2);
lean_inc_ref(x_27);
x_29 = lean_alloc_closure((void*)(lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg___lam__8___boxed), 17, 15);
lean_closure_set(x_29, 0, x_27);
lean_closure_set(x_29, 1, x_26);
lean_closure_set(x_29, 2, x_21);
lean_closure_set(x_29, 3, x_25);
lean_closure_set(x_29, 4, x_8);
lean_closure_set(x_29, 5, x_23);
lean_closure_set(x_29, 6, x_21);
lean_closure_set(x_29, 7, x_6);
lean_closure_set(x_29, 8, x_2);
lean_closure_set(x_29, 9, x_3);
lean_closure_set(x_29, 10, x_1);
lean_closure_set(x_29, 11, x_27);
lean_closure_set(x_29, 12, x_21);
lean_closure_set(x_29, 13, x_28);
lean_closure_set(x_29, 14, x_25);
return x_29;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_3);
x_5 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_4);
x_6 = lean_ctor_get(x_5, 2);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5___boxed), 2, 1);
lean_closure_set(x_7, 0, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg___lam__1___boxed), 5, 3);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_1);
lean_closure_set(x_8, 2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__3___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_24; 
x_24 = lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec(x_18);
lean_dec(x_17);
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__3___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__3___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__17___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__17(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__17___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__17___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9_spec__9___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_invertibleOfLeftInverse___at___00Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9_spec__9(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfLeftInverse___at___00Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9_spec__9___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_invertibleOfLeftInverse___at___00Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9_spec__9___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec(x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_13);
lean_dec(x_11);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec(x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_14);
lean_dec(x_11);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_9);
lean_dec(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_23; 
x_23 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22);
lean_dec(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec(x_18);
lean_dec(x_17);
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
return x_23;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_21; 
x_21 = lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20);
lean_dec(x_20);
lean_dec(x_19);
lean_dec(x_18);
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_9);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_invertibleMul___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12___redArg(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__18___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_sum___at___00Finset_sum___at___00dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__12_spec__12_spec__12___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec(x_8);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec(x_8);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Matrix_diagonal___at___00invertibleOne___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec(x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec(x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_inc(x_17);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__1___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18) {
_start:
{
lean_inc(x_16);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__2___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_inc(x_17);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__3___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_inc(x_11);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0_spec__0___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__10(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_inc(x_10);
lean_inc(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1), 3, 2);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_10);
lean_inc(x_9);
lean_inc(x_1);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_9);
lean_inc_ref(x_4);
lean_inc(x_3);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__2), 5, 4);
lean_closure_set(x_13, 0, x_2);
lean_closure_set(x_13, 1, x_3);
lean_closure_set(x_13, 2, x_4);
lean_closure_set(x_13, 3, x_12);
lean_inc_ref(x_4);
lean_inc(x_6);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__1), 5, 4);
lean_closure_set(x_14, 0, x_5);
lean_closure_set(x_14, 1, x_6);
lean_closure_set(x_14, 2, x_4);
lean_closure_set(x_14, 3, x_13);
lean_inc_ref(x_4);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___lam__8), 5, 4);
lean_closure_set(x_15, 0, x_7);
lean_closure_set(x_15, 1, x_6);
lean_closure_set(x_15, 2, x_4);
lean_closure_set(x_15, 3, x_14);
x_16 = lean_apply_2(x_1, x_9, x_10);
x_17 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_3, x_4, x_15, x_11);
x_18 = lean_apply_2(x_8, x_16, x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__1), 3, 2);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0_spec__0___redArg___lam__2), 3, 2);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_8);
lean_inc_ref(x_5);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__15___redArg___lam__8), 5, 4);
lean_closure_set(x_12, 0, x_3);
lean_closure_set(x_12, 1, x_4);
lean_closure_set(x_12, 2, x_5);
lean_closure_set(x_12, 3, x_11);
x_13 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_6, x_5, x_12, x_10);
x_14 = lean_apply_1(x_7, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__6(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__1_spec__0___redArg___lam__0), 3, 2);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__2), 3, 2);
lean_closure_set(x_11, 0, x_2);
lean_closure_set(x_11, 1, x_8);
lean_inc_ref(x_5);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__9___redArg___lam__2), 5, 4);
lean_closure_set(x_12, 0, x_3);
lean_closure_set(x_12, 1, x_4);
lean_closure_set(x_12, 2, x_5);
lean_closure_set(x_12, 3, x_11);
x_13 = lp_mathlib_dotProduct___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__3___redArg(x_6, x_5, x_12, x_10);
x_14 = lean_apply_1(x_7, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
lean_inc_ref(x_5);
x_12 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_5);
x_13 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_12);
x_14 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_13);
x_15 = lean_ctor_get(x_14, 1);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc_ref(x_5);
x_16 = lp_mathlib_Ring_toAddCommGroup___redArg(x_5);
x_17 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_16);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_17, 1);
lean_inc(x_18);
lean_dec_ref(x_17);
lean_inc(x_10);
lean_inc(x_2);
lean_inc(x_7);
lean_inc_ref(x_5);
lean_inc(x_1);
lean_inc(x_9);
lean_inc(x_6);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__10), 10, 8);
lean_closure_set(x_19, 0, x_6);
lean_closure_set(x_19, 1, x_9);
lean_closure_set(x_19, 2, x_1);
lean_closure_set(x_19, 3, x_5);
lean_closure_set(x_19, 4, x_7);
lean_closure_set(x_19, 5, x_2);
lean_closure_set(x_19, 6, x_10);
lean_closure_set(x_19, 7, x_15);
lean_inc(x_18);
lean_inc(x_1);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_7);
lean_inc(x_6);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__4), 9, 7);
lean_closure_set(x_20, 0, x_6);
lean_closure_set(x_20, 1, x_7);
lean_closure_set(x_20, 2, x_10);
lean_closure_set(x_20, 3, x_2);
lean_closure_set(x_20, 4, x_5);
lean_closure_set(x_20, 5, x_1);
lean_closure_set(x_20, 6, x_18);
lean_inc(x_7);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__6), 9, 7);
lean_closure_set(x_21, 0, x_7);
lean_closure_set(x_21, 1, x_6);
lean_closure_set(x_21, 2, x_9);
lean_closure_set(x_21, 3, x_1);
lean_closure_set(x_21, 4, x_5);
lean_closure_set(x_21, 5, x_2);
lean_closure_set(x_21, 6, x_18);
x_22 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_22, 0, lean_box(0));
lean_closure_set(x_22, 1, lean_box(0));
lean_closure_set(x_22, 2, lean_box(0));
lean_closure_set(x_22, 3, lean_box(0));
lean_closure_set(x_22, 4, lean_box(0));
lean_closure_set(x_22, 5, x_19);
lean_closure_set(x_22, 6, x_21);
lean_closure_set(x_22, 7, x_20);
lean_closure_set(x_22, 8, x_7);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21) {
_start:
{
lean_object* x_22; uint8_t x_23; 
x_22 = lean_ctor_get(x_6, 0);
lean_inc(x_22);
lean_dec_ref(x_6);
x_23 = !lean_is_exclusive(x_7);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_24 = lean_ctor_get(x_7, 0);
x_25 = lean_ctor_get(x_7, 1);
lean_dec(x_25);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___lam__0), 2, 1);
lean_closure_set(x_26, 0, x_13);
lean_inc_ref(x_26);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrix), 10, 8);
lean_closure_set(x_27, 0, lean_box(0));
lean_closure_set(x_27, 1, lean_box(0));
lean_closure_set(x_27, 2, lean_box(0));
lean_closure_set(x_27, 3, lean_box(0));
lean_closure_set(x_27, 4, lean_box(0));
lean_closure_set(x_27, 5, x_8);
lean_closure_set(x_27, 6, x_26);
lean_closure_set(x_27, 7, x_26);
x_28 = lean_apply_1(x_24, x_27);
x_29 = lean_apply_1(x_22, x_28);
lean_inc(x_29);
x_30 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2081_u2081), 8, 6);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, lean_box(0));
lean_closure_set(x_30, 2, lean_box(0));
lean_closure_set(x_30, 3, lean_box(0));
lean_closure_set(x_30, 4, lean_box(0));
lean_closure_set(x_30, 5, x_29);
x_31 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2082_u2082), 8, 6);
lean_closure_set(x_31, 0, lean_box(0));
lean_closure_set(x_31, 1, lean_box(0));
lean_closure_set(x_31, 2, lean_box(0));
lean_closure_set(x_31, 3, lean_box(0));
lean_closure_set(x_31, 4, lean_box(0));
lean_closure_set(x_31, 5, x_29);
lean_ctor_set(x_7, 1, x_31);
lean_ctor_set(x_7, 0, x_30);
return x_7;
}
else
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_32 = lean_ctor_get(x_7, 0);
lean_inc(x_32);
lean_dec(x_7);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___lam__0), 2, 1);
lean_closure_set(x_33, 0, x_13);
lean_inc_ref(x_33);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrix), 10, 8);
lean_closure_set(x_34, 0, lean_box(0));
lean_closure_set(x_34, 1, lean_box(0));
lean_closure_set(x_34, 2, lean_box(0));
lean_closure_set(x_34, 3, lean_box(0));
lean_closure_set(x_34, 4, lean_box(0));
lean_closure_set(x_34, 5, x_8);
lean_closure_set(x_34, 6, x_33);
lean_closure_set(x_34, 7, x_33);
x_35 = lean_apply_1(x_32, x_34);
x_36 = lean_apply_1(x_22, x_35);
lean_inc(x_36);
x_37 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2081_u2081), 8, 6);
lean_closure_set(x_37, 0, lean_box(0));
lean_closure_set(x_37, 1, lean_box(0));
lean_closure_set(x_37, 2, lean_box(0));
lean_closure_set(x_37, 3, lean_box(0));
lean_closure_set(x_37, 4, lean_box(0));
lean_closure_set(x_37, 5, x_36);
x_38 = lean_alloc_closure((void*)(lp_mathlib_Matrix_toBlocks_u2082_u2082), 8, 6);
lean_closure_set(x_38, 0, lean_box(0));
lean_closure_set(x_38, 1, lean_box(0));
lean_closure_set(x_38, 2, lean_box(0));
lean_closure_set(x_38, 3, lean_box(0));
lean_closure_set(x_38, 4, lean_box(0));
lean_closure_set(x_38, 5, x_36);
x_39 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_39, 0, x_37);
lean_ctor_set(x_39, 1, x_38);
return x_39;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
lean_inc_ref(x_5);
x_17 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_5);
x_18 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_17);
x_19 = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(x_18);
x_20 = lean_ctor_get(x_19, 1);
lean_inc(x_20);
lean_dec_ref(x_19);
lean_inc_ref(x_5);
x_21 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_5);
x_22 = lean_ctor_get(x_21, 1);
lean_inc_ref(x_22);
x_23 = lean_ctor_get(x_22, 2);
lean_inc(x_23);
lean_dec_ref(x_22);
x_24 = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(x_21);
lean_dec_ref(x_21);
x_25 = lean_ctor_get(x_24, 2);
lean_inc(x_25);
lean_dec_ref(x_24);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_6);
lean_inc(x_15);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__2), 6, 4);
lean_closure_set(x_26, 0, x_15);
lean_closure_set(x_26, 1, x_6);
lean_closure_set(x_26, 2, x_2);
lean_closure_set(x_26, 3, x_5);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_14);
lean_inc(x_6);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__3), 6, 4);
lean_closure_set(x_27, 0, x_6);
lean_closure_set(x_27, 1, x_14);
lean_closure_set(x_27, 2, x_2);
lean_closure_set(x_27, 3, x_5);
lean_inc(x_20);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocksZero_u2082_u2081Invertible___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_28, 0, x_20);
lean_inc(x_23);
x_29 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___redArg___lam__5___boxed), 2, 1);
lean_closure_set(x_29, 0, x_23);
lean_inc_ref(x_29);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
x_30 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_30, 0, x_4);
lean_closure_set(x_30, 1, x_5);
lean_closure_set(x_30, 2, x_29);
lean_inc_ref(x_5);
lean_inc_ref(x_3);
x_31 = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg___lam__0), 5, 3);
lean_closure_set(x_31, 0, x_3);
lean_closure_set(x_31, 1, x_5);
lean_closure_set(x_31, 2, x_29);
lean_inc(x_25);
lean_inc(x_13);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_6);
lean_inc(x_14);
lean_inc(x_15);
x_32 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__0), 9, 7);
lean_closure_set(x_32, 0, x_15);
lean_closure_set(x_32, 1, x_14);
lean_closure_set(x_32, 2, x_6);
lean_closure_set(x_32, 3, x_2);
lean_closure_set(x_32, 4, x_5);
lean_closure_set(x_32, 5, x_13);
lean_closure_set(x_32, 6, x_25);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_1);
lean_inc_ref(x_30);
lean_inc_ref(x_27);
lean_inc_ref(x_31);
lean_inc(x_16);
lean_inc_ref_n(x_28, 3);
lean_inc_ref(x_32);
x_33 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0___redArg___lam__1), 13, 11);
lean_closure_set(x_33, 0, x_32);
lean_closure_set(x_33, 1, x_28);
lean_closure_set(x_33, 2, x_28);
lean_closure_set(x_33, 3, x_16);
lean_closure_set(x_33, 4, x_31);
lean_closure_set(x_33, 5, x_27);
lean_closure_set(x_33, 6, x_28);
lean_closure_set(x_33, 7, x_30);
lean_closure_set(x_33, 8, x_1);
lean_closure_set(x_33, 9, x_2);
lean_closure_set(x_33, 10, x_5);
lean_inc_ref(x_30);
lean_inc_ref(x_28);
lean_inc_ref(x_31);
x_34 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_34, 0, lean_box(0));
lean_closure_set(x_34, 1, lean_box(0));
lean_closure_set(x_34, 2, lean_box(0));
lean_closure_set(x_34, 3, lean_box(0));
lean_closure_set(x_34, 4, lean_box(0));
lean_closure_set(x_34, 5, x_31);
lean_closure_set(x_34, 6, x_27);
lean_closure_set(x_34, 7, x_28);
lean_closure_set(x_34, 8, x_30);
lean_inc(x_16);
lean_inc_ref_n(x_28, 2);
x_35 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_35, 0, lean_box(0));
lean_closure_set(x_35, 1, lean_box(0));
lean_closure_set(x_35, 2, lean_box(0));
lean_closure_set(x_35, 3, lean_box(0));
lean_closure_set(x_35, 4, lean_box(0));
lean_closure_set(x_35, 5, x_32);
lean_closure_set(x_35, 6, x_28);
lean_closure_set(x_35, 7, x_28);
lean_closure_set(x_35, 8, x_16);
lean_inc(x_14);
lean_inc(x_6);
lean_inc(x_23);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc(x_2);
lean_inc(x_1);
lean_inc_ref(x_5);
x_36 = lp_mathlib_Invertible_mulLeft___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__11___redArg(x_5, x_1, x_2, x_3, x_4, x_23, x_6, x_14, x_34, x_35);
x_37 = lp_mathlib_Equiv_symm___redArg(x_36);
x_38 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_38, 0, lean_box(0));
lean_closure_set(x_38, 1, lean_box(0));
lean_closure_set(x_38, 2, lean_box(0));
lean_closure_set(x_38, 3, lean_box(0));
lean_closure_set(x_38, 4, lean_box(0));
lean_closure_set(x_38, 5, x_31);
lean_closure_set(x_38, 6, x_28);
lean_closure_set(x_38, 7, x_26);
lean_closure_set(x_38, 8, x_30);
lean_inc(x_6);
lean_inc(x_15);
lean_inc(x_23);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc(x_2);
lean_inc(x_1);
lean_inc_ref(x_5);
x_39 = lp_mathlib_Invertible_mulRight___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__16___redArg(x_5, x_1, x_2, x_3, x_4, x_23, x_15, x_6, x_33, x_38);
x_40 = lp_mathlib_Equiv_symm___redArg(x_39);
x_41 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4___redArg(x_1, x_2, x_3, x_4, x_5, x_37, x_40, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_23, x_20, x_6, x_25);
lean_dec(x_25);
lean_dec(x_6);
lean_dec(x_20);
lean_dec(x_23);
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_42 = lean_ctor_get(x_41, 0);
lean_inc(x_42);
lean_dec_ref(x_41);
return x_42;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; lean_object* x_13; 
x_12 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___closed__0;
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_8);
x_13 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3___redArg(x_2, x_1, x_4, x_3, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_11, x_10, x_9, x_8);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec(x_8);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_13, x_14, x_9, x_10, x_11, x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_13, x_14, x_9, x_10, x_11, x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc(x_2);
lean_inc(x_1);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0), 14, 13);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, lean_box(0));
lean_closure_set(x_11, 2, lean_box(0));
lean_closure_set(x_11, 3, x_1);
lean_closure_set(x_11, 4, x_2);
lean_closure_set(x_11, 5, x_3);
lean_closure_set(x_11, 6, x_4);
lean_closure_set(x_11, 7, x_5);
lean_closure_set(x_11, 8, x_6);
lean_closure_set(x_11, 9, x_7);
lean_closure_set(x_11, 10, x_8);
lean_closure_set(x_11, 11, x_9);
lean_closure_set(x_11, 12, x_10);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1___boxed), 14, 13);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, lean_box(0));
lean_closure_set(x_12, 2, lean_box(0));
lean_closure_set(x_12, 3, x_1);
lean_closure_set(x_12, 4, x_2);
lean_closure_set(x_12, 5, x_3);
lean_closure_set(x_12, 6, x_4);
lean_closure_set(x_12, 7, x_5);
lean_closure_set(x_12, 8, x_6);
lean_closure_set(x_12, 9, x_7);
lean_closure_set(x_12, 10, x_8);
lean_closure_set(x_12, 11, x_9);
lean_closure_set(x_12, 12, x_10);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_10, x_11, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_10, x_11, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23, lean_object* x_24, lean_object* x_25, lean_object* x_26, lean_object* x_27) {
_start:
{
lean_object* x_28; 
x_28 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24);
return x_28;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_20; 
x_20 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
lean_inc_ref(x_5);
x_13 = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(x_5);
x_14 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_13);
x_15 = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(x_14);
lean_inc_ref(x_5);
x_16 = lp_mathlib_Ring_toAddCommGroup___redArg(x_5);
x_17 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_16);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_17, 1);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lean_ctor_get(x_15, 1);
lean_inc(x_19);
lean_dec_ref(x_15);
x_20 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg___lam__0), 2, 1);
lean_closure_set(x_20, 0, x_11);
x_21 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg___lam__1), 2, 1);
lean_closure_set(x_21, 0, x_12);
lean_inc(x_18);
lean_inc(x_2);
lean_inc_ref(x_5);
lean_inc(x_1);
lean_inc(x_9);
lean_inc(x_6);
lean_inc(x_7);
x_22 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__6), 9, 7);
lean_closure_set(x_22, 0, x_7);
lean_closure_set(x_22, 1, x_6);
lean_closure_set(x_22, 2, x_9);
lean_closure_set(x_22, 3, x_1);
lean_closure_set(x_22, 4, x_5);
lean_closure_set(x_22, 5, x_2);
lean_closure_set(x_22, 6, x_18);
lean_inc(x_1);
lean_inc_ref(x_5);
lean_inc(x_2);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_6);
x_23 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__4), 9, 7);
lean_closure_set(x_23, 0, x_6);
lean_closure_set(x_23, 1, x_7);
lean_closure_set(x_23, 2, x_8);
lean_closure_set(x_23, 3, x_2);
lean_closure_set(x_23, 4, x_5);
lean_closure_set(x_23, 5, x_1);
lean_closure_set(x_23, 6, x_18);
lean_inc(x_7);
x_24 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___lam__10), 10, 8);
lean_closure_set(x_24, 0, x_6);
lean_closure_set(x_24, 1, x_9);
lean_closure_set(x_24, 2, x_1);
lean_closure_set(x_24, 3, x_5);
lean_closure_set(x_24, 4, x_7);
lean_closure_set(x_24, 5, x_2);
lean_closure_set(x_24, 6, x_8);
lean_closure_set(x_24, 7, x_19);
x_25 = lean_alloc_closure((void*)(lp_mathlib_Matrix_fromBlocks), 11, 9);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, lean_box(0));
lean_closure_set(x_25, 2, lean_box(0));
lean_closure_set(x_25, 3, lean_box(0));
lean_closure_set(x_25, 4, lean_box(0));
lean_closure_set(x_25, 5, x_7);
lean_closure_set(x_25, 6, x_23);
lean_closure_set(x_25, 7, x_22);
lean_closure_set(x_25, 8, x_24);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrix), 10, 8);
lean_closure_set(x_26, 0, lean_box(0));
lean_closure_set(x_26, 1, lean_box(0));
lean_closure_set(x_26, 2, lean_box(0));
lean_closure_set(x_26, 3, lean_box(0));
lean_closure_set(x_26, 4, lean_box(0));
lean_closure_set(x_26, 5, x_25);
lean_closure_set(x_26, 6, x_21);
lean_closure_set(x_26, 7, x_20);
return x_26;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_12, x_13, x_15, x_16, x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg___lam__0), 2, 1);
lean_closure_set(x_10, 0, x_8);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg___lam__1), 2, 1);
lean_closure_set(x_11, 0, x_9);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Matrix_submatrix), 10, 8);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, lean_box(0));
lean_closure_set(x_12, 2, lean_box(0));
lean_closure_set(x_12, 3, lean_box(0));
lean_closure_set(x_12, 4, lean_box(0));
lean_closure_set(x_12, 5, x_6);
lean_closure_set(x_12, 6, x_11);
lean_closure_set(x_12, 7, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__1___boxed(lean_object** _args) {
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
x_20 = lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec(x_17);
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__1___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Invertible_copy_x27___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__1___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__2___boxed(lean_object** _args) {
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
x_19 = lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18);
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__2___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__2___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__3___boxed(lean_object** _args) {
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
x_20 = lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec(x_17);
lean_dec(x_16);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__3___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Invertible_copy_x27___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__3___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_invertibleOfRightInverse___at___00Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRightInverse___at___00Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0_spec__0___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_invertibleOfRightInverse___at___00Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0_spec__0___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_14);
lean_dec(x_11);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_9);
lean_dec(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4___boxed(lean_object** _args) {
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
lean_object* x_26 = _args[25];
lean_object* x_27 = _args[26];
_start:
{
lean_object* x_28; 
x_28 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27);
lean_dec(x_27);
lean_dec(x_26);
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec(x_18);
lean_dec(x_17);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
return x_28;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3___boxed(lean_object** _args) {
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
x_20 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_11);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_10);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4___redArg___boxed(lean_object** _args) {
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
_start:
{
lean_object* x_22; 
x_22 = lp_mathlib_Matrix_invertibleOfFromBlocksZero_u2081_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3_spec__4___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec(x_18);
lean_dec(x_17);
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec(x_8);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Matrix_submatrixEquivInvertible___at___00Matrix_fromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__1_spec__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_10);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___at___00Matrix_invertibleEquivFromBlocks_u2081_u2081Invertible___elam__0_spec__0_spec__3___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec(x_8);
return x_17;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Invertible(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_NonsingularInverse(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_SchurComplement(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_NonsingularInverse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___closed__0 = _init_lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Matrix_invertibleOfFromBlocks_u2081_u2081Invertible___redArg___closed__0);
lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2___redArg___closed__0 = _init_lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Matrix_diagonal___at___00Matrix_invertibleOfFromBlocks_u2082_u2082Invertible___at___00Matrix_invertibleEquivFromBlocks_u2082_u2082Invertible___elam__0_spec__0_spec__2___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
