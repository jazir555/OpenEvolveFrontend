// Lean compiler output
// Module: Mathlib.Algebra.Lie.SkewAdjoint
// Imports: public import Init public import Mathlib.Algebra.Lie.Matrix public import Mathlib.LinearAlgebra.Matrix.SesquilinearForm public import Mathlib.Tactic.NoncommRing
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
lean_object* lp_mathlib_Matrix_lieConj___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LieRing_ofAssociativeRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquivTranspose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LieEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebraEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Module_End_instRing___redArg(lean_object*);
lean_object* lp_mathlib_LieEquiv_ofSubalgebras___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebraEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquivTranspose___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_lieConj___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebraEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Module_End_instAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_instAlgebra___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_toLieEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_instRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebraEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Algebra_id___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquivTranspose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_skewAdjointLieSubalgebra(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebraEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lp_mathlib_Module_End_instRing___redArg(x_2);
x_6 = lp_mathlib_LieRing_ofAssociativeRing___redArg(x_5);
x_7 = lp_mathlib_Module_End_instAlgebra___redArg(x_3);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_box(0);
x_10 = lp_mathlib_LinearEquiv_lieConj___redArg(x_4);
x_11 = lp_mathlib_LieEquiv_ofSubalgebras___redArg(x_1, x_6, x_8, x_9, x_10);
lean_dec(x_8);
lean_dec_ref(x_6);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebraEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_skewAdjointLieSubalgebraEquiv___redArg(x_3, x_4, x_5, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebraEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_skewAdjointLieSubalgebraEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_6);
lean_dec_ref(x_3);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointLieSubalgebraEquiv___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_skewAdjointLieSubalgebraEquiv___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_skewAdjointMatricesLieSubalgebra(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
lean_inc_ref(x_1);
lean_inc_ref(x_2);
lean_inc(x_3);
x_6 = lp_mathlib_Matrix_instRing___redArg(x_3, x_2, x_1);
x_7 = lp_mathlib_LieRing_ofAssociativeRing___redArg(x_6);
x_8 = lean_ctor_get(x_1, 0);
x_9 = lp_mathlib_Algebra_id___redArg(x_8);
lean_inc_ref(x_2);
x_10 = lp_mathlib_Matrix_instAlgebra___redArg(x_2, x_8, x_9);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lean_box(0);
lean_inc_ref(x_1);
x_13 = lp_mathlib_Matrix_lieConj___redArg(x_1, x_2, x_3, x_4, x_5);
x_14 = lp_mathlib_LieEquiv_symm___redArg(x_13);
x_15 = lp_mathlib_LieEquiv_ofSubalgebras___redArg(x_1, x_7, x_11, x_12, x_14);
lean_dec(x_11);
lean_dec_ref(x_7);
lean_dec_ref(x_1);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_skewAdjointMatricesLieSubalgebraEquiv___redArg(x_3, x_4, x_5, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_skewAdjointMatricesLieSubalgebraEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquivTranspose___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_5 = lp_mathlib_Matrix_instRing___redArg(x_3, x_2, x_1);
x_6 = lp_mathlib_LieRing_ofAssociativeRing___redArg(x_5);
x_7 = lean_ctor_get(x_1, 0);
x_8 = lp_mathlib_Algebra_id___redArg(x_7);
x_9 = lp_mathlib_Matrix_instAlgebra___redArg(x_2, x_7, x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lean_box(0);
x_12 = lp_mathlib_AlgEquiv_toLieEquiv___redArg(x_4);
x_13 = lp_mathlib_LieEquiv_ofSubalgebras___redArg(x_1, x_6, x_10, x_11, x_12);
lean_dec(x_10);
lean_dec_ref(x_6);
lean_dec_ref(x_1);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquivTranspose(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_skewAdjointMatricesLieSubalgebraEquivTranspose___redArg(x_3, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_skewAdjointMatricesLieSubalgebraEquivTranspose___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_skewAdjointMatricesLieSubalgebraEquivTranspose(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_12;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Matrix(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_SesquilinearForm(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NoncommRing(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Lie_SkewAdjoint(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Lie_Matrix(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_SesquilinearForm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NoncommRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
