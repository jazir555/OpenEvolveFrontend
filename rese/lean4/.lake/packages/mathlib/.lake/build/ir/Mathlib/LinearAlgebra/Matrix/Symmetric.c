// Lean compiler output
// Module: Mathlib.LinearAlgebra.Matrix.Symmetric
// Imports: public import Init public import Mathlib.Data.Matrix.Basic public import Mathlib.Data.Matrix.Block
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
LEAN_EXPORT uint8_t lp_mathlib_Matrix_instDecidableIsSymmOfEqTranspose(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_Matrix_instDecidableIsSymmOfEqTranspose___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instDecidableIsSymmOfEqTranspose___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instDecidableIsSymmOfEqTranspose___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Matrix_instDecidableIsSymmOfEqTranspose(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4) {
_start:
{
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Matrix_instDecidableIsSymmOfEqTranspose___redArg(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instDecidableIsSymmOfEqTranspose___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; uint8_t x_6; lean_object* x_7; 
x_5 = lean_unbox(x_4);
x_6 = lp_mathlib_Matrix_instDecidableIsSymmOfEqTranspose(x_1, x_2, x_3, x_5);
lean_dec(x_3);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instDecidableIsSymmOfEqTranspose___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_Matrix_instDecidableIsSymmOfEqTranspose___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Block(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Symmetric(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Block(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
