// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Rat
// Imports: public import Init public import Mathlib.Algebra.Algebra.Defs public import Mathlib.Algebra.Module.Equiv.Defs public import Mathlib.Data.Rat.Cast.CharZero
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
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toNNRatAlgebra(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toRatAlgebra(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toRatAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toNNRatAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toNNRatAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 5);
lean_inc(x_2);
x_3 = lp_mathlib_NNRat_castHom___redArg(x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toNNRatAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_DivisionSemiring_toNNRatAlgebra___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toRatAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 7);
lean_inc(x_2);
x_3 = lp_mathlib_Rat_castHom___redArg(x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toRatAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_DivisionRing_toRatAlgebra___redArg(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
