// Lean compiler output
// Module: Mathlib.Algebra.BrauerGroup.Defs
// Imports: public import Init public import Mathlib.Algebra.Category.AlgCat.Basic public import Mathlib.Algebra.Central.Defs public import Mathlib.LinearAlgebra.FiniteDimensional.Defs public import Mathlib.LinearAlgebra.Matrix.Reindex
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
LEAN_EXPORT lean_object* lp_mathlib_Brauer_CSA__Setoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Brauer_CSA__Setoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeSortCSAType(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeSortCSAType___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeSortCSAType(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeSortCSAType___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instCoeSortCSAType(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Brauer_CSA__Setoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Brauer_CSA__Setoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Brauer_CSA__Setoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_AlgCat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Central_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_FiniteDimensional_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Reindex(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_BrauerGroup_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_AlgCat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Central_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_FiniteDimensional_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Reindex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
