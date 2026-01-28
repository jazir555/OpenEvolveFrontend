// Lean compiler output
// Module: Mathlib.Algebra.Lie.Weights.RootSystem
// Imports: public import Init public import Mathlib.Algebra.Algebra.Rat public import Mathlib.Algebra.Lie.Weights.Killing public import Mathlib.Algebra.Module.Torsion.Free public import Mathlib.LinearAlgebra.RootSystem.Basic public import Mathlib.LinearAlgebra.RootSystem.Finite.CanonicalBilinear public import Mathlib.LinearAlgebra.RootSystem.Reduced
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
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Weights_Killing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_Finite_CanonicalBilinear(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_Reduced(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Weights_RootSystem(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Lie_Weights_Killing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_Finite_CanonicalBilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_Reduced(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
