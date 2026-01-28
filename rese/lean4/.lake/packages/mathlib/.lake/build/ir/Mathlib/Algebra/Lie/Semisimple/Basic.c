// Lean compiler output
// Module: Mathlib.Algebra.Lie.Semisimple.Basic
// Imports: public import Init public import Mathlib.Algebra.Lie.Semisimple.Defs public import Mathlib.Order.BooleanGenerators
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
LEAN_EXPORT lean_object* lp_mathlib_LieAlgebra_IsSemisimple_instDistribLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieAlgebra_IsSemisimple_instDistribLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieAlgebra_IsSemisimple_instDistribLattice___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LieSubmodule_instCompleteLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_IsCompactlyGenerated_BooleanGenerators_distribLattice__of__sSup__eq__top___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieAlgebra_IsSemisimple_instDistribLattice___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieAlgebra_IsSemisimple_instDistribLattice___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lp_mathlib_LieSubmodule_instCompleteLattice(lean_box(0), lean_box(0), lean_box(0), x_1, x_2, x_4, x_3, x_5);
x_7 = lp_mathlib_IsCompactlyGenerated_BooleanGenerators_distribLattice__of__sSup__eq__top___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieAlgebra_IsSemisimple_instDistribLattice(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_LieAlgebra_IsSemisimple_instDistribLattice___redArg(x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieAlgebra_IsSemisimple_instDistribLattice___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_LieAlgebra_IsSemisimple_instDistribLattice(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieAlgebra_IsSemisimple_instDistribLattice___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_LieAlgebra_IsSemisimple_instDistribLattice___redArg(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Semisimple_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BooleanGenerators(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Semisimple_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Lie_Semisimple_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BooleanGenerators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
