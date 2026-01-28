// Lean compiler output
// Module: Mathlib.Algebra.Lie.Weights.Killing
// Imports: public import Init public import Mathlib.Algebra.Lie.Derivation.Killing public import Mathlib.Algebra.Lie.Killing public import Mathlib.Algebra.Lie.Sl2 public import Mathlib.Algebra.Lie.Weights.Chain public import Mathlib.LinearAlgebra.Eigenspace.Semisimple public import Mathlib.LinearAlgebra.JordanChevalley
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
lean_object* lp_mathlib_LieRing_ofAssociativeRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieModule_Weight_instInvolutiveNegSubtypeMemLieSubalgebra___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieModule_Weight_instInvolutiveNegSubtypeMemLieSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieModule_Weight_instInvolutiveNegSubtypeMemLieSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieModule_Weight_instInvolutiveNegSubtypeMemLieSubalgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LieModule_Weight_instInvolutiveNegSubtypeMemLieSubalgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_2, x_3);
x_5 = lean_apply_1(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieModule_Weight_instInvolutiveNegSubtypeMemLieSubalgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lp_mathlib_Field_toDivisionRing___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
x_4 = lp_mathlib_LieRing_ofAssociativeRing___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_5);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_LieModule_Weight_instInvolutiveNegSubtypeMemLieSubalgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieModule_Weight_instInvolutiveNegSubtypeMemLieSubalgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_LieModule_Weight_instInvolutiveNegSubtypeMemLieSubalgebra___redArg(x_4);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LieModule_Weight_instInvolutiveNegSubtypeMemLieSubalgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_LieModule_Weight_instInvolutiveNegSubtypeMemLieSubalgebra(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Derivation_Killing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Killing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Sl2(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Weights_Chain(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Eigenspace_Semisimple(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_JordanChevalley(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Lie_Weights_Killing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Lie_Derivation_Killing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Lie_Killing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Lie_Sl2(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Lie_Weights_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Eigenspace_Semisimple(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_JordanChevalley(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
