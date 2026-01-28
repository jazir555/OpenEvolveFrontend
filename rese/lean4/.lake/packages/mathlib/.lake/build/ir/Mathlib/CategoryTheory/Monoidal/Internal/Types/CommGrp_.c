// Lean compiler output
// Module: Mathlib.CategoryTheory.Monoidal.Internal.Types.CommGrp_
// Imports: public import Init public import Mathlib.CategoryTheory.Monoidal.Internal.Types.Grp_ public import Mathlib.CategoryTheory.Monoidal.CommGrp_
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
LEAN_EXPORT lean_object* lp_mathlib_CommGrpTypeEquivalenceCommGrp_commGrpCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGrpTypeEquivalenceCommGrp_commGrpCommGroup(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_GrpTypeEquivalenceGrp_grpGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGrpTypeEquivalenceCommGrp_commGrpCommGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_GrpTypeEquivalenceGrp_grpGroup___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGrpTypeEquivalenceCommGrp_commGrpCommGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_GrpTypeEquivalenceGrp_grpGroup___redArg(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Internal_Types_Grp__(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_CommGrp__(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Internal_Types_CommGrp__(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_Internal_Types_Grp__(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Monoidal_CommGrp__(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
