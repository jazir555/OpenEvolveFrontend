// Lean compiler output
// Module: Mathlib.Tactic.FunProp
// Imports: public import Init public meta import Mathlib.Tactic.FunProp.Attr public meta import Mathlib.Tactic.FunProp.Core public meta import Mathlib.Tactic.FunProp.Decl public meta import Mathlib.Tactic.FunProp.Elab public meta import Mathlib.Tactic.FunProp.FunctionData public meta import Mathlib.Tactic.FunProp.Mor public meta import Mathlib.Tactic.FunProp.Theorems public meta import Mathlib.Tactic.FunProp.ToBatteries public meta import Mathlib.Tactic.FunProp.Types
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
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Core(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Decl(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Elab(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Mor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Theorems(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Types(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FunProp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Decl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Elab(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Mor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Theorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
