// Lean compiler output
// Module: Mathlib.Tactic.Linter
// Imports: public import Init public meta import Mathlib.Tactic.Linter.DeprecatedModule public meta import Mathlib.Tactic.Linter.HaveLetLinter public meta import Mathlib.Tactic.Linter.MinImports public meta import Mathlib.Tactic.Linter.PPRoundtrip public meta import Mathlib.Tactic.Linter.PrivateModule public meta import Mathlib.Tactic.Linter.UnusedInstancesInType public meta import Mathlib.Tactic.Linter.UpstreamableDecl
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
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedModule(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_HaveLetLinter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_MinImports(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_PPRoundtrip(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_PrivateModule(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_UnusedInstancesInType(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_UpstreamableDecl(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedModule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_HaveLetLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_PPRoundtrip(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_PrivateModule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_UnusedInstancesInType(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_UpstreamableDecl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
