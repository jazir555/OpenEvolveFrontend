// Lean compiler output
// Module: RESE
// Imports: public import Init public import RESE.Basic public import RESE.Constraint public import RESE.Templates public import RESE.TestCases public import RESE.Default
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
lean_object* initialize_rese_RESE_Basic(uint8_t builtin);
lean_object* initialize_rese_RESE_Constraint(uint8_t builtin);
lean_object* initialize_rese_RESE_Templates(uint8_t builtin);
lean_object* initialize_rese_RESE_TestCases(uint8_t builtin);
lean_object* initialize_rese_RESE_Default(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_rese_RESE(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_rese_RESE_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_rese_RESE_Constraint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_rese_RESE_Templates(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_rese_RESE_TestCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_rese_RESE_Default(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
