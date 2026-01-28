// Lean compiler output
// Module: ImportGraph
// Imports: public import Init public import ImportGraph.Cli public import ImportGraph.Imports public import ImportGraph.Meta public import ImportGraph.CurrentModule public import ImportGraph.FromSource public import ImportGraph.Lean.Name public import ImportGraph.RequiredModules
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
lean_object* initialize_importGraph_ImportGraph_Cli(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Imports(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Meta(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_CurrentModule(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_FromSource(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Lean_Name(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_RequiredModules(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_importGraph_ImportGraph(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Cli(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Imports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_CurrentModule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_FromSource(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Lean_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_RequiredModules(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
