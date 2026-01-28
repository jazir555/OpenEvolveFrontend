// Lean compiler output
// Module: Aesop.Tree
// Imports: public import Init public import Aesop.Tree.AddRapp public import Aesop.Tree.Check public import Aesop.Tree.Data public import Aesop.Tree.ExtractProof public import Aesop.Tree.ExtractScript public import Aesop.Tree.Free public import Aesop.Tree.RunMetaM public import Aesop.Tree.State public import Aesop.Tree.Tracing public import Aesop.Tree.Traversal public import Aesop.Tree.TreeM public import Aesop.Tree.UnsafeQueue
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
lean_object* initialize_aesop_Aesop_Tree_AddRapp(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Check(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Data(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_ExtractProof(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_ExtractScript(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Free(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_RunMetaM(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_State(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Tracing(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Traversal(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_UnsafeQueue(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_AddRapp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Data(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_ExtractProof(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_ExtractScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Free(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_RunMetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Traversal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_UnsafeQueue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
