// Lean compiler output
// Module: Batteries.Data.List
// Imports: public import Init public import Batteries.Data.List.ArrayMap public import Batteries.Data.List.Basic public import Batteries.Data.List.Count public import Batteries.Data.List.Init.Lemmas public import Batteries.Data.List.Lemmas public import Batteries.Data.List.Matcher public import Batteries.Data.List.Monadic public import Batteries.Data.List.Pairwise public import Batteries.Data.List.Perm public import Batteries.Data.List.Scan
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
lean_object* initialize_batteries_Batteries_Data_List_ArrayMap(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Count(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Init_Lemmas(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Lemmas(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Matcher(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Monadic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Pairwise(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Perm(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Scan(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_List(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_ArrayMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Init_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Matcher(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Monadic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Perm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Scan(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
