// Lean compiler output
// Module: Batteries.Data.List.ArrayMap
// Imports: public import Init
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toArrayMap___redArg(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_toArrayMap_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_List_toArrayMap___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_List_toArrayMap(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_toArrayMap_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_toArrayMap_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
lean_inc(x_1);
x_6 = lean_apply_1(x_1, x_4);
x_7 = lean_array_push(x_2, x_6);
x_2 = x_7;
x_3 = x_5;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_toArrayMap_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_List_foldl___at___00List_toArrayMap_spec__0___redArg(x_3, x_4, x_5);
return x_6;
}
}
static lean_object* _init_lp_batteries_List_toArrayMap___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toArrayMap___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_batteries_List_toArrayMap___redArg___closed__0;
x_4 = lp_batteries_List_foldl___at___00List_toArrayMap_spec__0___redArg(x_2, x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toArrayMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_List_toArrayMap___redArg(x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_List_ArrayMap(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_List_toArrayMap___redArg___closed__0 = _init_lp_batteries_List_toArrayMap___redArg___closed__0();
lean_mark_persistent(lp_batteries_List_toArrayMap___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
