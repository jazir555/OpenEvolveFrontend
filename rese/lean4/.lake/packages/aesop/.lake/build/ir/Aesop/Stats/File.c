// Lean compiler output
// Module: Aesop.Stats.File
// Imports: public import Init public import Aesop.Stats.Basic public import Lean.Data.Position
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
lean_object* l_Lean_JsonNumber_fromNat(lean_object*);
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_io_prim_handle_lock(lean_object*, uint8_t);
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Json_compress(lean_object*);
lean_object* lean_io_prim_handle_unlock(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0___boxed(lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1(lean_object*);
lean_object* l_Lean_instToJsonPosition_toJson(lean_object*);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__9;
static lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__1;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1(size_t, size_t, lean_object*);
static lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__1;
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__10;
lean_object* l_Lean_PrettyPrinter_ppCategory___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__12;
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__11;
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
lean_object* l_IO_FS_Handle_putStrLn(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__1;
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
extern lean_object* l_Lean_KVMap_instValueString;
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord___closed__0;
static lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__0;
lean_object* lean_array_to_list(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord;
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__4;
lean_object* lp_aesop_Aesop_instToJsonScriptGenerated_toJson(lean_object*);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__5;
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__7(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_instToJsonGoalStats_toJson(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__6(lean_object*);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__0;
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__13;
lean_object* l_IO_FS_withFile___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_instToJsonRuleStats_toJson(lean_object*);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__8;
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__0;
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__15;
size_t lean_usize_add(size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__14;
lean_object* lean_array_uget(lean_object*, size_t);
size_t lean_array_size(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
static lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1(size_t x_1, size_t x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_usize_dec_lt(x_2, x_1);
if (x_4 == 0)
{
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; size_t x_9; size_t x_10; lean_object* x_11; 
x_5 = lean_array_uget(x_3, x_2);
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_array_uset(x_3, x_2, x_6);
x_8 = lp_aesop_Aesop_instToJsonRuleStats_toJson(x_5);
x_9 = 1;
x_10 = lean_usize_add(x_2, x_9);
x_11 = lean_array_uset(x_7, x_2, x_8);
x_2 = x_10;
x_3 = x_11;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1(lean_object* x_1) {
_start:
{
size_t x_2; size_t x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_array_size(x_1);
x_3 = 0;
x_4 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1(x_2, x_3, x_1);
x_5 = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3_spec__3(size_t x_1, size_t x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_usize_dec_lt(x_2, x_1);
if (x_4 == 0)
{
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; size_t x_9; size_t x_10; lean_object* x_11; 
x_5 = lean_array_uget(x_3, x_2);
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_array_uset(x_3, x_2, x_6);
x_8 = lp_aesop_Aesop_instToJsonGoalStats_toJson(x_5);
x_9 = 1;
x_10 = lean_usize_add(x_2, x_9);
x_11 = lean_array_uset(x_7, x_2, x_8);
x_2 = x_10;
x_3 = x_11;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3(lean_object* x_1) {
_start:
{
size_t x_2; size_t x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_array_size(x_1);
x_3 = 0;
x_4 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3_spec__3(x_2, x_3, x_1);
x_5 = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lp_aesop_Aesop_instToJsonScriptGenerated_toJson(x_3);
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__5(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = l_Lean_instToJsonPosition_toJson(x_3);
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__7(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = lean_array_to_list(x_2);
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = l_List_foldl___at___00Array_appendList_spec__0___redArg(x_2, x_4);
x_1 = x_5;
x_2 = x_6;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__6(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
else
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; uint8_t x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = 1;
x_6 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_4, x_5);
lean_ctor_set_tag(x_1, 3);
lean_ctor_set(x_1, 0, x_6);
return x_1;
}
else
{
lean_object* x_7; uint8_t x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec(x_1);
x_8 = 1;
x_9 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_7, x_8);
x_10 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("total", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("configParsing", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ruleSetConstruction", 19, 19);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("search", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ruleSelection", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("script", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("forwardState", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("scriptGenerated", 15, 15);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ruleStats", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("goalStats", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("syntax", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("file", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("position", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("declaration", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("goalSolved", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
x_6 = lean_ctor_get(x_1, 4);
lean_inc(x_6);
x_7 = lean_ctor_get_uint8(x_1, sizeof(void*)*5);
lean_dec_ref(x_1);
x_8 = lean_ctor_get(x_2, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_2, 1);
lean_inc(x_9);
x_10 = lean_ctor_get(x_2, 2);
lean_inc(x_10);
x_11 = lean_ctor_get(x_2, 3);
lean_inc(x_11);
x_12 = lean_ctor_get(x_2, 4);
lean_inc(x_12);
x_13 = lean_ctor_get(x_2, 5);
lean_inc(x_13);
x_14 = lean_ctor_get(x_2, 6);
lean_inc(x_14);
x_15 = lean_ctor_get(x_2, 7);
lean_inc(x_15);
x_16 = lean_ctor_get(x_2, 8);
lean_inc_ref(x_16);
x_17 = lean_ctor_get(x_2, 9);
lean_inc_ref(x_17);
lean_dec_ref(x_2);
x_18 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__0;
x_19 = l_Lean_JsonNumber_fromNat(x_8);
x_20 = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(x_20, 0, x_19);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_18);
lean_ctor_set(x_21, 1, x_20);
x_22 = lean_box(0);
x_23 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_22);
x_24 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__1;
x_25 = l_Lean_JsonNumber_fromNat(x_9);
x_26 = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(x_26, 0, x_25);
x_27 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_27, 0, x_24);
lean_ctor_set(x_27, 1, x_26);
x_28 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_28, 0, x_27);
lean_ctor_set(x_28, 1, x_22);
x_29 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__2;
x_30 = l_Lean_JsonNumber_fromNat(x_10);
x_31 = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(x_31, 0, x_30);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_29);
lean_ctor_set(x_32, 1, x_31);
x_33 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_33, 1, x_22);
x_34 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__3;
x_35 = l_Lean_JsonNumber_fromNat(x_11);
x_36 = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(x_36, 0, x_35);
x_37 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_37, 0, x_34);
lean_ctor_set(x_37, 1, x_36);
x_38 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_22);
x_39 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__4;
x_40 = l_Lean_JsonNumber_fromNat(x_12);
x_41 = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(x_41, 0, x_40);
x_42 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_42, 0, x_39);
lean_ctor_set(x_42, 1, x_41);
x_43 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_43, 0, x_42);
lean_ctor_set(x_43, 1, x_22);
x_44 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__5;
x_45 = l_Lean_JsonNumber_fromNat(x_13);
x_46 = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(x_46, 0, x_45);
x_47 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_47, 0, x_44);
lean_ctor_set(x_47, 1, x_46);
x_48 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_48, 0, x_47);
lean_ctor_set(x_48, 1, x_22);
x_49 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__6;
x_50 = l_Lean_JsonNumber_fromNat(x_14);
x_51 = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(x_51, 0, x_50);
x_52 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_52, 0, x_49);
lean_ctor_set(x_52, 1, x_51);
x_53 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_53, 0, x_52);
lean_ctor_set(x_53, 1, x_22);
x_54 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__7;
x_55 = lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0(x_15);
lean_dec(x_15);
x_56 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_56, 0, x_54);
lean_ctor_set(x_56, 1, x_55);
x_57 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_57, 0, x_56);
lean_ctor_set(x_57, 1, x_22);
x_58 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__8;
x_59 = lp_aesop_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1(x_16);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_58);
lean_ctor_set(x_60, 1, x_59);
x_61 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_61, 0, x_60);
lean_ctor_set(x_61, 1, x_22);
x_62 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__9;
x_63 = lp_aesop_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3(x_17);
x_64 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_64, 0, x_62);
lean_ctor_set(x_64, 1, x_63);
x_65 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_65, 0, x_64);
lean_ctor_set(x_65, 1, x_22);
x_66 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__10;
x_67 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_67, 0, x_3);
x_68 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_68, 0, x_66);
lean_ctor_set(x_68, 1, x_67);
x_69 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_69, 0, x_68);
lean_ctor_set(x_69, 1, x_22);
x_70 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__11;
x_71 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_71, 0, x_4);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_70);
lean_ctor_set(x_72, 1, x_71);
x_73 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_73, 0, x_72);
lean_ctor_set(x_73, 1, x_22);
x_74 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__12;
x_75 = lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__5(x_5);
x_76 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_76, 0, x_74);
lean_ctor_set(x_76, 1, x_75);
x_77 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_77, 0, x_76);
lean_ctor_set(x_77, 1, x_22);
x_78 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__13;
x_79 = lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__6(x_6);
x_80 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_80, 0, x_78);
lean_ctor_set(x_80, 1, x_79);
x_81 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_81, 0, x_80);
lean_ctor_set(x_81, 1, x_22);
x_82 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__14;
x_83 = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(x_83, 0, x_7);
x_84 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_84, 0, x_82);
lean_ctor_set(x_84, 1, x_83);
x_85 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_85, 0, x_84);
lean_ctor_set(x_85, 1, x_22);
x_86 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_86, 0, x_85);
lean_ctor_set(x_86, 1, x_22);
x_87 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_87, 0, x_81);
lean_ctor_set(x_87, 1, x_86);
x_88 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_88, 0, x_77);
lean_ctor_set(x_88, 1, x_87);
x_89 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_89, 0, x_73);
lean_ctor_set(x_89, 1, x_88);
x_90 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_90, 0, x_69);
lean_ctor_set(x_90, 1, x_89);
x_91 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_91, 0, x_65);
lean_ctor_set(x_91, 1, x_90);
x_92 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_92, 0, x_61);
lean_ctor_set(x_92, 1, x_91);
x_93 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_93, 0, x_57);
lean_ctor_set(x_93, 1, x_92);
x_94 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_94, 0, x_53);
lean_ctor_set(x_94, 1, x_93);
x_95 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_95, 0, x_48);
lean_ctor_set(x_95, 1, x_94);
x_96 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_96, 0, x_43);
lean_ctor_set(x_96, 1, x_95);
x_97 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_97, 0, x_38);
lean_ctor_set(x_97, 1, x_96);
x_98 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_98, 0, x_33);
lean_ctor_set(x_98, 1, x_97);
x_99 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_99, 0, x_28);
lean_ctor_set(x_99, 1, x_98);
x_100 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_100, 0, x_23);
lean_ctor_set(x_100, 1, x_99);
x_101 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__15;
x_102 = lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__7(x_100, x_101);
x_103 = l_Lean_Json_mkObj(x_102);
return x_103;
}
}
LEAN_EXPORT lean_object* lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; size_t x_5; lean_object* x_6; 
x_4 = lean_unbox_usize(x_1);
lean_dec(x_1);
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1(x_4, x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; size_t x_5; lean_object* x_6; 
x_4 = lean_unbox_usize(x_1);
lean_dec(x_1);
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3_spec__3(x_4, x_5, x_3);
return x_6;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instToJsonStatsFileRecord() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instToJsonStatsFileRecord___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_cstr_to_nat("100000000000");
x_9 = lean_unsigned_to_nat(0u);
x_10 = l_Std_Format_pretty(x_7, x_8, x_9, x_9);
x_11 = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(x_11, 0, x_1);
lean_ctor_set(x_11, 1, x_10);
lean_ctor_set(x_11, 2, x_2);
lean_ctor_set(x_11, 3, x_3);
lean_ctor_set(x_11, 4, x_4);
lean_ctor_set_uint8(x_11, sizeof(void*)*5, x_5);
x_12 = lean_apply_2(x_6, lean_box(0), x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_5);
x_9 = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0(x_1, x_2, x_3, x_4, x_8, x_6, x_7);
return x_9;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = lean_box(x_4);
x_11 = lean_alloc_closure((void*)(lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0___boxed), 7, 6);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_2);
lean_closure_set(x_11, 2, x_3);
lean_closure_set(x_11, 3, x_9);
lean_closure_set(x_11, 4, x_10);
lean_closure_set(x_11, 5, x_5);
x_12 = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__1;
x_13 = lean_alloc_closure((void*)(l_Lean_PrettyPrinter_ppCategory___boxed), 5, 2);
lean_closure_set(x_13, 0, x_12);
lean_closure_set(x_13, 1, x_6);
x_14 = lean_apply_2(x_7, lean_box(0), x_13);
x_15 = lean_apply_4(x_8, lean_box(0), lean_box(0), x_14, x_11);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; lean_object* x_11; 
x_10 = lean_unbox(x_4);
x_11 = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1(x_1, x_2, x_3, x_10, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; uint8_t x_15; lean_object* x_16; 
x_15 = 0;
x_16 = l_Lean_Syntax_getPos_x3f(x_5, x_15);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_17; 
lean_dec_ref(x_9);
x_17 = lean_box(0);
x_10 = x_17;
goto block_14;
}
else
{
uint8_t x_18; 
x_18 = !lean_is_exclusive(x_16);
if (x_18 == 0)
{
lean_object* x_19; lean_object* x_20; 
x_19 = lean_ctor_get(x_16, 0);
x_20 = l_Lean_FileMap_toPosition(x_9, x_19);
lean_dec(x_19);
lean_ctor_set(x_16, 0, x_20);
x_10 = x_16;
goto block_14;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_21 = lean_ctor_get(x_16, 0);
lean_inc(x_21);
lean_dec(x_16);
x_22 = l_Lean_FileMap_toPosition(x_9, x_21);
lean_dec(x_21);
x_23 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_23, 0, x_22);
x_10 = x_23;
goto block_14;
}
}
block_14:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_11 = lean_box(x_3);
lean_inc(x_7);
x_12 = lean_alloc_closure((void*)(lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___boxed), 9, 8);
lean_closure_set(x_12, 0, x_1);
lean_closure_set(x_12, 1, x_2);
lean_closure_set(x_12, 2, x_10);
lean_closure_set(x_12, 3, x_11);
lean_closure_set(x_12, 4, x_4);
lean_closure_set(x_12, 5, x_5);
lean_closure_set(x_12, 6, x_6);
lean_closure_set(x_12, 7, x_7);
x_13 = lean_apply_4(x_7, lean_box(0), lean_box(0), x_8, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; lean_object* x_11; 
x_10 = lean_unbox(x_3);
x_11 = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2(x_1, x_2, x_10, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_10 = lean_box(x_2);
lean_inc(x_6);
x_11 = lean_alloc_closure((void*)(lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2___boxed), 9, 8);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_9);
lean_closure_set(x_11, 2, x_10);
lean_closure_set(x_11, 3, x_3);
lean_closure_set(x_11, 4, x_4);
lean_closure_set(x_11, 5, x_5);
lean_closure_set(x_11, 6, x_6);
lean_closure_set(x_11, 7, x_7);
x_12 = lean_apply_4(x_6, lean_box(0), lean_box(0), x_8, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; lean_object* x_11; 
x_10 = lean_unbox(x_2);
x_11 = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3(x_1, x_10, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_8 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_8);
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_dec_ref(x_1);
x_10 = lean_ctor_get(x_2, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_2, 2);
lean_inc(x_11);
lean_dec_ref(x_2);
x_12 = lean_ctor_get(x_8, 1);
lean_inc(x_12);
lean_dec_ref(x_8);
x_13 = lean_box(x_6);
lean_inc(x_9);
x_14 = lean_alloc_closure((void*)(lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3___boxed), 9, 8);
lean_closure_set(x_14, 0, x_7);
lean_closure_set(x_14, 1, x_13);
lean_closure_set(x_14, 2, x_12);
lean_closure_set(x_14, 3, x_5);
lean_closure_set(x_14, 4, x_4);
lean_closure_set(x_14, 5, x_9);
lean_closure_set(x_14, 6, x_3);
lean_closure_set(x_14, 7, x_10);
x_15 = lean_apply_4(x_9, lean_box(0), lean_box(0), x_11, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, uint8_t x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; lean_object* x_10; 
x_9 = lean_unbox(x_7);
x_10 = lp_aesop_Aesop_StatsFileRecord_ofStats(x_1, x_2, x_3, x_4, x_5, x_6, x_9, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_6);
x_9 = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg(x_1, x_2, x_3, x_4, x_5, x_8, x_7);
return x_9;
}
}
static lean_object* _init_lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_aesop_stats_file;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("", 0, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = 1;
x_5 = lean_io_prim_handle_lock(x_2, x_4);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_dec_ref(x_5);
x_6 = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson(x_1);
x_7 = l_Lean_Json_compress(x_6);
x_8 = l_IO_FS_Handle_putStrLn(x_2, x_7);
if (lean_obj_tag(x_8) == 0)
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_io_prim_handle_unlock(x_2);
if (lean_obj_tag(x_10) == 0)
{
uint8_t x_11; 
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; 
x_12 = lean_ctor_get(x_10, 0);
lean_dec(x_12);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
else
{
lean_object* x_13; 
lean_dec(x_10);
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_9);
return x_13;
}
}
else
{
lean_dec(x_9);
return x_10;
}
}
else
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_8, 0);
lean_inc(x_14);
lean_dec_ref(x_8);
x_15 = lean_io_prim_handle_unlock(x_2);
if (lean_obj_tag(x_15) == 0)
{
uint8_t x_16; 
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; 
x_17 = lean_ctor_get(x_15, 0);
lean_dec(x_17);
lean_ctor_set_tag(x_15, 1);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
else
{
lean_object* x_18; 
lean_dec(x_15);
x_18 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_18, 0, x_14);
return x_18;
}
}
else
{
lean_dec(x_14);
return x_15;
}
}
}
else
{
lean_dec_ref(x_1);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_alloc_closure((void*)(lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = 4;
x_6 = lean_box(x_5);
x_7 = lean_alloc_closure((void*)(l_IO_FS_withFile___boxed), 5, 4);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_1);
lean_closure_set(x_7, 2, x_6);
lean_closure_set(x_7, 3, x_4);
x_8 = lean_apply_2(x_2, lean_box(0), x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
x_12 = lean_apply_4(x_8, lean_box(0), lean_box(0), x_11, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_6);
x_12 = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2(x_1, x_2, x_3, x_4, x_5, x_11, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, uint8_t x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_13 = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__0;
x_14 = l_Lean_Option_get___redArg(x_1, x_12, x_13);
x_15 = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__1;
x_16 = lean_string_dec_eq(x_14, x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_17 = lean_ctor_get(x_2, 1);
lean_inc(x_17);
lean_dec_ref(x_2);
x_18 = lean_alloc_closure((void*)(lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__1), 3, 2);
lean_closure_set(x_18, 0, x_14);
lean_closure_set(x_18, 1, x_3);
x_19 = lean_box(x_9);
lean_inc(x_11);
x_20 = lean_alloc_closure((void*)(lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2___boxed), 10, 9);
lean_closure_set(x_20, 0, x_4);
lean_closure_set(x_20, 1, x_5);
lean_closure_set(x_20, 2, x_6);
lean_closure_set(x_20, 3, x_7);
lean_closure_set(x_20, 4, x_8);
lean_closure_set(x_20, 5, x_19);
lean_closure_set(x_20, 6, x_10);
lean_closure_set(x_20, 7, x_11);
lean_closure_set(x_20, 8, x_18);
x_21 = lean_box(0);
x_22 = lean_apply_2(x_17, lean_box(0), x_21);
x_23 = lean_apply_4(x_11, lean_box(0), lean_box(0), x_22, x_20);
return x_23;
}
else
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; 
lean_dec(x_14);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
x_24 = lean_ctor_get(x_2, 1);
lean_inc(x_24);
lean_dec_ref(x_2);
x_25 = lean_box(0);
x_26 = lean_apply_2(x_24, lean_box(0), x_25);
return x_26;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
uint8_t x_13; lean_object* x_14; 
x_13 = lean_unbox(x_9);
x_14 = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_13, x_10, x_11, x_12);
lean_dec(x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, uint8_t x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = l_Lean_KVMap_instValueString;
x_11 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_11);
x_12 = lean_ctor_get(x_1, 1);
lean_inc(x_12);
x_13 = lean_box(x_9);
lean_inc(x_12);
x_14 = lean_alloc_closure((void*)(lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___boxed), 12, 11);
lean_closure_set(x_14, 0, x_10);
lean_closure_set(x_14, 1, x_11);
lean_closure_set(x_14, 2, x_5);
lean_closure_set(x_14, 3, x_1);
lean_closure_set(x_14, 4, x_2);
lean_closure_set(x_14, 5, x_4);
lean_closure_set(x_14, 6, x_6);
lean_closure_set(x_14, 7, x_7);
lean_closure_set(x_14, 8, x_13);
lean_closure_set(x_14, 9, x_8);
lean_closure_set(x_14, 10, x_12);
x_15 = lean_apply_4(x_12, lean_box(0), lean_box(0), x_3, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, uint8_t x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_10);
x_12 = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; lean_object* x_11; 
x_10 = lean_unbox(x_9);
x_11 = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_10);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Stats_Basic(uint8_t builtin);
lean_object* initialize_Lean_Data_Position(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Stats_File(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Stats_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_Position(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__0 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__0);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__1 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__1();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__1);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__2 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__2();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__2);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__3 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__3();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__3);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__4 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__4();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__4);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__5 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__5();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__5);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__6 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__6();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__6);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__7 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__7();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__7);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__8 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__8();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__8);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__9 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__9();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__9);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__10 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__10();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__10);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__11 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__11();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__11);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__12 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__12();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__12);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__13 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__13();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__13);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__14 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__14();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__14);
lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__15 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__15();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__15);
lp_aesop_Aesop_instToJsonStatsFileRecord___closed__0 = _init_lp_aesop_Aesop_instToJsonStatsFileRecord___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord___closed__0);
lp_aesop_Aesop_instToJsonStatsFileRecord = _init_lp_aesop_Aesop_instToJsonStatsFileRecord();
lean_mark_persistent(lp_aesop_Aesop_instToJsonStatsFileRecord);
lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__0 = _init_lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__0();
lean_mark_persistent(lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__0);
lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__1 = _init_lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__1();
lean_mark_persistent(lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__1);
lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__0 = _init_lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__0();
lean_mark_persistent(lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__0);
lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__1 = _init_lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__1();
lean_mark_persistent(lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__3___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
