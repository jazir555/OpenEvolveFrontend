// Lean compiler output
// Module: Cache.Lean
// Imports: public import Init public import Lean.Data.Json public import Lean.Util.Path
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
LEAN_EXPORT lean_object* lp_mathlib_UInt64_asLTar(uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt64_asLTar___boxed(lean_object*);
lean_object* l_List_findM_x3f___at___00Lean_SearchPath_findWithExt_spec__0(lean_object*, lean_object*, lean_object*);
uint8_t lean_uint8_land(uint8_t, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_System_FilePath_withoutParent_go___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_fromComponents(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_System_FilePath_withoutParent_go(lean_object*, lean_object*);
uint32_t l_Nat_digitChar(lean_object*);
lean_object* lean_uint64_to_nat(uint64_t);
lean_object* l_System_FilePath_components(lean_object*);
lean_object* l_Lean_Name_getRoot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_toHexDigits(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_System_mkFilePath(lean_object*);
static lean_object* lp_mathlib_UInt64_asLTar___closed__1;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_updatePrefix(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_fromComponents_go(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_System_FilePath_withoutParent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_SearchPath_findWithExtBase(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_SearchPath_findWithExtBase___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
static lean_object* lp_mathlib_UInt64_asLTar___closed__0;
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint8_t lean_uint8_shift_right(uint8_t, uint8_t);
uint8_t lean_uint8_of_nat(lean_object*);
lean_object* lean_uint8_to_nat(uint8_t);
lean_object* lean_string_append(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_toHexDigits___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_toHexDigits(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_nat_dec_eq(x_2, x_4);
if (x_5 == 1)
{
lean_dec(x_2);
return x_3;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; uint8_t x_12; uint8_t x_13; lean_object* x_14; uint32_t x_15; lean_object* x_16; uint8_t x_17; uint8_t x_18; lean_object* x_19; uint32_t x_20; lean_object* x_21; 
x_6 = lean_unsigned_to_nat(1u);
x_7 = lean_nat_sub(x_2, x_6);
lean_dec(x_2);
x_8 = lean_unsigned_to_nat(8u);
x_9 = lean_nat_mul(x_7, x_8);
x_10 = lean_nat_shiftr(x_1, x_9);
lean_dec(x_9);
x_11 = lean_uint8_of_nat(x_10);
lean_dec(x_10);
x_12 = 4;
x_13 = lean_uint8_shift_right(x_11, x_12);
x_14 = lean_uint8_to_nat(x_13);
x_15 = l_Nat_digitChar(x_14);
x_16 = lean_string_push(x_3, x_15);
x_17 = 15;
x_18 = lean_uint8_land(x_11, x_17);
x_19 = lean_uint8_to_nat(x_18);
x_20 = l_Nat_digitChar(x_19);
x_21 = lean_string_push(x_16, x_20);
x_2 = x_7;
x_3 = x_21;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_toHexDigits___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Nat_toHexDigits(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_UInt64_asLTar___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("", 0, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_UInt64_asLTar___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(".ltar", 5, 5);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64_asLTar(uint64_t x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lean_uint64_to_nat(x_1);
x_3 = lean_unsigned_to_nat(8u);
x_4 = lp_mathlib_UInt64_asLTar___closed__0;
x_5 = lp_mathlib_Nat_toHexDigits(x_2, x_3, x_4);
lean_dec(x_2);
x_6 = lp_mathlib_UInt64_asLTar___closed__1;
x_7 = lean_string_append(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64_asLTar___boxed(lean_object* x_1) {
_start:
{
uint64_t x_2; lean_object* x_3; 
x_2 = lean_unbox_uint64(x_1);
lean_dec(x_1);
x_3 = lp_mathlib_UInt64_asLTar(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_fromComponents_go(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = l_Lean_Name_updatePrefix(x_3, x_1);
x_1 = x_5;
x_2 = x_4;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_fromComponents(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_box(0);
x_3 = lp_mathlib_Lean_Name_fromComponents_go(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_SearchPath_findWithExtBase(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_5; uint8_t x_6; lean_object* x_7; lean_object* x_8; 
x_5 = l_Lean_Name_getRoot(x_3);
x_6 = 0;
x_7 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_5, x_6);
x_8 = l_List_findM_x3f___at___00Lean_SearchPath_findWithExt_spec__0(x_7, x_2, x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_SearchPath_findWithExtBase___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Lean_SearchPath_findWithExtBase(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_System_FilePath_withoutParent_go(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
return x_1;
}
else
{
if (lean_obj_tag(x_2) == 0)
{
lean_inc_ref(x_1);
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_2, 1);
x_7 = lean_string_dec_eq(x_3, x_5);
if (x_7 == 0)
{
lean_inc_ref(x_1);
return x_1;
}
else
{
x_1 = x_4;
x_2 = x_6;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_System_FilePath_withoutParent_go___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_System_FilePath_withoutParent_go(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_System_FilePath_withoutParent(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = l_System_FilePath_components(x_1);
x_4 = l_System_FilePath_components(x_2);
x_5 = lp_mathlib_System_FilePath_withoutParent_go(x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
x_6 = l_System_mkFilePath(x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Data_Json(uint8_t builtin);
lean_object* initialize_Lean_Util_Path(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Cache_Lean(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_Json(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Util_Path(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_UInt64_asLTar___closed__0 = _init_lp_mathlib_UInt64_asLTar___closed__0();
lean_mark_persistent(lp_mathlib_UInt64_asLTar___closed__0);
lp_mathlib_UInt64_asLTar___closed__1 = _init_lp_mathlib_UInt64_asLTar___closed__1();
lean_mark_persistent(lp_mathlib_UInt64_asLTar___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
