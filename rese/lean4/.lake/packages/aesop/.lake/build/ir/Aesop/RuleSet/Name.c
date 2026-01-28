// Lean compiler output
// Module: Aesop.RuleSet.Name
// Imports: public import Init public import Lean
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
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0___boxed(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_builtinRuleSetNames___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_defaultRuleSetName;
LEAN_EXPORT lean_object* lp_aesop_Aesop_builtinRuleSetNames;
static lean_object* lp_aesop_Aesop_builtinRuleSetName___closed__1;
size_t lean_usize_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleSetName_isReserved___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_builtinRuleSetName;
static lean_object* lp_aesop_Aesop_defaultRuleSetName___closed__1;
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleSetName_isReserved(lean_object*);
static lean_object* lp_aesop_Aesop_defaultRuleSetName___closed__0;
uint8_t lean_name_eq(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_localRuleSetName___closed__1;
static lean_object* lp_aesop_Aesop_builtinRuleSetName___closed__0;
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_localRuleSetName___closed__0;
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_get_size(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_localRuleSetName;
static lean_object* lp_aesop_Aesop_builtinRuleSetNames___closed__1;
static lean_object* lp_aesop_Aesop_builtinRuleSetNames___closed__0;
static lean_object* _init_lp_aesop_Aesop_defaultRuleSetName___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("default", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_defaultRuleSetName___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_defaultRuleSetName___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_defaultRuleSetName() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_defaultRuleSetName___closed__1;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_builtinRuleSetName___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("builtin", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_builtinRuleSetName___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_builtinRuleSetName___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_builtinRuleSetName() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_builtinRuleSetName___closed__1;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_localRuleSetName___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("local", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_localRuleSetName___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_localRuleSetName___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_localRuleSetName() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_localRuleSetName___closed__1;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_builtinRuleSetNames___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_builtinRuleSetNames___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_defaultRuleSetName;
x_2 = lp_aesop_Aesop_builtinRuleSetNames___closed__0;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_builtinRuleSetNames___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_builtinRuleSetName;
x_2 = lp_aesop_Aesop_builtinRuleSetNames___closed__1;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_builtinRuleSetNames() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_builtinRuleSetNames___closed__2;
return x_1;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0(lean_object* x_1, lean_object* x_2, size_t x_3, size_t x_4) {
_start:
{
uint8_t x_5; 
x_5 = lean_usize_dec_eq(x_3, x_4);
if (x_5 == 0)
{
lean_object* x_6; uint8_t x_7; 
x_6 = lean_array_uget(x_2, x_3);
x_7 = lean_name_eq(x_1, x_6);
lean_dec(x_6);
if (x_7 == 0)
{
size_t x_8; size_t x_9; 
x_8 = 1;
x_9 = lean_usize_add(x_3, x_8);
x_3 = x_9;
goto _start;
}
else
{
return x_7;
}
}
else
{
uint8_t x_11; 
x_11 = 0;
return x_11;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_array_get_size(x_1);
x_5 = lean_nat_dec_lt(x_3, x_4);
if (x_5 == 0)
{
return x_5;
}
else
{
if (x_5 == 0)
{
return x_5;
}
else
{
size_t x_6; size_t x_7; uint8_t x_8; 
x_6 = 0;
x_7 = lean_usize_of_nat(x_4);
x_8 = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0(x_2, x_1, x_6, x_7);
return x_8;
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleSetName_isReserved(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lp_aesop_Aesop_localRuleSetName;
x_3 = lean_name_eq(x_1, x_2);
if (x_3 == 0)
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_aesop_Aesop_builtinRuleSetNames;
x_5 = lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0(x_4, x_1);
return x_5;
}
else
{
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleSetName_isReserved___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_RuleSetName_isReserved(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
size_t x_5; size_t x_6; uint8_t x_7; lean_object* x_8; 
x_5 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_6 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_7 = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0(x_1, x_2, x_5, x_6);
lean_dec_ref(x_2);
lean_dec(x_1);
x_8 = lean_box(x_7);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleSet_Name(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_defaultRuleSetName___closed__0 = _init_lp_aesop_Aesop_defaultRuleSetName___closed__0();
lean_mark_persistent(lp_aesop_Aesop_defaultRuleSetName___closed__0);
lp_aesop_Aesop_defaultRuleSetName___closed__1 = _init_lp_aesop_Aesop_defaultRuleSetName___closed__1();
lean_mark_persistent(lp_aesop_Aesop_defaultRuleSetName___closed__1);
lp_aesop_Aesop_defaultRuleSetName = _init_lp_aesop_Aesop_defaultRuleSetName();
lean_mark_persistent(lp_aesop_Aesop_defaultRuleSetName);
lp_aesop_Aesop_builtinRuleSetName___closed__0 = _init_lp_aesop_Aesop_builtinRuleSetName___closed__0();
lean_mark_persistent(lp_aesop_Aesop_builtinRuleSetName___closed__0);
lp_aesop_Aesop_builtinRuleSetName___closed__1 = _init_lp_aesop_Aesop_builtinRuleSetName___closed__1();
lean_mark_persistent(lp_aesop_Aesop_builtinRuleSetName___closed__1);
lp_aesop_Aesop_builtinRuleSetName = _init_lp_aesop_Aesop_builtinRuleSetName();
lean_mark_persistent(lp_aesop_Aesop_builtinRuleSetName);
lp_aesop_Aesop_localRuleSetName___closed__0 = _init_lp_aesop_Aesop_localRuleSetName___closed__0();
lean_mark_persistent(lp_aesop_Aesop_localRuleSetName___closed__0);
lp_aesop_Aesop_localRuleSetName___closed__1 = _init_lp_aesop_Aesop_localRuleSetName___closed__1();
lean_mark_persistent(lp_aesop_Aesop_localRuleSetName___closed__1);
lp_aesop_Aesop_localRuleSetName = _init_lp_aesop_Aesop_localRuleSetName();
lean_mark_persistent(lp_aesop_Aesop_localRuleSetName);
lp_aesop_Aesop_builtinRuleSetNames___closed__0 = _init_lp_aesop_Aesop_builtinRuleSetNames___closed__0();
lean_mark_persistent(lp_aesop_Aesop_builtinRuleSetNames___closed__0);
lp_aesop_Aesop_builtinRuleSetNames___closed__1 = _init_lp_aesop_Aesop_builtinRuleSetNames___closed__1();
lean_mark_persistent(lp_aesop_Aesop_builtinRuleSetNames___closed__1);
lp_aesop_Aesop_builtinRuleSetNames___closed__2 = _init_lp_aesop_Aesop_builtinRuleSetNames___closed__2();
lean_mark_persistent(lp_aesop_Aesop_builtinRuleSetNames___closed__2);
lp_aesop_Aesop_builtinRuleSetNames = _init_lp_aesop_Aesop_builtinRuleSetNames();
lean_mark_persistent(lp_aesop_Aesop_builtinRuleSetNames);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
