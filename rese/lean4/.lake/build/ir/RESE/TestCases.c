// Lean compiler output
// Module: RESE.TestCases
// Imports: public import Init public import RESE.Basic public import RESE.Constraint public import RESE.Templates
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
static lean_object* lp_rese_RESE_TestCases_tempMin___closed__3;
LEAN_EXPORT lean_object* lp_rese_RESE_TestCases_tempMax;
static lean_object* lp_rese_RESE_TestCases_tempMax___closed__0;
static lean_object* lp_rese_RESE_TestCases_pressureMax___closed__2;
static lean_object* lp_rese_RESE_TestCases_tempMin___closed__1;
LEAN_EXPORT lean_object* lp_rese_RESE_TestCases_tempMin;
static lean_object* lp_rese_RESE_TestCases_pressureMax___closed__0;
static lean_object* lp_rese_RESE_TestCases_tempMin___closed__0;
static lean_object* lp_rese_RESE_TestCases_tempMax___closed__1;
LEAN_EXPORT lean_object* lp_rese_RESE_TestCases_pressureMax;
static lean_object* lp_rese_RESE_TestCases_tempMax___closed__2;
static lean_object* lp_rese_RESE_TestCases_tempMax___closed__3;
static lean_object* lp_rese_RESE_TestCases_pressureMax___closed__1;
static lean_object* lp_rese_RESE_TestCases_tempMin___closed__2;
static lean_object* lp_rese_RESE_TestCases_pressureMax___closed__3;
static lean_object* _init_lp_rese_RESE_TestCases_tempMax___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("temp_max", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_tempMax___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Temperature must be less than 1000°C", 37, 36);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_tempMax___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("user_prompt", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_tempMax___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; uint8_t x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_rese_RESE_TestCases_tempMax___closed__2;
x_2 = lean_box(0);
x_3 = lp_rese_RESE_TestCases_tempMax___closed__1;
x_4 = 0;
x_5 = lp_rese_RESE_TestCases_tempMax___closed__0;
x_6 = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_3);
lean_ctor_set(x_6, 2, x_2);
lean_ctor_set(x_6, 3, x_1);
lean_ctor_set_uint8(x_6, sizeof(void*)*4, x_4);
return x_6;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_tempMax() {
_start:
{
lean_object* x_1; 
x_1 = lp_rese_RESE_TestCases_tempMax___closed__3;
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_tempMin___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("temp_min", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_tempMin___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Temperature must be greater than 500°C", 39, 38);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_tempMin___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_rese_RESE_TestCases_tempMax___closed__0;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_tempMin___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; uint8_t x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_rese_RESE_TestCases_tempMax___closed__2;
x_2 = lp_rese_RESE_TestCases_tempMin___closed__2;
x_3 = lp_rese_RESE_TestCases_tempMin___closed__1;
x_4 = 0;
x_5 = lp_rese_RESE_TestCases_tempMin___closed__0;
x_6 = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_3);
lean_ctor_set(x_6, 2, x_2);
lean_ctor_set(x_6, 3, x_1);
lean_ctor_set_uint8(x_6, sizeof(void*)*4, x_4);
return x_6;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_tempMin() {
_start:
{
lean_object* x_1; 
x_1 = lp_rese_RESE_TestCases_tempMin___closed__3;
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_pressureMax___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("pressure_max", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_pressureMax___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Pressure should preferably be below 10 bar", 42, 42);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_pressureMax___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("system_inferred", 15, 15);
return x_1;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_pressureMax___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; uint8_t x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_rese_RESE_TestCases_pressureMax___closed__2;
x_2 = lean_box(0);
x_3 = lp_rese_RESE_TestCases_pressureMax___closed__1;
x_4 = 1;
x_5 = lp_rese_RESE_TestCases_pressureMax___closed__0;
x_6 = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_3);
lean_ctor_set(x_6, 2, x_2);
lean_ctor_set(x_6, 3, x_1);
lean_ctor_set_uint8(x_6, sizeof(void*)*4, x_4);
return x_6;
}
}
static lean_object* _init_lp_rese_RESE_TestCases_pressureMax() {
_start:
{
lean_object* x_1; 
x_1 = lp_rese_RESE_TestCases_pressureMax___closed__3;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_rese_RESE_Basic(uint8_t builtin);
lean_object* initialize_rese_RESE_Constraint(uint8_t builtin);
lean_object* initialize_rese_RESE_Templates(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_rese_RESE_TestCases(uint8_t builtin) {
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
lp_rese_RESE_TestCases_tempMax___closed__0 = _init_lp_rese_RESE_TestCases_tempMax___closed__0();
lean_mark_persistent(lp_rese_RESE_TestCases_tempMax___closed__0);
lp_rese_RESE_TestCases_tempMax___closed__1 = _init_lp_rese_RESE_TestCases_tempMax___closed__1();
lean_mark_persistent(lp_rese_RESE_TestCases_tempMax___closed__1);
lp_rese_RESE_TestCases_tempMax___closed__2 = _init_lp_rese_RESE_TestCases_tempMax___closed__2();
lean_mark_persistent(lp_rese_RESE_TestCases_tempMax___closed__2);
lp_rese_RESE_TestCases_tempMax___closed__3 = _init_lp_rese_RESE_TestCases_tempMax___closed__3();
lean_mark_persistent(lp_rese_RESE_TestCases_tempMax___closed__3);
lp_rese_RESE_TestCases_tempMax = _init_lp_rese_RESE_TestCases_tempMax();
lean_mark_persistent(lp_rese_RESE_TestCases_tempMax);
lp_rese_RESE_TestCases_tempMin___closed__0 = _init_lp_rese_RESE_TestCases_tempMin___closed__0();
lean_mark_persistent(lp_rese_RESE_TestCases_tempMin___closed__0);
lp_rese_RESE_TestCases_tempMin___closed__1 = _init_lp_rese_RESE_TestCases_tempMin___closed__1();
lean_mark_persistent(lp_rese_RESE_TestCases_tempMin___closed__1);
lp_rese_RESE_TestCases_tempMin___closed__2 = _init_lp_rese_RESE_TestCases_tempMin___closed__2();
lean_mark_persistent(lp_rese_RESE_TestCases_tempMin___closed__2);
lp_rese_RESE_TestCases_tempMin___closed__3 = _init_lp_rese_RESE_TestCases_tempMin___closed__3();
lean_mark_persistent(lp_rese_RESE_TestCases_tempMin___closed__3);
lp_rese_RESE_TestCases_tempMin = _init_lp_rese_RESE_TestCases_tempMin();
lean_mark_persistent(lp_rese_RESE_TestCases_tempMin);
lp_rese_RESE_TestCases_pressureMax___closed__0 = _init_lp_rese_RESE_TestCases_pressureMax___closed__0();
lean_mark_persistent(lp_rese_RESE_TestCases_pressureMax___closed__0);
lp_rese_RESE_TestCases_pressureMax___closed__1 = _init_lp_rese_RESE_TestCases_pressureMax___closed__1();
lean_mark_persistent(lp_rese_RESE_TestCases_pressureMax___closed__1);
lp_rese_RESE_TestCases_pressureMax___closed__2 = _init_lp_rese_RESE_TestCases_pressureMax___closed__2();
lean_mark_persistent(lp_rese_RESE_TestCases_pressureMax___closed__2);
lp_rese_RESE_TestCases_pressureMax___closed__3 = _init_lp_rese_RESE_TestCases_pressureMax___closed__3();
lean_mark_persistent(lp_rese_RESE_TestCases_pressureMax___closed__3);
lp_rese_RESE_TestCases_pressureMax = _init_lp_rese_RESE_TestCases_pressureMax();
lean_mark_persistent(lp_rese_RESE_TestCases_pressureMax);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
