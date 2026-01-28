// Lean compiler output
// Module: Batteries.Data.Char.AsciiCasing
// Imports: public import Init public import Batteries.Data.Char.Basic public import Batteries.Tactic.Basic
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
LEAN_EXPORT lean_object* lp_batteries_Char_cmpCaseInsensitiveAsciiOnly___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Char_beqCaseInsensitiveAsciiOnly___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Char_cmpCaseInsensitiveAsciiOnly(uint32_t, uint32_t);
uint32_t l_Char_toLower(uint32_t);
LEAN_EXPORT lean_object* lp_batteries_Char_caseFoldAsciiOnly___boxed(lean_object*);
LEAN_EXPORT uint32_t lp_batteries_Char_caseFoldAsciiOnly(uint32_t);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
uint8_t lean_uint32_dec_lt(uint32_t, uint32_t);
LEAN_EXPORT lean_object* lp_batteries_Char_beqCaseInsensitiveAsciiOnly_isSetoid;
LEAN_EXPORT uint8_t lp_batteries_Char_beqCaseInsensitiveAsciiOnly(uint32_t, uint32_t);
LEAN_EXPORT uint32_t lp_batteries_Char_caseFoldAsciiOnly(uint32_t x_1) {
_start:
{
uint32_t x_2; 
x_2 = l_Char_toLower(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Char_caseFoldAsciiOnly___boxed(lean_object* x_1) {
_start:
{
uint32_t x_2; uint32_t x_3; lean_object* x_4; 
x_2 = lean_unbox_uint32(x_1);
lean_dec(x_1);
x_3 = lp_batteries_Char_caseFoldAsciiOnly(x_2);
x_4 = lean_box_uint32(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_batteries_Char_beqCaseInsensitiveAsciiOnly(uint32_t x_1, uint32_t x_2) {
_start:
{
uint32_t x_3; uint32_t x_4; uint8_t x_5; 
x_3 = l_Char_toLower(x_1);
x_4 = l_Char_toLower(x_2);
x_5 = lean_uint32_dec_eq(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Char_beqCaseInsensitiveAsciiOnly___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint32_t x_3; uint32_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox_uint32(x_1);
lean_dec(x_1);
x_4 = lean_unbox_uint32(x_2);
lean_dec(x_2);
x_5 = lp_batteries_Char_beqCaseInsensitiveAsciiOnly(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
static lean_object* _init_lp_batteries_Char_beqCaseInsensitiveAsciiOnly_isSetoid() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_batteries_Char_cmpCaseInsensitiveAsciiOnly(uint32_t x_1, uint32_t x_2) {
_start:
{
uint32_t x_3; uint32_t x_4; uint8_t x_5; 
x_3 = l_Char_toLower(x_1);
x_4 = l_Char_toLower(x_2);
x_5 = lean_uint32_dec_lt(x_3, x_4);
if (x_5 == 0)
{
uint8_t x_6; 
x_6 = lean_uint32_dec_eq(x_3, x_4);
if (x_6 == 0)
{
uint8_t x_7; 
x_7 = 2;
return x_7;
}
else
{
uint8_t x_8; 
x_8 = 1;
return x_8;
}
}
else
{
uint8_t x_9; 
x_9 = 0;
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Char_cmpCaseInsensitiveAsciiOnly___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint32_t x_3; uint32_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox_uint32(x_1);
lean_dec(x_1);
x_4 = lean_unbox_uint32(x_2);
lean_dec(x_2);
x_5 = lp_batteries_Char_cmpCaseInsensitiveAsciiOnly(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Char_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_Char_AsciiCasing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Char_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Char_beqCaseInsensitiveAsciiOnly_isSetoid = _init_lp_batteries_Char_beqCaseInsensitiveAsciiOnly_isSetoid();
lean_mark_persistent(lp_batteries_Char_beqCaseInsensitiveAsciiOnly_isSetoid);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
