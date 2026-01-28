// Lean compiler output
// Module: Batteries.Data.String.Basic
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
LEAN_EXPORT lean_object* lp_batteries_String_toAsciiByteArray_loop(lean_object*, lean_object*, lean_object*);
uint8_t lean_uint32_to_uint8(uint32_t);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
static lean_object* lp_batteries_instCoeStringRaw__batteries___closed__0;
LEAN_EXPORT lean_object* lp_batteries_instCoeStringRaw__batteries;
lean_object* lean_byte_array_push(lean_object*, uint8_t);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_String_toRawSubstring(lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0(uint32_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_ByteArray_empty;
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_utf8_at_end(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_toAsciiByteArray(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_toAsciiByteArray_loop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0___redArg(uint32_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_count(lean_object*, uint32_t);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint32_t lean_string_utf8_get(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
LEAN_EXPORT lean_object* lp_batteries_String_count___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_toAsciiByteArray___boxed(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_string_utf8_next(lean_object*, lean_object*);
lean_object* l_String_Slice_positions(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
static lean_object* _init_lp_batteries_instCoeStringRaw__batteries___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_String_toRawSubstring), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_instCoeStringRaw__batteries() {
_start:
{
lean_object* x_1; 
x_1 = lp_batteries_instCoeStringRaw__batteries___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0___redArg(uint32_t x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_6 = lean_ctor_get(x_2, 1);
x_7 = lean_ctor_get(x_2, 2);
x_8 = lean_nat_sub(x_7, x_6);
x_9 = lean_nat_dec_eq(x_4, x_8);
lean_dec(x_8);
if (x_9 == 0)
{
lean_object* x_10; uint32_t x_11; uint8_t x_12; 
x_10 = lean_string_utf8_next_fast(x_3, x_4);
x_11 = lean_string_utf8_get_fast(x_3, x_4);
lean_dec(x_4);
x_12 = lean_uint32_dec_eq(x_11, x_1);
if (x_12 == 0)
{
x_4 = x_10;
goto _start;
}
else
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_unsigned_to_nat(1u);
x_15 = lean_nat_add(x_5, x_14);
lean_dec(x_5);
x_4 = x_10;
x_5 = x_15;
goto _start;
}
}
else
{
lean_dec(x_4);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0(uint32_t x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0___redArg(x_1, x_2, x_3, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_count(lean_object* x_1, uint32_t x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_string_utf8_byte_size(x_1);
lean_inc_ref(x_1);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
x_6 = l_String_Slice_positions(x_5);
x_7 = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0___redArg(x_2, x_5, x_1, x_6, x_3);
lean_dec_ref(x_1);
lean_dec_ref(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint32_t x_9; lean_object* x_10; 
x_9 = lean_unbox_uint32(x_1);
lean_dec(x_1);
x_10 = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0(x_9, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_count___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint32_t x_3; lean_object* x_4; 
x_3 = lean_unbox_uint32(x_2);
lean_dec(x_2);
x_4 = lp_batteries_String_count(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint32_t x_6; lean_object* x_7; 
x_6 = lean_unbox_uint32(x_1);
lean_dec(x_1);
x_7 = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_count_spec__0___redArg(x_6, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_toAsciiByteArray_loop(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_string_utf8_at_end(x_1, x_2);
if (x_4 == 0)
{
uint32_t x_5; lean_object* x_6; uint8_t x_7; lean_object* x_8; 
x_5 = lean_string_utf8_get(x_1, x_2);
x_6 = lean_string_utf8_next(x_1, x_2);
lean_dec(x_2);
x_7 = lean_uint32_to_uint8(x_5);
x_8 = lean_byte_array_push(x_3, x_7);
x_2 = x_6;
x_3 = x_8;
goto _start;
}
else
{
lean_dec(x_2);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_String_toAsciiByteArray_loop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_String_toAsciiByteArray_loop(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_toAsciiByteArray(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = l_ByteArray_empty;
x_4 = lp_batteries_String_toAsciiByteArray_loop(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_toAsciiByteArray___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_String_toAsciiByteArray(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_String_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_instCoeStringRaw__batteries___closed__0 = _init_lp_batteries_instCoeStringRaw__batteries___closed__0();
lean_mark_persistent(lp_batteries_instCoeStringRaw__batteries___closed__0);
lp_batteries_instCoeStringRaw__batteries = _init_lp_batteries_instCoeStringRaw__batteries();
lean_mark_persistent(lp_batteries_instCoeStringRaw__batteries);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
