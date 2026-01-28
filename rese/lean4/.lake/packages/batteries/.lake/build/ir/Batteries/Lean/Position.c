// Lean compiler output
// Module: Batteries.Lean.Position
// Imports: public import Init public import Lean.Syntax public import Lean.Data.Lsp.Utf16
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
lean_object* l_Lean_FileMap_utf8RangeToLspRange(lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_prevAux_go___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0___boxed(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_next_x21(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_FileMap_rangeOfStx_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_pos_x21(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findLineStart(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lean_findLineStart___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_FileMap_rangeOfStx_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findIndentAndIsStart(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findIndentAndIsStart___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_FileMap_rangeOfStx_x3f(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = 0;
x_4 = l_Lean_Syntax_getRange_x3f(x_2, x_3);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; 
lean_dec_ref(x_1);
x_5 = lean_box(0);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_4);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_4, 0);
x_8 = l_Lean_FileMap_utf8RangeToLspRange(x_1, x_7);
lean_ctor_set(x_4, 0, x_8);
return x_4;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_4, 0);
lean_inc(x_9);
lean_dec(x_4);
x_10 = l_Lean_FileMap_utf8RangeToLspRange(x_1, x_9);
x_11 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_FileMap_rangeOfStx_x3f___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Lean_FileMap_rangeOfStx_x3f(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_3);
if (x_5 == 0)
{
lean_object* x_6; uint32_t x_7; lean_object* x_8; uint8_t x_9; 
x_6 = lean_ctor_get(x_3, 0);
x_7 = lean_ctor_get_uint32(x_3, sizeof(void*)*1);
x_8 = lean_unsigned_to_nat(0u);
x_9 = lean_nat_dec_eq(x_6, x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint32_t x_16; uint8_t x_17; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
x_12 = lean_unsigned_to_nat(1u);
x_13 = lean_nat_sub(x_6, x_12);
lean_dec(x_6);
x_14 = l_String_Slice_Pos_prevAux_go___redArg(x_2, x_13);
x_15 = lean_nat_add(x_11, x_14);
x_16 = lean_string_utf8_get_fast(x_10, x_15);
lean_dec(x_15);
x_17 = lean_uint32_dec_eq(x_16, x_7);
if (x_17 == 0)
{
lean_ctor_set(x_3, 0, x_14);
{
lean_object* _tmp_3 = x_1;
x_4 = _tmp_3;
}
goto _start;
}
else
{
lean_object* x_19; 
lean_free_object(x_3);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_14);
return x_19;
}
}
else
{
lean_free_object(x_3);
lean_dec(x_6);
lean_inc(x_4);
return x_4;
}
}
else
{
lean_object* x_20; uint32_t x_21; lean_object* x_22; uint8_t x_23; 
x_20 = lean_ctor_get(x_3, 0);
x_21 = lean_ctor_get_uint32(x_3, sizeof(void*)*1);
lean_inc(x_20);
lean_dec(x_3);
x_22 = lean_unsigned_to_nat(0u);
x_23 = lean_nat_dec_eq(x_20, x_22);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; uint32_t x_30; uint8_t x_31; 
x_24 = lean_ctor_get(x_2, 0);
x_25 = lean_ctor_get(x_2, 1);
x_26 = lean_unsigned_to_nat(1u);
x_27 = lean_nat_sub(x_20, x_26);
lean_dec(x_20);
x_28 = l_String_Slice_Pos_prevAux_go___redArg(x_2, x_27);
x_29 = lean_nat_add(x_25, x_28);
x_30 = lean_string_utf8_get_fast(x_24, x_29);
lean_dec(x_29);
x_31 = lean_uint32_dec_eq(x_30, x_21);
if (x_31 == 0)
{
lean_object* x_32; 
x_32 = lean_alloc_ctor(0, 1, 4);
lean_ctor_set(x_32, 0, x_28);
lean_ctor_set_uint32(x_32, sizeof(void*)*1, x_21);
{
lean_object* _tmp_2 = x_32;
lean_object* _tmp_3 = x_1;
x_3 = _tmp_2;
x_4 = _tmp_3;
}
goto _start;
}
else
{
lean_object* x_34; 
x_34 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_34, 0, x_28);
return x_34;
}
}
else
{
lean_dec(x_20);
lean_inc(x_4);
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg(x_1, x_2, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint32_t x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lean_ctor_get(x_1, 1);
x_3 = lean_ctor_get(x_1, 2);
x_4 = 10;
x_5 = lean_nat_sub(x_3, x_2);
x_6 = lean_alloc_ctor(0, 1, 4);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set_uint32(x_6, sizeof(void*)*1, x_4);
x_7 = lean_box(0);
x_8 = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg(x_7, x_1, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findLineStart(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_9 = lean_unsigned_to_nat(0u);
x_10 = lean_string_utf8_byte_size(x_1);
lean_inc_ref(x_1);
x_11 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_11, 0, x_1);
lean_ctor_set(x_11, 1, x_9);
lean_ctor_set(x_11, 2, x_10);
x_12 = l_String_Slice_pos_x21(x_11, x_2);
lean_dec_ref(x_11);
lean_inc_ref(x_1);
x_13 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_13, 0, x_1);
lean_ctor_set(x_13, 1, x_9);
lean_ctor_set(x_13, 2, x_12);
x_14 = lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0(x_13);
lean_dec_ref(x_13);
if (lean_obj_tag(x_14) == 0)
{
if (lean_obj_tag(x_14) == 0)
{
lean_dec_ref(x_1);
return x_9;
}
else
{
lean_object* x_15; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_3 = x_15;
goto block_8;
}
}
else
{
lean_object* x_16; 
x_16 = lean_ctor_get(x_14, 0);
lean_inc(x_16);
lean_dec_ref(x_14);
x_3 = x_16;
goto block_8;
}
block_8:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_string_utf8_byte_size(x_1);
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_1);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_5);
x_7 = l_String_Slice_Pos_next_x21(x_6, x_3);
lean_dec(x_3);
lean_dec_ref(x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findLineStart___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Lean_findLineStart(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_7 = lean_ctor_get(x_1, 1);
x_8 = lean_ctor_get(x_1, 2);
x_9 = lean_nat_sub(x_8, x_7);
x_10 = lean_nat_dec_eq(x_5, x_9);
lean_dec(x_9);
if (x_10 == 0)
{
lean_object* x_11; uint32_t x_12; uint32_t x_13; uint8_t x_14; 
x_11 = lean_nat_add(x_2, x_5);
x_12 = lean_string_utf8_get_fast(x_3, x_11);
x_13 = 32;
x_14 = lean_uint32_dec_eq(x_12, x_13);
if (x_14 == 0)
{
lean_object* x_15; 
lean_dec(x_11);
x_15 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_15, 0, x_5);
return x_15;
}
else
{
if (x_10 == 0)
{
lean_object* x_16; lean_object* x_17; 
lean_dec(x_5);
x_16 = lean_string_utf8_next_fast(x_3, x_11);
lean_dec(x_11);
x_17 = lean_nat_sub(x_16, x_2);
{
lean_object* _tmp_4 = x_17;
lean_object* _tmp_5 = x_4;
x_5 = _tmp_4;
x_6 = _tmp_5;
}
goto _start;
}
else
{
lean_object* x_19; 
lean_dec(x_11);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_5);
return x_19;
}
}
}
else
{
lean_dec(x_5);
lean_inc(x_6);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg(x_1, x_2, x_3, x_4, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findIndentAndIsStart(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_inc_ref(x_1);
x_3 = lp_batteries_Lean_findLineStart(x_1, x_2);
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_string_utf8_byte_size(x_1);
lean_inc_ref(x_1);
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_1);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_5);
x_7 = l_String_Slice_pos_x21(x_6, x_3);
lean_dec_ref(x_6);
lean_inc(x_7);
lean_inc_ref(x_1);
x_15 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_15, 0, x_1);
lean_ctor_set(x_15, 1, x_7);
lean_ctor_set(x_15, 2, x_5);
x_16 = lean_box(0);
x_17 = lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg(x_15, x_7, x_1, x_16, x_4, x_16);
lean_dec_ref(x_1);
lean_dec_ref(x_15);
if (lean_obj_tag(x_17) == 0)
{
lean_object* x_18; 
x_18 = lean_nat_sub(x_5, x_7);
x_8 = x_18;
goto block_14;
}
else
{
lean_object* x_19; 
x_19 = lean_ctor_get(x_17, 0);
lean_inc(x_19);
lean_dec_ref(x_17);
x_8 = x_19;
goto block_14;
}
block_14:
{
lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_nat_add(x_7, x_8);
lean_dec(x_8);
lean_dec(x_7);
x_10 = lean_nat_sub(x_9, x_3);
lean_dec(x_3);
x_11 = lean_nat_dec_eq(x_9, x_2);
lean_dec(x_9);
x_12 = lean_box(x_11);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_10);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_8);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findIndentAndIsStart___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Lean_findIndentAndIsStart(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Syntax(uint8_t builtin);
lean_object* initialize_Lean_Data_Lsp_Utf16(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Position(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_Lsp_Utf16(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
