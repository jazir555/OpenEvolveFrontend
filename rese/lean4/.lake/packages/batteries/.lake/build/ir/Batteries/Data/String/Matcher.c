// Lean compiler output
// Module: Batteries.Data.String.Matcher
// Imports: public import Init public import Batteries.Data.Array.Match public import Batteries.Data.String.Basic
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
LEAN_EXPORT lean_object* lp_batteries_Substring_Raw_containsSubstr___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_String_containsSubstr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Substring_findSubstr_x3f(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_containsSubstr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_find_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_PrefixTable_step___at___00Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0_spec__0(lean_object*, uint32_t, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_instDecidableEqChar___boxed(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_findAll(lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_findAllSubstr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Substring_Raw_findSubstr_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Substring_Raw_findAllSubstr(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_findSubstr_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_ofString(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_findAll_loop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Substring_findAllSubstr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_PrefixTable_step___at___00Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Array_Matcher_ofStream___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_ofSubstring(lean_object*);
uint32_t lean_string_utf8_get(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Substring_containsSubstr___boxed(lean_object*, lean_object*);
static lean_object* lp_batteries_String_Matcher_findAll___closed__0;
static lean_object* lp_batteries_String_Matcher_ofSubstring___closed__1;
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
static lean_object* lp_batteries_String_Matcher_ofSubstring___closed__0;
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_patternSize(lean_object*);
lean_object* lean_string_utf8_next(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Substring_containsSubstr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_patternSize___boxed(lean_object*);
lean_object* l_instStreamRawChar___lam__0(lean_object*);
lean_object* lean_array_get_size(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Substring_Raw_containsSubstr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
static lean_object* _init_lp_batteries_String_Matcher_ofSubstring___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instStreamRawChar___lam__0), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_String_Matcher_ofSubstring___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_alloc_closure((void*)(l_instDecidableEqChar___boxed), 2, 0);
x_2 = l_instBEqOfDecidableEq___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_ofSubstring(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_batteries_String_Matcher_ofSubstring___closed__0;
x_3 = lp_batteries_String_Matcher_ofSubstring___closed__1;
lean_inc_ref(x_1);
x_4 = lp_batteries_Array_Matcher_ofStream___redArg(x_3, x_2, x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_ofString(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_string_utf8_byte_size(x_1);
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_1);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_3);
x_5 = lp_batteries_String_Matcher_ofSubstring___closed__0;
x_6 = lp_batteries_String_Matcher_ofSubstring___closed__1;
lean_inc_ref(x_4);
x_7 = lp_batteries_Array_Matcher_ofStream___redArg(x_6, x_5, x_4);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_patternSize(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 1);
x_3 = lean_ctor_get(x_2, 1);
x_4 = lean_ctor_get(x_2, 2);
x_5 = lean_nat_sub(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_patternSize___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_String_Matcher_patternSize(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_PrefixTable_step___at___00Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0_spec__0(lean_object* x_1, uint32_t x_2, lean_object* x_3) {
_start:
{
lean_object* x_12; uint8_t x_13; 
x_12 = lean_array_get_size(x_1);
x_13 = lean_nat_dec_lt(x_3, x_12);
if (x_13 == 0)
{
goto block_11;
}
else
{
lean_object* x_14; lean_object* x_15; uint32_t x_16; uint8_t x_17; 
x_14 = lean_array_fget_borrowed(x_1, x_3);
x_15 = lean_ctor_get(x_14, 0);
x_16 = lean_unbox_uint32(x_15);
x_17 = lean_uint32_dec_eq(x_2, x_16);
if (x_17 == 0)
{
goto block_11;
}
else
{
lean_object* x_18; lean_object* x_19; 
x_18 = lean_unsigned_to_nat(1u);
x_19 = lean_nat_add(x_3, x_18);
lean_dec(x_3);
return x_19;
}
}
block_11:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_nat_dec_eq(x_3, x_4);
if (x_5 == 1)
{
lean_dec(x_3);
return x_4;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_unsigned_to_nat(1u);
x_7 = lean_nat_sub(x_3, x_6);
lean_dec(x_3);
x_8 = lean_array_fget_borrowed(x_1, x_7);
lean_dec(x_7);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
x_3 = x_9;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_2, 2);
x_7 = lean_nat_dec_lt(x_5, x_6);
if (x_7 == 0)
{
lean_object* x_8; 
lean_free_object(x_2);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_1);
x_8 = lean_box(0);
return x_8;
}
else
{
uint8_t x_9; 
x_9 = !lean_is_exclusive(x_1);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; uint32_t x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_10 = lean_ctor_get(x_1, 0);
x_11 = lean_ctor_get(x_1, 1);
x_12 = lean_string_utf8_get(x_4, x_5);
x_13 = lean_string_utf8_next(x_4, x_5);
lean_dec(x_5);
lean_ctor_set(x_2, 1, x_13);
x_14 = lp_batteries_Array_PrefixTable_step___at___00Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0_spec__0(x_10, x_12, x_11);
x_15 = lean_array_get_size(x_10);
x_16 = lean_nat_dec_eq(x_14, x_15);
if (x_16 == 0)
{
lean_ctor_set(x_1, 1, x_14);
goto _start;
}
else
{
lean_object* x_18; lean_object* x_19; 
lean_ctor_set(x_1, 1, x_14);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_2);
lean_ctor_set(x_18, 1, x_1);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
else
{
lean_object* x_20; lean_object* x_21; uint32_t x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; 
x_20 = lean_ctor_get(x_1, 0);
x_21 = lean_ctor_get(x_1, 1);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_1);
x_22 = lean_string_utf8_get(x_4, x_5);
x_23 = lean_string_utf8_next(x_4, x_5);
lean_dec(x_5);
lean_ctor_set(x_2, 1, x_23);
x_24 = lp_batteries_Array_PrefixTable_step___at___00Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0_spec__0(x_20, x_22, x_21);
x_25 = lean_array_get_size(x_20);
x_26 = lean_nat_dec_eq(x_24, x_25);
if (x_26 == 0)
{
lean_object* x_27; 
x_27 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_27, 0, x_20);
lean_ctor_set(x_27, 1, x_24);
x_1 = x_27;
goto _start;
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_20);
lean_ctor_set(x_29, 1, x_24);
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_2);
lean_ctor_set(x_30, 1, x_29);
x_31 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_31, 0, x_30);
return x_31;
}
}
}
}
else
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; uint8_t x_35; 
x_32 = lean_ctor_get(x_2, 0);
x_33 = lean_ctor_get(x_2, 1);
x_34 = lean_ctor_get(x_2, 2);
lean_inc(x_34);
lean_inc(x_33);
lean_inc(x_32);
lean_dec(x_2);
x_35 = lean_nat_dec_lt(x_33, x_34);
if (x_35 == 0)
{
lean_object* x_36; 
lean_dec(x_34);
lean_dec(x_33);
lean_dec_ref(x_32);
lean_dec_ref(x_1);
x_36 = lean_box(0);
return x_36;
}
else
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; uint32_t x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; uint8_t x_45; 
x_37 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_37);
x_38 = lean_ctor_get(x_1, 1);
lean_inc(x_38);
if (lean_is_exclusive(x_1)) {
 lean_ctor_release(x_1, 0);
 lean_ctor_release(x_1, 1);
 x_39 = x_1;
} else {
 lean_dec_ref(x_1);
 x_39 = lean_box(0);
}
x_40 = lean_string_utf8_get(x_32, x_33);
x_41 = lean_string_utf8_next(x_32, x_33);
lean_dec(x_33);
x_42 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_42, 0, x_32);
lean_ctor_set(x_42, 1, x_41);
lean_ctor_set(x_42, 2, x_34);
x_43 = lp_batteries_Array_PrefixTable_step___at___00Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0_spec__0(x_37, x_40, x_38);
x_44 = lean_array_get_size(x_37);
x_45 = lean_nat_dec_eq(x_43, x_44);
if (x_45 == 0)
{
lean_object* x_46; 
if (lean_is_scalar(x_39)) {
 x_46 = lean_alloc_ctor(0, 2, 0);
} else {
 x_46 = x_39;
}
lean_ctor_set(x_46, 0, x_37);
lean_ctor_set(x_46, 1, x_43);
x_1 = x_46;
x_2 = x_42;
goto _start;
}
else
{
lean_object* x_48; lean_object* x_49; lean_object* x_50; 
if (lean_is_scalar(x_39)) {
 x_48 = lean_alloc_ctor(0, 2, 0);
} else {
 x_48 = x_39;
}
lean_ctor_set(x_48, 0, x_37);
lean_ctor_set(x_48, 1, x_43);
x_49 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_49, 0, x_42);
lean_ctor_set(x_49, 1, x_48);
x_50 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_50, 0, x_49);
return x_50;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_findAll_loop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0(x_3, x_2);
if (lean_obj_tag(x_5) == 0)
{
lean_dec_ref(x_1);
return x_4;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 1);
lean_inc(x_9);
lean_dec(x_6);
x_10 = !lean_is_exclusive(x_7);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_11 = lean_ctor_get(x_7, 1);
x_12 = lean_ctor_get(x_7, 2);
x_13 = lean_ctor_get(x_7, 0);
lean_dec(x_13);
x_14 = lean_ctor_get(x_8, 0);
x_15 = lean_ctor_get(x_8, 1);
x_16 = lean_nat_sub(x_12, x_11);
lean_dec(x_11);
lean_dec(x_12);
x_17 = lean_nat_sub(x_15, x_16);
lean_dec(x_16);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_ctor_set(x_7, 2, x_15);
lean_ctor_set(x_7, 1, x_17);
lean_ctor_set(x_7, 0, x_14);
x_18 = lean_array_push(x_4, x_7);
x_2 = x_8;
x_3 = x_9;
x_4 = x_18;
goto _start;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_20 = lean_ctor_get(x_7, 1);
x_21 = lean_ctor_get(x_7, 2);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_7);
x_22 = lean_ctor_get(x_8, 0);
x_23 = lean_ctor_get(x_8, 1);
x_24 = lean_nat_sub(x_21, x_20);
lean_dec(x_20);
lean_dec(x_21);
x_25 = lean_nat_sub(x_23, x_24);
lean_dec(x_24);
lean_inc(x_23);
lean_inc_ref(x_22);
x_26 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_26, 0, x_22);
lean_ctor_set(x_26, 1, x_25);
lean_ctor_set(x_26, 2, x_23);
x_27 = lean_array_push(x_4, x_26);
x_2 = x_8;
x_3 = x_9;
x_4 = x_27;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_PrefixTable_step___at___00Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint32_t x_4; lean_object* x_5; 
x_4 = lean_unbox_uint32(x_2);
lean_dec(x_2);
x_5 = lp_batteries_Array_PrefixTable_step___at___00Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0_spec__0(x_1, x_4, x_3);
lean_dec_ref(x_1);
return x_5;
}
}
static lean_object* _init_lp_batteries_String_Matcher_findAll___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_findAll(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lp_batteries_String_Matcher_findAll___closed__0;
x_5 = lp_batteries_String_Matcher_findAll_loop(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Matcher_find_x3f(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lp_batteries_Array_Matcher_next_x3f___at___00String_Matcher_findAll_loop_spec__0(x_3, x_2);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; 
lean_dec_ref(x_4);
x_6 = lean_box(0);
return x_6;
}
else
{
uint8_t x_7; 
x_7 = !lean_is_exclusive(x_5);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_8 = lean_ctor_get(x_5, 0);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec(x_8);
x_10 = lean_ctor_get(x_4, 1);
lean_inc(x_10);
x_11 = lean_ctor_get(x_4, 2);
lean_inc(x_11);
lean_dec_ref(x_4);
x_12 = !lean_is_exclusive(x_9);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_13 = lean_ctor_get(x_9, 1);
x_14 = lean_ctor_get(x_9, 2);
lean_dec(x_14);
x_15 = lean_nat_sub(x_11, x_10);
lean_dec(x_10);
lean_dec(x_11);
x_16 = lean_nat_sub(x_13, x_15);
lean_dec(x_15);
lean_ctor_set(x_9, 2, x_13);
lean_ctor_set(x_9, 1, x_16);
lean_ctor_set(x_5, 0, x_9);
return x_5;
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_17 = lean_ctor_get(x_9, 0);
x_18 = lean_ctor_get(x_9, 1);
lean_inc(x_18);
lean_inc(x_17);
lean_dec(x_9);
x_19 = lean_nat_sub(x_11, x_10);
lean_dec(x_10);
lean_dec(x_11);
x_20 = lean_nat_sub(x_18, x_19);
lean_dec(x_19);
x_21 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_21, 0, x_17);
lean_ctor_set(x_21, 1, x_20);
lean_ctor_set(x_21, 2, x_18);
lean_ctor_set(x_5, 0, x_21);
return x_5;
}
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_22 = lean_ctor_get(x_5, 0);
lean_inc(x_22);
lean_dec(x_5);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec(x_22);
x_24 = lean_ctor_get(x_4, 1);
lean_inc(x_24);
x_25 = lean_ctor_get(x_4, 2);
lean_inc(x_25);
lean_dec_ref(x_4);
x_26 = lean_ctor_get(x_23, 0);
lean_inc_ref(x_26);
x_27 = lean_ctor_get(x_23, 1);
lean_inc(x_27);
if (lean_is_exclusive(x_23)) {
 lean_ctor_release(x_23, 0);
 lean_ctor_release(x_23, 1);
 lean_ctor_release(x_23, 2);
 x_28 = x_23;
} else {
 lean_dec_ref(x_23);
 x_28 = lean_box(0);
}
x_29 = lean_nat_sub(x_25, x_24);
lean_dec(x_24);
lean_dec(x_25);
x_30 = lean_nat_sub(x_27, x_29);
lean_dec(x_29);
if (lean_is_scalar(x_28)) {
 x_31 = lean_alloc_ctor(0, 3, 0);
} else {
 x_31 = x_28;
}
lean_ctor_set(x_31, 0, x_26);
lean_ctor_set(x_31, 1, x_30);
lean_ctor_set(x_31, 2, x_27);
x_32 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_32, 0, x_31);
return x_32;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Substring_Raw_findAllSubstr(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lp_batteries_String_Matcher_ofSubstring___closed__0;
x_4 = lp_batteries_String_Matcher_ofSubstring___closed__1;
lean_inc_ref(x_2);
x_5 = lp_batteries_Array_Matcher_ofStream___redArg(x_4, x_3, x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
x_7 = lp_batteries_String_Matcher_findAll(x_6, x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Substring_Raw_findSubstr_x3f(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lp_batteries_String_Matcher_ofSubstring___closed__0;
x_4 = lp_batteries_String_Matcher_ofSubstring___closed__1;
lean_inc_ref(x_2);
x_5 = lp_batteries_Array_Matcher_ofStream___redArg(x_4, x_3, x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
x_7 = lp_batteries_String_Matcher_find_x3f(x_6, x_1);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_batteries_Substring_Raw_containsSubstr(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lp_batteries_String_Matcher_ofSubstring___closed__0;
x_4 = lp_batteries_String_Matcher_ofSubstring___closed__1;
lean_inc_ref(x_2);
x_5 = lp_batteries_Array_Matcher_ofStream___redArg(x_4, x_3, x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
x_7 = lp_batteries_String_Matcher_find_x3f(x_6, x_1);
if (lean_obj_tag(x_7) == 0)
{
uint8_t x_8; 
x_8 = 0;
return x_8;
}
else
{
uint8_t x_9; 
lean_dec_ref(x_7);
x_9 = 1;
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Substring_Raw_containsSubstr___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_batteries_Substring_Raw_containsSubstr(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Substring_findAllSubstr(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lp_batteries_String_Matcher_ofSubstring___closed__0;
x_4 = lp_batteries_String_Matcher_ofSubstring___closed__1;
lean_inc_ref(x_2);
x_5 = lp_batteries_Array_Matcher_ofStream___redArg(x_4, x_3, x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
x_7 = lp_batteries_String_Matcher_findAll(x_6, x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Substring_findSubstr_x3f(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lp_batteries_String_Matcher_ofSubstring___closed__0;
x_4 = lp_batteries_String_Matcher_ofSubstring___closed__1;
lean_inc_ref(x_2);
x_5 = lp_batteries_Array_Matcher_ofStream___redArg(x_4, x_3, x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
x_7 = lp_batteries_String_Matcher_find_x3f(x_6, x_1);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_batteries_Substring_containsSubstr(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lp_batteries_String_Matcher_ofSubstring___closed__0;
x_4 = lp_batteries_String_Matcher_ofSubstring___closed__1;
lean_inc_ref(x_2);
x_5 = lp_batteries_Array_Matcher_ofStream___redArg(x_4, x_3, x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
x_7 = lp_batteries_String_Matcher_find_x3f(x_6, x_1);
if (lean_obj_tag(x_7) == 0)
{
uint8_t x_8; 
x_8 = 0;
return x_8;
}
else
{
uint8_t x_9; 
lean_dec_ref(x_7);
x_9 = 1;
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Substring_containsSubstr___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_batteries_Substring_containsSubstr(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_findAllSubstr(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_3 = lp_batteries_String_Matcher_ofSubstring___closed__0;
x_4 = lp_batteries_String_Matcher_ofSubstring___closed__1;
lean_inc_ref(x_2);
x_5 = lp_batteries_Array_Matcher_ofStream___redArg(x_4, x_3, x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_2);
x_7 = lean_unsigned_to_nat(0u);
x_8 = lean_string_utf8_byte_size(x_1);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_1);
lean_ctor_set(x_9, 1, x_7);
lean_ctor_set(x_9, 2, x_8);
x_10 = lp_batteries_String_Matcher_findAll(x_6, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_findSubstr_x3f(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_string_utf8_byte_size(x_1);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
x_6 = lp_batteries_String_Matcher_ofSubstring___closed__0;
x_7 = lp_batteries_String_Matcher_ofSubstring___closed__1;
lean_inc_ref(x_2);
x_8 = lp_batteries_Array_Matcher_ofStream___redArg(x_7, x_6, x_2);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
x_10 = lp_batteries_String_Matcher_find_x3f(x_9, x_5);
return x_10;
}
}
LEAN_EXPORT uint8_t lp_batteries_String_containsSubstr(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_string_utf8_byte_size(x_1);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
x_6 = lp_batteries_String_Matcher_ofSubstring___closed__0;
x_7 = lp_batteries_String_Matcher_ofSubstring___closed__1;
lean_inc_ref(x_2);
x_8 = lp_batteries_Array_Matcher_ofStream___redArg(x_7, x_6, x_2);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_2);
x_10 = lp_batteries_String_Matcher_find_x3f(x_9, x_5);
if (lean_obj_tag(x_10) == 0)
{
uint8_t x_11; 
x_11 = 0;
return x_11;
}
else
{
uint8_t x_12; 
lean_dec_ref(x_10);
x_12 = 1;
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_String_containsSubstr___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_batteries_String_containsSubstr(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Array_Match(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_String_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_String_Matcher(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Array_Match(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_String_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_String_Matcher_ofSubstring___closed__0 = _init_lp_batteries_String_Matcher_ofSubstring___closed__0();
lean_mark_persistent(lp_batteries_String_Matcher_ofSubstring___closed__0);
lp_batteries_String_Matcher_ofSubstring___closed__1 = _init_lp_batteries_String_Matcher_ofSubstring___closed__1();
lean_mark_persistent(lp_batteries_String_Matcher_ofSubstring___closed__1);
lp_batteries_String_Matcher_findAll___closed__0 = _init_lp_batteries_String_Matcher_findAll___closed__0();
lean_mark_persistent(lp_batteries_String_Matcher_findAll___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
