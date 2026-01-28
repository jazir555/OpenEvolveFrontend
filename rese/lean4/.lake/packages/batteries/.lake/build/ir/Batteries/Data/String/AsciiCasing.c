// Lean compiler output
// Module: Batteries.Data.String.AsciiCasing
// Imports: public import Init public import Batteries.Data.Char public import Batteries.Data.Char.AsciiCasing
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
uint8_t lp_batteries_Char_cmpCaseInsensitiveAsciiOnly(uint32_t, uint32_t);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_cmpCaseInsensitiveAsciiOnlyImpl___boxed(lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_cmpCaseInsensitiveAsciiOnlyImpl_loop(lean_object*, lean_object*);
lean_object* lp_batteries_Char_caseFoldAsciiOnly___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_caseFoldAsciiOnly(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_beqCaseInsensitiveAsciiOnlyImpl_loop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_beqCaseInsensitiveAsciiOnly_isSetoid;
lean_object* l_String_mapAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_beqCaseInsensitiveAsciiOnlyImpl(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint32_t lean_string_utf8_get(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_beqCaseInsensitiveAsciiOnlyImpl_loop(lean_object*, lean_object*);
lean_object* lean_string_utf8_next(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_cmpCaseInsensitiveAsciiOnlyImpl(lean_object*, lean_object*);
static lean_object* lp_batteries_String_caseFoldAsciiOnly___closed__0;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_cmpCaseInsensitiveAsciiOnlyImpl_loop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_beqCaseInsensitiveAsciiOnlyImpl___boxed(lean_object*, lean_object*);
uint8_t lp_batteries_Char_beqCaseInsensitiveAsciiOnly(uint32_t, uint32_t);
static lean_object* _init_lp_batteries_String_caseFoldAsciiOnly___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_batteries_Char_caseFoldAsciiOnly___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_caseFoldAsciiOnly(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_batteries_String_caseFoldAsciiOnly___closed__0;
x_3 = lean_unsigned_to_nat(0u);
x_4 = l_String_mapAux(x_2, x_1, x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_beqCaseInsensitiveAsciiOnlyImpl_loop(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_1, 2);
x_7 = lean_nat_dec_lt(x_5, x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; uint8_t x_10; 
lean_free_object(x_1);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
lean_dec_ref(x_2);
x_10 = lean_nat_dec_lt(x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
if (x_10 == 0)
{
uint8_t x_11; 
x_11 = 1;
return x_11;
}
else
{
return x_7;
}
}
else
{
uint8_t x_12; 
x_12 = !lean_is_exclusive(x_2);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_13 = lean_ctor_get(x_2, 0);
x_14 = lean_ctor_get(x_2, 1);
x_15 = lean_ctor_get(x_2, 2);
x_16 = lean_nat_dec_lt(x_14, x_15);
if (x_16 == 0)
{
lean_free_object(x_2);
lean_dec(x_15);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_free_object(x_1);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_16;
}
else
{
uint32_t x_17; uint32_t x_18; uint8_t x_19; 
x_17 = lean_string_utf8_get(x_4, x_5);
x_18 = lean_string_utf8_get(x_13, x_14);
x_19 = lp_batteries_Char_beqCaseInsensitiveAsciiOnly(x_17, x_18);
if (x_19 == 0)
{
lean_free_object(x_2);
lean_dec(x_15);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_free_object(x_1);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_19;
}
else
{
lean_object* x_20; lean_object* x_21; 
x_20 = lean_string_utf8_next(x_4, x_5);
lean_dec(x_5);
lean_ctor_set(x_2, 2, x_6);
lean_ctor_set(x_2, 1, x_20);
lean_ctor_set(x_2, 0, x_4);
x_21 = lean_string_utf8_next(x_13, x_14);
lean_dec(x_14);
lean_ctor_set(x_1, 2, x_15);
lean_ctor_set(x_1, 1, x_21);
lean_ctor_set(x_1, 0, x_13);
{
lean_object* _tmp_0 = x_2;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
}
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; 
x_23 = lean_ctor_get(x_2, 0);
x_24 = lean_ctor_get(x_2, 1);
x_25 = lean_ctor_get(x_2, 2);
lean_inc(x_25);
lean_inc(x_24);
lean_inc(x_23);
lean_dec(x_2);
x_26 = lean_nat_dec_lt(x_24, x_25);
if (x_26 == 0)
{
lean_dec(x_25);
lean_dec(x_24);
lean_dec_ref(x_23);
lean_free_object(x_1);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_26;
}
else
{
uint32_t x_27; uint32_t x_28; uint8_t x_29; 
x_27 = lean_string_utf8_get(x_4, x_5);
x_28 = lean_string_utf8_get(x_23, x_24);
x_29 = lp_batteries_Char_beqCaseInsensitiveAsciiOnly(x_27, x_28);
if (x_29 == 0)
{
lean_dec(x_25);
lean_dec(x_24);
lean_dec_ref(x_23);
lean_free_object(x_1);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_29;
}
else
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_30 = lean_string_utf8_next(x_4, x_5);
lean_dec(x_5);
x_31 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_31, 0, x_4);
lean_ctor_set(x_31, 1, x_30);
lean_ctor_set(x_31, 2, x_6);
x_32 = lean_string_utf8_next(x_23, x_24);
lean_dec(x_24);
lean_ctor_set(x_1, 2, x_25);
lean_ctor_set(x_1, 1, x_32);
lean_ctor_set(x_1, 0, x_23);
{
lean_object* _tmp_0 = x_31;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
}
}
}
}
else
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; uint8_t x_37; 
x_34 = lean_ctor_get(x_1, 0);
x_35 = lean_ctor_get(x_1, 1);
x_36 = lean_ctor_get(x_1, 2);
lean_inc(x_36);
lean_inc(x_35);
lean_inc(x_34);
lean_dec(x_1);
x_37 = lean_nat_dec_lt(x_35, x_36);
if (x_37 == 0)
{
lean_object* x_38; lean_object* x_39; uint8_t x_40; 
lean_dec(x_36);
lean_dec(x_35);
lean_dec_ref(x_34);
x_38 = lean_ctor_get(x_2, 1);
lean_inc(x_38);
x_39 = lean_ctor_get(x_2, 2);
lean_inc(x_39);
lean_dec_ref(x_2);
x_40 = lean_nat_dec_lt(x_38, x_39);
lean_dec(x_39);
lean_dec(x_38);
if (x_40 == 0)
{
uint8_t x_41; 
x_41 = 1;
return x_41;
}
else
{
return x_37;
}
}
else
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; uint8_t x_46; 
x_42 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_42);
x_43 = lean_ctor_get(x_2, 1);
lean_inc(x_43);
x_44 = lean_ctor_get(x_2, 2);
lean_inc(x_44);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 x_45 = x_2;
} else {
 lean_dec_ref(x_2);
 x_45 = lean_box(0);
}
x_46 = lean_nat_dec_lt(x_43, x_44);
if (x_46 == 0)
{
lean_dec(x_45);
lean_dec(x_44);
lean_dec(x_43);
lean_dec_ref(x_42);
lean_dec(x_36);
lean_dec(x_35);
lean_dec_ref(x_34);
return x_46;
}
else
{
uint32_t x_47; uint32_t x_48; uint8_t x_49; 
x_47 = lean_string_utf8_get(x_34, x_35);
x_48 = lean_string_utf8_get(x_42, x_43);
x_49 = lp_batteries_Char_beqCaseInsensitiveAsciiOnly(x_47, x_48);
if (x_49 == 0)
{
lean_dec(x_45);
lean_dec(x_44);
lean_dec(x_43);
lean_dec_ref(x_42);
lean_dec(x_36);
lean_dec(x_35);
lean_dec_ref(x_34);
return x_49;
}
else
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; 
x_50 = lean_string_utf8_next(x_34, x_35);
lean_dec(x_35);
if (lean_is_scalar(x_45)) {
 x_51 = lean_alloc_ctor(0, 3, 0);
} else {
 x_51 = x_45;
}
lean_ctor_set(x_51, 0, x_34);
lean_ctor_set(x_51, 1, x_50);
lean_ctor_set(x_51, 2, x_36);
x_52 = lean_string_utf8_next(x_42, x_43);
lean_dec(x_43);
x_53 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_53, 0, x_42);
lean_ctor_set(x_53, 1, x_52);
lean_ctor_set(x_53, 2, x_44);
x_1 = x_51;
x_2 = x_53;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_beqCaseInsensitiveAsciiOnlyImpl_loop___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_beqCaseInsensitiveAsciiOnlyImpl_loop(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_beqCaseInsensitiveAsciiOnlyImpl(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_string_length(x_1);
x_4 = lean_string_length(x_2);
x_5 = lean_nat_dec_eq(x_3, x_4);
if (x_5 == 0)
{
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_string_utf8_byte_size(x_1);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_1);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_7);
x_9 = lean_string_utf8_byte_size(x_2);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_2);
lean_ctor_set(x_10, 1, x_6);
lean_ctor_set(x_10, 2, x_9);
x_11 = lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_beqCaseInsensitiveAsciiOnlyImpl_loop(x_8, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_beqCaseInsensitiveAsciiOnlyImpl___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_beqCaseInsensitiveAsciiOnlyImpl(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_batteries_String_beqCaseInsensitiveAsciiOnly_isSetoid() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_cmpCaseInsensitiveAsciiOnlyImpl_loop(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_1, 2);
x_7 = lean_nat_dec_lt(x_5, x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; uint8_t x_10; 
lean_free_object(x_1);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
lean_dec_ref(x_2);
x_10 = lean_nat_dec_lt(x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
if (x_10 == 0)
{
uint8_t x_11; 
x_11 = 1;
return x_11;
}
else
{
uint8_t x_12; 
x_12 = 0;
return x_12;
}
}
else
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_2);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_14 = lean_ctor_get(x_2, 0);
x_15 = lean_ctor_get(x_2, 1);
x_16 = lean_ctor_get(x_2, 2);
x_17 = lean_nat_dec_lt(x_15, x_16);
if (x_17 == 0)
{
uint8_t x_18; 
lean_free_object(x_2);
lean_dec(x_16);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_free_object(x_1);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
x_18 = 2;
return x_18;
}
else
{
uint32_t x_19; uint32_t x_20; uint8_t x_21; 
x_19 = lean_string_utf8_get(x_4, x_5);
x_20 = lean_string_utf8_get(x_14, x_15);
x_21 = lp_batteries_Char_cmpCaseInsensitiveAsciiOnly(x_19, x_20);
if (x_21 == 1)
{
lean_object* x_22; lean_object* x_23; 
x_22 = lean_string_utf8_next(x_4, x_5);
lean_dec(x_5);
lean_ctor_set(x_2, 2, x_6);
lean_ctor_set(x_2, 1, x_22);
lean_ctor_set(x_2, 0, x_4);
x_23 = lean_string_utf8_next(x_14, x_15);
lean_dec(x_15);
lean_ctor_set(x_1, 2, x_16);
lean_ctor_set(x_1, 1, x_23);
lean_ctor_set(x_1, 0, x_14);
{
lean_object* _tmp_0 = x_2;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_free_object(x_2);
lean_dec(x_16);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_free_object(x_1);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_21;
}
}
}
else
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; uint8_t x_28; 
x_25 = lean_ctor_get(x_2, 0);
x_26 = lean_ctor_get(x_2, 1);
x_27 = lean_ctor_get(x_2, 2);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_25);
lean_dec(x_2);
x_28 = lean_nat_dec_lt(x_26, x_27);
if (x_28 == 0)
{
uint8_t x_29; 
lean_dec(x_27);
lean_dec(x_26);
lean_dec_ref(x_25);
lean_free_object(x_1);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
x_29 = 2;
return x_29;
}
else
{
uint32_t x_30; uint32_t x_31; uint8_t x_32; 
x_30 = lean_string_utf8_get(x_4, x_5);
x_31 = lean_string_utf8_get(x_25, x_26);
x_32 = lp_batteries_Char_cmpCaseInsensitiveAsciiOnly(x_30, x_31);
if (x_32 == 1)
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_33 = lean_string_utf8_next(x_4, x_5);
lean_dec(x_5);
x_34 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_34, 0, x_4);
lean_ctor_set(x_34, 1, x_33);
lean_ctor_set(x_34, 2, x_6);
x_35 = lean_string_utf8_next(x_25, x_26);
lean_dec(x_26);
lean_ctor_set(x_1, 2, x_27);
lean_ctor_set(x_1, 1, x_35);
lean_ctor_set(x_1, 0, x_25);
{
lean_object* _tmp_0 = x_34;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_dec(x_27);
lean_dec(x_26);
lean_dec_ref(x_25);
lean_free_object(x_1);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_32;
}
}
}
}
}
else
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; uint8_t x_40; 
x_37 = lean_ctor_get(x_1, 0);
x_38 = lean_ctor_get(x_1, 1);
x_39 = lean_ctor_get(x_1, 2);
lean_inc(x_39);
lean_inc(x_38);
lean_inc(x_37);
lean_dec(x_1);
x_40 = lean_nat_dec_lt(x_38, x_39);
if (x_40 == 0)
{
lean_object* x_41; lean_object* x_42; uint8_t x_43; 
lean_dec(x_39);
lean_dec(x_38);
lean_dec_ref(x_37);
x_41 = lean_ctor_get(x_2, 1);
lean_inc(x_41);
x_42 = lean_ctor_get(x_2, 2);
lean_inc(x_42);
lean_dec_ref(x_2);
x_43 = lean_nat_dec_lt(x_41, x_42);
lean_dec(x_42);
lean_dec(x_41);
if (x_43 == 0)
{
uint8_t x_44; 
x_44 = 1;
return x_44;
}
else
{
uint8_t x_45; 
x_45 = 0;
return x_45;
}
}
else
{
lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; uint8_t x_50; 
x_46 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_46);
x_47 = lean_ctor_get(x_2, 1);
lean_inc(x_47);
x_48 = lean_ctor_get(x_2, 2);
lean_inc(x_48);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 x_49 = x_2;
} else {
 lean_dec_ref(x_2);
 x_49 = lean_box(0);
}
x_50 = lean_nat_dec_lt(x_47, x_48);
if (x_50 == 0)
{
uint8_t x_51; 
lean_dec(x_49);
lean_dec(x_48);
lean_dec(x_47);
lean_dec_ref(x_46);
lean_dec(x_39);
lean_dec(x_38);
lean_dec_ref(x_37);
x_51 = 2;
return x_51;
}
else
{
uint32_t x_52; uint32_t x_53; uint8_t x_54; 
x_52 = lean_string_utf8_get(x_37, x_38);
x_53 = lean_string_utf8_get(x_46, x_47);
x_54 = lp_batteries_Char_cmpCaseInsensitiveAsciiOnly(x_52, x_53);
if (x_54 == 1)
{
lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; 
x_55 = lean_string_utf8_next(x_37, x_38);
lean_dec(x_38);
if (lean_is_scalar(x_49)) {
 x_56 = lean_alloc_ctor(0, 3, 0);
} else {
 x_56 = x_49;
}
lean_ctor_set(x_56, 0, x_37);
lean_ctor_set(x_56, 1, x_55);
lean_ctor_set(x_56, 2, x_39);
x_57 = lean_string_utf8_next(x_46, x_47);
lean_dec(x_47);
x_58 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_58, 0, x_46);
lean_ctor_set(x_58, 1, x_57);
lean_ctor_set(x_58, 2, x_48);
x_1 = x_56;
x_2 = x_58;
goto _start;
}
else
{
lean_dec(x_49);
lean_dec(x_48);
lean_dec(x_47);
lean_dec_ref(x_46);
lean_dec(x_39);
lean_dec(x_38);
lean_dec_ref(x_37);
return x_54;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_cmpCaseInsensitiveAsciiOnlyImpl_loop___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_cmpCaseInsensitiveAsciiOnlyImpl_loop(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_cmpCaseInsensitiveAsciiOnlyImpl(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_string_utf8_byte_size(x_1);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_4);
x_6 = lean_string_utf8_byte_size(x_2);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_2);
lean_ctor_set(x_7, 1, x_3);
lean_ctor_set(x_7, 2, x_6);
x_8 = lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_cmpCaseInsensitiveAsciiOnlyImpl_loop(x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_cmpCaseInsensitiveAsciiOnlyImpl___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_batteries___private_Batteries_Data_String_AsciiCasing_0__String_cmpCaseInsensitiveAsciiOnlyImpl(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Char(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Char_AsciiCasing(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_String_AsciiCasing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Char(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Char_AsciiCasing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_String_caseFoldAsciiOnly___closed__0 = _init_lp_batteries_String_caseFoldAsciiOnly___closed__0();
lean_mark_persistent(lp_batteries_String_caseFoldAsciiOnly___closed__0);
lp_batteries_String_beqCaseInsensitiveAsciiOnly_isSetoid = _init_lp_batteries_String_beqCaseInsensitiveAsciiOnly_isSetoid();
lean_mark_persistent(lp_batteries_String_beqCaseInsensitiveAsciiOnly_isSetoid);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
