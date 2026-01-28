// Lean compiler output
// Module: Batteries.Data.List.Matcher
// Imports: public import Init public import Batteries.Data.Array.Match
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
lean_object* l_Std_instStreamList___lam__0(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_batteries_Array_Matcher_next_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instStreamProdNat__batteries(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_ofList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instStreamProdNat__batteries___lam__0(lean_object*);
static lean_object* lp_batteries_List_Matcher_findAll___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_find_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_List_Matcher_findAll_loop___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll_loop___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_ofList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_containsInfix___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findAllInfix___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findInfix_x3f___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_List_Matcher_ofList___closed__0;
LEAN_EXPORT uint8_t lp_batteries_List_containsInfix(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Array_Matcher_ofStream___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_containsInfix___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_find_x3f___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findAllInfix(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_containsInfix___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findInfix_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_batteries_List_Matcher_ofList___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Std_instStreamList___lam__0), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_ofList(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_batteries_List_Matcher_ofList___closed__0;
lean_inc(x_3);
x_5 = lp_batteries_Array_Matcher_ofStream___redArg(x_2, x_4, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_ofList___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_batteries_List_Matcher_ofList___closed__0;
lean_inc(x_2);
x_4 = lp_batteries_Array_Matcher_ofStream___redArg(x_1, x_3, x_2);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instStreamProdNat__batteries___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_3; 
lean_dec_ref(x_1);
x_3 = lean_box(0);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_1, 0);
lean_dec(x_6);
x_7 = !lean_is_exclusive(x_2);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_ctor_get(x_2, 1);
x_9 = lean_unsigned_to_nat(1u);
x_10 = lean_nat_add(x_5, x_9);
lean_dec(x_5);
lean_ctor_set(x_1, 1, x_10);
lean_ctor_set(x_1, 0, x_8);
lean_ctor_set_tag(x_2, 0);
lean_ctor_set(x_2, 1, x_1);
x_11 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_11, 0, x_2);
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lean_ctor_get(x_2, 0);
x_13 = lean_ctor_get(x_2, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_2);
x_14 = lean_unsigned_to_nat(1u);
x_15 = lean_nat_add(x_5, x_14);
lean_dec(x_5);
lean_ctor_set(x_1, 1, x_15);
lean_ctor_set(x_1, 0, x_13);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_16, 1, x_1);
x_17 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_17, 0, x_16);
return x_17;
}
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_18 = lean_ctor_get(x_1, 1);
lean_inc(x_18);
lean_dec(x_1);
x_19 = lean_ctor_get(x_2, 0);
lean_inc(x_19);
x_20 = lean_ctor_get(x_2, 1);
lean_inc(x_20);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 x_21 = x_2;
} else {
 lean_dec_ref(x_2);
 x_21 = lean_box(0);
}
x_22 = lean_unsigned_to_nat(1u);
x_23 = lean_nat_add(x_18, x_22);
lean_dec(x_18);
x_24 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_24, 0, x_20);
lean_ctor_set(x_24, 1, x_23);
if (lean_is_scalar(x_21)) {
 x_25 = lean_alloc_ctor(0, 2, 0);
} else {
 x_25 = x_21;
 lean_ctor_set_tag(x_25, 0);
}
lean_ctor_set(x_25, 0, x_19);
lean_ctor_set(x_25, 1, x_24);
x_26 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_26, 0, x_25);
return x_26;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instStreamProdNat__batteries(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_batteries_List_instStreamProdNat__batteries___lam__0), 1, 0);
return x_2;
}
}
static lean_object* _init_lp_batteries_List_Matcher_findAll_loop___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_batteries_List_instStreamProdNat__batteries___lam__0), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll_loop___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_batteries_List_Matcher_findAll_loop___redArg___closed__0;
lean_inc_ref(x_1);
x_7 = lp_batteries_Array_Matcher_next_x3f___redArg(x_1, x_6, x_4, x_3);
if (lean_obj_tag(x_7) == 0)
{
lean_dec_ref(x_1);
return x_5;
}
else
{
lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
x_10 = !lean_is_exclusive(x_8);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_11 = lean_ctor_get(x_2, 0);
x_12 = lean_ctor_get(x_8, 1);
x_13 = lean_ctor_get(x_8, 0);
lean_dec(x_13);
x_14 = lean_ctor_get(x_9, 1);
x_15 = lean_ctor_get(x_11, 0);
x_16 = lean_array_get_size(x_15);
x_17 = lean_nat_sub(x_14, x_16);
lean_inc(x_14);
lean_ctor_set(x_8, 1, x_14);
lean_ctor_set(x_8, 0, x_17);
x_18 = lean_array_push(x_5, x_8);
x_3 = x_9;
x_4 = x_12;
x_5 = x_18;
goto _start;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_20 = lean_ctor_get(x_2, 0);
x_21 = lean_ctor_get(x_8, 1);
lean_inc(x_21);
lean_dec(x_8);
x_22 = lean_ctor_get(x_9, 1);
x_23 = lean_ctor_get(x_20, 0);
x_24 = lean_array_get_size(x_23);
x_25 = lean_nat_sub(x_22, x_24);
lean_inc(x_22);
x_26 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_22);
x_27 = lean_array_push(x_5, x_26);
x_3 = x_9;
x_4 = x_21;
x_5 = x_27;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll_loop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_List_Matcher_findAll_loop___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll_loop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_List_Matcher_findAll_loop(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll_loop___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_List_Matcher_findAll_loop___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
static lean_object* _init_lp_batteries_List_Matcher_findAll___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_4);
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_3);
lean_ctor_set(x_6, 1, x_5);
x_7 = lp_batteries_List_Matcher_findAll___redArg___closed__0;
x_8 = lp_batteries_List_Matcher_findAll_loop___redArg(x_1, x_2, x_6, x_4, x_7);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_findAll(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_List_Matcher_findAll___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_find_x3f___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_2);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_2, 1);
lean_dec(x_6);
x_7 = lp_batteries_List_Matcher_findAll_loop___redArg___closed__0;
x_8 = lean_unsigned_to_nat(0u);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_3);
lean_inc_ref(x_5);
x_9 = lp_batteries_Array_Matcher_next_x3f___redArg(x_1, x_7, x_5, x_2);
if (lean_obj_tag(x_9) == 0)
{
lean_object* x_10; 
lean_dec_ref(x_5);
x_10 = lean_box(0);
return x_10;
}
else
{
uint8_t x_11; 
x_11 = !lean_is_exclusive(x_9);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_12 = lean_ctor_get(x_9, 0);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec(x_12);
x_14 = !lean_is_exclusive(x_13);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_13, 1);
x_16 = lean_ctor_get(x_13, 0);
lean_dec(x_16);
x_17 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_17);
lean_dec_ref(x_5);
x_18 = lean_array_get_size(x_17);
lean_dec_ref(x_17);
x_19 = lean_nat_sub(x_15, x_18);
lean_ctor_set(x_13, 0, x_19);
lean_ctor_set(x_9, 0, x_13);
return x_9;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_20 = lean_ctor_get(x_13, 1);
lean_inc(x_20);
lean_dec(x_13);
x_21 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_21);
lean_dec_ref(x_5);
x_22 = lean_array_get_size(x_21);
lean_dec_ref(x_21);
x_23 = lean_nat_sub(x_20, x_22);
x_24 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_20);
lean_ctor_set(x_9, 0, x_24);
return x_9;
}
}
else
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_25 = lean_ctor_get(x_9, 0);
lean_inc(x_25);
lean_dec(x_9);
x_26 = lean_ctor_get(x_25, 0);
lean_inc(x_26);
lean_dec(x_25);
x_27 = lean_ctor_get(x_26, 1);
lean_inc(x_27);
if (lean_is_exclusive(x_26)) {
 lean_ctor_release(x_26, 0);
 lean_ctor_release(x_26, 1);
 x_28 = x_26;
} else {
 lean_dec_ref(x_26);
 x_28 = lean_box(0);
}
x_29 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_29);
lean_dec_ref(x_5);
x_30 = lean_array_get_size(x_29);
lean_dec_ref(x_29);
x_31 = lean_nat_sub(x_27, x_30);
if (lean_is_scalar(x_28)) {
 x_32 = lean_alloc_ctor(0, 2, 0);
} else {
 x_32 = x_28;
}
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_27);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_32);
return x_33;
}
}
}
else
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_34 = lean_ctor_get(x_2, 0);
lean_inc(x_34);
lean_dec(x_2);
x_35 = lp_batteries_List_Matcher_findAll_loop___redArg___closed__0;
x_36 = lean_unsigned_to_nat(0u);
x_37 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_37, 0, x_3);
lean_ctor_set(x_37, 1, x_36);
lean_inc_ref(x_34);
x_38 = lp_batteries_Array_Matcher_next_x3f___redArg(x_1, x_35, x_34, x_37);
if (lean_obj_tag(x_38) == 0)
{
lean_object* x_39; 
lean_dec_ref(x_34);
x_39 = lean_box(0);
return x_39;
}
else
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; 
x_40 = lean_ctor_get(x_38, 0);
lean_inc(x_40);
if (lean_is_exclusive(x_38)) {
 lean_ctor_release(x_38, 0);
 x_41 = x_38;
} else {
 lean_dec_ref(x_38);
 x_41 = lean_box(0);
}
x_42 = lean_ctor_get(x_40, 0);
lean_inc(x_42);
lean_dec(x_40);
x_43 = lean_ctor_get(x_42, 1);
lean_inc(x_43);
if (lean_is_exclusive(x_42)) {
 lean_ctor_release(x_42, 0);
 lean_ctor_release(x_42, 1);
 x_44 = x_42;
} else {
 lean_dec_ref(x_42);
 x_44 = lean_box(0);
}
x_45 = lean_ctor_get(x_34, 0);
lean_inc_ref(x_45);
lean_dec_ref(x_34);
x_46 = lean_array_get_size(x_45);
lean_dec_ref(x_45);
x_47 = lean_nat_sub(x_43, x_46);
if (lean_is_scalar(x_44)) {
 x_48 = lean_alloc_ctor(0, 2, 0);
} else {
 x_48 = x_44;
}
lean_ctor_set(x_48, 0, x_47);
lean_ctor_set(x_48, 1, x_43);
if (lean_is_scalar(x_41)) {
 x_49 = lean_alloc_ctor(1, 1, 0);
} else {
 x_49 = x_41;
}
lean_ctor_set(x_49, 0, x_48);
return x_49;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Matcher_find_x3f(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_List_Matcher_find_x3f___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findAllInfix(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lp_batteries_List_Matcher_ofList___closed__0;
lean_inc(x_4);
lean_inc_ref(x_2);
x_6 = lp_batteries_Array_Matcher_ofStream___redArg(x_2, x_5, x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
x_8 = lp_batteries_List_Matcher_findAll___redArg(x_2, x_7, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findAllInfix___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lp_batteries_List_Matcher_ofList___closed__0;
lean_inc(x_3);
lean_inc_ref(x_1);
x_5 = lp_batteries_Array_Matcher_ofStream___redArg(x_1, x_4, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_3);
x_7 = lp_batteries_List_Matcher_findAll___redArg(x_1, x_6, x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findInfix_x3f(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lp_batteries_List_Matcher_ofList___closed__0;
lean_inc(x_4);
lean_inc_ref(x_2);
x_6 = lp_batteries_Array_Matcher_ofStream___redArg(x_2, x_5, x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
x_8 = lp_batteries_List_Matcher_find_x3f___redArg(x_2, x_7, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findInfix_x3f___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lp_batteries_List_Matcher_ofList___closed__0;
lean_inc(x_3);
lean_inc_ref(x_1);
x_5 = lp_batteries_Array_Matcher_ofStream___redArg(x_1, x_4, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_3);
x_7 = lp_batteries_List_Matcher_find_x3f___redArg(x_1, x_6, x_2);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_containsInfix(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lp_batteries_List_Matcher_ofList___closed__0;
lean_inc(x_4);
lean_inc_ref(x_2);
x_6 = lp_batteries_Array_Matcher_ofStream___redArg(x_2, x_5, x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
x_8 = lp_batteries_List_Matcher_find_x3f___redArg(x_2, x_7, x_3);
if (lean_obj_tag(x_8) == 0)
{
uint8_t x_9; 
x_9 = 0;
return x_9;
}
else
{
uint8_t x_10; 
lean_dec_ref(x_8);
x_10 = 1;
return x_10;
}
}
}
LEAN_EXPORT uint8_t lp_batteries_List_containsInfix___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lp_batteries_List_Matcher_ofList___closed__0;
lean_inc(x_3);
lean_inc_ref(x_1);
x_5 = lp_batteries_Array_Matcher_ofStream___redArg(x_1, x_4, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_3);
x_7 = lp_batteries_List_Matcher_find_x3f___redArg(x_1, x_6, x_2);
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
LEAN_EXPORT lean_object* lp_batteries_List_containsInfix___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_batteries_List_containsInfix(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_containsInfix___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_batteries_List_containsInfix___redArg(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Array_Match(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_List_Matcher(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Array_Match(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_List_Matcher_ofList___closed__0 = _init_lp_batteries_List_Matcher_ofList___closed__0();
lean_mark_persistent(lp_batteries_List_Matcher_ofList___closed__0);
lp_batteries_List_Matcher_findAll_loop___redArg___closed__0 = _init_lp_batteries_List_Matcher_findAll_loop___redArg___closed__0();
lean_mark_persistent(lp_batteries_List_Matcher_findAll_loop___redArg___closed__0);
lp_batteries_List_Matcher_findAll___redArg___closed__0 = _init_lp_batteries_List_Matcher_findAll___redArg___closed__0();
lean_mark_persistent(lp_batteries_List_Matcher_findAll___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
