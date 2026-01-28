// Lean compiler output
// Module: Aesop.Index.DiscrTreeConfig
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkDiscrTreePath___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Config_toConfigWithKey(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_indexConfig;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_indexConfig___closed__0;
lean_object* l_Lean_Meta_DiscrTree_getUnify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_mkPath(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkDiscrTreePath(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_indexConfig___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_getMatch___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_indexConfig___closed__0() {
_start:
{
uint8_t x_1; uint8_t x_2; uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_1 = 0;
x_2 = 0;
x_3 = 2;
x_4 = 1;
x_5 = 0;
x_6 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_6, 0, x_5);
lean_ctor_set_uint8(x_6, 1, x_5);
lean_ctor_set_uint8(x_6, 2, x_5);
lean_ctor_set_uint8(x_6, 3, x_5);
lean_ctor_set_uint8(x_6, 4, x_5);
lean_ctor_set_uint8(x_6, 5, x_4);
lean_ctor_set_uint8(x_6, 6, x_4);
lean_ctor_set_uint8(x_6, 7, x_5);
lean_ctor_set_uint8(x_6, 8, x_4);
lean_ctor_set_uint8(x_6, 9, x_3);
lean_ctor_set_uint8(x_6, 10, x_2);
lean_ctor_set_uint8(x_6, 11, x_4);
lean_ctor_set_uint8(x_6, 12, x_4);
lean_ctor_set_uint8(x_6, 13, x_4);
lean_ctor_set_uint8(x_6, 14, x_1);
lean_ctor_set_uint8(x_6, 15, x_4);
lean_ctor_set_uint8(x_6, 16, x_4);
lean_ctor_set_uint8(x_6, 17, x_4);
lean_ctor_set_uint8(x_6, 18, x_4);
return x_6;
}
}
static lean_object* _init_lp_aesop_Aesop_indexConfig___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_indexConfig___closed__0;
x_2 = l_Lean_Meta_Config_toConfigWithKey(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_indexConfig() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_indexConfig___closed__1;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkDiscrTreePath(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; uint8_t x_8; 
x_7 = lp_aesop_Aesop_indexConfig;
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
uint8_t x_9; 
x_9 = !lean_is_exclusive(x_2);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; uint64_t x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_7, 0);
x_11 = lean_ctor_get(x_2, 0);
lean_dec(x_11);
x_12 = 0;
x_13 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_10);
lean_ctor_set_uint64(x_7, sizeof(void*)*1, x_13);
lean_ctor_set(x_2, 0, x_7);
x_14 = l_Lean_Meta_DiscrTree_mkPath(x_1, x_12, x_2, x_3, x_4, x_5);
return x_14;
}
else
{
lean_object* x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; uint8_t x_24; uint8_t x_25; uint64_t x_26; lean_object* x_27; lean_object* x_28; 
x_15 = lean_ctor_get(x_7, 0);
x_16 = lean_ctor_get_uint8(x_2, sizeof(void*)*7);
x_17 = lean_ctor_get(x_2, 1);
x_18 = lean_ctor_get(x_2, 2);
x_19 = lean_ctor_get(x_2, 3);
x_20 = lean_ctor_get(x_2, 4);
x_21 = lean_ctor_get(x_2, 5);
x_22 = lean_ctor_get(x_2, 6);
x_23 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 1);
x_24 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 2);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_dec(x_2);
x_25 = 0;
x_26 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_15);
lean_ctor_set_uint64(x_7, sizeof(void*)*1, x_26);
x_27 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_27, 0, x_7);
lean_ctor_set(x_27, 1, x_17);
lean_ctor_set(x_27, 2, x_18);
lean_ctor_set(x_27, 3, x_19);
lean_ctor_set(x_27, 4, x_20);
lean_ctor_set(x_27, 5, x_21);
lean_ctor_set(x_27, 6, x_22);
lean_ctor_set_uint8(x_27, sizeof(void*)*7, x_16);
lean_ctor_set_uint8(x_27, sizeof(void*)*7 + 1, x_23);
lean_ctor_set_uint8(x_27, sizeof(void*)*7 + 2, x_24);
x_28 = l_Lean_Meta_DiscrTree_mkPath(x_1, x_25, x_27, x_3, x_4, x_5);
return x_28;
}
}
else
{
lean_object* x_29; uint8_t x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; uint8_t x_37; uint8_t x_38; lean_object* x_39; uint8_t x_40; uint64_t x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; 
x_29 = lean_ctor_get(x_7, 0);
lean_inc(x_29);
lean_dec(x_7);
x_30 = lean_ctor_get_uint8(x_2, sizeof(void*)*7);
x_31 = lean_ctor_get(x_2, 1);
lean_inc(x_31);
x_32 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_32);
x_33 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_33);
x_34 = lean_ctor_get(x_2, 4);
lean_inc(x_34);
x_35 = lean_ctor_get(x_2, 5);
lean_inc(x_35);
x_36 = lean_ctor_get(x_2, 6);
lean_inc(x_36);
x_37 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 1);
x_38 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 2);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 lean_ctor_release(x_2, 3);
 lean_ctor_release(x_2, 4);
 lean_ctor_release(x_2, 5);
 lean_ctor_release(x_2, 6);
 x_39 = x_2;
} else {
 lean_dec_ref(x_2);
 x_39 = lean_box(0);
}
x_40 = 0;
x_41 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_29);
x_42 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_42, 0, x_29);
lean_ctor_set_uint64(x_42, sizeof(void*)*1, x_41);
if (lean_is_scalar(x_39)) {
 x_43 = lean_alloc_ctor(0, 7, 3);
} else {
 x_43 = x_39;
}
lean_ctor_set(x_43, 0, x_42);
lean_ctor_set(x_43, 1, x_31);
lean_ctor_set(x_43, 2, x_32);
lean_ctor_set(x_43, 3, x_33);
lean_ctor_set(x_43, 4, x_34);
lean_ctor_set(x_43, 5, x_35);
lean_ctor_set(x_43, 6, x_36);
lean_ctor_set_uint8(x_43, sizeof(void*)*7, x_30);
lean_ctor_set_uint8(x_43, sizeof(void*)*7 + 1, x_37);
lean_ctor_set_uint8(x_43, sizeof(void*)*7 + 2, x_38);
x_44 = l_Lean_Meta_DiscrTree_mkPath(x_1, x_40, x_43, x_3, x_4, x_5);
return x_44;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkDiscrTreePath___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Aesop_mkDiscrTreePath(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; uint8_t x_9; 
x_8 = lp_aesop_Aesop_indexConfig;
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
uint8_t x_10; 
x_10 = !lean_is_exclusive(x_3);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; uint64_t x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_8, 0);
x_12 = lean_ctor_get(x_3, 0);
lean_dec(x_12);
x_13 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_11);
lean_ctor_set_uint64(x_8, sizeof(void*)*1, x_13);
lean_ctor_set(x_3, 0, x_8);
x_14 = l_Lean_Meta_DiscrTree_getUnify___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_14;
}
else
{
lean_object* x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; uint8_t x_24; uint64_t x_25; lean_object* x_26; lean_object* x_27; 
x_15 = lean_ctor_get(x_8, 0);
x_16 = lean_ctor_get_uint8(x_3, sizeof(void*)*7);
x_17 = lean_ctor_get(x_3, 1);
x_18 = lean_ctor_get(x_3, 2);
x_19 = lean_ctor_get(x_3, 3);
x_20 = lean_ctor_get(x_3, 4);
x_21 = lean_ctor_get(x_3, 5);
x_22 = lean_ctor_get(x_3, 6);
x_23 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 1);
x_24 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 2);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_dec(x_3);
x_25 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_15);
lean_ctor_set_uint64(x_8, sizeof(void*)*1, x_25);
x_26 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_26, 0, x_8);
lean_ctor_set(x_26, 1, x_17);
lean_ctor_set(x_26, 2, x_18);
lean_ctor_set(x_26, 3, x_19);
lean_ctor_set(x_26, 4, x_20);
lean_ctor_set(x_26, 5, x_21);
lean_ctor_set(x_26, 6, x_22);
lean_ctor_set_uint8(x_26, sizeof(void*)*7, x_16);
lean_ctor_set_uint8(x_26, sizeof(void*)*7 + 1, x_23);
lean_ctor_set_uint8(x_26, sizeof(void*)*7 + 2, x_24);
x_27 = l_Lean_Meta_DiscrTree_getUnify___redArg(x_1, x_2, x_26, x_4, x_5, x_6);
return x_27;
}
}
else
{
lean_object* x_28; uint8_t x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; uint8_t x_36; uint8_t x_37; lean_object* x_38; uint64_t x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_28 = lean_ctor_get(x_8, 0);
lean_inc(x_28);
lean_dec(x_8);
x_29 = lean_ctor_get_uint8(x_3, sizeof(void*)*7);
x_30 = lean_ctor_get(x_3, 1);
lean_inc(x_30);
x_31 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_31);
x_32 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_32);
x_33 = lean_ctor_get(x_3, 4);
lean_inc(x_33);
x_34 = lean_ctor_get(x_3, 5);
lean_inc(x_34);
x_35 = lean_ctor_get(x_3, 6);
lean_inc(x_35);
x_36 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 1);
x_37 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 2);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 lean_ctor_release(x_3, 2);
 lean_ctor_release(x_3, 3);
 lean_ctor_release(x_3, 4);
 lean_ctor_release(x_3, 5);
 lean_ctor_release(x_3, 6);
 x_38 = x_3;
} else {
 lean_dec_ref(x_3);
 x_38 = lean_box(0);
}
x_39 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_28);
x_40 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_40, 0, x_28);
lean_ctor_set_uint64(x_40, sizeof(void*)*1, x_39);
if (lean_is_scalar(x_38)) {
 x_41 = lean_alloc_ctor(0, 7, 3);
} else {
 x_41 = x_38;
}
lean_ctor_set(x_41, 0, x_40);
lean_ctor_set(x_41, 1, x_30);
lean_ctor_set(x_41, 2, x_31);
lean_ctor_set(x_41, 3, x_32);
lean_ctor_set(x_41, 4, x_33);
lean_ctor_set(x_41, 5, x_34);
lean_ctor_set(x_41, 6, x_35);
lean_ctor_set_uint8(x_41, sizeof(void*)*7, x_29);
lean_ctor_set_uint8(x_41, sizeof(void*)*7 + 1, x_36);
lean_ctor_set_uint8(x_41, sizeof(void*)*7 + 2, x_37);
x_42 = l_Lean_Meta_DiscrTree_getUnify___redArg(x_1, x_2, x_41, x_4, x_5, x_6);
return x_42;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_getUnify___redArg(x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_getUnify(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Aesop_getUnify___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; uint8_t x_9; 
x_8 = lp_aesop_Aesop_indexConfig;
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
uint8_t x_10; 
x_10 = !lean_is_exclusive(x_3);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; uint64_t x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_8, 0);
x_12 = lean_ctor_get(x_3, 0);
lean_dec(x_12);
x_13 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_11);
lean_ctor_set_uint64(x_8, sizeof(void*)*1, x_13);
lean_ctor_set(x_3, 0, x_8);
x_14 = l_Lean_Meta_DiscrTree_getMatch___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_14;
}
else
{
lean_object* x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; uint8_t x_24; uint64_t x_25; lean_object* x_26; lean_object* x_27; 
x_15 = lean_ctor_get(x_8, 0);
x_16 = lean_ctor_get_uint8(x_3, sizeof(void*)*7);
x_17 = lean_ctor_get(x_3, 1);
x_18 = lean_ctor_get(x_3, 2);
x_19 = lean_ctor_get(x_3, 3);
x_20 = lean_ctor_get(x_3, 4);
x_21 = lean_ctor_get(x_3, 5);
x_22 = lean_ctor_get(x_3, 6);
x_23 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 1);
x_24 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 2);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_dec(x_3);
x_25 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_15);
lean_ctor_set_uint64(x_8, sizeof(void*)*1, x_25);
x_26 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_26, 0, x_8);
lean_ctor_set(x_26, 1, x_17);
lean_ctor_set(x_26, 2, x_18);
lean_ctor_set(x_26, 3, x_19);
lean_ctor_set(x_26, 4, x_20);
lean_ctor_set(x_26, 5, x_21);
lean_ctor_set(x_26, 6, x_22);
lean_ctor_set_uint8(x_26, sizeof(void*)*7, x_16);
lean_ctor_set_uint8(x_26, sizeof(void*)*7 + 1, x_23);
lean_ctor_set_uint8(x_26, sizeof(void*)*7 + 2, x_24);
x_27 = l_Lean_Meta_DiscrTree_getMatch___redArg(x_1, x_2, x_26, x_4, x_5, x_6);
return x_27;
}
}
else
{
lean_object* x_28; uint8_t x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; uint8_t x_36; uint8_t x_37; lean_object* x_38; uint64_t x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_28 = lean_ctor_get(x_8, 0);
lean_inc(x_28);
lean_dec(x_8);
x_29 = lean_ctor_get_uint8(x_3, sizeof(void*)*7);
x_30 = lean_ctor_get(x_3, 1);
lean_inc(x_30);
x_31 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_31);
x_32 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_32);
x_33 = lean_ctor_get(x_3, 4);
lean_inc(x_33);
x_34 = lean_ctor_get(x_3, 5);
lean_inc(x_34);
x_35 = lean_ctor_get(x_3, 6);
lean_inc(x_35);
x_36 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 1);
x_37 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 2);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 lean_ctor_release(x_3, 2);
 lean_ctor_release(x_3, 3);
 lean_ctor_release(x_3, 4);
 lean_ctor_release(x_3, 5);
 lean_ctor_release(x_3, 6);
 x_38 = x_3;
} else {
 lean_dec_ref(x_3);
 x_38 = lean_box(0);
}
x_39 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_28);
x_40 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_40, 0, x_28);
lean_ctor_set_uint64(x_40, sizeof(void*)*1, x_39);
if (lean_is_scalar(x_38)) {
 x_41 = lean_alloc_ctor(0, 7, 3);
} else {
 x_41 = x_38;
}
lean_ctor_set(x_41, 0, x_40);
lean_ctor_set(x_41, 1, x_30);
lean_ctor_set(x_41, 2, x_31);
lean_ctor_set(x_41, 3, x_32);
lean_ctor_set(x_41, 4, x_33);
lean_ctor_set(x_41, 5, x_34);
lean_ctor_set(x_41, 6, x_35);
lean_ctor_set_uint8(x_41, sizeof(void*)*7, x_29);
lean_ctor_set_uint8(x_41, sizeof(void*)*7 + 1, x_36);
lean_ctor_set_uint8(x_41, sizeof(void*)*7 + 2, x_37);
x_42 = l_Lean_Meta_DiscrTree_getMatch___redArg(x_1, x_2, x_41, x_4, x_5, x_6);
return x_42;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_getMatch___redArg(x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_getMatch(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Aesop_getMatch___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Index_DiscrTreeConfig(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_indexConfig___closed__0 = _init_lp_aesop_Aesop_indexConfig___closed__0();
lean_mark_persistent(lp_aesop_Aesop_indexConfig___closed__0);
lp_aesop_Aesop_indexConfig___closed__1 = _init_lp_aesop_Aesop_indexConfig___closed__1();
lean_mark_persistent(lp_aesop_Aesop_indexConfig___closed__1);
lp_aesop_Aesop_indexConfig = _init_lp_aesop_Aesop_indexConfig();
lean_mark_persistent(lp_aesop_Aesop_indexConfig);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
