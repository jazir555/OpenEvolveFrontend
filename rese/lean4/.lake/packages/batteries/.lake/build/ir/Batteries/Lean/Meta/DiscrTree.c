// Lean compiler output
// Module: Batteries.Lean.Meta.DiscrTree
// Imports: public import Init public import Lean.Meta.DiscrTree public import Batteries.Data.Array.Merge public import Batteries.Lean.Meta.Expr public import Batteries.Lean.PersistentHashMap
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
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries;
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_find_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_Key_hash___boxed(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t lean_string_dec_lt(lean_object*, lean_object*);
static lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__1;
LEAN_EXPORT uint8_t lp_batteries_Lean_Meta_DiscrTree_Key_cmp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Key_cmp___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_instBEqKey_beq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren(lean_object*, lean_object*, lean_object*);
lean_object* l_Subarray_toArray___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_Key_ctorIdx(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries___closed__0;
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Name_quickCmp(lean_object*, lean_object*);
static lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__0;
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_foldl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Meta_DiscrTree_Key_cmp(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
switch (lean_obj_tag(x_1)) {
case 2:
{
if (lean_obj_tag(x_2) == 2)
{
lean_object* x_13; 
x_13 = lean_ctor_get(x_1, 0);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; 
x_14 = lean_ctor_get(x_2, 0);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_15 = lean_ctor_get(x_13, 0);
x_16 = lean_ctor_get(x_14, 0);
x_17 = lean_nat_dec_lt(x_15, x_16);
if (x_17 == 0)
{
uint8_t x_18; 
x_18 = lean_nat_dec_eq(x_15, x_16);
if (x_18 == 0)
{
uint8_t x_19; 
x_19 = 2;
return x_19;
}
else
{
uint8_t x_20; 
x_20 = 1;
return x_20;
}
}
else
{
uint8_t x_21; 
x_21 = 0;
return x_21;
}
}
else
{
uint8_t x_22; 
x_22 = 0;
return x_22;
}
}
else
{
lean_object* x_23; 
x_23 = lean_ctor_get(x_2, 0);
if (lean_obj_tag(x_23) == 0)
{
uint8_t x_24; 
x_24 = 2;
return x_24;
}
else
{
lean_object* x_25; lean_object* x_26; uint8_t x_27; 
x_25 = lean_ctor_get(x_13, 0);
x_26 = lean_ctor_get(x_23, 0);
x_27 = lean_string_dec_lt(x_25, x_26);
if (x_27 == 0)
{
uint8_t x_28; 
x_28 = lean_string_dec_eq(x_25, x_26);
if (x_28 == 0)
{
uint8_t x_29; 
x_29 = 2;
return x_29;
}
else
{
uint8_t x_30; 
x_30 = 1;
return x_30;
}
}
else
{
uint8_t x_31; 
x_31 = 0;
return x_31;
}
}
}
}
else
{
x_3 = x_1;
x_4 = x_2;
goto block_12;
}
}
case 3:
{
if (lean_obj_tag(x_2) == 3)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; uint8_t x_36; 
x_32 = lean_ctor_get(x_1, 0);
x_33 = lean_ctor_get(x_1, 1);
x_34 = lean_ctor_get(x_2, 0);
x_35 = lean_ctor_get(x_2, 1);
x_36 = l_Lean_Name_quickCmp(x_32, x_34);
if (x_36 == 1)
{
uint8_t x_37; 
x_37 = lean_nat_dec_lt(x_33, x_35);
if (x_37 == 0)
{
uint8_t x_38; 
x_38 = lean_nat_dec_eq(x_33, x_35);
if (x_38 == 0)
{
uint8_t x_39; 
x_39 = 2;
return x_39;
}
else
{
return x_36;
}
}
else
{
uint8_t x_40; 
x_40 = 0;
return x_40;
}
}
else
{
return x_36;
}
}
else
{
x_3 = x_1;
x_4 = x_2;
goto block_12;
}
}
case 4:
{
if (lean_obj_tag(x_2) == 4)
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; uint8_t x_45; 
x_41 = lean_ctor_get(x_1, 0);
x_42 = lean_ctor_get(x_1, 1);
x_43 = lean_ctor_get(x_2, 0);
x_44 = lean_ctor_get(x_2, 1);
x_45 = l_Lean_Name_quickCmp(x_41, x_43);
if (x_45 == 1)
{
uint8_t x_46; 
x_46 = lean_nat_dec_lt(x_42, x_44);
if (x_46 == 0)
{
uint8_t x_47; 
x_47 = lean_nat_dec_eq(x_42, x_44);
if (x_47 == 0)
{
uint8_t x_48; 
x_48 = 2;
return x_48;
}
else
{
return x_45;
}
}
else
{
uint8_t x_49; 
x_49 = 0;
return x_49;
}
}
else
{
return x_45;
}
}
else
{
x_3 = x_1;
x_4 = x_2;
goto block_12;
}
}
case 6:
{
if (lean_obj_tag(x_2) == 6)
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; uint8_t x_56; 
x_50 = lean_ctor_get(x_1, 0);
x_51 = lean_ctor_get(x_1, 1);
x_52 = lean_ctor_get(x_1, 2);
x_53 = lean_ctor_get(x_2, 0);
x_54 = lean_ctor_get(x_2, 1);
x_55 = lean_ctor_get(x_2, 2);
x_56 = l_Lean_Name_quickCmp(x_50, x_53);
if (x_56 == 1)
{
uint8_t x_57; 
x_57 = lean_nat_dec_lt(x_51, x_54);
if (x_57 == 0)
{
uint8_t x_58; 
x_58 = lean_nat_dec_eq(x_51, x_54);
if (x_58 == 0)
{
uint8_t x_59; 
x_59 = 2;
return x_59;
}
else
{
uint8_t x_60; 
x_60 = lean_nat_dec_lt(x_52, x_55);
if (x_60 == 0)
{
uint8_t x_61; 
x_61 = lean_nat_dec_eq(x_52, x_55);
if (x_61 == 0)
{
uint8_t x_62; 
x_62 = 2;
return x_62;
}
else
{
return x_56;
}
}
else
{
uint8_t x_63; 
x_63 = 0;
return x_63;
}
}
}
else
{
uint8_t x_64; 
x_64 = 0;
return x_64;
}
}
else
{
return x_56;
}
}
else
{
x_3 = x_1;
x_4 = x_2;
goto block_12;
}
}
default: 
{
x_3 = x_1;
x_4 = x_2;
goto block_12;
}
}
block_12:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = l_Lean_Meta_DiscrTree_Key_ctorIdx(x_3);
x_6 = l_Lean_Meta_DiscrTree_Key_ctorIdx(x_4);
x_7 = lean_nat_dec_lt(x_5, x_6);
if (x_7 == 0)
{
uint8_t x_8; 
x_8 = lean_nat_dec_eq(x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
if (x_8 == 0)
{
uint8_t x_9; 
x_9 = 2;
return x_9;
}
else
{
uint8_t x_10; 
x_10 = 1;
return x_10;
}
}
else
{
uint8_t x_11; 
lean_dec(x_6);
lean_dec(x_5);
x_11 = 0;
return x_11;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Key_cmp___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_batteries_Lean_Meta_DiscrTree_Key_cmp(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_DiscrTree_Key_cmp___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries() {
_start:
{
lean_object* x_1; 
x_1 = lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; uint8_t x_8; 
x_7 = lean_array_get_size(x_1);
x_8 = lean_nat_dec_le(x_7, x_5);
if (x_8 == 0)
{
lean_object* x_9; uint8_t x_10; 
x_9 = lean_array_get_size(x_2);
x_10 = lean_nat_dec_le(x_9, x_6);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_11 = lean_array_fget_borrowed(x_1, x_5);
x_12 = lean_ctor_get(x_11, 0);
x_13 = lean_array_fget_borrowed(x_2, x_6);
x_14 = lean_ctor_get(x_13, 0);
x_15 = lp_batteries_Lean_Meta_DiscrTree_Key_cmp(x_12, x_14);
switch (x_15) {
case 0:
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_inc(x_11);
x_16 = lean_array_push(x_4, x_11);
x_17 = lean_unsigned_to_nat(1u);
x_18 = lean_nat_add(x_5, x_17);
lean_dec(x_5);
x_4 = x_16;
x_5 = x_18;
goto _start;
}
case 1:
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
lean_inc_ref(x_3);
lean_inc(x_13);
lean_inc(x_11);
x_20 = lean_apply_2(x_3, x_11, x_13);
x_21 = lean_array_push(x_4, x_20);
x_22 = lean_unsigned_to_nat(1u);
x_23 = lean_nat_add(x_5, x_22);
lean_dec(x_5);
x_24 = lean_nat_add(x_6, x_22);
lean_dec(x_6);
x_4 = x_21;
x_5 = x_23;
x_6 = x_24;
goto _start;
}
default: 
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; 
lean_inc(x_13);
x_26 = lean_array_push(x_4, x_13);
x_27 = lean_unsigned_to_nat(1u);
x_28 = lean_nat_add(x_6, x_27);
lean_dec(x_6);
x_4 = x_26;
x_6 = x_28;
goto _start;
}
}
}
else
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; 
lean_dec(x_6);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_30 = l_Array_toSubarray___redArg(x_1, x_5, x_7);
x_31 = l_Subarray_toArray___redArg(x_30);
x_32 = l_Array_append___redArg(x_4, x_31);
lean_dec_ref(x_31);
return x_32;
}
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_33 = lean_array_get_size(x_2);
x_34 = l_Array_toSubarray___redArg(x_2, x_6, x_33);
x_35 = l_Subarray_toArray___redArg(x_34);
x_36 = l_Array_append___redArg(x_4, x_35);
lean_dec_ref(x_35);
return x_36;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_array_get_size(x_1);
x_5 = lean_array_get_size(x_2);
x_6 = lean_nat_add(x_4, x_5);
x_7 = lean_mk_empty_array_with_capacity(x_6);
lean_dec(x_6);
x_8 = lean_unsigned_to_nat(0u);
x_9 = lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__0___redArg(x_1, x_2, x_3, x_7, x_8, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__0___redArg(x_2, x_3, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_2, 1);
x_7 = lean_ctor_get(x_2, 0);
lean_dec(x_7);
x_8 = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(x_4, x_6);
lean_ctor_set(x_2, 1, x_8);
lean_ctor_set(x_2, 0, x_3);
return x_2;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_2, 1);
lean_inc(x_9);
lean_dec(x_2);
x_10 = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(x_4, x_9);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_3);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg___lam__0), 2, 0);
x_4 = lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = l_Array_append___redArg(x_3, x_6);
lean_dec_ref(x_6);
x_9 = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg(x_4, x_7);
lean_ctor_set(x_2, 1, x_9);
lean_ctor_set(x_2, 0, x_8);
return x_2;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_2);
x_12 = l_Array_append___redArg(x_3, x_10);
lean_dec_ref(x_10);
x_13 = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg(x_4, x_11);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg(x_2, x_3);
return x_4;
}
}
static lean_object* _init_lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_DiscrTree_instBEqKey_beq___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_DiscrTree_Key_hash___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_6 = l_Lean_PersistentHashMap_find_x3f___redArg(x_1, x_2, x_3, x_4);
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_7; 
x_7 = l_Lean_PersistentHashMap_insert___redArg(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(x_8, x_5);
x_10 = l_Lean_PersistentHashMap_insert___redArg(x_1, x_2, x_3, x_4, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__0;
x_5 = lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__1;
x_6 = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___lam__0), 5, 2);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_5);
x_7 = l_Lean_PersistentHashMap_foldl___redArg(x_3, x_6, x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__0;
x_4 = lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__1;
x_5 = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___lam__0), 5, 2);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
x_6 = l_Lean_PersistentHashMap_foldl___redArg(x_2, x_5, x_1);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_DiscrTree(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Array_Merge(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_Expr(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_PersistentHashMap(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Meta_DiscrTree(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Array_Merge(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_PersistentHashMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries___closed__0 = _init_lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries___closed__0();
lean_mark_persistent(lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries___closed__0);
lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries = _init_lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries();
lean_mark_persistent(lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries);
lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__0 = _init_lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__0();
lean_mark_persistent(lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__0);
lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__1 = _init_lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__1();
lean_mark_persistent(lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
