// Lean compiler output
// Module: Mathlib.Tactic.Linter.TacticDocumentation
// Imports: public import Init public meta import Batteries.Tactic.Lint.Basic public meta import Lean.Elab.Tactic.Doc public meta import Lean.Parser.Tactic.Doc import Mathlib.Tactic.Linter.Header
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
static lean_object* lp_mathlib_tacticDocs___closed__5;
static lean_object* lp_mathlib_tacticDocs___closed__1;
static lean_object* lp_mathlib_tacticDocs___lam__0___closed__0;
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib_tacticDocs___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
static lean_object* lp_mathlib_tacticDocs___lam__0___closed__3;
lean_object* lean_string_utf8_byte_size(lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_Parser_Tactic_Doc_alternativeOfTactic(lean_object*, lean_object*);
uint8_t l_Lean_Parser_Tactic_Doc_isTactic(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_tacticDocs;
static lean_object* lp_mathlib_tacticDocs___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1(lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Elab_Tactic_Doc_allTacticDocs(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_tacticDocs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0(lean_object*, size_t, size_t);
static lean_object* lp_mathlib_tacticDocs___closed__4;
static lean_object* lp_mathlib_tacticDocs___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_quickCmp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0___boxed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
static lean_object* lp_mathlib_tacticDocs___closed__0;
static lean_object* lp_mathlib_tacticDocs___lam__0___closed__2;
static lean_object* lp_mathlib_tacticDocs___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0(lean_object* x_1, size_t x_2, size_t x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_usize_dec_eq(x_2, x_3);
if (x_4 == 0)
{
uint8_t x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_5 = 1;
x_6 = lean_array_uget(x_1, x_2);
x_7 = lean_string_utf8_byte_size(x_6);
lean_dec(x_6);
x_8 = lean_unsigned_to_nat(0u);
x_9 = lean_nat_dec_eq(x_7, x_8);
if (x_9 == 0)
{
return x_5;
}
else
{
if (x_4 == 0)
{
size_t x_10; size_t x_11; 
x_10 = 1;
x_11 = lean_usize_add(x_2, x_10);
x_2 = x_11;
goto _start;
}
else
{
return x_5;
}
}
}
else
{
uint8_t x_13; 
x_13 = 0;
return x_13;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 3);
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_3 = lean_ctor_get(x_1, 4);
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_array_get_size(x_3);
x_6 = lean_nat_dec_lt(x_4, x_5);
if (x_6 == 0)
{
return x_6;
}
else
{
if (x_6 == 0)
{
return x_6;
}
else
{
size_t x_7; size_t x_8; uint8_t x_9; 
x_7 = 0;
x_8 = lean_usize_of_nat(x_5);
x_9 = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0(x_3, x_7, x_8);
return x_9;
}
}
}
else
{
uint8_t x_10; 
x_10 = 1;
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; size_t x_5; uint8_t x_6; lean_object* x_7; 
x_4 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_5 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_6 = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0(x_1, x_4, x_5);
lean_dec_ref(x_1);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc(x_1);
lean_dec_ref(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_1, 2);
x_5 = lean_ctor_get(x_1, 3);
x_6 = lean_ctor_get(x_1, 4);
x_7 = l_Lean_Name_quickCmp(x_2, x_3);
switch (x_7) {
case 0:
{
x_1 = x_5;
goto _start;
}
case 1:
{
lean_object* x_9; 
lean_inc(x_4);
x_9 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_9, 0, x_4);
return x_9;
}
default: 
{
x_1 = x_6;
goto _start;
}
}
}
else
{
lean_object* x_11; 
x_11 = lean_box(0);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg(x_2, x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("No tactics are missing documentation.", 37, 37);
return x_1;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_tacticDocs___closed__0;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_tacticDocs___closed__1;
x_2 = l_Lean_MessageData_ofFormat(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("TACTICS ARE MISSING DOCUMENTATION STRINGS:", 42, 42);
return x_1;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_tacticDocs___closed__3;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_tacticDocs___closed__4;
x_2 = l_Lean_MessageData_ofFormat(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tactic `", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_tacticDocs___lam__0___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___lam__0___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("` missing documentation string", 30, 30);
return x_1;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___lam__0___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_tacticDocs___lam__0___closed__2;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1(lean_object* x_1, size_t x_2, size_t x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = lean_usize_dec_eq(x_2, x_3);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; size_t x_9; size_t x_10; 
x_6 = lean_array_uget(x_1, x_2);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(x_7, x_6, x_4);
x_9 = 1;
x_10 = lean_usize_add(x_2, x_9);
x_2 = x_10;
x_4 = x_8;
goto _start;
}
else
{
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_tacticDocs___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_11; uint8_t x_12; 
x_7 = lean_st_ref_get(x_5);
x_11 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_11);
lean_dec(x_7);
lean_inc_ref(x_11);
x_12 = l_Lean_Parser_Tactic_Doc_isTactic(x_11, x_1);
if (x_12 == 0)
{
lean_dec_ref(x_11);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
goto block_10;
}
else
{
lean_object* x_13; 
lean_inc(x_1);
x_13 = l_Lean_Parser_Tactic_Doc_alternativeOfTactic(x_11, x_1);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; 
x_14 = l_Lean_Elab_Tactic_Doc_allTacticDocs(x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_26; lean_object* x_27; lean_object* x_37; lean_object* x_43; lean_object* x_44; lean_object* x_45; uint8_t x_46; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
if (lean_is_exclusive(x_14)) {
 lean_ctor_release(x_14, 0);
 x_16 = x_14;
} else {
 lean_dec_ref(x_14);
 x_16 = lean_box(0);
}
x_43 = lean_box(1);
x_44 = lean_unsigned_to_nat(0u);
x_45 = lean_array_get_size(x_15);
x_46 = lean_nat_dec_lt(x_44, x_45);
if (x_46 == 0)
{
lean_dec(x_15);
x_37 = x_43;
goto block_42;
}
else
{
uint8_t x_47; 
x_47 = lean_nat_dec_le(x_45, x_45);
if (x_47 == 0)
{
lean_dec(x_15);
x_37 = x_43;
goto block_42;
}
else
{
size_t x_48; size_t x_49; lean_object* x_50; 
x_48 = 0;
x_49 = lean_usize_of_nat(x_45);
x_50 = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1(x_15, x_48, x_49, x_43);
lean_dec(x_15);
x_37 = x_50;
goto block_42;
}
}
block_25:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_18 = lp_mathlib_tacticDocs___lam__0___closed__1;
x_19 = l_Lean_stringToMessageData(x_17);
x_20 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_20, 0, x_18);
lean_ctor_set(x_20, 1, x_19);
x_21 = lp_mathlib_tacticDocs___lam__0___closed__3;
x_22 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_22, 0, x_20);
lean_ctor_set(x_22, 1, x_21);
x_23 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_23, 0, x_22);
if (lean_is_scalar(x_16)) {
 x_24 = lean_alloc_ctor(0, 1, 0);
} else {
 x_24 = x_16;
}
lean_ctor_set(x_24, 0, x_23);
return x_24;
}
block_36:
{
if (lean_obj_tag(x_26) == 1)
{
uint8_t x_28; 
x_28 = !lean_is_exclusive(x_26);
if (x_28 == 0)
{
lean_object* x_29; uint8_t x_30; 
x_29 = lean_ctor_get(x_26, 0);
x_30 = lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc(x_29);
lean_dec(x_29);
if (x_30 == 0)
{
lean_free_object(x_26);
x_17 = x_27;
goto block_25;
}
else
{
lean_object* x_31; 
lean_dec_ref(x_27);
lean_dec(x_16);
x_31 = lean_box(0);
lean_ctor_set_tag(x_26, 0);
lean_ctor_set(x_26, 0, x_31);
return x_26;
}
}
else
{
lean_object* x_32; uint8_t x_33; 
x_32 = lean_ctor_get(x_26, 0);
lean_inc(x_32);
lean_dec(x_26);
x_33 = lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc(x_32);
lean_dec(x_32);
if (x_33 == 0)
{
x_17 = x_27;
goto block_25;
}
else
{
lean_object* x_34; lean_object* x_35; 
lean_dec_ref(x_27);
lean_dec(x_16);
x_34 = lean_box(0);
x_35 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_35, 0, x_34);
return x_35;
}
}
}
else
{
lean_dec(x_26);
x_17 = x_27;
goto block_25;
}
}
block_42:
{
lean_object* x_38; 
x_38 = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg(x_37, x_1);
lean_dec(x_37);
if (lean_obj_tag(x_38) == 0)
{
lean_object* x_39; 
x_39 = l_Lean_Name_toString(x_1, x_12);
x_26 = x_38;
x_27 = x_39;
goto block_36;
}
else
{
lean_object* x_40; lean_object* x_41; 
lean_dec(x_1);
x_40 = lean_ctor_get(x_38, 0);
lean_inc(x_40);
x_41 = lean_ctor_get(x_40, 1);
lean_inc_ref(x_41);
lean_dec(x_40);
x_26 = x_38;
x_27 = x_41;
goto block_36;
}
}
}
else
{
uint8_t x_51; 
lean_dec(x_1);
x_51 = !lean_is_exclusive(x_14);
if (x_51 == 0)
{
return x_14;
}
else
{
lean_object* x_52; lean_object* x_53; 
x_52 = lean_ctor_get(x_14, 0);
lean_inc(x_52);
lean_dec(x_14);
x_53 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_53, 0, x_52);
return x_53;
}
}
}
else
{
lean_dec_ref(x_13);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
goto block_10;
}
}
block_10:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_box(0);
x_9 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_tacticDocs___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_tacticDocs___lam__0(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
static lean_object* _init_lp_mathlib_tacticDocs() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; uint8_t x_4; lean_object* x_5; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_tacticDocs___lam__0___boxed), 6, 0);
x_2 = lp_mathlib_tacticDocs___closed__2;
x_3 = lp_mathlib_tacticDocs___closed__5;
x_4 = 1;
x_5 = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_2);
lean_ctor_set(x_5, 2, x_3);
lean_ctor_set_uint8(x_5, sizeof(void*)*3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
size_t x_5; size_t x_6; lean_object* x_7; 
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1(x_1, x_5, x_6, x_4);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Doc(uint8_t builtin);
lean_object* initialize_Lean_Parser_Tactic_Doc(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_TacticDocumentation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Doc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Parser_Tactic_Doc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_tacticDocs___closed__0 = _init_lp_mathlib_tacticDocs___closed__0();
lean_mark_persistent(lp_mathlib_tacticDocs___closed__0);
lp_mathlib_tacticDocs___closed__1 = _init_lp_mathlib_tacticDocs___closed__1();
lean_mark_persistent(lp_mathlib_tacticDocs___closed__1);
lp_mathlib_tacticDocs___closed__2 = _init_lp_mathlib_tacticDocs___closed__2();
lean_mark_persistent(lp_mathlib_tacticDocs___closed__2);
lp_mathlib_tacticDocs___closed__3 = _init_lp_mathlib_tacticDocs___closed__3();
lean_mark_persistent(lp_mathlib_tacticDocs___closed__3);
lp_mathlib_tacticDocs___closed__4 = _init_lp_mathlib_tacticDocs___closed__4();
lean_mark_persistent(lp_mathlib_tacticDocs___closed__4);
lp_mathlib_tacticDocs___closed__5 = _init_lp_mathlib_tacticDocs___closed__5();
lean_mark_persistent(lp_mathlib_tacticDocs___closed__5);
lp_mathlib_tacticDocs___lam__0___closed__0 = _init_lp_mathlib_tacticDocs___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_tacticDocs___lam__0___closed__0);
lp_mathlib_tacticDocs___lam__0___closed__1 = _init_lp_mathlib_tacticDocs___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_tacticDocs___lam__0___closed__1);
lp_mathlib_tacticDocs___lam__0___closed__2 = _init_lp_mathlib_tacticDocs___lam__0___closed__2();
lean_mark_persistent(lp_mathlib_tacticDocs___lam__0___closed__2);
lp_mathlib_tacticDocs___lam__0___closed__3 = _init_lp_mathlib_tacticDocs___lam__0___closed__3();
lean_mark_persistent(lp_mathlib_tacticDocs___lam__0___closed__3);
lp_mathlib_tacticDocs = _init_lp_mathlib_tacticDocs();
lean_mark_persistent(lp_mathlib_tacticDocs);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
