// Lean compiler output
// Module: Mathlib.Tactic.Zify
// Imports: public import Init public meta import Mathlib.Tactic.Basic public meta import Mathlib.Tactic.Attr.Register public meta import Mathlib.Data.Int.Cast.Basic public meta import Mathlib.Order.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6;
lean_object* l_Lean_Meta_getSimpTheorems___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0(lean_object*, size_t, size_t, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___closed__0;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__26;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__36;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14;
extern lean_object* l_Lean_Parser_Tactic_location;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1;
lean_object* l_Lean_Syntax_getArgs(lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__6;
lean_object* l_Lean_Meta_mkExpectedTypeHint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__16;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7;
lean_object* l_Array_mkArray0(lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15;
lean_object* l_Lean_Meta_mkEqMP(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13;
lean_object* l_Array_mkArray1___redArg(lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11;
uint8_t lean_expr_eqv(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33;
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__18;
lean_object* l_Array_empty(lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__2;
lean_object* l_Lean_Meta_simp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2;
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__7;
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__0;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__8;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__4;
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__16;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__39;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__8;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__5;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__21;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__0;
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0___redArg(size_t, size_t, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__10;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_mkSimpContext(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29;
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__41;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__20;
uint8_t l_Lean_Syntax_isNone(lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__8;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19;
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__15;
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30;
extern lean_object* l_Lean_Parser_Tactic_simpArgs;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23;
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7;
size_t lean_array_size(lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40;
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3;
uint8_t lean_usize_dec_lt(size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42;
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__38;
lean_object* l_Array_mkArray4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Mathlib", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Zify", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("zify", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__2;
x_3 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1;
x_4 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("andthen", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__5;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__7() {
_start:
{
uint8_t x_1; lean_object* x_2; lean_object* x_3; 
x_1 = 0;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3;
x_3 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set_uint8(x_3, sizeof(void*)*1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("optional", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__8;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_Parser_Tactic_simpArgs;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__7;
x_3 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_Parser_Tactic_location;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12;
x_3 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__15;
x_2 = lean_unsigned_to_nat(1022u);
x_3 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4;
x_4 = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__16;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lean", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Parser", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("simp", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1;
x_3 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1;
x_4 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("optConfig", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__4;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1;
x_3 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1;
x_4 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("null", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__6;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("configItem", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__8;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1;
x_3 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1;
x_4 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("negConfigItem", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__10;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1;
x_3 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1;
x_4 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("-", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("decide", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__16() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Decidable", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13;
x_2 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__16;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1;
x_3 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1;
x_4 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__21() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__20;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__21;
x_2 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__18;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_mkArray0(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("only", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("[", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__26() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("simpLemma", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__26;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1;
x_3 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1;
x_4 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("zify_simps", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(",", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("push_cast", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("]", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__36() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("location", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__36;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1;
x_3 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1;
x_4 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__38() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("at", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__39() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__41() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("simpArgs", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__41;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1;
x_3 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1;
x_4 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_83; uint8_t x_84; 
x_83 = lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4;
lean_inc(x_1);
x_84 = l_Lean_Syntax_isOfKind(x_1, x_83);
if (x_84 == 0)
{
lean_object* x_85; lean_object* x_86; 
lean_dec_ref(x_2);
lean_dec(x_1);
x_85 = lean_box(1);
x_86 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_86, 0, x_85);
lean_ctor_set(x_86, 1, x_3);
return x_86;
}
else
{
lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_115; uint8_t x_116; 
x_87 = lean_unsigned_to_nat(0u);
x_96 = lean_unsigned_to_nat(1u);
x_115 = l_Lean_Syntax_getArg(x_1, x_96);
x_116 = l_Lean_Syntax_isNone(x_115);
if (x_116 == 0)
{
uint8_t x_117; 
lean_inc(x_115);
x_117 = l_Lean_Syntax_matchesNull(x_115, x_96);
if (x_117 == 0)
{
lean_object* x_118; lean_object* x_119; 
lean_dec(x_115);
lean_dec_ref(x_2);
lean_dec(x_1);
x_118 = lean_box(1);
x_119 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_119, 0, x_118);
lean_ctor_set(x_119, 1, x_3);
return x_119;
}
else
{
lean_object* x_120; lean_object* x_121; uint8_t x_122; 
x_120 = l_Lean_Syntax_getArg(x_115, x_87);
lean_dec(x_115);
x_121 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42;
lean_inc(x_120);
x_122 = l_Lean_Syntax_isOfKind(x_120, x_121);
if (x_122 == 0)
{
lean_object* x_123; lean_object* x_124; 
lean_dec(x_120);
lean_dec_ref(x_2);
lean_dec(x_1);
x_123 = lean_box(1);
x_124 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_124, 0, x_123);
lean_ctor_set(x_124, 1, x_3);
return x_124;
}
else
{
lean_object* x_125; lean_object* x_126; lean_object* x_127; 
x_125 = l_Lean_Syntax_getArg(x_120, x_96);
lean_dec(x_120);
x_126 = l_Lean_Syntax_getArgs(x_125);
lean_dec(x_125);
x_127 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_127, 0, x_126);
x_97 = x_127;
x_98 = x_2;
x_99 = x_3;
goto block_114;
}
}
}
else
{
lean_object* x_128; 
lean_dec(x_115);
x_128 = lean_box(0);
x_97 = x_128;
x_98 = x_2;
x_99 = x_3;
goto block_114;
}
block_95:
{
if (lean_obj_tag(x_88) == 0)
{
lean_object* x_92; 
x_92 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40;
x_20 = x_91;
x_21 = x_89;
x_22 = x_90;
x_23 = x_92;
goto block_82;
}
else
{
lean_object* x_93; lean_object* x_94; 
x_93 = lean_ctor_get(x_88, 0);
lean_inc(x_93);
lean_dec_ref(x_88);
x_94 = l_Lean_Syntax_TSepArray_getElems___redArg(x_93);
lean_dec(x_93);
x_20 = x_91;
x_21 = x_89;
x_22 = x_90;
x_23 = x_94;
goto block_82;
}
}
block_114:
{
lean_object* x_100; lean_object* x_101; uint8_t x_102; 
x_100 = lean_unsigned_to_nat(2u);
x_101 = l_Lean_Syntax_getArg(x_1, x_100);
lean_dec(x_1);
x_102 = l_Lean_Syntax_isNone(x_101);
if (x_102 == 0)
{
uint8_t x_103; 
lean_inc(x_101);
x_103 = l_Lean_Syntax_matchesNull(x_101, x_96);
if (x_103 == 0)
{
lean_object* x_104; lean_object* x_105; 
lean_dec(x_101);
lean_dec_ref(x_98);
lean_dec(x_97);
x_104 = lean_box(1);
x_105 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_105, 0, x_104);
lean_ctor_set(x_105, 1, x_99);
return x_105;
}
else
{
lean_object* x_106; lean_object* x_107; uint8_t x_108; 
x_106 = l_Lean_Syntax_getArg(x_101, x_87);
lean_dec(x_101);
x_107 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37;
lean_inc(x_106);
x_108 = l_Lean_Syntax_isOfKind(x_106, x_107);
if (x_108 == 0)
{
lean_object* x_109; lean_object* x_110; 
lean_dec(x_106);
lean_dec_ref(x_98);
lean_dec(x_97);
x_109 = lean_box(1);
x_110 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_110, 0, x_109);
lean_ctor_set(x_110, 1, x_99);
return x_110;
}
else
{
lean_object* x_111; lean_object* x_112; 
x_111 = l_Lean_Syntax_getArg(x_106, x_96);
lean_dec(x_106);
x_112 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_112, 0, x_111);
x_88 = x_97;
x_89 = x_112;
x_90 = x_98;
x_91 = x_99;
goto block_95;
}
}
}
else
{
lean_object* x_113; 
lean_dec(x_101);
x_113 = lean_box(0);
x_88 = x_97;
x_89 = x_113;
x_90 = x_98;
x_91 = x_99;
goto block_95;
}
}
}
block_19:
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_15 = l_Array_append___redArg(x_7, x_14);
lean_dec_ref(x_14);
lean_inc(x_12);
x_16 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_16, 1, x_6);
lean_ctor_set(x_16, 2, x_15);
x_17 = l_Lean_Syntax_node6(x_12, x_8, x_9, x_5, x_11, x_13, x_10, x_16);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_4);
return x_18;
}
block_82:
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; uint8_t x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; 
x_24 = lean_ctor_get(x_22, 1);
lean_inc(x_24);
x_25 = lean_ctor_get(x_22, 2);
lean_inc(x_25);
x_26 = lean_ctor_get(x_22, 5);
lean_inc(x_26);
lean_dec_ref(x_22);
x_27 = 0;
x_28 = l_Lean_SourceInfo_fromRef(x_26, x_27);
lean_dec(x_26);
x_29 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2;
x_30 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3;
lean_inc(x_28);
x_31 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_31, 0, x_28);
lean_ctor_set(x_31, 1, x_29);
x_32 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5;
x_33 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7;
x_34 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9;
x_35 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11;
x_36 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12;
lean_inc(x_28);
x_37 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_37, 0, x_28);
lean_ctor_set(x_37, 1, x_36);
x_38 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14;
x_39 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15;
lean_inc(x_25);
lean_inc(x_24);
x_40 = l_Lean_addMacroScope(x_24, x_39, x_25);
x_41 = lean_box(0);
x_42 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22;
lean_inc(x_28);
x_43 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_43, 0, x_28);
lean_ctor_set(x_43, 1, x_38);
lean_ctor_set(x_43, 2, x_40);
lean_ctor_set(x_43, 3, x_42);
lean_inc(x_28);
x_44 = l_Lean_Syntax_node2(x_28, x_35, x_37, x_43);
lean_inc(x_28);
x_45 = l_Lean_Syntax_node1(x_28, x_34, x_44);
lean_inc(x_28);
x_46 = l_Lean_Syntax_node1(x_28, x_33, x_45);
lean_inc(x_28);
x_47 = l_Lean_Syntax_node1(x_28, x_32, x_46);
x_48 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23;
lean_inc(x_28);
x_49 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_49, 0, x_28);
lean_ctor_set(x_49, 1, x_33);
lean_ctor_set(x_49, 2, x_48);
x_50 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24;
lean_inc(x_28);
x_51 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_51, 0, x_28);
lean_ctor_set(x_51, 1, x_50);
lean_inc(x_28);
x_52 = l_Lean_Syntax_node1(x_28, x_33, x_51);
x_53 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25;
lean_inc(x_28);
x_54 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_54, 0, x_28);
lean_ctor_set(x_54, 1, x_53);
x_55 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27;
x_56 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29;
x_57 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30;
lean_inc(x_25);
lean_inc(x_24);
x_58 = l_Lean_addMacroScope(x_24, x_57, x_25);
lean_inc(x_28);
x_59 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_59, 0, x_28);
lean_ctor_set(x_59, 1, x_56);
lean_ctor_set(x_59, 2, x_58);
lean_ctor_set(x_59, 3, x_41);
lean_inc_ref_n(x_49, 2);
lean_inc(x_28);
x_60 = l_Lean_Syntax_node3(x_28, x_55, x_49, x_49, x_59);
x_61 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31;
lean_inc(x_28);
x_62 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_62, 0, x_28);
lean_ctor_set(x_62, 1, x_61);
x_63 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33;
x_64 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34;
x_65 = l_Lean_addMacroScope(x_24, x_64, x_25);
lean_inc(x_28);
x_66 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_66, 0, x_28);
lean_ctor_set(x_66, 1, x_63);
lean_ctor_set(x_66, 2, x_65);
lean_ctor_set(x_66, 3, x_41);
lean_inc_ref_n(x_49, 2);
lean_inc(x_28);
x_67 = l_Lean_Syntax_node3(x_28, x_55, x_49, x_49, x_66);
lean_inc_ref(x_62);
x_68 = l_Array_mkArray4___redArg(x_60, x_62, x_67, x_62);
x_69 = l_Lean_Syntax_SepArray_ofElems(x_61, x_23);
lean_dec_ref(x_23);
x_70 = l_Array_append___redArg(x_68, x_69);
lean_dec_ref(x_69);
lean_inc(x_28);
x_71 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_71, 0, x_28);
lean_ctor_set(x_71, 1, x_33);
lean_ctor_set(x_71, 2, x_70);
x_72 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35;
lean_inc(x_28);
x_73 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_73, 0, x_28);
lean_ctor_set(x_73, 1, x_72);
lean_inc(x_28);
x_74 = l_Lean_Syntax_node3(x_28, x_33, x_54, x_71, x_73);
if (lean_obj_tag(x_21) == 1)
{
lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; 
x_75 = lean_ctor_get(x_21, 0);
lean_inc(x_75);
lean_dec_ref(x_21);
x_76 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37;
x_77 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__38;
lean_inc(x_28);
x_78 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_78, 0, x_28);
lean_ctor_set(x_78, 1, x_77);
lean_inc(x_28);
x_79 = l_Lean_Syntax_node2(x_28, x_76, x_78, x_75);
x_80 = l_Array_mkArray1___redArg(x_79);
x_4 = x_20;
x_5 = x_47;
x_6 = x_33;
x_7 = x_48;
x_8 = x_30;
x_9 = x_31;
x_10 = x_74;
x_11 = x_49;
x_12 = x_28;
x_13 = x_52;
x_14 = x_80;
goto block_19;
}
else
{
lean_object* x_81; 
lean_dec(x_21);
x_81 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__39;
x_4 = x_20;
x_5 = x_47;
x_6 = x_33;
x_7 = x_48;
x_8 = x_30;
x_9 = x_31;
x_10 = x_74;
x_11 = x_49;
x_12 = x_28;
x_13 = x_52;
x_14 = x_81;
goto block_19;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0___redArg(size_t x_1, size_t x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_usize_dec_lt(x_2, x_1);
if (x_4 == 0)
{
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; size_t x_8; size_t x_9; lean_object* x_10; 
x_5 = lean_array_uget(x_3, x_2);
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_array_uset(x_3, x_2, x_6);
x_8 = 1;
x_9 = lean_usize_add(x_2, x_8);
x_10 = lean_array_uset(x_7, x_2, x_5);
x_2 = x_9;
x_3 = x_10;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0(lean_object* x_1, size_t x_2, size_t x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0___redArg(x_2, x_3, x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_getSimpTheorems___boxed), 3, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_71; 
x_71 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40;
x_11 = x_71;
goto block_70;
}
else
{
lean_object* x_72; lean_object* x_73; 
x_72 = lean_ctor_get(x_1, 0);
x_73 = l_Lean_Syntax_TSepArray_getElems___redArg(x_72);
x_11 = x_73;
goto block_70;
}
block_70:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; size_t x_57; size_t x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; uint8_t x_67; lean_object* x_68; lean_object* x_69; 
x_12 = lean_ctor_get(x_8, 5);
x_13 = lean_ctor_get(x_8, 10);
x_14 = lean_ctor_get(x_8, 11);
x_15 = 0;
x_16 = l_Lean_SourceInfo_fromRef(x_12, x_15);
x_17 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2;
x_18 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3;
lean_inc(x_16);
x_19 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_19, 0, x_16);
lean_ctor_set(x_19, 1, x_17);
x_20 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5;
x_21 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7;
x_22 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9;
x_23 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11;
x_24 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12;
lean_inc(x_16);
x_25 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_25, 0, x_16);
lean_ctor_set(x_25, 1, x_24);
x_26 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14;
x_27 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15;
lean_inc(x_14);
lean_inc(x_13);
x_28 = l_Lean_addMacroScope(x_13, x_27, x_14);
x_29 = lean_box(0);
x_30 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22;
lean_inc(x_16);
x_31 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_31, 0, x_16);
lean_ctor_set(x_31, 1, x_26);
lean_ctor_set(x_31, 2, x_28);
lean_ctor_set(x_31, 3, x_30);
lean_inc(x_16);
x_32 = l_Lean_Syntax_node2(x_16, x_23, x_25, x_31);
lean_inc(x_16);
x_33 = l_Lean_Syntax_node1(x_16, x_22, x_32);
lean_inc(x_16);
x_34 = l_Lean_Syntax_node1(x_16, x_21, x_33);
lean_inc(x_16);
x_35 = l_Lean_Syntax_node1(x_16, x_20, x_34);
x_36 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23;
lean_inc(x_16);
x_37 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_37, 0, x_16);
lean_ctor_set(x_37, 1, x_21);
lean_ctor_set(x_37, 2, x_36);
x_38 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24;
lean_inc(x_16);
x_39 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_39, 0, x_16);
lean_ctor_set(x_39, 1, x_38);
lean_inc(x_16);
x_40 = l_Lean_Syntax_node1(x_16, x_21, x_39);
x_41 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25;
lean_inc(x_16);
x_42 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_42, 0, x_16);
lean_ctor_set(x_42, 1, x_41);
x_43 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27;
x_44 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29;
x_45 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30;
lean_inc(x_14);
lean_inc(x_13);
x_46 = l_Lean_addMacroScope(x_13, x_45, x_14);
lean_inc(x_16);
x_47 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_47, 0, x_16);
lean_ctor_set(x_47, 1, x_44);
lean_ctor_set(x_47, 2, x_46);
lean_ctor_set(x_47, 3, x_29);
lean_inc_ref_n(x_37, 2);
lean_inc(x_16);
x_48 = l_Lean_Syntax_node3(x_16, x_43, x_37, x_37, x_47);
x_49 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31;
lean_inc(x_16);
x_50 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_50, 0, x_16);
lean_ctor_set(x_50, 1, x_49);
x_51 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33;
x_52 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34;
lean_inc(x_14);
lean_inc(x_13);
x_53 = l_Lean_addMacroScope(x_13, x_52, x_14);
lean_inc(x_16);
x_54 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_54, 0, x_16);
lean_ctor_set(x_54, 1, x_51);
lean_ctor_set(x_54, 2, x_53);
lean_ctor_set(x_54, 3, x_29);
lean_inc_ref_n(x_37, 2);
lean_inc(x_16);
x_55 = l_Lean_Syntax_node3(x_16, x_43, x_37, x_37, x_54);
lean_inc_ref(x_50);
x_56 = l_Array_mkArray4___redArg(x_48, x_50, x_55, x_50);
x_57 = lean_array_size(x_11);
x_58 = 0;
x_59 = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0___redArg(x_57, x_58, x_11);
x_60 = l_Lean_Syntax_SepArray_ofElems(x_49, x_59);
lean_dec_ref(x_59);
x_61 = l_Array_append___redArg(x_56, x_60);
lean_dec_ref(x_60);
lean_inc(x_16);
x_62 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_62, 0, x_16);
lean_ctor_set(x_62, 1, x_21);
lean_ctor_set(x_62, 2, x_61);
x_63 = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35;
lean_inc(x_16);
x_64 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_64, 0, x_16);
lean_ctor_set(x_64, 1, x_63);
lean_inc(x_16);
x_65 = l_Lean_Syntax_node3(x_16, x_21, x_42, x_62, x_64);
lean_inc_ref(x_37);
x_66 = l_Lean_Syntax_node6(x_16, x_18, x_19, x_35, x_37, x_40, x_65, x_37);
x_67 = 0;
x_68 = lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___closed__0;
x_69 = l_Lean_Elab_Tactic_mkSimpContext(x_66, x_15, x_67, x_15, x_68, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_66);
return x_69;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
size_t x_5; size_t x_6; lean_object* x_7; 
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0(x_1, x_5, x_6, x_4);
lean_dec(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; size_t x_5; lean_object* x_6; 
x_4 = lean_unbox_usize(x_1);
lean_dec(x_1);
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0___redArg(x_4, x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_1);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lean_ctor_get(x_3, 1);
if (lean_obj_tag(x_9) == 0)
{
lean_object* x_10; uint8_t x_11; 
x_10 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_3);
x_11 = lean_expr_eqv(x_10, x_2);
if (x_11 == 0)
{
lean_object* x_12; 
lean_inc_ref(x_10);
x_12 = l_Lean_Meta_mkExpectedTypeHint(x_1, x_10, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_12) == 0)
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_12, 0);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_10);
lean_ctor_set(x_12, 0, x_15);
return x_12;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lean_ctor_get(x_12, 0);
lean_inc(x_16);
lean_dec(x_12);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_10);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
}
else
{
uint8_t x_19; 
lean_dec_ref(x_10);
x_19 = !lean_is_exclusive(x_12);
if (x_19 == 0)
{
return x_12;
}
else
{
lean_object* x_20; lean_object* x_21; 
x_20 = lean_ctor_get(x_12, 0);
lean_inc(x_20);
lean_dec(x_12);
x_21 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
}
}
else
{
lean_object* x_22; lean_object* x_23; 
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_1);
lean_ctor_set(x_22, 1, x_10);
x_23 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_23, 0, x_22);
return x_23;
}
}
else
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; 
lean_inc_ref(x_9);
x_24 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_24);
lean_dec_ref(x_3);
x_25 = lean_ctor_get(x_9, 0);
lean_inc(x_25);
lean_dec_ref(x_9);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc_ref(x_4);
x_26 = l_Lean_Meta_mkEqMP(x_25, x_1, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_26) == 0)
{
lean_object* x_27; lean_object* x_28; 
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
lean_dec_ref(x_26);
lean_inc_ref(x_24);
x_28 = l_Lean_Meta_mkExpectedTypeHint(x_27, x_24, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_28) == 0)
{
uint8_t x_29; 
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; 
x_30 = lean_ctor_get(x_28, 0);
x_31 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_31, 0, x_30);
lean_ctor_set(x_31, 1, x_24);
lean_ctor_set(x_28, 0, x_31);
return x_28;
}
else
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_32 = lean_ctor_get(x_28, 0);
lean_inc(x_32);
lean_dec(x_28);
x_33 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_33, 1, x_24);
x_34 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_34, 0, x_33);
return x_34;
}
}
else
{
uint8_t x_35; 
lean_dec_ref(x_24);
x_35 = !lean_is_exclusive(x_28);
if (x_35 == 0)
{
return x_28;
}
else
{
lean_object* x_36; lean_object* x_37; 
x_36 = lean_ctor_get(x_28, 0);
lean_inc(x_36);
lean_dec(x_28);
x_37 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_37, 0, x_36);
return x_37;
}
}
}
else
{
uint8_t x_38; 
lean_dec_ref(x_24);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
x_38 = !lean_is_exclusive(x_26);
if (x_38 == 0)
{
return x_26;
}
else
{
lean_object* x_39; lean_object* x_40; 
x_39 = lean_ctor_get(x_26, 0);
lean_inc(x_39);
lean_dec(x_26);
x_40 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_40, 0, x_39);
return x_40;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_2);
return x_9;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(32u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6() {
_start:
{
size_t x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = 5;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4;
x_4 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5;
x_5 = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_2);
lean_ctor_set(x_5, 3, x_2);
lean_ctor_set_usize(x_5, 4, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2;
x_3 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_2);
lean_ctor_set(x_3, 2, x_2);
lean_ctor_set(x_3, 3, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7;
x_2 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
x_13 = lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext(x_1, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lean_ctor_get(x_14, 0);
lean_inc_ref(x_15);
lean_dec(x_14);
x_16 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__0;
x_17 = lean_box(0);
x_18 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__8;
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc_ref(x_3);
x_19 = l_Lean_Meta_simp(x_3, x_15, x_16, x_17, x_18, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_19) == 0)
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec(x_20);
x_22 = lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27(x_2, x_3, x_21, x_8, x_9, x_10, x_11);
lean_dec_ref(x_3);
return x_22;
}
else
{
uint8_t x_23; 
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_23 = !lean_is_exclusive(x_19);
if (x_23 == 0)
{
return x_19;
}
else
{
lean_object* x_24; lean_object* x_25; 
x_24 = lean_ctor_get(x_19, 0);
lean_inc(x_24);
lean_dec(x_19);
x_25 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_25, 0, x_24);
return x_25;
}
}
}
else
{
uint8_t x_26; 
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_26 = !lean_is_exclusive(x_13);
if (x_26 == 0)
{
return x_13;
}
else
{
lean_object* x_27; lean_object* x_28; 
x_27 = lean_ctor_get(x_13, 0);
lean_inc(x_27);
lean_dec(x_13);
x_28 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_28, 0, x_27);
return x_28;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Mathlib_Tactic_Zify_zifyProof(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_1);
return x_13;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Zify(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__0 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__0();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__0);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__2 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__2();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__2);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__5 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__5();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__5);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__7 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__7();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__7);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__8 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__8();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__8);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__15 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__15();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__15);
lp_mathlib_Mathlib_Tactic_Zify_zify___closed__16 = _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__16();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__16);
lp_mathlib_Mathlib_Tactic_Zify_zify = _init_lp_mathlib_Mathlib_Tactic_Zify_zify();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__4 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__4();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__4);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__6 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__6();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__6);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__8 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__8();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__8);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__10 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__10();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__10);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__16 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__16();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__16);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__18 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__18();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__18);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__20 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__20();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__20);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__21 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__21();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__21);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__26 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__26();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__26);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__36 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__36();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__36);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__38 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__38();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__38);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__39 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__39();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__39);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__41 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__41();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__41);
lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42 = _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42);
lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___closed__0 = _init_lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___closed__0();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___closed__0);
lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__0 = _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__0();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__0);
lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1 = _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1);
lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2 = _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2);
lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3 = _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3);
lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4 = _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4);
lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5 = _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5);
lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6 = _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6);
lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7 = _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7);
lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__8 = _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__8();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__8);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
