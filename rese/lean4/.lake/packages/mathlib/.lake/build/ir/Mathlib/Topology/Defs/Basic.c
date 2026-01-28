// Lean compiler output
// Module: Mathlib.Topology.Defs.Basic
// Imports: public import Init public import Mathlib.Order.SetNotation public import Mathlib.Tactic.Continuity public import Mathlib.Tactic.FunProp public import Mathlib.Tactic.MkIffOfInductiveProp public import Mathlib.Tactic.ToAdditive public import Mathlib.Util.AssertExists
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
static lean_object* lp_mathlib_Topology_Continuous__of___closed__4;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__11;
static lean_object* lp_mathlib_Topology_closure__of___closed__1;
static lean_object* lp_mathlib_Topology_Continuous__of___closed__1;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__0;
static lean_object* lp_mathlib_Topology_IsClosed__of___closed__5;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__1;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__3;
static lean_object* lp_mathlib_Topology_closure__of___closed__3;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__6;
static lean_object* lp_mathlib_Topology_Continuous__of___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Topology_IsClosed__of;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__1;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__2;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__0;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__11;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__3;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__2;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__10;
static lean_object* lp_mathlib_Topology_IsClosed__of___closed__4;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__17;
static lean_object* lp_mathlib_Topology_Continuous__of___closed__9;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__2;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__15;
static lean_object* lp_mathlib_Topology_IsClosed__of___closed__6;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Topology_Continuous__of___closed__2;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__18;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__5;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__8;
static lean_object* lp_mathlib_Topology_IsClosed__of___closed__3;
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Topology_IsOpen__of;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__6;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__12;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__13;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Topology_Continuous__of;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__14;
static lean_object* lp_mathlib_Topology_IsClosed__of___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Topology_closure__of;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__14;
static lean_object* lp_mathlib_Topology_closure__of___closed__2;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__2;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__4;
static lean_object* lp_mathlib_Topology_Continuous__of___closed__3;
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Topology_Continuous__of___closed__0;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__7;
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__2;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__5;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__6;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__4;
static lean_object* lp_mathlib_Topology_Continuous__of___closed__6;
lean_object* l_String_toRawSubstring_x27(lean_object*);
static lean_object* lp_mathlib_Topology_Continuous__of___closed__10;
static lean_object* lp_mathlib_Topology_IsClosed__of___closed__2;
static lean_object* lp_mathlib_Topology_closure__of___closed__6;
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Topology_IsClosed__of___closed__1;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__1;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__12;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__5;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__7;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__19;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__9;
static lean_object* lp_mathlib_Topology_closure__of___closed__4;
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__3;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__0;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__9;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__4;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__4;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__4;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__5;
static lean_object* lp_mathlib_Topology_closure__of___closed__0;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Topology_closure__of___closed__5;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__0;
static lean_object* lp_mathlib_Topology_Continuous__of___closed__8;
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__8;
static lean_object* lp_mathlib_Topology_Continuous__of___closed__5;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__13;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__1;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__6;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__16;
static lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Topology_IsOpen__of___closed__3;
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Topology", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("IsOpen_of", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__1;
x_2 = lp_mathlib_Topology_IsOpen__of___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("andthen", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__3;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("IsOpen[", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__5;
x_2 = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("term", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__7;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lp_mathlib_Topology_IsOpen__of___closed__8;
x_3 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__9;
x_2 = lp_mathlib_Topology_IsOpen__of___closed__6;
x_3 = lp_mathlib_Topology_IsOpen__of___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("]", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__11;
x_2 = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__12;
x_2 = lp_mathlib_Topology_IsOpen__of___closed__10;
x_3 = lp_mathlib_Topology_IsOpen__of___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__13;
x_2 = lean_unsigned_to_nat(1024u);
x_3 = lp_mathlib_Topology_IsOpen__of___closed__2;
x_4 = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_IsOpen__of() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__14;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lean", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Parser", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Term", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("app", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__3;
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__2;
x_3 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__1;
x_4 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("explicit", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__5;
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__2;
x_3 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__1;
x_4 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("@", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("IsOpen", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__8;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__8;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__10;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__10;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__12;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__13;
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__11;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("null", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__15;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__17() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("hole", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__17;
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__2;
x_3 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__1;
x_4 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__19() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("_", 1, 1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib_Topology_IsOpen__of___closed__2;
lean_inc(x_1);
x_5 = l_Lean_Syntax_isOfKind(x_1, x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
lean_dec_ref(x_2);
lean_dec(x_1);
x_6 = lean_box(1);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_2, 5);
lean_inc(x_10);
lean_dec_ref(x_2);
x_11 = lean_unsigned_to_nat(1u);
x_12 = l_Lean_Syntax_getArg(x_1, x_11);
lean_dec(x_1);
x_13 = 0;
x_14 = l_Lean_SourceInfo_fromRef(x_10, x_13);
lean_dec(x_10);
x_15 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__4;
x_16 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__6;
x_17 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__7;
lean_inc(x_14);
x_18 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_18, 0, x_14);
lean_ctor_set(x_18, 1, x_17);
x_19 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__9;
x_20 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__10;
x_21 = l_Lean_addMacroScope(x_8, x_20, x_9);
x_22 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__14;
lean_inc(x_14);
x_23 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_23, 0, x_14);
lean_ctor_set(x_23, 1, x_19);
lean_ctor_set(x_23, 2, x_21);
lean_ctor_set(x_23, 3, x_22);
lean_inc(x_14);
x_24 = l_Lean_Syntax_node2(x_14, x_16, x_18, x_23);
x_25 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__16;
x_26 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__18;
x_27 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__19;
lean_inc(x_14);
x_28 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_28, 0, x_14);
lean_ctor_set(x_28, 1, x_27);
lean_inc(x_14);
x_29 = l_Lean_Syntax_node1(x_14, x_26, x_28);
lean_inc(x_14);
x_30 = l_Lean_Syntax_node2(x_14, x_25, x_29, x_12);
x_31 = l_Lean_Syntax_node2(x_14, x_15, x_24, x_30);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_3);
return x_32;
}
}
}
static lean_object* _init_lp_mathlib_Topology_IsClosed__of___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("IsClosed_of", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_IsClosed__of___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Topology_IsClosed__of___closed__0;
x_2 = lp_mathlib_Topology_IsOpen__of___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology_IsClosed__of___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("IsClosed[", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_IsClosed__of___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology_IsClosed__of___closed__2;
x_2 = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology_IsClosed__of___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__9;
x_2 = lp_mathlib_Topology_IsClosed__of___closed__3;
x_3 = lp_mathlib_Topology_IsOpen__of___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_IsClosed__of___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__12;
x_2 = lp_mathlib_Topology_IsClosed__of___closed__4;
x_3 = lp_mathlib_Topology_IsOpen__of___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_IsClosed__of___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_IsClosed__of___closed__5;
x_2 = lean_unsigned_to_nat(1024u);
x_3 = lp_mathlib_Topology_IsClosed__of___closed__1;
x_4 = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_IsClosed__of() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Topology_IsClosed__of___closed__6;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("IsClosed", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__0;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__2;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__2;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__4;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__5;
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__3;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib_Topology_IsClosed__of___closed__1;
lean_inc(x_1);
x_5 = l_Lean_Syntax_isOfKind(x_1, x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
lean_dec_ref(x_2);
lean_dec(x_1);
x_6 = lean_box(1);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_2, 5);
lean_inc(x_10);
lean_dec_ref(x_2);
x_11 = lean_unsigned_to_nat(1u);
x_12 = l_Lean_Syntax_getArg(x_1, x_11);
lean_dec(x_1);
x_13 = 0;
x_14 = l_Lean_SourceInfo_fromRef(x_10, x_13);
lean_dec(x_10);
x_15 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__4;
x_16 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__6;
x_17 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__7;
lean_inc(x_14);
x_18 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_18, 0, x_14);
lean_ctor_set(x_18, 1, x_17);
x_19 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__1;
x_20 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__2;
x_21 = l_Lean_addMacroScope(x_8, x_20, x_9);
x_22 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__6;
lean_inc(x_14);
x_23 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_23, 0, x_14);
lean_ctor_set(x_23, 1, x_19);
lean_ctor_set(x_23, 2, x_21);
lean_ctor_set(x_23, 3, x_22);
lean_inc(x_14);
x_24 = l_Lean_Syntax_node2(x_14, x_16, x_18, x_23);
x_25 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__16;
x_26 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__18;
x_27 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__19;
lean_inc(x_14);
x_28 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_28, 0, x_14);
lean_ctor_set(x_28, 1, x_27);
lean_inc(x_14);
x_29 = l_Lean_Syntax_node1(x_14, x_26, x_28);
lean_inc(x_14);
x_30 = l_Lean_Syntax_node2(x_14, x_25, x_29, x_12);
x_31 = l_Lean_Syntax_node2(x_14, x_15, x_24, x_30);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_3);
return x_32;
}
}
}
static lean_object* _init_lp_mathlib_Topology_closure__of___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("closure_of", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_closure__of___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Topology_closure__of___closed__0;
x_2 = lp_mathlib_Topology_IsOpen__of___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology_closure__of___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("closure[", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_closure__of___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology_closure__of___closed__2;
x_2 = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology_closure__of___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__9;
x_2 = lp_mathlib_Topology_closure__of___closed__3;
x_3 = lp_mathlib_Topology_IsOpen__of___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_closure__of___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__12;
x_2 = lp_mathlib_Topology_closure__of___closed__4;
x_3 = lp_mathlib_Topology_IsOpen__of___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_closure__of___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_closure__of___closed__5;
x_2 = lean_unsigned_to_nat(1024u);
x_3 = lp_mathlib_Topology_closure__of___closed__1;
x_4 = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_closure__of() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Topology_closure__of___closed__6;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("closure", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__0;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__2;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__3;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib_Topology_closure__of___closed__1;
lean_inc(x_1);
x_5 = l_Lean_Syntax_isOfKind(x_1, x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
lean_dec_ref(x_2);
lean_dec(x_1);
x_6 = lean_box(1);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_2, 5);
lean_inc(x_10);
lean_dec_ref(x_2);
x_11 = lean_unsigned_to_nat(1u);
x_12 = l_Lean_Syntax_getArg(x_1, x_11);
lean_dec(x_1);
x_13 = 0;
x_14 = l_Lean_SourceInfo_fromRef(x_10, x_13);
lean_dec(x_10);
x_15 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__4;
x_16 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__6;
x_17 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__7;
lean_inc(x_14);
x_18 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_18, 0, x_14);
lean_ctor_set(x_18, 1, x_17);
x_19 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__1;
x_20 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__2;
x_21 = l_Lean_addMacroScope(x_8, x_20, x_9);
x_22 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__4;
lean_inc(x_14);
x_23 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_23, 0, x_14);
lean_ctor_set(x_23, 1, x_19);
lean_ctor_set(x_23, 2, x_21);
lean_ctor_set(x_23, 3, x_22);
lean_inc(x_14);
x_24 = l_Lean_Syntax_node2(x_14, x_16, x_18, x_23);
x_25 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__16;
x_26 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__18;
x_27 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__19;
lean_inc(x_14);
x_28 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_28, 0, x_14);
lean_ctor_set(x_28, 1, x_27);
lean_inc(x_14);
x_29 = l_Lean_Syntax_node1(x_14, x_26, x_28);
lean_inc(x_14);
x_30 = l_Lean_Syntax_node2(x_14, x_25, x_29, x_12);
x_31 = l_Lean_Syntax_node2(x_14, x_15, x_24, x_30);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_3);
return x_32;
}
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Continuous_of", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Topology_Continuous__of___closed__0;
x_2 = lp_mathlib_Topology_IsOpen__of___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Continuous[", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology_Continuous__of___closed__2;
x_2 = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__9;
x_2 = lp_mathlib_Topology_Continuous__of___closed__3;
x_3 = lp_mathlib_Topology_IsOpen__of___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(", ", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology_Continuous__of___closed__5;
x_2 = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_Continuous__of___closed__6;
x_2 = lp_mathlib_Topology_Continuous__of___closed__4;
x_3 = lp_mathlib_Topology_IsOpen__of___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__9;
x_2 = lp_mathlib_Topology_Continuous__of___closed__7;
x_3 = lp_mathlib_Topology_IsOpen__of___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_IsOpen__of___closed__12;
x_2 = lp_mathlib_Topology_Continuous__of___closed__8;
x_3 = lp_mathlib_Topology_IsOpen__of___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Topology_Continuous__of___closed__9;
x_2 = lean_unsigned_to_nat(1024u);
x_3 = lp_mathlib_Topology_Continuous__of___closed__1;
x_4 = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Topology_Continuous__of() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Topology_Continuous__of___closed__10;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Continuous", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__0;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__2;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__2;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__4;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__5;
x_2 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__3;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib_Topology_Continuous__of___closed__1;
lean_inc(x_1);
x_5 = l_Lean_Syntax_isOfKind(x_1, x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
lean_dec_ref(x_2);
lean_dec(x_1);
x_6 = lean_box(1);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_2, 5);
lean_inc(x_10);
lean_dec_ref(x_2);
x_11 = lean_unsigned_to_nat(1u);
x_12 = l_Lean_Syntax_getArg(x_1, x_11);
x_13 = lean_unsigned_to_nat(3u);
x_14 = l_Lean_Syntax_getArg(x_1, x_13);
lean_dec(x_1);
x_15 = 0;
x_16 = l_Lean_SourceInfo_fromRef(x_10, x_15);
lean_dec(x_10);
x_17 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__4;
x_18 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__6;
x_19 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__7;
lean_inc(x_16);
x_20 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_20, 0, x_16);
lean_ctor_set(x_20, 1, x_19);
x_21 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__1;
x_22 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__2;
x_23 = l_Lean_addMacroScope(x_8, x_22, x_9);
x_24 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__6;
lean_inc(x_16);
x_25 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_25, 0, x_16);
lean_ctor_set(x_25, 1, x_21);
lean_ctor_set(x_25, 2, x_23);
lean_ctor_set(x_25, 3, x_24);
lean_inc(x_16);
x_26 = l_Lean_Syntax_node2(x_16, x_18, x_20, x_25);
x_27 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__16;
x_28 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__18;
x_29 = lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__19;
lean_inc(x_16);
x_30 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_30, 0, x_16);
lean_ctor_set(x_30, 1, x_29);
lean_inc(x_16);
x_31 = l_Lean_Syntax_node1(x_16, x_28, x_30);
lean_inc(x_31);
lean_inc(x_16);
x_32 = l_Lean_Syntax_node4(x_16, x_27, x_31, x_31, x_12, x_14);
x_33 = l_Lean_Syntax_node2(x_16, x_17, x_26, x_32);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_3);
return x_34;
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Continuity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AssertExists(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Defs_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Continuity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AssertExists(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Topology_IsOpen__of___closed__0 = _init_lp_mathlib_Topology_IsOpen__of___closed__0();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__0);
lp_mathlib_Topology_IsOpen__of___closed__1 = _init_lp_mathlib_Topology_IsOpen__of___closed__1();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__1);
lp_mathlib_Topology_IsOpen__of___closed__2 = _init_lp_mathlib_Topology_IsOpen__of___closed__2();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__2);
lp_mathlib_Topology_IsOpen__of___closed__3 = _init_lp_mathlib_Topology_IsOpen__of___closed__3();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__3);
lp_mathlib_Topology_IsOpen__of___closed__4 = _init_lp_mathlib_Topology_IsOpen__of___closed__4();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__4);
lp_mathlib_Topology_IsOpen__of___closed__5 = _init_lp_mathlib_Topology_IsOpen__of___closed__5();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__5);
lp_mathlib_Topology_IsOpen__of___closed__6 = _init_lp_mathlib_Topology_IsOpen__of___closed__6();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__6);
lp_mathlib_Topology_IsOpen__of___closed__7 = _init_lp_mathlib_Topology_IsOpen__of___closed__7();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__7);
lp_mathlib_Topology_IsOpen__of___closed__8 = _init_lp_mathlib_Topology_IsOpen__of___closed__8();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__8);
lp_mathlib_Topology_IsOpen__of___closed__9 = _init_lp_mathlib_Topology_IsOpen__of___closed__9();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__9);
lp_mathlib_Topology_IsOpen__of___closed__10 = _init_lp_mathlib_Topology_IsOpen__of___closed__10();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__10);
lp_mathlib_Topology_IsOpen__of___closed__11 = _init_lp_mathlib_Topology_IsOpen__of___closed__11();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__11);
lp_mathlib_Topology_IsOpen__of___closed__12 = _init_lp_mathlib_Topology_IsOpen__of___closed__12();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__12);
lp_mathlib_Topology_IsOpen__of___closed__13 = _init_lp_mathlib_Topology_IsOpen__of___closed__13();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__13);
lp_mathlib_Topology_IsOpen__of___closed__14 = _init_lp_mathlib_Topology_IsOpen__of___closed__14();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of___closed__14);
lp_mathlib_Topology_IsOpen__of = _init_lp_mathlib_Topology_IsOpen__of();
lean_mark_persistent(lp_mathlib_Topology_IsOpen__of);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__0 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__0();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__0);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__1 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__1();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__1);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__2 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__2();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__2);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__3 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__3();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__3);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__4 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__4();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__4);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__5 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__5();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__5);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__6 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__6();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__6);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__7 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__7();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__7);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__8 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__8();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__8);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__9 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__9();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__9);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__10 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__10();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__10);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__11 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__11();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__11);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__12 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__12();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__12);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__13 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__13();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__13);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__14 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__14();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__14);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__15 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__15();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__15);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__16 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__16();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__16);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__17 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__17();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__17);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__18 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__18();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__18);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__19 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__19();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsOpen__of__1___closed__19);
lp_mathlib_Topology_IsClosed__of___closed__0 = _init_lp_mathlib_Topology_IsClosed__of___closed__0();
lean_mark_persistent(lp_mathlib_Topology_IsClosed__of___closed__0);
lp_mathlib_Topology_IsClosed__of___closed__1 = _init_lp_mathlib_Topology_IsClosed__of___closed__1();
lean_mark_persistent(lp_mathlib_Topology_IsClosed__of___closed__1);
lp_mathlib_Topology_IsClosed__of___closed__2 = _init_lp_mathlib_Topology_IsClosed__of___closed__2();
lean_mark_persistent(lp_mathlib_Topology_IsClosed__of___closed__2);
lp_mathlib_Topology_IsClosed__of___closed__3 = _init_lp_mathlib_Topology_IsClosed__of___closed__3();
lean_mark_persistent(lp_mathlib_Topology_IsClosed__of___closed__3);
lp_mathlib_Topology_IsClosed__of___closed__4 = _init_lp_mathlib_Topology_IsClosed__of___closed__4();
lean_mark_persistent(lp_mathlib_Topology_IsClosed__of___closed__4);
lp_mathlib_Topology_IsClosed__of___closed__5 = _init_lp_mathlib_Topology_IsClosed__of___closed__5();
lean_mark_persistent(lp_mathlib_Topology_IsClosed__of___closed__5);
lp_mathlib_Topology_IsClosed__of___closed__6 = _init_lp_mathlib_Topology_IsClosed__of___closed__6();
lean_mark_persistent(lp_mathlib_Topology_IsClosed__of___closed__6);
lp_mathlib_Topology_IsClosed__of = _init_lp_mathlib_Topology_IsClosed__of();
lean_mark_persistent(lp_mathlib_Topology_IsClosed__of);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__0 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__0();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__0);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__1 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__1();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__1);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__2 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__2();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__2);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__3 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__3();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__3);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__4 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__4();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__4);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__5 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__5();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__5);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__6 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__6();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__IsClosed__of__1___closed__6);
lp_mathlib_Topology_closure__of___closed__0 = _init_lp_mathlib_Topology_closure__of___closed__0();
lean_mark_persistent(lp_mathlib_Topology_closure__of___closed__0);
lp_mathlib_Topology_closure__of___closed__1 = _init_lp_mathlib_Topology_closure__of___closed__1();
lean_mark_persistent(lp_mathlib_Topology_closure__of___closed__1);
lp_mathlib_Topology_closure__of___closed__2 = _init_lp_mathlib_Topology_closure__of___closed__2();
lean_mark_persistent(lp_mathlib_Topology_closure__of___closed__2);
lp_mathlib_Topology_closure__of___closed__3 = _init_lp_mathlib_Topology_closure__of___closed__3();
lean_mark_persistent(lp_mathlib_Topology_closure__of___closed__3);
lp_mathlib_Topology_closure__of___closed__4 = _init_lp_mathlib_Topology_closure__of___closed__4();
lean_mark_persistent(lp_mathlib_Topology_closure__of___closed__4);
lp_mathlib_Topology_closure__of___closed__5 = _init_lp_mathlib_Topology_closure__of___closed__5();
lean_mark_persistent(lp_mathlib_Topology_closure__of___closed__5);
lp_mathlib_Topology_closure__of___closed__6 = _init_lp_mathlib_Topology_closure__of___closed__6();
lean_mark_persistent(lp_mathlib_Topology_closure__of___closed__6);
lp_mathlib_Topology_closure__of = _init_lp_mathlib_Topology_closure__of();
lean_mark_persistent(lp_mathlib_Topology_closure__of);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__0 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__0();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__0);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__1 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__1();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__1);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__2 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__2();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__2);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__3 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__3();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__3);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__4 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__4();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__closure__of__1___closed__4);
lp_mathlib_Topology_Continuous__of___closed__0 = _init_lp_mathlib_Topology_Continuous__of___closed__0();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of___closed__0);
lp_mathlib_Topology_Continuous__of___closed__1 = _init_lp_mathlib_Topology_Continuous__of___closed__1();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of___closed__1);
lp_mathlib_Topology_Continuous__of___closed__2 = _init_lp_mathlib_Topology_Continuous__of___closed__2();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of___closed__2);
lp_mathlib_Topology_Continuous__of___closed__3 = _init_lp_mathlib_Topology_Continuous__of___closed__3();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of___closed__3);
lp_mathlib_Topology_Continuous__of___closed__4 = _init_lp_mathlib_Topology_Continuous__of___closed__4();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of___closed__4);
lp_mathlib_Topology_Continuous__of___closed__5 = _init_lp_mathlib_Topology_Continuous__of___closed__5();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of___closed__5);
lp_mathlib_Topology_Continuous__of___closed__6 = _init_lp_mathlib_Topology_Continuous__of___closed__6();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of___closed__6);
lp_mathlib_Topology_Continuous__of___closed__7 = _init_lp_mathlib_Topology_Continuous__of___closed__7();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of___closed__7);
lp_mathlib_Topology_Continuous__of___closed__8 = _init_lp_mathlib_Topology_Continuous__of___closed__8();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of___closed__8);
lp_mathlib_Topology_Continuous__of___closed__9 = _init_lp_mathlib_Topology_Continuous__of___closed__9();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of___closed__9);
lp_mathlib_Topology_Continuous__of___closed__10 = _init_lp_mathlib_Topology_Continuous__of___closed__10();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of___closed__10);
lp_mathlib_Topology_Continuous__of = _init_lp_mathlib_Topology_Continuous__of();
lean_mark_persistent(lp_mathlib_Topology_Continuous__of);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__0 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__0();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__0);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__1 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__1();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__1);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__2 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__2();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__2);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__3 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__3();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__3);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__4 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__4();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__4);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__5 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__5();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__5);
lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__6 = _init_lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__6();
lean_mark_persistent(lp_mathlib_Topology___aux__Mathlib__Topology__Defs__Basic______macroRules__Topology__Continuous__of__1___closed__6);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
