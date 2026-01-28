// Lean compiler output
// Module: Mathlib.Order.ConditionallyCompleteLattice.Finset
// Imports: public import Init public import Mathlib.Data.Finset.Max public import Mathlib.Data.Set.Finite.Lattice public import Mathlib.Order.ConditionallyCompleteLattice.Indexed
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
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__71;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__55;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40;
lean_object* lean_array_push(lean_object*, lean_object*);
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__107;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__113;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__34;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__112;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45;
lean_object* lean_string_utf8_byte_size(lean_object*);
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__57;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__51;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__104;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__15;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__106;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__3;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__105;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__87;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__75;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12;
lean_object* l_Array_empty(lean_object*);
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__74;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96;
LEAN_EXPORT lean_object* lp_mathlib_Finset_ciInf__eq__min_x27__image___auto__1;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__8;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__114;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__20;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__59;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__46;
LEAN_EXPORT lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__103;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__6;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__17;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__110;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__33;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__25;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__29;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__27;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__32;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__21;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__108;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__43;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88;
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__111;
lean_object* l_Lean_mkAtom(lean_object*);
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__47;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__109;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__68;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__66;
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lean", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Parser", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticSeq", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__3;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1;
x_4 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticSeq1Indented", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__6;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1;
x_4 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("null", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__8;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("classical", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1;
x_4 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("exact", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1;
x_4 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Term", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("app", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1;
x_4 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__21() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("image_nonempty.mpr", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__21;
x_2 = lean_string_utf8_byte_size(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__21;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("image_nonempty", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__25() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mpr", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__25;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__27() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23;
x_4 = lean_box(2);
x_5 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_2);
lean_ctor_set(x_5, 3, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__27;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__29() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("paren", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__29;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1;
x_4 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("hygienicLParen", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__32() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1;
x_4 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__33() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("(", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__34() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__33;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__34;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("hygieneInfo", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("[anonymous]", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38;
x_2 = lean_string_utf8_byte_size(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40;
x_4 = lean_box(2);
x_5 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_2);
lean_ctor_set(x_5, 3, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__43() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__43;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__32;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__46() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__47() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("h.imp", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__47;
x_2 = lean_string_utf8_byte_size(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__47;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("h", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__51() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("imp", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__51;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49;
x_4 = lean_box(2);
x_5 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_2);
lean_ctor_set(x_5, 3, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__55() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("fun", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__55;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1;
x_4 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__57() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__55;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__57;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__59() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("basicFun", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__59;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1;
x_4 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("hole", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1;
x_4 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("_", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__66() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__66;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__68() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__68;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__71() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("↦", 3, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__74() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__71;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__75() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("And.left", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__75;
x_2 = lean_string_utf8_byte_size(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__75;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("And", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("left", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80;
x_3 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77;
x_4 = lean_box(2);
x_5 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_2);
lean_ctor_set(x_5, 3, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__74;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__87() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__87;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__20;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__46;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(")", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91;
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__20;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__17;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__15;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__103() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__104() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__103;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__105() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__104;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__106() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__105;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__107() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__106;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__108() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__107;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__109() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__108;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__110() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__109;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__111() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__110;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__112() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__111;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__113() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__112;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__114() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__113;
x_2 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4;
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__114;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Finset_ciInf__eq__min_x27__image___auto__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__114;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Indexed(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Indexed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__3 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__3();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__3);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__6 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__6();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__6);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__8 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__8();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__8);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__15 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__15();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__15);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__17 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__17();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__17);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__20 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__20();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__20);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__21 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__21();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__21);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__25 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__25();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__25);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__27 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__27();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__27);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__29 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__29();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__29);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__32 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__32();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__32);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__33 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__33();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__33);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__34 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__34();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__34);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__43 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__43();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__43);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__46 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__46();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__46);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__47 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__47();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__47);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__51 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__51();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__51);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__55 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__55();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__55);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__57 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__57();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__57);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__59 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__59();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__59);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__66 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__66();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__66);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__68 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__68();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__68);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__71 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__71();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__71);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__74 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__74();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__74);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__75 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__75();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__75);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__87 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__87();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__87);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__103 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__103();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__103);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__104 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__104();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__104);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__105 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__105();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__105);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__106 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__106();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__106);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__107 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__107();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__107);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__108 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__108();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__108);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__109 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__109();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__109);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__110 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__110();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__110);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__111 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__111();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__111);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__112 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__112();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__112);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__113 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__113();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__113);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__114 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__114();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__114);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1);
lp_mathlib_Finset_ciInf__eq__min_x27__image___auto__1 = _init_lp_mathlib_Finset_ciInf__eq__min_x27__image___auto__1();
lean_mark_persistent(lp_mathlib_Finset_ciInf__eq__min_x27__image___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
