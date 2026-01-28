// Lean compiler output
// Module: Mathlib.Data.ENat.Defs
// Imports: public import Init public import Mathlib.Data.Nat.Notation public import Mathlib.Order.TypeTags
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
static lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_term_u2115_u221e;
LEAN_EXPORT lean_object* lp_mathlib_ENat_instNatCast;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTopENat;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2;
static lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__0;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
static lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1;
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
static lean_object* lp_mathlib_term_u2115_u221e___closed__0;
static lean_object* lp_mathlib_ENat_instNatCast___closed__0;
lean_object* lp_mathlib_WithTop_some(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedENat;
static lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__4;
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_term_u2115_u221e___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_term_u2115_u221e___closed__3;
static lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__1;
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_term_u2115_u221e___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
static lean_object* lp_mathlib_term_u2115_u221e___closed__1;
static lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_instTopENat() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instInhabitedENat() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_term_u2115_u221e___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("termℕ∞", 10, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_term_u2115_u221e___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_term_u2115_u221e___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_term_u2115_u221e___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ℕ∞", 6, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_term_u2115_u221e___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_term_u2115_u221e___closed__2;
x_2 = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_term_u2115_u221e___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_term_u2115_u221e___closed__3;
x_2 = lean_unsigned_to_nat(1024u);
x_3 = lp_mathlib_term_u2115_u221e___closed__1;
x_4 = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_term_u2115_u221e() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_term_u2115_u221e___closed__4;
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ENat", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__3;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib_term_u2115_u221e___closed__1;
x_5 = l_Lean_Syntax_isOfKind(x_1, x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
lean_dec_ref(x_2);
x_6 = lean_box(1);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_2, 5);
lean_inc(x_10);
lean_dec_ref(x_2);
x_11 = 0;
x_12 = l_Lean_SourceInfo_fromRef(x_10, x_11);
lean_dec(x_10);
x_13 = lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1;
x_14 = lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2;
x_15 = l_Lean_addMacroScope(x_8, x_14, x_9);
x_16 = lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__4;
x_17 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_17, 0, x_12);
lean_ctor_set(x_17, 1, x_13);
lean_ctor_set(x_17, 2, x_15);
lean_ctor_set(x_17, 3, x_16);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_3);
return x_18;
}
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ident", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__1;
lean_inc(x_1);
x_5 = l_Lean_Syntax_isOfKind(x_1, x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_1);
x_6 = lean_box(0);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
return x_7;
}
else
{
lean_object* x_8; uint8_t x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_8 = l_Lean_replaceRef(x_1, x_2);
lean_dec(x_1);
x_9 = 0;
x_10 = l_Lean_SourceInfo_fromRef(x_8, x_9);
lean_dec(x_8);
x_11 = lp_mathlib_term_u2115_u221e___closed__1;
x_12 = lp_mathlib_term_u2115_u221e___closed__2;
lean_inc(x_10);
x_13 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_13, 0, x_10);
lean_ctor_set(x_13, 1, x_12);
x_14 = l_Lean_Syntax_node1(x_10, x_11, x_13);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_3);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_ENat_instNatCast___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_WithTop_some), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_ENat_instNatCast() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_ENat_instNatCast___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_2);
lean_inc(x_1);
return x_1;
}
else
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ENat_recTopCoe___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ENat_recTopCoe(x_1, x_2, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_ENat_recTopCoe___redArg(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_TypeTags(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ENat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instTopENat = _init_lp_mathlib_instTopENat();
lean_mark_persistent(lp_mathlib_instTopENat);
lp_mathlib_instInhabitedENat = _init_lp_mathlib_instInhabitedENat();
lean_mark_persistent(lp_mathlib_instInhabitedENat);
lp_mathlib_term_u2115_u221e___closed__0 = _init_lp_mathlib_term_u2115_u221e___closed__0();
lean_mark_persistent(lp_mathlib_term_u2115_u221e___closed__0);
lp_mathlib_term_u2115_u221e___closed__1 = _init_lp_mathlib_term_u2115_u221e___closed__1();
lean_mark_persistent(lp_mathlib_term_u2115_u221e___closed__1);
lp_mathlib_term_u2115_u221e___closed__2 = _init_lp_mathlib_term_u2115_u221e___closed__2();
lean_mark_persistent(lp_mathlib_term_u2115_u221e___closed__2);
lp_mathlib_term_u2115_u221e___closed__3 = _init_lp_mathlib_term_u2115_u221e___closed__3();
lean_mark_persistent(lp_mathlib_term_u2115_u221e___closed__3);
lp_mathlib_term_u2115_u221e___closed__4 = _init_lp_mathlib_term_u2115_u221e___closed__4();
lean_mark_persistent(lp_mathlib_term_u2115_u221e___closed__4);
lp_mathlib_term_u2115_u221e = _init_lp_mathlib_term_u2115_u221e();
lean_mark_persistent(lp_mathlib_term_u2115_u221e);
lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0 = _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0);
lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1 = _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1);
lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2 = _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2);
lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__3 = _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__3();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__3);
lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__4 = _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__4();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__4);
lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__0 = _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__0();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__0);
lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__1 = _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__1();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__1);
lp_mathlib_ENat_instNatCast___closed__0 = _init_lp_mathlib_ENat_instNatCast___closed__0();
lean_mark_persistent(lp_mathlib_ENat_instNatCast___closed__0);
lp_mathlib_ENat_instNatCast = _init_lp_mathlib_ENat_instNatCast();
lean_mark_persistent(lp_mathlib_ENat_instNatCast);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
