// Lean compiler output
// Module: Aesop.BuiltinRules.Rfl
// Imports: public import Init public import Aesop.Frontend.Attribute
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__2;
static lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__1;
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__5;
static lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__0;
static lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4;
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_ofTacticSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__3;
static lean_object* _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticRfl", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Parser", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lean", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__3;
x_2 = lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__2;
x_3 = lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__1;
x_4 = lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("rfl", 3, 3);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; uint8_t x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_7 = lean_ctor_get(x_4, 5);
x_8 = 0;
x_9 = l_Lean_SourceInfo_fromRef(x_7, x_8);
x_10 = lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4;
x_11 = lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__5;
lean_inc(x_9);
x_12 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_12, 0, x_9);
lean_ctor_set(x_12, 1, x_11);
x_13 = l_Lean_Syntax_node1(x_9, x_10, x_12);
x_14 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Aesop_BuiltinRules_rfl___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_alloc_closure((void*)(lp_aesop_Aesop_BuiltinRules_rfl___lam__0___boxed), 6, 0);
x_9 = lp_aesop_Aesop_RuleTac_ofTacticSyntax(x_8, x_1, x_2, x_3, x_4, x_5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Aesop_BuiltinRules_rfl(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_BuiltinRules_Rfl(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__3 = _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__3();
lean_mark_persistent(lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__3);
lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__2 = _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__2();
lean_mark_persistent(lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__2);
lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__1 = _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__1();
lean_mark_persistent(lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__1);
lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__0 = _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__0();
lean_mark_persistent(lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__0);
lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4 = _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4();
lean_mark_persistent(lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4);
lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__5 = _init_lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__5();
lean_mark_persistent(lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__5);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
