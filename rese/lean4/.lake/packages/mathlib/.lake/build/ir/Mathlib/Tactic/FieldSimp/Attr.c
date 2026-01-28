// Lean compiler output
// Module: Mathlib.Tactic.FieldSimp.Attr
// Imports: public import Init public import Mathlib.Init
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
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_();
static lean_object* lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2____boxed(lean_object*);
static lean_object* lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_;
static lean_object* lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib_fieldSimpExt;
lean_object* l_Lean_Meta_Simp_registerSimprocAttr(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_;
lean_object* l_Lean_Name_mkStr1(lean_object*);
static lean_object* lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_;
static lean_object* _init_lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("field", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Attribute grouping the simprocs associated to the field_simp tactic", 67, 67);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("fieldSimpExt", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_() {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_;
x_3 = lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_;
x_4 = lean_box(0);
x_5 = lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_;
x_6 = l_Lean_Meta_Simp_registerSimprocAttr(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2____boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_initFn_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_();
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FieldSimp_Attr(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_);
res = lp_mathlib_initFn_00___x40_Mathlib_Tactic_FieldSimp_Attr_4203525765____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_fieldSimpExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_fieldSimpExt);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
