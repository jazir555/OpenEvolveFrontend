// Lean compiler output
// Module: Mathlib.Tactic.Measurability.Init
// Imports: public import Init public import Mathlib.Init public meta import Aesop
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
static lean_object* lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
uint8_t lean_usize_dec_eq(size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_();
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
static uint8_t lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3__spec__0(lean_object*, size_t, size_t, lean_object*);
static lean_object* lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3____boxed(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
static uint8_t lp_mathlib_initFn___closed__5_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_aesop_Aesop_Frontend_declareRuleSetUnchecked(lean_object*, uint8_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
static size_t lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
static lean_object* lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
static lean_object* lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3__spec__0(lean_object* x_1, size_t x_2, size_t x_3, lean_object* x_4) {
_start:
{
uint8_t x_6; 
x_6 = lean_usize_dec_eq(x_2, x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_array_uget(x_1, x_2);
x_8 = lp_aesop_Aesop_Frontend_declareRuleSetUnchecked(x_7, x_6);
if (lean_obj_tag(x_8) == 0)
{
lean_object* x_9; size_t x_10; size_t x_11; 
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = 1;
x_11 = lean_usize_add(x_2, x_10);
x_2 = x_11;
x_4 = x_9;
goto _start;
}
else
{
return x_8;
}
}
else
{
lean_object* x_13; 
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_4);
return x_13;
}
}
}
static lean_object* _init_lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Measurable", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
x_2 = l_Array_mkArray1___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
x_2 = lean_array_get_size(x_1);
return x_2;
}
}
static uint8_t _init_lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_() {
_start:
{
lean_object* x_1; lean_object* x_2; uint8_t x_3; 
x_1 = lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_nat_dec_lt(x_2, x_1);
return x_3;
}
}
static uint8_t _init_lp_mathlib_initFn___closed__5_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_() {
_start:
{
lean_object* x_1; uint8_t x_2; 
x_1 = lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
x_2 = lean_nat_dec_le(x_1, x_1);
return x_2;
}
}
static size_t _init_lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_() {
_start:
{
lean_object* x_1; size_t x_2; 
x_1 = lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
x_2 = lean_usize_of_nat(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_() {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
x_3 = lean_box(0);
x_4 = lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
if (x_4 == 0)
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_5, 0, x_3);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = lp_mathlib_initFn___closed__5_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_7, 0, x_3);
return x_7;
}
else
{
size_t x_8; size_t x_9; lean_object* x_10; 
x_8 = 0;
x_9 = lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_;
x_10 = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3__spec__0(x_2, x_8, x_9, x_3);
return x_10;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3__spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
size_t x_6; size_t x_7; lean_object* x_8; 
x_6 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_7 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_8 = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3__spec__0(x_1, x_6, x_7, x_4);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3____boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_();
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Measurability_Init(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_ = _init_lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_();
lean_mark_persistent(lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_);
lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_ = _init_lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_();
lean_mark_persistent(lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_);
lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_ = _init_lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_();
lean_mark_persistent(lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_);
lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_ = _init_lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_();
lean_mark_persistent(lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_);
lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_ = _init_lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_();
lp_mathlib_initFn___closed__5_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_ = _init_lp_mathlib_initFn___closed__5_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_();
lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_ = _init_lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_();
res = lp_mathlib_initFn_00___x40_Mathlib_Tactic_Measurability_Init_3256345150____hygCtx___hyg_3_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
