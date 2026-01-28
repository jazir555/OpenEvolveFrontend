// Lean compiler output
// Module: Mathlib.Util.PPOptions
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
static lean_object* lp_mathlib_Mathlib_initFn___closed__7_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
static lean_object* lp_mathlib_Mathlib_getPPBinderPredicates___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_pp_mathlib_binderPredicates;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_getPPBinderPredicates___boxed(lean_object*);
lean_object* l_Lean_KVMap_find(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_initFn___closed__1_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_();
static lean_object* lp_mathlib_Mathlib_initFn___closed__2_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_initFn___closed__3_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_getPPBinderPredicates(lean_object*);
static lean_object* lp_mathlib_Mathlib_initFn___closed__4_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
static lean_object* lp_mathlib_Mathlib_initFn___closed__0_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
lean_object* lean_register_option(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_initFn___closed__5_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4____boxed(lean_object*);
uint8_t l_Lean_getPPAll(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_initFn___closed__6_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
static lean_object* _init_lp_mathlib_Mathlib_initFn___closed__0_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("pp", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_initFn___closed__1_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mathlib", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_initFn___closed__2_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("binderPredicates", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_initFn___closed__3_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Mathlib_initFn___closed__2_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
x_2 = lp_mathlib_Mathlib_initFn___closed__1_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
x_3 = lp_mathlib_Mathlib_initFn___closed__0_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
x_4 = l_Lean_Name_mkStr3(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Mathlib_initFn___closed__4_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("(pretty printer) pretty prints binders such as `∀ (x : α) (x < 2), p x` as `∀ x < 2, p x`", 94, 89);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_initFn___closed__5_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_() {
_start:
{
lean_object* x_1; uint8_t x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Mathlib_initFn___closed__4_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
x_2 = 1;
x_3 = lean_box(x_2);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Mathlib_initFn___closed__6_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Mathlib", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_initFn___closed__7_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_initFn___closed__2_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
x_2 = lp_mathlib_Mathlib_initFn___closed__1_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
x_3 = lp_mathlib_Mathlib_initFn___closed__0_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
x_4 = lp_mathlib_Mathlib_initFn___closed__6_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4__spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_alloc_ctor(1, 0, 1);
x_9 = lean_unbox(x_6);
lean_ctor_set_uint8(x_8, 0, x_9);
lean_inc(x_1);
x_10 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_10, 0, x_1);
lean_ctor_set(x_10, 1, x_3);
lean_ctor_set(x_10, 2, x_8);
lean_ctor_set(x_10, 3, x_7);
lean_inc(x_1);
x_11 = lean_register_option(x_1, x_10);
if (lean_obj_tag(x_11) == 0)
{
uint8_t x_12; 
x_12 = !lean_is_exclusive(x_11);
if (x_12 == 0)
{
lean_object* x_13; 
x_13 = lean_ctor_get(x_11, 0);
lean_dec(x_13);
lean_ctor_set(x_2, 1, x_6);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_11, 0, x_2);
return x_11;
}
else
{
lean_object* x_14; 
lean_dec(x_11);
lean_ctor_set(x_2, 1, x_6);
lean_ctor_set(x_2, 0, x_1);
x_14 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_14, 0, x_2);
return x_14;
}
}
else
{
uint8_t x_15; 
lean_free_object(x_2);
lean_dec(x_6);
lean_dec(x_1);
x_15 = !lean_is_exclusive(x_11);
if (x_15 == 0)
{
return x_11;
}
else
{
lean_object* x_16; lean_object* x_17; 
x_16 = lean_ctor_get(x_11, 0);
lean_inc(x_16);
lean_dec(x_11);
x_17 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_17, 0, x_16);
return x_17;
}
}
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; lean_object* x_22; lean_object* x_23; 
x_18 = lean_ctor_get(x_2, 0);
x_19 = lean_ctor_get(x_2, 1);
lean_inc(x_19);
lean_inc(x_18);
lean_dec(x_2);
x_20 = lean_alloc_ctor(1, 0, 1);
x_21 = lean_unbox(x_18);
lean_ctor_set_uint8(x_20, 0, x_21);
lean_inc(x_1);
x_22 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_22, 0, x_1);
lean_ctor_set(x_22, 1, x_3);
lean_ctor_set(x_22, 2, x_20);
lean_ctor_set(x_22, 3, x_19);
lean_inc(x_1);
x_23 = lean_register_option(x_1, x_22);
if (lean_obj_tag(x_23) == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; 
if (lean_is_exclusive(x_23)) {
 lean_ctor_release(x_23, 0);
 x_24 = x_23;
} else {
 lean_dec_ref(x_23);
 x_24 = lean_box(0);
}
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_1);
lean_ctor_set(x_25, 1, x_18);
if (lean_is_scalar(x_24)) {
 x_26 = lean_alloc_ctor(0, 1, 0);
} else {
 x_26 = x_24;
}
lean_ctor_set(x_26, 0, x_25);
return x_26;
}
else
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; 
lean_dec(x_18);
lean_dec(x_1);
x_27 = lean_ctor_get(x_23, 0);
lean_inc(x_27);
if (lean_is_exclusive(x_23)) {
 lean_ctor_release(x_23, 0);
 x_28 = x_23;
} else {
 lean_dec_ref(x_23);
 x_28 = lean_box(0);
}
if (lean_is_scalar(x_28)) {
 x_29 = lean_alloc_ctor(1, 1, 0);
} else {
 x_29 = x_28;
}
lean_ctor_set(x_29, 0, x_27);
return x_29;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_() {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_mathlib_Mathlib_initFn___closed__3_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
x_3 = lp_mathlib_Mathlib_initFn___closed__5_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
x_4 = lp_mathlib_Mathlib_initFn___closed__7_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_;
x_5 = lp_mathlib_Lean_Option_register___at___00Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4__spec__0(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4____boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_();
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4__spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Lean_Option_register___at___00Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4__spec__0(x_1, x_2, x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_getPPBinderPredicates___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Mathlib_pp_mathlib_binderPredicates;
return x_1;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_getPPBinderPredicates(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; uint8_t x_9; 
x_2 = lp_mathlib_Mathlib_getPPBinderPredicates___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_9 = l_Lean_getPPAll(x_1);
if (x_9 == 0)
{
uint8_t x_10; 
x_10 = 1;
x_4 = x_10;
goto block_8;
}
else
{
uint8_t x_11; 
x_11 = 0;
x_4 = x_11;
goto block_8;
}
block_8:
{
lean_object* x_5; 
x_5 = l_Lean_KVMap_find(x_1, x_3);
lean_dec(x_3);
if (lean_obj_tag(x_5) == 0)
{
return x_4;
}
else
{
lean_object* x_6; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
if (lean_obj_tag(x_6) == 1)
{
uint8_t x_7; 
x_7 = lean_ctor_get_uint8(x_6, 0);
lean_dec_ref(x_6);
return x_7;
}
else
{
lean_dec(x_6);
return x_4;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_getPPBinderPredicates___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Mathlib_getPPBinderPredicates(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_PPOptions(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_initFn___closed__0_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_ = _init_lp_mathlib_Mathlib_initFn___closed__0_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_();
lean_mark_persistent(lp_mathlib_Mathlib_initFn___closed__0_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_);
lp_mathlib_Mathlib_initFn___closed__1_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_ = _init_lp_mathlib_Mathlib_initFn___closed__1_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_();
lean_mark_persistent(lp_mathlib_Mathlib_initFn___closed__1_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_);
lp_mathlib_Mathlib_initFn___closed__2_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_ = _init_lp_mathlib_Mathlib_initFn___closed__2_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_();
lean_mark_persistent(lp_mathlib_Mathlib_initFn___closed__2_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_);
lp_mathlib_Mathlib_initFn___closed__3_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_ = _init_lp_mathlib_Mathlib_initFn___closed__3_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_();
lean_mark_persistent(lp_mathlib_Mathlib_initFn___closed__3_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_);
lp_mathlib_Mathlib_initFn___closed__4_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_ = _init_lp_mathlib_Mathlib_initFn___closed__4_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_();
lean_mark_persistent(lp_mathlib_Mathlib_initFn___closed__4_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_);
lp_mathlib_Mathlib_initFn___closed__5_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_ = _init_lp_mathlib_Mathlib_initFn___closed__5_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_();
lean_mark_persistent(lp_mathlib_Mathlib_initFn___closed__5_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_);
lp_mathlib_Mathlib_initFn___closed__6_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_ = _init_lp_mathlib_Mathlib_initFn___closed__6_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_();
lean_mark_persistent(lp_mathlib_Mathlib_initFn___closed__6_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_);
lp_mathlib_Mathlib_initFn___closed__7_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_ = _init_lp_mathlib_Mathlib_initFn___closed__7_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_();
lean_mark_persistent(lp_mathlib_Mathlib_initFn___closed__7_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_);
res = lp_mathlib_Mathlib_initFn_00___x40_Mathlib_Util_PPOptions_274037740____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_pp_mathlib_binderPredicates = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_pp_mathlib_binderPredicates);
lean_dec_ref(res);
lp_mathlib_Mathlib_getPPBinderPredicates___closed__0 = _init_lp_mathlib_Mathlib_getPPBinderPredicates___closed__0();
lean_mark_persistent(lp_mathlib_Mathlib_getPPBinderPredicates___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
