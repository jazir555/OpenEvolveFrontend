// Lean compiler output
// Module: Mathlib.Data.Array.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Array_cyclicPermute_x21_cyclicPermuteAux_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Array_cyclicPermute_x21_cyclicPermuteAux_spec__0(lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__3;
static lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_permute_x21(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__2;
lean_object* lean_array_fget(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_permute_x21___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_permute_x21___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Init.Data.Array.Basic", 21, 21);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Array.swapAt!", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("index ", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" out of bounds", 14, 14);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_5; 
x_5 = lean_array_set(x_1, x_4, x_3);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_2);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 1);
x_9 = lean_array_get_size(x_1);
x_10 = lean_nat_dec_lt(x_7, x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
lean_ctor_set_tag(x_2, 0);
lean_ctor_set(x_2, 1, x_1);
lean_ctor_set(x_2, 0, x_3);
x_11 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__0;
x_12 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__1;
x_13 = lean_unsigned_to_nat(419u);
x_14 = lean_unsigned_to_nat(4u);
x_15 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__2;
x_16 = l_Nat_reprFast(x_7);
x_17 = lean_string_append(x_15, x_16);
lean_dec_ref(x_16);
x_18 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__3;
x_19 = lean_string_append(x_17, x_18);
x_20 = l_mkPanicMessageWithDecl(x_11, x_12, x_13, x_14, x_19);
lean_dec_ref(x_19);
x_21 = lean_panic_fn(x_2, x_20);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
x_23 = lean_ctor_get(x_21, 1);
lean_inc(x_23);
lean_dec(x_21);
x_1 = x_23;
x_2 = x_8;
x_3 = x_22;
goto _start;
}
else
{
lean_object* x_25; lean_object* x_26; 
lean_free_object(x_2);
x_25 = lean_array_fget(x_1, x_7);
x_26 = lean_array_fset(x_1, x_7, x_3);
lean_dec(x_7);
x_1 = x_26;
x_2 = x_8;
x_3 = x_25;
goto _start;
}
}
else
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; uint8_t x_31; 
x_28 = lean_ctor_get(x_2, 0);
x_29 = lean_ctor_get(x_2, 1);
lean_inc(x_29);
lean_inc(x_28);
lean_dec(x_2);
x_30 = lean_array_get_size(x_1);
x_31 = lean_nat_dec_lt(x_28, x_30);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; 
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_3);
lean_ctor_set(x_32, 1, x_1);
x_33 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__0;
x_34 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__1;
x_35 = lean_unsigned_to_nat(419u);
x_36 = lean_unsigned_to_nat(4u);
x_37 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__2;
x_38 = l_Nat_reprFast(x_28);
x_39 = lean_string_append(x_37, x_38);
lean_dec_ref(x_38);
x_40 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__3;
x_41 = lean_string_append(x_39, x_40);
x_42 = l_mkPanicMessageWithDecl(x_33, x_34, x_35, x_36, x_41);
lean_dec_ref(x_41);
x_43 = lean_panic_fn(x_32, x_42);
x_44 = lean_ctor_get(x_43, 0);
lean_inc(x_44);
x_45 = lean_ctor_get(x_43, 1);
lean_inc(x_45);
lean_dec(x_43);
x_1 = x_45;
x_2 = x_29;
x_3 = x_44;
goto _start;
}
else
{
lean_object* x_47; lean_object* x_48; 
x_47 = lean_array_fget(x_1, x_28);
x_48 = lean_array_fset(x_1, x_28, x_3);
lean_dec(x_28);
x_1 = x_48;
x_2 = x_29;
x_3 = x_47;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Array_cyclicPermute_x21_cyclicPermuteAux_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_panic_fn(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Array_cyclicPermute_x21_cyclicPermuteAux_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_panic_fn(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_array_get(x_1, x_2, x_4);
x_7 = lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg(x_2, x_5, x_6, x_4);
lean_dec(x_4);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_cyclicPermute_x21(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Array_cyclicPermute_x21___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_permute_x21___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Array_cyclicPermute_x21___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_permute_x21___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Array_permute_x21___redArg___lam__0), 3, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = l_List_foldl___redArg(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_permute_x21(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Array_permute_x21___redArg(x_2, x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Array_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__0 = _init_lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__0);
lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__1 = _init_lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__1();
lean_mark_persistent(lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__1);
lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__2 = _init_lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__2();
lean_mark_persistent(lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__2);
lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__3 = _init_lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__3();
lean_mark_persistent(lp_mathlib_Array_cyclicPermute_x21_cyclicPermuteAux___redArg___closed__3);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
