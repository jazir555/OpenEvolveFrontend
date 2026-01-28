// Lean compiler output
// Module: Mathlib.Control.ULift
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
static lean_object* lp_mathlib_PLift_instMonad__mathlib___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ULift_pure(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_map___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_PLift_instMonad__mathlib___closed__3;
static lean_object* lp_mathlib_ULift_instMonad__mathlib___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_ULift_pure___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_pure___redArg(lean_object*);
static lean_object* lp_mathlib_ULift_instMonad__mathlib___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_PLift_seq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_bind(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_bind___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_ULift_instMonad__mathlib___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_ULift_pure___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_bind(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib;
LEAN_EXPORT lean_object* lp_mathlib_PLift_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_seq___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMonad__mathlib;
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_pure___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_ULift_instMonad__mathlib___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ULift_instMonad__mathlib___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_PLift_pure___redArg(lean_object*);
static lean_object* lp_mathlib_ULift_instMonad__mathlib___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_ULift_seq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_map___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_ULift_instMonad__mathlib___closed__8;
static lean_object* lp_mathlib_ULift_instMonad__mathlib___closed__6;
static lean_object* lp_mathlib_ULift_instMonad__mathlib___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_ULift_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_bind___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_seq___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_PLift_instMonad__mathlib___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PLift_pure(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_PLift_instMonad__mathlib___closed__2;
static lean_object* lp_mathlib_ULift_instMonad__mathlib___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_PLift_pure___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_1(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_map___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_pure(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_pure___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_pure___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_PLift_pure(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_pure___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_PLift_pure___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_seq___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_box(0);
x_4 = lean_apply_1(x_2, x_3);
x_5 = lean_apply_1(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_seq(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_PLift_seq___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_bind(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_1(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_bind___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_box(0);
x_6 = lean_apply_1(x_4, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_PLift_instMonad__mathlib___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PLift_map), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_PLift_instMonad__mathlib___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PLift_pure___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_PLift_instMonad__mathlib___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PLift_seq), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_PLift_instMonad__mathlib___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PLift_bind), 4, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_PLift_instMonad__mathlib___lam__0(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_PLift_instMonad__mathlib___lam__1(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_instMonad__mathlib___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_PLift_instMonad__mathlib___lam__2(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_PLift_instMonad__mathlib() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PLift_instMonad__mathlib___lam__0___boxed), 4, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_PLift_instMonad__mathlib___lam__1___boxed), 4, 0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_PLift_instMonad__mathlib___lam__2___boxed), 4, 0);
x_4 = lp_mathlib_PLift_instMonad__mathlib___closed__0;
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_1);
x_6 = lp_mathlib_PLift_instMonad__mathlib___closed__1;
x_7 = lp_mathlib_PLift_instMonad__mathlib___closed__2;
x_8 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_7);
lean_ctor_set(x_8, 3, x_2);
lean_ctor_set(x_8, 4, x_3);
x_9 = lp_mathlib_PLift_instMonad__mathlib___closed__3;
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_8);
lean_ctor_set(x_10, 1, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_1(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_map___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_pure(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_pure___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_pure___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_pure(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_pure___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ULift_pure___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_seq___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_box(0);
x_4 = lean_apply_1(x_2, x_3);
x_5 = lean_apply_1(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_seq(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ULift_seq___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_bind(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_1(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_bind___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ULift_instMonad__mathlib___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PLift_instMonad__mathlib___lam__0___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ULift_instMonad__mathlib___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PLift_instMonad__mathlib___lam__1___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ULift_instMonad__mathlib___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PLift_instMonad__mathlib___lam__2___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ULift_instMonad__mathlib___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_ULift_map), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ULift_instMonad__mathlib___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_ULift_instMonad__mathlib___closed__0;
x_2 = lp_mathlib_ULift_instMonad__mathlib___closed__3;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ULift_instMonad__mathlib___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_ULift_pure___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ULift_instMonad__mathlib___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_ULift_seq), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ULift_instMonad__mathlib___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_mathlib_ULift_instMonad__mathlib___closed__2;
x_2 = lp_mathlib_ULift_instMonad__mathlib___closed__1;
x_3 = lp_mathlib_ULift_instMonad__mathlib___closed__6;
x_4 = lp_mathlib_ULift_instMonad__mathlib___closed__5;
x_5 = lp_mathlib_ULift_instMonad__mathlib___closed__4;
x_6 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_3);
lean_ctor_set(x_6, 3, x_2);
lean_ctor_set(x_6, 4, x_1);
return x_6;
}
}
static lean_object* _init_lp_mathlib_ULift_instMonad__mathlib___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_ULift_bind), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ULift_instMonad__mathlib___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_ULift_instMonad__mathlib___closed__8;
x_2 = lp_mathlib_ULift_instMonad__mathlib___closed__7;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ULift_instMonad__mathlib() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_ULift_instMonad__mathlib___closed__9;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Control_ULift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PLift_instMonad__mathlib___closed__0 = _init_lp_mathlib_PLift_instMonad__mathlib___closed__0();
lean_mark_persistent(lp_mathlib_PLift_instMonad__mathlib___closed__0);
lp_mathlib_PLift_instMonad__mathlib___closed__1 = _init_lp_mathlib_PLift_instMonad__mathlib___closed__1();
lean_mark_persistent(lp_mathlib_PLift_instMonad__mathlib___closed__1);
lp_mathlib_PLift_instMonad__mathlib___closed__2 = _init_lp_mathlib_PLift_instMonad__mathlib___closed__2();
lean_mark_persistent(lp_mathlib_PLift_instMonad__mathlib___closed__2);
lp_mathlib_PLift_instMonad__mathlib___closed__3 = _init_lp_mathlib_PLift_instMonad__mathlib___closed__3();
lean_mark_persistent(lp_mathlib_PLift_instMonad__mathlib___closed__3);
lp_mathlib_PLift_instMonad__mathlib = _init_lp_mathlib_PLift_instMonad__mathlib();
lean_mark_persistent(lp_mathlib_PLift_instMonad__mathlib);
lp_mathlib_ULift_instMonad__mathlib___closed__0 = _init_lp_mathlib_ULift_instMonad__mathlib___closed__0();
lean_mark_persistent(lp_mathlib_ULift_instMonad__mathlib___closed__0);
lp_mathlib_ULift_instMonad__mathlib___closed__1 = _init_lp_mathlib_ULift_instMonad__mathlib___closed__1();
lean_mark_persistent(lp_mathlib_ULift_instMonad__mathlib___closed__1);
lp_mathlib_ULift_instMonad__mathlib___closed__2 = _init_lp_mathlib_ULift_instMonad__mathlib___closed__2();
lean_mark_persistent(lp_mathlib_ULift_instMonad__mathlib___closed__2);
lp_mathlib_ULift_instMonad__mathlib___closed__3 = _init_lp_mathlib_ULift_instMonad__mathlib___closed__3();
lean_mark_persistent(lp_mathlib_ULift_instMonad__mathlib___closed__3);
lp_mathlib_ULift_instMonad__mathlib___closed__4 = _init_lp_mathlib_ULift_instMonad__mathlib___closed__4();
lean_mark_persistent(lp_mathlib_ULift_instMonad__mathlib___closed__4);
lp_mathlib_ULift_instMonad__mathlib___closed__5 = _init_lp_mathlib_ULift_instMonad__mathlib___closed__5();
lean_mark_persistent(lp_mathlib_ULift_instMonad__mathlib___closed__5);
lp_mathlib_ULift_instMonad__mathlib___closed__6 = _init_lp_mathlib_ULift_instMonad__mathlib___closed__6();
lean_mark_persistent(lp_mathlib_ULift_instMonad__mathlib___closed__6);
lp_mathlib_ULift_instMonad__mathlib___closed__7 = _init_lp_mathlib_ULift_instMonad__mathlib___closed__7();
lean_mark_persistent(lp_mathlib_ULift_instMonad__mathlib___closed__7);
lp_mathlib_ULift_instMonad__mathlib___closed__8 = _init_lp_mathlib_ULift_instMonad__mathlib___closed__8();
lean_mark_persistent(lp_mathlib_ULift_instMonad__mathlib___closed__8);
lp_mathlib_ULift_instMonad__mathlib___closed__9 = _init_lp_mathlib_ULift_instMonad__mathlib___closed__9();
lean_mark_persistent(lp_mathlib_ULift_instMonad__mathlib___closed__9);
lp_mathlib_ULift_instMonad__mathlib = _init_lp_mathlib_ULift_instMonad__mathlib();
lean_mark_persistent(lp_mathlib_ULift_instMonad__mathlib);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
