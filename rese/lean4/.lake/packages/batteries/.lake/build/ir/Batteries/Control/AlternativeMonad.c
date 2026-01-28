// Lean compiler output
// Module: Batteries.Control.AlternativeMonad
// Imports: public import Init public import Batteries.Control.Lemmas public import Batteries.Control.OptionT import all Init.Control.Option import all Init.Control.State import all Init.Control.Reader import all Init.Control.StateRef
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
LEAN_EXPORT lean_object* lp_batteries_StateRefT_x27_instAlternativeMonad(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__2;
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__3;
lean_object* l_instMonadOption___lam__0(lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_StateT_instAlternativeMonad___redArg(lean_object*);
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__12;
lean_object* l_instFunctorOption___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_AlternativeMonad_toMonad___redArg(lean_object*);
lean_object* l_instMonadOption___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__1;
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__11;
lean_object* l_Option_bind(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__6;
LEAN_EXPORT lean_object* lp_batteries_OptionT_instAlternativeMonadOfMonad___redArg(lean_object*);
lean_object* l_instMonadOption___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_AlternativeMonad_toMonad(lean_object*, lean_object*);
lean_object* l_ReaderT_instAlternativeOfMonad___redArg(lean_object*, lean_object*);
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__9;
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__7;
lean_object* l_Option_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ReaderT_instAlternativeMonad___redArg(lean_object*);
lean_object* l_StateT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Option_instAlternativeMonad;
lean_object* l_OptionT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__4;
lean_object* l_OptionT_instAlternative___redArg(lean_object*);
lean_object* l_instAlternativeOption___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_OptionT_instAlternativeMonadOfMonad(lean_object*, lean_object*);
lean_object* l_StateT_instAlternative___redArg(lean_object*, lean_object*);
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__5;
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__8;
LEAN_EXPORT lean_object* lp_batteries_StateT_instAlternativeMonad(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ReaderT_instAlternativeMonad(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__10;
static lean_object* lp_batteries_Option_instAlternativeMonad___closed__0;
LEAN_EXPORT lean_object* lp_batteries_StateRefT_x27_instAlternativeMonad___redArg(lean_object*);
lean_object* l_instAlternativeOption___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_AlternativeMonad_toMonad___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_1);
x_7 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_AlternativeMonad_toMonad(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_AlternativeMonad_toMonad___redArg(x_2);
return x_3;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instAlternativeOption___lam__0), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instAlternativeOption___lam__1___boxed), 3, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instMonadOption___lam__0), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instMonadOption___lam__1), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instMonadOption___lam__2___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instMonadOption___lam__3___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instFunctorOption___lam__0), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Option_map), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_batteries_Option_instAlternativeMonad___closed__6;
x_2 = lp_batteries_Option_instAlternativeMonad___closed__7;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_batteries_Option_instAlternativeMonad___closed__5;
x_2 = lp_batteries_Option_instAlternativeMonad___closed__4;
x_3 = lp_batteries_Option_instAlternativeMonad___closed__3;
x_4 = lp_batteries_Option_instAlternativeMonad___closed__2;
x_5 = lp_batteries_Option_instAlternativeMonad___closed__8;
x_6 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_3);
lean_ctor_set(x_6, 3, x_2);
lean_ctor_set(x_6, 4, x_1);
return x_6;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_batteries_Option_instAlternativeMonad___closed__1;
x_2 = lp_batteries_Option_instAlternativeMonad___closed__0;
x_3 = lp_batteries_Option_instAlternativeMonad___closed__9;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Option_bind), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_batteries_Option_instAlternativeMonad___closed__11;
x_2 = lp_batteries_Option_instAlternativeMonad___closed__10;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_batteries_Option_instAlternativeMonad() {
_start:
{
lean_object* x_1; 
x_1 = lp_batteries_Option_instAlternativeMonad___closed__12;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_OptionT_instAlternativeMonadOfMonad___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = l_OptionT_instAlternative___redArg(x_1);
x_3 = lean_alloc_closure((void*)(l_OptionT_bind), 6, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_OptionT_instAlternativeMonadOfMonad(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_OptionT_instAlternativeMonadOfMonad___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_StateT_instAlternativeMonad___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
lean_inc_ref(x_1);
x_2 = lp_batteries_AlternativeMonad_toMonad___redArg(x_1);
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
lean_dec(x_5);
lean_inc_ref(x_2);
x_6 = l_StateT_instAlternative___redArg(x_2, x_4);
x_7 = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, x_2);
lean_ctor_set(x_1, 1, x_7);
lean_ctor_set(x_1, 0, x_6);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_ctor_get(x_1, 0);
lean_inc(x_8);
lean_dec(x_1);
lean_inc_ref(x_2);
x_9 = l_StateT_instAlternative___redArg(x_2, x_8);
x_10 = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, x_2);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_StateT_instAlternativeMonad(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_StateT_instAlternativeMonad___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_ReaderT_instAlternativeMonad___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_batteries_AlternativeMonad_toMonad___redArg(x_1);
lean_inc_ref(x_3);
x_4 = l_ReaderT_instAlternativeOfMonad___redArg(x_2, x_3);
x_5 = l_ReaderT_instMonad___redArg(x_3);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 0);
lean_dec(x_7);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
else
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec(x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_4);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_ReaderT_instAlternativeMonad(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_ReaderT_instAlternativeMonad___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_StateRefT_x27_instAlternativeMonad___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lp_batteries_AlternativeMonad_toMonad___redArg(x_1);
lean_inc_ref(x_3);
x_4 = l_ReaderT_instAlternativeOfMonad___redArg(x_2, x_3);
x_5 = l_ReaderT_instMonad___redArg(x_3);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 0);
lean_dec(x_7);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
else
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec(x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_4);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_StateRefT_x27_instAlternativeMonad(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_StateRefT_x27_instAlternativeMonad___redArg(x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Control_Lemmas(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Control_OptionT(uint8_t builtin);
lean_object* initialize_Init_Control_Option(uint8_t builtin);
lean_object* initialize_Init_Control_State(uint8_t builtin);
lean_object* initialize_Init_Control_Reader(uint8_t builtin);
lean_object* initialize_Init_Control_StateRef(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Control_AlternativeMonad(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Control_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Control_OptionT(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init_Control_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init_Control_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init_Control_Reader(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init_Control_StateRef(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Option_instAlternativeMonad___closed__0 = _init_lp_batteries_Option_instAlternativeMonad___closed__0();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__0);
lp_batteries_Option_instAlternativeMonad___closed__1 = _init_lp_batteries_Option_instAlternativeMonad___closed__1();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__1);
lp_batteries_Option_instAlternativeMonad___closed__2 = _init_lp_batteries_Option_instAlternativeMonad___closed__2();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__2);
lp_batteries_Option_instAlternativeMonad___closed__3 = _init_lp_batteries_Option_instAlternativeMonad___closed__3();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__3);
lp_batteries_Option_instAlternativeMonad___closed__4 = _init_lp_batteries_Option_instAlternativeMonad___closed__4();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__4);
lp_batteries_Option_instAlternativeMonad___closed__5 = _init_lp_batteries_Option_instAlternativeMonad___closed__5();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__5);
lp_batteries_Option_instAlternativeMonad___closed__6 = _init_lp_batteries_Option_instAlternativeMonad___closed__6();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__6);
lp_batteries_Option_instAlternativeMonad___closed__7 = _init_lp_batteries_Option_instAlternativeMonad___closed__7();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__7);
lp_batteries_Option_instAlternativeMonad___closed__8 = _init_lp_batteries_Option_instAlternativeMonad___closed__8();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__8);
lp_batteries_Option_instAlternativeMonad___closed__9 = _init_lp_batteries_Option_instAlternativeMonad___closed__9();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__9);
lp_batteries_Option_instAlternativeMonad___closed__10 = _init_lp_batteries_Option_instAlternativeMonad___closed__10();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__10);
lp_batteries_Option_instAlternativeMonad___closed__11 = _init_lp_batteries_Option_instAlternativeMonad___closed__11();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__11);
lp_batteries_Option_instAlternativeMonad___closed__12 = _init_lp_batteries_Option_instAlternativeMonad___closed__12();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad___closed__12);
lp_batteries_Option_instAlternativeMonad = _init_lp_batteries_Option_instAlternativeMonad();
lean_mark_persistent(lp_batteries_Option_instAlternativeMonad);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
