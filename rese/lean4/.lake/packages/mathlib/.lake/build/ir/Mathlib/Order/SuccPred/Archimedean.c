// Lean compiler output
// Module: Mathlib.Order.SuccPred.Archimedean
// Imports: public import Init public import Mathlib.Order.SuccPred.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderDual_instPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
lean_inc(x_3);
lean_inc(x_2);
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
lean_dec(x_2);
return x_3;
}
else
{
lean_dec(x_3);
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
lean_inc(x_3);
lean_inc(x_2);
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
lean_dec(x_3);
return x_2;
}
else
{
lean_dec(x_2);
return x_3;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
lean_inc(x_4);
lean_inc(x_3);
x_5 = lean_apply_2(x_1, x_3, x_4);
x_6 = lean_unbox(x_5);
if (x_6 == 0)
{
lean_object* x_7; uint8_t x_8; 
x_7 = lean_apply_2(x_2, x_3, x_4);
x_8 = lean_unbox(x_7);
if (x_8 == 0)
{
uint8_t x_9; 
x_9 = 2;
return x_9;
}
else
{
uint8_t x_10; 
x_10 = 1;
return x_10;
}
}
else
{
uint8_t x_11; 
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_11 = 0;
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_6);
x_9 = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__0), 3, 1);
lean_closure_set(x_9, 0, x_6);
lean_inc_ref(x_6);
x_10 = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__1), 3, 1);
lean_closure_set(x_10, 0, x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_7);
x_11 = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2___boxed), 4, 2);
lean_closure_set(x_11, 0, x_7);
lean_closure_set(x_11, 1, x_5);
x_12 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_9);
lean_ctor_set(x_12, 2, x_10);
lean_ctor_set(x_12, 3, x_11);
lean_ctor_set(x_12, 4, x_6);
lean_ctor_set(x_12, 5, x_5);
lean_ctor_set(x_12, 6, x_7);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_3);
lean_inc_ref(x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__1), 3, 1);
lean_closure_set(x_6, 0, x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2___boxed), 4, 2);
lean_closure_set(x_7, 0, x_4);
lean_closure_set(x_7, 1, x_2);
x_8 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_8, 0, x_1);
lean_ctor_set(x_8, 1, x_5);
lean_ctor_set(x_8, 2, x_6);
lean_ctor_set(x_8, 3, x_7);
lean_ctor_set(x_8, 4, x_3);
lean_ctor_set(x_8, 5, x_2);
lean_ctor_set(x_8, 6, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_IsSuccArchimedean_linearOrder(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_3);
return x_9;
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_3, x_2);
x_5 = lean_unbox(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
lean_inc(x_2);
lean_inc(x_3);
x_4 = lean_apply_2(x_1, x_3, x_2);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
lean_dec(x_3);
return x_2;
}
else
{
lean_dec(x_2);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
lean_inc(x_2);
lean_inc(x_3);
x_4 = lean_apply_2(x_1, x_3, x_2);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
lean_dec(x_2);
return x_3;
}
else
{
lean_dec(x_3);
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
lean_inc(x_3);
lean_inc(x_4);
x_5 = lean_apply_2(x_1, x_4, x_3);
x_6 = lean_unbox(x_5);
if (x_6 == 0)
{
lean_object* x_7; uint8_t x_8; 
x_7 = lean_apply_2(x_2, x_3, x_4);
x_8 = lean_unbox(x_7);
if (x_8 == 0)
{
uint8_t x_9; 
x_9 = 2;
return x_9;
}
else
{
uint8_t x_10; 
x_10 = 1;
return x_10;
}
}
else
{
uint8_t x_11; 
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_11 = 0;
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__3(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
lean_inc_ref(x_6);
x_9 = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_9, 0, x_6);
lean_inc_ref(x_7);
x_10 = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_10, 0, x_7);
lean_inc_ref(x_6);
x_11 = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__2), 3, 1);
lean_closure_set(x_11, 0, x_6);
x_12 = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1), 3, 1);
lean_closure_set(x_12, 0, x_6);
lean_inc_ref(x_5);
x_13 = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__3___boxed), 4, 2);
lean_closure_set(x_13, 0, x_7);
lean_closure_set(x_13, 1, x_5);
x_14 = lp_mathlib_OrderDual_instPreorder(lean_box(0), x_2);
x_15 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_11);
lean_ctor_set(x_15, 2, x_12);
lean_ctor_set(x_15, 3, x_13);
lean_ctor_set(x_15, 4, x_9);
lean_ctor_set(x_15, 5, x_5);
lean_ctor_set(x_15, 6, x_10);
x_16 = lp_mathlib_OrderDual_instLinearOrder___redArg(x_15);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_5, 0, x_3);
lean_inc_ref(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_6, 0, x_4);
lean_inc_ref(x_3);
x_7 = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__2), 3, 1);
lean_closure_set(x_7, 0, x_3);
x_8 = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1), 3, 1);
lean_closure_set(x_8, 0, x_3);
lean_inc_ref(x_2);
x_9 = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__3___boxed), 4, 2);
lean_closure_set(x_9, 0, x_4);
lean_closure_set(x_9, 1, x_2);
x_10 = lp_mathlib_OrderDual_instPreorder(lean_box(0), x_1);
x_11 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_7);
lean_ctor_set(x_11, 2, x_8);
lean_ctor_set(x_11, 3, x_9);
lean_ctor_set(x_11, 4, x_5);
lean_ctor_set(x_11, 5, x_2);
lean_ctor_set(x_11, 6, x_6);
x_12 = lp_mathlib_OrderDual_instLinearOrder___redArg(x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_IsPredArchimedean_linearOrder(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_IsPredArchimedean_linearOrder___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
