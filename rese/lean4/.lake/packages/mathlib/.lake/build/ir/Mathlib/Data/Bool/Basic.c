// Lean compiler output
// Module: Mathlib.Data.Bool.Basic
// Imports: public import Init public import Mathlib.Logic.Basic public import Mathlib.Order.Defs.LinearOrder
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
static lean_object* lp_mathlib_Bool_linearOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Bool_linearOrder;
LEAN_EXPORT lean_object* lp_mathlib_Bool_ofNat___boxed(lean_object*);
lean_object* l_Bool_instMin___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Bool_instMax___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Bool_instDecidableLe___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Bool_linearOrder___closed__5;
static lean_object* lp_mathlib_Bool_linearOrder___closed__2;
LEAN_EXPORT uint8_t lp_mathlib_Bool_carry(uint8_t, uint8_t, uint8_t);
lean_object* l_instOrdBool___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_instDecidableEqBool___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bool_carry___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Bool_linearOrder___closed__6;
lean_object* l_Bool_instDecidableLt___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Bool_linearOrder___closed__3;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
static lean_object* lp_mathlib_Bool_linearOrder___closed__4;
static lean_object* lp_mathlib_Bool_linearOrder___closed__1;
LEAN_EXPORT uint8_t lp_mathlib_Bool_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bool_xor3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Bool_xor3(uint8_t, uint8_t, uint8_t);
static lean_object* _init_lp_mathlib_Bool_linearOrder___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Bool_instMin___lam__0___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Bool_linearOrder___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Bool_instMax___lam__0___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Bool_linearOrder___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instOrdBool___lam__0___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Bool_linearOrder___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Bool_linearOrder___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Bool_instDecidableLe___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Bool_linearOrder___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Bool_instDecidableLt___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Bool_linearOrder___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_1 = lp_mathlib_Bool_linearOrder___closed__5;
x_2 = lean_alloc_closure((void*)(l_instDecidableEqBool___boxed), 2, 0);
x_3 = lp_mathlib_Bool_linearOrder___closed__4;
x_4 = lp_mathlib_Bool_linearOrder___closed__2;
x_5 = lp_mathlib_Bool_linearOrder___closed__1;
x_6 = lp_mathlib_Bool_linearOrder___closed__0;
x_7 = lp_mathlib_Bool_linearOrder___closed__3;
x_8 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_5);
lean_ctor_set(x_8, 3, x_4);
lean_ctor_set(x_8, 4, x_3);
lean_ctor_set(x_8, 5, x_2);
lean_ctor_set(x_8, 6, x_1);
return x_8;
}
}
static lean_object* _init_lp_mathlib_Bool_linearOrder() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Bool_linearOrder___closed__6;
return x_1;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Bool_ofNat(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_nat_dec_eq(x_1, x_2);
if (x_3 == 0)
{
uint8_t x_4; 
x_4 = 1;
return x_4;
}
else
{
uint8_t x_5; 
x_5 = 0;
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_ofNat___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Bool_ofNat(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Bool_xor3(uint8_t x_1, uint8_t x_2, uint8_t x_3) {
_start:
{
if (x_1 == 0)
{
if (x_2 == 0)
{
return x_3;
}
else
{
goto block_6;
}
}
else
{
if (x_2 == 0)
{
goto block_6;
}
else
{
return x_3;
}
}
block_6:
{
if (x_3 == 0)
{
uint8_t x_4; 
x_4 = 1;
return x_4;
}
else
{
uint8_t x_5; 
x_5 = 0;
return x_5;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_xor3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; uint8_t x_5; uint8_t x_6; uint8_t x_7; lean_object* x_8; 
x_4 = lean_unbox(x_1);
x_5 = lean_unbox(x_2);
x_6 = lean_unbox(x_3);
x_7 = lp_mathlib_Bool_xor3(x_4, x_5, x_6);
x_8 = lean_box(x_7);
return x_8;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Bool_carry(uint8_t x_1, uint8_t x_2, uint8_t x_3) {
_start:
{
if (x_1 == 0)
{
goto block_4;
}
else
{
if (x_2 == 0)
{
goto block_4;
}
else
{
return x_2;
}
}
block_4:
{
if (x_1 == 0)
{
if (x_2 == 0)
{
return x_2;
}
else
{
return x_3;
}
}
else
{
if (x_3 == 0)
{
if (x_2 == 0)
{
return x_2;
}
else
{
return x_3;
}
}
else
{
return x_3;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_carry___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; uint8_t x_5; uint8_t x_6; uint8_t x_7; lean_object* x_8; 
x_4 = lean_unbox(x_1);
x_5 = lean_unbox(x_2);
x_6 = lean_unbox(x_3);
x_7 = lp_mathlib_Bool_carry(x_4, x_5, x_6);
x_8 = lean_box(x_7);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Defs_LinearOrder(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Bool_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Defs_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Bool_linearOrder___closed__0 = _init_lp_mathlib_Bool_linearOrder___closed__0();
lean_mark_persistent(lp_mathlib_Bool_linearOrder___closed__0);
lp_mathlib_Bool_linearOrder___closed__1 = _init_lp_mathlib_Bool_linearOrder___closed__1();
lean_mark_persistent(lp_mathlib_Bool_linearOrder___closed__1);
lp_mathlib_Bool_linearOrder___closed__2 = _init_lp_mathlib_Bool_linearOrder___closed__2();
lean_mark_persistent(lp_mathlib_Bool_linearOrder___closed__2);
lp_mathlib_Bool_linearOrder___closed__3 = _init_lp_mathlib_Bool_linearOrder___closed__3();
lean_mark_persistent(lp_mathlib_Bool_linearOrder___closed__3);
lp_mathlib_Bool_linearOrder___closed__4 = _init_lp_mathlib_Bool_linearOrder___closed__4();
lean_mark_persistent(lp_mathlib_Bool_linearOrder___closed__4);
lp_mathlib_Bool_linearOrder___closed__5 = _init_lp_mathlib_Bool_linearOrder___closed__5();
lean_mark_persistent(lp_mathlib_Bool_linearOrder___closed__5);
lp_mathlib_Bool_linearOrder___closed__6 = _init_lp_mathlib_Bool_linearOrder___closed__6();
lean_mark_persistent(lp_mathlib_Bool_linearOrder___closed__6);
lp_mathlib_Bool_linearOrder = _init_lp_mathlib_Bool_linearOrder();
lean_mark_persistent(lp_mathlib_Bool_linearOrder);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
