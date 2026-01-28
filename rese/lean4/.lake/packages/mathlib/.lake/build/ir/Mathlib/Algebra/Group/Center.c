// Lean compiler output
// Module: Mathlib.Algebra.Group.Center
// Imports: public import Init public import Mathlib.Algebra.Group.Invertible.Basic public import Mathlib.Algebra.Notation.Prod public import Mathlib.Data.Set.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAddCenter___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAddCentralizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAddCenter___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemCentralizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAddCentralizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemCenter___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAddCenter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemCenter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAddCentralizer___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAddCenter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemCenter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemCentralizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemCentralizer___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemCenter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemCentralizer___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAddCentralizer___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemCentralizer(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
x_6 = lean_apply_1(x_4, x_5);
x_7 = lean_unbox(x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemCentralizer___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_unbox(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemCentralizer___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_Set_decidableMemCentralizer(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_3);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemCentralizer___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_Set_decidableMemCentralizer___redArg(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAddCentralizer(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
x_6 = lean_apply_1(x_4, x_5);
x_7 = lean_unbox(x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAddCentralizer___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_unbox(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAddCentralizer___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_Set_decidableMemAddCentralizer(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_3);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAddCentralizer___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_Set_decidableMemAddCentralizer___redArg(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemCenter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_apply_1(x_3, x_4);
x_6 = lean_unbox(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemCenter___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_unbox(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemCenter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Set_decidableMemCenter(x_1, x_2, x_3, x_4);
lean_dec(x_2);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemCenter___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_Set_decidableMemCenter___redArg(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAddCenter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_apply_1(x_3, x_4);
x_6 = lean_unbox(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAddCenter___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_unbox(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAddCenter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Set_decidableMemAddCenter(x_1, x_2, x_3, x_4);
lean_dec(x_2);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAddCenter___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_Set_decidableMemAddCenter___redArg(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Center(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
