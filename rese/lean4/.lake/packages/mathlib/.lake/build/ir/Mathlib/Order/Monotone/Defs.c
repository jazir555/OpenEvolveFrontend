// Lean compiler output
// Module: Mathlib.Order.Monotone.Defs
// Imports: public import Init public import Mathlib.Data.Set.Operations public import Mathlib.Logic.Function.Iterate public import Mathlib.Order.Basic public import Mathlib.Tactic.Coe public import Mathlib.Util.AssertExists
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
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictMonoOfForallForallForallLt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableMonotoneOfForallForallForallLe___redArg(uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictAntiOnOfForallForallMemSetForallForallForallLt___redArg(uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableMonotoneOnOfForallForallMemSetForallForallForallLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableAntitoneOfForallForallForallLe___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableMonotoneOnOfForallForallMemSetForallForallForallLe___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictAntiOfForallForallForallLt___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableAntitoneOnOfForallForallMemSetForallForallForallLe___redArg(uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictAntiOfForallForallForallLt___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictMonoOnOfForallForallMemSetForallForallForallLt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictMonoOfForallForallForallLt___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictMonoOnOfForallForallMemSetForallForallForallLt___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableAntitoneOfForallForallForallLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableMonotoneOnOfForallForallMemSetForallForallForallLe___redArg(uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictAntiOfForallForallForallLt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableMonotoneOnOfForallForallMemSetForallForallForallLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableAntitoneOfForallForallForallLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictAntiOnOfForallForallMemSetForallForallForallLt___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableMonotoneOfForallForallForallLe___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictMonoOfForallForallForallLt___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictMonoOnOfForallForallMemSetForallForallForallLt___redArg(uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictMonoOfForallForallForallLt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableAntitoneOnOfForallForallMemSetForallForallForallLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictAntiOnOfForallForallMemSetForallForallForallLt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableAntitoneOnOfForallForallMemSetForallForallForallLe___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableMonotoneOfForallForallForallLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictAntiOnOfForallForallMemSetForallForallForallLt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictAntiOfForallForallForallLt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableAntitoneOnOfForallForallMemSetForallForallForallLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictMonoOnOfForallForallMemSetForallForallForallLt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableMonotoneOfForallForallForallLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableAntitoneOfForallForallForallLe___redArg(uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableMonotoneOfForallForallForallLe(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6) {
_start:
{
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableMonotoneOfForallForallForallLe___redArg(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableMonotoneOfForallForallForallLe___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; uint8_t x_8; lean_object* x_9; 
x_7 = lean_unbox(x_6);
x_8 = lp_mathlib_instDecidableMonotoneOfForallForallForallLe(x_1, x_2, x_3, x_4, x_5, x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_9 = lean_box(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableMonotoneOfForallForallForallLe___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_instDecidableMonotoneOfForallForallForallLe___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableAntitoneOfForallForallForallLe(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6) {
_start:
{
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableAntitoneOfForallForallForallLe___redArg(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableAntitoneOfForallForallForallLe___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; uint8_t x_8; lean_object* x_9; 
x_7 = lean_unbox(x_6);
x_8 = lp_mathlib_instDecidableAntitoneOfForallForallForallLe(x_1, x_2, x_3, x_4, x_5, x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_9 = lean_box(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableAntitoneOfForallForallForallLe___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_instDecidableAntitoneOfForallForallForallLe___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableMonotoneOnOfForallForallMemSetForallForallForallLe(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, uint8_t x_7) {
_start:
{
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableMonotoneOnOfForallForallMemSetForallForallForallLe___redArg(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableMonotoneOnOfForallForallMemSetForallForallForallLe___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; uint8_t x_9; lean_object* x_10; 
x_8 = lean_unbox(x_7);
x_9 = lp_mathlib_instDecidableMonotoneOnOfForallForallMemSetForallForallForallLe(x_1, x_2, x_3, x_4, x_5, x_6, x_8);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_10 = lean_box(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableMonotoneOnOfForallForallMemSetForallForallForallLe___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_instDecidableMonotoneOnOfForallForallMemSetForallForallForallLe___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableAntitoneOnOfForallForallMemSetForallForallForallLe(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, uint8_t x_7) {
_start:
{
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableAntitoneOnOfForallForallMemSetForallForallForallLe___redArg(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableAntitoneOnOfForallForallMemSetForallForallForallLe___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; uint8_t x_9; lean_object* x_10; 
x_8 = lean_unbox(x_7);
x_9 = lp_mathlib_instDecidableAntitoneOnOfForallForallMemSetForallForallForallLe(x_1, x_2, x_3, x_4, x_5, x_6, x_8);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_10 = lean_box(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableAntitoneOnOfForallForallMemSetForallForallForallLe___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_instDecidableAntitoneOnOfForallForallMemSetForallForallForallLe___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictMonoOfForallForallForallLt(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6) {
_start:
{
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictMonoOfForallForallForallLt___redArg(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictMonoOfForallForallForallLt___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; uint8_t x_8; lean_object* x_9; 
x_7 = lean_unbox(x_6);
x_8 = lp_mathlib_instDecidableStrictMonoOfForallForallForallLt(x_1, x_2, x_3, x_4, x_5, x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_9 = lean_box(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictMonoOfForallForallForallLt___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_instDecidableStrictMonoOfForallForallForallLt___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictAntiOfForallForallForallLt(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6) {
_start:
{
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictAntiOfForallForallForallLt___redArg(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictAntiOfForallForallForallLt___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; uint8_t x_8; lean_object* x_9; 
x_7 = lean_unbox(x_6);
x_8 = lp_mathlib_instDecidableStrictAntiOfForallForallForallLt(x_1, x_2, x_3, x_4, x_5, x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_9 = lean_box(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictAntiOfForallForallForallLt___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_instDecidableStrictAntiOfForallForallForallLt___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictMonoOnOfForallForallMemSetForallForallForallLt(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, uint8_t x_7) {
_start:
{
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictMonoOnOfForallForallMemSetForallForallForallLt___redArg(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictMonoOnOfForallForallMemSetForallForallForallLt___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; uint8_t x_9; lean_object* x_10; 
x_8 = lean_unbox(x_7);
x_9 = lp_mathlib_instDecidableStrictMonoOnOfForallForallMemSetForallForallForallLt(x_1, x_2, x_3, x_4, x_5, x_6, x_8);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_10 = lean_box(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictMonoOnOfForallForallMemSetForallForallForallLt___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_instDecidableStrictMonoOnOfForallForallMemSetForallForallForallLt___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictAntiOnOfForallForallMemSetForallForallForallLt(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, uint8_t x_7) {
_start:
{
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableStrictAntiOnOfForallForallMemSetForallForallForallLt___redArg(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictAntiOnOfForallForallMemSetForallForallForallLt___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; uint8_t x_9; lean_object* x_10; 
x_8 = lean_unbox(x_7);
x_9 = lp_mathlib_instDecidableStrictAntiOnOfForallForallMemSetForallForallForallLt(x_1, x_2, x_3, x_4, x_5, x_6, x_8);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_10 = lean_box(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableStrictAntiOnOfForallForallMemSetForallForallForallLt___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_instDecidableStrictAntiOnOfForallForallMemSetForallForallForallLt___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AssertExists(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Monotone_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AssertExists(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
