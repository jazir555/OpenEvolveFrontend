// Lean compiler output
// Module: Mathlib.Data.Set.Insert
// Imports: public import Init public import Mathlib.Data.Set.Disjoint
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
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
lean_inc(x_3);
x_4 = lean_apply_2(x_1, x_3, x_2);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
lean_object* x_6; 
x_6 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_6, 0, x_3);
return x_6;
}
else
{
lean_object* x_7; 
lean_dec(x_3);
x_7 = lean_box(0);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_inc(x_1);
return x_1;
}
else
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_4, 0, x_2);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Set_subtypeInsertEquivOption___redArg(x_2, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Set_uniqueSingleton(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Set_uniqueSingleton___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4) {
_start:
{
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton___redArg(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; uint8_t x_6; lean_object* x_7; 
x_5 = lean_unbox(x_4);
x_6 = lp_mathlib_Set_decidableSingleton(x_1, x_2, x_3, x_5);
lean_dec(x_3);
lean_dec(x_2);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_Set_decidableSingleton___redArg(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Disjoint(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Insert(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
