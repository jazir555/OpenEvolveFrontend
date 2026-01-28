// Lean compiler output
// Module: Mathlib.Data.Finset.BooleanAlgebra
// Imports: public import Init public import Mathlib.Data.Finset.Basic public import Mathlib.Data.Finset.Image public import Mathlib.Data.Fintype.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_instGeneralizedBooleanAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_sub___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_boundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_boundedOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableIsCompl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableIsCompl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableIsCompl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Finset_decidableDisjoint___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableMem___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_ndunion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableIsCompl___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_boundedOrder___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_boundedOrder(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_boundedOrder___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Multiset_sub___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_1);
x_5 = lp_mathlib_Multiset_sub___redArg(x_1, x_2, x_3);
x_6 = lp_mathlib_Multiset_ndunion___redArg(x_1, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_2);
x_3 = lp_mathlib_Finset_instGeneralizedBooleanAlgebra___redArg(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 2);
lean_inc(x_6);
lean_dec_ref(x_3);
lean_inc(x_1);
lean_inc_ref(x_2);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Finset_booleanAlgebra___redArg___lam__0), 3, 2);
lean_closure_set(x_7, 0, x_2);
lean_closure_set(x_7, 1, x_1);
lean_inc(x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Finset_booleanAlgebra___redArg___lam__1), 4, 2);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_1);
x_9 = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(x_9, 0, x_4);
lean_ctor_set(x_9, 1, x_7);
lean_ctor_set(x_9, 2, x_5);
lean_ctor_set(x_9, 3, x_8);
lean_ctor_set(x_9, 4, x_1);
lean_ctor_set(x_9, 5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_booleanAlgebra___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
lean_inc(x_4);
lean_inc_ref(x_1);
x_5 = lp_mathlib_Multiset_decidableMem___redArg(x_1, x_4, x_2);
if (x_5 == 0)
{
uint8_t x_6; 
x_6 = lp_mathlib_Multiset_decidableMem___redArg(x_1, x_4, x_3);
return x_6;
}
else
{
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_5, 0, x_4);
lean_closure_set(x_5, 1, x_1);
lean_closure_set(x_5, 2, x_2);
x_6 = lp_mathlib_Fintype_decidableForallFintype___redArg(x_5, x_3);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = lp_mathlib_Finset_decidableCodisjoint___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_Finset_decidableCodisjoint(x_1, x_2, x_3, x_4, x_5);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Finset_decidableCodisjoint___redArg(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableIsCompl___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
lean_inc(x_2);
lean_inc(x_1);
lean_inc_ref(x_4);
x_5 = lp_mathlib_Finset_decidableDisjoint___redArg(x_4, x_1, x_2);
if (x_5 == 0)
{
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = lp_mathlib_Finset_decidableCodisjoint___redArg(x_1, x_2, x_3, x_4);
return x_6;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableIsCompl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = lp_mathlib_Finset_decidableIsCompl___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableIsCompl___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_Finset_decidableIsCompl(x_1, x_2, x_3, x_4, x_5);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableIsCompl___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Finset_decidableIsCompl___redArg(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
