// Lean compiler output
// Module: Mathlib.LinearAlgebra.Matrix.RowCol
// Imports: public import Init public import Mathlib.LinearAlgebra.Matrix.ConjTranspose
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
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateCol___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateRow___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateCol(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateRow___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_update___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_updateRow___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateCol___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateRow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_updateCol___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_update___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Matrix_replicateCol___redArg___closed__0;
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_updateCol(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_updateRow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateRow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_updateCol___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateCol___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Matrix_replicateCol___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateCol___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_1(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateCol___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Matrix_replicateCol___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateCol___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lp_mathlib_Matrix_replicateCol___redArg___closed__0;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Matrix_replicateCol___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_6, 0, x_1);
x_7 = lean_apply_3(x_5, x_6, x_2, x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateCol(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_replicateCol___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateRow___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateRow___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Matrix_replicateRow___redArg___lam__0(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateRow___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lp_mathlib_Matrix_replicateCol___redArg___closed__0;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Matrix_replicateRow___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_6, 0, x_1);
x_7 = lean_apply_3(x_5, x_6, x_2, x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_replicateRow(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Matrix_replicateRow___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_updateRow___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lp_mathlib_Matrix_replicateCol___redArg___closed__0;
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Function_update___boxed), 7, 6);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, x_1);
lean_closure_set(x_9, 3, x_2);
lean_closure_set(x_9, 4, x_3);
lean_closure_set(x_9, 5, x_4);
x_10 = lean_apply_3(x_8, x_9, x_5, x_6);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_updateRow(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_updateRow___redArg(x_4, x_5, x_6, x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_updateCol___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_5);
x_7 = lean_apply_1(x_1, x_5);
x_8 = lean_apply_1(x_2, x_5);
x_9 = lp_mathlib_Function_update___redArg(x_3, x_7, x_4, x_8, x_6);
lean_dec(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_updateCol___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lp_mathlib_Matrix_replicateCol___redArg___closed__0;
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Matrix_updateCol___redArg___lam__0), 6, 4);
lean_closure_set(x_9, 0, x_2);
lean_closure_set(x_9, 1, x_4);
lean_closure_set(x_9, 2, x_1);
lean_closure_set(x_9, 3, x_3);
x_10 = lean_apply_3(x_8, x_9, x_5, x_6);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_updateCol(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Matrix_updateCol___redArg(x_4, x_5, x_6, x_7, x_8, x_9);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ConjTranspose(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_RowCol(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_ConjTranspose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Matrix_replicateCol___redArg___closed__0 = _init_lp_mathlib_Matrix_replicateCol___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Matrix_replicateCol___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
