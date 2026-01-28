// Lean compiler output
// Module: Mathlib.Data.Real.ENatENNReal
// Imports: public import Init public import Mathlib.Data.ENat.Basic public import Mathlib.Data.ENNReal.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withTopMap___at___00ENat_toENNRealOrderEmbedding_spec__1___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_castOrderEmbedding___at___00ENat_toENNRealOrderEmbedding_spec__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ENat_toENNReal(lean_object*);
static lean_object* lp_mathlib_Nat_castOrderEmbedding___at___00ENat_toENNRealOrderEmbedding_spec__0___closed__0;
lean_object* lp_mathlib_WithTop_map___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ENat_toENNRealOrderEmbedding;
LEAN_EXPORT lean_object* lp_mathlib_ENat_hasCoeENNReal;
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
static lean_object* lp_mathlib_ENat_hasCoeENNReal___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withTopMap___at___00ENat_toENNRealOrderEmbedding_spec__1(lean_object*);
extern lean_object* lp_mathlib_instSemiringNNReal;
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
static lean_object* _init_lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_instSemiringNNReal;
x_2 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__0;
x_2 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__1;
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_apply_1(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_toENNReal(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0), 1, 0);
x_3 = lp_mathlib_WithTop_map___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ENat_hasCoeENNReal___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_ENat_toENNReal), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ENat_hasCoeENNReal() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_ENat_hasCoeENNReal___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_castOrderEmbedding___at___00ENat_toENNRealOrderEmbedding_spec__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_castOrderEmbedding___at___00ENat_toENNRealOrderEmbedding_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Nat_castOrderEmbedding___at___00ENat_toENNRealOrderEmbedding_spec__0___closed__0;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withTopMap___at___00ENat_toENNRealOrderEmbedding_spec__1___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withTopMap___at___00ENat_toENNRealOrderEmbedding_spec__1(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_OrderEmbedding_withTopMap___at___00ENat_toENNRealOrderEmbedding_spec__1___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, lean_box(0));
lean_closure_set(x_3, 2, x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ENat_toENNRealOrderEmbedding() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_castOrderEmbedding___at___00ENat_toENNRealOrderEmbedding_spec__0(lean_box(0), lean_box(0), lean_box(0));
x_2 = lp_mathlib_OrderEmbedding_withTopMap___at___00ENat_toENNRealOrderEmbedding_spec__1(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ENat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ENNReal_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Real_ENatENNReal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENNReal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__0 = _init_lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__0();
lean_mark_persistent(lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__0);
lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__1 = _init_lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__1();
lean_mark_persistent(lp_mathlib_Nat_cast___at___00ENat_toENNReal_spec__0___closed__1);
lp_mathlib_ENat_hasCoeENNReal___closed__0 = _init_lp_mathlib_ENat_hasCoeENNReal___closed__0();
lean_mark_persistent(lp_mathlib_ENat_hasCoeENNReal___closed__0);
lp_mathlib_ENat_hasCoeENNReal = _init_lp_mathlib_ENat_hasCoeENNReal();
lean_mark_persistent(lp_mathlib_ENat_hasCoeENNReal);
lp_mathlib_Nat_castOrderEmbedding___at___00ENat_toENNRealOrderEmbedding_spec__0___closed__0 = _init_lp_mathlib_Nat_castOrderEmbedding___at___00ENat_toENNRealOrderEmbedding_spec__0___closed__0();
lean_mark_persistent(lp_mathlib_Nat_castOrderEmbedding___at___00ENat_toENNRealOrderEmbedding_spec__0___closed__0);
lp_mathlib_ENat_toENNRealOrderEmbedding = _init_lp_mathlib_ENat_toENNRealOrderEmbedding();
lean_mark_persistent(lp_mathlib_ENat_toENNRealOrderEmbedding);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
