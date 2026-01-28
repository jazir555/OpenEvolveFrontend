// Lean compiler output
// Module: Mathlib.Data.Fin.Embedding
// Imports: public import Init public import Mathlib.Data.Fin.SuccPred public import Mathlib.Logic.Embedding.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddEmb(lean_object*, lean_object*);
lean_object* l_Fin_succ___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAddEmb(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveEmb(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding(lean_object*);
lean_object* lp_mathlib_finCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddEmb___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succEmb(lean_object*);
static lean_object* lp_mathlib_Fin_castAddEmb___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccEmb___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___lam__0(lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccEmb(lean_object*);
lean_object* lp_mathlib_Fin_succAbove___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb(lean_object*, lean_object*);
lean_object* l_Fin_natAdd___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_valEmbedding___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Fin_valEmbedding___lam__0___boxed), 1, 0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_valEmbedding(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succEmb(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(l_Fin_succ___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_castLEEmb___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Fin_castLEEmb___lam__0___boxed), 1, 0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Fin_castLEEmb(x_1, x_2, x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Fin_castAddEmb___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Fin_castLEEmb___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddEmb(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_castAddEmb___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddEmb___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_castAddEmb(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccEmb(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_castAddEmb___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccEmb___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Fin_castSuccEmb(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_nat_add(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_addNatEmb___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Fin_addNatEmb___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_addNatEmb___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_addNatEmb(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAddEmb(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveEmb(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Fin_succAbove___boxed), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_nat_sub(x_2, x_1);
lean_inc(x_3);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Fin_addNatEmb___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lean_nat_add(x_1, x_3);
lean_dec(x_3);
x_6 = lp_mathlib_finCongr(x_5, x_2, lean_box(0));
lean_dec(x_5);
x_7 = lp_mathlib_Equiv_toEmbedding___redArg(x_6);
x_8 = lp_mathlib_Function_Embedding_trans___redArg(x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Fin_natAdd__castLEEmb___redArg(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Fin_natAdd__castLEEmb(x_1, x_2, x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Fin_natAdd__castLEEmb___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fin_Embedding(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Fin_castAddEmb___closed__0 = _init_lp_mathlib_Fin_castAddEmb___closed__0();
lean_mark_persistent(lp_mathlib_Fin_castAddEmb___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
