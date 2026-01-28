// Lean compiler output
// Module: Mathlib.Order.OrderIsoNat
// Imports: public import Init public import Mathlib.Data.Nat.Lattice public import Mathlib.Logic.Denumerable public import Mathlib.Logic.Function.Iterate public import Mathlib.Order.Hom.Basic public import Mathlib.Data.Set.Subsingleton
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
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Nat_orderEmbeddingOfSet___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___redArg(lean_object*);
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Nat_Subtype_ofNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___redArg___boxed(lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_orderEmbeddingOfSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_orderEmbeddingOfSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RelEmbedding_natLT(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RelEmbedding_natLT___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RelEmbedding_natGT(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_RelEmbedding_natGT___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_orderEmbeddingOfSet___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_subtype___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_orderEmbeddingOfSet___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Nat_Subtype_ofNat___boxed), 4, 3);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
lean_closure_set(x_2, 2, lean_box(0));
x_3 = lp_mathlib_Nat_orderEmbeddingOfSet___redArg___closed__0;
x_4 = lp_mathlib_Function_Embedding_trans___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_orderEmbeddingOfSet(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Nat_orderEmbeddingOfSet___redArg(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Denumerable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Subsingleton(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_OrderIsoNat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Denumerable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_orderEmbeddingOfSet___redArg___closed__0 = _init_lp_mathlib_Nat_orderEmbeddingOfSet___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Nat_orderEmbeddingOfSet___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
