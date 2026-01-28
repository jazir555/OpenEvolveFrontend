// Lean compiler output
// Module: Mathlib.Data.Fintype.Vector
// Imports: public import Init public import Mathlib.Data.Fintype.Pi public import Mathlib.Data.Sym.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSym_x27OfDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Vector_fintype___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSym_x27OfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Vector_fintype___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Vector_fintype___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSym_x27OfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instFintype___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_vectorEquivFin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSymOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Quotient_fintype___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_List_decidablePerm___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Vector_fintype(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instDecidableEqFin___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSymOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instFintypeSym_x27OfDecidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Vector_fintype___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Vector_fintype___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Vector_fintype___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Vector_fintype___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Vector_fintype___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_3, 0, x_1);
lean_inc(x_2);
x_4 = lean_alloc_closure((void*)(l_instDecidableEqFin___boxed), 3, 1);
lean_closure_set(x_4, 0, x_2);
lean_inc(x_2);
x_5 = l_List_finRange(x_2);
x_6 = lp_mathlib_Pi_instFintype___redArg(x_4, x_5, x_3);
x_7 = lp_mathlib_Equiv_vectorEquivFin___redArg(x_2);
x_8 = lp_mathlib_Equiv_symm___redArg(x_7);
x_9 = lp_mathlib_Fintype_ofEquiv___redArg(x_6, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Vector_fintype(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Vector_fintype___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instFintypeSym_x27OfDecidableEq___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = l_List_decidablePerm___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSym_x27OfDecidableEq___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_instFintypeSym_x27OfDecidableEq___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSym_x27OfDecidableEq___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_instFintypeSym_x27OfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lp_mathlib_Vector_fintype___redArg(x_2, x_3);
x_6 = lean_box(0);
x_7 = lp_mathlib_Quotient_fintype___redArg(x_5, x_6, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSym_x27OfDecidableEq(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instFintypeSym_x27OfDecidableEq___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSymOfDecidableEq___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_3);
x_4 = lp_mathlib_instFintypeSym_x27OfDecidableEq___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0(lean_box(0), x_3, lean_box(0), lean_box(0), lean_box(0), lean_box(0));
lean_dec(x_3);
x_6 = lp_mathlib_Equiv_symm___redArg(x_5);
x_7 = lp_mathlib_Fintype_ofEquiv___redArg(x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSymOfDecidableEq(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instFintypeSymOfDecidableEq___redArg(x_2, x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sym_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Vector(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sym_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
