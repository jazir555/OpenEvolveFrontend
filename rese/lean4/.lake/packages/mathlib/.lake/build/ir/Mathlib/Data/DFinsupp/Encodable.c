// Lean compiler output
// Module: Mathlib.Data.DFinsupp.Encodable
// Imports: public import Init public import Mathlib.Data.DFinsupp.Defs public import Mathlib.Logic.Encodable.Pi
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
lean_object* l_List_lengthTR___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_encodable___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Sigma_encodable___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Encodable_finPi___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_sigmaFinsetFunEquiv___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_piCongrLeft_x27___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Encodable_ofEquiv___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableMem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Encodable_decidableEqOfEncodable___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Encodable_fintypeEquivFin___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_encodable___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Encodable_decidableEqOfEncodable___boxed), 4, 2);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_1);
x_5 = lp_mathlib_Multiset_decidableMem___redArg(x_4, x_3, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_7 = lp_mathlib_Subtype_encodable___redArg(x_1, x_2);
x_8 = lp_mathlib_Encodable_fintypeEquivFin___redArg(x_3, x_7);
x_9 = lp_mathlib_Equiv_symm___redArg(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lean_apply_1(x_10, x_6);
lean_inc(x_11);
x_12 = lean_apply_1(x_4, x_11);
x_13 = lean_apply_1(x_5, x_11);
x_14 = lp_mathlib_Subtype_encodable___redArg(x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_inc(x_4);
lean_inc_ref(x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_4);
x_6 = lp_mathlib_Multiset_attach___redArg(x_4);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__2), 6, 5);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_5);
lean_closure_set(x_7, 2, x_6);
lean_closure_set(x_7, 3, x_2);
lean_closure_set(x_7, 4, x_3);
x_8 = l_List_lengthTR___redArg(x_6);
x_9 = lp_mathlib_Encodable_finPi___redArg(x_8, x_7);
x_10 = lp_mathlib_Subtype_encodable___redArg(x_1, x_5);
x_11 = lp_mathlib_Encodable_fintypeEquivFin___redArg(x_6, x_10);
x_12 = lp_mathlib_Equiv_piCongrLeft_x27___redArg(x_11);
x_13 = lp_mathlib_Encodable_ofEquiv___redArg(x_9, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_4);
lean_inc_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg___lam__1), 4, 3);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_3);
lean_closure_set(x_5, 2, x_4);
lean_inc_ref(x_2);
x_6 = lp_mathlib_Finset_encodable___redArg(x_2);
x_7 = lp_mathlib_Sigma_encodable___redArg(x_6, x_5);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Encodable_decidableEqOfEncodable___boxed), 4, 2);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, x_2);
x_9 = lp_mathlib_DFinsupp_sigmaFinsetFunEquiv___redArg(x_8, x_1, x_4);
x_10 = lp_mathlib_Encodable_ofEquiv___redArg(x_7, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Encodable_Pi(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Encodable(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Encodable_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
