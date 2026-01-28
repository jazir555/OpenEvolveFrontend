// Lean compiler output
// Module: Mathlib.Data.Int.NatAbs
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Hom public import Mathlib.Algebra.GroupWithZero.Nat public import Mathlib.Algebra.Ring.Int.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Int_natAbsHom;
static lean_object* lp_mathlib_Int_natAbsHom___closed__0;
lean_object* l_Int_natAbs___boxed(lean_object*);
static lean_object* _init_lp_mathlib_Int_natAbsHom___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Int_natAbs___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Int_natAbsHom() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Int_natAbsHom___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_NatAbs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_natAbsHom___closed__0 = _init_lp_mathlib_Int_natAbsHom___closed__0();
lean_mark_persistent(lp_mathlib_Int_natAbsHom___closed__0);
lp_mathlib_Int_natAbsHom = _init_lp_mathlib_Int_natAbsHom();
lean_mark_persistent(lp_mathlib_Int_natAbsHom);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
