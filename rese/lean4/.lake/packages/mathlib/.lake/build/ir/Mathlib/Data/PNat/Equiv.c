// Lean compiler output
// Module: Mathlib.Data.PNat.Equiv
// Imports: public import Init public import Mathlib.Data.PNat.Defs public import Mathlib.Logic.Equiv.Defs
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
static lean_object* lp_mathlib_Equiv_pnatEquivNat___closed__2;
static lean_object* lp_mathlib_Equiv_pnatEquivNat___closed__1;
lean_object* lp_mathlib_Nat_succPNat___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pnatEquivNat;
static lean_object* lp_mathlib_Equiv_pnatEquivNat___closed__0;
lean_object* lp_mathlib_PNat_natPred___boxed(lean_object*);
static lean_object* _init_lp_mathlib_Equiv_pnatEquivNat___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_PNat_natPred___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_pnatEquivNat___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_succPNat___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_pnatEquivNat___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Equiv_pnatEquivNat___closed__1;
x_2 = lp_mathlib_Equiv_pnatEquivNat___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Equiv_pnatEquivNat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_pnatEquivNat___closed__2;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_PNat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_PNat_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_PNat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Equiv_pnatEquivNat___closed__0 = _init_lp_mathlib_Equiv_pnatEquivNat___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_pnatEquivNat___closed__0);
lp_mathlib_Equiv_pnatEquivNat___closed__1 = _init_lp_mathlib_Equiv_pnatEquivNat___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_pnatEquivNat___closed__1);
lp_mathlib_Equiv_pnatEquivNat___closed__2 = _init_lp_mathlib_Equiv_pnatEquivNat___closed__2();
lean_mark_persistent(lp_mathlib_Equiv_pnatEquivNat___closed__2);
lp_mathlib_Equiv_pnatEquivNat = _init_lp_mathlib_Equiv_pnatEquivNat();
lean_mark_persistent(lp_mathlib_Equiv_pnatEquivNat);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
