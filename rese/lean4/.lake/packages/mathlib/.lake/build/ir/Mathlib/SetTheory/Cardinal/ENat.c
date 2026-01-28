// Lean compiler output
// Module: Mathlib.SetTheory.Cardinal.ENat
// Imports: public import Init public import Mathlib.Algebra.Order.Hom.Ring public import Mathlib.Data.ENat.Basic public import Mathlib.SetTheory.Cardinal.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_ofENatHom;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_ofENat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instCoeENat;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Cardinal_ofENat_spec__0(lean_object*);
static lean_object* lp_mathlib_Cardinal_instCoeENat___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Cardinal_ofENat_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_ofENat___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Cardinal_ofENat_spec__0(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_ofENat(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
return lean_box(0);
}
else
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lp_mathlib_Nat_cast___at___00Cardinal_ofENat_spec__0(x_2);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Cardinal_ofENat_spec__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nat_cast___at___00Cardinal_ofENat_spec__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_ofENat___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Cardinal_ofENat(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instCoeENat___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Cardinal_ofENat___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instCoeENat() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Cardinal_instCoeENat___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Cardinal_ofENatHom() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Cardinal_instCoeENat___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ENat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_ENat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Cardinal_instCoeENat___closed__0 = _init_lp_mathlib_Cardinal_instCoeENat___closed__0();
lean_mark_persistent(lp_mathlib_Cardinal_instCoeENat___closed__0);
lp_mathlib_Cardinal_instCoeENat = _init_lp_mathlib_Cardinal_instCoeENat();
lean_mark_persistent(lp_mathlib_Cardinal_instCoeENat);
lp_mathlib_Cardinal_ofENatHom = _init_lp_mathlib_Cardinal_ofENatHom();
lean_mark_persistent(lp_mathlib_Cardinal_ofENatHom);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
