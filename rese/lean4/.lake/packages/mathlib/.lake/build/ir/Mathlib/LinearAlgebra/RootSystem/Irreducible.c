// Lean compiler output
// Module: Mathlib.LinearAlgebra.RootSystem.Irreducible
// Imports: public import Init public import Mathlib.LinearAlgebra.RootSystem.RootPositive public import Mathlib.LinearAlgebra.RootSystem.WeylGroup public import Mathlib.RepresentationTheory.Submodule
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
static lean_object* lp_mathlib_RootPairing_invtRootSubmodule___closed__0;
lean_object* lp_mathlib_Sublattice_instInfSet___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_instBoundedOrderSubtypeSubmoduleMemSublatticeInvtRootSubmodule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_invtRootSubmodule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_RootPairing_instBoundedOrderSubtypeSubmoduleMemSublatticeInvtRootSubmodule___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_instBoundedOrderSubtypeSubmoduleMemSublatticeInvtRootSubmodule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_invtRootSubmodule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_RootPairing_invtRootSubmodule___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Sublattice_instInfSet___lam__0(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_invtRootSubmodule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_RootPairing_invtRootSubmodule___closed__0;
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_invtRootSubmodule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_RootPairing_invtRootSubmodule(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_11;
}
}
static lean_object* _init_lp_mathlib_RootPairing_instBoundedOrderSubtypeSubmoduleMemSublatticeInvtRootSubmodule___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_instBoundedOrderSubtypeSubmoduleMemSublatticeInvtRootSubmodule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_RootPairing_instBoundedOrderSubtypeSubmoduleMemSublatticeInvtRootSubmodule___closed__0;
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RootPairing_instBoundedOrderSubtypeSubmoduleMemSublatticeInvtRootSubmodule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_RootPairing_instBoundedOrderSubtypeSubmoduleMemSublatticeInvtRootSubmodule(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_RootPositive(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_WeylGroup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RepresentationTheory_Submodule(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_Irreducible(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_RootPositive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_RootSystem_WeylGroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RepresentationTheory_Submodule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_RootPairing_invtRootSubmodule___closed__0 = _init_lp_mathlib_RootPairing_invtRootSubmodule___closed__0();
lean_mark_persistent(lp_mathlib_RootPairing_invtRootSubmodule___closed__0);
lp_mathlib_RootPairing_instBoundedOrderSubtypeSubmoduleMemSublatticeInvtRootSubmodule___closed__0 = _init_lp_mathlib_RootPairing_instBoundedOrderSubtypeSubmoduleMemSublatticeInvtRootSubmodule___closed__0();
lean_mark_persistent(lp_mathlib_RootPairing_instBoundedOrderSubtypeSubmoduleMemSublatticeInvtRootSubmodule___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
