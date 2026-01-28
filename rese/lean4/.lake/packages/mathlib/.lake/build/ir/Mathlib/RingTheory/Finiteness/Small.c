// Lean compiler output
// Module: Mathlib.RingTheory.Finiteness.Small
// Imports: public import Init public import Mathlib.LinearAlgebra.Finsupp.LinearCombination public import Mathlib.RingTheory.FiniteType public import Mathlib.LinearAlgebra.DFinsupp public import Mathlib.Algebra.Algebra.Subalgebra.Basic public import Mathlib.LinearAlgebra.Basis.Cardinality public import Mathlib.LinearAlgebra.StdBasis public import Mathlib.RingTheory.Finiteness.Basic public import Mathlib.RingTheory.MvPolynomial.Basic public import Mathlib.Data.DFinsupp.Small
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
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInhabitedSubtypeSmallMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_completeLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CompleteLattice_instOmegaCompletePartialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInhabitedSubtypeSmallMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_partialOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lp_mathlib_Submodule_completeLattice(lean_box(0), lean_box(0), x_1, x_2, x_3);
x_5 = lp_mathlib_CompleteLattice_instOmegaCompletePartialOrder___redArg(x_4);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_5, 0);
x_8 = lean_ctor_get(x_5, 1);
lean_dec(x_8);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___redArg___lam__0), 2, 0);
x_10 = lp_mathlib_Subtype_partialOrder(lean_box(0), x_7, lean_box(0));
lean_dec_ref(x_7);
lean_ctor_set(x_5, 1, x_9);
lean_ctor_set(x_5, 0, x_10);
return x_5;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_5, 0);
lean_inc(x_11);
lean_dec(x_5);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___redArg___lam__0), 2, 0);
x_13 = lp_mathlib_Subtype_partialOrder(lean_box(0), x_11, lean_box(0));
lean_dec_ref(x_11);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_12);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Submodule_instSemilatticeSupSubtypeSmallMem___redArg(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInhabitedSubtypeSmallMem(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInhabitedSubtypeSmallMem___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Submodule_instInhabitedSubtypeSmallMem(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_FiniteType(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Basis_Cardinality(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_StdBasis(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_MvPolynomial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Small(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Small(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_FiniteType(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Basis_Cardinality(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_StdBasis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_MvPolynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Small(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
