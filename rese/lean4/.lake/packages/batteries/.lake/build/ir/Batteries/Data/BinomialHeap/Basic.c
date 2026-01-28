// Lean compiler output
// Module: Batteries.Data.BinomialHeap.Basic
// Imports: public import Init public import Batteries.Classes.Order public import Batteries.Control.ForInStep.Basic
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
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_foldM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_nil_elim___redArg(lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__6;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_nil_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_realSize_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__4;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_foldM_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length(lean_object*, lean_object*);
lean_object* lp_batteries_ForInStep_run___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_push___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_deleteMin___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_empty___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_singleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_head_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_empty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instStream___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_singleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_foldM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_insert___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_singleton(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__4;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_mkBinomialHeap(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_isEmpty___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___redArg(lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__6;
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__4;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___redArg___boxed(lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_forIn___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArray___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_panic___redArg(lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_cons_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_toHeap_go_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_findMin_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_combine(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___redArg(lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__8;
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___boxed(lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_fold___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_singleton(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_realSize_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTree___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___boxed(lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__1;
static lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___closed__2;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_deleteMin(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_isEmpty(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_mkBinomialHeap___boxed(lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_isEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_nil_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_cons_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instStream(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_head_x3f(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__7;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_foldM_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg(lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap(lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap(lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_rankTR_go_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg___boxed(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instEmptyCollection(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_toHeap_go_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___redArg___boxed(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_shiftl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instEmptyCollection___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_merge(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_node_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toList(lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__2;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_merge___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_insert(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__2;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_nil_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_realSize_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0___boxed(lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___redArg(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_node_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_findMin_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_isEmpty___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_combine___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toList___redArg(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_realSize_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_rankTR_go_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
else
{
lean_object* x_3; 
x_3 = lean_unsigned_to_nat(1u);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_3(x_2, x_3, x_4, x_5);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_nil_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_nil_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_node_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_node_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Batteries.BinomialHeap.Imp.HeapNode.nil", 39, 39);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__0;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Batteries.BinomialHeap.Imp.HeapNode.node", 40, 40);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__4;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(1);
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__5;
x_3 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_11; uint8_t x_12; 
lean_dec_ref(x_1);
x_11 = lean_unsigned_to_nat(1024u);
x_12 = lean_nat_dec_le(x_11, x_3);
if (x_12 == 0)
{
lean_object* x_13; 
x_13 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2;
x_4 = x_13;
goto block_10;
}
else
{
lean_object* x_14; 
x_14 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3;
x_4 = x_14;
goto block_10;
}
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; uint8_t x_35; 
x_15 = lean_ctor_get(x_2, 0);
lean_inc(x_15);
x_16 = lean_ctor_get(x_2, 1);
lean_inc(x_16);
x_17 = lean_ctor_get(x_2, 2);
lean_inc(x_17);
lean_dec_ref(x_2);
x_18 = lean_unsigned_to_nat(1024u);
x_35 = lean_nat_dec_le(x_18, x_3);
if (x_35 == 0)
{
lean_object* x_36; 
x_36 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2;
x_19 = x_36;
goto block_34;
}
else
{
lean_object* x_37; 
x_37 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3;
x_19 = x_37;
goto block_34;
}
block_34:
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; uint8_t x_31; lean_object* x_32; lean_object* x_33; 
x_20 = lean_box(1);
x_21 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__6;
lean_inc_ref(x_1);
x_22 = lean_apply_2(x_1, x_15, x_18);
x_23 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_22);
x_24 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_20);
lean_inc_ref(x_1);
x_25 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(x_1, x_16, x_18);
x_26 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_26, 0, x_24);
lean_ctor_set(x_26, 1, x_25);
x_27 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_20);
x_28 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(x_1, x_17, x_18);
x_29 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_29, 0, x_27);
lean_ctor_set(x_29, 1, x_28);
x_30 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_30, 0, x_19);
lean_ctor_set(x_30, 1, x_29);
x_31 = 0;
x_32 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_32, 0, x_30);
lean_ctor_set_uint8(x_32, sizeof(void*)*1, x_31);
x_33 = l_Repr_addAppParen(x_32, x_3);
return x_33;
}
}
block_10:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__1;
x_6 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
x_7 = 0;
x_8 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set_uint8(x_8, sizeof(void*)*1, x_7);
x_9 = l_Repr_addAppParen(x_8, x_3);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___boxed), 4, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___boxed), 4, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_1, 2);
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(x_3);
x_6 = lean_unsigned_to_nat(1u);
x_7 = lean_nat_add(x_5, x_6);
lean_dec(x_5);
x_8 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(x_4);
x_9 = lean_nat_add(x_7, x_8);
lean_dec(x_8);
lean_dec(x_7);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_realSize_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_3);
x_4 = lean_box(0);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_dec(x_2);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_1, 2);
lean_inc(x_8);
lean_dec_ref(x_1);
x_9 = lean_apply_3(x_3, x_6, x_7, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_realSize_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_realSize_match__1_splitter___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_singleton___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
lean_ctor_set(x_3, 2, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_singleton(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_singleton___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 2);
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(x_3);
x_5 = lean_unsigned_to_nat(1u);
x_6 = lean_nat_add(x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 2);
x_4 = lean_unsigned_to_nat(1u);
x_5 = lean_nat_add(x_2, x_4);
lean_dec(x_2);
x_1 = x_3;
x_2 = x_5;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_rankTR_go_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_5; 
lean_dec(x_4);
x_5 = lean_apply_1(x_3, x_2);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_dec(x_3);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_1, 2);
lean_inc(x_8);
lean_dec_ref(x_1);
x_9 = lean_apply_4(x_4, x_6, x_7, x_8, x_2);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_rankTR_go_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_rankTR_go_match__1_splitter___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
else
{
lean_object* x_3; 
x_3 = lean_unsigned_to_nat(1u);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
x_6 = lean_ctor_get(x_1, 3);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_4(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_nil_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_nil_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_cons_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_cons_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Batteries.BinomialHeap.Imp.Heap.nil", 35, 35);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__0;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Batteries.BinomialHeap.Imp.Heap.cons", 36, 36);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__2;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(1);
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__3;
x_3 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_11; uint8_t x_12; 
lean_dec_ref(x_1);
x_11 = lean_unsigned_to_nat(1024u);
x_12 = lean_nat_dec_le(x_11, x_3);
if (x_12 == 0)
{
lean_object* x_13; 
x_13 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2;
x_4 = x_13;
goto block_10;
}
else
{
lean_object* x_14; 
x_14 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3;
x_4 = x_14;
goto block_10;
}
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_40; 
x_15 = lean_ctor_get(x_2, 0);
lean_inc(x_15);
x_16 = lean_ctor_get(x_2, 1);
lean_inc(x_16);
x_17 = lean_ctor_get(x_2, 2);
lean_inc(x_17);
x_18 = lean_ctor_get(x_2, 3);
lean_inc(x_18);
lean_dec_ref(x_2);
x_19 = lean_unsigned_to_nat(1024u);
x_40 = lean_nat_dec_le(x_19, x_3);
if (x_40 == 0)
{
lean_object* x_41; 
x_41 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2;
x_20 = x_41;
goto block_39;
}
else
{
lean_object* x_42; 
x_42 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3;
x_20 = x_42;
goto block_39;
}
block_39:
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; uint8_t x_36; lean_object* x_37; lean_object* x_38; 
x_21 = lean_box(1);
x_22 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__4;
x_23 = l_Nat_reprFast(x_15);
x_24 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_24, 0, x_23);
x_25 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_25, 0, x_22);
lean_ctor_set(x_25, 1, x_24);
x_26 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_21);
lean_inc_ref(x_1);
x_27 = lean_apply_2(x_1, x_16, x_19);
x_28 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_28, 0, x_26);
lean_ctor_set(x_28, 1, x_27);
x_29 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_21);
lean_inc_ref(x_1);
x_30 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(x_1, x_17, x_19);
x_31 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_31, 0, x_29);
lean_ctor_set(x_31, 1, x_30);
x_32 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_21);
x_33 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg(x_1, x_18, x_19);
x_34 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_34, 0, x_32);
lean_ctor_set(x_34, 1, x_33);
x_35 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_35, 0, x_20);
lean_ctor_set(x_35, 1, x_34);
x_36 = 0;
x_37 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_37, 0, x_35);
lean_ctor_set_uint8(x_37, sizeof(void*)*1, x_36);
x_38 = l_Repr_addAppParen(x_37, x_3);
return x_38;
}
}
block_10:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__1;
x_6 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
x_7 = 0;
x_8 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set_uint8(x_8, sizeof(void*)*1, x_7);
x_9 = l_Repr_addAppParen(x_8, x_3);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___boxed), 4, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___boxed), 4, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lean_ctor_get(x_1, 2);
x_4 = lean_ctor_get(x_1, 3);
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(x_3);
x_6 = lean_unsigned_to_nat(1u);
x_7 = lean_nat_add(x_5, x_6);
lean_dec(x_5);
x_8 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg(x_4);
x_9 = lean_nat_add(x_7, x_8);
lean_dec(x_8);
lean_dec(x_7);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_realSize_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_3);
x_4 = lean_box(0);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_dec(x_2);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_1, 2);
lean_inc(x_8);
x_9 = lean_ctor_get(x_1, 3);
lean_inc(x_9);
lean_dec_ref(x_1);
x_10 = lean_apply_4(x_3, x_6, x_7, x_8, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_realSize_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_realSize_match__1_splitter___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 3);
x_5 = lean_unsigned_to_nat(1u);
x_6 = lean_nat_shiftl(x_5, x_3);
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(x_4);
x_8 = lean_nat_add(x_6, x_7);
lean_dec(x_7);
lean_dec(x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
uint8_t x_3; 
x_3 = 1;
return x_3;
}
else
{
uint8_t x_4; 
x_4 = 0;
return x_4;
}
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty(x_1, x_2);
lean_dec(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___redArg(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_singleton(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_box(0);
x_5 = lean_box(0);
x_6 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_6, 0, x_3);
lean_ctor_set(x_6, 1, x_2);
lean_ctor_set(x_6, 2, x_4);
lean_ctor_set(x_6, 3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_singleton___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_box(0);
x_4 = lean_box(0);
x_5 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_1);
lean_ctor_set(x_5, 2, x_3);
lean_ctor_set(x_5, 3, x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_3; 
x_3 = 1;
return x_3;
}
else
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_nat_dec_lt(x_2, x_4);
return x_5;
}
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 3);
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg(x_3);
x_5 = lean_unsigned_to_nat(1u);
x_6 = lean_nat_add(x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_length(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_combine(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; uint8_t x_8; 
lean_inc(x_4);
lean_inc(x_3);
x_7 = lean_apply_2(x_2, x_3, x_4);
x_8 = lean_unbox(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_9, 0, x_3);
lean_ctor_set(x_9, 1, x_5);
lean_ctor_set(x_9, 2, x_6);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_4);
lean_ctor_set(x_10, 1, x_9);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_11, 0, x_4);
lean_ctor_set(x_11, 1, x_6);
lean_ctor_set(x_11, 2, x_5);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_3);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_combine___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
lean_inc(x_3);
lean_inc(x_2);
x_6 = lean_apply_2(x_1, x_2, x_3);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_8, 0, x_2);
lean_ctor_set(x_8, 1, x_4);
lean_ctor_set(x_8, 2, x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_3);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_10, 0, x_3);
lean_ctor_set(x_10, 1, x_5);
lean_ctor_set(x_10, 2, x_4);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_2);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_dec_ref(x_1);
return x_3;
}
else
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec_ref(x_1);
return x_2;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_28; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_2, 2);
x_7 = lean_ctor_get(x_2, 3);
x_8 = lean_ctor_get(x_3, 0);
x_9 = lean_ctor_get(x_3, 1);
x_10 = lean_ctor_get(x_3, 2);
x_11 = lean_ctor_get(x_3, 3);
x_28 = lean_nat_dec_lt(x_4, x_8);
if (x_28 == 0)
{
uint8_t x_29; 
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_8);
x_29 = !lean_is_exclusive(x_3);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; uint8_t x_34; 
x_30 = lean_ctor_get(x_3, 3);
lean_dec(x_30);
x_31 = lean_ctor_get(x_3, 2);
lean_dec(x_31);
x_32 = lean_ctor_get(x_3, 1);
lean_dec(x_32);
x_33 = lean_ctor_get(x_3, 0);
lean_dec(x_33);
x_34 = lean_nat_dec_lt(x_8, x_4);
if (x_34 == 0)
{
lean_object* x_35; uint8_t x_36; 
lean_free_object(x_3);
lean_dec(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
lean_inc(x_9);
lean_inc(x_5);
x_35 = lean_apply_2(x_1, x_5, x_9);
x_36 = lean_unbox(x_35);
if (x_36 == 0)
{
lean_object* x_37; 
x_37 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_37, 0, x_5);
lean_ctor_set(x_37, 1, x_6);
lean_ctor_set(x_37, 2, x_10);
x_12 = x_9;
x_13 = x_37;
goto block_27;
}
else
{
lean_object* x_38; 
x_38 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_38, 0, x_9);
lean_ctor_set(x_38, 1, x_10);
lean_ctor_set(x_38, 2, x_6);
x_12 = x_5;
x_13 = x_38;
goto block_27;
}
}
else
{
lean_object* x_39; 
x_39 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(x_1, x_2, x_11);
lean_ctor_set(x_3, 3, x_39);
return x_3;
}
}
else
{
uint8_t x_40; 
lean_dec(x_3);
x_40 = lean_nat_dec_lt(x_8, x_4);
if (x_40 == 0)
{
lean_object* x_41; uint8_t x_42; 
lean_dec(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
lean_inc(x_9);
lean_inc(x_5);
x_41 = lean_apply_2(x_1, x_5, x_9);
x_42 = lean_unbox(x_41);
if (x_42 == 0)
{
lean_object* x_43; 
x_43 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_43, 0, x_5);
lean_ctor_set(x_43, 1, x_6);
lean_ctor_set(x_43, 2, x_10);
x_12 = x_9;
x_13 = x_43;
goto block_27;
}
else
{
lean_object* x_44; 
x_44 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_44, 0, x_9);
lean_ctor_set(x_44, 1, x_10);
lean_ctor_set(x_44, 2, x_6);
x_12 = x_5;
x_13 = x_44;
goto block_27;
}
}
else
{
lean_object* x_45; lean_object* x_46; 
x_45 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(x_1, x_2, x_11);
x_46 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_46, 0, x_8);
lean_ctor_set(x_46, 1, x_9);
lean_ctor_set(x_46, 2, x_10);
lean_ctor_set(x_46, 3, x_45);
return x_46;
}
}
}
else
{
uint8_t x_47; 
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
x_47 = !lean_is_exclusive(x_2);
if (x_47 == 0)
{
lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; 
x_48 = lean_ctor_get(x_2, 3);
lean_dec(x_48);
x_49 = lean_ctor_get(x_2, 2);
lean_dec(x_49);
x_50 = lean_ctor_get(x_2, 1);
lean_dec(x_50);
x_51 = lean_ctor_get(x_2, 0);
lean_dec(x_51);
x_52 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(x_1, x_7, x_3);
lean_ctor_set(x_2, 3, x_52);
return x_2;
}
else
{
lean_object* x_53; lean_object* x_54; 
lean_dec(x_2);
x_53 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(x_1, x_7, x_3);
x_54 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_54, 0, x_4);
lean_ctor_set(x_54, 1, x_5);
lean_ctor_set(x_54, 2, x_6);
lean_ctor_set(x_54, 3, x_53);
return x_54;
}
}
block_27:
{
lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_14 = lean_unsigned_to_nat(1u);
x_15 = lean_nat_add(x_4, x_14);
lean_dec(x_4);
x_16 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(x_7, x_15);
if (x_16 == 0)
{
uint8_t x_17; 
x_17 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(x_11, x_15);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; 
x_18 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(x_1, x_7, x_11);
x_19 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_19, 0, x_15);
lean_ctor_set(x_19, 1, x_12);
lean_ctor_set(x_19, 2, x_13);
lean_ctor_set(x_19, 3, x_18);
return x_19;
}
else
{
lean_object* x_20; 
x_20 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_20, 0, x_15);
lean_ctor_set(x_20, 1, x_12);
lean_ctor_set(x_20, 2, x_13);
lean_ctor_set(x_20, 3, x_11);
x_2 = x_7;
x_3 = x_20;
goto _start;
}
}
else
{
uint8_t x_22; 
x_22 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(x_11, x_15);
if (x_22 == 0)
{
lean_object* x_23; 
x_23 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_23, 0, x_15);
lean_ctor_set(x_23, 1, x_12);
lean_ctor_set(x_23, 2, x_13);
lean_ctor_set(x_23, 3, x_7);
x_2 = x_23;
x_3 = x_11;
goto _start;
}
else
{
lean_object* x_25; lean_object* x_26; 
x_25 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(x_1, x_7, x_11);
x_26 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_26, 0, x_15);
lean_ctor_set(x_26, 1, x_12);
lean_ctor_set(x_26, 2, x_13);
lean_ctor_set(x_26, 3, x_25);
return x_26;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__3_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_8; 
lean_dec(x_7);
lean_dec(x_6);
x_8 = lean_apply_1(x_5, x_4);
return x_8;
}
else
{
lean_dec(x_5);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_9; 
lean_dec(x_7);
x_9 = lean_apply_2(x_6, x_3, lean_box(0));
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_dec(x_6);
x_10 = lean_ctor_get(x_3, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
x_12 = lean_ctor_get(x_3, 2);
lean_inc(x_12);
x_13 = lean_ctor_get(x_3, 3);
lean_inc(x_13);
lean_dec_ref(x_3);
x_14 = lean_ctor_get(x_4, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_4, 1);
lean_inc(x_15);
x_16 = lean_ctor_get(x_4, 2);
lean_inc(x_16);
x_17 = lean_ctor_get(x_4, 3);
lean_inc(x_17);
lean_dec_ref(x_4);
x_18 = lean_apply_8(x_7, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
return x_18;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__3_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_6; 
lean_dec(x_5);
lean_dec(x_4);
x_6 = lean_apply_1(x_3, x_2);
return x_6;
}
else
{
lean_dec(x_3);
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_7; 
lean_dec(x_5);
x_7 = lean_apply_2(x_4, x_1, lean_box(0));
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
lean_dec(x_4);
x_8 = lean_ctor_get(x_1, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
x_10 = lean_ctor_get(x_1, 2);
lean_inc(x_10);
x_11 = lean_ctor_get(x_1, 3);
lean_inc(x_11);
lean_dec_ref(x_1);
x_12 = lean_ctor_get(x_2, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_2, 1);
lean_inc(x_13);
x_14 = lean_ctor_get(x_2, 2);
lean_inc(x_14);
x_15 = lean_ctor_get(x_2, 3);
lean_inc(x_15);
lean_dec_ref(x_2);
x_16 = lean_apply_8(x_5, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_16;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__1_splitter___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_dec(x_2);
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_1, 2);
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_sub(x_2, x_7);
lean_dec(x_2);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_8);
x_9 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_4);
lean_ctor_set(x_9, 2, x_5);
lean_ctor_set(x_9, 3, x_3);
x_1 = x_6;
x_2 = x_8;
x_3 = x_9;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go(x_1, x_2, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(x_1);
x_3 = lean_box(0);
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec_ref(x_1);
return x_2;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc(x_5);
lean_dec_ref(x_3);
lean_inc_ref(x_1);
lean_inc(x_4);
lean_inc(x_2);
x_6 = lean_apply_2(x_1, x_2, x_4);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
lean_dec(x_2);
x_2 = x_4;
x_3 = x_5;
goto _start;
}
else
{
lean_dec(x_4);
x_3 = x_5;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_head_x3f(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
lean_dec_ref(x_2);
x_4 = lean_box(0);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 3);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(x_2, x_5, x_6);
x_8 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_head_x3f___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_3; 
lean_dec_ref(x_1);
x_3 = lean_box(0);
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 3);
lean_inc(x_5);
lean_dec_ref(x_2);
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(x_1, x_4, x_5);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_2);
lean_ctor_set(x_5, 2, x_3);
lean_ctor_set(x_5, 3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 2);
lean_inc(x_7);
x_8 = lean_ctor_get(x_3, 3);
lean_inc(x_8);
lean_dec_ref(x_3);
x_9 = lean_ctor_get(x_4, 1);
lean_inc(x_7);
lean_inc(x_6);
x_10 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg___lam__0), 4, 3);
lean_closure_set(x_10, 0, x_5);
lean_closure_set(x_10, 1, x_6);
lean_closure_set(x_10, 2, x_7);
lean_inc_ref(x_2);
x_11 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, lean_box(0));
lean_closure_set(x_11, 2, lean_box(0));
lean_closure_set(x_11, 3, x_2);
lean_closure_set(x_11, 4, x_10);
lean_inc_ref(x_1);
lean_inc(x_6);
lean_inc(x_9);
x_12 = lean_apply_2(x_1, x_9, x_6);
x_13 = lean_unbox(x_12);
if (x_13 == 0)
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_4);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_15 = lean_ctor_get(x_4, 3);
lean_dec(x_15);
x_16 = lean_ctor_get(x_4, 2);
lean_dec(x_16);
x_17 = lean_ctor_get(x_4, 1);
lean_dec(x_17);
x_18 = lean_ctor_get(x_4, 0);
lean_dec(x_18);
lean_inc(x_8);
lean_ctor_set(x_4, 3, x_8);
lean_ctor_set(x_4, 2, x_7);
lean_ctor_set(x_4, 1, x_6);
lean_ctor_set(x_4, 0, x_2);
x_2 = x_11;
x_3 = x_8;
goto _start;
}
else
{
lean_object* x_20; 
lean_dec(x_4);
lean_inc(x_8);
x_20 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_20, 0, x_2);
lean_ctor_set(x_20, 1, x_6);
lean_ctor_set(x_20, 2, x_7);
lean_ctor_set(x_20, 3, x_8);
x_2 = x_11;
x_3 = x_8;
x_4 = x_20;
goto _start;
}
}
else
{
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_2);
x_2 = x_11;
x_3 = x_8;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_6, 0, x_1);
lean_ctor_set(x_6, 1, x_2);
lean_ctor_set(x_6, 2, x_3);
lean_ctor_set(x_6, 3, x_5);
x_7 = lean_apply_1(x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 2);
lean_inc(x_7);
x_8 = lean_ctor_get(x_3, 3);
lean_inc(x_8);
lean_dec_ref(x_3);
x_9 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_2);
lean_inc(x_7);
lean_inc(x_6);
x_10 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg___lam__0), 5, 4);
lean_closure_set(x_10, 0, x_5);
lean_closure_set(x_10, 1, x_6);
lean_closure_set(x_10, 2, x_7);
lean_closure_set(x_10, 3, x_2);
lean_inc_ref(x_1);
lean_inc(x_6);
lean_inc(x_9);
x_11 = lean_apply_2(x_1, x_9, x_6);
x_12 = lean_unbox(x_11);
if (x_12 == 0)
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_4);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_14 = lean_ctor_get(x_4, 3);
lean_dec(x_14);
x_15 = lean_ctor_get(x_4, 2);
lean_dec(x_15);
x_16 = lean_ctor_get(x_4, 1);
lean_dec(x_16);
x_17 = lean_ctor_get(x_4, 0);
lean_dec(x_17);
lean_inc(x_8);
lean_ctor_set(x_4, 3, x_8);
lean_ctor_set(x_4, 2, x_7);
lean_ctor_set(x_4, 1, x_6);
lean_ctor_set(x_4, 0, x_2);
x_2 = x_10;
x_3 = x_8;
goto _start;
}
else
{
lean_object* x_19; 
lean_dec(x_4);
lean_inc(x_8);
x_19 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_19, 0, x_2);
lean_ctor_set(x_19, 1, x_6);
lean_ctor_set(x_19, 2, x_7);
lean_ctor_set(x_19, 3, x_8);
x_2 = x_10;
x_3 = x_8;
x_4 = x_19;
goto _start;
}
}
else
{
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_2);
x_2 = x_10;
x_3 = x_8;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_dec_ref(x_1);
return x_3;
}
else
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec_ref(x_1);
return x_2;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_28; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_2, 2);
x_7 = lean_ctor_get(x_2, 3);
x_8 = lean_ctor_get(x_3, 0);
x_9 = lean_ctor_get(x_3, 1);
x_10 = lean_ctor_get(x_3, 2);
x_11 = lean_ctor_get(x_3, 3);
x_28 = lean_nat_dec_lt(x_4, x_8);
if (x_28 == 0)
{
uint8_t x_29; 
lean_inc(x_11);
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_8);
x_29 = !lean_is_exclusive(x_3);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; uint8_t x_34; 
x_30 = lean_ctor_get(x_3, 3);
lean_dec(x_30);
x_31 = lean_ctor_get(x_3, 2);
lean_dec(x_31);
x_32 = lean_ctor_get(x_3, 1);
lean_dec(x_32);
x_33 = lean_ctor_get(x_3, 0);
lean_dec(x_33);
x_34 = lean_nat_dec_lt(x_8, x_4);
if (x_34 == 0)
{
lean_object* x_35; uint8_t x_36; 
lean_free_object(x_3);
lean_dec(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
lean_inc(x_9);
lean_inc(x_5);
x_35 = lean_apply_2(x_1, x_5, x_9);
x_36 = lean_unbox(x_35);
if (x_36 == 0)
{
lean_object* x_37; 
x_37 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_37, 0, x_5);
lean_ctor_set(x_37, 1, x_6);
lean_ctor_set(x_37, 2, x_10);
x_12 = x_9;
x_13 = x_37;
goto block_27;
}
else
{
lean_object* x_38; 
x_38 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_38, 0, x_9);
lean_ctor_set(x_38, 1, x_10);
lean_ctor_set(x_38, 2, x_6);
x_12 = x_5;
x_13 = x_38;
goto block_27;
}
}
else
{
lean_object* x_39; 
x_39 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(x_1, x_2, x_11);
lean_ctor_set(x_3, 3, x_39);
return x_3;
}
}
else
{
uint8_t x_40; 
lean_dec(x_3);
x_40 = lean_nat_dec_lt(x_8, x_4);
if (x_40 == 0)
{
lean_object* x_41; uint8_t x_42; 
lean_dec(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
lean_inc(x_9);
lean_inc(x_5);
x_41 = lean_apply_2(x_1, x_5, x_9);
x_42 = lean_unbox(x_41);
if (x_42 == 0)
{
lean_object* x_43; 
x_43 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_43, 0, x_5);
lean_ctor_set(x_43, 1, x_6);
lean_ctor_set(x_43, 2, x_10);
x_12 = x_9;
x_13 = x_43;
goto block_27;
}
else
{
lean_object* x_44; 
x_44 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_44, 0, x_9);
lean_ctor_set(x_44, 1, x_10);
lean_ctor_set(x_44, 2, x_6);
x_12 = x_5;
x_13 = x_44;
goto block_27;
}
}
else
{
lean_object* x_45; lean_object* x_46; 
x_45 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(x_1, x_2, x_11);
x_46 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_46, 0, x_8);
lean_ctor_set(x_46, 1, x_9);
lean_ctor_set(x_46, 2, x_10);
lean_ctor_set(x_46, 3, x_45);
return x_46;
}
}
}
else
{
uint8_t x_47; 
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
x_47 = !lean_is_exclusive(x_2);
if (x_47 == 0)
{
lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; 
x_48 = lean_ctor_get(x_2, 3);
lean_dec(x_48);
x_49 = lean_ctor_get(x_2, 2);
lean_dec(x_49);
x_50 = lean_ctor_get(x_2, 1);
lean_dec(x_50);
x_51 = lean_ctor_get(x_2, 0);
lean_dec(x_51);
x_52 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(x_1, x_7, x_3);
lean_ctor_set(x_2, 3, x_52);
return x_2;
}
else
{
lean_object* x_53; lean_object* x_54; 
lean_dec(x_2);
x_53 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(x_1, x_7, x_3);
x_54 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_54, 0, x_4);
lean_ctor_set(x_54, 1, x_5);
lean_ctor_set(x_54, 2, x_6);
lean_ctor_set(x_54, 3, x_53);
return x_54;
}
}
block_27:
{
lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_14 = lean_unsigned_to_nat(1u);
x_15 = lean_nat_add(x_4, x_14);
lean_dec(x_4);
x_16 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(x_7, x_15);
if (x_16 == 0)
{
uint8_t x_17; 
x_17 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(x_11, x_15);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; 
x_18 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(x_1, x_7, x_11);
x_19 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_19, 0, x_15);
lean_ctor_set(x_19, 1, x_12);
lean_ctor_set(x_19, 2, x_13);
lean_ctor_set(x_19, 3, x_18);
return x_19;
}
else
{
lean_object* x_20; 
x_20 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_20, 0, x_15);
lean_ctor_set(x_20, 1, x_12);
lean_ctor_set(x_20, 2, x_13);
lean_ctor_set(x_20, 3, x_11);
x_2 = x_7;
x_3 = x_20;
goto _start;
}
}
else
{
uint8_t x_22; 
x_22 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(x_11, x_15);
if (x_22 == 0)
{
lean_object* x_23; 
x_23 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_23, 0, x_15);
lean_ctor_set(x_23, 1, x_12);
lean_ctor_set(x_23, 2, x_13);
lean_ctor_set(x_23, 3, x_7);
x_2 = x_23;
x_3 = x_11;
goto _start;
}
else
{
lean_object* x_25; lean_object* x_26; 
x_25 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(x_1, x_7, x_11);
x_26 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_26, 0, x_15);
lean_ctor_set(x_26, 1, x_12);
lean_ctor_set(x_26, 2, x_13);
lean_ctor_set(x_26, 3, x_25);
return x_26;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_3; 
lean_dec_ref(x_1);
x_3 = lean_box(0);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_2);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_2, 1);
x_7 = lean_ctor_get(x_2, 2);
x_8 = lean_ctor_get(x_2, 3);
x_9 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0___boxed), 1, 0);
lean_inc(x_7);
lean_inc(x_6);
x_10 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg___lam__0), 4, 3);
lean_closure_set(x_10, 0, x_5);
lean_closure_set(x_10, 1, x_6);
lean_closure_set(x_10, 2, x_7);
lean_inc(x_8);
lean_ctor_set_tag(x_2, 0);
lean_ctor_set(x_2, 0, x_9);
lean_inc_ref(x_1);
x_11 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg(x_1, x_10, x_8, x_2);
x_12 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_12);
x_13 = lean_ctor_get(x_11, 1);
lean_inc(x_13);
x_14 = lean_ctor_get(x_11, 2);
lean_inc(x_14);
x_15 = lean_ctor_get(x_11, 3);
lean_inc(x_15);
lean_dec_ref(x_11);
x_16 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg(x_14);
lean_dec(x_14);
x_17 = lean_apply_1(x_12, x_15);
x_18 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(x_1, x_16, x_17);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_13);
lean_ctor_set(x_19, 1, x_18);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
return x_20;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_21 = lean_ctor_get(x_2, 0);
x_22 = lean_ctor_get(x_2, 1);
x_23 = lean_ctor_get(x_2, 2);
x_24 = lean_ctor_get(x_2, 3);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_21);
lean_dec(x_2);
x_25 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0___boxed), 1, 0);
lean_inc(x_23);
lean_inc(x_22);
x_26 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg___lam__0), 4, 3);
lean_closure_set(x_26, 0, x_21);
lean_closure_set(x_26, 1, x_22);
lean_closure_set(x_26, 2, x_23);
lean_inc(x_24);
x_27 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_27, 0, x_25);
lean_ctor_set(x_27, 1, x_22);
lean_ctor_set(x_27, 2, x_23);
lean_ctor_set(x_27, 3, x_24);
lean_inc_ref(x_1);
x_28 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg(x_1, x_26, x_24, x_27);
x_29 = lean_ctor_get(x_28, 0);
lean_inc_ref(x_29);
x_30 = lean_ctor_get(x_28, 1);
lean_inc(x_30);
x_31 = lean_ctor_get(x_28, 2);
lean_inc(x_31);
x_32 = lean_ctor_get(x_28, 3);
lean_inc(x_32);
lean_dec_ref(x_28);
x_33 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg(x_31);
lean_dec(x_31);
x_34 = lean_apply_1(x_29, x_32);
x_35 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(x_1, x_33, x_34);
x_36 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_36, 0, x_30);
lean_ctor_set(x_36, 1, x_35);
x_37 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_37, 0, x_36);
return x_37;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail_x3f(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_2, x_3);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_4);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_4, 0);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
lean_dec(x_7);
lean_ctor_set(x_4, 0, x_8);
return x_4;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_4, 0);
lean_inc(x_9);
lean_dec(x_4);
x_10 = lean_ctor_get(x_9, 1);
lean_inc(x_10);
lean_dec(x_9);
x_11 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail_x3f___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_3);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_3, 0);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec(x_6);
lean_ctor_set(x_3, 0, x_7);
return x_3;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
lean_dec(x_3);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
lean_dec(x_8);
x_10 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_2, x_3);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_4, 0);
lean_inc(x_6);
lean_dec_ref(x_4);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec(x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec(x_5);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_findMin_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_5; 
lean_dec(x_4);
x_5 = lean_apply_1(x_3, x_2);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_dec(x_3);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_1, 2);
lean_inc(x_8);
x_9 = lean_ctor_get(x_1, 3);
lean_inc(x_9);
lean_dec_ref(x_1);
x_10 = lean_apply_5(x_4, x_6, x_7, x_8, x_9, x_2);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_findMin_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_findMin_match__1_splitter___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_toHeap_go_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_6; 
lean_dec(x_5);
x_6 = lean_apply_2(x_4, x_2, x_3);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_dec(x_4);
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
lean_dec_ref(x_1);
x_10 = lean_apply_5(x_5, x_7, x_8, x_9, x_2, x_3);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_toHeap_go_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_toHeap_go_match__1_splitter___redArg(x_3, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_1, x_2, x_3, x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc_ref(x_2);
x_9 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_2, x_3);
if (lean_obj_tag(x_9) == 0)
{
lean_object* x_10; 
lean_inc(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_10 = lean_apply_2(x_8, lean_box(0), x_4);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_9, 0);
lean_inc(x_11);
lean_dec_ref(x_9);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_11, 1);
lean_inc(x_13);
lean_dec(x_11);
lean_inc(x_5);
x_14 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg___lam__0), 5, 4);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_13);
lean_closure_set(x_14, 3, x_5);
x_15 = lean_apply_2(x_5, x_4, x_12);
x_16 = lean_apply_4(x_7, lean_box(0), lean_box(0), x_15, x_14);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_foldM_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; 
lean_dec(x_3);
x_4 = lean_apply_1(x_2, lean_box(0));
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_dec(x_2);
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec(x_5);
x_8 = lean_apply_3(x_3, x_6, x_7, lean_box(0));
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_foldM_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_foldM_match__1_splitter___redArg(x_3, x_4, x_5);
return x_6;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__0), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__1___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__2___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__3), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__4___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__5___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__6), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__1;
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__5;
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__4;
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__3;
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__2;
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__7;
x_6 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_3);
lean_ctor_set(x_6, 3, x_2);
lean_ctor_set(x_6, 4, x_1);
return x_6;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__6;
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__8;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_8 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_7, x_3, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_5, x_1, x_2, x_3, x_4);
return x_6;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Array_push___boxed), 3, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0;
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1;
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_6, x_2, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0;
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1;
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_5, x_1, x_2, x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toList(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0;
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1;
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_6, x_2, x_3, x_4, x_5);
x_8 = lean_array_to_list(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toList___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0;
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1;
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_5, x_1, x_2, x_3, x_4);
x_7 = lean_array_to_list(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_3(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc(x_1);
x_8 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__0), 4, 3);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, x_7);
x_9 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(x_3, x_4, x_1, x_5);
x_10 = lean_apply_4(x_6, lean_box(0), lean_box(0), x_9, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, lean_box(0), x_2);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_4, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_4, 1);
lean_inc(x_10);
x_11 = lean_ctor_get(x_4, 2);
lean_inc(x_11);
lean_dec_ref(x_4);
lean_inc(x_8);
lean_inc(x_2);
lean_inc_ref(x_1);
lean_inc(x_3);
x_12 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__1), 7, 6);
lean_closure_set(x_12, 0, x_3);
lean_closure_set(x_12, 1, x_9);
lean_closure_set(x_12, 2, x_1);
lean_closure_set(x_12, 3, x_2);
lean_closure_set(x_12, 4, x_11);
lean_closure_set(x_12, 5, x_8);
x_13 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(x_1, x_2, x_3, x_10);
x_14 = lean_apply_4(x_8, lean_box(0), lean_box(0), x_13, x_12);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_3(x_1, x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc(x_1);
x_8 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__0), 4, 3);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, x_7);
x_9 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(x_3, x_4, x_1, x_5);
x_10 = lean_apply_4(x_6, lean_box(0), lean_box(0), x_9, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, lean_box(0), x_2);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_4, 1);
lean_inc(x_9);
x_10 = lean_ctor_get(x_4, 2);
lean_inc(x_10);
x_11 = lean_ctor_get(x_4, 3);
lean_inc(x_11);
lean_dec_ref(x_4);
lean_inc(x_8);
lean_inc(x_2);
lean_inc_ref(x_1);
lean_inc(x_3);
x_12 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__1), 7, 6);
lean_closure_set(x_12, 0, x_3);
lean_closure_set(x_12, 1, x_9);
lean_closure_set(x_12, 2, x_1);
lean_closure_set(x_12, 3, x_2);
lean_closure_set(x_12, 4, x_11);
lean_closure_set(x_12, 5, x_8);
x_13 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(x_1, x_2, x_3, x_10);
x_14 = lean_apply_4(x_8, lean_box(0), lean_box(0), x_13, x_12);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTree(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(x_6, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTree___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(x_4, x_1, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_2, 1);
x_7 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_1);
x_8 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(x_1, x_7, x_3);
x_9 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(x_1, x_6, x_8);
lean_inc(x_5);
x_10 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_2, 2);
x_7 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_1);
x_8 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(x_1, x_7, x_3);
x_9 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(x_1, x_6, x_8);
lean_inc(x_5);
x_10 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0___boxed), 1, 0);
x_3 = lean_box(0);
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(x_2, x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_2, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 1);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 2);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_array_push(x_3, x_5);
lean_inc_ref(x_1);
x_9 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0___redArg(x_1, x_6, x_8);
x_2 = x_7;
x_3 = x_9;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 2);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 3);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_array_push(x_3, x_5);
lean_inc_ref(x_1);
x_9 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0___redArg(x_1, x_6, x_8);
x_2 = x_7;
x_3 = x_9;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0___boxed), 1, 0);
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0;
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0___redArg(x_2, x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_6 = lean_ctor_get(x_4, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_4, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_4, 2);
lean_inc(x_8);
x_9 = lean_ctor_get(x_4, 3);
lean_inc(x_9);
lean_dec_ref(x_4);
x_10 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_1);
lean_inc(x_7);
lean_inc(x_10);
x_11 = lean_apply_2(x_1, x_10, x_7);
x_12 = lean_unbox(x_11);
if (x_12 == 0)
{
uint8_t x_13; 
lean_dec(x_5);
x_13 = !lean_is_exclusive(x_3);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_14 = lean_ctor_get(x_3, 3);
lean_dec(x_14);
x_15 = lean_ctor_get(x_3, 2);
lean_dec(x_15);
x_16 = lean_ctor_get(x_3, 1);
lean_dec(x_16);
x_17 = lean_ctor_get(x_3, 0);
lean_dec(x_17);
lean_inc_ref(x_2);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_6);
x_18 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg___lam__0), 5, 4);
lean_closure_set(x_18, 0, x_6);
lean_closure_set(x_18, 1, x_7);
lean_closure_set(x_18, 2, x_8);
lean_closure_set(x_18, 3, x_2);
lean_inc(x_9);
lean_ctor_set(x_3, 3, x_9);
lean_ctor_set(x_3, 2, x_8);
lean_ctor_set(x_3, 1, x_7);
lean_ctor_set(x_3, 0, x_2);
x_2 = x_18;
x_4 = x_9;
x_5 = x_6;
goto _start;
}
else
{
lean_object* x_20; lean_object* x_21; 
lean_dec(x_3);
lean_inc_ref(x_2);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_6);
x_20 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg___lam__0), 5, 4);
lean_closure_set(x_20, 0, x_6);
lean_closure_set(x_20, 1, x_7);
lean_closure_set(x_20, 2, x_8);
lean_closure_set(x_20, 3, x_2);
lean_inc(x_9);
x_21 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_21, 0, x_2);
lean_ctor_set(x_21, 1, x_7);
lean_ctor_set(x_21, 2, x_8);
lean_ctor_set(x_21, 3, x_9);
x_2 = x_20;
x_3 = x_21;
x_4 = x_9;
x_5 = x_6;
goto _start;
}
}
else
{
lean_object* x_23; 
x_23 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg___lam__0), 5, 4);
lean_closure_set(x_23, 0, x_6);
lean_closure_set(x_23, 1, x_7);
lean_closure_set(x_23, 2, x_8);
lean_closure_set(x_23, 3, x_2);
x_2 = x_23;
x_4 = x_9;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin___redArg(x_2, x_4, x_5, x_6, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_3);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_mkBinomialHeap(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_mkBinomialHeap___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_mkBinomialHeap(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_empty(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_empty___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_empty(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instEmptyCollection(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instEmptyCollection___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_instEmptyCollection(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instInhabited___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_instInhabited(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_isEmpty(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
uint8_t x_4; 
x_4 = 1;
return x_4;
}
else
{
uint8_t x_5; 
x_5 = 0;
return x_5;
}
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_isEmpty___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_isEmpty___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_batteries_Batteries_BinomialHeap_isEmpty(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_isEmpty___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_batteries_Batteries_BinomialHeap_isEmpty___redArg(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_size(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_size___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_box(0);
x_6 = lean_box(0);
x_7 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_7, 0, x_4);
lean_ctor_set(x_7, 1, x_3);
lean_ctor_set(x_7, 2, x_5);
lean_ctor_set(x_7, 3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_box(0);
x_4 = lean_box(0);
x_5 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_1);
lean_ctor_set(x_5, 2, x_3);
lean_ctor_set(x_5, 3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_singleton(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_merge(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_merge___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_insert(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_box(0);
x_7 = lean_box(0);
x_8 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_3);
lean_ctor_set(x_8, 2, x_6);
lean_ctor_set(x_8, 3, x_7);
x_9 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(x_2, x_8, x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_insert___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_box(0);
x_6 = lean_box(0);
x_7 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_7, 0, x_4);
lean_ctor_set(x_7, 1, x_2);
lean_ctor_set(x_7, 2, x_5);
lean_ctor_set(x_7, 3, x_6);
x_8 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(x_1, x_7, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec_ref(x_1);
return x_2;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_4 = lean_ctor_get(x_3, 0);
x_5 = lean_ctor_get(x_3, 1);
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_box(0);
x_8 = lean_box(0);
lean_inc(x_4);
x_9 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_4);
lean_ctor_set(x_9, 2, x_7);
lean_ctor_set(x_9, 3, x_8);
lean_inc_ref(x_1);
x_10 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(x_1, x_9, x_2);
x_2 = x_10;
x_3 = x_5;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_box(0);
x_4 = lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_ofList___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_ofList(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_ofList___redArg(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg(lean_object* x_1, lean_object* x_2, size_t x_3, size_t x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = lean_usize_dec_eq(x_3, x_4);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; size_t x_13; size_t x_14; 
x_7 = lean_array_uget(x_2, x_3);
x_8 = lean_unsigned_to_nat(0u);
x_9 = lean_box(0);
x_10 = lean_box(0);
x_11 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_11, 0, x_8);
lean_ctor_set(x_11, 1, x_7);
lean_ctor_set(x_11, 2, x_9);
lean_ctor_set(x_11, 3, x_10);
lean_inc_ref(x_1);
x_12 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(x_1, x_11, x_5);
x_13 = 1;
x_14 = lean_usize_add(x_3, x_13);
x_3 = x_14;
x_5 = x_12;
goto _start;
}
else
{
lean_dec_ref(x_1);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_3 = lean_box(0);
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_array_get_size(x_2);
x_6 = lean_nat_dec_lt(x_4, x_5);
if (x_6 == 0)
{
lean_dec_ref(x_1);
return x_3;
}
else
{
uint8_t x_7; 
x_7 = lean_nat_dec_le(x_5, x_5);
if (x_7 == 0)
{
lean_dec_ref(x_1);
return x_3;
}
else
{
size_t x_8; size_t x_9; lean_object* x_10; 
x_8 = 0;
x_9 = lean_usize_of_nat(x_5);
x_10 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg(x_1, x_2, x_8, x_9, x_3);
return x_10;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_ofArray___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, size_t x_4, size_t x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_ofArray(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
size_t x_7; size_t x_8; lean_object* x_9; 
x_7 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_8 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_9 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0(x_1, x_2, x_3, x_7, x_8, x_6);
lean_dec_ref(x_3);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
size_t x_6; size_t x_7; lean_object* x_8; 
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_8 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg(x_1, x_2, x_6, x_7, x_5);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_ofArray___redArg(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_deleteMin(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_2, x_3);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_4);
if (x_6 == 0)
{
lean_object* x_7; uint8_t x_8; 
x_7 = lean_ctor_get(x_4, 0);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
return x_4;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_7, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
lean_ctor_set(x_4, 0, x_11);
return x_4;
}
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lean_ctor_get(x_4, 0);
lean_inc(x_12);
lean_dec(x_4);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_12, 1);
lean_inc(x_14);
if (lean_is_exclusive(x_12)) {
 lean_ctor_release(x_12, 0);
 lean_ctor_release(x_12, 1);
 x_15 = x_12;
} else {
 lean_dec_ref(x_12);
 x_15 = lean_box(0);
}
if (lean_is_scalar(x_15)) {
 x_16 = lean_alloc_ctor(0, 2, 0);
} else {
 x_16 = x_15;
}
lean_ctor_set(x_16, 0, x_13);
lean_ctor_set(x_16, 1, x_14);
x_17 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_17, 0, x_16);
return x_17;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_deleteMin___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_3);
if (x_5 == 0)
{
lean_object* x_6; uint8_t x_7; 
x_6 = lean_ctor_get(x_3, 0);
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
return x_3;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_6);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_8);
lean_ctor_set(x_10, 1, x_9);
lean_ctor_set(x_3, 0, x_10);
return x_3;
}
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_3, 0);
lean_inc(x_11);
lean_dec(x_3);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_11, 1);
lean_inc(x_13);
if (lean_is_exclusive(x_11)) {
 lean_ctor_release(x_11, 0);
 lean_ctor_release(x_11, 1);
 x_14 = x_11;
} else {
 lean_dec_ref(x_11);
 x_14 = lean_box(0);
}
if (lean_is_scalar(x_14)) {
 x_15 = lean_alloc_ctor(0, 2, 0);
} else {
 x_15 = x_14;
}
lean_ctor_set(x_15, 0, x_12);
lean_ctor_set(x_15, 1, x_13);
x_16 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_16, 0, x_15);
return x_16;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instStream(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_deleteMin), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instStream___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_deleteMin), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_5; 
lean_dec(x_4);
lean_dec(x_2);
x_5 = lean_apply_2(x_1, lean_box(0), x_3);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_1);
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = lean_apply_2(x_2, x_4, x_6);
return x_7;
}
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_forIn___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_batteries_ForInStep_run___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_6, 0);
x_8 = lean_ctor_get(x_6, 1);
x_9 = lean_ctor_get(x_7, 0);
lean_inc(x_9);
lean_inc(x_8);
x_10 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_forIn___redArg___lam__0), 4, 2);
lean_closure_set(x_10, 0, x_8);
lean_closure_set(x_10, 1, x_5);
x_11 = lp_batteries_Batteries_BinomialHeap_forIn___redArg___closed__0;
x_12 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_12, 0, x_4);
x_13 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_2, x_1, x_3, x_12, x_10);
x_14 = lean_apply_4(x_9, lean_box(0), lean_box(0), x_11, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_batteries_Batteries_BinomialHeap_forIn___redArg(x_2, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Batteries_BinomialHeap_forIn___redArg(x_1, x_2, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg___lam__0), 6, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x3f(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
lean_dec_ref(x_2);
x_4 = lean_box(0);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 3);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(x_2, x_5, x_6);
x_8 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x3f___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_3; 
lean_dec_ref(x_1);
x_3 = lean_box(0);
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 3);
lean_inc(x_5);
lean_dec_ref(x_2);
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(x_1, x_4, x_5);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_head_x21___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Init.Data.Option.BasicAux", 25, 25);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_head_x21___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Option.get!", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_head_x21___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("value is none", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_head_x21___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_batteries_Batteries_BinomialHeap_head_x21___closed__2;
x_2 = lean_unsigned_to_nat(14u);
x_3 = lean_unsigned_to_nat(22u);
x_4 = lp_batteries_Batteries_BinomialHeap_head_x21___closed__1;
x_5 = lp_batteries_Batteries_BinomialHeap_head_x21___closed__0;
x_6 = l_mkPanicMessageWithDecl(x_5, x_4, x_3, x_2, x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec_ref(x_2);
x_5 = lp_batteries_Batteries_BinomialHeap_head_x21___closed__3;
x_6 = l_panic___redArg(x_3, x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_dec(x_3);
x_7 = lean_ctor_get(x_4, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_4, 3);
lean_inc(x_8);
lean_dec_ref(x_4);
x_9 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(x_2, x_7, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec_ref(x_1);
x_4 = lp_batteries_Batteries_BinomialHeap_head_x21___closed__3;
x_5 = l_panic___redArg(x_2, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_dec(x_2);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 3);
lean_inc(x_7);
lean_dec_ref(x_3);
x_8 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(x_1, x_6, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_dec_ref(x_2);
lean_inc(x_3);
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 3);
lean_inc(x_6);
lean_dec_ref(x_4);
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(x_2, x_5, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec_ref(x_1);
lean_inc(x_2);
return x_2;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 3);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(x_1, x_4, x_5);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Batteries_BinomialHeap_headI(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_headI___redArg(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail_x3f(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_2, x_3);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_4);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_4, 0);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
lean_dec(x_7);
lean_ctor_set(x_4, 0, x_8);
return x_4;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_4, 0);
lean_inc(x_9);
lean_dec(x_4);
x_10 = lean_ctor_get(x_9, 1);
lean_inc(x_10);
lean_dec(x_9);
x_11 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail_x3f___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_3);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_3, 0);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec(x_6);
lean_ctor_set(x_3, 0, x_7);
return x_3;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
lean_dec(x_3);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
lean_dec(x_8);
x_10 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_2, x_3);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_4, 0);
lean_inc(x_6);
lean_dec_ref(x_4);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
lean_dec(x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec(x_5);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_foldM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_5, x_2, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_foldM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_2, x_1, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_fold(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_8 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_7, x_2, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_fold___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_5, x_1, x_2, x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toList(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0;
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1;
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_6, x_2, x_3, x_4, x_5);
x_8 = lean_array_to_list(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toList___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0;
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1;
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_5, x_1, x_2, x_3, x_4);
x_7 = lean_array_to_list(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArray(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0;
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1;
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_7 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_6, x_2, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArray___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0;
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1;
x_5 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9;
x_6 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(x_5, x_1, x_2, x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_toListUnordered(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_toListUnordered___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Batteries_BinomialHeap_toArrayUnordered(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Classes_Order(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Control_ForInStep_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_BinomialHeap_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Classes_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Control_ForInStep_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__0 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__0();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__0);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__1 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__1();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__1);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__4 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__4();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__4);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__5 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__5();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__5);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__6 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__6();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__6);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__0 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__0();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__0);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__1 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__1();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__1);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__2 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__2();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__2);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__3 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__3();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__3);
lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__4 = _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__4();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__4);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__0 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__0();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__0);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__1 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__1();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__1);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__2 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__2();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__2);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__3 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__3();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__3);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__4 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__4();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__4);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__5 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__5();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__5);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__6 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__6();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__6);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__7 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__7();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__7);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__8 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__8();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__8);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___closed__9);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__0);
lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1 = _init_lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___closed__1);
lp_batteries_Batteries_BinomialHeap_forIn___redArg___closed__0 = _init_lp_batteries_Batteries_BinomialHeap_forIn___redArg___closed__0();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_forIn___redArg___closed__0);
lp_batteries_Batteries_BinomialHeap_head_x21___closed__0 = _init_lp_batteries_Batteries_BinomialHeap_head_x21___closed__0();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_head_x21___closed__0);
lp_batteries_Batteries_BinomialHeap_head_x21___closed__1 = _init_lp_batteries_Batteries_BinomialHeap_head_x21___closed__1();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_head_x21___closed__1);
lp_batteries_Batteries_BinomialHeap_head_x21___closed__2 = _init_lp_batteries_Batteries_BinomialHeap_head_x21___closed__2();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_head_x21___closed__2);
lp_batteries_Batteries_BinomialHeap_head_x21___closed__3 = _init_lp_batteries_Batteries_BinomialHeap_head_x21___closed__3();
lean_mark_persistent(lp_batteries_Batteries_BinomialHeap_head_x21___closed__3);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
