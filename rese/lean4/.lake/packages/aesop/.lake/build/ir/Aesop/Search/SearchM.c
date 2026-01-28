// Lean compiler output
// Module: Aesop.Search.SearchM
// Imports: public import Init public import Aesop.Options public import Aesop.Search.Queue.Class public import Aesop.Stats.Basic public import Aesop.RuleSet public import Aesop.Tree.TreeM
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
lean_object* l_ReaderT_read(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__0;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__14;
static lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonad___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__3;
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonad(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__6;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__4;
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3;
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_mkInitialTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__11;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__16;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__1(lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__12;
lean_object* lean_st_ref_take(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEST(lean_object*, lean_object*);
lean_object* l_Array_empty(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Queue_init_x27___redArg(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__1___boxed(lean_object*);
extern lean_object* l_Lean_Meta_Simp_instInhabitedContext_default;
lean_object* lean_st_ref_get(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormSimpContext;
lean_object* lean_st_mk_ref(lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__5;
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__8;
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormSimpContext_default;
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadRef___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__0;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__2;
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7;
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__0;
lean_object* l_Lean_instAddMessageContextOfMonadLift___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__17;
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___redArg___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__9;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadReaderContext(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__2;
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__6;
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__10;
lean_object* lean_array_fget(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5;
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__13;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadRef(lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__15;
size_t lean_array_size(lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState___redArg(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___redArg(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
static lean_object* lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadReaderContext___boxed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__8;
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_Meta_Simp_instInhabitedContext_default;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; uint8_t x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1;
x_2 = lean_box(0);
x_3 = 0;
x_4 = lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__0;
x_5 = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_2);
lean_ctor_set(x_5, 2, x_1);
lean_ctor_set_uint8(x_5, sizeof(void*)*3, x_3);
lean_ctor_set_uint8(x_5, sizeof(void*)*3 + 1, x_3);
return x_5;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__2;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormSimpContext() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedNormSimpContext_default;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = 0;
x_4 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_1);
lean_ctor_set_uint8(x_4, sizeof(void*)*2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_SearchM_instInhabitedState_default___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_SearchM_instInhabitedState_default(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_SearchM_instInhabitedState_default___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_SearchM_instInhabitedState_default___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_SearchM_instInhabitedState(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonad___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_instMonadEST(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonad___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_SearchM_instMonad___closed__0;
x_2 = l_ReaderT_instMonad___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonad___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Core_instMonadCoreM___lam__0___boxed), 5, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonad___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Core_instMonadCoreM___lam__1___boxed), 7, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonad___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_instMonadMetaM___lam__0___boxed), 7, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonad___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_instMonadMetaM___lam__1___boxed), 9, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonad(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lp_aesop_Aesop_SearchM_instMonad___closed__1;
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_dec(x_6);
x_7 = !lean_is_exclusive(x_5);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; uint8_t x_22; 
x_8 = lean_ctor_get(x_5, 0);
x_9 = lean_ctor_get(x_5, 2);
x_10 = lean_ctor_get(x_5, 3);
x_11 = lean_ctor_get(x_5, 4);
x_12 = lean_ctor_get(x_5, 1);
lean_dec(x_12);
x_13 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_14 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_8);
x_15 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_15, 0, x_8);
x_16 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_16, 0, x_8);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
x_18 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_18, 0, x_11);
x_19 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_19, 0, x_10);
x_20 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_20, 0, x_9);
lean_ctor_set(x_5, 4, x_18);
lean_ctor_set(x_5, 3, x_19);
lean_ctor_set(x_5, 2, x_20);
lean_ctor_set(x_5, 1, x_13);
lean_ctor_set(x_5, 0, x_17);
lean_ctor_set(x_3, 1, x_14);
x_21 = l_ReaderT_instMonad___redArg(x_3);
x_22 = !lean_is_exclusive(x_21);
if (x_22 == 0)
{
lean_object* x_23; lean_object* x_24; uint8_t x_25; 
x_23 = lean_ctor_get(x_21, 0);
x_24 = lean_ctor_get(x_21, 1);
lean_dec(x_24);
x_25 = !lean_is_exclusive(x_23);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_26 = lean_ctor_get(x_23, 0);
x_27 = lean_ctor_get(x_23, 2);
x_28 = lean_ctor_get(x_23, 3);
x_29 = lean_ctor_get(x_23, 4);
x_30 = lean_ctor_get(x_23, 1);
lean_dec(x_30);
x_31 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_32 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_26);
x_33 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_33, 0, x_26);
x_34 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_34, 0, x_26);
x_35 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_35, 0, x_33);
lean_ctor_set(x_35, 1, x_34);
x_36 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_36, 0, x_29);
x_37 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_37, 0, x_28);
x_38 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_38, 0, x_27);
lean_ctor_set(x_23, 4, x_36);
lean_ctor_set(x_23, 3, x_37);
lean_ctor_set(x_23, 2, x_38);
lean_ctor_set(x_23, 1, x_31);
lean_ctor_set(x_23, 0, x_35);
lean_ctor_set(x_21, 1, x_32);
x_39 = l_ReaderT_instMonad___redArg(x_21);
x_40 = l_ReaderT_instMonad___redArg(x_39);
x_41 = l_ReaderT_instMonad___redArg(x_40);
x_42 = l_ReaderT_instMonad___redArg(x_41);
return x_42;
}
else
{
lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; 
x_43 = lean_ctor_get(x_23, 0);
x_44 = lean_ctor_get(x_23, 2);
x_45 = lean_ctor_get(x_23, 3);
x_46 = lean_ctor_get(x_23, 4);
lean_inc(x_46);
lean_inc(x_45);
lean_inc(x_44);
lean_inc(x_43);
lean_dec(x_23);
x_47 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_48 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_43);
x_49 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_49, 0, x_43);
x_50 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_50, 0, x_43);
x_51 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_51, 0, x_49);
lean_ctor_set(x_51, 1, x_50);
x_52 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_52, 0, x_46);
x_53 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_53, 0, x_45);
x_54 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_54, 0, x_44);
x_55 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_55, 0, x_51);
lean_ctor_set(x_55, 1, x_47);
lean_ctor_set(x_55, 2, x_54);
lean_ctor_set(x_55, 3, x_53);
lean_ctor_set(x_55, 4, x_52);
lean_ctor_set(x_21, 1, x_48);
lean_ctor_set(x_21, 0, x_55);
x_56 = l_ReaderT_instMonad___redArg(x_21);
x_57 = l_ReaderT_instMonad___redArg(x_56);
x_58 = l_ReaderT_instMonad___redArg(x_57);
x_59 = l_ReaderT_instMonad___redArg(x_58);
return x_59;
}
}
else
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; 
x_60 = lean_ctor_get(x_21, 0);
lean_inc(x_60);
lean_dec(x_21);
x_61 = lean_ctor_get(x_60, 0);
lean_inc_ref(x_61);
x_62 = lean_ctor_get(x_60, 2);
lean_inc(x_62);
x_63 = lean_ctor_get(x_60, 3);
lean_inc(x_63);
x_64 = lean_ctor_get(x_60, 4);
lean_inc(x_64);
if (lean_is_exclusive(x_60)) {
 lean_ctor_release(x_60, 0);
 lean_ctor_release(x_60, 1);
 lean_ctor_release(x_60, 2);
 lean_ctor_release(x_60, 3);
 lean_ctor_release(x_60, 4);
 x_65 = x_60;
} else {
 lean_dec_ref(x_60);
 x_65 = lean_box(0);
}
x_66 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_67 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_61);
x_68 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_68, 0, x_61);
x_69 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_69, 0, x_61);
x_70 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_70, 0, x_68);
lean_ctor_set(x_70, 1, x_69);
x_71 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_71, 0, x_64);
x_72 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_72, 0, x_63);
x_73 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_73, 0, x_62);
if (lean_is_scalar(x_65)) {
 x_74 = lean_alloc_ctor(0, 5, 0);
} else {
 x_74 = x_65;
}
lean_ctor_set(x_74, 0, x_70);
lean_ctor_set(x_74, 1, x_66);
lean_ctor_set(x_74, 2, x_73);
lean_ctor_set(x_74, 3, x_72);
lean_ctor_set(x_74, 4, x_71);
x_75 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_75, 0, x_74);
lean_ctor_set(x_75, 1, x_67);
x_76 = l_ReaderT_instMonad___redArg(x_75);
x_77 = l_ReaderT_instMonad___redArg(x_76);
x_78 = l_ReaderT_instMonad___redArg(x_77);
x_79 = l_ReaderT_instMonad___redArg(x_78);
return x_79;
}
}
else
{
lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; 
x_80 = lean_ctor_get(x_5, 0);
x_81 = lean_ctor_get(x_5, 2);
x_82 = lean_ctor_get(x_5, 3);
x_83 = lean_ctor_get(x_5, 4);
lean_inc(x_83);
lean_inc(x_82);
lean_inc(x_81);
lean_inc(x_80);
lean_dec(x_5);
x_84 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_85 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_80);
x_86 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_86, 0, x_80);
x_87 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_87, 0, x_80);
x_88 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_88, 0, x_86);
lean_ctor_set(x_88, 1, x_87);
x_89 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_89, 0, x_83);
x_90 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_90, 0, x_82);
x_91 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_91, 0, x_81);
x_92 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_92, 0, x_88);
lean_ctor_set(x_92, 1, x_84);
lean_ctor_set(x_92, 2, x_91);
lean_ctor_set(x_92, 3, x_90);
lean_ctor_set(x_92, 4, x_89);
lean_ctor_set(x_3, 1, x_85);
lean_ctor_set(x_3, 0, x_92);
x_93 = l_ReaderT_instMonad___redArg(x_3);
x_94 = lean_ctor_get(x_93, 0);
lean_inc_ref(x_94);
if (lean_is_exclusive(x_93)) {
 lean_ctor_release(x_93, 0);
 lean_ctor_release(x_93, 1);
 x_95 = x_93;
} else {
 lean_dec_ref(x_93);
 x_95 = lean_box(0);
}
x_96 = lean_ctor_get(x_94, 0);
lean_inc_ref(x_96);
x_97 = lean_ctor_get(x_94, 2);
lean_inc(x_97);
x_98 = lean_ctor_get(x_94, 3);
lean_inc(x_98);
x_99 = lean_ctor_get(x_94, 4);
lean_inc(x_99);
if (lean_is_exclusive(x_94)) {
 lean_ctor_release(x_94, 0);
 lean_ctor_release(x_94, 1);
 lean_ctor_release(x_94, 2);
 lean_ctor_release(x_94, 3);
 lean_ctor_release(x_94, 4);
 x_100 = x_94;
} else {
 lean_dec_ref(x_94);
 x_100 = lean_box(0);
}
x_101 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_102 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_96);
x_103 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_103, 0, x_96);
x_104 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_104, 0, x_96);
x_105 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_105, 0, x_103);
lean_ctor_set(x_105, 1, x_104);
x_106 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_106, 0, x_99);
x_107 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_107, 0, x_98);
x_108 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_108, 0, x_97);
if (lean_is_scalar(x_100)) {
 x_109 = lean_alloc_ctor(0, 5, 0);
} else {
 x_109 = x_100;
}
lean_ctor_set(x_109, 0, x_105);
lean_ctor_set(x_109, 1, x_101);
lean_ctor_set(x_109, 2, x_108);
lean_ctor_set(x_109, 3, x_107);
lean_ctor_set(x_109, 4, x_106);
if (lean_is_scalar(x_95)) {
 x_110 = lean_alloc_ctor(0, 2, 0);
} else {
 x_110 = x_95;
}
lean_ctor_set(x_110, 0, x_109);
lean_ctor_set(x_110, 1, x_102);
x_111 = l_ReaderT_instMonad___redArg(x_110);
x_112 = l_ReaderT_instMonad___redArg(x_111);
x_113 = l_ReaderT_instMonad___redArg(x_112);
x_114 = l_ReaderT_instMonad___redArg(x_113);
return x_114;
}
}
else
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; 
x_115 = lean_ctor_get(x_3, 0);
lean_inc(x_115);
lean_dec(x_3);
x_116 = lean_ctor_get(x_115, 0);
lean_inc_ref(x_116);
x_117 = lean_ctor_get(x_115, 2);
lean_inc(x_117);
x_118 = lean_ctor_get(x_115, 3);
lean_inc(x_118);
x_119 = lean_ctor_get(x_115, 4);
lean_inc(x_119);
if (lean_is_exclusive(x_115)) {
 lean_ctor_release(x_115, 0);
 lean_ctor_release(x_115, 1);
 lean_ctor_release(x_115, 2);
 lean_ctor_release(x_115, 3);
 lean_ctor_release(x_115, 4);
 x_120 = x_115;
} else {
 lean_dec_ref(x_115);
 x_120 = lean_box(0);
}
x_121 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_122 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_116);
x_123 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_123, 0, x_116);
x_124 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_124, 0, x_116);
x_125 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_125, 0, x_123);
lean_ctor_set(x_125, 1, x_124);
x_126 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_126, 0, x_119);
x_127 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_127, 0, x_118);
x_128 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_128, 0, x_117);
if (lean_is_scalar(x_120)) {
 x_129 = lean_alloc_ctor(0, 5, 0);
} else {
 x_129 = x_120;
}
lean_ctor_set(x_129, 0, x_125);
lean_ctor_set(x_129, 1, x_121);
lean_ctor_set(x_129, 2, x_128);
lean_ctor_set(x_129, 3, x_127);
lean_ctor_set(x_129, 4, x_126);
x_130 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_130, 0, x_129);
lean_ctor_set(x_130, 1, x_122);
x_131 = l_ReaderT_instMonad___redArg(x_130);
x_132 = lean_ctor_get(x_131, 0);
lean_inc_ref(x_132);
if (lean_is_exclusive(x_131)) {
 lean_ctor_release(x_131, 0);
 lean_ctor_release(x_131, 1);
 x_133 = x_131;
} else {
 lean_dec_ref(x_131);
 x_133 = lean_box(0);
}
x_134 = lean_ctor_get(x_132, 0);
lean_inc_ref(x_134);
x_135 = lean_ctor_get(x_132, 2);
lean_inc(x_135);
x_136 = lean_ctor_get(x_132, 3);
lean_inc(x_136);
x_137 = lean_ctor_get(x_132, 4);
lean_inc(x_137);
if (lean_is_exclusive(x_132)) {
 lean_ctor_release(x_132, 0);
 lean_ctor_release(x_132, 1);
 lean_ctor_release(x_132, 2);
 lean_ctor_release(x_132, 3);
 lean_ctor_release(x_132, 4);
 x_138 = x_132;
} else {
 lean_dec_ref(x_132);
 x_138 = lean_box(0);
}
x_139 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_140 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_134);
x_141 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_141, 0, x_134);
x_142 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_142, 0, x_134);
x_143 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_143, 0, x_141);
lean_ctor_set(x_143, 1, x_142);
x_144 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_144, 0, x_137);
x_145 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_145, 0, x_136);
x_146 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_146, 0, x_135);
if (lean_is_scalar(x_138)) {
 x_147 = lean_alloc_ctor(0, 5, 0);
} else {
 x_147 = x_138;
}
lean_ctor_set(x_147, 0, x_143);
lean_ctor_set(x_147, 1, x_139);
lean_ctor_set(x_147, 2, x_146);
lean_ctor_set(x_147, 3, x_145);
lean_ctor_set(x_147, 4, x_144);
if (lean_is_scalar(x_133)) {
 x_148 = lean_alloc_ctor(0, 2, 0);
} else {
 x_148 = x_133;
}
lean_ctor_set(x_148, 0, x_147);
lean_ctor_set(x_148, 1, x_140);
x_149 = l_ReaderT_instMonad___redArg(x_148);
x_150 = l_ReaderT_instMonad___redArg(x_149);
x_151 = l_ReaderT_instMonad___redArg(x_150);
x_152 = l_ReaderT_instMonad___redArg(x_151);
return x_152;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonad___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_SearchM_instMonad(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_ReaderT_instMonadFunctor___lam__0), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 3);
lean_closure_set(x_1, 0, lean_box(0));
lean_closure_set(x_1, 1, lean_box(0));
lean_closure_set(x_1, 2, lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = l_Lean_Core_instMonadQuotationCoreM;
x_2 = lp_aesop_Aesop_SearchM_instMonadRef___closed__1;
x_3 = lp_aesop_Aesop_SearchM_instMonadRef___closed__0;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop_Aesop_SearchM_instMonadRef___closed__3;
x_2 = lp_aesop_Aesop_SearchM_instMonadRef___closed__2;
x_3 = lp_aesop_Aesop_SearchM_instMonadRef___closed__0;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop_Aesop_SearchM_instMonadRef___closed__4;
x_2 = lp_aesop_Aesop_SearchM_instMonadRef___closed__1;
x_3 = lp_aesop_Aesop_SearchM_instMonadRef___closed__0;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop_Aesop_SearchM_instMonadRef___closed__5;
x_2 = lp_aesop_Aesop_SearchM_instMonadRef___closed__1;
x_3 = lp_aesop_Aesop_SearchM_instMonadRef___closed__0;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop_Aesop_SearchM_instMonadRef___closed__6;
x_2 = lp_aesop_Aesop_SearchM_instMonadRef___closed__1;
x_3 = lp_aesop_Aesop_SearchM_instMonadRef___closed__0;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop_Aesop_SearchM_instMonadRef___closed__7;
x_2 = lp_aesop_Aesop_SearchM_instMonadRef___closed__2;
x_3 = lp_aesop_Aesop_SearchM_instMonadRef___closed__0;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadRef(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_SearchM_instMonadRef___closed__8;
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadRef___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_SearchM_instMonadRef(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = l_Lean_instMonadExceptOfExceptionCoreM;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = l_Lean_instMonadExceptOfExceptionCoreM;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1;
x_2 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2;
x_2 = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2;
x_2 = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4;
x_2 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("failed", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__6;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; uint8_t x_11; 
x_10 = lp_aesop_Aesop_SearchM_instMonad___closed__1;
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
lean_dec(x_13);
x_14 = !lean_is_exclusive(x_12);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; 
x_15 = lean_ctor_get(x_12, 0);
x_16 = lean_ctor_get(x_12, 2);
x_17 = lean_ctor_get(x_12, 3);
x_18 = lean_ctor_get(x_12, 4);
x_19 = lean_ctor_get(x_12, 1);
lean_dec(x_19);
x_20 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_21 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_15);
x_22 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_22, 0, x_15);
x_23 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_23, 0, x_15);
x_24 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_24, 0, x_22);
lean_ctor_set(x_24, 1, x_23);
x_25 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_25, 0, x_18);
x_26 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_26, 0, x_17);
x_27 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_27, 0, x_16);
lean_ctor_set(x_12, 4, x_25);
lean_ctor_set(x_12, 3, x_26);
lean_ctor_set(x_12, 2, x_27);
lean_ctor_set(x_12, 1, x_20);
lean_ctor_set(x_12, 0, x_24);
lean_ctor_set(x_10, 1, x_21);
x_28 = l_ReaderT_instMonad___redArg(x_10);
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_30 = lean_ctor_get(x_28, 0);
x_31 = lean_ctor_get(x_28, 1);
lean_dec(x_31);
x_32 = !lean_is_exclusive(x_30);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; 
x_33 = lean_ctor_get(x_30, 0);
x_34 = lean_ctor_get(x_30, 2);
x_35 = lean_ctor_get(x_30, 3);
x_36 = lean_ctor_get(x_30, 4);
x_37 = lean_ctor_get(x_30, 1);
lean_dec(x_37);
x_38 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_39 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_33);
x_40 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_40, 0, x_33);
x_41 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_41, 0, x_33);
x_42 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_42, 0, x_40);
lean_ctor_set(x_42, 1, x_41);
x_43 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_43, 0, x_36);
x_44 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_44, 0, x_35);
x_45 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_45, 0, x_34);
lean_ctor_set(x_30, 4, x_43);
lean_ctor_set(x_30, 3, x_44);
lean_ctor_set(x_30, 2, x_45);
lean_ctor_set(x_30, 1, x_38);
lean_ctor_set(x_30, 0, x_42);
lean_ctor_set(x_28, 1, x_39);
x_46 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5;
x_47 = lp_aesop_Aesop_SearchM_instMonadRef___closed__4;
x_48 = lean_ctor_get(x_47, 0);
lean_inc_ref(x_48);
x_49 = l_Lean_Meta_instAddMessageContextMetaM;
lean_inc_ref(x_28);
x_50 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_49, x_28);
x_51 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_51, 0, x_46);
lean_ctor_set(x_51, 1, x_48);
lean_ctor_set(x_51, 2, x_50);
x_52 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7;
x_53 = l_Lean_throwError___redArg(x_28, x_51, x_52);
x_54 = lean_apply_5(x_53, x_5, x_6, x_7, x_8, lean_box(0));
return x_54;
}
else
{
lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; 
x_55 = lean_ctor_get(x_30, 0);
x_56 = lean_ctor_get(x_30, 2);
x_57 = lean_ctor_get(x_30, 3);
x_58 = lean_ctor_get(x_30, 4);
lean_inc(x_58);
lean_inc(x_57);
lean_inc(x_56);
lean_inc(x_55);
lean_dec(x_30);
x_59 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_60 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_55);
x_61 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_61, 0, x_55);
x_62 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_62, 0, x_55);
x_63 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_63, 0, x_61);
lean_ctor_set(x_63, 1, x_62);
x_64 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_64, 0, x_58);
x_65 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_65, 0, x_57);
x_66 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_66, 0, x_56);
x_67 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_67, 0, x_63);
lean_ctor_set(x_67, 1, x_59);
lean_ctor_set(x_67, 2, x_66);
lean_ctor_set(x_67, 3, x_65);
lean_ctor_set(x_67, 4, x_64);
lean_ctor_set(x_28, 1, x_60);
lean_ctor_set(x_28, 0, x_67);
x_68 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5;
x_69 = lp_aesop_Aesop_SearchM_instMonadRef___closed__4;
x_70 = lean_ctor_get(x_69, 0);
lean_inc_ref(x_70);
x_71 = l_Lean_Meta_instAddMessageContextMetaM;
lean_inc_ref(x_28);
x_72 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_71, x_28);
x_73 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_73, 0, x_68);
lean_ctor_set(x_73, 1, x_70);
lean_ctor_set(x_73, 2, x_72);
x_74 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7;
x_75 = l_Lean_throwError___redArg(x_28, x_73, x_74);
x_76 = lean_apply_5(x_75, x_5, x_6, x_7, x_8, lean_box(0));
return x_76;
}
}
else
{
lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; 
x_77 = lean_ctor_get(x_28, 0);
lean_inc(x_77);
lean_dec(x_28);
x_78 = lean_ctor_get(x_77, 0);
lean_inc_ref(x_78);
x_79 = lean_ctor_get(x_77, 2);
lean_inc(x_79);
x_80 = lean_ctor_get(x_77, 3);
lean_inc(x_80);
x_81 = lean_ctor_get(x_77, 4);
lean_inc(x_81);
if (lean_is_exclusive(x_77)) {
 lean_ctor_release(x_77, 0);
 lean_ctor_release(x_77, 1);
 lean_ctor_release(x_77, 2);
 lean_ctor_release(x_77, 3);
 lean_ctor_release(x_77, 4);
 x_82 = x_77;
} else {
 lean_dec_ref(x_77);
 x_82 = lean_box(0);
}
x_83 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_84 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_78);
x_85 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_85, 0, x_78);
x_86 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_86, 0, x_78);
x_87 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_87, 0, x_85);
lean_ctor_set(x_87, 1, x_86);
x_88 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_88, 0, x_81);
x_89 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_89, 0, x_80);
x_90 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_90, 0, x_79);
if (lean_is_scalar(x_82)) {
 x_91 = lean_alloc_ctor(0, 5, 0);
} else {
 x_91 = x_82;
}
lean_ctor_set(x_91, 0, x_87);
lean_ctor_set(x_91, 1, x_83);
lean_ctor_set(x_91, 2, x_90);
lean_ctor_set(x_91, 3, x_89);
lean_ctor_set(x_91, 4, x_88);
x_92 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_92, 0, x_91);
lean_ctor_set(x_92, 1, x_84);
x_93 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5;
x_94 = lp_aesop_Aesop_SearchM_instMonadRef___closed__4;
x_95 = lean_ctor_get(x_94, 0);
lean_inc_ref(x_95);
x_96 = l_Lean_Meta_instAddMessageContextMetaM;
lean_inc_ref(x_92);
x_97 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_96, x_92);
x_98 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_98, 0, x_93);
lean_ctor_set(x_98, 1, x_95);
lean_ctor_set(x_98, 2, x_97);
x_99 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7;
x_100 = l_Lean_throwError___redArg(x_92, x_98, x_99);
x_101 = lean_apply_5(x_100, x_5, x_6, x_7, x_8, lean_box(0));
return x_101;
}
}
else
{
lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; 
x_102 = lean_ctor_get(x_12, 0);
x_103 = lean_ctor_get(x_12, 2);
x_104 = lean_ctor_get(x_12, 3);
x_105 = lean_ctor_get(x_12, 4);
lean_inc(x_105);
lean_inc(x_104);
lean_inc(x_103);
lean_inc(x_102);
lean_dec(x_12);
x_106 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_107 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_102);
x_108 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_108, 0, x_102);
x_109 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_109, 0, x_102);
x_110 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_110, 0, x_108);
lean_ctor_set(x_110, 1, x_109);
x_111 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_111, 0, x_105);
x_112 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_112, 0, x_104);
x_113 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_113, 0, x_103);
x_114 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_114, 0, x_110);
lean_ctor_set(x_114, 1, x_106);
lean_ctor_set(x_114, 2, x_113);
lean_ctor_set(x_114, 3, x_112);
lean_ctor_set(x_114, 4, x_111);
lean_ctor_set(x_10, 1, x_107);
lean_ctor_set(x_10, 0, x_114);
x_115 = l_ReaderT_instMonad___redArg(x_10);
x_116 = lean_ctor_get(x_115, 0);
lean_inc_ref(x_116);
if (lean_is_exclusive(x_115)) {
 lean_ctor_release(x_115, 0);
 lean_ctor_release(x_115, 1);
 x_117 = x_115;
} else {
 lean_dec_ref(x_115);
 x_117 = lean_box(0);
}
x_118 = lean_ctor_get(x_116, 0);
lean_inc_ref(x_118);
x_119 = lean_ctor_get(x_116, 2);
lean_inc(x_119);
x_120 = lean_ctor_get(x_116, 3);
lean_inc(x_120);
x_121 = lean_ctor_get(x_116, 4);
lean_inc(x_121);
if (lean_is_exclusive(x_116)) {
 lean_ctor_release(x_116, 0);
 lean_ctor_release(x_116, 1);
 lean_ctor_release(x_116, 2);
 lean_ctor_release(x_116, 3);
 lean_ctor_release(x_116, 4);
 x_122 = x_116;
} else {
 lean_dec_ref(x_116);
 x_122 = lean_box(0);
}
x_123 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_124 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_118);
x_125 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_125, 0, x_118);
x_126 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_126, 0, x_118);
x_127 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_127, 0, x_125);
lean_ctor_set(x_127, 1, x_126);
x_128 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_128, 0, x_121);
x_129 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_129, 0, x_120);
x_130 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_130, 0, x_119);
if (lean_is_scalar(x_122)) {
 x_131 = lean_alloc_ctor(0, 5, 0);
} else {
 x_131 = x_122;
}
lean_ctor_set(x_131, 0, x_127);
lean_ctor_set(x_131, 1, x_123);
lean_ctor_set(x_131, 2, x_130);
lean_ctor_set(x_131, 3, x_129);
lean_ctor_set(x_131, 4, x_128);
if (lean_is_scalar(x_117)) {
 x_132 = lean_alloc_ctor(0, 2, 0);
} else {
 x_132 = x_117;
}
lean_ctor_set(x_132, 0, x_131);
lean_ctor_set(x_132, 1, x_124);
x_133 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5;
x_134 = lp_aesop_Aesop_SearchM_instMonadRef___closed__4;
x_135 = lean_ctor_get(x_134, 0);
lean_inc_ref(x_135);
x_136 = l_Lean_Meta_instAddMessageContextMetaM;
lean_inc_ref(x_132);
x_137 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_136, x_132);
x_138 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_138, 0, x_133);
lean_ctor_set(x_138, 1, x_135);
lean_ctor_set(x_138, 2, x_137);
x_139 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7;
x_140 = l_Lean_throwError___redArg(x_132, x_138, x_139);
x_141 = lean_apply_5(x_140, x_5, x_6, x_7, x_8, lean_box(0));
return x_141;
}
}
else
{
lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; 
x_142 = lean_ctor_get(x_10, 0);
lean_inc(x_142);
lean_dec(x_10);
x_143 = lean_ctor_get(x_142, 0);
lean_inc_ref(x_143);
x_144 = lean_ctor_get(x_142, 2);
lean_inc(x_144);
x_145 = lean_ctor_get(x_142, 3);
lean_inc(x_145);
x_146 = lean_ctor_get(x_142, 4);
lean_inc(x_146);
if (lean_is_exclusive(x_142)) {
 lean_ctor_release(x_142, 0);
 lean_ctor_release(x_142, 1);
 lean_ctor_release(x_142, 2);
 lean_ctor_release(x_142, 3);
 lean_ctor_release(x_142, 4);
 x_147 = x_142;
} else {
 lean_dec_ref(x_142);
 x_147 = lean_box(0);
}
x_148 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_149 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_143);
x_150 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_150, 0, x_143);
x_151 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_151, 0, x_143);
x_152 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_152, 0, x_150);
lean_ctor_set(x_152, 1, x_151);
x_153 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_153, 0, x_146);
x_154 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_154, 0, x_145);
x_155 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_155, 0, x_144);
if (lean_is_scalar(x_147)) {
 x_156 = lean_alloc_ctor(0, 5, 0);
} else {
 x_156 = x_147;
}
lean_ctor_set(x_156, 0, x_152);
lean_ctor_set(x_156, 1, x_148);
lean_ctor_set(x_156, 2, x_155);
lean_ctor_set(x_156, 3, x_154);
lean_ctor_set(x_156, 4, x_153);
x_157 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_157, 0, x_156);
lean_ctor_set(x_157, 1, x_149);
x_158 = l_ReaderT_instMonad___redArg(x_157);
x_159 = lean_ctor_get(x_158, 0);
lean_inc_ref(x_159);
if (lean_is_exclusive(x_158)) {
 lean_ctor_release(x_158, 0);
 lean_ctor_release(x_158, 1);
 x_160 = x_158;
} else {
 lean_dec_ref(x_158);
 x_160 = lean_box(0);
}
x_161 = lean_ctor_get(x_159, 0);
lean_inc_ref(x_161);
x_162 = lean_ctor_get(x_159, 2);
lean_inc(x_162);
x_163 = lean_ctor_get(x_159, 3);
lean_inc(x_163);
x_164 = lean_ctor_get(x_159, 4);
lean_inc(x_164);
if (lean_is_exclusive(x_159)) {
 lean_ctor_release(x_159, 0);
 lean_ctor_release(x_159, 1);
 lean_ctor_release(x_159, 2);
 lean_ctor_release(x_159, 3);
 lean_ctor_release(x_159, 4);
 x_165 = x_159;
} else {
 lean_dec_ref(x_159);
 x_165 = lean_box(0);
}
x_166 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_167 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_161);
x_168 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_168, 0, x_161);
x_169 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_169, 0, x_161);
x_170 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_170, 0, x_168);
lean_ctor_set(x_170, 1, x_169);
x_171 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_171, 0, x_164);
x_172 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_172, 0, x_163);
x_173 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_173, 0, x_162);
if (lean_is_scalar(x_165)) {
 x_174 = lean_alloc_ctor(0, 5, 0);
} else {
 x_174 = x_165;
}
lean_ctor_set(x_174, 0, x_170);
lean_ctor_set(x_174, 1, x_166);
lean_ctor_set(x_174, 2, x_173);
lean_ctor_set(x_174, 3, x_172);
lean_ctor_set(x_174, 4, x_171);
if (lean_is_scalar(x_160)) {
 x_175 = lean_alloc_ctor(0, 2, 0);
} else {
 x_175 = x_160;
}
lean_ctor_set(x_175, 0, x_174);
lean_ctor_set(x_175, 1, x_167);
x_176 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5;
x_177 = lp_aesop_Aesop_SearchM_instMonadRef___closed__4;
x_178 = lean_ctor_get(x_177, 0);
lean_inc_ref(x_178);
x_179 = l_Lean_Meta_instAddMessageContextMetaM;
lean_inc_ref(x_175);
x_180 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_179, x_175);
x_181 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_181, 0, x_176);
lean_ctor_set(x_181, 1, x_178);
lean_ctor_set(x_181, 2, x_180);
x_182 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7;
x_183 = l_Lean_throwError___redArg(x_175, x_181, x_182);
x_184 = lean_apply_5(x_183, x_5, x_6, x_7, x_8, lean_box(0));
return x_184;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_SearchM_instInhabited___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instInhabited___lam__0___boxed), 9, 0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_SearchM_instInhabited(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_st_ref_get(x_2);
x_11 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_SearchM_instMonadStateState___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_st_ref_set(x_3, x_1);
x_12 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_SearchM_instMonadStateState___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lean_st_ref_take(x_4);
x_13 = lean_apply_1(x_2, x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_13, 1);
lean_inc(x_15);
lean_dec_ref(x_13);
x_16 = lean_st_ref_set(x_4, x_15);
x_17 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_17, 0, x_14);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_SearchM_instMonadStateState___lam__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadStateState___lam__0___boxed), 9, 0);
x_4 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadStateState___lam__1___boxed), 10, 0);
x_5 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadStateState___lam__2___boxed), 11, 0);
x_6 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_6, 0, x_3);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_SearchM_instMonadStateState(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadReaderContext(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lp_aesop_Aesop_SearchM_instMonad___closed__1;
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_dec(x_6);
x_7 = !lean_is_exclusive(x_5);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; uint8_t x_22; 
x_8 = lean_ctor_get(x_5, 0);
x_9 = lean_ctor_get(x_5, 2);
x_10 = lean_ctor_get(x_5, 3);
x_11 = lean_ctor_get(x_5, 4);
x_12 = lean_ctor_get(x_5, 1);
lean_dec(x_12);
x_13 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_14 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_8);
x_15 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_15, 0, x_8);
x_16 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_16, 0, x_8);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
x_18 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_18, 0, x_11);
x_19 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_19, 0, x_10);
x_20 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_20, 0, x_9);
lean_ctor_set(x_5, 4, x_18);
lean_ctor_set(x_5, 3, x_19);
lean_ctor_set(x_5, 2, x_20);
lean_ctor_set(x_5, 1, x_13);
lean_ctor_set(x_5, 0, x_17);
lean_ctor_set(x_3, 1, x_14);
x_21 = l_ReaderT_instMonad___redArg(x_3);
x_22 = !lean_is_exclusive(x_21);
if (x_22 == 0)
{
lean_object* x_23; lean_object* x_24; uint8_t x_25; 
x_23 = lean_ctor_get(x_21, 0);
x_24 = lean_ctor_get(x_21, 1);
lean_dec(x_24);
x_25 = !lean_is_exclusive(x_23);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_26 = lean_ctor_get(x_23, 0);
x_27 = lean_ctor_get(x_23, 2);
x_28 = lean_ctor_get(x_23, 3);
x_29 = lean_ctor_get(x_23, 4);
x_30 = lean_ctor_get(x_23, 1);
lean_dec(x_30);
x_31 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_32 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_26);
x_33 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_33, 0, x_26);
x_34 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_34, 0, x_26);
x_35 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_35, 0, x_33);
lean_ctor_set(x_35, 1, x_34);
x_36 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_36, 0, x_29);
x_37 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_37, 0, x_28);
x_38 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_38, 0, x_27);
lean_ctor_set(x_23, 4, x_36);
lean_ctor_set(x_23, 3, x_37);
lean_ctor_set(x_23, 2, x_38);
lean_ctor_set(x_23, 1, x_31);
lean_ctor_set(x_23, 0, x_35);
lean_ctor_set(x_21, 1, x_32);
x_39 = l_ReaderT_instMonad___redArg(x_21);
x_40 = l_ReaderT_instMonad___redArg(x_39);
x_41 = l_ReaderT_instMonad___redArg(x_40);
x_42 = lean_alloc_closure((void*)(l_ReaderT_read), 4, 3);
lean_closure_set(x_42, 0, lean_box(0));
lean_closure_set(x_42, 1, lean_box(0));
lean_closure_set(x_42, 2, x_41);
return x_42;
}
else
{
lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; 
x_43 = lean_ctor_get(x_23, 0);
x_44 = lean_ctor_get(x_23, 2);
x_45 = lean_ctor_get(x_23, 3);
x_46 = lean_ctor_get(x_23, 4);
lean_inc(x_46);
lean_inc(x_45);
lean_inc(x_44);
lean_inc(x_43);
lean_dec(x_23);
x_47 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_48 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_43);
x_49 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_49, 0, x_43);
x_50 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_50, 0, x_43);
x_51 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_51, 0, x_49);
lean_ctor_set(x_51, 1, x_50);
x_52 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_52, 0, x_46);
x_53 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_53, 0, x_45);
x_54 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_54, 0, x_44);
x_55 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_55, 0, x_51);
lean_ctor_set(x_55, 1, x_47);
lean_ctor_set(x_55, 2, x_54);
lean_ctor_set(x_55, 3, x_53);
lean_ctor_set(x_55, 4, x_52);
lean_ctor_set(x_21, 1, x_48);
lean_ctor_set(x_21, 0, x_55);
x_56 = l_ReaderT_instMonad___redArg(x_21);
x_57 = l_ReaderT_instMonad___redArg(x_56);
x_58 = l_ReaderT_instMonad___redArg(x_57);
x_59 = lean_alloc_closure((void*)(l_ReaderT_read), 4, 3);
lean_closure_set(x_59, 0, lean_box(0));
lean_closure_set(x_59, 1, lean_box(0));
lean_closure_set(x_59, 2, x_58);
return x_59;
}
}
else
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; 
x_60 = lean_ctor_get(x_21, 0);
lean_inc(x_60);
lean_dec(x_21);
x_61 = lean_ctor_get(x_60, 0);
lean_inc_ref(x_61);
x_62 = lean_ctor_get(x_60, 2);
lean_inc(x_62);
x_63 = lean_ctor_get(x_60, 3);
lean_inc(x_63);
x_64 = lean_ctor_get(x_60, 4);
lean_inc(x_64);
if (lean_is_exclusive(x_60)) {
 lean_ctor_release(x_60, 0);
 lean_ctor_release(x_60, 1);
 lean_ctor_release(x_60, 2);
 lean_ctor_release(x_60, 3);
 lean_ctor_release(x_60, 4);
 x_65 = x_60;
} else {
 lean_dec_ref(x_60);
 x_65 = lean_box(0);
}
x_66 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_67 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_61);
x_68 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_68, 0, x_61);
x_69 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_69, 0, x_61);
x_70 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_70, 0, x_68);
lean_ctor_set(x_70, 1, x_69);
x_71 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_71, 0, x_64);
x_72 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_72, 0, x_63);
x_73 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_73, 0, x_62);
if (lean_is_scalar(x_65)) {
 x_74 = lean_alloc_ctor(0, 5, 0);
} else {
 x_74 = x_65;
}
lean_ctor_set(x_74, 0, x_70);
lean_ctor_set(x_74, 1, x_66);
lean_ctor_set(x_74, 2, x_73);
lean_ctor_set(x_74, 3, x_72);
lean_ctor_set(x_74, 4, x_71);
x_75 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_75, 0, x_74);
lean_ctor_set(x_75, 1, x_67);
x_76 = l_ReaderT_instMonad___redArg(x_75);
x_77 = l_ReaderT_instMonad___redArg(x_76);
x_78 = l_ReaderT_instMonad___redArg(x_77);
x_79 = lean_alloc_closure((void*)(l_ReaderT_read), 4, 3);
lean_closure_set(x_79, 0, lean_box(0));
lean_closure_set(x_79, 1, lean_box(0));
lean_closure_set(x_79, 2, x_78);
return x_79;
}
}
else
{
lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; 
x_80 = lean_ctor_get(x_5, 0);
x_81 = lean_ctor_get(x_5, 2);
x_82 = lean_ctor_get(x_5, 3);
x_83 = lean_ctor_get(x_5, 4);
lean_inc(x_83);
lean_inc(x_82);
lean_inc(x_81);
lean_inc(x_80);
lean_dec(x_5);
x_84 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_85 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_80);
x_86 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_86, 0, x_80);
x_87 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_87, 0, x_80);
x_88 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_88, 0, x_86);
lean_ctor_set(x_88, 1, x_87);
x_89 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_89, 0, x_83);
x_90 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_90, 0, x_82);
x_91 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_91, 0, x_81);
x_92 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_92, 0, x_88);
lean_ctor_set(x_92, 1, x_84);
lean_ctor_set(x_92, 2, x_91);
lean_ctor_set(x_92, 3, x_90);
lean_ctor_set(x_92, 4, x_89);
lean_ctor_set(x_3, 1, x_85);
lean_ctor_set(x_3, 0, x_92);
x_93 = l_ReaderT_instMonad___redArg(x_3);
x_94 = lean_ctor_get(x_93, 0);
lean_inc_ref(x_94);
if (lean_is_exclusive(x_93)) {
 lean_ctor_release(x_93, 0);
 lean_ctor_release(x_93, 1);
 x_95 = x_93;
} else {
 lean_dec_ref(x_93);
 x_95 = lean_box(0);
}
x_96 = lean_ctor_get(x_94, 0);
lean_inc_ref(x_96);
x_97 = lean_ctor_get(x_94, 2);
lean_inc(x_97);
x_98 = lean_ctor_get(x_94, 3);
lean_inc(x_98);
x_99 = lean_ctor_get(x_94, 4);
lean_inc(x_99);
if (lean_is_exclusive(x_94)) {
 lean_ctor_release(x_94, 0);
 lean_ctor_release(x_94, 1);
 lean_ctor_release(x_94, 2);
 lean_ctor_release(x_94, 3);
 lean_ctor_release(x_94, 4);
 x_100 = x_94;
} else {
 lean_dec_ref(x_94);
 x_100 = lean_box(0);
}
x_101 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_102 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_96);
x_103 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_103, 0, x_96);
x_104 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_104, 0, x_96);
x_105 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_105, 0, x_103);
lean_ctor_set(x_105, 1, x_104);
x_106 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_106, 0, x_99);
x_107 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_107, 0, x_98);
x_108 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_108, 0, x_97);
if (lean_is_scalar(x_100)) {
 x_109 = lean_alloc_ctor(0, 5, 0);
} else {
 x_109 = x_100;
}
lean_ctor_set(x_109, 0, x_105);
lean_ctor_set(x_109, 1, x_101);
lean_ctor_set(x_109, 2, x_108);
lean_ctor_set(x_109, 3, x_107);
lean_ctor_set(x_109, 4, x_106);
if (lean_is_scalar(x_95)) {
 x_110 = lean_alloc_ctor(0, 2, 0);
} else {
 x_110 = x_95;
}
lean_ctor_set(x_110, 0, x_109);
lean_ctor_set(x_110, 1, x_102);
x_111 = l_ReaderT_instMonad___redArg(x_110);
x_112 = l_ReaderT_instMonad___redArg(x_111);
x_113 = l_ReaderT_instMonad___redArg(x_112);
x_114 = lean_alloc_closure((void*)(l_ReaderT_read), 4, 3);
lean_closure_set(x_114, 0, lean_box(0));
lean_closure_set(x_114, 1, lean_box(0));
lean_closure_set(x_114, 2, x_113);
return x_114;
}
}
else
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; 
x_115 = lean_ctor_get(x_3, 0);
lean_inc(x_115);
lean_dec(x_3);
x_116 = lean_ctor_get(x_115, 0);
lean_inc_ref(x_116);
x_117 = lean_ctor_get(x_115, 2);
lean_inc(x_117);
x_118 = lean_ctor_get(x_115, 3);
lean_inc(x_118);
x_119 = lean_ctor_get(x_115, 4);
lean_inc(x_119);
if (lean_is_exclusive(x_115)) {
 lean_ctor_release(x_115, 0);
 lean_ctor_release(x_115, 1);
 lean_ctor_release(x_115, 2);
 lean_ctor_release(x_115, 3);
 lean_ctor_release(x_115, 4);
 x_120 = x_115;
} else {
 lean_dec_ref(x_115);
 x_120 = lean_box(0);
}
x_121 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_122 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_116);
x_123 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_123, 0, x_116);
x_124 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_124, 0, x_116);
x_125 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_125, 0, x_123);
lean_ctor_set(x_125, 1, x_124);
x_126 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_126, 0, x_119);
x_127 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_127, 0, x_118);
x_128 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_128, 0, x_117);
if (lean_is_scalar(x_120)) {
 x_129 = lean_alloc_ctor(0, 5, 0);
} else {
 x_129 = x_120;
}
lean_ctor_set(x_129, 0, x_125);
lean_ctor_set(x_129, 1, x_121);
lean_ctor_set(x_129, 2, x_128);
lean_ctor_set(x_129, 3, x_127);
lean_ctor_set(x_129, 4, x_126);
x_130 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_130, 0, x_129);
lean_ctor_set(x_130, 1, x_122);
x_131 = l_ReaderT_instMonad___redArg(x_130);
x_132 = lean_ctor_get(x_131, 0);
lean_inc_ref(x_132);
if (lean_is_exclusive(x_131)) {
 lean_ctor_release(x_131, 0);
 lean_ctor_release(x_131, 1);
 x_133 = x_131;
} else {
 lean_dec_ref(x_131);
 x_133 = lean_box(0);
}
x_134 = lean_ctor_get(x_132, 0);
lean_inc_ref(x_134);
x_135 = lean_ctor_get(x_132, 2);
lean_inc(x_135);
x_136 = lean_ctor_get(x_132, 3);
lean_inc(x_136);
x_137 = lean_ctor_get(x_132, 4);
lean_inc(x_137);
if (lean_is_exclusive(x_132)) {
 lean_ctor_release(x_132, 0);
 lean_ctor_release(x_132, 1);
 lean_ctor_release(x_132, 2);
 lean_ctor_release(x_132, 3);
 lean_ctor_release(x_132, 4);
 x_138 = x_132;
} else {
 lean_dec_ref(x_132);
 x_138 = lean_box(0);
}
x_139 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_140 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_134);
x_141 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_141, 0, x_134);
x_142 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_142, 0, x_134);
x_143 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_143, 0, x_141);
lean_ctor_set(x_143, 1, x_142);
x_144 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_144, 0, x_137);
x_145 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_145, 0, x_136);
x_146 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_146, 0, x_135);
if (lean_is_scalar(x_138)) {
 x_147 = lean_alloc_ctor(0, 5, 0);
} else {
 x_147 = x_138;
}
lean_ctor_set(x_147, 0, x_143);
lean_ctor_set(x_147, 1, x_139);
lean_ctor_set(x_147, 2, x_146);
lean_ctor_set(x_147, 3, x_145);
lean_ctor_set(x_147, 4, x_144);
if (lean_is_scalar(x_133)) {
 x_148 = lean_alloc_ctor(0, 2, 0);
} else {
 x_148 = x_133;
}
lean_ctor_set(x_148, 0, x_147);
lean_ctor_set(x_148, 1, x_140);
x_149 = l_ReaderT_instMonad___redArg(x_148);
x_150 = l_ReaderT_instMonad___redArg(x_149);
x_151 = l_ReaderT_instMonad___redArg(x_150);
x_152 = lean_alloc_closure((void*)(l_ReaderT_read), 4, 3);
lean_closure_set(x_152, 0, lean_box(0));
lean_closure_set(x_152, 1, lean_box(0));
lean_closure_set(x_152, 2, x_151);
return x_152;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadReaderContext___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_SearchM_instMonadReaderContext(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_12 = lean_st_ref_get(x_4);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec(x_12);
x_14 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_14);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
x_16 = lean_apply_8(x_2, x_15, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
return x_16;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_SearchM_instMonadLiftTreeM(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_11 = lean_st_mk_ref(x_3);
x_12 = lean_st_mk_ref(x_2);
lean_inc(x_11);
lean_inc(x_12);
x_13 = lean_apply_9(x_4, x_1, x_12, x_11, x_5, x_6, x_7, x_8, x_9, lean_box(0));
if (lean_obj_tag(x_13) == 0)
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_13);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_13, 0);
x_16 = lean_st_ref_get(x_12);
lean_dec(x_12);
x_17 = lean_st_ref_get(x_11);
lean_dec(x_11);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_15);
lean_ctor_set(x_19, 1, x_18);
lean_ctor_set(x_13, 0, x_19);
return x_13;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_20 = lean_ctor_get(x_13, 0);
lean_inc(x_20);
lean_dec(x_13);
x_21 = lean_st_ref_get(x_12);
lean_dec(x_12);
x_22 = lean_st_ref_get(x_11);
lean_dec(x_11);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_22);
x_24 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_24, 0, x_20);
lean_ctor_set(x_24, 1, x_23);
x_25 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_25, 0, x_24);
return x_25;
}
}
else
{
uint8_t x_26; 
lean_dec(x_12);
lean_dec(x_11);
x_26 = !lean_is_exclusive(x_13);
if (x_26 == 0)
{
return x_13;
}
else
{
lean_object* x_27; lean_object* x_28; 
x_27 = lean_ctor_get(x_13, 0);
lean_inc(x_27);
lean_dec(x_13);
x_28 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_28, 0, x_27);
return x_28;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; 
x_14 = lp_aesop_Aesop_SearchM_run_x27___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_aesop_Aesop_SearchM_run_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_2);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_SearchM_run_x27___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_SearchM_run___redArg___closed__1;
x_2 = lp_aesop_Aesop_SearchM_run___redArg___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = l_Lean_Meta_instAddMessageContextMetaM;
x_2 = lp_aesop_Aesop_SearchM_instMonadRef___closed__1;
x_3 = l_Lean_instAddMessageContextOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_SearchM_run___redArg___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__6), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__5___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__4___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__3), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__2___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__1___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__0), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_SearchM_run___redArg___closed__5;
x_2 = lp_aesop_Aesop_SearchM_run___redArg___closed__4;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_aesop_Aesop_SearchM_run___redArg___closed__9;
x_2 = lp_aesop_Aesop_SearchM_run___redArg___closed__8;
x_3 = lp_aesop_Aesop_SearchM_run___redArg___closed__7;
x_4 = lp_aesop_Aesop_SearchM_run___redArg___closed__6;
x_5 = lp_aesop_Aesop_SearchM_run___redArg___closed__11;
x_6 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_3);
lean_ctor_set(x_6, 3, x_2);
lean_ctor_set(x_6, 4, x_1);
return x_6;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_SearchM_run___redArg___closed__10;
x_2 = lp_aesop_Aesop_SearchM_run___redArg___closed__12;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_treeImpl;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop: internal error: root mvar cluster does not contain exactly one goal.", 75, 75);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_SearchM_run___redArg___closed__15;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__1___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_SearchM_run___redArg___lam__1(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; uint8_t x_15; 
x_14 = lp_aesop_Aesop_SearchM_instMonad___closed__1;
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; uint8_t x_18; 
x_16 = lean_ctor_get(x_14, 0);
x_17 = lean_ctor_get(x_14, 1);
lean_dec(x_17);
x_18 = !lean_is_exclusive(x_16);
if (x_18 == 0)
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_19 = lean_ctor_get(x_16, 0);
x_20 = lean_ctor_get(x_16, 2);
x_21 = lean_ctor_get(x_16, 3);
x_22 = lean_ctor_get(x_16, 4);
x_23 = lean_ctor_get(x_16, 1);
lean_dec(x_23);
x_24 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_25 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_19);
x_26 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_26, 0, x_19);
x_27 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_27, 0, x_19);
x_28 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_28, 0, x_26);
lean_ctor_set(x_28, 1, x_27);
x_29 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_29, 0, x_22);
x_30 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_30, 0, x_21);
x_31 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_31, 0, x_20);
lean_ctor_set(x_16, 4, x_29);
lean_ctor_set(x_16, 3, x_30);
lean_ctor_set(x_16, 2, x_31);
lean_ctor_set(x_16, 1, x_24);
lean_ctor_set(x_16, 0, x_28);
lean_ctor_set(x_14, 1, x_25);
x_32 = l_ReaderT_instMonad___redArg(x_14);
x_33 = !lean_is_exclusive(x_32);
if (x_33 == 0)
{
lean_object* x_34; lean_object* x_35; uint8_t x_36; 
x_34 = lean_ctor_get(x_32, 0);
x_35 = lean_ctor_get(x_32, 1);
lean_dec(x_35);
x_36 = !lean_is_exclusive(x_34);
if (x_36 == 0)
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; 
x_37 = lean_ctor_get(x_34, 0);
x_38 = lean_ctor_get(x_34, 2);
x_39 = lean_ctor_get(x_34, 3);
x_40 = lean_ctor_get(x_34, 4);
x_41 = lean_ctor_get(x_34, 1);
lean_dec(x_41);
x_42 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_43 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_37);
x_44 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_44, 0, x_37);
x_45 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_45, 0, x_37);
x_46 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_46, 0, x_44);
lean_ctor_set(x_46, 1, x_45);
x_47 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_47, 0, x_40);
x_48 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_48, 0, x_39);
x_49 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_49, 0, x_38);
lean_ctor_set(x_34, 4, x_47);
lean_ctor_set(x_34, 3, x_48);
lean_ctor_set(x_34, 2, x_49);
lean_ctor_set(x_34, 1, x_42);
lean_ctor_set(x_34, 0, x_46);
lean_ctor_set(x_32, 1, x_43);
x_50 = l_ReaderT_instMonad___redArg(x_32);
x_51 = lp_aesop_Aesop_SearchM_run___redArg___closed__2;
x_52 = lp_aesop_Aesop_SearchM_instMonadRef___closed__5;
x_53 = lean_ctor_get(x_52, 0);
lean_inc_ref(x_53);
x_54 = lp_aesop_Aesop_SearchM_run___redArg___closed__3;
lean_inc_ref(x_50);
x_55 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_54, x_50);
x_56 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_56, 0, x_51);
lean_ctor_set(x_56, 1, x_53);
lean_ctor_set(x_56, 2, x_55);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_2);
x_57 = lp_aesop_Aesop_mkInitialTree(x_6, x_2, x_8, x_9, x_10, x_11, x_12);
if (lean_obj_tag(x_57) == 0)
{
lean_object* x_58; lean_object* x_59; 
x_58 = lean_ctor_get(x_57, 0);
lean_inc(x_58);
lean_dec_ref(x_57);
x_59 = l_Lean_Meta_getSimpCongrTheorems___redArg(x_12);
if (lean_obj_tag(x_59) == 0)
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; size_t x_65; size_t x_66; lean_object* x_67; lean_object* x_68; 
x_60 = lean_ctor_get(x_59, 0);
lean_inc(x_60);
lean_dec_ref(x_59);
x_61 = lean_ctor_get(x_2, 1);
x_62 = lean_ctor_get(x_2, 2);
x_63 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_run___redArg___lam__0___boxed), 1, 0);
x_64 = lp_aesop_Aesop_SearchM_run___redArg___closed__13;
x_65 = lean_array_size(x_61);
x_66 = 0;
lean_inc_ref(x_61);
x_67 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_64, x_63, x_65, x_66, x_61);
x_68 = l_Lean_Meta_Simp_mkContext___redArg(x_4, x_67, x_60, x_9, x_12);
if (lean_obj_tag(x_68) == 0)
{
lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; uint8_t x_78; 
x_69 = lean_ctor_get(x_68, 0);
lean_inc(x_69);
lean_dec_ref(x_68);
x_70 = lean_ctor_get(x_58, 0);
x_71 = lean_st_ref_get(x_70);
x_72 = lp_aesop_Aesop_SearchM_run___redArg___closed__14;
x_73 = lean_ctor_get(x_72, 5);
lean_inc_ref(x_73);
x_74 = lean_apply_1(x_73, x_71);
x_75 = lean_ctor_get(x_74, 1);
lean_inc_ref(x_75);
lean_dec_ref(x_74);
x_76 = lean_array_get_size(x_75);
x_77 = lean_unsigned_to_nat(1u);
x_78 = lean_nat_dec_eq(x_76, x_77);
if (x_78 == 0)
{
lean_object* x_79; lean_object* x_80; lean_object* x_81; 
lean_dec_ref(x_75);
lean_dec(x_69);
lean_dec(x_58);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_79 = lp_aesop_Aesop_SearchM_run___redArg___closed__16;
x_80 = l_Lean_throwError___redArg(x_50, x_56, x_79);
x_81 = lean_apply_6(x_80, x_8, x_9, x_10, x_11, x_12, lean_box(0));
return x_81;
}
else
{
lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; uint8_t x_88; uint8_t x_89; lean_object* x_90; size_t x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; uint8_t x_95; lean_object* x_96; lean_object* x_97; 
lean_dec_ref(x_56);
lean_dec_ref(x_50);
x_82 = lean_unsigned_to_nat(0u);
x_83 = lean_array_fget(x_75, x_82);
lean_dec_ref(x_75);
x_84 = lp_aesop_Aesop_SearchM_run___redArg___closed__17;
x_85 = lean_array_push(x_84, x_83);
x_86 = lp_aesop_Aesop_Queue_init_x27___redArg(x_1, x_85);
x_87 = lean_ctor_get(x_3, 0);
x_88 = lean_ctor_get_uint8(x_87, sizeof(void*)*6 + 7);
x_89 = lean_ctor_get_uint8(x_87, sizeof(void*)*6 + 8);
x_90 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_run___redArg___lam__1___boxed), 1, 0);
x_91 = lean_array_size(x_62);
lean_inc_ref(x_62);
x_92 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_64, x_90, x_91, x_66, x_62);
x_93 = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(x_93, 0, x_69);
lean_ctor_set(x_93, 1, x_5);
lean_ctor_set(x_93, 2, x_92);
lean_ctor_set_uint8(x_93, sizeof(void*)*3, x_88);
lean_ctor_set_uint8(x_93, sizeof(void*)*3 + 1, x_89);
x_94 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_94, 0, x_2);
lean_ctor_set(x_94, 1, x_93);
lean_ctor_set(x_94, 2, x_3);
x_95 = 0;
x_96 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_96, 0, x_77);
lean_ctor_set(x_96, 1, x_86);
lean_ctor_set_uint8(x_96, sizeof(void*)*2, x_95);
x_97 = lp_aesop_Aesop_SearchM_run_x27___redArg(x_94, x_96, x_58, x_7, x_8, x_9, x_10, x_11, x_12);
return x_97;
}
}
else
{
uint8_t x_98; 
lean_dec(x_58);
lean_dec_ref(x_56);
lean_dec_ref(x_50);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_98 = !lean_is_exclusive(x_68);
if (x_98 == 0)
{
return x_68;
}
else
{
lean_object* x_99; lean_object* x_100; 
x_99 = lean_ctor_get(x_68, 0);
lean_inc(x_99);
lean_dec(x_68);
x_100 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_100, 0, x_99);
return x_100;
}
}
}
else
{
uint8_t x_101; 
lean_dec(x_58);
lean_dec_ref(x_56);
lean_dec_ref(x_50);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_101 = !lean_is_exclusive(x_59);
if (x_101 == 0)
{
return x_59;
}
else
{
lean_object* x_102; lean_object* x_103; 
x_102 = lean_ctor_get(x_59, 0);
lean_inc(x_102);
lean_dec(x_59);
x_103 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_103, 0, x_102);
return x_103;
}
}
}
else
{
uint8_t x_104; 
lean_dec_ref(x_56);
lean_dec_ref(x_50);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_104 = !lean_is_exclusive(x_57);
if (x_104 == 0)
{
return x_57;
}
else
{
lean_object* x_105; lean_object* x_106; 
x_105 = lean_ctor_get(x_57, 0);
lean_inc(x_105);
lean_dec(x_57);
x_106 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_106, 0, x_105);
return x_106;
}
}
}
else
{
lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; 
x_107 = lean_ctor_get(x_34, 0);
x_108 = lean_ctor_get(x_34, 2);
x_109 = lean_ctor_get(x_34, 3);
x_110 = lean_ctor_get(x_34, 4);
lean_inc(x_110);
lean_inc(x_109);
lean_inc(x_108);
lean_inc(x_107);
lean_dec(x_34);
x_111 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_112 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_107);
x_113 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_113, 0, x_107);
x_114 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_114, 0, x_107);
x_115 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_115, 0, x_113);
lean_ctor_set(x_115, 1, x_114);
x_116 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_116, 0, x_110);
x_117 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_117, 0, x_109);
x_118 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_118, 0, x_108);
x_119 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_119, 0, x_115);
lean_ctor_set(x_119, 1, x_111);
lean_ctor_set(x_119, 2, x_118);
lean_ctor_set(x_119, 3, x_117);
lean_ctor_set(x_119, 4, x_116);
lean_ctor_set(x_32, 1, x_112);
lean_ctor_set(x_32, 0, x_119);
x_120 = l_ReaderT_instMonad___redArg(x_32);
x_121 = lp_aesop_Aesop_SearchM_run___redArg___closed__2;
x_122 = lp_aesop_Aesop_SearchM_instMonadRef___closed__5;
x_123 = lean_ctor_get(x_122, 0);
lean_inc_ref(x_123);
x_124 = lp_aesop_Aesop_SearchM_run___redArg___closed__3;
lean_inc_ref(x_120);
x_125 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_124, x_120);
x_126 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_126, 0, x_121);
lean_ctor_set(x_126, 1, x_123);
lean_ctor_set(x_126, 2, x_125);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_2);
x_127 = lp_aesop_Aesop_mkInitialTree(x_6, x_2, x_8, x_9, x_10, x_11, x_12);
if (lean_obj_tag(x_127) == 0)
{
lean_object* x_128; lean_object* x_129; 
x_128 = lean_ctor_get(x_127, 0);
lean_inc(x_128);
lean_dec_ref(x_127);
x_129 = l_Lean_Meta_getSimpCongrTheorems___redArg(x_12);
if (lean_obj_tag(x_129) == 0)
{
lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; size_t x_135; size_t x_136; lean_object* x_137; lean_object* x_138; 
x_130 = lean_ctor_get(x_129, 0);
lean_inc(x_130);
lean_dec_ref(x_129);
x_131 = lean_ctor_get(x_2, 1);
x_132 = lean_ctor_get(x_2, 2);
x_133 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_run___redArg___lam__0___boxed), 1, 0);
x_134 = lp_aesop_Aesop_SearchM_run___redArg___closed__13;
x_135 = lean_array_size(x_131);
x_136 = 0;
lean_inc_ref(x_131);
x_137 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_134, x_133, x_135, x_136, x_131);
x_138 = l_Lean_Meta_Simp_mkContext___redArg(x_4, x_137, x_130, x_9, x_12);
if (lean_obj_tag(x_138) == 0)
{
lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; uint8_t x_148; 
x_139 = lean_ctor_get(x_138, 0);
lean_inc(x_139);
lean_dec_ref(x_138);
x_140 = lean_ctor_get(x_128, 0);
x_141 = lean_st_ref_get(x_140);
x_142 = lp_aesop_Aesop_SearchM_run___redArg___closed__14;
x_143 = lean_ctor_get(x_142, 5);
lean_inc_ref(x_143);
x_144 = lean_apply_1(x_143, x_141);
x_145 = lean_ctor_get(x_144, 1);
lean_inc_ref(x_145);
lean_dec_ref(x_144);
x_146 = lean_array_get_size(x_145);
x_147 = lean_unsigned_to_nat(1u);
x_148 = lean_nat_dec_eq(x_146, x_147);
if (x_148 == 0)
{
lean_object* x_149; lean_object* x_150; lean_object* x_151; 
lean_dec_ref(x_145);
lean_dec(x_139);
lean_dec(x_128);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_149 = lp_aesop_Aesop_SearchM_run___redArg___closed__16;
x_150 = l_Lean_throwError___redArg(x_120, x_126, x_149);
x_151 = lean_apply_6(x_150, x_8, x_9, x_10, x_11, x_12, lean_box(0));
return x_151;
}
else
{
lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; uint8_t x_158; uint8_t x_159; lean_object* x_160; size_t x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; uint8_t x_165; lean_object* x_166; lean_object* x_167; 
lean_dec_ref(x_126);
lean_dec_ref(x_120);
x_152 = lean_unsigned_to_nat(0u);
x_153 = lean_array_fget(x_145, x_152);
lean_dec_ref(x_145);
x_154 = lp_aesop_Aesop_SearchM_run___redArg___closed__17;
x_155 = lean_array_push(x_154, x_153);
x_156 = lp_aesop_Aesop_Queue_init_x27___redArg(x_1, x_155);
x_157 = lean_ctor_get(x_3, 0);
x_158 = lean_ctor_get_uint8(x_157, sizeof(void*)*6 + 7);
x_159 = lean_ctor_get_uint8(x_157, sizeof(void*)*6 + 8);
x_160 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_run___redArg___lam__1___boxed), 1, 0);
x_161 = lean_array_size(x_132);
lean_inc_ref(x_132);
x_162 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_134, x_160, x_161, x_136, x_132);
x_163 = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(x_163, 0, x_139);
lean_ctor_set(x_163, 1, x_5);
lean_ctor_set(x_163, 2, x_162);
lean_ctor_set_uint8(x_163, sizeof(void*)*3, x_158);
lean_ctor_set_uint8(x_163, sizeof(void*)*3 + 1, x_159);
x_164 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_164, 0, x_2);
lean_ctor_set(x_164, 1, x_163);
lean_ctor_set(x_164, 2, x_3);
x_165 = 0;
x_166 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_166, 0, x_147);
lean_ctor_set(x_166, 1, x_156);
lean_ctor_set_uint8(x_166, sizeof(void*)*2, x_165);
x_167 = lp_aesop_Aesop_SearchM_run_x27___redArg(x_164, x_166, x_128, x_7, x_8, x_9, x_10, x_11, x_12);
return x_167;
}
}
else
{
lean_object* x_168; lean_object* x_169; lean_object* x_170; 
lean_dec(x_128);
lean_dec_ref(x_126);
lean_dec_ref(x_120);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_168 = lean_ctor_get(x_138, 0);
lean_inc(x_168);
if (lean_is_exclusive(x_138)) {
 lean_ctor_release(x_138, 0);
 x_169 = x_138;
} else {
 lean_dec_ref(x_138);
 x_169 = lean_box(0);
}
if (lean_is_scalar(x_169)) {
 x_170 = lean_alloc_ctor(1, 1, 0);
} else {
 x_170 = x_169;
}
lean_ctor_set(x_170, 0, x_168);
return x_170;
}
}
else
{
lean_object* x_171; lean_object* x_172; lean_object* x_173; 
lean_dec(x_128);
lean_dec_ref(x_126);
lean_dec_ref(x_120);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_171 = lean_ctor_get(x_129, 0);
lean_inc(x_171);
if (lean_is_exclusive(x_129)) {
 lean_ctor_release(x_129, 0);
 x_172 = x_129;
} else {
 lean_dec_ref(x_129);
 x_172 = lean_box(0);
}
if (lean_is_scalar(x_172)) {
 x_173 = lean_alloc_ctor(1, 1, 0);
} else {
 x_173 = x_172;
}
lean_ctor_set(x_173, 0, x_171);
return x_173;
}
}
else
{
lean_object* x_174; lean_object* x_175; lean_object* x_176; 
lean_dec_ref(x_126);
lean_dec_ref(x_120);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_174 = lean_ctor_get(x_127, 0);
lean_inc(x_174);
if (lean_is_exclusive(x_127)) {
 lean_ctor_release(x_127, 0);
 x_175 = x_127;
} else {
 lean_dec_ref(x_127);
 x_175 = lean_box(0);
}
if (lean_is_scalar(x_175)) {
 x_176 = lean_alloc_ctor(1, 1, 0);
} else {
 x_176 = x_175;
}
lean_ctor_set(x_176, 0, x_174);
return x_176;
}
}
}
else
{
lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; 
x_177 = lean_ctor_get(x_32, 0);
lean_inc(x_177);
lean_dec(x_32);
x_178 = lean_ctor_get(x_177, 0);
lean_inc_ref(x_178);
x_179 = lean_ctor_get(x_177, 2);
lean_inc(x_179);
x_180 = lean_ctor_get(x_177, 3);
lean_inc(x_180);
x_181 = lean_ctor_get(x_177, 4);
lean_inc(x_181);
if (lean_is_exclusive(x_177)) {
 lean_ctor_release(x_177, 0);
 lean_ctor_release(x_177, 1);
 lean_ctor_release(x_177, 2);
 lean_ctor_release(x_177, 3);
 lean_ctor_release(x_177, 4);
 x_182 = x_177;
} else {
 lean_dec_ref(x_177);
 x_182 = lean_box(0);
}
x_183 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_184 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_178);
x_185 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_185, 0, x_178);
x_186 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_186, 0, x_178);
x_187 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_187, 0, x_185);
lean_ctor_set(x_187, 1, x_186);
x_188 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_188, 0, x_181);
x_189 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_189, 0, x_180);
x_190 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_190, 0, x_179);
if (lean_is_scalar(x_182)) {
 x_191 = lean_alloc_ctor(0, 5, 0);
} else {
 x_191 = x_182;
}
lean_ctor_set(x_191, 0, x_187);
lean_ctor_set(x_191, 1, x_183);
lean_ctor_set(x_191, 2, x_190);
lean_ctor_set(x_191, 3, x_189);
lean_ctor_set(x_191, 4, x_188);
x_192 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_192, 0, x_191);
lean_ctor_set(x_192, 1, x_184);
x_193 = l_ReaderT_instMonad___redArg(x_192);
x_194 = lp_aesop_Aesop_SearchM_run___redArg___closed__2;
x_195 = lp_aesop_Aesop_SearchM_instMonadRef___closed__5;
x_196 = lean_ctor_get(x_195, 0);
lean_inc_ref(x_196);
x_197 = lp_aesop_Aesop_SearchM_run___redArg___closed__3;
lean_inc_ref(x_193);
x_198 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_197, x_193);
x_199 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_199, 0, x_194);
lean_ctor_set(x_199, 1, x_196);
lean_ctor_set(x_199, 2, x_198);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_2);
x_200 = lp_aesop_Aesop_mkInitialTree(x_6, x_2, x_8, x_9, x_10, x_11, x_12);
if (lean_obj_tag(x_200) == 0)
{
lean_object* x_201; lean_object* x_202; 
x_201 = lean_ctor_get(x_200, 0);
lean_inc(x_201);
lean_dec_ref(x_200);
x_202 = l_Lean_Meta_getSimpCongrTheorems___redArg(x_12);
if (lean_obj_tag(x_202) == 0)
{
lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; size_t x_208; size_t x_209; lean_object* x_210; lean_object* x_211; 
x_203 = lean_ctor_get(x_202, 0);
lean_inc(x_203);
lean_dec_ref(x_202);
x_204 = lean_ctor_get(x_2, 1);
x_205 = lean_ctor_get(x_2, 2);
x_206 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_run___redArg___lam__0___boxed), 1, 0);
x_207 = lp_aesop_Aesop_SearchM_run___redArg___closed__13;
x_208 = lean_array_size(x_204);
x_209 = 0;
lean_inc_ref(x_204);
x_210 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_207, x_206, x_208, x_209, x_204);
x_211 = l_Lean_Meta_Simp_mkContext___redArg(x_4, x_210, x_203, x_9, x_12);
if (lean_obj_tag(x_211) == 0)
{
lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; uint8_t x_221; 
x_212 = lean_ctor_get(x_211, 0);
lean_inc(x_212);
lean_dec_ref(x_211);
x_213 = lean_ctor_get(x_201, 0);
x_214 = lean_st_ref_get(x_213);
x_215 = lp_aesop_Aesop_SearchM_run___redArg___closed__14;
x_216 = lean_ctor_get(x_215, 5);
lean_inc_ref(x_216);
x_217 = lean_apply_1(x_216, x_214);
x_218 = lean_ctor_get(x_217, 1);
lean_inc_ref(x_218);
lean_dec_ref(x_217);
x_219 = lean_array_get_size(x_218);
x_220 = lean_unsigned_to_nat(1u);
x_221 = lean_nat_dec_eq(x_219, x_220);
if (x_221 == 0)
{
lean_object* x_222; lean_object* x_223; lean_object* x_224; 
lean_dec_ref(x_218);
lean_dec(x_212);
lean_dec(x_201);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_222 = lp_aesop_Aesop_SearchM_run___redArg___closed__16;
x_223 = l_Lean_throwError___redArg(x_193, x_199, x_222);
x_224 = lean_apply_6(x_223, x_8, x_9, x_10, x_11, x_12, lean_box(0));
return x_224;
}
else
{
lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; uint8_t x_231; uint8_t x_232; lean_object* x_233; size_t x_234; lean_object* x_235; lean_object* x_236; lean_object* x_237; uint8_t x_238; lean_object* x_239; lean_object* x_240; 
lean_dec_ref(x_199);
lean_dec_ref(x_193);
x_225 = lean_unsigned_to_nat(0u);
x_226 = lean_array_fget(x_218, x_225);
lean_dec_ref(x_218);
x_227 = lp_aesop_Aesop_SearchM_run___redArg___closed__17;
x_228 = lean_array_push(x_227, x_226);
x_229 = lp_aesop_Aesop_Queue_init_x27___redArg(x_1, x_228);
x_230 = lean_ctor_get(x_3, 0);
x_231 = lean_ctor_get_uint8(x_230, sizeof(void*)*6 + 7);
x_232 = lean_ctor_get_uint8(x_230, sizeof(void*)*6 + 8);
x_233 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_run___redArg___lam__1___boxed), 1, 0);
x_234 = lean_array_size(x_205);
lean_inc_ref(x_205);
x_235 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_207, x_233, x_234, x_209, x_205);
x_236 = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(x_236, 0, x_212);
lean_ctor_set(x_236, 1, x_5);
lean_ctor_set(x_236, 2, x_235);
lean_ctor_set_uint8(x_236, sizeof(void*)*3, x_231);
lean_ctor_set_uint8(x_236, sizeof(void*)*3 + 1, x_232);
x_237 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_237, 0, x_2);
lean_ctor_set(x_237, 1, x_236);
lean_ctor_set(x_237, 2, x_3);
x_238 = 0;
x_239 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_239, 0, x_220);
lean_ctor_set(x_239, 1, x_229);
lean_ctor_set_uint8(x_239, sizeof(void*)*2, x_238);
x_240 = lp_aesop_Aesop_SearchM_run_x27___redArg(x_237, x_239, x_201, x_7, x_8, x_9, x_10, x_11, x_12);
return x_240;
}
}
else
{
lean_object* x_241; lean_object* x_242; lean_object* x_243; 
lean_dec(x_201);
lean_dec_ref(x_199);
lean_dec_ref(x_193);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_241 = lean_ctor_get(x_211, 0);
lean_inc(x_241);
if (lean_is_exclusive(x_211)) {
 lean_ctor_release(x_211, 0);
 x_242 = x_211;
} else {
 lean_dec_ref(x_211);
 x_242 = lean_box(0);
}
if (lean_is_scalar(x_242)) {
 x_243 = lean_alloc_ctor(1, 1, 0);
} else {
 x_243 = x_242;
}
lean_ctor_set(x_243, 0, x_241);
return x_243;
}
}
else
{
lean_object* x_244; lean_object* x_245; lean_object* x_246; 
lean_dec(x_201);
lean_dec_ref(x_199);
lean_dec_ref(x_193);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_244 = lean_ctor_get(x_202, 0);
lean_inc(x_244);
if (lean_is_exclusive(x_202)) {
 lean_ctor_release(x_202, 0);
 x_245 = x_202;
} else {
 lean_dec_ref(x_202);
 x_245 = lean_box(0);
}
if (lean_is_scalar(x_245)) {
 x_246 = lean_alloc_ctor(1, 1, 0);
} else {
 x_246 = x_245;
}
lean_ctor_set(x_246, 0, x_244);
return x_246;
}
}
else
{
lean_object* x_247; lean_object* x_248; lean_object* x_249; 
lean_dec_ref(x_199);
lean_dec_ref(x_193);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_247 = lean_ctor_get(x_200, 0);
lean_inc(x_247);
if (lean_is_exclusive(x_200)) {
 lean_ctor_release(x_200, 0);
 x_248 = x_200;
} else {
 lean_dec_ref(x_200);
 x_248 = lean_box(0);
}
if (lean_is_scalar(x_248)) {
 x_249 = lean_alloc_ctor(1, 1, 0);
} else {
 x_249 = x_248;
}
lean_ctor_set(x_249, 0, x_247);
return x_249;
}
}
}
else
{
lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; lean_object* x_259; lean_object* x_260; lean_object* x_261; lean_object* x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; lean_object* x_269; lean_object* x_270; lean_object* x_271; lean_object* x_272; lean_object* x_273; lean_object* x_274; lean_object* x_275; lean_object* x_276; lean_object* x_277; lean_object* x_278; lean_object* x_279; lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; lean_object* x_288; 
x_250 = lean_ctor_get(x_16, 0);
x_251 = lean_ctor_get(x_16, 2);
x_252 = lean_ctor_get(x_16, 3);
x_253 = lean_ctor_get(x_16, 4);
lean_inc(x_253);
lean_inc(x_252);
lean_inc(x_251);
lean_inc(x_250);
lean_dec(x_16);
x_254 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_255 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_250);
x_256 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_256, 0, x_250);
x_257 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_257, 0, x_250);
x_258 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_258, 0, x_256);
lean_ctor_set(x_258, 1, x_257);
x_259 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_259, 0, x_253);
x_260 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_260, 0, x_252);
x_261 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_261, 0, x_251);
x_262 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_262, 0, x_258);
lean_ctor_set(x_262, 1, x_254);
lean_ctor_set(x_262, 2, x_261);
lean_ctor_set(x_262, 3, x_260);
lean_ctor_set(x_262, 4, x_259);
lean_ctor_set(x_14, 1, x_255);
lean_ctor_set(x_14, 0, x_262);
x_263 = l_ReaderT_instMonad___redArg(x_14);
x_264 = lean_ctor_get(x_263, 0);
lean_inc_ref(x_264);
if (lean_is_exclusive(x_263)) {
 lean_ctor_release(x_263, 0);
 lean_ctor_release(x_263, 1);
 x_265 = x_263;
} else {
 lean_dec_ref(x_263);
 x_265 = lean_box(0);
}
x_266 = lean_ctor_get(x_264, 0);
lean_inc_ref(x_266);
x_267 = lean_ctor_get(x_264, 2);
lean_inc(x_267);
x_268 = lean_ctor_get(x_264, 3);
lean_inc(x_268);
x_269 = lean_ctor_get(x_264, 4);
lean_inc(x_269);
if (lean_is_exclusive(x_264)) {
 lean_ctor_release(x_264, 0);
 lean_ctor_release(x_264, 1);
 lean_ctor_release(x_264, 2);
 lean_ctor_release(x_264, 3);
 lean_ctor_release(x_264, 4);
 x_270 = x_264;
} else {
 lean_dec_ref(x_264);
 x_270 = lean_box(0);
}
x_271 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_272 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_266);
x_273 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_273, 0, x_266);
x_274 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_274, 0, x_266);
x_275 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_275, 0, x_273);
lean_ctor_set(x_275, 1, x_274);
x_276 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_276, 0, x_269);
x_277 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_277, 0, x_268);
x_278 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_278, 0, x_267);
if (lean_is_scalar(x_270)) {
 x_279 = lean_alloc_ctor(0, 5, 0);
} else {
 x_279 = x_270;
}
lean_ctor_set(x_279, 0, x_275);
lean_ctor_set(x_279, 1, x_271);
lean_ctor_set(x_279, 2, x_278);
lean_ctor_set(x_279, 3, x_277);
lean_ctor_set(x_279, 4, x_276);
if (lean_is_scalar(x_265)) {
 x_280 = lean_alloc_ctor(0, 2, 0);
} else {
 x_280 = x_265;
}
lean_ctor_set(x_280, 0, x_279);
lean_ctor_set(x_280, 1, x_272);
x_281 = l_ReaderT_instMonad___redArg(x_280);
x_282 = lp_aesop_Aesop_SearchM_run___redArg___closed__2;
x_283 = lp_aesop_Aesop_SearchM_instMonadRef___closed__5;
x_284 = lean_ctor_get(x_283, 0);
lean_inc_ref(x_284);
x_285 = lp_aesop_Aesop_SearchM_run___redArg___closed__3;
lean_inc_ref(x_281);
x_286 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_285, x_281);
x_287 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_287, 0, x_282);
lean_ctor_set(x_287, 1, x_284);
lean_ctor_set(x_287, 2, x_286);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_2);
x_288 = lp_aesop_Aesop_mkInitialTree(x_6, x_2, x_8, x_9, x_10, x_11, x_12);
if (lean_obj_tag(x_288) == 0)
{
lean_object* x_289; lean_object* x_290; 
x_289 = lean_ctor_get(x_288, 0);
lean_inc(x_289);
lean_dec_ref(x_288);
x_290 = l_Lean_Meta_getSimpCongrTheorems___redArg(x_12);
if (lean_obj_tag(x_290) == 0)
{
lean_object* x_291; lean_object* x_292; lean_object* x_293; lean_object* x_294; lean_object* x_295; size_t x_296; size_t x_297; lean_object* x_298; lean_object* x_299; 
x_291 = lean_ctor_get(x_290, 0);
lean_inc(x_291);
lean_dec_ref(x_290);
x_292 = lean_ctor_get(x_2, 1);
x_293 = lean_ctor_get(x_2, 2);
x_294 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_run___redArg___lam__0___boxed), 1, 0);
x_295 = lp_aesop_Aesop_SearchM_run___redArg___closed__13;
x_296 = lean_array_size(x_292);
x_297 = 0;
lean_inc_ref(x_292);
x_298 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_295, x_294, x_296, x_297, x_292);
x_299 = l_Lean_Meta_Simp_mkContext___redArg(x_4, x_298, x_291, x_9, x_12);
if (lean_obj_tag(x_299) == 0)
{
lean_object* x_300; lean_object* x_301; lean_object* x_302; lean_object* x_303; lean_object* x_304; lean_object* x_305; lean_object* x_306; lean_object* x_307; lean_object* x_308; uint8_t x_309; 
x_300 = lean_ctor_get(x_299, 0);
lean_inc(x_300);
lean_dec_ref(x_299);
x_301 = lean_ctor_get(x_289, 0);
x_302 = lean_st_ref_get(x_301);
x_303 = lp_aesop_Aesop_SearchM_run___redArg___closed__14;
x_304 = lean_ctor_get(x_303, 5);
lean_inc_ref(x_304);
x_305 = lean_apply_1(x_304, x_302);
x_306 = lean_ctor_get(x_305, 1);
lean_inc_ref(x_306);
lean_dec_ref(x_305);
x_307 = lean_array_get_size(x_306);
x_308 = lean_unsigned_to_nat(1u);
x_309 = lean_nat_dec_eq(x_307, x_308);
if (x_309 == 0)
{
lean_object* x_310; lean_object* x_311; lean_object* x_312; 
lean_dec_ref(x_306);
lean_dec(x_300);
lean_dec(x_289);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_310 = lp_aesop_Aesop_SearchM_run___redArg___closed__16;
x_311 = l_Lean_throwError___redArg(x_281, x_287, x_310);
x_312 = lean_apply_6(x_311, x_8, x_9, x_10, x_11, x_12, lean_box(0));
return x_312;
}
else
{
lean_object* x_313; lean_object* x_314; lean_object* x_315; lean_object* x_316; lean_object* x_317; lean_object* x_318; uint8_t x_319; uint8_t x_320; lean_object* x_321; size_t x_322; lean_object* x_323; lean_object* x_324; lean_object* x_325; uint8_t x_326; lean_object* x_327; lean_object* x_328; 
lean_dec_ref(x_287);
lean_dec_ref(x_281);
x_313 = lean_unsigned_to_nat(0u);
x_314 = lean_array_fget(x_306, x_313);
lean_dec_ref(x_306);
x_315 = lp_aesop_Aesop_SearchM_run___redArg___closed__17;
x_316 = lean_array_push(x_315, x_314);
x_317 = lp_aesop_Aesop_Queue_init_x27___redArg(x_1, x_316);
x_318 = lean_ctor_get(x_3, 0);
x_319 = lean_ctor_get_uint8(x_318, sizeof(void*)*6 + 7);
x_320 = lean_ctor_get_uint8(x_318, sizeof(void*)*6 + 8);
x_321 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_run___redArg___lam__1___boxed), 1, 0);
x_322 = lean_array_size(x_293);
lean_inc_ref(x_293);
x_323 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_295, x_321, x_322, x_297, x_293);
x_324 = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(x_324, 0, x_300);
lean_ctor_set(x_324, 1, x_5);
lean_ctor_set(x_324, 2, x_323);
lean_ctor_set_uint8(x_324, sizeof(void*)*3, x_319);
lean_ctor_set_uint8(x_324, sizeof(void*)*3 + 1, x_320);
x_325 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_325, 0, x_2);
lean_ctor_set(x_325, 1, x_324);
lean_ctor_set(x_325, 2, x_3);
x_326 = 0;
x_327 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_327, 0, x_308);
lean_ctor_set(x_327, 1, x_317);
lean_ctor_set_uint8(x_327, sizeof(void*)*2, x_326);
x_328 = lp_aesop_Aesop_SearchM_run_x27___redArg(x_325, x_327, x_289, x_7, x_8, x_9, x_10, x_11, x_12);
return x_328;
}
}
else
{
lean_object* x_329; lean_object* x_330; lean_object* x_331; 
lean_dec(x_289);
lean_dec_ref(x_287);
lean_dec_ref(x_281);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_329 = lean_ctor_get(x_299, 0);
lean_inc(x_329);
if (lean_is_exclusive(x_299)) {
 lean_ctor_release(x_299, 0);
 x_330 = x_299;
} else {
 lean_dec_ref(x_299);
 x_330 = lean_box(0);
}
if (lean_is_scalar(x_330)) {
 x_331 = lean_alloc_ctor(1, 1, 0);
} else {
 x_331 = x_330;
}
lean_ctor_set(x_331, 0, x_329);
return x_331;
}
}
else
{
lean_object* x_332; lean_object* x_333; lean_object* x_334; 
lean_dec(x_289);
lean_dec_ref(x_287);
lean_dec_ref(x_281);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_332 = lean_ctor_get(x_290, 0);
lean_inc(x_332);
if (lean_is_exclusive(x_290)) {
 lean_ctor_release(x_290, 0);
 x_333 = x_290;
} else {
 lean_dec_ref(x_290);
 x_333 = lean_box(0);
}
if (lean_is_scalar(x_333)) {
 x_334 = lean_alloc_ctor(1, 1, 0);
} else {
 x_334 = x_333;
}
lean_ctor_set(x_334, 0, x_332);
return x_334;
}
}
else
{
lean_object* x_335; lean_object* x_336; lean_object* x_337; 
lean_dec_ref(x_287);
lean_dec_ref(x_281);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_335 = lean_ctor_get(x_288, 0);
lean_inc(x_335);
if (lean_is_exclusive(x_288)) {
 lean_ctor_release(x_288, 0);
 x_336 = x_288;
} else {
 lean_dec_ref(x_288);
 x_336 = lean_box(0);
}
if (lean_is_scalar(x_336)) {
 x_337 = lean_alloc_ctor(1, 1, 0);
} else {
 x_337 = x_336;
}
lean_ctor_set(x_337, 0, x_335);
return x_337;
}
}
}
else
{
lean_object* x_338; lean_object* x_339; lean_object* x_340; lean_object* x_341; lean_object* x_342; lean_object* x_343; lean_object* x_344; lean_object* x_345; lean_object* x_346; lean_object* x_347; lean_object* x_348; lean_object* x_349; lean_object* x_350; lean_object* x_351; lean_object* x_352; lean_object* x_353; lean_object* x_354; lean_object* x_355; lean_object* x_356; lean_object* x_357; lean_object* x_358; lean_object* x_359; lean_object* x_360; lean_object* x_361; lean_object* x_362; lean_object* x_363; lean_object* x_364; lean_object* x_365; lean_object* x_366; lean_object* x_367; lean_object* x_368; lean_object* x_369; lean_object* x_370; lean_object* x_371; lean_object* x_372; lean_object* x_373; lean_object* x_374; lean_object* x_375; lean_object* x_376; lean_object* x_377; lean_object* x_378; lean_object* x_379; 
x_338 = lean_ctor_get(x_14, 0);
lean_inc(x_338);
lean_dec(x_14);
x_339 = lean_ctor_get(x_338, 0);
lean_inc_ref(x_339);
x_340 = lean_ctor_get(x_338, 2);
lean_inc(x_340);
x_341 = lean_ctor_get(x_338, 3);
lean_inc(x_341);
x_342 = lean_ctor_get(x_338, 4);
lean_inc(x_342);
if (lean_is_exclusive(x_338)) {
 lean_ctor_release(x_338, 0);
 lean_ctor_release(x_338, 1);
 lean_ctor_release(x_338, 2);
 lean_ctor_release(x_338, 3);
 lean_ctor_release(x_338, 4);
 x_343 = x_338;
} else {
 lean_dec_ref(x_338);
 x_343 = lean_box(0);
}
x_344 = lp_aesop_Aesop_SearchM_instMonad___closed__2;
x_345 = lp_aesop_Aesop_SearchM_instMonad___closed__3;
lean_inc_ref(x_339);
x_346 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_346, 0, x_339);
x_347 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_347, 0, x_339);
x_348 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_348, 0, x_346);
lean_ctor_set(x_348, 1, x_347);
x_349 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_349, 0, x_342);
x_350 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_350, 0, x_341);
x_351 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_351, 0, x_340);
if (lean_is_scalar(x_343)) {
 x_352 = lean_alloc_ctor(0, 5, 0);
} else {
 x_352 = x_343;
}
lean_ctor_set(x_352, 0, x_348);
lean_ctor_set(x_352, 1, x_344);
lean_ctor_set(x_352, 2, x_351);
lean_ctor_set(x_352, 3, x_350);
lean_ctor_set(x_352, 4, x_349);
x_353 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_353, 0, x_352);
lean_ctor_set(x_353, 1, x_345);
x_354 = l_ReaderT_instMonad___redArg(x_353);
x_355 = lean_ctor_get(x_354, 0);
lean_inc_ref(x_355);
if (lean_is_exclusive(x_354)) {
 lean_ctor_release(x_354, 0);
 lean_ctor_release(x_354, 1);
 x_356 = x_354;
} else {
 lean_dec_ref(x_354);
 x_356 = lean_box(0);
}
x_357 = lean_ctor_get(x_355, 0);
lean_inc_ref(x_357);
x_358 = lean_ctor_get(x_355, 2);
lean_inc(x_358);
x_359 = lean_ctor_get(x_355, 3);
lean_inc(x_359);
x_360 = lean_ctor_get(x_355, 4);
lean_inc(x_360);
if (lean_is_exclusive(x_355)) {
 lean_ctor_release(x_355, 0);
 lean_ctor_release(x_355, 1);
 lean_ctor_release(x_355, 2);
 lean_ctor_release(x_355, 3);
 lean_ctor_release(x_355, 4);
 x_361 = x_355;
} else {
 lean_dec_ref(x_355);
 x_361 = lean_box(0);
}
x_362 = lp_aesop_Aesop_SearchM_instMonad___closed__4;
x_363 = lp_aesop_Aesop_SearchM_instMonad___closed__5;
lean_inc_ref(x_357);
x_364 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_364, 0, x_357);
x_365 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_365, 0, x_357);
x_366 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_366, 0, x_364);
lean_ctor_set(x_366, 1, x_365);
x_367 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_367, 0, x_360);
x_368 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_368, 0, x_359);
x_369 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_369, 0, x_358);
if (lean_is_scalar(x_361)) {
 x_370 = lean_alloc_ctor(0, 5, 0);
} else {
 x_370 = x_361;
}
lean_ctor_set(x_370, 0, x_366);
lean_ctor_set(x_370, 1, x_362);
lean_ctor_set(x_370, 2, x_369);
lean_ctor_set(x_370, 3, x_368);
lean_ctor_set(x_370, 4, x_367);
if (lean_is_scalar(x_356)) {
 x_371 = lean_alloc_ctor(0, 2, 0);
} else {
 x_371 = x_356;
}
lean_ctor_set(x_371, 0, x_370);
lean_ctor_set(x_371, 1, x_363);
x_372 = l_ReaderT_instMonad___redArg(x_371);
x_373 = lp_aesop_Aesop_SearchM_run___redArg___closed__2;
x_374 = lp_aesop_Aesop_SearchM_instMonadRef___closed__5;
x_375 = lean_ctor_get(x_374, 0);
lean_inc_ref(x_375);
x_376 = lp_aesop_Aesop_SearchM_run___redArg___closed__3;
lean_inc_ref(x_372);
x_377 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_376, x_372);
x_378 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_378, 0, x_373);
lean_ctor_set(x_378, 1, x_375);
lean_ctor_set(x_378, 2, x_377);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_2);
x_379 = lp_aesop_Aesop_mkInitialTree(x_6, x_2, x_8, x_9, x_10, x_11, x_12);
if (lean_obj_tag(x_379) == 0)
{
lean_object* x_380; lean_object* x_381; 
x_380 = lean_ctor_get(x_379, 0);
lean_inc(x_380);
lean_dec_ref(x_379);
x_381 = l_Lean_Meta_getSimpCongrTheorems___redArg(x_12);
if (lean_obj_tag(x_381) == 0)
{
lean_object* x_382; lean_object* x_383; lean_object* x_384; lean_object* x_385; lean_object* x_386; size_t x_387; size_t x_388; lean_object* x_389; lean_object* x_390; 
x_382 = lean_ctor_get(x_381, 0);
lean_inc(x_382);
lean_dec_ref(x_381);
x_383 = lean_ctor_get(x_2, 1);
x_384 = lean_ctor_get(x_2, 2);
x_385 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_run___redArg___lam__0___boxed), 1, 0);
x_386 = lp_aesop_Aesop_SearchM_run___redArg___closed__13;
x_387 = lean_array_size(x_383);
x_388 = 0;
lean_inc_ref(x_383);
x_389 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_386, x_385, x_387, x_388, x_383);
x_390 = l_Lean_Meta_Simp_mkContext___redArg(x_4, x_389, x_382, x_9, x_12);
if (lean_obj_tag(x_390) == 0)
{
lean_object* x_391; lean_object* x_392; lean_object* x_393; lean_object* x_394; lean_object* x_395; lean_object* x_396; lean_object* x_397; lean_object* x_398; lean_object* x_399; uint8_t x_400; 
x_391 = lean_ctor_get(x_390, 0);
lean_inc(x_391);
lean_dec_ref(x_390);
x_392 = lean_ctor_get(x_380, 0);
x_393 = lean_st_ref_get(x_392);
x_394 = lp_aesop_Aesop_SearchM_run___redArg___closed__14;
x_395 = lean_ctor_get(x_394, 5);
lean_inc_ref(x_395);
x_396 = lean_apply_1(x_395, x_393);
x_397 = lean_ctor_get(x_396, 1);
lean_inc_ref(x_397);
lean_dec_ref(x_396);
x_398 = lean_array_get_size(x_397);
x_399 = lean_unsigned_to_nat(1u);
x_400 = lean_nat_dec_eq(x_398, x_399);
if (x_400 == 0)
{
lean_object* x_401; lean_object* x_402; lean_object* x_403; 
lean_dec_ref(x_397);
lean_dec(x_391);
lean_dec(x_380);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_401 = lp_aesop_Aesop_SearchM_run___redArg___closed__16;
x_402 = l_Lean_throwError___redArg(x_372, x_378, x_401);
x_403 = lean_apply_6(x_402, x_8, x_9, x_10, x_11, x_12, lean_box(0));
return x_403;
}
else
{
lean_object* x_404; lean_object* x_405; lean_object* x_406; lean_object* x_407; lean_object* x_408; lean_object* x_409; uint8_t x_410; uint8_t x_411; lean_object* x_412; size_t x_413; lean_object* x_414; lean_object* x_415; lean_object* x_416; uint8_t x_417; lean_object* x_418; lean_object* x_419; 
lean_dec_ref(x_378);
lean_dec_ref(x_372);
x_404 = lean_unsigned_to_nat(0u);
x_405 = lean_array_fget(x_397, x_404);
lean_dec_ref(x_397);
x_406 = lp_aesop_Aesop_SearchM_run___redArg___closed__17;
x_407 = lean_array_push(x_406, x_405);
x_408 = lp_aesop_Aesop_Queue_init_x27___redArg(x_1, x_407);
x_409 = lean_ctor_get(x_3, 0);
x_410 = lean_ctor_get_uint8(x_409, sizeof(void*)*6 + 7);
x_411 = lean_ctor_get_uint8(x_409, sizeof(void*)*6 + 8);
x_412 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_run___redArg___lam__1___boxed), 1, 0);
x_413 = lean_array_size(x_384);
lean_inc_ref(x_384);
x_414 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_386, x_412, x_413, x_388, x_384);
x_415 = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(x_415, 0, x_391);
lean_ctor_set(x_415, 1, x_5);
lean_ctor_set(x_415, 2, x_414);
lean_ctor_set_uint8(x_415, sizeof(void*)*3, x_410);
lean_ctor_set_uint8(x_415, sizeof(void*)*3 + 1, x_411);
x_416 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_416, 0, x_2);
lean_ctor_set(x_416, 1, x_415);
lean_ctor_set(x_416, 2, x_3);
x_417 = 0;
x_418 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_418, 0, x_399);
lean_ctor_set(x_418, 1, x_408);
lean_ctor_set_uint8(x_418, sizeof(void*)*2, x_417);
x_419 = lp_aesop_Aesop_SearchM_run_x27___redArg(x_416, x_418, x_380, x_7, x_8, x_9, x_10, x_11, x_12);
return x_419;
}
}
else
{
lean_object* x_420; lean_object* x_421; lean_object* x_422; 
lean_dec(x_380);
lean_dec_ref(x_378);
lean_dec_ref(x_372);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_420 = lean_ctor_get(x_390, 0);
lean_inc(x_420);
if (lean_is_exclusive(x_390)) {
 lean_ctor_release(x_390, 0);
 x_421 = x_390;
} else {
 lean_dec_ref(x_390);
 x_421 = lean_box(0);
}
if (lean_is_scalar(x_421)) {
 x_422 = lean_alloc_ctor(1, 1, 0);
} else {
 x_422 = x_421;
}
lean_ctor_set(x_422, 0, x_420);
return x_422;
}
}
else
{
lean_object* x_423; lean_object* x_424; lean_object* x_425; 
lean_dec(x_380);
lean_dec_ref(x_378);
lean_dec_ref(x_372);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_423 = lean_ctor_get(x_381, 0);
lean_inc(x_423);
if (lean_is_exclusive(x_381)) {
 lean_ctor_release(x_381, 0);
 x_424 = x_381;
} else {
 lean_dec_ref(x_381);
 x_424 = lean_box(0);
}
if (lean_is_scalar(x_424)) {
 x_425 = lean_alloc_ctor(1, 1, 0);
} else {
 x_425 = x_424;
}
lean_ctor_set(x_425, 0, x_423);
return x_425;
}
}
else
{
lean_object* x_426; lean_object* x_427; lean_object* x_428; 
lean_dec_ref(x_378);
lean_dec_ref(x_372);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_426 = lean_ctor_get(x_379, 0);
lean_inc(x_426);
if (lean_is_exclusive(x_379)) {
 lean_ctor_release(x_379, 0);
 x_427 = x_379;
} else {
 lean_dec_ref(x_379);
 x_427 = lean_box(0);
}
if (lean_is_scalar(x_427)) {
 x_428 = lean_alloc_ctor(1, 1, 0);
} else {
 x_428 = x_427;
}
lean_ctor_set(x_428, 0, x_426);
return x_428;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_16; 
x_16 = lp_aesop_Aesop_SearchM_run___redArg(x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_aesop_Aesop_SearchM_run(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_aesop_Aesop_SearchM_run___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_st_ref_get(x_1);
lean_dec(x_4);
x_5 = lean_st_ref_get(x_2);
x_6 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_6, 0, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_getTree___redArg(x_4, x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_getTree(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_getTree___redArg(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_st_ref_get(x_2);
lean_dec(x_5);
x_6 = lean_st_ref_take(x_3);
lean_dec(x_6);
x_7 = lean_st_ref_set(x_3, x_1);
x_8 = lean_box(0);
x_9 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_setTree___redArg(x_3, x_5, x_6);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_setTree(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_setTree___redArg(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_st_ref_get(x_2);
lean_dec(x_5);
x_6 = lean_st_ref_take(x_3);
x_7 = lean_apply_1(x_1, x_6);
x_8 = lean_st_ref_set(x_3, x_7);
x_9 = lean_box(0);
x_10 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_modifyTree___redArg(x_3, x_5, x_6);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_modifyTree(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_modifyTree___redArg(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___redArg(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_st_ref_get(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec(x_3);
x_5 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_getIteration___redArg(x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_getIteration(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_getIteration___redArg(x_1);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___redArg(lean_object* x_1) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_st_ref_take(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_unsigned_to_nat(1u);
x_7 = lean_nat_add(x_5, x_6);
lean_dec(x_5);
lean_ctor_set(x_3, 0, x_7);
x_8 = lean_st_ref_set(x_1, x_3);
x_9 = lean_box(0);
x_10 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_11 = lean_ctor_get(x_3, 0);
x_12 = lean_ctor_get(x_3, 1);
x_13 = lean_ctor_get_uint8(x_3, sizeof(void*)*2);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_3);
x_14 = lean_unsigned_to_nat(1u);
x_15 = lean_nat_add(x_11, x_14);
lean_dec(x_11);
x_16 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_12);
lean_ctor_set_uint8(x_16, sizeof(void*)*2, x_13);
x_17 = lean_st_ref_set(x_1, x_16);
x_18 = lean_box(0);
x_19 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_incrementIteration___redArg(x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_incrementIteration(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_incrementIteration___redArg(x_1);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_4 = lean_st_ref_get(x_2);
x_5 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
x_6 = lean_st_ref_get(x_2);
lean_dec(x_6);
x_7 = !lean_is_exclusive(x_4);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_8 = lean_ctor_get(x_4, 1);
x_9 = lean_apply_2(x_5, x_8, lean_box(0));
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_9, 1);
lean_inc(x_11);
lean_dec_ref(x_9);
lean_ctor_set(x_4, 1, x_11);
x_12 = lean_st_ref_set(x_2, x_4);
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_10);
return x_13;
}
else
{
lean_object* x_14; lean_object* x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_14 = lean_ctor_get(x_4, 0);
x_15 = lean_ctor_get(x_4, 1);
x_16 = lean_ctor_get_uint8(x_4, sizeof(void*)*2);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_4);
x_17 = lean_apply_2(x_5, x_15, lean_box(0));
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
x_19 = lean_ctor_get(x_17, 1);
lean_inc(x_19);
lean_dec_ref(x_17);
x_20 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_20, 0, x_14);
lean_ctor_set(x_20, 1, x_19);
lean_ctor_set_uint8(x_20, sizeof(void*)*2, x_16);
x_21 = lean_st_ref_set(x_2, x_20);
x_22 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_22, 0, x_18);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_popGoal_x3f___redArg(x_2, x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_popGoal_x3f(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_popGoal_x3f___redArg(x_1, x_2);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_5 = lean_st_ref_get(x_3);
x_6 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_6);
lean_dec_ref(x_1);
x_7 = lean_st_ref_get(x_3);
lean_dec(x_7);
x_8 = !lean_is_exclusive(x_5);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_5, 1);
x_10 = lean_apply_3(x_6, x_9, x_2, lean_box(0));
lean_ctor_set(x_5, 1, x_10);
x_11 = lean_st_ref_set(x_3, x_5);
x_12 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
else
{
lean_object* x_13; lean_object* x_14; uint8_t x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_13 = lean_ctor_get(x_5, 0);
x_14 = lean_ctor_get(x_5, 1);
x_15 = lean_ctor_get_uint8(x_5, sizeof(void*)*2);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_5);
x_16 = lean_apply_3(x_6, x_14, x_2, lean_box(0));
x_17 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_17, 0, x_13);
lean_ctor_set(x_17, 1, x_16);
lean_ctor_set_uint8(x_17, sizeof(void*)*2, x_15);
x_18 = lean_st_ref_set(x_3, x_17);
x_19 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_enqueueGoals___redArg(x_2, x_3, x_5);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_enqueueGoals(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_enqueueGoals___redArg(x_1, x_2, x_3);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(lean_object* x_1) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_st_ref_take(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
uint8_t x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = 1;
lean_ctor_set_uint8(x_3, sizeof(void*)*2, x_5);
x_6 = lean_st_ref_set(x_1, x_3);
x_7 = lean_box(0);
x_8 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_9 = lean_ctor_get(x_3, 0);
x_10 = lean_ctor_get(x_3, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_3);
x_11 = 1;
x_12 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_12, 0, x_9);
lean_ctor_set(x_12, 1, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*2, x_11);
x_13 = lean_st_ref_set(x_1, x_12);
x_14 = lean_box(0);
x_15 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_setMaxRuleApplicationDepthReached(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(x_1);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(lean_object* x_1) {
_start:
{
lean_object* x_3; uint8_t x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_st_ref_get(x_1);
x_4 = lean_ctor_get_uint8(x_3, sizeof(void*)*2);
lean_dec(x_3);
x_5 = lean_box(x_4);
x_6 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_6, 0, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_wasMaxRuleApplicationDepthReached(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(x_1);
lean_dec(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Options(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_Queue_Class(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Stats_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleSet(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_SearchM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Options(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Queue_Class(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Stats_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__0 = _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__0);
lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1 = _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1);
lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__2 = _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__2();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__2);
lp_aesop_Aesop_instInhabitedNormSimpContext_default = _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormSimpContext_default);
lp_aesop_Aesop_instInhabitedNormSimpContext = _init_lp_aesop_Aesop_instInhabitedNormSimpContext();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormSimpContext);
lp_aesop_Aesop_SearchM_instMonad___closed__0 = _init_lp_aesop_Aesop_SearchM_instMonad___closed__0();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonad___closed__0);
lp_aesop_Aesop_SearchM_instMonad___closed__1 = _init_lp_aesop_Aesop_SearchM_instMonad___closed__1();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonad___closed__1);
lp_aesop_Aesop_SearchM_instMonad___closed__2 = _init_lp_aesop_Aesop_SearchM_instMonad___closed__2();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonad___closed__2);
lp_aesop_Aesop_SearchM_instMonad___closed__3 = _init_lp_aesop_Aesop_SearchM_instMonad___closed__3();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonad___closed__3);
lp_aesop_Aesop_SearchM_instMonad___closed__4 = _init_lp_aesop_Aesop_SearchM_instMonad___closed__4();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonad___closed__4);
lp_aesop_Aesop_SearchM_instMonad___closed__5 = _init_lp_aesop_Aesop_SearchM_instMonad___closed__5();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonad___closed__5);
lp_aesop_Aesop_SearchM_instMonadRef___closed__0 = _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__0();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonadRef___closed__0);
lp_aesop_Aesop_SearchM_instMonadRef___closed__1 = _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__1();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonadRef___closed__1);
lp_aesop_Aesop_SearchM_instMonadRef___closed__2 = _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__2();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonadRef___closed__2);
lp_aesop_Aesop_SearchM_instMonadRef___closed__3 = _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__3();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonadRef___closed__3);
lp_aesop_Aesop_SearchM_instMonadRef___closed__4 = _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__4();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonadRef___closed__4);
lp_aesop_Aesop_SearchM_instMonadRef___closed__5 = _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__5();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonadRef___closed__5);
lp_aesop_Aesop_SearchM_instMonadRef___closed__6 = _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__6();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonadRef___closed__6);
lp_aesop_Aesop_SearchM_instMonadRef___closed__7 = _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__7();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonadRef___closed__7);
lp_aesop_Aesop_SearchM_instMonadRef___closed__8 = _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__8();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instMonadRef___closed__8);
lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1 = _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1);
lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0 = _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0);
lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2 = _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2);
lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4 = _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4);
lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3 = _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3);
lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5 = _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5);
lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__6 = _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__6();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__6);
lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7 = _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7();
lean_mark_persistent(lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7);
lp_aesop_Aesop_SearchM_run___redArg___closed__1 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__1);
lp_aesop_Aesop_SearchM_run___redArg___closed__0 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__0);
lp_aesop_Aesop_SearchM_run___redArg___closed__2 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__2);
lp_aesop_Aesop_SearchM_run___redArg___closed__3 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__3);
lp_aesop_Aesop_SearchM_run___redArg___closed__10 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__10();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__10);
lp_aesop_Aesop_SearchM_run___redArg___closed__9 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__9();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__9);
lp_aesop_Aesop_SearchM_run___redArg___closed__8 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__8();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__8);
lp_aesop_Aesop_SearchM_run___redArg___closed__7 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__7();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__7);
lp_aesop_Aesop_SearchM_run___redArg___closed__6 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__6();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__6);
lp_aesop_Aesop_SearchM_run___redArg___closed__5 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__5();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__5);
lp_aesop_Aesop_SearchM_run___redArg___closed__4 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__4();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__4);
lp_aesop_Aesop_SearchM_run___redArg___closed__11 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__11();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__11);
lp_aesop_Aesop_SearchM_run___redArg___closed__12 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__12();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__12);
lp_aesop_Aesop_SearchM_run___redArg___closed__13 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__13();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__13);
lp_aesop_Aesop_SearchM_run___redArg___closed__14 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__14();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__14);
lp_aesop_Aesop_SearchM_run___redArg___closed__15 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__15();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__15);
lp_aesop_Aesop_SearchM_run___redArg___closed__16 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__16();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__16);
lp_aesop_Aesop_SearchM_run___redArg___closed__17 = _init_lp_aesop_Aesop_SearchM_run___redArg___closed__17();
lean_mark_persistent(lp_aesop_Aesop_SearchM_run___redArg___closed__17);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
