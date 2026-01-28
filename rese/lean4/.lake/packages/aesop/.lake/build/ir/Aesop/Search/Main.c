// Lean compiler output
// Module: Aesop.Search.Main
// Imports: public import Init public import Aesop.Check public import Aesop.Options public import Aesop.RuleSet public import Aesop.Script.Check public import Aesop.Script.Main public import Aesop.Search.Expansion public import Aesop.Search.ExpandSafePrefix public import Aesop.Search.Queue public import Aesop.Tree public import Aesop.Tree.Stats public import Aesop.Frontend.Extension
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
lean_object* l_Lean_MVarId_withContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadTraceCoreM;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__3;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__4;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
double lp_aesop_Aesop_Goal_priority(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__6;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
lean_object* l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___boxed(lean_object**);
lean_object* lp_aesop_Aesop_popGoal_x3f___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Options_queue(lean_object*);
lean_object* lp_aesop_Aesop_expandSafePrefix___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_aesop_Aesop_Percent_toHumanString(double);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__3;
lean_object* lp_aesop_Aesop_RegularRule_name(lean_object*);
lean_object* lp_aesop_Aesop_Script_UScript_optimize(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__20;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_collectGoalStatsIfEnabled(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__4;
lean_object* l_instMonadControlTOfPure___redArg(lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__5;
extern lean_object* lp_aesop_Aesop_Check_script;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__7;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__17;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__2___boxed(lean_object**);
lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getRootGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_GoalRef_markForcedUnprovable(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t l_Array_isEmpty___redArg(lean_object*);
lean_object* l_Lean_MVarId_checkNotAssigned(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getRootMVarCluster___redArg(lean_object*);
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__8;
lean_object* l_Lean_instMonadLogOfMonadLift___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5;
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_extractProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_expandGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__11;
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__4;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__9;
static lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1___closed__1;
lean_object* l_Lean_Core_checkSystem(lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_Check_get(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__8;
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed(lean_object*);
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__4;
uint8_t lean_usize_dec_eq(size_t, size_t);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__14;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__7;
lean_object* l_Lean_KVMap_find(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__9;
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__6;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__2;
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__1;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__9;
LEAN_EXPORT lean_object* lp_aesop_Aesop_search(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_incrementIteration___redArg(lean_object*);
uint8_t lp_aesop_Aesop_NodeState_isUnprovable(uint8_t);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__20;
lean_object* l_Lean_instMonadTraceOfMonadLift___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_extractSafePrefixScript(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__5;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__1;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonad(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__10;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_stats;
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__2;
lean_object* lp_aesop_Aesop_Goal_traceMetadata___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__6;
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__6;
extern lean_object* lp_aesop_Aesop_Check_script_steps;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__4;
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__10;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__5;
lean_object* lp_aesop_Aesop_extractSafePrefix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__19;
lean_object* lp_aesop_Aesop_enqueueGoals___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___lam__2___closed__0;
lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg(lean_object*);
lean_object* lp_aesop_Aesop_getTree___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__0;
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__16;
lean_object* l_Lean_logWarning___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0;
extern lean_object* lp_aesop_Aesop_aesop_dev_generateScript;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__8;
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3;
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__1;
lean_object* l_ReaderT_pure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__16;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18;
size_t lean_usize_of_nat(lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__11;
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_withTraceNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__0;
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeHasProgress(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__12;
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__15;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__13;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__16;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__9;
static lean_object* lp_aesop_Aesop_traceScript___redArg___lam__2___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__15;
lean_object* lean_st_ref_take(lean_object*);
lean_object* lp_aesop_Aesop_Frontend_getDefaultGlobalRuleSets(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__7;
lean_object* l_instMonadExceptOfEST(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___closed__3;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__0;
lean_object* lp_aesop_Aesop_SearchM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__7;
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEST(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__13;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traceTree___redArg___closed__0;
static lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__5;
lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__0;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__17;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__4;
lean_object* l_IO_instMonadLiftSTRealWorldBaseIO___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_warn_nonterminal;
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__12;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__8;
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__7;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
extern lean_object* l_Lean_KVMap_instValueString;
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
static lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__0;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__8;
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__3;
lean_object* lean_array_to_list(lean_object*);
lean_object* lp_aesop_Aesop_Script_UScript_checkIfEnabled(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__17;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__14;
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__11;
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__25;
extern lean_object* lp_aesop_Aesop_TraceOption_script;
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__16;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__19;
static lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0;
extern lean_object* lp_aesop_Aesop_BaseM_instMonadStats;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__5;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__18;
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__5;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__6;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__4;
uint8_t lean_name_eq(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__4;
lean_object* l_Lean_getExprMVarAssignment_x3f___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0___closed__0;
extern lean_object* l_Lean_Core_instMonadLogCoreM;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__18;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadLCtxMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__18;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_newNodeEmoji;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__11;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__9;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__15;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__2;
static lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3;
lean_object* l_Lean_instMonadAlwaysExceptReaderT___redArg(lean_object*);
lean_object* l_Lean_instAddMessageContextOfMonadLift___redArg(lean_object*, lean_object*);
lean_object* l_Lean_throwMaxRecDepthAt___redArg(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___boxed(lean_object**);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_withPPAnalyze___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_mkLocalRuleSet(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkRappLimit___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__12;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ExtractScriptM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_RegularRule_isUnsafe(lean_object*);
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg___lam__0(lean_object*);
lean_object* lp_aesop_Aesop_checkInvariantsIfEnabled___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lp_aesop_Aesop_getRootMetaState___redArg(lean_object*);
lean_object* lp_aesop_Aesop_Goal_traceTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt(lean_object*, lean_object*, lean_object*, double, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__7;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
extern lean_object* lp_aesop_Aesop_TreeM_instMonad;
static lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__8;
extern lean_object* lp_aesop_Aesop_aesop_smallErrorMessages;
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instantiateMVars___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__0;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__3;
lean_object* lp_aesop_Aesop_getRootMVarId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__2;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__0;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__15;
static lean_object* lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
extern lean_object* lp_aesop_Aesop_TraceOption_steps;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__19;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__0;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__2;
extern lean_object* l_Lean_Meta_instMonadEnvMetaM;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__21;
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_freeTree___redArg(lean_object*);
lean_object* lp_aesop_Aesop_RuleResult_toEmoji___boxed(lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_search___closed__0;
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_ReaderT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__22;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadLiftBaseIOEIO___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_preprocessRule;
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_proof;
lean_object* l_Lean_Meta_getMVarsNoDelayed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonadRef(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__1;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__22;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16;
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__11;
lean_object* l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__18;
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__9;
uint8_t lp_aesop_Aesop_NodeState_isProven(uint8_t);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0(lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeHasProgress___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___boxed(lean_object**);
lean_object* l_instMonadControlReaderT(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed(lean_object**);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Rapp_traceMetadata___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__26;
uint8_t lp_aesop_Aesop_instBEqScopeName_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___closed__0;
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__11;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__12;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkRappLimit___redArg___closed__0;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__0;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__10;
lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__1;
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__4;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__0;
lean_object* l_instMonadLiftT___lam__0___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__14;
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
lean_object* lp_aesop_Aesop_BaseM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1;
lean_object* l_Lean_Core_liftIOCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
extern lean_object* l_Lean_Core_instMonadWithOptionsCoreM;
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__8;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__10;
lean_object* lean_array_uget(lean_object*, size_t);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__5;
lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___boxed(lean_object**);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__13;
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__1;
lean_object* l_Lean_addMessageContextFull___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
lean_object* l_Lean_Name_mkStr1(lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__11;
lean_object* lp_aesop_Aesop_getIteration___redArg(lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_GoalRef_extractScriptCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__8;
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__6;
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__2;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__9;
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instMonadMCtxMetaM;
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_checkRappLimit___redArg___closed__1;
lean_object* lp_aesop_Aesop_getRootGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__2;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__13;
lean_object* l_Lean_instMonadWithOptionsOfMonadFunctor___redArg(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_exceptEmoji___redArg(lean_object*);
lean_object* lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__0;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__7;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2___boxed(lean_object**);
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__3;
extern lean_object* lp_aesop_Aesop_TraceOption_tree;
uint8_t lp_aesop_Aesop_Goal_isActive(lean_object*);
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1;
static lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__2;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__1;
lean_object* l_instMonadLiftTOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__13;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__15;
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg(lean_object*, double, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__12;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__4;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__14;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__10;
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lp_aesop_Aesop_clearForwardImplDetailHyps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__10;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__17;
static lean_object* lp_aesop_Aesop_traceScript___redArg___lam__2___closed__2;
static lean_object* lp_aesop_Aesop_checkRappLimit___redArg___closed__3;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__14;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__2;
uint8_t l_Lean_Expr_hasExprMVar(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__21;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0;
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = l_Lean_instMonadExceptOfExceptionCoreM;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = l_Lean_instMonadExceptOfExceptionCoreM;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__3;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__2;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__4;
x_2 = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__4;
x_2 = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__6;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__5;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__7;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__7;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__9;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__8;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__10;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__10;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__12;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__11;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__13;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__13;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__19;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__18;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__22() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__20;
x_2 = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__21() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__20;
x_2 = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__22;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__21;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__1() {
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
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = l_Lean_Meta_instAddMessageContextMetaM;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_3 = l_Lean_instAddMessageContextOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__14;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_3 = l_Lean_instAddMessageContextOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__15;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_3 = l_Lean_instAddMessageContextOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__17() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__16;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
x_3 = l_Lean_instAddMessageContextOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__25() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop/expandNextGoal: internal error: no active goals left", 58, 58);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__26() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__25;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_11 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
x_12 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_13 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_14 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_11);
x_15 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_14, x_11);
x_16 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_16, 1, x_13);
lean_ctor_set(x_16, 2, x_15);
lean_inc_ref(x_1);
x_17 = lp_aesop_Aesop_popGoal_x3f___redArg(x_1, x_3);
if (lean_obj_tag(x_17) == 0)
{
uint8_t x_18; 
x_18 = !lean_is_exclusive(x_17);
if (x_18 == 0)
{
lean_object* x_19; 
x_19 = lean_ctor_get(x_17, 0);
if (lean_obj_tag(x_19) == 1)
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; 
lean_dec_ref(x_16);
lean_dec_ref(x_11);
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = lean_st_ref_get(x_3);
lean_dec(x_21);
x_22 = lean_st_ref_get(x_20);
x_23 = lean_st_ref_get(x_3);
lean_dec(x_23);
x_24 = lp_aesop_Aesop_Goal_isActive(x_22);
if (x_24 == 0)
{
lean_dec(x_20);
lean_free_object(x_17);
goto _start;
}
else
{
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
lean_ctor_set(x_17, 0, x_20);
return x_17;
}
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; 
lean_free_object(x_17);
lean_dec(x_19);
lean_dec_ref(x_1);
x_26 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__26;
x_27 = l_Lean_throwError___redArg(x_11, x_16, x_26);
x_28 = lean_apply_9(x_27, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_28;
}
}
else
{
lean_object* x_29; 
x_29 = lean_ctor_get(x_17, 0);
lean_inc(x_29);
lean_dec(x_17);
if (lean_obj_tag(x_29) == 1)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; uint8_t x_34; 
lean_dec_ref(x_16);
lean_dec_ref(x_11);
x_30 = lean_ctor_get(x_29, 0);
lean_inc(x_30);
lean_dec_ref(x_29);
x_31 = lean_st_ref_get(x_3);
lean_dec(x_31);
x_32 = lean_st_ref_get(x_30);
x_33 = lean_st_ref_get(x_3);
lean_dec(x_33);
x_34 = lp_aesop_Aesop_Goal_isActive(x_32);
if (x_34 == 0)
{
lean_dec(x_30);
goto _start;
}
else
{
lean_object* x_36; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_36 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_36, 0, x_30);
return x_36;
}
}
else
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; 
lean_dec(x_29);
lean_dec_ref(x_1);
x_37 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__26;
x_38 = l_Lean_throwError___redArg(x_11, x_16, x_37);
x_39 = lean_apply_9(x_38, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_39;
}
}
}
else
{
uint8_t x_40; 
lean_dec_ref(x_16);
lean_dec_ref(x_11);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_40 = !lean_is_exclusive(x_17);
if (x_40 == 0)
{
return x_17;
}
else
{
lean_object* x_41; lean_object* x_42; 
x_41 = lean_ctor_get(x_17, 0);
lean_inc(x_41);
lean_dec(x_17);
x_42 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_42, 0, x_41);
return x_42;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_nextActiveGoal___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_nextActiveGoal(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_nextActiveGoal___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_instMonadEST(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0;
x_2 = l_ReaderT_instMonad___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Core_instMonadCoreM___lam__0___boxed), 5, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Core_instMonadCoreM___lam__1___boxed), 7, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_instMonadMetaM___lam__0___boxed), 7, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_instMonadMetaM___lam__1___boxed), 9, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_instMonadLCtxMetaM___lam__0___boxed), 5, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed), 3, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__7;
x_2 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, lean_box(0));
lean_closure_set(x_2, 3, lean_box(0));
lean_closure_set(x_2, 4, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__8;
x_2 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; 
x_7 = l_Lean_MVarId_getType(x_1, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_7) == 0)
{
lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1;
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_11 = lean_ctor_get(x_9, 0);
x_12 = lean_ctor_get(x_9, 1);
lean_dec(x_12);
x_13 = !lean_is_exclusive(x_11);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; uint8_t x_28; 
x_14 = lean_ctor_get(x_11, 0);
x_15 = lean_ctor_get(x_11, 2);
x_16 = lean_ctor_get(x_11, 3);
x_17 = lean_ctor_get(x_11, 4);
x_18 = lean_ctor_get(x_11, 1);
lean_dec(x_18);
x_19 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_20 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_14);
x_21 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_21, 0, x_14);
x_22 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_22, 0, x_14);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_22);
x_24 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_24, 0, x_17);
x_25 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_25, 0, x_16);
x_26 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_26, 0, x_15);
lean_ctor_set(x_11, 4, x_24);
lean_ctor_set(x_11, 3, x_25);
lean_ctor_set(x_11, 2, x_26);
lean_ctor_set(x_11, 1, x_19);
lean_ctor_set(x_11, 0, x_23);
lean_ctor_set(x_9, 1, x_20);
x_27 = l_ReaderT_instMonad___redArg(x_9);
x_28 = !lean_is_exclusive(x_27);
if (x_28 == 0)
{
lean_object* x_29; lean_object* x_30; uint8_t x_31; 
x_29 = lean_ctor_get(x_27, 0);
x_30 = lean_ctor_get(x_27, 1);
lean_dec(x_30);
x_31 = !lean_is_exclusive(x_29);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; 
x_32 = lean_ctor_get(x_29, 0);
x_33 = lean_ctor_get(x_29, 2);
x_34 = lean_ctor_get(x_29, 3);
x_35 = lean_ctor_get(x_29, 4);
x_36 = lean_ctor_get(x_29, 1);
lean_dec(x_36);
x_37 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_38 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_32);
x_39 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_39, 0, x_32);
x_40 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_40, 0, x_32);
x_41 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_41, 0, x_39);
lean_ctor_set(x_41, 1, x_40);
x_42 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_42, 0, x_35);
x_43 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_43, 0, x_34);
x_44 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_44, 0, x_33);
lean_ctor_set(x_29, 4, x_42);
lean_ctor_set(x_29, 3, x_43);
lean_ctor_set(x_29, 2, x_44);
lean_ctor_set(x_29, 1, x_37);
lean_ctor_set(x_29, 0, x_41);
lean_ctor_set(x_27, 1, x_38);
x_45 = l_Lean_Meta_instMonadEnvMetaM;
x_46 = l_Lean_Meta_instMonadMCtxMetaM;
x_47 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6;
x_48 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
x_49 = l_Lean_MessageData_ofExpr(x_8);
x_50 = l_Lean_addMessageContextFull___redArg(x_27, x_45, x_46, x_47, x_48, x_49);
x_51 = lean_apply_5(x_50, x_2, x_3, x_4, x_5, lean_box(0));
return x_51;
}
else
{
lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; 
x_52 = lean_ctor_get(x_29, 0);
x_53 = lean_ctor_get(x_29, 2);
x_54 = lean_ctor_get(x_29, 3);
x_55 = lean_ctor_get(x_29, 4);
lean_inc(x_55);
lean_inc(x_54);
lean_inc(x_53);
lean_inc(x_52);
lean_dec(x_29);
x_56 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_57 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_52);
x_58 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_58, 0, x_52);
x_59 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_59, 0, x_52);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_58);
lean_ctor_set(x_60, 1, x_59);
x_61 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_61, 0, x_55);
x_62 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_62, 0, x_54);
x_63 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_63, 0, x_53);
x_64 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_64, 0, x_60);
lean_ctor_set(x_64, 1, x_56);
lean_ctor_set(x_64, 2, x_63);
lean_ctor_set(x_64, 3, x_62);
lean_ctor_set(x_64, 4, x_61);
lean_ctor_set(x_27, 1, x_57);
lean_ctor_set(x_27, 0, x_64);
x_65 = l_Lean_Meta_instMonadEnvMetaM;
x_66 = l_Lean_Meta_instMonadMCtxMetaM;
x_67 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6;
x_68 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
x_69 = l_Lean_MessageData_ofExpr(x_8);
x_70 = l_Lean_addMessageContextFull___redArg(x_27, x_65, x_66, x_67, x_68, x_69);
x_71 = lean_apply_5(x_70, x_2, x_3, x_4, x_5, lean_box(0));
return x_71;
}
}
else
{
lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; 
x_72 = lean_ctor_get(x_27, 0);
lean_inc(x_72);
lean_dec(x_27);
x_73 = lean_ctor_get(x_72, 0);
lean_inc_ref(x_73);
x_74 = lean_ctor_get(x_72, 2);
lean_inc(x_74);
x_75 = lean_ctor_get(x_72, 3);
lean_inc(x_75);
x_76 = lean_ctor_get(x_72, 4);
lean_inc(x_76);
if (lean_is_exclusive(x_72)) {
 lean_ctor_release(x_72, 0);
 lean_ctor_release(x_72, 1);
 lean_ctor_release(x_72, 2);
 lean_ctor_release(x_72, 3);
 lean_ctor_release(x_72, 4);
 x_77 = x_72;
} else {
 lean_dec_ref(x_72);
 x_77 = lean_box(0);
}
x_78 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_79 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_73);
x_80 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_80, 0, x_73);
x_81 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_81, 0, x_73);
x_82 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_82, 0, x_80);
lean_ctor_set(x_82, 1, x_81);
x_83 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_83, 0, x_76);
x_84 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_84, 0, x_75);
x_85 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_85, 0, x_74);
if (lean_is_scalar(x_77)) {
 x_86 = lean_alloc_ctor(0, 5, 0);
} else {
 x_86 = x_77;
}
lean_ctor_set(x_86, 0, x_82);
lean_ctor_set(x_86, 1, x_78);
lean_ctor_set(x_86, 2, x_85);
lean_ctor_set(x_86, 3, x_84);
lean_ctor_set(x_86, 4, x_83);
x_87 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_87, 0, x_86);
lean_ctor_set(x_87, 1, x_79);
x_88 = l_Lean_Meta_instMonadEnvMetaM;
x_89 = l_Lean_Meta_instMonadMCtxMetaM;
x_90 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6;
x_91 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
x_92 = l_Lean_MessageData_ofExpr(x_8);
x_93 = l_Lean_addMessageContextFull___redArg(x_87, x_88, x_89, x_90, x_91, x_92);
x_94 = lean_apply_5(x_93, x_2, x_3, x_4, x_5, lean_box(0));
return x_94;
}
}
else
{
lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; 
x_95 = lean_ctor_get(x_11, 0);
x_96 = lean_ctor_get(x_11, 2);
x_97 = lean_ctor_get(x_11, 3);
x_98 = lean_ctor_get(x_11, 4);
lean_inc(x_98);
lean_inc(x_97);
lean_inc(x_96);
lean_inc(x_95);
lean_dec(x_11);
x_99 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_100 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_95);
x_101 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_101, 0, x_95);
x_102 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_102, 0, x_95);
x_103 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_103, 0, x_101);
lean_ctor_set(x_103, 1, x_102);
x_104 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_104, 0, x_98);
x_105 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_105, 0, x_97);
x_106 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_106, 0, x_96);
x_107 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_107, 0, x_103);
lean_ctor_set(x_107, 1, x_99);
lean_ctor_set(x_107, 2, x_106);
lean_ctor_set(x_107, 3, x_105);
lean_ctor_set(x_107, 4, x_104);
lean_ctor_set(x_9, 1, x_100);
lean_ctor_set(x_9, 0, x_107);
x_108 = l_ReaderT_instMonad___redArg(x_9);
x_109 = lean_ctor_get(x_108, 0);
lean_inc_ref(x_109);
if (lean_is_exclusive(x_108)) {
 lean_ctor_release(x_108, 0);
 lean_ctor_release(x_108, 1);
 x_110 = x_108;
} else {
 lean_dec_ref(x_108);
 x_110 = lean_box(0);
}
x_111 = lean_ctor_get(x_109, 0);
lean_inc_ref(x_111);
x_112 = lean_ctor_get(x_109, 2);
lean_inc(x_112);
x_113 = lean_ctor_get(x_109, 3);
lean_inc(x_113);
x_114 = lean_ctor_get(x_109, 4);
lean_inc(x_114);
if (lean_is_exclusive(x_109)) {
 lean_ctor_release(x_109, 0);
 lean_ctor_release(x_109, 1);
 lean_ctor_release(x_109, 2);
 lean_ctor_release(x_109, 3);
 lean_ctor_release(x_109, 4);
 x_115 = x_109;
} else {
 lean_dec_ref(x_109);
 x_115 = lean_box(0);
}
x_116 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_117 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_111);
x_118 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_118, 0, x_111);
x_119 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_119, 0, x_111);
x_120 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_120, 0, x_118);
lean_ctor_set(x_120, 1, x_119);
x_121 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_121, 0, x_114);
x_122 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_122, 0, x_113);
x_123 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_123, 0, x_112);
if (lean_is_scalar(x_115)) {
 x_124 = lean_alloc_ctor(0, 5, 0);
} else {
 x_124 = x_115;
}
lean_ctor_set(x_124, 0, x_120);
lean_ctor_set(x_124, 1, x_116);
lean_ctor_set(x_124, 2, x_123);
lean_ctor_set(x_124, 3, x_122);
lean_ctor_set(x_124, 4, x_121);
if (lean_is_scalar(x_110)) {
 x_125 = lean_alloc_ctor(0, 2, 0);
} else {
 x_125 = x_110;
}
lean_ctor_set(x_125, 0, x_124);
lean_ctor_set(x_125, 1, x_117);
x_126 = l_Lean_Meta_instMonadEnvMetaM;
x_127 = l_Lean_Meta_instMonadMCtxMetaM;
x_128 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6;
x_129 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
x_130 = l_Lean_MessageData_ofExpr(x_8);
x_131 = l_Lean_addMessageContextFull___redArg(x_125, x_126, x_127, x_128, x_129, x_130);
x_132 = lean_apply_5(x_131, x_2, x_3, x_4, x_5, lean_box(0));
return x_132;
}
}
else
{
lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; 
x_133 = lean_ctor_get(x_9, 0);
lean_inc(x_133);
lean_dec(x_9);
x_134 = lean_ctor_get(x_133, 0);
lean_inc_ref(x_134);
x_135 = lean_ctor_get(x_133, 2);
lean_inc(x_135);
x_136 = lean_ctor_get(x_133, 3);
lean_inc(x_136);
x_137 = lean_ctor_get(x_133, 4);
lean_inc(x_137);
if (lean_is_exclusive(x_133)) {
 lean_ctor_release(x_133, 0);
 lean_ctor_release(x_133, 1);
 lean_ctor_release(x_133, 2);
 lean_ctor_release(x_133, 3);
 lean_ctor_release(x_133, 4);
 x_138 = x_133;
} else {
 lean_dec_ref(x_133);
 x_138 = lean_box(0);
}
x_139 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_140 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
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
x_148 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_148, 0, x_147);
lean_ctor_set(x_148, 1, x_140);
x_149 = l_ReaderT_instMonad___redArg(x_148);
x_150 = lean_ctor_get(x_149, 0);
lean_inc_ref(x_150);
if (lean_is_exclusive(x_149)) {
 lean_ctor_release(x_149, 0);
 lean_ctor_release(x_149, 1);
 x_151 = x_149;
} else {
 lean_dec_ref(x_149);
 x_151 = lean_box(0);
}
x_152 = lean_ctor_get(x_150, 0);
lean_inc_ref(x_152);
x_153 = lean_ctor_get(x_150, 2);
lean_inc(x_153);
x_154 = lean_ctor_get(x_150, 3);
lean_inc(x_154);
x_155 = lean_ctor_get(x_150, 4);
lean_inc(x_155);
if (lean_is_exclusive(x_150)) {
 lean_ctor_release(x_150, 0);
 lean_ctor_release(x_150, 1);
 lean_ctor_release(x_150, 2);
 lean_ctor_release(x_150, 3);
 lean_ctor_release(x_150, 4);
 x_156 = x_150;
} else {
 lean_dec_ref(x_150);
 x_156 = lean_box(0);
}
x_157 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_158 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_152);
x_159 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_159, 0, x_152);
x_160 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_160, 0, x_152);
x_161 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_161, 0, x_159);
lean_ctor_set(x_161, 1, x_160);
x_162 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_162, 0, x_155);
x_163 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_163, 0, x_154);
x_164 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_164, 0, x_153);
if (lean_is_scalar(x_156)) {
 x_165 = lean_alloc_ctor(0, 5, 0);
} else {
 x_165 = x_156;
}
lean_ctor_set(x_165, 0, x_161);
lean_ctor_set(x_165, 1, x_157);
lean_ctor_set(x_165, 2, x_164);
lean_ctor_set(x_165, 3, x_163);
lean_ctor_set(x_165, 4, x_162);
if (lean_is_scalar(x_151)) {
 x_166 = lean_alloc_ctor(0, 2, 0);
} else {
 x_166 = x_151;
}
lean_ctor_set(x_166, 0, x_165);
lean_ctor_set(x_166, 1, x_158);
x_167 = l_Lean_Meta_instMonadEnvMetaM;
x_168 = l_Lean_Meta_instMonadMCtxMetaM;
x_169 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6;
x_170 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
x_171 = l_Lean_MessageData_ofExpr(x_8);
x_172 = l_Lean_addMessageContextFull___redArg(x_166, x_167, x_168, x_169, x_170, x_171);
x_173 = lean_apply_5(x_172, x_2, x_3, x_4, x_5, lean_box(0));
return x_173;
}
}
else
{
uint8_t x_174; 
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_174 = !lean_is_exclusive(x_7);
if (x_174 == 0)
{
return x_7;
}
else
{
lean_object* x_175; lean_object* x_176; 
x_175 = lean_ctor_get(x_7, 0);
lean_inc(x_175);
lean_dec(x_7);
x_176 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_176, 0, x_175);
return x_176;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleResult_toEmoji___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" (G", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__1;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(") [", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__3;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("] ⋯ ⊢ ", 10, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__5;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg(lean_object* x_1, double x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_12 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1;
x_13 = lean_ctor_get(x_12, 0);
lean_inc_ref(x_13);
x_14 = !lean_is_exclusive(x_13);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_15 = lean_ctor_get(x_13, 0);
x_16 = lean_ctor_get(x_13, 2);
x_17 = lean_ctor_get(x_13, 3);
x_18 = lean_ctor_get(x_13, 4);
x_19 = lean_ctor_get(x_13, 1);
lean_dec(x_19);
x_20 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_21 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
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
lean_ctor_set(x_13, 4, x_25);
lean_ctor_set(x_13, 3, x_26);
lean_ctor_set(x_13, 2, x_27);
lean_ctor_set(x_13, 1, x_20);
lean_ctor_set(x_13, 0, x_24);
x_28 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_28, 0, x_13);
lean_ctor_set(x_28, 1, x_21);
x_29 = l_ReaderT_instMonad___redArg(x_28);
x_30 = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, lean_box(0));
lean_closure_set(x_30, 2, x_29);
x_31 = l_instMonadControlTOfPure___redArg(x_30);
x_32 = !lean_is_exclusive(x_12);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; uint8_t x_35; 
x_33 = lean_ctor_get(x_12, 0);
x_34 = lean_ctor_get(x_12, 1);
lean_dec(x_34);
x_35 = !lean_is_exclusive(x_33);
if (x_35 == 0)
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; uint8_t x_48; 
x_36 = lean_ctor_get(x_33, 0);
x_37 = lean_ctor_get(x_33, 2);
x_38 = lean_ctor_get(x_33, 3);
x_39 = lean_ctor_get(x_33, 4);
x_40 = lean_ctor_get(x_33, 1);
lean_dec(x_40);
lean_inc_ref(x_36);
x_41 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_41, 0, x_36);
x_42 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_42, 0, x_36);
x_43 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_43, 0, x_41);
lean_ctor_set(x_43, 1, x_42);
x_44 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_44, 0, x_39);
x_45 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_45, 0, x_38);
x_46 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_46, 0, x_37);
lean_ctor_set(x_33, 4, x_44);
lean_ctor_set(x_33, 3, x_45);
lean_ctor_set(x_33, 2, x_46);
lean_ctor_set(x_33, 1, x_20);
lean_ctor_set(x_33, 0, x_43);
lean_ctor_set(x_12, 1, x_21);
x_47 = l_ReaderT_instMonad___redArg(x_12);
x_48 = !lean_is_exclusive(x_47);
if (x_48 == 0)
{
lean_object* x_49; lean_object* x_50; uint8_t x_51; 
x_49 = lean_ctor_get(x_47, 0);
x_50 = lean_ctor_get(x_47, 1);
lean_dec(x_50);
x_51 = !lean_is_exclusive(x_49);
if (x_51 == 0)
{
lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_52 = lean_ctor_get(x_49, 0);
x_53 = lean_ctor_get(x_49, 2);
x_54 = lean_ctor_get(x_49, 3);
x_55 = lean_ctor_get(x_49, 4);
x_56 = lean_ctor_get(x_49, 1);
lean_dec(x_56);
x_57 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_58 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_52);
x_59 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_59, 0, x_52);
x_60 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_60, 0, x_52);
x_61 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_61, 0, x_59);
lean_ctor_set(x_61, 1, x_60);
x_62 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_62, 0, x_55);
x_63 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_63, 0, x_54);
x_64 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_64, 0, x_53);
lean_ctor_set(x_49, 4, x_62);
lean_ctor_set(x_49, 3, x_63);
lean_ctor_set(x_49, 2, x_64);
lean_ctor_set(x_49, 1, x_57);
lean_ctor_set(x_49, 0, x_61);
lean_ctor_set(x_47, 1, x_58);
x_65 = lean_st_ref_get(x_6);
lean_dec(x_65);
lean_inc(x_3);
x_66 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___boxed), 6, 1);
lean_closure_set(x_66, 0, x_3);
x_67 = l_Lean_MVarId_withContext___redArg(x_31, x_47, x_3, x_66);
x_68 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_4, x_67, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_68) == 0)
{
uint8_t x_69; 
x_69 = !lean_is_exclusive(x_68);
if (x_69 == 0)
{
lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; 
x_70 = lean_ctor_get(x_68, 0);
x_71 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0;
x_72 = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(x_71, x_5);
x_73 = l_Lean_stringToMessageData(x_72);
x_74 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2;
x_75 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_75, 0, x_73);
lean_ctor_set(x_75, 1, x_74);
x_76 = l_Nat_reprFast(x_1);
x_77 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_77, 0, x_76);
x_78 = l_Lean_MessageData_ofFormat(x_77);
x_79 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_79, 0, x_75);
lean_ctor_set(x_79, 1, x_78);
x_80 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4;
x_81 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_81, 0, x_79);
lean_ctor_set(x_81, 1, x_80);
x_82 = lp_aesop_Aesop_Percent_toHumanString(x_2);
x_83 = l_Lean_stringToMessageData(x_82);
x_84 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_84, 0, x_81);
lean_ctor_set(x_84, 1, x_83);
x_85 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6;
x_86 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_86, 0, x_84);
lean_ctor_set(x_86, 1, x_85);
x_87 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_87, 0, x_86);
lean_ctor_set(x_87, 1, x_70);
lean_ctor_set(x_68, 0, x_87);
return x_68;
}
else
{
lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; 
x_88 = lean_ctor_get(x_68, 0);
lean_inc(x_88);
lean_dec(x_68);
x_89 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0;
x_90 = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(x_89, x_5);
x_91 = l_Lean_stringToMessageData(x_90);
x_92 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2;
x_93 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_93, 0, x_91);
lean_ctor_set(x_93, 1, x_92);
x_94 = l_Nat_reprFast(x_1);
x_95 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_95, 0, x_94);
x_96 = l_Lean_MessageData_ofFormat(x_95);
x_97 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_97, 0, x_93);
lean_ctor_set(x_97, 1, x_96);
x_98 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4;
x_99 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_99, 0, x_97);
lean_ctor_set(x_99, 1, x_98);
x_100 = lp_aesop_Aesop_Percent_toHumanString(x_2);
x_101 = l_Lean_stringToMessageData(x_100);
x_102 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_102, 0, x_99);
lean_ctor_set(x_102, 1, x_101);
x_103 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6;
x_104 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_104, 0, x_102);
lean_ctor_set(x_104, 1, x_103);
x_105 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_105, 0, x_104);
lean_ctor_set(x_105, 1, x_88);
x_106 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_106, 0, x_105);
return x_106;
}
}
else
{
lean_dec_ref(x_5);
lean_dec(x_1);
return x_68;
}
}
else
{
lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; 
x_107 = lean_ctor_get(x_49, 0);
x_108 = lean_ctor_get(x_49, 2);
x_109 = lean_ctor_get(x_49, 3);
x_110 = lean_ctor_get(x_49, 4);
lean_inc(x_110);
lean_inc(x_109);
lean_inc(x_108);
lean_inc(x_107);
lean_dec(x_49);
x_111 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_112 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
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
lean_ctor_set(x_47, 1, x_112);
lean_ctor_set(x_47, 0, x_119);
x_120 = lean_st_ref_get(x_6);
lean_dec(x_120);
lean_inc(x_3);
x_121 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___boxed), 6, 1);
lean_closure_set(x_121, 0, x_3);
x_122 = l_Lean_MVarId_withContext___redArg(x_31, x_47, x_3, x_121);
x_123 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_4, x_122, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_123) == 0)
{
lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; 
x_124 = lean_ctor_get(x_123, 0);
lean_inc(x_124);
if (lean_is_exclusive(x_123)) {
 lean_ctor_release(x_123, 0);
 x_125 = x_123;
} else {
 lean_dec_ref(x_123);
 x_125 = lean_box(0);
}
x_126 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0;
x_127 = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(x_126, x_5);
x_128 = l_Lean_stringToMessageData(x_127);
x_129 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2;
x_130 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_130, 0, x_128);
lean_ctor_set(x_130, 1, x_129);
x_131 = l_Nat_reprFast(x_1);
x_132 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_132, 0, x_131);
x_133 = l_Lean_MessageData_ofFormat(x_132);
x_134 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_134, 0, x_130);
lean_ctor_set(x_134, 1, x_133);
x_135 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4;
x_136 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_136, 0, x_134);
lean_ctor_set(x_136, 1, x_135);
x_137 = lp_aesop_Aesop_Percent_toHumanString(x_2);
x_138 = l_Lean_stringToMessageData(x_137);
x_139 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_139, 0, x_136);
lean_ctor_set(x_139, 1, x_138);
x_140 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6;
x_141 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_141, 0, x_139);
lean_ctor_set(x_141, 1, x_140);
x_142 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_142, 0, x_141);
lean_ctor_set(x_142, 1, x_124);
if (lean_is_scalar(x_125)) {
 x_143 = lean_alloc_ctor(0, 1, 0);
} else {
 x_143 = x_125;
}
lean_ctor_set(x_143, 0, x_142);
return x_143;
}
else
{
lean_dec_ref(x_5);
lean_dec(x_1);
return x_123;
}
}
}
else
{
lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; 
x_144 = lean_ctor_get(x_47, 0);
lean_inc(x_144);
lean_dec(x_47);
x_145 = lean_ctor_get(x_144, 0);
lean_inc_ref(x_145);
x_146 = lean_ctor_get(x_144, 2);
lean_inc(x_146);
x_147 = lean_ctor_get(x_144, 3);
lean_inc(x_147);
x_148 = lean_ctor_get(x_144, 4);
lean_inc(x_148);
if (lean_is_exclusive(x_144)) {
 lean_ctor_release(x_144, 0);
 lean_ctor_release(x_144, 1);
 lean_ctor_release(x_144, 2);
 lean_ctor_release(x_144, 3);
 lean_ctor_release(x_144, 4);
 x_149 = x_144;
} else {
 lean_dec_ref(x_144);
 x_149 = lean_box(0);
}
x_150 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_151 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_145);
x_152 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_152, 0, x_145);
x_153 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_153, 0, x_145);
x_154 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_154, 0, x_152);
lean_ctor_set(x_154, 1, x_153);
x_155 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_155, 0, x_148);
x_156 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_156, 0, x_147);
x_157 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_157, 0, x_146);
if (lean_is_scalar(x_149)) {
 x_158 = lean_alloc_ctor(0, 5, 0);
} else {
 x_158 = x_149;
}
lean_ctor_set(x_158, 0, x_154);
lean_ctor_set(x_158, 1, x_150);
lean_ctor_set(x_158, 2, x_157);
lean_ctor_set(x_158, 3, x_156);
lean_ctor_set(x_158, 4, x_155);
x_159 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_159, 0, x_158);
lean_ctor_set(x_159, 1, x_151);
x_160 = lean_st_ref_get(x_6);
lean_dec(x_160);
lean_inc(x_3);
x_161 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___boxed), 6, 1);
lean_closure_set(x_161, 0, x_3);
x_162 = l_Lean_MVarId_withContext___redArg(x_31, x_159, x_3, x_161);
x_163 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_4, x_162, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_163) == 0)
{
lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; 
x_164 = lean_ctor_get(x_163, 0);
lean_inc(x_164);
if (lean_is_exclusive(x_163)) {
 lean_ctor_release(x_163, 0);
 x_165 = x_163;
} else {
 lean_dec_ref(x_163);
 x_165 = lean_box(0);
}
x_166 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0;
x_167 = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(x_166, x_5);
x_168 = l_Lean_stringToMessageData(x_167);
x_169 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2;
x_170 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_170, 0, x_168);
lean_ctor_set(x_170, 1, x_169);
x_171 = l_Nat_reprFast(x_1);
x_172 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_172, 0, x_171);
x_173 = l_Lean_MessageData_ofFormat(x_172);
x_174 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_174, 0, x_170);
lean_ctor_set(x_174, 1, x_173);
x_175 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4;
x_176 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_176, 0, x_174);
lean_ctor_set(x_176, 1, x_175);
x_177 = lp_aesop_Aesop_Percent_toHumanString(x_2);
x_178 = l_Lean_stringToMessageData(x_177);
x_179 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_179, 0, x_176);
lean_ctor_set(x_179, 1, x_178);
x_180 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6;
x_181 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_181, 0, x_179);
lean_ctor_set(x_181, 1, x_180);
x_182 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_182, 0, x_181);
lean_ctor_set(x_182, 1, x_164);
if (lean_is_scalar(x_165)) {
 x_183 = lean_alloc_ctor(0, 1, 0);
} else {
 x_183 = x_165;
}
lean_ctor_set(x_183, 0, x_182);
return x_183;
}
else
{
lean_dec_ref(x_5);
lean_dec(x_1);
return x_163;
}
}
}
else
{
lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; 
x_184 = lean_ctor_get(x_33, 0);
x_185 = lean_ctor_get(x_33, 2);
x_186 = lean_ctor_get(x_33, 3);
x_187 = lean_ctor_get(x_33, 4);
lean_inc(x_187);
lean_inc(x_186);
lean_inc(x_185);
lean_inc(x_184);
lean_dec(x_33);
lean_inc_ref(x_184);
x_188 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_188, 0, x_184);
x_189 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_189, 0, x_184);
x_190 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_190, 0, x_188);
lean_ctor_set(x_190, 1, x_189);
x_191 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_191, 0, x_187);
x_192 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_192, 0, x_186);
x_193 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_193, 0, x_185);
x_194 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_194, 0, x_190);
lean_ctor_set(x_194, 1, x_20);
lean_ctor_set(x_194, 2, x_193);
lean_ctor_set(x_194, 3, x_192);
lean_ctor_set(x_194, 4, x_191);
lean_ctor_set(x_12, 1, x_21);
lean_ctor_set(x_12, 0, x_194);
x_195 = l_ReaderT_instMonad___redArg(x_12);
x_196 = lean_ctor_get(x_195, 0);
lean_inc_ref(x_196);
if (lean_is_exclusive(x_195)) {
 lean_ctor_release(x_195, 0);
 lean_ctor_release(x_195, 1);
 x_197 = x_195;
} else {
 lean_dec_ref(x_195);
 x_197 = lean_box(0);
}
x_198 = lean_ctor_get(x_196, 0);
lean_inc_ref(x_198);
x_199 = lean_ctor_get(x_196, 2);
lean_inc(x_199);
x_200 = lean_ctor_get(x_196, 3);
lean_inc(x_200);
x_201 = lean_ctor_get(x_196, 4);
lean_inc(x_201);
if (lean_is_exclusive(x_196)) {
 lean_ctor_release(x_196, 0);
 lean_ctor_release(x_196, 1);
 lean_ctor_release(x_196, 2);
 lean_ctor_release(x_196, 3);
 lean_ctor_release(x_196, 4);
 x_202 = x_196;
} else {
 lean_dec_ref(x_196);
 x_202 = lean_box(0);
}
x_203 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_204 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_198);
x_205 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_205, 0, x_198);
x_206 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_206, 0, x_198);
x_207 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_207, 0, x_205);
lean_ctor_set(x_207, 1, x_206);
x_208 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_208, 0, x_201);
x_209 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_209, 0, x_200);
x_210 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_210, 0, x_199);
if (lean_is_scalar(x_202)) {
 x_211 = lean_alloc_ctor(0, 5, 0);
} else {
 x_211 = x_202;
}
lean_ctor_set(x_211, 0, x_207);
lean_ctor_set(x_211, 1, x_203);
lean_ctor_set(x_211, 2, x_210);
lean_ctor_set(x_211, 3, x_209);
lean_ctor_set(x_211, 4, x_208);
if (lean_is_scalar(x_197)) {
 x_212 = lean_alloc_ctor(0, 2, 0);
} else {
 x_212 = x_197;
}
lean_ctor_set(x_212, 0, x_211);
lean_ctor_set(x_212, 1, x_204);
x_213 = lean_st_ref_get(x_6);
lean_dec(x_213);
lean_inc(x_3);
x_214 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___boxed), 6, 1);
lean_closure_set(x_214, 0, x_3);
x_215 = l_Lean_MVarId_withContext___redArg(x_31, x_212, x_3, x_214);
x_216 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_4, x_215, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_216) == 0)
{
lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; lean_object* x_236; 
x_217 = lean_ctor_get(x_216, 0);
lean_inc(x_217);
if (lean_is_exclusive(x_216)) {
 lean_ctor_release(x_216, 0);
 x_218 = x_216;
} else {
 lean_dec_ref(x_216);
 x_218 = lean_box(0);
}
x_219 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0;
x_220 = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(x_219, x_5);
x_221 = l_Lean_stringToMessageData(x_220);
x_222 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2;
x_223 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_223, 0, x_221);
lean_ctor_set(x_223, 1, x_222);
x_224 = l_Nat_reprFast(x_1);
x_225 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_225, 0, x_224);
x_226 = l_Lean_MessageData_ofFormat(x_225);
x_227 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_227, 0, x_223);
lean_ctor_set(x_227, 1, x_226);
x_228 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4;
x_229 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_229, 0, x_227);
lean_ctor_set(x_229, 1, x_228);
x_230 = lp_aesop_Aesop_Percent_toHumanString(x_2);
x_231 = l_Lean_stringToMessageData(x_230);
x_232 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_232, 0, x_229);
lean_ctor_set(x_232, 1, x_231);
x_233 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6;
x_234 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_234, 0, x_232);
lean_ctor_set(x_234, 1, x_233);
x_235 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_235, 0, x_234);
lean_ctor_set(x_235, 1, x_217);
if (lean_is_scalar(x_218)) {
 x_236 = lean_alloc_ctor(0, 1, 0);
} else {
 x_236 = x_218;
}
lean_ctor_set(x_236, 0, x_235);
return x_236;
}
else
{
lean_dec_ref(x_5);
lean_dec(x_1);
return x_216;
}
}
}
else
{
lean_object* x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; lean_object* x_241; lean_object* x_242; lean_object* x_243; lean_object* x_244; lean_object* x_245; lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; lean_object* x_259; lean_object* x_260; lean_object* x_261; lean_object* x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; lean_object* x_269; lean_object* x_270; lean_object* x_271; lean_object* x_272; 
x_237 = lean_ctor_get(x_12, 0);
lean_inc(x_237);
lean_dec(x_12);
x_238 = lean_ctor_get(x_237, 0);
lean_inc_ref(x_238);
x_239 = lean_ctor_get(x_237, 2);
lean_inc(x_239);
x_240 = lean_ctor_get(x_237, 3);
lean_inc(x_240);
x_241 = lean_ctor_get(x_237, 4);
lean_inc(x_241);
if (lean_is_exclusive(x_237)) {
 lean_ctor_release(x_237, 0);
 lean_ctor_release(x_237, 1);
 lean_ctor_release(x_237, 2);
 lean_ctor_release(x_237, 3);
 lean_ctor_release(x_237, 4);
 x_242 = x_237;
} else {
 lean_dec_ref(x_237);
 x_242 = lean_box(0);
}
lean_inc_ref(x_238);
x_243 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_243, 0, x_238);
x_244 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_244, 0, x_238);
x_245 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_245, 0, x_243);
lean_ctor_set(x_245, 1, x_244);
x_246 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_246, 0, x_241);
x_247 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_247, 0, x_240);
x_248 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_248, 0, x_239);
if (lean_is_scalar(x_242)) {
 x_249 = lean_alloc_ctor(0, 5, 0);
} else {
 x_249 = x_242;
}
lean_ctor_set(x_249, 0, x_245);
lean_ctor_set(x_249, 1, x_20);
lean_ctor_set(x_249, 2, x_248);
lean_ctor_set(x_249, 3, x_247);
lean_ctor_set(x_249, 4, x_246);
x_250 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_250, 0, x_249);
lean_ctor_set(x_250, 1, x_21);
x_251 = l_ReaderT_instMonad___redArg(x_250);
x_252 = lean_ctor_get(x_251, 0);
lean_inc_ref(x_252);
if (lean_is_exclusive(x_251)) {
 lean_ctor_release(x_251, 0);
 lean_ctor_release(x_251, 1);
 x_253 = x_251;
} else {
 lean_dec_ref(x_251);
 x_253 = lean_box(0);
}
x_254 = lean_ctor_get(x_252, 0);
lean_inc_ref(x_254);
x_255 = lean_ctor_get(x_252, 2);
lean_inc(x_255);
x_256 = lean_ctor_get(x_252, 3);
lean_inc(x_256);
x_257 = lean_ctor_get(x_252, 4);
lean_inc(x_257);
if (lean_is_exclusive(x_252)) {
 lean_ctor_release(x_252, 0);
 lean_ctor_release(x_252, 1);
 lean_ctor_release(x_252, 2);
 lean_ctor_release(x_252, 3);
 lean_ctor_release(x_252, 4);
 x_258 = x_252;
} else {
 lean_dec_ref(x_252);
 x_258 = lean_box(0);
}
x_259 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_260 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_254);
x_261 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_261, 0, x_254);
x_262 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_262, 0, x_254);
x_263 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_263, 0, x_261);
lean_ctor_set(x_263, 1, x_262);
x_264 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_264, 0, x_257);
x_265 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_265, 0, x_256);
x_266 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_266, 0, x_255);
if (lean_is_scalar(x_258)) {
 x_267 = lean_alloc_ctor(0, 5, 0);
} else {
 x_267 = x_258;
}
lean_ctor_set(x_267, 0, x_263);
lean_ctor_set(x_267, 1, x_259);
lean_ctor_set(x_267, 2, x_266);
lean_ctor_set(x_267, 3, x_265);
lean_ctor_set(x_267, 4, x_264);
if (lean_is_scalar(x_253)) {
 x_268 = lean_alloc_ctor(0, 2, 0);
} else {
 x_268 = x_253;
}
lean_ctor_set(x_268, 0, x_267);
lean_ctor_set(x_268, 1, x_260);
x_269 = lean_st_ref_get(x_6);
lean_dec(x_269);
lean_inc(x_3);
x_270 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___boxed), 6, 1);
lean_closure_set(x_270, 0, x_3);
x_271 = l_Lean_MVarId_withContext___redArg(x_31, x_268, x_3, x_270);
x_272 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_4, x_271, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_272) == 0)
{
lean_object* x_273; lean_object* x_274; lean_object* x_275; lean_object* x_276; lean_object* x_277; lean_object* x_278; lean_object* x_279; lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; lean_object* x_291; lean_object* x_292; 
x_273 = lean_ctor_get(x_272, 0);
lean_inc(x_273);
if (lean_is_exclusive(x_272)) {
 lean_ctor_release(x_272, 0);
 x_274 = x_272;
} else {
 lean_dec_ref(x_272);
 x_274 = lean_box(0);
}
x_275 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0;
x_276 = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(x_275, x_5);
x_277 = l_Lean_stringToMessageData(x_276);
x_278 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2;
x_279 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_279, 0, x_277);
lean_ctor_set(x_279, 1, x_278);
x_280 = l_Nat_reprFast(x_1);
x_281 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_281, 0, x_280);
x_282 = l_Lean_MessageData_ofFormat(x_281);
x_283 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_283, 0, x_279);
lean_ctor_set(x_283, 1, x_282);
x_284 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4;
x_285 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_285, 0, x_283);
lean_ctor_set(x_285, 1, x_284);
x_286 = lp_aesop_Aesop_Percent_toHumanString(x_2);
x_287 = l_Lean_stringToMessageData(x_286);
x_288 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_288, 0, x_285);
lean_ctor_set(x_288, 1, x_287);
x_289 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6;
x_290 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_290, 0, x_288);
lean_ctor_set(x_290, 1, x_289);
x_291 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_291, 0, x_290);
lean_ctor_set(x_291, 1, x_273);
if (lean_is_scalar(x_274)) {
 x_292 = lean_alloc_ctor(0, 1, 0);
} else {
 x_292 = x_274;
}
lean_ctor_set(x_292, 0, x_291);
return x_292;
}
else
{
lean_dec_ref(x_5);
lean_dec(x_1);
return x_272;
}
}
}
else
{
lean_object* x_293; lean_object* x_294; lean_object* x_295; lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; lean_object* x_301; lean_object* x_302; lean_object* x_303; lean_object* x_304; lean_object* x_305; lean_object* x_306; lean_object* x_307; lean_object* x_308; lean_object* x_309; lean_object* x_310; lean_object* x_311; lean_object* x_312; lean_object* x_313; lean_object* x_314; lean_object* x_315; lean_object* x_316; lean_object* x_317; lean_object* x_318; lean_object* x_319; lean_object* x_320; lean_object* x_321; lean_object* x_322; lean_object* x_323; lean_object* x_324; lean_object* x_325; lean_object* x_326; lean_object* x_327; lean_object* x_328; lean_object* x_329; lean_object* x_330; lean_object* x_331; lean_object* x_332; lean_object* x_333; lean_object* x_334; lean_object* x_335; lean_object* x_336; lean_object* x_337; lean_object* x_338; lean_object* x_339; lean_object* x_340; lean_object* x_341; lean_object* x_342; lean_object* x_343; lean_object* x_344; lean_object* x_345; lean_object* x_346; 
x_293 = lean_ctor_get(x_13, 0);
x_294 = lean_ctor_get(x_13, 2);
x_295 = lean_ctor_get(x_13, 3);
x_296 = lean_ctor_get(x_13, 4);
lean_inc(x_296);
lean_inc(x_295);
lean_inc(x_294);
lean_inc(x_293);
lean_dec(x_13);
x_297 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_298 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_293);
x_299 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_299, 0, x_293);
x_300 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_300, 0, x_293);
x_301 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_301, 0, x_299);
lean_ctor_set(x_301, 1, x_300);
x_302 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_302, 0, x_296);
x_303 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_303, 0, x_295);
x_304 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_304, 0, x_294);
x_305 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_305, 0, x_301);
lean_ctor_set(x_305, 1, x_297);
lean_ctor_set(x_305, 2, x_304);
lean_ctor_set(x_305, 3, x_303);
lean_ctor_set(x_305, 4, x_302);
x_306 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_306, 0, x_305);
lean_ctor_set(x_306, 1, x_298);
x_307 = l_ReaderT_instMonad___redArg(x_306);
x_308 = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(x_308, 0, lean_box(0));
lean_closure_set(x_308, 1, lean_box(0));
lean_closure_set(x_308, 2, x_307);
x_309 = l_instMonadControlTOfPure___redArg(x_308);
x_310 = lean_ctor_get(x_12, 0);
lean_inc_ref(x_310);
if (lean_is_exclusive(x_12)) {
 lean_ctor_release(x_12, 0);
 lean_ctor_release(x_12, 1);
 x_311 = x_12;
} else {
 lean_dec_ref(x_12);
 x_311 = lean_box(0);
}
x_312 = lean_ctor_get(x_310, 0);
lean_inc_ref(x_312);
x_313 = lean_ctor_get(x_310, 2);
lean_inc(x_313);
x_314 = lean_ctor_get(x_310, 3);
lean_inc(x_314);
x_315 = lean_ctor_get(x_310, 4);
lean_inc(x_315);
if (lean_is_exclusive(x_310)) {
 lean_ctor_release(x_310, 0);
 lean_ctor_release(x_310, 1);
 lean_ctor_release(x_310, 2);
 lean_ctor_release(x_310, 3);
 lean_ctor_release(x_310, 4);
 x_316 = x_310;
} else {
 lean_dec_ref(x_310);
 x_316 = lean_box(0);
}
lean_inc_ref(x_312);
x_317 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_317, 0, x_312);
x_318 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_318, 0, x_312);
x_319 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_319, 0, x_317);
lean_ctor_set(x_319, 1, x_318);
x_320 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_320, 0, x_315);
x_321 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_321, 0, x_314);
x_322 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_322, 0, x_313);
if (lean_is_scalar(x_316)) {
 x_323 = lean_alloc_ctor(0, 5, 0);
} else {
 x_323 = x_316;
}
lean_ctor_set(x_323, 0, x_319);
lean_ctor_set(x_323, 1, x_297);
lean_ctor_set(x_323, 2, x_322);
lean_ctor_set(x_323, 3, x_321);
lean_ctor_set(x_323, 4, x_320);
if (lean_is_scalar(x_311)) {
 x_324 = lean_alloc_ctor(0, 2, 0);
} else {
 x_324 = x_311;
}
lean_ctor_set(x_324, 0, x_323);
lean_ctor_set(x_324, 1, x_298);
x_325 = l_ReaderT_instMonad___redArg(x_324);
x_326 = lean_ctor_get(x_325, 0);
lean_inc_ref(x_326);
if (lean_is_exclusive(x_325)) {
 lean_ctor_release(x_325, 0);
 lean_ctor_release(x_325, 1);
 x_327 = x_325;
} else {
 lean_dec_ref(x_325);
 x_327 = lean_box(0);
}
x_328 = lean_ctor_get(x_326, 0);
lean_inc_ref(x_328);
x_329 = lean_ctor_get(x_326, 2);
lean_inc(x_329);
x_330 = lean_ctor_get(x_326, 3);
lean_inc(x_330);
x_331 = lean_ctor_get(x_326, 4);
lean_inc(x_331);
if (lean_is_exclusive(x_326)) {
 lean_ctor_release(x_326, 0);
 lean_ctor_release(x_326, 1);
 lean_ctor_release(x_326, 2);
 lean_ctor_release(x_326, 3);
 lean_ctor_release(x_326, 4);
 x_332 = x_326;
} else {
 lean_dec_ref(x_326);
 x_332 = lean_box(0);
}
x_333 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_334 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_328);
x_335 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_335, 0, x_328);
x_336 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_336, 0, x_328);
x_337 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_337, 0, x_335);
lean_ctor_set(x_337, 1, x_336);
x_338 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_338, 0, x_331);
x_339 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_339, 0, x_330);
x_340 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_340, 0, x_329);
if (lean_is_scalar(x_332)) {
 x_341 = lean_alloc_ctor(0, 5, 0);
} else {
 x_341 = x_332;
}
lean_ctor_set(x_341, 0, x_337);
lean_ctor_set(x_341, 1, x_333);
lean_ctor_set(x_341, 2, x_340);
lean_ctor_set(x_341, 3, x_339);
lean_ctor_set(x_341, 4, x_338);
if (lean_is_scalar(x_327)) {
 x_342 = lean_alloc_ctor(0, 2, 0);
} else {
 x_342 = x_327;
}
lean_ctor_set(x_342, 0, x_341);
lean_ctor_set(x_342, 1, x_334);
x_343 = lean_st_ref_get(x_6);
lean_dec(x_343);
lean_inc(x_3);
x_344 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___boxed), 6, 1);
lean_closure_set(x_344, 0, x_3);
x_345 = l_Lean_MVarId_withContext___redArg(x_309, x_342, x_3, x_344);
x_346 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_4, x_345, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_346) == 0)
{
lean_object* x_347; lean_object* x_348; lean_object* x_349; lean_object* x_350; lean_object* x_351; lean_object* x_352; lean_object* x_353; lean_object* x_354; lean_object* x_355; lean_object* x_356; lean_object* x_357; lean_object* x_358; lean_object* x_359; lean_object* x_360; lean_object* x_361; lean_object* x_362; lean_object* x_363; lean_object* x_364; lean_object* x_365; lean_object* x_366; 
x_347 = lean_ctor_get(x_346, 0);
lean_inc(x_347);
if (lean_is_exclusive(x_346)) {
 lean_ctor_release(x_346, 0);
 x_348 = x_346;
} else {
 lean_dec_ref(x_346);
 x_348 = lean_box(0);
}
x_349 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0;
x_350 = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(x_349, x_5);
x_351 = l_Lean_stringToMessageData(x_350);
x_352 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2;
x_353 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_353, 0, x_351);
lean_ctor_set(x_353, 1, x_352);
x_354 = l_Nat_reprFast(x_1);
x_355 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_355, 0, x_354);
x_356 = l_Lean_MessageData_ofFormat(x_355);
x_357 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_357, 0, x_353);
lean_ctor_set(x_357, 1, x_356);
x_358 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4;
x_359 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_359, 0, x_357);
lean_ctor_set(x_359, 1, x_358);
x_360 = lp_aesop_Aesop_Percent_toHumanString(x_2);
x_361 = l_Lean_stringToMessageData(x_360);
x_362 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_362, 0, x_359);
lean_ctor_set(x_362, 1, x_361);
x_363 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6;
x_364 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_364, 0, x_362);
lean_ctor_set(x_364, 1, x_363);
x_365 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_365, 0, x_364);
lean_ctor_set(x_365, 1, x_347);
if (lean_is_scalar(x_348)) {
 x_366 = lean_alloc_ctor(0, 1, 0);
} else {
 x_366 = x_348;
}
lean_ctor_set(x_366, 0, x_365);
return x_366;
}
else
{
lean_dec_ref(x_5);
lean_dec(x_1);
return x_346;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt(lean_object* x_1, lean_object* x_2, lean_object* x_3, double x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_17; 
x_17 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg(x_3, x_4, x_5, x_6, x_7, x_9, x_12, x_13, x_14, x_15);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
double x_17; lean_object* x_18; 
x_17 = lean_unbox_float(x_4);
lean_dec_ref(x_4);
x_18 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt(x_1, x_2, x_3, x_17, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_2);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
double x_12; lean_object* x_13; 
x_12 = lean_unbox_float(x_2);
lean_dec_ref(x_2);
x_13 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg(x_1, x_12, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_6);
return x_13;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_BaseM_instMonadStats;
x_2 = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0;
x_2 = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1;
x_2 = lp_aesop_Aesop_instMonadStatsReaderT___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_TraceOption_steps;
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" ", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_newNodeEmoji;
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__3;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__1;
x_3 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__4;
x_8 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_1);
x_9 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Metadata", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__1;
x_2 = l_Lean_MessageData_ofFormat(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2;
x_8 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_7;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = l_Lean_Core_instMonadTraceCoreM;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_3 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__0;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_3 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_ReaderT_instMonadFunctor___lam__0), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = l_Lean_Core_instMonadQuotationCoreM;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_3 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__3;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_3 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = l_instMonadExceptOfEST(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__5;
x_2 = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__6;
x_2 = l_Lean_instMonadAlwaysExceptReaderT___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__7;
x_2 = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__8;
x_2 = l_Lean_instMonadAlwaysExceptReaderT___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Core_liftIOCore___boxed), 5, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instMonadLiftBaseIOEIO___lam__0___boxed), 3, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instMonadLiftT___lam__0___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__11;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__12;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__10;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__13;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__14;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__15;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("", 0, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_treeImpl;
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__17() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_IO_instMonadLiftSTRealWorldBaseIO___lam__0___boxed), 3, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__17;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__12;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__11;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__18;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__10;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__19;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__21() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__20;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__22() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__21;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, uint8_t x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_19; lean_object* x_20; 
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_19 = l_Lean_addTrace___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_inc(x_17);
lean_inc_ref(x_16);
lean_inc(x_15);
lean_inc_ref(x_14);
x_20 = lean_apply_5(x_19, x_14, x_15, x_16, x_17, lean_box(0));
if (lean_obj_tag(x_20) == 0)
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
lean_dec_ref(x_20);
x_21 = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_traceMetadata___boxed), 7, 2);
lean_closure_set(x_21, 0, x_7);
lean_closure_set(x_21, 1, x_8);
x_22 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_23 = l_Lean_withTraceNode___redArg(x_1, x_2, x_3, x_4, x_9, x_10, x_11, x_5, x_12, x_21, x_13, x_22);
x_24 = lean_apply_5(x_23, x_14, x_15, x_16, x_17, lean_box(0));
return x_24;
}
else
{
lean_dec(x_17);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_20;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
_start:
{
uint8_t x_19; lean_object* x_20; 
x_19 = lean_unbox(x_13);
x_20 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_19, x_14, x_15, x_16, x_17);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, uint8_t x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_18 = lean_st_ref_get(x_12);
x_19 = lean_ctor_get(x_1, 0);
x_20 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_21 = lean_ctor_get(x_20, 1);
lean_inc_ref(x_21);
lean_inc(x_18);
x_22 = lean_apply_1(x_21, x_18);
x_23 = lean_ctor_get(x_22, 5);
lean_inc(x_23);
lean_dec_ref(x_22);
x_24 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_24, 0, x_23);
x_25 = lean_box(x_10);
lean_inc_ref(x_1);
lean_inc(x_18);
lean_inc(x_19);
x_26 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___boxed), 18, 13);
lean_closure_set(x_26, 0, x_2);
lean_closure_set(x_26, 1, x_3);
lean_closure_set(x_26, 2, x_4);
lean_closure_set(x_26, 3, x_5);
lean_closure_set(x_26, 4, x_19);
lean_closure_set(x_26, 5, x_24);
lean_closure_set(x_26, 6, x_18);
lean_closure_set(x_26, 7, x_1);
lean_closure_set(x_26, 8, x_6);
lean_closure_set(x_26, 9, x_7);
lean_closure_set(x_26, 10, x_8);
lean_closure_set(x_26, 11, x_9);
lean_closure_set(x_26, 12, x_25);
x_27 = lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg(x_18, x_1, x_26, x_10, x_11, x_13, x_14, x_15, x_16);
return x_27;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
_start:
{
uint8_t x_18; lean_object* x_19; 
x_18 = lean_unbox(x_10);
x_19 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_18, x_11, x_12, x_13, x_14, x_15, x_16);
lean_dec(x_12);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_18 = lean_st_ref_get(x_10);
lean_dec(x_18);
x_19 = lean_st_ref_get(x_6);
x_20 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1;
x_21 = !lean_is_exclusive(x_20);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; uint8_t x_24; 
x_22 = lean_ctor_get(x_20, 0);
x_23 = lean_ctor_get(x_20, 1);
lean_dec(x_23);
x_24 = !lean_is_exclusive(x_22);
if (x_24 == 0)
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; uint8_t x_39; 
x_25 = lean_ctor_get(x_22, 0);
x_26 = lean_ctor_get(x_22, 2);
x_27 = lean_ctor_get(x_22, 3);
x_28 = lean_ctor_get(x_22, 4);
x_29 = lean_ctor_get(x_22, 1);
lean_dec(x_29);
x_30 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_31 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_25);
x_32 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_32, 0, x_25);
x_33 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_33, 0, x_25);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_32);
lean_ctor_set(x_34, 1, x_33);
x_35 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_35, 0, x_28);
x_36 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_36, 0, x_27);
x_37 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_37, 0, x_26);
lean_ctor_set(x_22, 4, x_35);
lean_ctor_set(x_22, 3, x_36);
lean_ctor_set(x_22, 2, x_37);
lean_ctor_set(x_22, 1, x_30);
lean_ctor_set(x_22, 0, x_34);
lean_ctor_set(x_20, 1, x_31);
x_38 = l_ReaderT_instMonad___redArg(x_20);
x_39 = !lean_is_exclusive(x_38);
if (x_39 == 0)
{
lean_object* x_40; lean_object* x_41; uint8_t x_42; 
x_40 = lean_ctor_get(x_38, 0);
x_41 = lean_ctor_get(x_38, 1);
lean_dec(x_41);
x_42 = !lean_is_exclusive(x_40);
if (x_42 == 0)
{
lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_43 = lean_ctor_get(x_40, 0);
x_44 = lean_ctor_get(x_40, 2);
x_45 = lean_ctor_get(x_40, 3);
x_46 = lean_ctor_get(x_40, 4);
x_47 = lean_ctor_get(x_40, 1);
lean_dec(x_47);
x_48 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_49 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_43);
x_50 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_50, 0, x_43);
x_51 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_51, 0, x_43);
x_52 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_52, 0, x_50);
lean_ctor_set(x_52, 1, x_51);
x_53 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_53, 0, x_46);
x_54 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_54, 0, x_45);
x_55 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_55, 0, x_44);
lean_ctor_set(x_40, 4, x_53);
lean_ctor_set(x_40, 3, x_54);
lean_ctor_set(x_40, 2, x_55);
lean_ctor_set(x_40, 1, x_48);
lean_ctor_set(x_40, 0, x_52);
lean_ctor_set(x_38, 1, x_49);
x_56 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1;
x_57 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__4;
x_58 = lean_ctor_get(x_57, 0);
lean_inc_ref(x_58);
x_59 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__9;
x_60 = lean_st_ref_get(x_10);
lean_dec(x_60);
x_61 = lean_ctor_get(x_1, 0);
x_62 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__16;
x_63 = l_Lean_Meta_instAddMessageContextMetaM;
x_64 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
lean_inc_ref(x_1);
lean_inc(x_19);
x_65 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_traceMetadata___boxed), 7, 2);
lean_closure_set(x_65, 0, x_19);
lean_closure_set(x_65, 1, x_1);
x_66 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
lean_inc_ref(x_2);
lean_inc(x_61);
lean_inc_ref(x_58);
lean_inc_ref(x_38);
x_67 = l_Lean_withTraceNode___redArg(x_38, x_56, x_58, x_63, x_64, x_59, x_62, x_61, x_2, x_65, x_3, x_66);
lean_inc(x_16);
lean_inc_ref(x_15);
lean_inc(x_14);
lean_inc_ref(x_13);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
lean_inc(x_19);
x_68 = lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(x_19, x_1, x_67, x_3, x_4, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_68) == 0)
{
lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; 
lean_dec_ref(x_68);
x_69 = lean_st_ref_get(x_10);
lean_dec(x_69);
x_70 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_71 = lean_ctor_get(x_70, 3);
lean_inc_ref(x_71);
lean_inc(x_19);
x_72 = lean_apply_1(x_71, x_19);
x_73 = lean_ctor_get(x_72, 6);
lean_inc_ref(x_73);
lean_dec_ref(x_72);
x_74 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__22;
x_75 = lean_box(x_3);
lean_inc_ref(x_38);
x_76 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___boxed), 17, 11);
lean_closure_set(x_76, 0, x_1);
lean_closure_set(x_76, 1, x_38);
lean_closure_set(x_76, 2, x_56);
lean_closure_set(x_76, 3, x_58);
lean_closure_set(x_76, 4, x_63);
lean_closure_set(x_76, 5, x_64);
lean_closure_set(x_76, 6, x_59);
lean_closure_set(x_76, 7, x_62);
lean_closure_set(x_76, 8, x_2);
lean_closure_set(x_76, 9, x_75);
lean_closure_set(x_76, 10, x_4);
x_77 = lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(x_38, x_74, x_76, x_19);
x_78 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_73, x_77, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_78) == 0)
{
uint8_t x_79; 
x_79 = !lean_is_exclusive(x_78);
if (x_79 == 0)
{
lean_object* x_80; lean_object* x_81; 
x_80 = lean_ctor_get(x_78, 0);
lean_dec(x_80);
x_81 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_81, 0, x_5);
lean_ctor_set(x_78, 0, x_81);
return x_78;
}
else
{
lean_object* x_82; lean_object* x_83; 
lean_dec(x_78);
x_82 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_82, 0, x_5);
x_83 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_83, 0, x_82);
return x_83;
}
}
else
{
uint8_t x_84; 
x_84 = !lean_is_exclusive(x_78);
if (x_84 == 0)
{
return x_78;
}
else
{
lean_object* x_85; lean_object* x_86; 
x_85 = lean_ctor_get(x_78, 0);
lean_inc(x_85);
lean_dec(x_78);
x_86 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_86, 0, x_85);
return x_86;
}
}
}
else
{
uint8_t x_87; 
lean_dec_ref(x_58);
lean_dec_ref(x_38);
lean_dec(x_19);
lean_dec(x_16);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_87 = !lean_is_exclusive(x_68);
if (x_87 == 0)
{
return x_68;
}
else
{
lean_object* x_88; lean_object* x_89; 
x_88 = lean_ctor_get(x_68, 0);
lean_inc(x_88);
lean_dec(x_68);
x_89 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_89, 0, x_88);
return x_89;
}
}
}
else
{
lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; 
x_90 = lean_ctor_get(x_40, 0);
x_91 = lean_ctor_get(x_40, 2);
x_92 = lean_ctor_get(x_40, 3);
x_93 = lean_ctor_get(x_40, 4);
lean_inc(x_93);
lean_inc(x_92);
lean_inc(x_91);
lean_inc(x_90);
lean_dec(x_40);
x_94 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_95 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_90);
x_96 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_96, 0, x_90);
x_97 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_97, 0, x_90);
x_98 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_98, 0, x_96);
lean_ctor_set(x_98, 1, x_97);
x_99 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_99, 0, x_93);
x_100 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_100, 0, x_92);
x_101 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_101, 0, x_91);
x_102 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_102, 0, x_98);
lean_ctor_set(x_102, 1, x_94);
lean_ctor_set(x_102, 2, x_101);
lean_ctor_set(x_102, 3, x_100);
lean_ctor_set(x_102, 4, x_99);
lean_ctor_set(x_38, 1, x_95);
lean_ctor_set(x_38, 0, x_102);
x_103 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1;
x_104 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__4;
x_105 = lean_ctor_get(x_104, 0);
lean_inc_ref(x_105);
x_106 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__9;
x_107 = lean_st_ref_get(x_10);
lean_dec(x_107);
x_108 = lean_ctor_get(x_1, 0);
x_109 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__16;
x_110 = l_Lean_Meta_instAddMessageContextMetaM;
x_111 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
lean_inc_ref(x_1);
lean_inc(x_19);
x_112 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_traceMetadata___boxed), 7, 2);
lean_closure_set(x_112, 0, x_19);
lean_closure_set(x_112, 1, x_1);
x_113 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
lean_inc_ref(x_2);
lean_inc(x_108);
lean_inc_ref(x_105);
lean_inc_ref(x_38);
x_114 = l_Lean_withTraceNode___redArg(x_38, x_103, x_105, x_110, x_111, x_106, x_109, x_108, x_2, x_112, x_3, x_113);
lean_inc(x_16);
lean_inc_ref(x_15);
lean_inc(x_14);
lean_inc_ref(x_13);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
lean_inc(x_19);
x_115 = lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(x_19, x_1, x_114, x_3, x_4, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_115) == 0)
{
lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; 
lean_dec_ref(x_115);
x_116 = lean_st_ref_get(x_10);
lean_dec(x_116);
x_117 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_118 = lean_ctor_get(x_117, 3);
lean_inc_ref(x_118);
lean_inc(x_19);
x_119 = lean_apply_1(x_118, x_19);
x_120 = lean_ctor_get(x_119, 6);
lean_inc_ref(x_120);
lean_dec_ref(x_119);
x_121 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__22;
x_122 = lean_box(x_3);
lean_inc_ref(x_38);
x_123 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___boxed), 17, 11);
lean_closure_set(x_123, 0, x_1);
lean_closure_set(x_123, 1, x_38);
lean_closure_set(x_123, 2, x_103);
lean_closure_set(x_123, 3, x_105);
lean_closure_set(x_123, 4, x_110);
lean_closure_set(x_123, 5, x_111);
lean_closure_set(x_123, 6, x_106);
lean_closure_set(x_123, 7, x_109);
lean_closure_set(x_123, 8, x_2);
lean_closure_set(x_123, 9, x_122);
lean_closure_set(x_123, 10, x_4);
x_124 = lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(x_38, x_121, x_123, x_19);
x_125 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_120, x_124, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_125) == 0)
{
lean_object* x_126; lean_object* x_127; lean_object* x_128; 
if (lean_is_exclusive(x_125)) {
 lean_ctor_release(x_125, 0);
 x_126 = x_125;
} else {
 lean_dec_ref(x_125);
 x_126 = lean_box(0);
}
x_127 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_127, 0, x_5);
if (lean_is_scalar(x_126)) {
 x_128 = lean_alloc_ctor(0, 1, 0);
} else {
 x_128 = x_126;
}
lean_ctor_set(x_128, 0, x_127);
return x_128;
}
else
{
lean_object* x_129; lean_object* x_130; lean_object* x_131; 
x_129 = lean_ctor_get(x_125, 0);
lean_inc(x_129);
if (lean_is_exclusive(x_125)) {
 lean_ctor_release(x_125, 0);
 x_130 = x_125;
} else {
 lean_dec_ref(x_125);
 x_130 = lean_box(0);
}
if (lean_is_scalar(x_130)) {
 x_131 = lean_alloc_ctor(1, 1, 0);
} else {
 x_131 = x_130;
}
lean_ctor_set(x_131, 0, x_129);
return x_131;
}
}
else
{
lean_object* x_132; lean_object* x_133; lean_object* x_134; 
lean_dec_ref(x_105);
lean_dec_ref(x_38);
lean_dec(x_19);
lean_dec(x_16);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_132 = lean_ctor_get(x_115, 0);
lean_inc(x_132);
if (lean_is_exclusive(x_115)) {
 lean_ctor_release(x_115, 0);
 x_133 = x_115;
} else {
 lean_dec_ref(x_115);
 x_133 = lean_box(0);
}
if (lean_is_scalar(x_133)) {
 x_134 = lean_alloc_ctor(1, 1, 0);
} else {
 x_134 = x_133;
}
lean_ctor_set(x_134, 0, x_132);
return x_134;
}
}
}
else
{
lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; 
x_135 = lean_ctor_get(x_38, 0);
lean_inc(x_135);
lean_dec(x_38);
x_136 = lean_ctor_get(x_135, 0);
lean_inc_ref(x_136);
x_137 = lean_ctor_get(x_135, 2);
lean_inc(x_137);
x_138 = lean_ctor_get(x_135, 3);
lean_inc(x_138);
x_139 = lean_ctor_get(x_135, 4);
lean_inc(x_139);
if (lean_is_exclusive(x_135)) {
 lean_ctor_release(x_135, 0);
 lean_ctor_release(x_135, 1);
 lean_ctor_release(x_135, 2);
 lean_ctor_release(x_135, 3);
 lean_ctor_release(x_135, 4);
 x_140 = x_135;
} else {
 lean_dec_ref(x_135);
 x_140 = lean_box(0);
}
x_141 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_142 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_136);
x_143 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_143, 0, x_136);
x_144 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_144, 0, x_136);
x_145 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_145, 0, x_143);
lean_ctor_set(x_145, 1, x_144);
x_146 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_146, 0, x_139);
x_147 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_147, 0, x_138);
x_148 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_148, 0, x_137);
if (lean_is_scalar(x_140)) {
 x_149 = lean_alloc_ctor(0, 5, 0);
} else {
 x_149 = x_140;
}
lean_ctor_set(x_149, 0, x_145);
lean_ctor_set(x_149, 1, x_141);
lean_ctor_set(x_149, 2, x_148);
lean_ctor_set(x_149, 3, x_147);
lean_ctor_set(x_149, 4, x_146);
x_150 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_150, 0, x_149);
lean_ctor_set(x_150, 1, x_142);
x_151 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1;
x_152 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__4;
x_153 = lean_ctor_get(x_152, 0);
lean_inc_ref(x_153);
x_154 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__9;
x_155 = lean_st_ref_get(x_10);
lean_dec(x_155);
x_156 = lean_ctor_get(x_1, 0);
x_157 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__16;
x_158 = l_Lean_Meta_instAddMessageContextMetaM;
x_159 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
lean_inc_ref(x_1);
lean_inc(x_19);
x_160 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_traceMetadata___boxed), 7, 2);
lean_closure_set(x_160, 0, x_19);
lean_closure_set(x_160, 1, x_1);
x_161 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
lean_inc_ref(x_2);
lean_inc(x_156);
lean_inc_ref(x_153);
lean_inc_ref(x_150);
x_162 = l_Lean_withTraceNode___redArg(x_150, x_151, x_153, x_158, x_159, x_154, x_157, x_156, x_2, x_160, x_3, x_161);
lean_inc(x_16);
lean_inc_ref(x_15);
lean_inc(x_14);
lean_inc_ref(x_13);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
lean_inc(x_19);
x_163 = lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(x_19, x_1, x_162, x_3, x_4, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_163) == 0)
{
lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; 
lean_dec_ref(x_163);
x_164 = lean_st_ref_get(x_10);
lean_dec(x_164);
x_165 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_166 = lean_ctor_get(x_165, 3);
lean_inc_ref(x_166);
lean_inc(x_19);
x_167 = lean_apply_1(x_166, x_19);
x_168 = lean_ctor_get(x_167, 6);
lean_inc_ref(x_168);
lean_dec_ref(x_167);
x_169 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__22;
x_170 = lean_box(x_3);
lean_inc_ref(x_150);
x_171 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___boxed), 17, 11);
lean_closure_set(x_171, 0, x_1);
lean_closure_set(x_171, 1, x_150);
lean_closure_set(x_171, 2, x_151);
lean_closure_set(x_171, 3, x_153);
lean_closure_set(x_171, 4, x_158);
lean_closure_set(x_171, 5, x_159);
lean_closure_set(x_171, 6, x_154);
lean_closure_set(x_171, 7, x_157);
lean_closure_set(x_171, 8, x_2);
lean_closure_set(x_171, 9, x_170);
lean_closure_set(x_171, 10, x_4);
x_172 = lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(x_150, x_169, x_171, x_19);
x_173 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_168, x_172, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_173) == 0)
{
lean_object* x_174; lean_object* x_175; lean_object* x_176; 
if (lean_is_exclusive(x_173)) {
 lean_ctor_release(x_173, 0);
 x_174 = x_173;
} else {
 lean_dec_ref(x_173);
 x_174 = lean_box(0);
}
x_175 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_175, 0, x_5);
if (lean_is_scalar(x_174)) {
 x_176 = lean_alloc_ctor(0, 1, 0);
} else {
 x_176 = x_174;
}
lean_ctor_set(x_176, 0, x_175);
return x_176;
}
else
{
lean_object* x_177; lean_object* x_178; lean_object* x_179; 
x_177 = lean_ctor_get(x_173, 0);
lean_inc(x_177);
if (lean_is_exclusive(x_173)) {
 lean_ctor_release(x_173, 0);
 x_178 = x_173;
} else {
 lean_dec_ref(x_173);
 x_178 = lean_box(0);
}
if (lean_is_scalar(x_178)) {
 x_179 = lean_alloc_ctor(1, 1, 0);
} else {
 x_179 = x_178;
}
lean_ctor_set(x_179, 0, x_177);
return x_179;
}
}
else
{
lean_object* x_180; lean_object* x_181; lean_object* x_182; 
lean_dec_ref(x_153);
lean_dec_ref(x_150);
lean_dec(x_19);
lean_dec(x_16);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_180 = lean_ctor_get(x_163, 0);
lean_inc(x_180);
if (lean_is_exclusive(x_163)) {
 lean_ctor_release(x_163, 0);
 x_181 = x_163;
} else {
 lean_dec_ref(x_163);
 x_181 = lean_box(0);
}
if (lean_is_scalar(x_181)) {
 x_182 = lean_alloc_ctor(1, 1, 0);
} else {
 x_182 = x_181;
}
lean_ctor_set(x_182, 0, x_180);
return x_182;
}
}
}
else
{
lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; 
x_183 = lean_ctor_get(x_22, 0);
x_184 = lean_ctor_get(x_22, 2);
x_185 = lean_ctor_get(x_22, 3);
x_186 = lean_ctor_get(x_22, 4);
lean_inc(x_186);
lean_inc(x_185);
lean_inc(x_184);
lean_inc(x_183);
lean_dec(x_22);
x_187 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_188 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_183);
x_189 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_189, 0, x_183);
x_190 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_190, 0, x_183);
x_191 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_191, 0, x_189);
lean_ctor_set(x_191, 1, x_190);
x_192 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_192, 0, x_186);
x_193 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_193, 0, x_185);
x_194 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_194, 0, x_184);
x_195 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_195, 0, x_191);
lean_ctor_set(x_195, 1, x_187);
lean_ctor_set(x_195, 2, x_194);
lean_ctor_set(x_195, 3, x_193);
lean_ctor_set(x_195, 4, x_192);
lean_ctor_set(x_20, 1, x_188);
lean_ctor_set(x_20, 0, x_195);
x_196 = l_ReaderT_instMonad___redArg(x_20);
x_197 = lean_ctor_get(x_196, 0);
lean_inc_ref(x_197);
if (lean_is_exclusive(x_196)) {
 lean_ctor_release(x_196, 0);
 lean_ctor_release(x_196, 1);
 x_198 = x_196;
} else {
 lean_dec_ref(x_196);
 x_198 = lean_box(0);
}
x_199 = lean_ctor_get(x_197, 0);
lean_inc_ref(x_199);
x_200 = lean_ctor_get(x_197, 2);
lean_inc(x_200);
x_201 = lean_ctor_get(x_197, 3);
lean_inc(x_201);
x_202 = lean_ctor_get(x_197, 4);
lean_inc(x_202);
if (lean_is_exclusive(x_197)) {
 lean_ctor_release(x_197, 0);
 lean_ctor_release(x_197, 1);
 lean_ctor_release(x_197, 2);
 lean_ctor_release(x_197, 3);
 lean_ctor_release(x_197, 4);
 x_203 = x_197;
} else {
 lean_dec_ref(x_197);
 x_203 = lean_box(0);
}
x_204 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_205 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_199);
x_206 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_206, 0, x_199);
x_207 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_207, 0, x_199);
x_208 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_208, 0, x_206);
lean_ctor_set(x_208, 1, x_207);
x_209 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_209, 0, x_202);
x_210 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_210, 0, x_201);
x_211 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_211, 0, x_200);
if (lean_is_scalar(x_203)) {
 x_212 = lean_alloc_ctor(0, 5, 0);
} else {
 x_212 = x_203;
}
lean_ctor_set(x_212, 0, x_208);
lean_ctor_set(x_212, 1, x_204);
lean_ctor_set(x_212, 2, x_211);
lean_ctor_set(x_212, 3, x_210);
lean_ctor_set(x_212, 4, x_209);
if (lean_is_scalar(x_198)) {
 x_213 = lean_alloc_ctor(0, 2, 0);
} else {
 x_213 = x_198;
}
lean_ctor_set(x_213, 0, x_212);
lean_ctor_set(x_213, 1, x_205);
x_214 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1;
x_215 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__4;
x_216 = lean_ctor_get(x_215, 0);
lean_inc_ref(x_216);
x_217 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__9;
x_218 = lean_st_ref_get(x_10);
lean_dec(x_218);
x_219 = lean_ctor_get(x_1, 0);
x_220 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__16;
x_221 = l_Lean_Meta_instAddMessageContextMetaM;
x_222 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
lean_inc_ref(x_1);
lean_inc(x_19);
x_223 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_traceMetadata___boxed), 7, 2);
lean_closure_set(x_223, 0, x_19);
lean_closure_set(x_223, 1, x_1);
x_224 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
lean_inc_ref(x_2);
lean_inc(x_219);
lean_inc_ref(x_216);
lean_inc_ref(x_213);
x_225 = l_Lean_withTraceNode___redArg(x_213, x_214, x_216, x_221, x_222, x_217, x_220, x_219, x_2, x_223, x_3, x_224);
lean_inc(x_16);
lean_inc_ref(x_15);
lean_inc(x_14);
lean_inc_ref(x_13);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
lean_inc(x_19);
x_226 = lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(x_19, x_1, x_225, x_3, x_4, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_226) == 0)
{
lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; lean_object* x_236; 
lean_dec_ref(x_226);
x_227 = lean_st_ref_get(x_10);
lean_dec(x_227);
x_228 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_229 = lean_ctor_get(x_228, 3);
lean_inc_ref(x_229);
lean_inc(x_19);
x_230 = lean_apply_1(x_229, x_19);
x_231 = lean_ctor_get(x_230, 6);
lean_inc_ref(x_231);
lean_dec_ref(x_230);
x_232 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__22;
x_233 = lean_box(x_3);
lean_inc_ref(x_213);
x_234 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___boxed), 17, 11);
lean_closure_set(x_234, 0, x_1);
lean_closure_set(x_234, 1, x_213);
lean_closure_set(x_234, 2, x_214);
lean_closure_set(x_234, 3, x_216);
lean_closure_set(x_234, 4, x_221);
lean_closure_set(x_234, 5, x_222);
lean_closure_set(x_234, 6, x_217);
lean_closure_set(x_234, 7, x_220);
lean_closure_set(x_234, 8, x_2);
lean_closure_set(x_234, 9, x_233);
lean_closure_set(x_234, 10, x_4);
x_235 = lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(x_213, x_232, x_234, x_19);
x_236 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_231, x_235, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_236) == 0)
{
lean_object* x_237; lean_object* x_238; lean_object* x_239; 
if (lean_is_exclusive(x_236)) {
 lean_ctor_release(x_236, 0);
 x_237 = x_236;
} else {
 lean_dec_ref(x_236);
 x_237 = lean_box(0);
}
x_238 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_238, 0, x_5);
if (lean_is_scalar(x_237)) {
 x_239 = lean_alloc_ctor(0, 1, 0);
} else {
 x_239 = x_237;
}
lean_ctor_set(x_239, 0, x_238);
return x_239;
}
else
{
lean_object* x_240; lean_object* x_241; lean_object* x_242; 
x_240 = lean_ctor_get(x_236, 0);
lean_inc(x_240);
if (lean_is_exclusive(x_236)) {
 lean_ctor_release(x_236, 0);
 x_241 = x_236;
} else {
 lean_dec_ref(x_236);
 x_241 = lean_box(0);
}
if (lean_is_scalar(x_241)) {
 x_242 = lean_alloc_ctor(1, 1, 0);
} else {
 x_242 = x_241;
}
lean_ctor_set(x_242, 0, x_240);
return x_242;
}
}
else
{
lean_object* x_243; lean_object* x_244; lean_object* x_245; 
lean_dec_ref(x_216);
lean_dec_ref(x_213);
lean_dec(x_19);
lean_dec(x_16);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_243 = lean_ctor_get(x_226, 0);
lean_inc(x_243);
if (lean_is_exclusive(x_226)) {
 lean_ctor_release(x_226, 0);
 x_244 = x_226;
} else {
 lean_dec_ref(x_226);
 x_244 = lean_box(0);
}
if (lean_is_scalar(x_244)) {
 x_245 = lean_alloc_ctor(1, 1, 0);
} else {
 x_245 = x_244;
}
lean_ctor_set(x_245, 0, x_243);
return x_245;
}
}
}
else
{
lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; lean_object* x_259; lean_object* x_260; lean_object* x_261; lean_object* x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; lean_object* x_269; lean_object* x_270; lean_object* x_271; lean_object* x_272; lean_object* x_273; lean_object* x_274; lean_object* x_275; lean_object* x_276; lean_object* x_277; lean_object* x_278; lean_object* x_279; lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; lean_object* x_291; lean_object* x_292; 
x_246 = lean_ctor_get(x_20, 0);
lean_inc(x_246);
lean_dec(x_20);
x_247 = lean_ctor_get(x_246, 0);
lean_inc_ref(x_247);
x_248 = lean_ctor_get(x_246, 2);
lean_inc(x_248);
x_249 = lean_ctor_get(x_246, 3);
lean_inc(x_249);
x_250 = lean_ctor_get(x_246, 4);
lean_inc(x_250);
if (lean_is_exclusive(x_246)) {
 lean_ctor_release(x_246, 0);
 lean_ctor_release(x_246, 1);
 lean_ctor_release(x_246, 2);
 lean_ctor_release(x_246, 3);
 lean_ctor_release(x_246, 4);
 x_251 = x_246;
} else {
 lean_dec_ref(x_246);
 x_251 = lean_box(0);
}
x_252 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_253 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_247);
x_254 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_254, 0, x_247);
x_255 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_255, 0, x_247);
x_256 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_256, 0, x_254);
lean_ctor_set(x_256, 1, x_255);
x_257 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_257, 0, x_250);
x_258 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_258, 0, x_249);
x_259 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_259, 0, x_248);
if (lean_is_scalar(x_251)) {
 x_260 = lean_alloc_ctor(0, 5, 0);
} else {
 x_260 = x_251;
}
lean_ctor_set(x_260, 0, x_256);
lean_ctor_set(x_260, 1, x_252);
lean_ctor_set(x_260, 2, x_259);
lean_ctor_set(x_260, 3, x_258);
lean_ctor_set(x_260, 4, x_257);
x_261 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_261, 0, x_260);
lean_ctor_set(x_261, 1, x_253);
x_262 = l_ReaderT_instMonad___redArg(x_261);
x_263 = lean_ctor_get(x_262, 0);
lean_inc_ref(x_263);
if (lean_is_exclusive(x_262)) {
 lean_ctor_release(x_262, 0);
 lean_ctor_release(x_262, 1);
 x_264 = x_262;
} else {
 lean_dec_ref(x_262);
 x_264 = lean_box(0);
}
x_265 = lean_ctor_get(x_263, 0);
lean_inc_ref(x_265);
x_266 = lean_ctor_get(x_263, 2);
lean_inc(x_266);
x_267 = lean_ctor_get(x_263, 3);
lean_inc(x_267);
x_268 = lean_ctor_get(x_263, 4);
lean_inc(x_268);
if (lean_is_exclusive(x_263)) {
 lean_ctor_release(x_263, 0);
 lean_ctor_release(x_263, 1);
 lean_ctor_release(x_263, 2);
 lean_ctor_release(x_263, 3);
 lean_ctor_release(x_263, 4);
 x_269 = x_263;
} else {
 lean_dec_ref(x_263);
 x_269 = lean_box(0);
}
x_270 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_271 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_265);
x_272 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_272, 0, x_265);
x_273 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_273, 0, x_265);
x_274 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_274, 0, x_272);
lean_ctor_set(x_274, 1, x_273);
x_275 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_275, 0, x_268);
x_276 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_276, 0, x_267);
x_277 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_277, 0, x_266);
if (lean_is_scalar(x_269)) {
 x_278 = lean_alloc_ctor(0, 5, 0);
} else {
 x_278 = x_269;
}
lean_ctor_set(x_278, 0, x_274);
lean_ctor_set(x_278, 1, x_270);
lean_ctor_set(x_278, 2, x_277);
lean_ctor_set(x_278, 3, x_276);
lean_ctor_set(x_278, 4, x_275);
if (lean_is_scalar(x_264)) {
 x_279 = lean_alloc_ctor(0, 2, 0);
} else {
 x_279 = x_264;
}
lean_ctor_set(x_279, 0, x_278);
lean_ctor_set(x_279, 1, x_271);
x_280 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1;
x_281 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__4;
x_282 = lean_ctor_get(x_281, 0);
lean_inc_ref(x_282);
x_283 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__9;
x_284 = lean_st_ref_get(x_10);
lean_dec(x_284);
x_285 = lean_ctor_get(x_1, 0);
x_286 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__16;
x_287 = l_Lean_Meta_instAddMessageContextMetaM;
x_288 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
lean_inc_ref(x_1);
lean_inc(x_19);
x_289 = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_traceMetadata___boxed), 7, 2);
lean_closure_set(x_289, 0, x_19);
lean_closure_set(x_289, 1, x_1);
x_290 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
lean_inc_ref(x_2);
lean_inc(x_285);
lean_inc_ref(x_282);
lean_inc_ref(x_279);
x_291 = l_Lean_withTraceNode___redArg(x_279, x_280, x_282, x_287, x_288, x_283, x_286, x_285, x_2, x_289, x_3, x_290);
lean_inc(x_16);
lean_inc_ref(x_15);
lean_inc(x_14);
lean_inc_ref(x_13);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
lean_inc(x_19);
x_292 = lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(x_19, x_1, x_291, x_3, x_4, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_292) == 0)
{
lean_object* x_293; lean_object* x_294; lean_object* x_295; lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; lean_object* x_301; lean_object* x_302; 
lean_dec_ref(x_292);
x_293 = lean_st_ref_get(x_10);
lean_dec(x_293);
x_294 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_295 = lean_ctor_get(x_294, 3);
lean_inc_ref(x_295);
lean_inc(x_19);
x_296 = lean_apply_1(x_295, x_19);
x_297 = lean_ctor_get(x_296, 6);
lean_inc_ref(x_297);
lean_dec_ref(x_296);
x_298 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__22;
x_299 = lean_box(x_3);
lean_inc_ref(x_279);
x_300 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___boxed), 17, 11);
lean_closure_set(x_300, 0, x_1);
lean_closure_set(x_300, 1, x_279);
lean_closure_set(x_300, 2, x_280);
lean_closure_set(x_300, 3, x_282);
lean_closure_set(x_300, 4, x_287);
lean_closure_set(x_300, 5, x_288);
lean_closure_set(x_300, 6, x_283);
lean_closure_set(x_300, 7, x_286);
lean_closure_set(x_300, 8, x_2);
lean_closure_set(x_300, 9, x_299);
lean_closure_set(x_300, 10, x_4);
x_301 = lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(x_279, x_298, x_300, x_19);
x_302 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_297, x_301, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_302) == 0)
{
lean_object* x_303; lean_object* x_304; lean_object* x_305; 
if (lean_is_exclusive(x_302)) {
 lean_ctor_release(x_302, 0);
 x_303 = x_302;
} else {
 lean_dec_ref(x_302);
 x_303 = lean_box(0);
}
x_304 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_304, 0, x_5);
if (lean_is_scalar(x_303)) {
 x_305 = lean_alloc_ctor(0, 1, 0);
} else {
 x_305 = x_303;
}
lean_ctor_set(x_305, 0, x_304);
return x_305;
}
else
{
lean_object* x_306; lean_object* x_307; lean_object* x_308; 
x_306 = lean_ctor_get(x_302, 0);
lean_inc(x_306);
if (lean_is_exclusive(x_302)) {
 lean_ctor_release(x_302, 0);
 x_307 = x_302;
} else {
 lean_dec_ref(x_302);
 x_307 = lean_box(0);
}
if (lean_is_scalar(x_307)) {
 x_308 = lean_alloc_ctor(1, 1, 0);
} else {
 x_308 = x_307;
}
lean_ctor_set(x_308, 0, x_306);
return x_308;
}
}
else
{
lean_object* x_309; lean_object* x_310; lean_object* x_311; 
lean_dec_ref(x_282);
lean_dec_ref(x_279);
lean_dec(x_19);
lean_dec(x_16);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_309 = lean_ctor_get(x_292, 0);
lean_inc(x_309);
if (lean_is_exclusive(x_292)) {
 lean_ctor_release(x_292, 0);
 x_310 = x_292;
} else {
 lean_dec_ref(x_292);
 x_310 = lean_box(0);
}
if (lean_is_scalar(x_310)) {
 x_311 = lean_alloc_ctor(1, 1, 0);
} else {
 x_311 = x_310;
}
lean_ctor_set(x_311, 0, x_309);
return x_311;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
_start:
{
uint8_t x_18; lean_object* x_19; 
x_18 = lean_unbox(x_3);
x_19 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4(x_1, x_2, x_18, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
lean_dec(x_12);
lean_dec(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_6);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
x_13 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3;
lean_inc_ref(x_12);
x_16 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_12, x_14, x_15);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_17 = lean_apply_9(x_16, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_17) == 0)
{
uint8_t x_18; 
x_18 = !lean_is_exclusive(x_17);
if (x_18 == 0)
{
lean_object* x_19; uint8_t x_20; 
x_19 = lean_ctor_get(x_17, 0);
x_20 = lean_unbox(x_19);
if (x_20 == 0)
{
lean_object* x_21; 
lean_dec(x_19);
lean_dec_ref(x_12);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_21 = lean_box(0);
lean_ctor_set(x_17, 0, x_21);
return x_17;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; size_t x_26; size_t x_27; lean_object* x_28; lean_object* x_29; 
lean_free_object(x_17);
x_22 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___boxed), 6, 0);
x_23 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___boxed), 6, 0);
x_24 = lean_box(0);
x_25 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___boxed), 17, 5);
lean_closure_set(x_25, 0, x_15);
lean_closure_set(x_25, 1, x_23);
lean_closure_set(x_25, 2, x_19);
lean_closure_set(x_25, 3, x_22);
lean_closure_set(x_25, 4, x_24);
x_26 = lean_array_size(x_2);
x_27 = 0;
x_28 = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), x_12, x_2, x_25, x_26, x_27, x_24);
x_29 = lean_apply_9(x_28, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_29) == 0)
{
uint8_t x_30; 
x_30 = !lean_is_exclusive(x_29);
if (x_30 == 0)
{
lean_object* x_31; 
x_31 = lean_ctor_get(x_29, 0);
lean_dec(x_31);
lean_ctor_set(x_29, 0, x_24);
return x_29;
}
else
{
lean_object* x_32; 
lean_dec(x_29);
x_32 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_32, 0, x_24);
return x_32;
}
}
else
{
return x_29;
}
}
}
else
{
lean_object* x_33; uint8_t x_34; 
x_33 = lean_ctor_get(x_17, 0);
lean_inc(x_33);
lean_dec(x_17);
x_34 = lean_unbox(x_33);
if (x_34 == 0)
{
lean_object* x_35; lean_object* x_36; 
lean_dec(x_33);
lean_dec_ref(x_12);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_35 = lean_box(0);
x_36 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_36, 0, x_35);
return x_36;
}
else
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; size_t x_41; size_t x_42; lean_object* x_43; lean_object* x_44; 
x_37 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___boxed), 6, 0);
x_38 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___boxed), 6, 0);
x_39 = lean_box(0);
x_40 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___boxed), 17, 5);
lean_closure_set(x_40, 0, x_15);
lean_closure_set(x_40, 1, x_38);
lean_closure_set(x_40, 2, x_33);
lean_closure_set(x_40, 3, x_37);
lean_closure_set(x_40, 4, x_39);
x_41 = lean_array_size(x_2);
x_42 = 0;
x_43 = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), x_12, x_2, x_40, x_41, x_42, x_39);
x_44 = lean_apply_9(x_43, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_44) == 0)
{
lean_object* x_45; lean_object* x_46; 
if (lean_is_exclusive(x_44)) {
 lean_ctor_release(x_44, 0);
 x_45 = x_44;
} else {
 lean_dec_ref(x_44);
 x_45 = lean_box(0);
}
if (lean_is_scalar(x_45)) {
 x_46 = lean_alloc_ctor(0, 1, 0);
} else {
 x_46 = x_45;
}
lean_ctor_set(x_46, 0, x_39);
return x_46;
}
else
{
return x_44;
}
}
}
}
else
{
uint8_t x_47; 
lean_dec_ref(x_12);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_47 = !lean_is_exclusive(x_17);
if (x_47 == 0)
{
return x_17;
}
else
{
lean_object* x_48; lean_object* x_49; 
x_48 = lean_ctor_get(x_17, 0);
lean_inc(x_48);
lean_dec(x_17);
x_49 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_49, 0, x_48);
return x_49;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_2);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_1);
return x_12;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_3 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_expandNextGoal___redArg___closed__0;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_3 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_expandNextGoal___redArg___closed__1;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_3 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_expandNextGoal___redArg___closed__2;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
x_3 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__9;
x_2 = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_expandNextGoal___redArg___closed__3;
x_2 = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_expandNextGoal___redArg___closed__4;
x_2 = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_expandNextGoal___redArg___closed__6;
x_2 = l_Lean_instMonadAlwaysExceptReaderT___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_11 = lean_st_ref_get(x_3);
lean_dec(x_11);
x_12 = lean_st_ref_get(x_1);
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_expandNextGoal___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_11;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__16;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_2 = lp_aesop_Aesop_expandNextGoal___redArg___closed__8;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_2 = lp_aesop_Aesop_expandNextGoal___redArg___closed__9;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
x_2 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Initial goal:", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; lean_object* x_15; 
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_14 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_1, x_2, x_3);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_inc_ref(x_9);
x_15 = lean_apply_5(x_14, x_9, x_10, x_11, x_12, lean_box(0));
if (lean_obj_tag(x_15) == 0)
{
uint8_t x_16; 
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; uint8_t x_18; 
x_17 = lean_ctor_get(x_15, 0);
x_18 = lean_unbox(x_17);
lean_dec(x_17);
if (x_18 == 0)
{
lean_object* x_19; 
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_19 = lean_box(0);
lean_ctor_set(x_15, 0, x_19);
return x_15;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; 
lean_free_object(x_15);
x_20 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_21 = l_Lean_Core_instMonadQuotationCoreM;
x_22 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_20, x_4, x_21);
x_23 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_20, x_5, x_22);
x_24 = lean_ctor_get(x_23, 0);
lean_inc_ref(x_24);
lean_dec_ref(x_23);
x_25 = !lean_is_exclusive(x_3);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_26 = lean_ctor_get(x_3, 0);
x_27 = lean_ctor_get(x_3, 1);
lean_dec(x_27);
x_28 = lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1;
x_29 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_29, 0, x_6);
x_30 = l_Lean_indentD(x_29);
lean_ctor_set_tag(x_3, 7);
lean_ctor_set(x_3, 1, x_30);
lean_ctor_set(x_3, 0, x_28);
x_31 = l_Lean_addTrace___redArg(x_1, x_7, x_24, x_8, x_26, x_3);
x_32 = lean_apply_5(x_31, x_9, x_10, x_11, x_12, lean_box(0));
return x_32;
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_33 = lean_ctor_get(x_3, 0);
lean_inc(x_33);
lean_dec(x_3);
x_34 = lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1;
x_35 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_35, 0, x_6);
x_36 = l_Lean_indentD(x_35);
x_37 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_37, 0, x_34);
lean_ctor_set(x_37, 1, x_36);
x_38 = l_Lean_addTrace___redArg(x_1, x_7, x_24, x_8, x_33, x_37);
x_39 = lean_apply_5(x_38, x_9, x_10, x_11, x_12, lean_box(0));
return x_39;
}
}
}
else
{
lean_object* x_40; uint8_t x_41; 
x_40 = lean_ctor_get(x_15, 0);
lean_inc(x_40);
lean_dec(x_15);
x_41 = lean_unbox(x_40);
lean_dec(x_40);
if (x_41 == 0)
{
lean_object* x_42; lean_object* x_43; 
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_42 = lean_box(0);
x_43 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_43, 0, x_42);
return x_43;
}
else
{
lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; 
x_44 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_45 = l_Lean_Core_instMonadQuotationCoreM;
x_46 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_44, x_4, x_45);
x_47 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_44, x_5, x_46);
x_48 = lean_ctor_get(x_47, 0);
lean_inc_ref(x_48);
lean_dec_ref(x_47);
x_49 = lean_ctor_get(x_3, 0);
lean_inc(x_49);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 x_50 = x_3;
} else {
 lean_dec_ref(x_3);
 x_50 = lean_box(0);
}
x_51 = lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1;
x_52 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_52, 0, x_6);
x_53 = l_Lean_indentD(x_52);
if (lean_is_scalar(x_50)) {
 x_54 = lean_alloc_ctor(7, 2, 0);
} else {
 x_54 = x_50;
 lean_ctor_set_tag(x_54, 7);
}
lean_ctor_set(x_54, 0, x_51);
lean_ctor_set(x_54, 1, x_53);
x_55 = l_Lean_addTrace___redArg(x_1, x_7, x_48, x_8, x_49, x_54);
x_56 = lean_apply_5(x_55, x_9, x_10, x_11, x_12, lean_box(0));
return x_56;
}
}
}
else
{
uint8_t x_57; 
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_57 = !lean_is_exclusive(x_15);
if (x_57 == 0)
{
return x_15;
}
else
{
lean_object* x_58; lean_object* x_59; 
x_58 = lean_ctor_get(x_15, 0);
lean_inc(x_58);
lean_dec(x_15);
x_59 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_59, 0, x_58);
return x_59;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_aesop_Aesop_expandNextGoal___redArg___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_st_ref_get(x_4);
lean_dec(x_12);
x_13 = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(x_1, x_2, x_7, x_8, x_9, x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_expandNextGoal___redArg___lam__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_12;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Treating the goal as unprovable since it is beyond the maximum rule application depth (", 87, 87);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(").", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18) {
_start:
{
lean_object* x_20; lean_object* x_21; lean_object* x_34; 
lean_inc(x_18);
lean_inc_ref(x_17);
lean_inc(x_16);
lean_inc_ref(x_15);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_inc_ref(x_11);
x_34 = lean_apply_9(x_1, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, lean_box(0));
if (lean_obj_tag(x_34) == 0)
{
lean_object* x_35; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; uint8_t x_147; 
x_35 = lean_ctor_get(x_34, 0);
lean_inc(x_35);
lean_dec_ref(x_34);
x_143 = lean_ctor_get(x_11, 2);
x_144 = lean_ctor_get(x_143, 0);
x_145 = lean_ctor_get(x_144, 0);
x_146 = lean_unsigned_to_nat(0u);
x_147 = lean_nat_dec_eq(x_145, x_146);
if (x_147 == 0)
{
lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; uint8_t x_152; 
x_148 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_149 = lean_ctor_get(x_148, 1);
lean_inc_ref(x_149);
x_150 = lean_apply_1(x_149, x_35);
x_151 = lean_ctor_get(x_150, 4);
lean_inc(x_151);
lean_dec_ref(x_150);
x_152 = lean_nat_dec_le(x_145, x_151);
lean_dec(x_151);
if (x_152 == 0)
{
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
goto block_142;
}
else
{
lean_object* x_153; lean_object* x_154; 
lean_dec_ref(x_2);
lean_inc_ref(x_6);
lean_inc_ref(x_4);
x_153 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_4, x_5, x_6);
lean_inc(x_18);
lean_inc_ref(x_17);
lean_inc(x_16);
lean_inc_ref(x_15);
lean_inc(x_14);
lean_inc(x_13);
lean_inc(x_12);
lean_inc_ref(x_11);
x_154 = lean_apply_9(x_153, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, lean_box(0));
if (lean_obj_tag(x_154) == 0)
{
lean_object* x_155; uint8_t x_156; 
x_155 = lean_ctor_get(x_154, 0);
lean_inc(x_155);
lean_dec_ref(x_154);
x_156 = lean_unbox(x_155);
lean_dec(x_155);
if (x_156 == 0)
{
lean_dec(x_18);
lean_dec_ref(x_17);
lean_dec(x_16);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
x_20 = x_12;
x_21 = lean_box(0);
goto block_33;
}
else
{
uint8_t x_157; 
x_157 = !lean_is_exclusive(x_6);
if (x_157 == 0)
{
lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; 
x_158 = lean_ctor_get(x_6, 0);
x_159 = lean_ctor_get(x_6, 1);
lean_dec(x_159);
x_160 = lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__2;
lean_inc(x_145);
x_161 = l_Nat_reprFast(x_145);
x_162 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_162, 0, x_161);
x_163 = l_Lean_MessageData_ofFormat(x_162);
lean_ctor_set_tag(x_6, 7);
lean_ctor_set(x_6, 1, x_163);
lean_ctor_set(x_6, 0, x_160);
x_164 = lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__4;
x_165 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_165, 0, x_6);
lean_ctor_set(x_165, 1, x_164);
x_166 = l_Lean_addTrace___redArg(x_4, x_7, x_8, x_9, x_158, x_165);
lean_inc(x_12);
x_167 = lean_apply_9(x_166, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, lean_box(0));
if (lean_obj_tag(x_167) == 0)
{
lean_dec_ref(x_167);
x_20 = x_12;
x_21 = lean_box(0);
goto block_33;
}
else
{
uint8_t x_168; 
lean_dec(x_12);
lean_dec(x_3);
x_168 = !lean_is_exclusive(x_167);
if (x_168 == 0)
{
return x_167;
}
else
{
lean_object* x_169; lean_object* x_170; 
x_169 = lean_ctor_get(x_167, 0);
lean_inc(x_169);
lean_dec(x_167);
x_170 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_170, 0, x_169);
return x_170;
}
}
}
else
{
lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; 
x_171 = lean_ctor_get(x_6, 0);
lean_inc(x_171);
lean_dec(x_6);
x_172 = lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__2;
lean_inc(x_145);
x_173 = l_Nat_reprFast(x_145);
x_174 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_174, 0, x_173);
x_175 = l_Lean_MessageData_ofFormat(x_174);
x_176 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_176, 0, x_172);
lean_ctor_set(x_176, 1, x_175);
x_177 = lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__4;
x_178 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_178, 0, x_176);
lean_ctor_set(x_178, 1, x_177);
x_179 = l_Lean_addTrace___redArg(x_4, x_7, x_8, x_9, x_171, x_178);
lean_inc(x_12);
x_180 = lean_apply_9(x_179, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, lean_box(0));
if (lean_obj_tag(x_180) == 0)
{
lean_dec_ref(x_180);
x_20 = x_12;
x_21 = lean_box(0);
goto block_33;
}
else
{
lean_object* x_181; lean_object* x_182; lean_object* x_183; 
lean_dec(x_12);
lean_dec(x_3);
x_181 = lean_ctor_get(x_180, 0);
lean_inc(x_181);
if (lean_is_exclusive(x_180)) {
 lean_ctor_release(x_180, 0);
 x_182 = x_180;
} else {
 lean_dec_ref(x_180);
 x_182 = lean_box(0);
}
if (lean_is_scalar(x_182)) {
 x_183 = lean_alloc_ctor(1, 1, 0);
} else {
 x_183 = x_182;
}
lean_ctor_set(x_183, 0, x_181);
return x_183;
}
}
}
}
else
{
uint8_t x_184; 
lean_dec(x_18);
lean_dec_ref(x_17);
lean_dec(x_16);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec(x_3);
x_184 = !lean_is_exclusive(x_154);
if (x_184 == 0)
{
return x_154;
}
else
{
lean_object* x_185; lean_object* x_186; 
x_185 = lean_ctor_get(x_154, 0);
lean_inc(x_185);
lean_dec(x_154);
x_186 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_186, 0, x_185);
return x_186;
}
}
}
}
else
{
lean_dec(x_35);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
goto block_142;
}
block_142:
{
lean_object* x_36; 
lean_inc(x_12);
lean_inc(x_3);
lean_inc_ref(x_2);
x_36 = lp_aesop_Aesop_expandGoal___redArg(x_2, x_3, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18);
if (lean_obj_tag(x_36) == 0)
{
lean_object* x_37; lean_object* x_38; 
x_37 = lean_ctor_get(x_36, 0);
lean_inc(x_37);
lean_dec_ref(x_36);
x_38 = lp_aesop_Aesop_getIteration___redArg(x_12);
if (lean_obj_tag(x_38) == 0)
{
uint8_t x_39; 
x_39 = !lean_is_exclusive(x_38);
if (x_39 == 0)
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; uint8_t x_47; 
x_40 = lean_ctor_get(x_38, 0);
x_41 = lean_st_ref_get(x_12);
lean_dec(x_41);
x_42 = lean_st_ref_take(x_3);
x_43 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_44 = lean_ctor_get(x_43, 0);
lean_inc(x_44);
x_45 = lean_ctor_get(x_43, 1);
lean_inc_ref(x_45);
x_46 = lean_apply_1(x_45, x_42);
x_47 = !lean_is_exclusive(x_46);
if (x_47 == 0)
{
lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; uint8_t x_54; 
x_48 = lean_ctor_get(x_46, 11);
lean_dec(x_48);
lean_ctor_set(x_46, 11, x_40);
x_49 = lean_apply_1(x_44, x_46);
x_50 = lean_st_ref_set(x_3, x_49);
x_51 = lean_st_ref_get(x_12);
lean_dec(x_51);
x_52 = lean_st_ref_get(x_3);
x_53 = lean_st_ref_get(x_12);
lean_dec(x_53);
x_54 = lp_aesop_Aesop_Goal_isActive(x_52);
if (x_54 == 0)
{
lean_dec(x_12);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_ctor_set(x_38, 0, x_37);
return x_38;
}
else
{
lean_object* x_55; lean_object* x_56; lean_object* x_57; 
lean_free_object(x_38);
x_55 = lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0;
x_56 = lean_array_push(x_55, x_3);
x_57 = lp_aesop_Aesop_enqueueGoals___redArg(x_2, x_56, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_57) == 0)
{
uint8_t x_58; 
x_58 = !lean_is_exclusive(x_57);
if (x_58 == 0)
{
lean_object* x_59; 
x_59 = lean_ctor_get(x_57, 0);
lean_dec(x_59);
lean_ctor_set(x_57, 0, x_37);
return x_57;
}
else
{
lean_object* x_60; 
lean_dec(x_57);
x_60 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_60, 0, x_37);
return x_60;
}
}
else
{
uint8_t x_61; 
lean_dec(x_37);
x_61 = !lean_is_exclusive(x_57);
if (x_61 == 0)
{
return x_57;
}
else
{
lean_object* x_62; lean_object* x_63; 
x_62 = lean_ctor_get(x_57, 0);
lean_inc(x_62);
lean_dec(x_57);
x_63 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_63, 0, x_62);
return x_63;
}
}
}
}
else
{
lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; uint8_t x_69; uint8_t x_70; uint8_t x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; double x_77; lean_object* x_78; uint8_t x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; uint8_t x_88; 
x_64 = lean_ctor_get(x_46, 0);
x_65 = lean_ctor_get(x_46, 1);
x_66 = lean_ctor_get(x_46, 2);
x_67 = lean_ctor_get(x_46, 3);
x_68 = lean_ctor_get(x_46, 4);
x_69 = lean_ctor_get_uint8(x_46, sizeof(void*)*14 + 8);
x_70 = lean_ctor_get_uint8(x_46, sizeof(void*)*14 + 9);
x_71 = lean_ctor_get_uint8(x_46, sizeof(void*)*14 + 10);
x_72 = lean_ctor_get(x_46, 5);
x_73 = lean_ctor_get(x_46, 6);
x_74 = lean_ctor_get(x_46, 7);
x_75 = lean_ctor_get(x_46, 8);
x_76 = lean_ctor_get(x_46, 9);
x_77 = lean_ctor_get_float(x_46, sizeof(void*)*14);
x_78 = lean_ctor_get(x_46, 10);
x_79 = lean_ctor_get_uint8(x_46, sizeof(void*)*14 + 11);
x_80 = lean_ctor_get(x_46, 12);
x_81 = lean_ctor_get(x_46, 13);
lean_inc(x_81);
lean_inc(x_80);
lean_inc(x_78);
lean_inc(x_76);
lean_inc(x_75);
lean_inc(x_74);
lean_inc(x_73);
lean_inc(x_72);
lean_inc(x_68);
lean_inc(x_67);
lean_inc(x_66);
lean_inc(x_65);
lean_inc(x_64);
lean_dec(x_46);
x_82 = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(x_82, 0, x_64);
lean_ctor_set(x_82, 1, x_65);
lean_ctor_set(x_82, 2, x_66);
lean_ctor_set(x_82, 3, x_67);
lean_ctor_set(x_82, 4, x_68);
lean_ctor_set(x_82, 5, x_72);
lean_ctor_set(x_82, 6, x_73);
lean_ctor_set(x_82, 7, x_74);
lean_ctor_set(x_82, 8, x_75);
lean_ctor_set(x_82, 9, x_76);
lean_ctor_set(x_82, 10, x_78);
lean_ctor_set(x_82, 11, x_40);
lean_ctor_set(x_82, 12, x_80);
lean_ctor_set(x_82, 13, x_81);
lean_ctor_set_uint8(x_82, sizeof(void*)*14 + 8, x_69);
lean_ctor_set_uint8(x_82, sizeof(void*)*14 + 9, x_70);
lean_ctor_set_uint8(x_82, sizeof(void*)*14 + 10, x_71);
lean_ctor_set_float(x_82, sizeof(void*)*14, x_77);
lean_ctor_set_uint8(x_82, sizeof(void*)*14 + 11, x_79);
x_83 = lean_apply_1(x_44, x_82);
x_84 = lean_st_ref_set(x_3, x_83);
x_85 = lean_st_ref_get(x_12);
lean_dec(x_85);
x_86 = lean_st_ref_get(x_3);
x_87 = lean_st_ref_get(x_12);
lean_dec(x_87);
x_88 = lp_aesop_Aesop_Goal_isActive(x_86);
if (x_88 == 0)
{
lean_dec(x_12);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_ctor_set(x_38, 0, x_37);
return x_38;
}
else
{
lean_object* x_89; lean_object* x_90; lean_object* x_91; 
lean_free_object(x_38);
x_89 = lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0;
x_90 = lean_array_push(x_89, x_3);
x_91 = lp_aesop_Aesop_enqueueGoals___redArg(x_2, x_90, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_91) == 0)
{
lean_object* x_92; lean_object* x_93; 
if (lean_is_exclusive(x_91)) {
 lean_ctor_release(x_91, 0);
 x_92 = x_91;
} else {
 lean_dec_ref(x_91);
 x_92 = lean_box(0);
}
if (lean_is_scalar(x_92)) {
 x_93 = lean_alloc_ctor(0, 1, 0);
} else {
 x_93 = x_92;
}
lean_ctor_set(x_93, 0, x_37);
return x_93;
}
else
{
lean_object* x_94; lean_object* x_95; lean_object* x_96; 
lean_dec(x_37);
x_94 = lean_ctor_get(x_91, 0);
lean_inc(x_94);
if (lean_is_exclusive(x_91)) {
 lean_ctor_release(x_91, 0);
 x_95 = x_91;
} else {
 lean_dec_ref(x_91);
 x_95 = lean_box(0);
}
if (lean_is_scalar(x_95)) {
 x_96 = lean_alloc_ctor(1, 1, 0);
} else {
 x_96 = x_95;
}
lean_ctor_set(x_96, 0, x_94);
return x_96;
}
}
}
}
else
{
lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; uint8_t x_109; uint8_t x_110; uint8_t x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; double x_117; lean_object* x_118; uint8_t x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; uint8_t x_129; 
x_97 = lean_ctor_get(x_38, 0);
lean_inc(x_97);
lean_dec(x_38);
x_98 = lean_st_ref_get(x_12);
lean_dec(x_98);
x_99 = lean_st_ref_take(x_3);
x_100 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_101 = lean_ctor_get(x_100, 0);
lean_inc(x_101);
x_102 = lean_ctor_get(x_100, 1);
lean_inc_ref(x_102);
x_103 = lean_apply_1(x_102, x_99);
x_104 = lean_ctor_get(x_103, 0);
lean_inc(x_104);
x_105 = lean_ctor_get(x_103, 1);
lean_inc(x_105);
x_106 = lean_ctor_get(x_103, 2);
lean_inc_ref(x_106);
x_107 = lean_ctor_get(x_103, 3);
lean_inc(x_107);
x_108 = lean_ctor_get(x_103, 4);
lean_inc(x_108);
x_109 = lean_ctor_get_uint8(x_103, sizeof(void*)*14 + 8);
x_110 = lean_ctor_get_uint8(x_103, sizeof(void*)*14 + 9);
x_111 = lean_ctor_get_uint8(x_103, sizeof(void*)*14 + 10);
x_112 = lean_ctor_get(x_103, 5);
lean_inc(x_112);
x_113 = lean_ctor_get(x_103, 6);
lean_inc(x_113);
x_114 = lean_ctor_get(x_103, 7);
lean_inc_ref(x_114);
x_115 = lean_ctor_get(x_103, 8);
lean_inc_ref(x_115);
x_116 = lean_ctor_get(x_103, 9);
lean_inc_ref(x_116);
x_117 = lean_ctor_get_float(x_103, sizeof(void*)*14);
x_118 = lean_ctor_get(x_103, 10);
lean_inc(x_118);
x_119 = lean_ctor_get_uint8(x_103, sizeof(void*)*14 + 11);
x_120 = lean_ctor_get(x_103, 12);
lean_inc_ref(x_120);
x_121 = lean_ctor_get(x_103, 13);
lean_inc_ref(x_121);
if (lean_is_exclusive(x_103)) {
 lean_ctor_release(x_103, 0);
 lean_ctor_release(x_103, 1);
 lean_ctor_release(x_103, 2);
 lean_ctor_release(x_103, 3);
 lean_ctor_release(x_103, 4);
 lean_ctor_release(x_103, 5);
 lean_ctor_release(x_103, 6);
 lean_ctor_release(x_103, 7);
 lean_ctor_release(x_103, 8);
 lean_ctor_release(x_103, 9);
 lean_ctor_release(x_103, 10);
 lean_ctor_release(x_103, 11);
 lean_ctor_release(x_103, 12);
 lean_ctor_release(x_103, 13);
 x_122 = x_103;
} else {
 lean_dec_ref(x_103);
 x_122 = lean_box(0);
}
if (lean_is_scalar(x_122)) {
 x_123 = lean_alloc_ctor(0, 14, 12);
} else {
 x_123 = x_122;
}
lean_ctor_set(x_123, 0, x_104);
lean_ctor_set(x_123, 1, x_105);
lean_ctor_set(x_123, 2, x_106);
lean_ctor_set(x_123, 3, x_107);
lean_ctor_set(x_123, 4, x_108);
lean_ctor_set(x_123, 5, x_112);
lean_ctor_set(x_123, 6, x_113);
lean_ctor_set(x_123, 7, x_114);
lean_ctor_set(x_123, 8, x_115);
lean_ctor_set(x_123, 9, x_116);
lean_ctor_set(x_123, 10, x_118);
lean_ctor_set(x_123, 11, x_97);
lean_ctor_set(x_123, 12, x_120);
lean_ctor_set(x_123, 13, x_121);
lean_ctor_set_uint8(x_123, sizeof(void*)*14 + 8, x_109);
lean_ctor_set_uint8(x_123, sizeof(void*)*14 + 9, x_110);
lean_ctor_set_uint8(x_123, sizeof(void*)*14 + 10, x_111);
lean_ctor_set_float(x_123, sizeof(void*)*14, x_117);
lean_ctor_set_uint8(x_123, sizeof(void*)*14 + 11, x_119);
x_124 = lean_apply_1(x_101, x_123);
x_125 = lean_st_ref_set(x_3, x_124);
x_126 = lean_st_ref_get(x_12);
lean_dec(x_126);
x_127 = lean_st_ref_get(x_3);
x_128 = lean_st_ref_get(x_12);
lean_dec(x_128);
x_129 = lp_aesop_Aesop_Goal_isActive(x_127);
if (x_129 == 0)
{
lean_object* x_130; 
lean_dec(x_12);
lean_dec(x_3);
lean_dec_ref(x_2);
x_130 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_130, 0, x_37);
return x_130;
}
else
{
lean_object* x_131; lean_object* x_132; lean_object* x_133; 
x_131 = lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0;
x_132 = lean_array_push(x_131, x_3);
x_133 = lp_aesop_Aesop_enqueueGoals___redArg(x_2, x_132, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_133) == 0)
{
lean_object* x_134; lean_object* x_135; 
if (lean_is_exclusive(x_133)) {
 lean_ctor_release(x_133, 0);
 x_134 = x_133;
} else {
 lean_dec_ref(x_133);
 x_134 = lean_box(0);
}
if (lean_is_scalar(x_134)) {
 x_135 = lean_alloc_ctor(0, 1, 0);
} else {
 x_135 = x_134;
}
lean_ctor_set(x_135, 0, x_37);
return x_135;
}
else
{
lean_object* x_136; lean_object* x_137; lean_object* x_138; 
lean_dec(x_37);
x_136 = lean_ctor_get(x_133, 0);
lean_inc(x_136);
if (lean_is_exclusive(x_133)) {
 lean_ctor_release(x_133, 0);
 x_137 = x_133;
} else {
 lean_dec_ref(x_133);
 x_137 = lean_box(0);
}
if (lean_is_scalar(x_137)) {
 x_138 = lean_alloc_ctor(1, 1, 0);
} else {
 x_138 = x_137;
}
lean_ctor_set(x_138, 0, x_136);
return x_138;
}
}
}
}
else
{
uint8_t x_139; 
lean_dec(x_37);
lean_dec(x_12);
lean_dec(x_3);
lean_dec_ref(x_2);
x_139 = !lean_is_exclusive(x_38);
if (x_139 == 0)
{
return x_38;
}
else
{
lean_object* x_140; lean_object* x_141; 
x_140 = lean_ctor_get(x_38, 0);
lean_inc(x_140);
lean_dec(x_38);
x_141 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_141, 0, x_140);
return x_141;
}
}
}
else
{
lean_dec(x_12);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_36;
}
}
}
else
{
uint8_t x_187; 
lean_dec(x_18);
lean_dec_ref(x_17);
lean_dec(x_16);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_187 = !lean_is_exclusive(x_34);
if (x_187 == 0)
{
return x_34;
}
else
{
lean_object* x_188; lean_object* x_189; 
x_188 = lean_ctor_get(x_34, 0);
lean_inc(x_188);
lean_dec(x_34);
x_189 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_189, 0, x_188);
return x_189;
}
}
block_33:
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_22 = lean_st_ref_get(x_20);
lean_dec(x_22);
x_23 = lp_aesop_Aesop_GoalRef_markForcedUnprovable(x_3);
lean_dec(x_3);
x_24 = lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(x_20);
lean_dec(x_20);
if (lean_obj_tag(x_24) == 0)
{
uint8_t x_25; 
x_25 = !lean_is_exclusive(x_24);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; 
x_26 = lean_ctor_get(x_24, 0);
lean_dec(x_26);
x_27 = lean_box(2);
lean_ctor_set(x_24, 0, x_27);
return x_24;
}
else
{
lean_object* x_28; lean_object* x_29; 
lean_dec(x_24);
x_28 = lean_box(2);
x_29 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_29, 0, x_28);
return x_29;
}
}
else
{
uint8_t x_30; 
x_30 = !lean_is_exclusive(x_24);
if (x_30 == 0)
{
return x_24;
}
else
{
lean_object* x_31; lean_object* x_32; 
x_31 = lean_ctor_get(x_24, 0);
lean_inc(x_31);
lean_dec(x_24);
x_32 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_32, 0, x_31);
return x_32;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
lean_object* x_19 = _args[18];
_start:
{
lean_object* x_20; 
x_20 = lp_aesop_Aesop_expandNextGoal___redArg___lam__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18);
return x_20;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_11 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_12 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_13 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1;
x_14 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1;
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
x_24 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_25 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
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
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; 
x_37 = lean_ctor_get(x_34, 0);
x_38 = lean_ctor_get(x_34, 2);
x_39 = lean_ctor_get(x_34, 3);
x_40 = lean_ctor_get(x_34, 4);
x_41 = lean_ctor_get(x_34, 1);
lean_dec(x_41);
x_42 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_43 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
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
x_50 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
lean_inc_ref(x_32);
x_51 = l_ReaderT_instMonad___redArg(x_32);
x_52 = l_ReaderT_instMonad___redArg(x_51);
x_53 = l_ReaderT_instMonad___redArg(x_52);
x_54 = lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
x_55 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_56 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_57 = lean_ctor_get(x_56, 0);
lean_inc(x_57);
x_58 = lp_aesop_Aesop_expandNextGoal___redArg___closed__7;
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_59 = lp_aesop_Aesop_nextActiveGoal___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_59) == 0)
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; 
x_60 = lean_ctor_get(x_59, 0);
lean_inc(x_60);
lean_dec_ref(x_59);
lean_inc(x_60);
x_61 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__0___boxed), 10, 1);
lean_closure_set(x_61, 0, x_60);
x_62 = lp_aesop_Aesop_expandNextGoal___redArg___lam__0(x_60, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
x_63 = lean_ctor_get(x_62, 0);
lean_inc(x_63);
lean_dec_ref(x_62);
x_64 = lean_st_ref_get(x_3);
lean_dec(x_64);
x_65 = lp_aesop_Aesop_getRootMetaState___redArg(x_4);
if (lean_obj_tag(x_65) == 0)
{
lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_66 = lean_ctor_get(x_65, 0);
lean_inc(x_66);
lean_dec_ref(x_65);
x_67 = lean_st_ref_get(x_3);
lean_dec(x_67);
lean_inc(x_63);
x_68 = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(x_63, x_66);
lean_dec(x_66);
if (lean_obj_tag(x_68) == 0)
{
lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; double x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; uint8_t x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; 
x_69 = lean_ctor_get(x_68, 0);
lean_inc(x_69);
lean_dec_ref(x_68);
x_70 = lean_ctor_get(x_69, 0);
lean_inc(x_70);
x_71 = lean_ctor_get(x_69, 1);
lean_inc(x_71);
lean_dec(x_69);
x_72 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_73 = lean_ctor_get(x_72, 1);
lean_inc_ref(x_73);
lean_inc(x_63);
x_74 = lean_apply_1(x_73, x_63);
x_75 = lean_ctor_get(x_74, 0);
lean_inc(x_75);
lean_dec_ref(x_74);
x_76 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3;
x_77 = lean_ctor_get(x_76, 0);
lean_inc(x_77);
x_78 = l_Lean_Meta_instAddMessageContextMetaM;
x_79 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
x_80 = lp_aesop_Aesop_expandNextGoal___redArg___closed__11;
x_81 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc(x_70);
x_82 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__1___boxed), 13, 8);
lean_closure_set(x_82, 0, x_32);
lean_closure_set(x_82, 1, x_79);
lean_closure_set(x_82, 2, x_76);
lean_closure_set(x_82, 3, x_12);
lean_closure_set(x_82, 4, x_11);
lean_closure_set(x_82, 5, x_70);
lean_closure_set(x_82, 6, x_13);
lean_closure_set(x_82, 7, x_78);
lean_inc(x_71);
x_83 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__2___boxed), 11, 2);
lean_closure_set(x_83, 0, x_71);
lean_closure_set(x_83, 1, x_82);
lean_inc_ref(x_55);
lean_inc(x_57);
lean_inc_ref(x_50);
lean_inc_ref(x_1);
x_84 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___boxed), 19, 9);
lean_closure_set(x_84, 0, x_61);
lean_closure_set(x_84, 1, x_1);
lean_closure_set(x_84, 2, x_60);
lean_closure_set(x_84, 3, x_50);
lean_closure_set(x_84, 4, x_57);
lean_closure_set(x_84, 5, x_76);
lean_closure_set(x_84, 6, x_54);
lean_closure_set(x_84, 7, x_55);
lean_closure_set(x_84, 8, x_81);
x_85 = lp_aesop_Aesop_Goal_priority(x_63);
x_86 = lean_box_float(x_85);
lean_inc_ref(x_1);
x_87 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___boxed), 16, 6);
lean_closure_set(x_87, 0, lean_box(0));
lean_closure_set(x_87, 1, x_1);
lean_closure_set(x_87, 2, x_75);
lean_closure_set(x_87, 3, x_86);
lean_closure_set(x_87, 4, x_70);
lean_closure_set(x_87, 5, x_71);
x_88 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_88, 0, lean_box(0));
lean_closure_set(x_88, 1, lean_box(0));
lean_closure_set(x_88, 2, x_53);
lean_closure_set(x_88, 3, lean_box(0));
lean_closure_set(x_88, 4, lean_box(0));
lean_closure_set(x_88, 5, x_83);
lean_closure_set(x_88, 6, x_84);
x_89 = 1;
x_90 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_91 = l_Lean_withTraceNode___redArg(x_50, x_54, x_55, x_81, x_57, x_58, x_80, x_77, x_87, x_88, x_89, x_90);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_92 = lean_apply_9(x_91, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
if (lean_obj_tag(x_92) == 0)
{
uint8_t x_93; 
x_93 = !lean_is_exclusive(x_92);
if (x_93 == 0)
{
lean_object* x_94; 
x_94 = lean_ctor_get(x_92, 0);
if (lean_obj_tag(x_94) == 2)
{
lean_object* x_95; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_95 = lean_box(0);
lean_ctor_set(x_92, 0, x_95);
return x_92;
}
else
{
lean_object* x_96; lean_object* x_97; 
lean_free_object(x_92);
x_96 = lean_ctor_get(x_94, 0);
lean_inc_ref(x_96);
lean_dec(x_94);
x_97 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(x_1, x_96, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_1);
return x_97;
}
}
else
{
lean_object* x_98; 
x_98 = lean_ctor_get(x_92, 0);
lean_inc(x_98);
lean_dec(x_92);
if (lean_obj_tag(x_98) == 2)
{
lean_object* x_99; lean_object* x_100; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_99 = lean_box(0);
x_100 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_100, 0, x_99);
return x_100;
}
else
{
lean_object* x_101; lean_object* x_102; 
x_101 = lean_ctor_get(x_98, 0);
lean_inc_ref(x_101);
lean_dec(x_98);
x_102 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(x_1, x_101, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_1);
return x_102;
}
}
}
else
{
uint8_t x_103; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_103 = !lean_is_exclusive(x_92);
if (x_103 == 0)
{
return x_92;
}
else
{
lean_object* x_104; lean_object* x_105; 
x_104 = lean_ctor_get(x_92, 0);
lean_inc(x_104);
lean_dec(x_92);
x_105 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_105, 0, x_104);
return x_105;
}
}
}
else
{
uint8_t x_106; 
lean_dec(x_63);
lean_dec_ref(x_61);
lean_dec(x_60);
lean_dec(x_57);
lean_dec_ref(x_55);
lean_dec_ref(x_53);
lean_dec_ref(x_50);
lean_dec_ref(x_32);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_106 = !lean_is_exclusive(x_68);
if (x_106 == 0)
{
return x_68;
}
else
{
lean_object* x_107; lean_object* x_108; 
x_107 = lean_ctor_get(x_68, 0);
lean_inc(x_107);
lean_dec(x_68);
x_108 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_108, 0, x_107);
return x_108;
}
}
}
else
{
uint8_t x_109; 
lean_dec(x_63);
lean_dec_ref(x_61);
lean_dec(x_60);
lean_dec(x_57);
lean_dec_ref(x_55);
lean_dec_ref(x_53);
lean_dec_ref(x_50);
lean_dec_ref(x_32);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_109 = !lean_is_exclusive(x_65);
if (x_109 == 0)
{
return x_65;
}
else
{
lean_object* x_110; lean_object* x_111; 
x_110 = lean_ctor_get(x_65, 0);
lean_inc(x_110);
lean_dec(x_65);
x_111 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_111, 0, x_110);
return x_111;
}
}
}
else
{
uint8_t x_112; 
lean_dec(x_57);
lean_dec_ref(x_55);
lean_dec_ref(x_53);
lean_dec_ref(x_50);
lean_dec_ref(x_32);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_112 = !lean_is_exclusive(x_59);
if (x_112 == 0)
{
return x_59;
}
else
{
lean_object* x_113; lean_object* x_114; 
x_113 = lean_ctor_get(x_59, 0);
lean_inc(x_113);
lean_dec(x_59);
x_114 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_114, 0, x_113);
return x_114;
}
}
}
else
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; 
x_115 = lean_ctor_get(x_34, 0);
x_116 = lean_ctor_get(x_34, 2);
x_117 = lean_ctor_get(x_34, 3);
x_118 = lean_ctor_get(x_34, 4);
lean_inc(x_118);
lean_inc(x_117);
lean_inc(x_116);
lean_inc(x_115);
lean_dec(x_34);
x_119 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_120 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_115);
x_121 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_121, 0, x_115);
x_122 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_122, 0, x_115);
x_123 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_123, 0, x_121);
lean_ctor_set(x_123, 1, x_122);
x_124 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_124, 0, x_118);
x_125 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_125, 0, x_117);
x_126 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_126, 0, x_116);
x_127 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_127, 0, x_123);
lean_ctor_set(x_127, 1, x_119);
lean_ctor_set(x_127, 2, x_126);
lean_ctor_set(x_127, 3, x_125);
lean_ctor_set(x_127, 4, x_124);
lean_ctor_set(x_32, 1, x_120);
lean_ctor_set(x_32, 0, x_127);
x_128 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
lean_inc_ref(x_32);
x_129 = l_ReaderT_instMonad___redArg(x_32);
x_130 = l_ReaderT_instMonad___redArg(x_129);
x_131 = l_ReaderT_instMonad___redArg(x_130);
x_132 = lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
x_133 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_134 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_135 = lean_ctor_get(x_134, 0);
lean_inc(x_135);
x_136 = lp_aesop_Aesop_expandNextGoal___redArg___closed__7;
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_137 = lp_aesop_Aesop_nextActiveGoal___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_137) == 0)
{
lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; 
x_138 = lean_ctor_get(x_137, 0);
lean_inc(x_138);
lean_dec_ref(x_137);
lean_inc(x_138);
x_139 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__0___boxed), 10, 1);
lean_closure_set(x_139, 0, x_138);
x_140 = lp_aesop_Aesop_expandNextGoal___redArg___lam__0(x_138, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
x_141 = lean_ctor_get(x_140, 0);
lean_inc(x_141);
lean_dec_ref(x_140);
x_142 = lean_st_ref_get(x_3);
lean_dec(x_142);
x_143 = lp_aesop_Aesop_getRootMetaState___redArg(x_4);
if (lean_obj_tag(x_143) == 0)
{
lean_object* x_144; lean_object* x_145; lean_object* x_146; 
x_144 = lean_ctor_get(x_143, 0);
lean_inc(x_144);
lean_dec_ref(x_143);
x_145 = lean_st_ref_get(x_3);
lean_dec(x_145);
lean_inc(x_141);
x_146 = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(x_141, x_144);
lean_dec(x_144);
if (lean_obj_tag(x_146) == 0)
{
lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; double x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; uint8_t x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; 
x_147 = lean_ctor_get(x_146, 0);
lean_inc(x_147);
lean_dec_ref(x_146);
x_148 = lean_ctor_get(x_147, 0);
lean_inc(x_148);
x_149 = lean_ctor_get(x_147, 1);
lean_inc(x_149);
lean_dec(x_147);
x_150 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_151 = lean_ctor_get(x_150, 1);
lean_inc_ref(x_151);
lean_inc(x_141);
x_152 = lean_apply_1(x_151, x_141);
x_153 = lean_ctor_get(x_152, 0);
lean_inc(x_153);
lean_dec_ref(x_152);
x_154 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3;
x_155 = lean_ctor_get(x_154, 0);
lean_inc(x_155);
x_156 = l_Lean_Meta_instAddMessageContextMetaM;
x_157 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
x_158 = lp_aesop_Aesop_expandNextGoal___redArg___closed__11;
x_159 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc(x_148);
x_160 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__1___boxed), 13, 8);
lean_closure_set(x_160, 0, x_32);
lean_closure_set(x_160, 1, x_157);
lean_closure_set(x_160, 2, x_154);
lean_closure_set(x_160, 3, x_12);
lean_closure_set(x_160, 4, x_11);
lean_closure_set(x_160, 5, x_148);
lean_closure_set(x_160, 6, x_13);
lean_closure_set(x_160, 7, x_156);
lean_inc(x_149);
x_161 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__2___boxed), 11, 2);
lean_closure_set(x_161, 0, x_149);
lean_closure_set(x_161, 1, x_160);
lean_inc_ref(x_133);
lean_inc(x_135);
lean_inc_ref(x_128);
lean_inc_ref(x_1);
x_162 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___boxed), 19, 9);
lean_closure_set(x_162, 0, x_139);
lean_closure_set(x_162, 1, x_1);
lean_closure_set(x_162, 2, x_138);
lean_closure_set(x_162, 3, x_128);
lean_closure_set(x_162, 4, x_135);
lean_closure_set(x_162, 5, x_154);
lean_closure_set(x_162, 6, x_132);
lean_closure_set(x_162, 7, x_133);
lean_closure_set(x_162, 8, x_159);
x_163 = lp_aesop_Aesop_Goal_priority(x_141);
x_164 = lean_box_float(x_163);
lean_inc_ref(x_1);
x_165 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___boxed), 16, 6);
lean_closure_set(x_165, 0, lean_box(0));
lean_closure_set(x_165, 1, x_1);
lean_closure_set(x_165, 2, x_153);
lean_closure_set(x_165, 3, x_164);
lean_closure_set(x_165, 4, x_148);
lean_closure_set(x_165, 5, x_149);
x_166 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_166, 0, lean_box(0));
lean_closure_set(x_166, 1, lean_box(0));
lean_closure_set(x_166, 2, x_131);
lean_closure_set(x_166, 3, lean_box(0));
lean_closure_set(x_166, 4, lean_box(0));
lean_closure_set(x_166, 5, x_161);
lean_closure_set(x_166, 6, x_162);
x_167 = 1;
x_168 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_169 = l_Lean_withTraceNode___redArg(x_128, x_132, x_133, x_159, x_135, x_136, x_158, x_155, x_165, x_166, x_167, x_168);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_170 = lean_apply_9(x_169, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
if (lean_obj_tag(x_170) == 0)
{
lean_object* x_171; lean_object* x_172; 
x_171 = lean_ctor_get(x_170, 0);
lean_inc(x_171);
if (lean_is_exclusive(x_170)) {
 lean_ctor_release(x_170, 0);
 x_172 = x_170;
} else {
 lean_dec_ref(x_170);
 x_172 = lean_box(0);
}
if (lean_obj_tag(x_171) == 2)
{
lean_object* x_173; lean_object* x_174; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_173 = lean_box(0);
if (lean_is_scalar(x_172)) {
 x_174 = lean_alloc_ctor(0, 1, 0);
} else {
 x_174 = x_172;
}
lean_ctor_set(x_174, 0, x_173);
return x_174;
}
else
{
lean_object* x_175; lean_object* x_176; 
lean_dec(x_172);
x_175 = lean_ctor_get(x_171, 0);
lean_inc_ref(x_175);
lean_dec(x_171);
x_176 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(x_1, x_175, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_1);
return x_176;
}
}
else
{
lean_object* x_177; lean_object* x_178; lean_object* x_179; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_177 = lean_ctor_get(x_170, 0);
lean_inc(x_177);
if (lean_is_exclusive(x_170)) {
 lean_ctor_release(x_170, 0);
 x_178 = x_170;
} else {
 lean_dec_ref(x_170);
 x_178 = lean_box(0);
}
if (lean_is_scalar(x_178)) {
 x_179 = lean_alloc_ctor(1, 1, 0);
} else {
 x_179 = x_178;
}
lean_ctor_set(x_179, 0, x_177);
return x_179;
}
}
else
{
lean_object* x_180; lean_object* x_181; lean_object* x_182; 
lean_dec(x_141);
lean_dec_ref(x_139);
lean_dec(x_138);
lean_dec(x_135);
lean_dec_ref(x_133);
lean_dec_ref(x_131);
lean_dec_ref(x_128);
lean_dec_ref(x_32);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_180 = lean_ctor_get(x_146, 0);
lean_inc(x_180);
if (lean_is_exclusive(x_146)) {
 lean_ctor_release(x_146, 0);
 x_181 = x_146;
} else {
 lean_dec_ref(x_146);
 x_181 = lean_box(0);
}
if (lean_is_scalar(x_181)) {
 x_182 = lean_alloc_ctor(1, 1, 0);
} else {
 x_182 = x_181;
}
lean_ctor_set(x_182, 0, x_180);
return x_182;
}
}
else
{
lean_object* x_183; lean_object* x_184; lean_object* x_185; 
lean_dec(x_141);
lean_dec_ref(x_139);
lean_dec(x_138);
lean_dec(x_135);
lean_dec_ref(x_133);
lean_dec_ref(x_131);
lean_dec_ref(x_128);
lean_dec_ref(x_32);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_183 = lean_ctor_get(x_143, 0);
lean_inc(x_183);
if (lean_is_exclusive(x_143)) {
 lean_ctor_release(x_143, 0);
 x_184 = x_143;
} else {
 lean_dec_ref(x_143);
 x_184 = lean_box(0);
}
if (lean_is_scalar(x_184)) {
 x_185 = lean_alloc_ctor(1, 1, 0);
} else {
 x_185 = x_184;
}
lean_ctor_set(x_185, 0, x_183);
return x_185;
}
}
else
{
lean_object* x_186; lean_object* x_187; lean_object* x_188; 
lean_dec(x_135);
lean_dec_ref(x_133);
lean_dec_ref(x_131);
lean_dec_ref(x_128);
lean_dec_ref(x_32);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_186 = lean_ctor_get(x_137, 0);
lean_inc(x_186);
if (lean_is_exclusive(x_137)) {
 lean_ctor_release(x_137, 0);
 x_187 = x_137;
} else {
 lean_dec_ref(x_137);
 x_187 = lean_box(0);
}
if (lean_is_scalar(x_187)) {
 x_188 = lean_alloc_ctor(1, 1, 0);
} else {
 x_188 = x_187;
}
lean_ctor_set(x_188, 0, x_186);
return x_188;
}
}
}
else
{
lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; 
x_189 = lean_ctor_get(x_32, 0);
lean_inc(x_189);
lean_dec(x_32);
x_190 = lean_ctor_get(x_189, 0);
lean_inc_ref(x_190);
x_191 = lean_ctor_get(x_189, 2);
lean_inc(x_191);
x_192 = lean_ctor_get(x_189, 3);
lean_inc(x_192);
x_193 = lean_ctor_get(x_189, 4);
lean_inc(x_193);
if (lean_is_exclusive(x_189)) {
 lean_ctor_release(x_189, 0);
 lean_ctor_release(x_189, 1);
 lean_ctor_release(x_189, 2);
 lean_ctor_release(x_189, 3);
 lean_ctor_release(x_189, 4);
 x_194 = x_189;
} else {
 lean_dec_ref(x_189);
 x_194 = lean_box(0);
}
x_195 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_196 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_190);
x_197 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_197, 0, x_190);
x_198 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_198, 0, x_190);
x_199 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_199, 0, x_197);
lean_ctor_set(x_199, 1, x_198);
x_200 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_200, 0, x_193);
x_201 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_201, 0, x_192);
x_202 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_202, 0, x_191);
if (lean_is_scalar(x_194)) {
 x_203 = lean_alloc_ctor(0, 5, 0);
} else {
 x_203 = x_194;
}
lean_ctor_set(x_203, 0, x_199);
lean_ctor_set(x_203, 1, x_195);
lean_ctor_set(x_203, 2, x_202);
lean_ctor_set(x_203, 3, x_201);
lean_ctor_set(x_203, 4, x_200);
x_204 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_204, 0, x_203);
lean_ctor_set(x_204, 1, x_196);
x_205 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
lean_inc_ref(x_204);
x_206 = l_ReaderT_instMonad___redArg(x_204);
x_207 = l_ReaderT_instMonad___redArg(x_206);
x_208 = l_ReaderT_instMonad___redArg(x_207);
x_209 = lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
x_210 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_211 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_212 = lean_ctor_get(x_211, 0);
lean_inc(x_212);
x_213 = lp_aesop_Aesop_expandNextGoal___redArg___closed__7;
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_214 = lp_aesop_Aesop_nextActiveGoal___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_214) == 0)
{
lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; 
x_215 = lean_ctor_get(x_214, 0);
lean_inc(x_215);
lean_dec_ref(x_214);
lean_inc(x_215);
x_216 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__0___boxed), 10, 1);
lean_closure_set(x_216, 0, x_215);
x_217 = lp_aesop_Aesop_expandNextGoal___redArg___lam__0(x_215, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
x_218 = lean_ctor_get(x_217, 0);
lean_inc(x_218);
lean_dec_ref(x_217);
x_219 = lean_st_ref_get(x_3);
lean_dec(x_219);
x_220 = lp_aesop_Aesop_getRootMetaState___redArg(x_4);
if (lean_obj_tag(x_220) == 0)
{
lean_object* x_221; lean_object* x_222; lean_object* x_223; 
x_221 = lean_ctor_get(x_220, 0);
lean_inc(x_221);
lean_dec_ref(x_220);
x_222 = lean_st_ref_get(x_3);
lean_dec(x_222);
lean_inc(x_218);
x_223 = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(x_218, x_221);
lean_dec(x_221);
if (lean_obj_tag(x_223) == 0)
{
lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; lean_object* x_236; lean_object* x_237; lean_object* x_238; lean_object* x_239; double x_240; lean_object* x_241; lean_object* x_242; lean_object* x_243; uint8_t x_244; lean_object* x_245; lean_object* x_246; lean_object* x_247; 
x_224 = lean_ctor_get(x_223, 0);
lean_inc(x_224);
lean_dec_ref(x_223);
x_225 = lean_ctor_get(x_224, 0);
lean_inc(x_225);
x_226 = lean_ctor_get(x_224, 1);
lean_inc(x_226);
lean_dec(x_224);
x_227 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_228 = lean_ctor_get(x_227, 1);
lean_inc_ref(x_228);
lean_inc(x_218);
x_229 = lean_apply_1(x_228, x_218);
x_230 = lean_ctor_get(x_229, 0);
lean_inc(x_230);
lean_dec_ref(x_229);
x_231 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3;
x_232 = lean_ctor_get(x_231, 0);
lean_inc(x_232);
x_233 = l_Lean_Meta_instAddMessageContextMetaM;
x_234 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
x_235 = lp_aesop_Aesop_expandNextGoal___redArg___closed__11;
x_236 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc(x_225);
x_237 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__1___boxed), 13, 8);
lean_closure_set(x_237, 0, x_204);
lean_closure_set(x_237, 1, x_234);
lean_closure_set(x_237, 2, x_231);
lean_closure_set(x_237, 3, x_12);
lean_closure_set(x_237, 4, x_11);
lean_closure_set(x_237, 5, x_225);
lean_closure_set(x_237, 6, x_13);
lean_closure_set(x_237, 7, x_233);
lean_inc(x_226);
x_238 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__2___boxed), 11, 2);
lean_closure_set(x_238, 0, x_226);
lean_closure_set(x_238, 1, x_237);
lean_inc_ref(x_210);
lean_inc(x_212);
lean_inc_ref(x_205);
lean_inc_ref(x_1);
x_239 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___boxed), 19, 9);
lean_closure_set(x_239, 0, x_216);
lean_closure_set(x_239, 1, x_1);
lean_closure_set(x_239, 2, x_215);
lean_closure_set(x_239, 3, x_205);
lean_closure_set(x_239, 4, x_212);
lean_closure_set(x_239, 5, x_231);
lean_closure_set(x_239, 6, x_209);
lean_closure_set(x_239, 7, x_210);
lean_closure_set(x_239, 8, x_236);
x_240 = lp_aesop_Aesop_Goal_priority(x_218);
x_241 = lean_box_float(x_240);
lean_inc_ref(x_1);
x_242 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___boxed), 16, 6);
lean_closure_set(x_242, 0, lean_box(0));
lean_closure_set(x_242, 1, x_1);
lean_closure_set(x_242, 2, x_230);
lean_closure_set(x_242, 3, x_241);
lean_closure_set(x_242, 4, x_225);
lean_closure_set(x_242, 5, x_226);
x_243 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_243, 0, lean_box(0));
lean_closure_set(x_243, 1, lean_box(0));
lean_closure_set(x_243, 2, x_208);
lean_closure_set(x_243, 3, lean_box(0));
lean_closure_set(x_243, 4, lean_box(0));
lean_closure_set(x_243, 5, x_238);
lean_closure_set(x_243, 6, x_239);
x_244 = 1;
x_245 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_246 = l_Lean_withTraceNode___redArg(x_205, x_209, x_210, x_236, x_212, x_213, x_235, x_232, x_242, x_243, x_244, x_245);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_247 = lean_apply_9(x_246, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
if (lean_obj_tag(x_247) == 0)
{
lean_object* x_248; lean_object* x_249; 
x_248 = lean_ctor_get(x_247, 0);
lean_inc(x_248);
if (lean_is_exclusive(x_247)) {
 lean_ctor_release(x_247, 0);
 x_249 = x_247;
} else {
 lean_dec_ref(x_247);
 x_249 = lean_box(0);
}
if (lean_obj_tag(x_248) == 2)
{
lean_object* x_250; lean_object* x_251; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_250 = lean_box(0);
if (lean_is_scalar(x_249)) {
 x_251 = lean_alloc_ctor(0, 1, 0);
} else {
 x_251 = x_249;
}
lean_ctor_set(x_251, 0, x_250);
return x_251;
}
else
{
lean_object* x_252; lean_object* x_253; 
lean_dec(x_249);
x_252 = lean_ctor_get(x_248, 0);
lean_inc_ref(x_252);
lean_dec(x_248);
x_253 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(x_1, x_252, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_1);
return x_253;
}
}
else
{
lean_object* x_254; lean_object* x_255; lean_object* x_256; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_254 = lean_ctor_get(x_247, 0);
lean_inc(x_254);
if (lean_is_exclusive(x_247)) {
 lean_ctor_release(x_247, 0);
 x_255 = x_247;
} else {
 lean_dec_ref(x_247);
 x_255 = lean_box(0);
}
if (lean_is_scalar(x_255)) {
 x_256 = lean_alloc_ctor(1, 1, 0);
} else {
 x_256 = x_255;
}
lean_ctor_set(x_256, 0, x_254);
return x_256;
}
}
else
{
lean_object* x_257; lean_object* x_258; lean_object* x_259; 
lean_dec(x_218);
lean_dec_ref(x_216);
lean_dec(x_215);
lean_dec(x_212);
lean_dec_ref(x_210);
lean_dec_ref(x_208);
lean_dec_ref(x_205);
lean_dec_ref(x_204);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_257 = lean_ctor_get(x_223, 0);
lean_inc(x_257);
if (lean_is_exclusive(x_223)) {
 lean_ctor_release(x_223, 0);
 x_258 = x_223;
} else {
 lean_dec_ref(x_223);
 x_258 = lean_box(0);
}
if (lean_is_scalar(x_258)) {
 x_259 = lean_alloc_ctor(1, 1, 0);
} else {
 x_259 = x_258;
}
lean_ctor_set(x_259, 0, x_257);
return x_259;
}
}
else
{
lean_object* x_260; lean_object* x_261; lean_object* x_262; 
lean_dec(x_218);
lean_dec_ref(x_216);
lean_dec(x_215);
lean_dec(x_212);
lean_dec_ref(x_210);
lean_dec_ref(x_208);
lean_dec_ref(x_205);
lean_dec_ref(x_204);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_260 = lean_ctor_get(x_220, 0);
lean_inc(x_260);
if (lean_is_exclusive(x_220)) {
 lean_ctor_release(x_220, 0);
 x_261 = x_220;
} else {
 lean_dec_ref(x_220);
 x_261 = lean_box(0);
}
if (lean_is_scalar(x_261)) {
 x_262 = lean_alloc_ctor(1, 1, 0);
} else {
 x_262 = x_261;
}
lean_ctor_set(x_262, 0, x_260);
return x_262;
}
}
else
{
lean_object* x_263; lean_object* x_264; lean_object* x_265; 
lean_dec(x_212);
lean_dec_ref(x_210);
lean_dec_ref(x_208);
lean_dec_ref(x_205);
lean_dec_ref(x_204);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_263 = lean_ctor_get(x_214, 0);
lean_inc(x_263);
if (lean_is_exclusive(x_214)) {
 lean_ctor_release(x_214, 0);
 x_264 = x_214;
} else {
 lean_dec_ref(x_214);
 x_264 = lean_box(0);
}
if (lean_is_scalar(x_264)) {
 x_265 = lean_alloc_ctor(1, 1, 0);
} else {
 x_265 = x_264;
}
lean_ctor_set(x_265, 0, x_263);
return x_265;
}
}
}
else
{
lean_object* x_266; lean_object* x_267; lean_object* x_268; lean_object* x_269; lean_object* x_270; lean_object* x_271; lean_object* x_272; lean_object* x_273; lean_object* x_274; lean_object* x_275; lean_object* x_276; lean_object* x_277; lean_object* x_278; lean_object* x_279; lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; lean_object* x_291; lean_object* x_292; lean_object* x_293; lean_object* x_294; lean_object* x_295; lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; lean_object* x_301; lean_object* x_302; lean_object* x_303; lean_object* x_304; lean_object* x_305; lean_object* x_306; 
x_266 = lean_ctor_get(x_16, 0);
x_267 = lean_ctor_get(x_16, 2);
x_268 = lean_ctor_get(x_16, 3);
x_269 = lean_ctor_get(x_16, 4);
lean_inc(x_269);
lean_inc(x_268);
lean_inc(x_267);
lean_inc(x_266);
lean_dec(x_16);
x_270 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_271 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_266);
x_272 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_272, 0, x_266);
x_273 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_273, 0, x_266);
x_274 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_274, 0, x_272);
lean_ctor_set(x_274, 1, x_273);
x_275 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_275, 0, x_269);
x_276 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_276, 0, x_268);
x_277 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_277, 0, x_267);
x_278 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_278, 0, x_274);
lean_ctor_set(x_278, 1, x_270);
lean_ctor_set(x_278, 2, x_277);
lean_ctor_set(x_278, 3, x_276);
lean_ctor_set(x_278, 4, x_275);
lean_ctor_set(x_14, 1, x_271);
lean_ctor_set(x_14, 0, x_278);
x_279 = l_ReaderT_instMonad___redArg(x_14);
x_280 = lean_ctor_get(x_279, 0);
lean_inc_ref(x_280);
if (lean_is_exclusive(x_279)) {
 lean_ctor_release(x_279, 0);
 lean_ctor_release(x_279, 1);
 x_281 = x_279;
} else {
 lean_dec_ref(x_279);
 x_281 = lean_box(0);
}
x_282 = lean_ctor_get(x_280, 0);
lean_inc_ref(x_282);
x_283 = lean_ctor_get(x_280, 2);
lean_inc(x_283);
x_284 = lean_ctor_get(x_280, 3);
lean_inc(x_284);
x_285 = lean_ctor_get(x_280, 4);
lean_inc(x_285);
if (lean_is_exclusive(x_280)) {
 lean_ctor_release(x_280, 0);
 lean_ctor_release(x_280, 1);
 lean_ctor_release(x_280, 2);
 lean_ctor_release(x_280, 3);
 lean_ctor_release(x_280, 4);
 x_286 = x_280;
} else {
 lean_dec_ref(x_280);
 x_286 = lean_box(0);
}
x_287 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_288 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_282);
x_289 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_289, 0, x_282);
x_290 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_290, 0, x_282);
x_291 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_291, 0, x_289);
lean_ctor_set(x_291, 1, x_290);
x_292 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_292, 0, x_285);
x_293 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_293, 0, x_284);
x_294 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_294, 0, x_283);
if (lean_is_scalar(x_286)) {
 x_295 = lean_alloc_ctor(0, 5, 0);
} else {
 x_295 = x_286;
}
lean_ctor_set(x_295, 0, x_291);
lean_ctor_set(x_295, 1, x_287);
lean_ctor_set(x_295, 2, x_294);
lean_ctor_set(x_295, 3, x_293);
lean_ctor_set(x_295, 4, x_292);
if (lean_is_scalar(x_281)) {
 x_296 = lean_alloc_ctor(0, 2, 0);
} else {
 x_296 = x_281;
}
lean_ctor_set(x_296, 0, x_295);
lean_ctor_set(x_296, 1, x_288);
x_297 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
lean_inc_ref(x_296);
x_298 = l_ReaderT_instMonad___redArg(x_296);
x_299 = l_ReaderT_instMonad___redArg(x_298);
x_300 = l_ReaderT_instMonad___redArg(x_299);
x_301 = lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
x_302 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_303 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_304 = lean_ctor_get(x_303, 0);
lean_inc(x_304);
x_305 = lp_aesop_Aesop_expandNextGoal___redArg___closed__7;
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_306 = lp_aesop_Aesop_nextActiveGoal___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_306) == 0)
{
lean_object* x_307; lean_object* x_308; lean_object* x_309; lean_object* x_310; lean_object* x_311; lean_object* x_312; 
x_307 = lean_ctor_get(x_306, 0);
lean_inc(x_307);
lean_dec_ref(x_306);
lean_inc(x_307);
x_308 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__0___boxed), 10, 1);
lean_closure_set(x_308, 0, x_307);
x_309 = lp_aesop_Aesop_expandNextGoal___redArg___lam__0(x_307, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
x_310 = lean_ctor_get(x_309, 0);
lean_inc(x_310);
lean_dec_ref(x_309);
x_311 = lean_st_ref_get(x_3);
lean_dec(x_311);
x_312 = lp_aesop_Aesop_getRootMetaState___redArg(x_4);
if (lean_obj_tag(x_312) == 0)
{
lean_object* x_313; lean_object* x_314; lean_object* x_315; 
x_313 = lean_ctor_get(x_312, 0);
lean_inc(x_313);
lean_dec_ref(x_312);
x_314 = lean_st_ref_get(x_3);
lean_dec(x_314);
lean_inc(x_310);
x_315 = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(x_310, x_313);
lean_dec(x_313);
if (lean_obj_tag(x_315) == 0)
{
lean_object* x_316; lean_object* x_317; lean_object* x_318; lean_object* x_319; lean_object* x_320; lean_object* x_321; lean_object* x_322; lean_object* x_323; lean_object* x_324; lean_object* x_325; lean_object* x_326; lean_object* x_327; lean_object* x_328; lean_object* x_329; lean_object* x_330; lean_object* x_331; double x_332; lean_object* x_333; lean_object* x_334; lean_object* x_335; uint8_t x_336; lean_object* x_337; lean_object* x_338; lean_object* x_339; 
x_316 = lean_ctor_get(x_315, 0);
lean_inc(x_316);
lean_dec_ref(x_315);
x_317 = lean_ctor_get(x_316, 0);
lean_inc(x_317);
x_318 = lean_ctor_get(x_316, 1);
lean_inc(x_318);
lean_dec(x_316);
x_319 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_320 = lean_ctor_get(x_319, 1);
lean_inc_ref(x_320);
lean_inc(x_310);
x_321 = lean_apply_1(x_320, x_310);
x_322 = lean_ctor_get(x_321, 0);
lean_inc(x_322);
lean_dec_ref(x_321);
x_323 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3;
x_324 = lean_ctor_get(x_323, 0);
lean_inc(x_324);
x_325 = l_Lean_Meta_instAddMessageContextMetaM;
x_326 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
x_327 = lp_aesop_Aesop_expandNextGoal___redArg___closed__11;
x_328 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc(x_317);
x_329 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__1___boxed), 13, 8);
lean_closure_set(x_329, 0, x_296);
lean_closure_set(x_329, 1, x_326);
lean_closure_set(x_329, 2, x_323);
lean_closure_set(x_329, 3, x_12);
lean_closure_set(x_329, 4, x_11);
lean_closure_set(x_329, 5, x_317);
lean_closure_set(x_329, 6, x_13);
lean_closure_set(x_329, 7, x_325);
lean_inc(x_318);
x_330 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__2___boxed), 11, 2);
lean_closure_set(x_330, 0, x_318);
lean_closure_set(x_330, 1, x_329);
lean_inc_ref(x_302);
lean_inc(x_304);
lean_inc_ref(x_297);
lean_inc_ref(x_1);
x_331 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___boxed), 19, 9);
lean_closure_set(x_331, 0, x_308);
lean_closure_set(x_331, 1, x_1);
lean_closure_set(x_331, 2, x_307);
lean_closure_set(x_331, 3, x_297);
lean_closure_set(x_331, 4, x_304);
lean_closure_set(x_331, 5, x_323);
lean_closure_set(x_331, 6, x_301);
lean_closure_set(x_331, 7, x_302);
lean_closure_set(x_331, 8, x_328);
x_332 = lp_aesop_Aesop_Goal_priority(x_310);
x_333 = lean_box_float(x_332);
lean_inc_ref(x_1);
x_334 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___boxed), 16, 6);
lean_closure_set(x_334, 0, lean_box(0));
lean_closure_set(x_334, 1, x_1);
lean_closure_set(x_334, 2, x_322);
lean_closure_set(x_334, 3, x_333);
lean_closure_set(x_334, 4, x_317);
lean_closure_set(x_334, 5, x_318);
x_335 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_335, 0, lean_box(0));
lean_closure_set(x_335, 1, lean_box(0));
lean_closure_set(x_335, 2, x_300);
lean_closure_set(x_335, 3, lean_box(0));
lean_closure_set(x_335, 4, lean_box(0));
lean_closure_set(x_335, 5, x_330);
lean_closure_set(x_335, 6, x_331);
x_336 = 1;
x_337 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_338 = l_Lean_withTraceNode___redArg(x_297, x_301, x_302, x_328, x_304, x_305, x_327, x_324, x_334, x_335, x_336, x_337);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_339 = lean_apply_9(x_338, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
if (lean_obj_tag(x_339) == 0)
{
lean_object* x_340; lean_object* x_341; 
x_340 = lean_ctor_get(x_339, 0);
lean_inc(x_340);
if (lean_is_exclusive(x_339)) {
 lean_ctor_release(x_339, 0);
 x_341 = x_339;
} else {
 lean_dec_ref(x_339);
 x_341 = lean_box(0);
}
if (lean_obj_tag(x_340) == 2)
{
lean_object* x_342; lean_object* x_343; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_342 = lean_box(0);
if (lean_is_scalar(x_341)) {
 x_343 = lean_alloc_ctor(0, 1, 0);
} else {
 x_343 = x_341;
}
lean_ctor_set(x_343, 0, x_342);
return x_343;
}
else
{
lean_object* x_344; lean_object* x_345; 
lean_dec(x_341);
x_344 = lean_ctor_get(x_340, 0);
lean_inc_ref(x_344);
lean_dec(x_340);
x_345 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(x_1, x_344, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_1);
return x_345;
}
}
else
{
lean_object* x_346; lean_object* x_347; lean_object* x_348; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_346 = lean_ctor_get(x_339, 0);
lean_inc(x_346);
if (lean_is_exclusive(x_339)) {
 lean_ctor_release(x_339, 0);
 x_347 = x_339;
} else {
 lean_dec_ref(x_339);
 x_347 = lean_box(0);
}
if (lean_is_scalar(x_347)) {
 x_348 = lean_alloc_ctor(1, 1, 0);
} else {
 x_348 = x_347;
}
lean_ctor_set(x_348, 0, x_346);
return x_348;
}
}
else
{
lean_object* x_349; lean_object* x_350; lean_object* x_351; 
lean_dec(x_310);
lean_dec_ref(x_308);
lean_dec(x_307);
lean_dec(x_304);
lean_dec_ref(x_302);
lean_dec_ref(x_300);
lean_dec_ref(x_297);
lean_dec_ref(x_296);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_349 = lean_ctor_get(x_315, 0);
lean_inc(x_349);
if (lean_is_exclusive(x_315)) {
 lean_ctor_release(x_315, 0);
 x_350 = x_315;
} else {
 lean_dec_ref(x_315);
 x_350 = lean_box(0);
}
if (lean_is_scalar(x_350)) {
 x_351 = lean_alloc_ctor(1, 1, 0);
} else {
 x_351 = x_350;
}
lean_ctor_set(x_351, 0, x_349);
return x_351;
}
}
else
{
lean_object* x_352; lean_object* x_353; lean_object* x_354; 
lean_dec(x_310);
lean_dec_ref(x_308);
lean_dec(x_307);
lean_dec(x_304);
lean_dec_ref(x_302);
lean_dec_ref(x_300);
lean_dec_ref(x_297);
lean_dec_ref(x_296);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_352 = lean_ctor_get(x_312, 0);
lean_inc(x_352);
if (lean_is_exclusive(x_312)) {
 lean_ctor_release(x_312, 0);
 x_353 = x_312;
} else {
 lean_dec_ref(x_312);
 x_353 = lean_box(0);
}
if (lean_is_scalar(x_353)) {
 x_354 = lean_alloc_ctor(1, 1, 0);
} else {
 x_354 = x_353;
}
lean_ctor_set(x_354, 0, x_352);
return x_354;
}
}
else
{
lean_object* x_355; lean_object* x_356; lean_object* x_357; 
lean_dec(x_304);
lean_dec_ref(x_302);
lean_dec_ref(x_300);
lean_dec_ref(x_297);
lean_dec_ref(x_296);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_355 = lean_ctor_get(x_306, 0);
lean_inc(x_355);
if (lean_is_exclusive(x_306)) {
 lean_ctor_release(x_306, 0);
 x_356 = x_306;
} else {
 lean_dec_ref(x_306);
 x_356 = lean_box(0);
}
if (lean_is_scalar(x_356)) {
 x_357 = lean_alloc_ctor(1, 1, 0);
} else {
 x_357 = x_356;
}
lean_ctor_set(x_357, 0, x_355);
return x_357;
}
}
}
else
{
lean_object* x_358; lean_object* x_359; lean_object* x_360; lean_object* x_361; lean_object* x_362; lean_object* x_363; lean_object* x_364; lean_object* x_365; lean_object* x_366; lean_object* x_367; lean_object* x_368; lean_object* x_369; lean_object* x_370; lean_object* x_371; lean_object* x_372; lean_object* x_373; lean_object* x_374; lean_object* x_375; lean_object* x_376; lean_object* x_377; lean_object* x_378; lean_object* x_379; lean_object* x_380; lean_object* x_381; lean_object* x_382; lean_object* x_383; lean_object* x_384; lean_object* x_385; lean_object* x_386; lean_object* x_387; lean_object* x_388; lean_object* x_389; lean_object* x_390; lean_object* x_391; lean_object* x_392; lean_object* x_393; lean_object* x_394; lean_object* x_395; lean_object* x_396; lean_object* x_397; lean_object* x_398; lean_object* x_399; lean_object* x_400; lean_object* x_401; 
x_358 = lean_ctor_get(x_14, 0);
lean_inc(x_358);
lean_dec(x_14);
x_359 = lean_ctor_get(x_358, 0);
lean_inc_ref(x_359);
x_360 = lean_ctor_get(x_358, 2);
lean_inc(x_360);
x_361 = lean_ctor_get(x_358, 3);
lean_inc(x_361);
x_362 = lean_ctor_get(x_358, 4);
lean_inc(x_362);
if (lean_is_exclusive(x_358)) {
 lean_ctor_release(x_358, 0);
 lean_ctor_release(x_358, 1);
 lean_ctor_release(x_358, 2);
 lean_ctor_release(x_358, 3);
 lean_ctor_release(x_358, 4);
 x_363 = x_358;
} else {
 lean_dec_ref(x_358);
 x_363 = lean_box(0);
}
x_364 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_365 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_359);
x_366 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_366, 0, x_359);
x_367 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_367, 0, x_359);
x_368 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_368, 0, x_366);
lean_ctor_set(x_368, 1, x_367);
x_369 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_369, 0, x_362);
x_370 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_370, 0, x_361);
x_371 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_371, 0, x_360);
if (lean_is_scalar(x_363)) {
 x_372 = lean_alloc_ctor(0, 5, 0);
} else {
 x_372 = x_363;
}
lean_ctor_set(x_372, 0, x_368);
lean_ctor_set(x_372, 1, x_364);
lean_ctor_set(x_372, 2, x_371);
lean_ctor_set(x_372, 3, x_370);
lean_ctor_set(x_372, 4, x_369);
x_373 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_373, 0, x_372);
lean_ctor_set(x_373, 1, x_365);
x_374 = l_ReaderT_instMonad___redArg(x_373);
x_375 = lean_ctor_get(x_374, 0);
lean_inc_ref(x_375);
if (lean_is_exclusive(x_374)) {
 lean_ctor_release(x_374, 0);
 lean_ctor_release(x_374, 1);
 x_376 = x_374;
} else {
 lean_dec_ref(x_374);
 x_376 = lean_box(0);
}
x_377 = lean_ctor_get(x_375, 0);
lean_inc_ref(x_377);
x_378 = lean_ctor_get(x_375, 2);
lean_inc(x_378);
x_379 = lean_ctor_get(x_375, 3);
lean_inc(x_379);
x_380 = lean_ctor_get(x_375, 4);
lean_inc(x_380);
if (lean_is_exclusive(x_375)) {
 lean_ctor_release(x_375, 0);
 lean_ctor_release(x_375, 1);
 lean_ctor_release(x_375, 2);
 lean_ctor_release(x_375, 3);
 lean_ctor_release(x_375, 4);
 x_381 = x_375;
} else {
 lean_dec_ref(x_375);
 x_381 = lean_box(0);
}
x_382 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_383 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_377);
x_384 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_384, 0, x_377);
x_385 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_385, 0, x_377);
x_386 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_386, 0, x_384);
lean_ctor_set(x_386, 1, x_385);
x_387 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_387, 0, x_380);
x_388 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_388, 0, x_379);
x_389 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_389, 0, x_378);
if (lean_is_scalar(x_381)) {
 x_390 = lean_alloc_ctor(0, 5, 0);
} else {
 x_390 = x_381;
}
lean_ctor_set(x_390, 0, x_386);
lean_ctor_set(x_390, 1, x_382);
lean_ctor_set(x_390, 2, x_389);
lean_ctor_set(x_390, 3, x_388);
lean_ctor_set(x_390, 4, x_387);
if (lean_is_scalar(x_376)) {
 x_391 = lean_alloc_ctor(0, 2, 0);
} else {
 x_391 = x_376;
}
lean_ctor_set(x_391, 0, x_390);
lean_ctor_set(x_391, 1, x_383);
x_392 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
lean_inc_ref(x_391);
x_393 = l_ReaderT_instMonad___redArg(x_391);
x_394 = l_ReaderT_instMonad___redArg(x_393);
x_395 = l_ReaderT_instMonad___redArg(x_394);
x_396 = lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
x_397 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_398 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_399 = lean_ctor_get(x_398, 0);
lean_inc(x_399);
x_400 = lp_aesop_Aesop_expandNextGoal___redArg___closed__7;
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_401 = lp_aesop_Aesop_nextActiveGoal___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_401) == 0)
{
lean_object* x_402; lean_object* x_403; lean_object* x_404; lean_object* x_405; lean_object* x_406; lean_object* x_407; 
x_402 = lean_ctor_get(x_401, 0);
lean_inc(x_402);
lean_dec_ref(x_401);
lean_inc(x_402);
x_403 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__0___boxed), 10, 1);
lean_closure_set(x_403, 0, x_402);
x_404 = lp_aesop_Aesop_expandNextGoal___redArg___lam__0(x_402, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
x_405 = lean_ctor_get(x_404, 0);
lean_inc(x_405);
lean_dec_ref(x_404);
x_406 = lean_st_ref_get(x_3);
lean_dec(x_406);
x_407 = lp_aesop_Aesop_getRootMetaState___redArg(x_4);
if (lean_obj_tag(x_407) == 0)
{
lean_object* x_408; lean_object* x_409; lean_object* x_410; 
x_408 = lean_ctor_get(x_407, 0);
lean_inc(x_408);
lean_dec_ref(x_407);
x_409 = lean_st_ref_get(x_3);
lean_dec(x_409);
lean_inc(x_405);
x_410 = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(x_405, x_408);
lean_dec(x_408);
if (lean_obj_tag(x_410) == 0)
{
lean_object* x_411; lean_object* x_412; lean_object* x_413; lean_object* x_414; lean_object* x_415; lean_object* x_416; lean_object* x_417; lean_object* x_418; lean_object* x_419; lean_object* x_420; lean_object* x_421; lean_object* x_422; lean_object* x_423; lean_object* x_424; lean_object* x_425; lean_object* x_426; double x_427; lean_object* x_428; lean_object* x_429; lean_object* x_430; uint8_t x_431; lean_object* x_432; lean_object* x_433; lean_object* x_434; 
x_411 = lean_ctor_get(x_410, 0);
lean_inc(x_411);
lean_dec_ref(x_410);
x_412 = lean_ctor_get(x_411, 0);
lean_inc(x_412);
x_413 = lean_ctor_get(x_411, 1);
lean_inc(x_413);
lean_dec(x_411);
x_414 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_415 = lean_ctor_get(x_414, 1);
lean_inc_ref(x_415);
lean_inc(x_405);
x_416 = lean_apply_1(x_415, x_405);
x_417 = lean_ctor_get(x_416, 0);
lean_inc(x_417);
lean_dec_ref(x_416);
x_418 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3;
x_419 = lean_ctor_get(x_418, 0);
lean_inc(x_419);
x_420 = l_Lean_Meta_instAddMessageContextMetaM;
x_421 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9;
x_422 = lp_aesop_Aesop_expandNextGoal___redArg___closed__11;
x_423 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc(x_412);
x_424 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__1___boxed), 13, 8);
lean_closure_set(x_424, 0, x_391);
lean_closure_set(x_424, 1, x_421);
lean_closure_set(x_424, 2, x_418);
lean_closure_set(x_424, 3, x_12);
lean_closure_set(x_424, 4, x_11);
lean_closure_set(x_424, 5, x_412);
lean_closure_set(x_424, 6, x_13);
lean_closure_set(x_424, 7, x_420);
lean_inc(x_413);
x_425 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__2___boxed), 11, 2);
lean_closure_set(x_425, 0, x_413);
lean_closure_set(x_425, 1, x_424);
lean_inc_ref(x_397);
lean_inc(x_399);
lean_inc_ref(x_392);
lean_inc_ref(x_1);
x_426 = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___boxed), 19, 9);
lean_closure_set(x_426, 0, x_403);
lean_closure_set(x_426, 1, x_1);
lean_closure_set(x_426, 2, x_402);
lean_closure_set(x_426, 3, x_392);
lean_closure_set(x_426, 4, x_399);
lean_closure_set(x_426, 5, x_418);
lean_closure_set(x_426, 6, x_396);
lean_closure_set(x_426, 7, x_397);
lean_closure_set(x_426, 8, x_423);
x_427 = lp_aesop_Aesop_Goal_priority(x_405);
x_428 = lean_box_float(x_427);
lean_inc_ref(x_1);
x_429 = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___boxed), 16, 6);
lean_closure_set(x_429, 0, lean_box(0));
lean_closure_set(x_429, 1, x_1);
lean_closure_set(x_429, 2, x_417);
lean_closure_set(x_429, 3, x_428);
lean_closure_set(x_429, 4, x_412);
lean_closure_set(x_429, 5, x_413);
x_430 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_430, 0, lean_box(0));
lean_closure_set(x_430, 1, lean_box(0));
lean_closure_set(x_430, 2, x_395);
lean_closure_set(x_430, 3, lean_box(0));
lean_closure_set(x_430, 4, lean_box(0));
lean_closure_set(x_430, 5, x_425);
lean_closure_set(x_430, 6, x_426);
x_431 = 1;
x_432 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_433 = l_Lean_withTraceNode___redArg(x_392, x_396, x_397, x_423, x_399, x_400, x_422, x_419, x_429, x_430, x_431, x_432);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_434 = lean_apply_9(x_433, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
if (lean_obj_tag(x_434) == 0)
{
lean_object* x_435; lean_object* x_436; 
x_435 = lean_ctor_get(x_434, 0);
lean_inc(x_435);
if (lean_is_exclusive(x_434)) {
 lean_ctor_release(x_434, 0);
 x_436 = x_434;
} else {
 lean_dec_ref(x_434);
 x_436 = lean_box(0);
}
if (lean_obj_tag(x_435) == 2)
{
lean_object* x_437; lean_object* x_438; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_437 = lean_box(0);
if (lean_is_scalar(x_436)) {
 x_438 = lean_alloc_ctor(0, 1, 0);
} else {
 x_438 = x_436;
}
lean_ctor_set(x_438, 0, x_437);
return x_438;
}
else
{
lean_object* x_439; lean_object* x_440; 
lean_dec(x_436);
x_439 = lean_ctor_get(x_435, 0);
lean_inc_ref(x_439);
lean_dec(x_435);
x_440 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(x_1, x_439, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_1);
return x_440;
}
}
else
{
lean_object* x_441; lean_object* x_442; lean_object* x_443; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_441 = lean_ctor_get(x_434, 0);
lean_inc(x_441);
if (lean_is_exclusive(x_434)) {
 lean_ctor_release(x_434, 0);
 x_442 = x_434;
} else {
 lean_dec_ref(x_434);
 x_442 = lean_box(0);
}
if (lean_is_scalar(x_442)) {
 x_443 = lean_alloc_ctor(1, 1, 0);
} else {
 x_443 = x_442;
}
lean_ctor_set(x_443, 0, x_441);
return x_443;
}
}
else
{
lean_object* x_444; lean_object* x_445; lean_object* x_446; 
lean_dec(x_405);
lean_dec_ref(x_403);
lean_dec(x_402);
lean_dec(x_399);
lean_dec_ref(x_397);
lean_dec_ref(x_395);
lean_dec_ref(x_392);
lean_dec_ref(x_391);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_444 = lean_ctor_get(x_410, 0);
lean_inc(x_444);
if (lean_is_exclusive(x_410)) {
 lean_ctor_release(x_410, 0);
 x_445 = x_410;
} else {
 lean_dec_ref(x_410);
 x_445 = lean_box(0);
}
if (lean_is_scalar(x_445)) {
 x_446 = lean_alloc_ctor(1, 1, 0);
} else {
 x_446 = x_445;
}
lean_ctor_set(x_446, 0, x_444);
return x_446;
}
}
else
{
lean_object* x_447; lean_object* x_448; lean_object* x_449; 
lean_dec(x_405);
lean_dec_ref(x_403);
lean_dec(x_402);
lean_dec(x_399);
lean_dec_ref(x_397);
lean_dec_ref(x_395);
lean_dec_ref(x_392);
lean_dec_ref(x_391);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_447 = lean_ctor_get(x_407, 0);
lean_inc(x_447);
if (lean_is_exclusive(x_407)) {
 lean_ctor_release(x_407, 0);
 x_448 = x_407;
} else {
 lean_dec_ref(x_407);
 x_448 = lean_box(0);
}
if (lean_is_scalar(x_448)) {
 x_449 = lean_alloc_ctor(1, 1, 0);
} else {
 x_449 = x_448;
}
lean_ctor_set(x_449, 0, x_447);
return x_449;
}
}
else
{
lean_object* x_450; lean_object* x_451; lean_object* x_452; 
lean_dec(x_399);
lean_dec_ref(x_397);
lean_dec_ref(x_395);
lean_dec_ref(x_392);
lean_dec_ref(x_391);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_450 = lean_ctor_get(x_401, 0);
lean_inc(x_450);
if (lean_is_exclusive(x_401)) {
 lean_ctor_release(x_401, 0);
 x_451 = x_401;
} else {
 lean_dec_ref(x_401);
 x_451 = lean_box(0);
}
if (lean_is_scalar(x_451)) {
 x_452 = lean_alloc_ctor(1, 1, 0);
} else {
 x_452 = x_451;
}
lean_ctor_set(x_452, 0, x_450);
return x_452;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_expandNextGoal___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_expandNextGoal(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_expandNextGoal___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
static lean_object* _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("maximum number of goals (", 25, 25);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkGoalLimit___redArg___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(") reached. Set the 'maxGoals' option to increase the limit.", 59, 59);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkGoalLimit___redArg___closed__2;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_getTree___redArg(x_2, x_3);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
if (lean_is_exclusive(x_5)) {
 lean_ctor_release(x_5, 0);
 x_7 = x_5;
} else {
 lean_dec_ref(x_5);
 x_7 = lean_box(0);
}
x_11 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_11);
lean_dec_ref(x_1);
x_12 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_11);
x_13 = lean_ctor_get(x_12, 2);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_6, 2);
lean_inc(x_14);
lean_dec(x_6);
x_15 = lean_unsigned_to_nat(0u);
x_16 = lean_nat_dec_eq(x_13, x_15);
if (x_16 == 0)
{
uint8_t x_17; 
x_17 = lean_nat_dec_le(x_13, x_14);
lean_dec(x_14);
if (x_17 == 0)
{
lean_dec(x_13);
goto block_10;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
lean_dec(x_7);
x_18 = lp_aesop_Aesop_checkGoalLimit___redArg___closed__1;
x_19 = l_Nat_reprFast(x_13);
x_20 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_20, 0, x_19);
x_21 = l_Lean_MessageData_ofFormat(x_20);
x_22 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_22, 0, x_18);
lean_ctor_set(x_22, 1, x_21);
x_23 = lp_aesop_Aesop_checkGoalLimit___redArg___closed__3;
x_24 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_24, 0, x_22);
lean_ctor_set(x_24, 1, x_23);
x_25 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_25, 0, x_24);
x_26 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_26, 0, x_25);
return x_26;
}
}
else
{
lean_dec(x_14);
lean_dec(x_13);
goto block_10;
}
block_10:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_box(0);
if (lean_is_scalar(x_7)) {
 x_9 = lean_alloc_ctor(0, 1, 0);
} else {
 x_9 = x_7;
}
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
}
else
{
uint8_t x_27; 
lean_dec_ref(x_1);
x_27 = !lean_is_exclusive(x_5);
if (x_27 == 0)
{
return x_5;
}
else
{
lean_object* x_28; lean_object* x_29; 
x_28 = lean_ctor_get(x_5, 0);
lean_inc(x_28);
lean_dec(x_5);
x_29 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_29, 0, x_28);
return x_29;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_checkGoalLimit___redArg(x_3, x_4, x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_checkGoalLimit(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_checkGoalLimit___redArg(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("maximum number of rule applications (", 37, 37);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkRappLimit___redArg___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(") reached. Set the 'maxRuleApplications' option to increase the limit.", 70, 70);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkRappLimit___redArg___closed__2;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_getTree___redArg(x_2, x_3);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
if (lean_is_exclusive(x_5)) {
 lean_ctor_release(x_5, 0);
 x_7 = x_5;
} else {
 lean_dec_ref(x_5);
 x_7 = lean_box(0);
}
x_11 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_11);
lean_dec_ref(x_1);
x_12 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_12);
lean_dec_ref(x_11);
x_13 = lean_ctor_get(x_12, 1);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_6, 3);
lean_inc(x_14);
lean_dec(x_6);
x_15 = lean_unsigned_to_nat(0u);
x_16 = lean_nat_dec_eq(x_13, x_15);
if (x_16 == 0)
{
uint8_t x_17; 
x_17 = lean_nat_dec_le(x_13, x_14);
lean_dec(x_14);
if (x_17 == 0)
{
lean_dec(x_13);
goto block_10;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
lean_dec(x_7);
x_18 = lp_aesop_Aesop_checkRappLimit___redArg___closed__1;
x_19 = l_Nat_reprFast(x_13);
x_20 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_20, 0, x_19);
x_21 = l_Lean_MessageData_ofFormat(x_20);
x_22 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_22, 0, x_18);
lean_ctor_set(x_22, 1, x_21);
x_23 = lp_aesop_Aesop_checkRappLimit___redArg___closed__3;
x_24 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_24, 0, x_22);
lean_ctor_set(x_24, 1, x_23);
x_25 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_25, 0, x_24);
x_26 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_26, 0, x_25);
return x_26;
}
}
else
{
lean_dec(x_14);
lean_dec(x_13);
goto block_10;
}
block_10:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_box(0);
if (lean_is_scalar(x_7)) {
 x_9 = lean_alloc_ctor(0, 1, 0);
} else {
 x_9 = x_7;
}
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
}
else
{
uint8_t x_27; 
lean_dec_ref(x_1);
x_27 = !lean_is_exclusive(x_5);
if (x_27 == 0)
{
return x_5;
}
else
{
lean_object* x_28; lean_object* x_29; 
x_28 = lean_ctor_get(x_5, 0);
lean_inc(x_28);
lean_dec(x_5);
x_29 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_29, 0, x_28);
return x_29;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_checkRappLimit___redArg(x_3, x_4, x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_checkRappLimit(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_checkRappLimit___redArg(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("failed to prove the goal after exhaustive search.", 49, 49);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkRootUnprovable___redArg___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("failed to prove the goal. Some goals were not explored because the maximum rule application depth (", 99, 99);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkRootUnprovable___redArg___closed__2;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(") was reached. Set option 'maxRuleApplicationDepth' to increase the limit.", 74, 74);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_checkRootUnprovable___redArg___closed__4;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_getTree___redArg(x_2, x_3);
if (lean_obj_tag(x_5) == 0)
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; uint8_t x_15; 
x_7 = lean_ctor_get(x_5, 0);
x_8 = lean_st_ref_get(x_2);
lean_dec(x_8);
x_9 = lean_ctor_get(x_7, 0);
lean_inc(x_9);
lean_dec(x_7);
x_10 = lean_st_ref_get(x_9);
lean_dec(x_9);
x_11 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_12 = lean_ctor_get(x_11, 5);
lean_inc_ref(x_12);
x_13 = lean_apply_1(x_12, x_10);
x_14 = lean_ctor_get_uint8(x_13, sizeof(void*)*2 + 1);
lean_dec_ref(x_13);
x_15 = lp_aesop_Aesop_NodeState_isUnprovable(x_14);
if (x_15 == 0)
{
lean_object* x_16; 
lean_dec_ref(x_1);
x_16 = lean_box(0);
lean_ctor_set(x_5, 0, x_16);
return x_5;
}
else
{
lean_object* x_17; 
lean_free_object(x_5);
x_17 = lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(x_2);
if (lean_obj_tag(x_17) == 0)
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_24; 
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
if (lean_is_exclusive(x_17)) {
 lean_ctor_release(x_17, 0);
 x_19 = x_17;
} else {
 lean_dec_ref(x_17);
 x_19 = lean_box(0);
}
x_24 = lean_unbox(x_18);
lean_dec(x_18);
if (x_24 == 0)
{
lean_object* x_25; 
lean_dec_ref(x_1);
x_25 = lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1;
x_20 = x_25;
goto block_23;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_26 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_26);
lean_dec_ref(x_1);
x_27 = lean_ctor_get(x_26, 0);
lean_inc_ref(x_27);
lean_dec_ref(x_26);
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
lean_dec_ref(x_27);
x_29 = lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3;
x_30 = l_Nat_reprFast(x_28);
x_31 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_31, 0, x_30);
x_32 = l_Lean_MessageData_ofFormat(x_31);
x_33 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_33, 0, x_29);
lean_ctor_set(x_33, 1, x_32);
x_34 = lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5;
x_35 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_35, 0, x_33);
lean_ctor_set(x_35, 1, x_34);
x_20 = x_35;
goto block_23;
}
block_23:
{
lean_object* x_21; lean_object* x_22; 
x_21 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_21, 0, x_20);
if (lean_is_scalar(x_19)) {
 x_22 = lean_alloc_ctor(0, 1, 0);
} else {
 x_22 = x_19;
}
lean_ctor_set(x_22, 0, x_21);
return x_22;
}
}
else
{
uint8_t x_36; 
lean_dec_ref(x_1);
x_36 = !lean_is_exclusive(x_17);
if (x_36 == 0)
{
return x_17;
}
else
{
lean_object* x_37; lean_object* x_38; 
x_37 = lean_ctor_get(x_17, 0);
lean_inc(x_37);
lean_dec(x_17);
x_38 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_38, 0, x_37);
return x_38;
}
}
}
}
else
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; uint8_t x_46; uint8_t x_47; 
x_39 = lean_ctor_get(x_5, 0);
lean_inc(x_39);
lean_dec(x_5);
x_40 = lean_st_ref_get(x_2);
lean_dec(x_40);
x_41 = lean_ctor_get(x_39, 0);
lean_inc(x_41);
lean_dec(x_39);
x_42 = lean_st_ref_get(x_41);
lean_dec(x_41);
x_43 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_44 = lean_ctor_get(x_43, 5);
lean_inc_ref(x_44);
x_45 = lean_apply_1(x_44, x_42);
x_46 = lean_ctor_get_uint8(x_45, sizeof(void*)*2 + 1);
lean_dec_ref(x_45);
x_47 = lp_aesop_Aesop_NodeState_isUnprovable(x_46);
if (x_47 == 0)
{
lean_object* x_48; lean_object* x_49; 
lean_dec_ref(x_1);
x_48 = lean_box(0);
x_49 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_49, 0, x_48);
return x_49;
}
else
{
lean_object* x_50; 
x_50 = lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(x_2);
if (lean_obj_tag(x_50) == 0)
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; uint8_t x_57; 
x_51 = lean_ctor_get(x_50, 0);
lean_inc(x_51);
if (lean_is_exclusive(x_50)) {
 lean_ctor_release(x_50, 0);
 x_52 = x_50;
} else {
 lean_dec_ref(x_50);
 x_52 = lean_box(0);
}
x_57 = lean_unbox(x_51);
lean_dec(x_51);
if (x_57 == 0)
{
lean_object* x_58; 
lean_dec_ref(x_1);
x_58 = lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1;
x_53 = x_58;
goto block_56;
}
else
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_59 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_59);
lean_dec_ref(x_1);
x_60 = lean_ctor_get(x_59, 0);
lean_inc_ref(x_60);
lean_dec_ref(x_59);
x_61 = lean_ctor_get(x_60, 0);
lean_inc(x_61);
lean_dec_ref(x_60);
x_62 = lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3;
x_63 = l_Nat_reprFast(x_61);
x_64 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_64, 0, x_63);
x_65 = l_Lean_MessageData_ofFormat(x_64);
x_66 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_66, 0, x_62);
lean_ctor_set(x_66, 1, x_65);
x_67 = lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5;
x_68 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_68, 0, x_66);
lean_ctor_set(x_68, 1, x_67);
x_53 = x_68;
goto block_56;
}
block_56:
{
lean_object* x_54; lean_object* x_55; 
x_54 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_54, 0, x_53);
if (lean_is_scalar(x_52)) {
 x_55 = lean_alloc_ctor(0, 1, 0);
} else {
 x_55 = x_52;
}
lean_ctor_set(x_55, 0, x_54);
return x_55;
}
}
else
{
lean_object* x_69; lean_object* x_70; lean_object* x_71; 
lean_dec_ref(x_1);
x_69 = lean_ctor_get(x_50, 0);
lean_inc(x_69);
if (lean_is_exclusive(x_50)) {
 lean_ctor_release(x_50, 0);
 x_70 = x_50;
} else {
 lean_dec_ref(x_50);
 x_70 = lean_box(0);
}
if (lean_is_scalar(x_70)) {
 x_71 = lean_alloc_ctor(1, 1, 0);
} else {
 x_71 = x_70;
}
lean_ctor_set(x_71, 0, x_69);
return x_71;
}
}
}
}
else
{
uint8_t x_72; 
lean_dec_ref(x_1);
x_72 = !lean_is_exclusive(x_5);
if (x_72 == 0)
{
return x_5;
}
else
{
lean_object* x_73; lean_object* x_74; 
x_73 = lean_ctor_get(x_5, 0);
lean_inc(x_73);
lean_dec(x_5);
x_74 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_74, 0, x_73);
return x_74;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_checkRootUnprovable___redArg(x_3, x_4, x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_checkRootUnprovable(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_checkRootUnprovable___redArg(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; uint8_t x_12; 
x_11 = l_Lean_Meta_instMonadMCtxMetaM;
x_12 = !lean_is_exclusive(x_11);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_13 = lean_ctor_get(x_11, 0);
x_14 = lean_ctor_get(x_11, 1);
x_15 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_16 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_17 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_17, 0, x_14);
lean_closure_set(x_17, 1, x_16);
x_18 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, lean_box(0));
lean_closure_set(x_18, 2, lean_box(0));
lean_closure_set(x_18, 3, lean_box(0));
lean_closure_set(x_18, 4, x_13);
x_19 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_19, 0, x_17);
lean_closure_set(x_19, 1, x_16);
x_20 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, lean_box(0));
lean_closure_set(x_20, 2, lean_box(0));
lean_closure_set(x_20, 3, lean_box(0));
lean_closure_set(x_20, 4, x_18);
x_21 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_21, 0, x_19);
lean_closure_set(x_21, 1, x_15);
x_22 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_22, 0, lean_box(0));
lean_closure_set(x_22, 1, x_20);
x_23 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
x_24 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
x_25 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_25, 0, x_21);
lean_closure_set(x_25, 1, x_24);
x_26 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 2);
lean_closure_set(x_26, 0, lean_box(0));
lean_closure_set(x_26, 1, x_22);
lean_ctor_set(x_11, 1, x_25);
lean_ctor_set(x_11, 0, x_26);
x_27 = lean_st_ref_get(x_3);
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
lean_dec(x_27);
x_29 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_29);
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_28);
lean_ctor_set(x_30, 1, x_29);
x_31 = lp_aesop_Aesop_getRootMVarId(x_30, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_30);
if (lean_obj_tag(x_31) == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_32 = lean_ctor_get(x_31, 0);
lean_inc(x_32);
lean_dec_ref(x_31);
x_33 = l_Lean_getExprMVarAssignment_x3f___redArg(x_23, x_11, x_32);
x_34 = lean_apply_9(x_33, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_34;
}
else
{
uint8_t x_35; 
lean_dec_ref(x_11);
lean_dec_ref(x_23);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_35 = !lean_is_exclusive(x_31);
if (x_35 == 0)
{
return x_31;
}
else
{
lean_object* x_36; lean_object* x_37; 
x_36 = lean_ctor_get(x_31, 0);
lean_inc(x_36);
lean_dec(x_31);
x_37 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_37, 0, x_36);
return x_37;
}
}
}
else
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; 
x_38 = lean_ctor_get(x_11, 0);
x_39 = lean_ctor_get(x_11, 1);
lean_inc(x_39);
lean_inc(x_38);
lean_dec(x_11);
x_40 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_41 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_42 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_42, 0, x_39);
lean_closure_set(x_42, 1, x_41);
x_43 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_43, 0, lean_box(0));
lean_closure_set(x_43, 1, lean_box(0));
lean_closure_set(x_43, 2, lean_box(0));
lean_closure_set(x_43, 3, lean_box(0));
lean_closure_set(x_43, 4, x_38);
x_44 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_44, 0, x_42);
lean_closure_set(x_44, 1, x_41);
x_45 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_45, 0, lean_box(0));
lean_closure_set(x_45, 1, lean_box(0));
lean_closure_set(x_45, 2, lean_box(0));
lean_closure_set(x_45, 3, lean_box(0));
lean_closure_set(x_45, 4, x_43);
x_46 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_46, 0, x_44);
lean_closure_set(x_46, 1, x_40);
x_47 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_47, 0, lean_box(0));
lean_closure_set(x_47, 1, x_45);
x_48 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
x_49 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
x_50 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_50, 0, x_46);
lean_closure_set(x_50, 1, x_49);
x_51 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 2);
lean_closure_set(x_51, 0, lean_box(0));
lean_closure_set(x_51, 1, x_47);
x_52 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_52, 0, x_51);
lean_ctor_set(x_52, 1, x_50);
x_53 = lean_st_ref_get(x_3);
x_54 = lean_ctor_get(x_53, 0);
lean_inc(x_54);
lean_dec(x_53);
x_55 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_55);
x_56 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_56, 0, x_54);
lean_ctor_set(x_56, 1, x_55);
x_57 = lp_aesop_Aesop_getRootMVarId(x_56, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_56);
if (lean_obj_tag(x_57) == 0)
{
lean_object* x_58; lean_object* x_59; lean_object* x_60; 
x_58 = lean_ctor_get(x_57, 0);
lean_inc(x_58);
lean_dec_ref(x_57);
x_59 = l_Lean_getExprMVarAssignment_x3f___redArg(x_48, x_52, x_58);
x_60 = lean_apply_9(x_59, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_60;
}
else
{
lean_object* x_61; lean_object* x_62; lean_object* x_63; 
lean_dec_ref(x_52);
lean_dec_ref(x_48);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_61 = lean_ctor_get(x_57, 0);
lean_inc(x_61);
if (lean_is_exclusive(x_57)) {
 lean_ctor_release(x_57, 0);
 x_62 = x_57;
} else {
 lean_dec_ref(x_57);
 x_62 = lean_box(0);
}
if (lean_is_scalar(x_62)) {
 x_63 = lean_alloc_ctor(1, 1, 0);
} else {
 x_63 = x_62;
}
lean_ctor_set(x_63, 0, x_61);
return x_63;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_getProof_x3f___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_getProof_x3f(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_getProof_x3f___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_1);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__1(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_instMonadControlReaderT(lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_st_ref_get(x_2);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec(x_10);
x_12 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_12);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
x_14 = lp_aesop_Aesop_extractProof(x_13, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_finalizeProof___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_finalizeProof___redArg___lam__1(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = l_Lean_Core_instMonadWithOptionsCoreM;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_3 = l_Lean_instMonadWithOptionsOfMonadFunctor___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___closed__1;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_3 = l_Lean_instMonadWithOptionsOfMonadFunctor___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___closed__2;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_3 = l_Lean_instMonadWithOptionsOfMonadFunctor___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___closed__3;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_3 = l_Lean_instMonadWithOptionsOfMonadFunctor___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___closed__4;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_3 = l_Lean_instMonadWithOptionsOfMonadFunctor___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___closed__5;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_3 = l_Lean_instMonadWithOptionsOfMonadFunctor___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_TraceOption_proof;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Final proof:", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, uint8_t x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18) {
_start:
{
if (x_10 == 0)
{
lean_object* x_20; lean_object* x_21; 
lean_dec(x_18);
lean_dec_ref(x_17);
lean_dec(x_16);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_20 = lean_box(0);
x_21 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; uint8_t x_27; 
x_22 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1;
x_23 = l_Lean_instMonadTraceOfMonadLift___redArg(x_1, x_22);
x_24 = l_Lean_instMonadTraceOfMonadLift___redArg(x_2, x_23);
x_25 = l_Lean_instMonadTraceOfMonadLift___redArg(x_3, x_24);
x_26 = l_Lean_instMonadTraceOfMonadLift___redArg(x_4, x_25);
x_27 = !lean_is_exclusive(x_5);
if (x_27 == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_28 = lean_ctor_get(x_5, 0);
x_29 = lean_ctor_get(x_5, 1);
lean_dec(x_29);
x_30 = lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1;
x_31 = l_Lean_indentExpr(x_6);
lean_ctor_set_tag(x_5, 7);
lean_ctor_set(x_5, 1, x_31);
lean_ctor_set(x_5, 0, x_30);
x_32 = l_Lean_addTrace___redArg(x_7, x_26, x_8, x_9, x_28, x_5);
x_33 = lean_apply_9(x_32, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, lean_box(0));
return x_33;
}
else
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_34 = lean_ctor_get(x_5, 0);
lean_inc(x_34);
lean_dec(x_5);
x_35 = lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1;
x_36 = l_Lean_indentExpr(x_6);
x_37 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_37, 0, x_35);
lean_ctor_set(x_37, 1, x_36);
x_38 = l_Lean_addTrace___redArg(x_7, x_26, x_8, x_9, x_34, x_37);
x_39 = lean_apply_9(x_38, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, lean_box(0));
return x_39;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
lean_object* x_19 = _args[18];
_start:
{
uint8_t x_20; lean_object* x_21; 
x_20 = lean_unbox(x_10);
x_21 = lp_aesop_Aesop_finalizeProof___redArg___lam__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_20, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18);
return x_21;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Proof: ", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("\nUnassigned metavariables: ", 27, 27);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__6), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__5___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__4___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__3), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__2___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__1___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__0), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__6;
x_2 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__5;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__10;
x_2 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__9;
x_3 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__8;
x_4 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__7;
x_5 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__12;
x_6 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_3);
lean_ctor_set(x_6, 3, x_2);
lean_ctor_set(x_6, 4, x_1);
return x_6;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__11;
x_2 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__13;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_MessageData_ofName), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop: internal error: extracted proof has metavariables.", 57, 57);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop: internal error: root goal is proven but its metavariable is not assigned", 79, 79);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23) {
_start:
{
lean_object* x_25; 
lean_inc(x_23);
lean_inc_ref(x_22);
lean_inc(x_21);
lean_inc_ref(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc_ref(x_16);
x_25 = lp_aesop_Aesop_getProof_x3f___redArg(x_1, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23);
if (lean_obj_tag(x_25) == 0)
{
lean_object* x_26; 
x_26 = lean_ctor_get(x_25, 0);
lean_inc(x_26);
lean_dec_ref(x_25);
if (lean_obj_tag(x_26) == 1)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_44; lean_object* x_45; 
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
lean_dec_ref(x_26);
lean_inc(x_27);
lean_inc_ref(x_2);
x_44 = l_Lean_instantiateMVars___redArg(x_2, x_3, x_27);
lean_inc(x_23);
lean_inc_ref(x_22);
lean_inc(x_21);
lean_inc_ref(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc_ref(x_16);
x_45 = lean_apply_9(x_44, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, lean_box(0));
if (lean_obj_tag(x_45) == 0)
{
lean_object* x_46; uint8_t x_47; 
x_46 = lean_ctor_get(x_45, 0);
lean_inc(x_46);
lean_dec_ref(x_45);
x_47 = l_Lean_Expr_hasExprMVar(x_46);
lean_dec(x_46);
if (x_47 == 0)
{
lean_dec_ref(x_14);
lean_dec_ref(x_13);
x_28 = x_16;
x_29 = x_17;
x_30 = x_18;
x_31 = x_19;
x_32 = x_20;
x_33 = x_21;
x_34 = x_22;
x_35 = x_23;
x_36 = lean_box(0);
goto block_43;
}
else
{
lean_object* x_48; lean_object* x_49; 
x_48 = lean_st_ref_get(x_17);
lean_dec(x_48);
lean_inc(x_27);
x_49 = l_Lean_Meta_getMVarsNoDelayed(x_27, x_20, x_21, x_22, x_23);
if (lean_obj_tag(x_49) == 0)
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; size_t x_57; size_t x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; 
x_50 = lean_ctor_get(x_49, 0);
lean_inc(x_50);
lean_dec_ref(x_49);
x_51 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__2;
lean_inc(x_27);
x_52 = l_Lean_MessageData_ofExpr(x_27);
x_53 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_53, 0, x_51);
lean_ctor_set(x_53, 1, x_52);
x_54 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__4;
x_55 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_55, 0, x_53);
lean_ctor_set(x_55, 1, x_54);
x_56 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__14;
x_57 = lean_array_size(x_50);
x_58 = 0;
x_59 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_56, x_13, x_57, x_58, x_50);
x_60 = lean_array_to_list(x_59);
x_61 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__15;
x_62 = lean_box(0);
x_63 = l_List_mapTR_loop___redArg(x_61, x_60, x_62);
x_64 = l_Lean_MessageData_ofList(x_63);
x_65 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_65, 0, x_55);
lean_ctor_set(x_65, 1, x_64);
x_66 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__17;
x_67 = l_Lean_indentD(x_65);
x_68 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_68, 0, x_66);
lean_ctor_set(x_68, 1, x_67);
lean_inc_ref(x_2);
x_69 = l_Lean_throwError___redArg(x_2, x_14, x_68);
lean_inc(x_23);
lean_inc_ref(x_22);
lean_inc(x_21);
lean_inc_ref(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc_ref(x_16);
x_70 = lean_apply_9(x_69, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, lean_box(0));
if (lean_obj_tag(x_70) == 0)
{
lean_dec_ref(x_70);
x_28 = x_16;
x_29 = x_17;
x_30 = x_18;
x_31 = x_19;
x_32 = x_20;
x_33 = x_21;
x_34 = x_22;
x_35 = x_23;
x_36 = lean_box(0);
goto block_43;
}
else
{
lean_dec(x_27);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec_ref(x_20);
lean_dec(x_19);
lean_dec(x_18);
lean_dec(x_17);
lean_dec_ref(x_16);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
return x_70;
}
}
else
{
uint8_t x_71; 
lean_dec(x_27);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec_ref(x_20);
lean_dec(x_19);
lean_dec(x_18);
lean_dec(x_17);
lean_dec_ref(x_16);
lean_dec_ref(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
x_71 = !lean_is_exclusive(x_49);
if (x_71 == 0)
{
return x_49;
}
else
{
lean_object* x_72; lean_object* x_73; 
x_72 = lean_ctor_get(x_49, 0);
lean_inc(x_72);
lean_dec(x_49);
x_73 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_73, 0, x_72);
return x_73;
}
}
}
}
else
{
uint8_t x_74; 
lean_dec(x_27);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec_ref(x_20);
lean_dec(x_19);
lean_dec(x_18);
lean_dec(x_17);
lean_dec_ref(x_16);
lean_dec_ref(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
x_74 = !lean_is_exclusive(x_45);
if (x_74 == 0)
{
return x_45;
}
else
{
lean_object* x_75; lean_object* x_76; 
x_75 = lean_ctor_get(x_45, 0);
lean_inc(x_75);
lean_dec(x_45);
x_76 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_76, 0, x_75);
return x_76;
}
}
block_43:
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_37 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__0;
lean_inc_ref(x_2);
x_38 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__2___boxed), 19, 9);
lean_closure_set(x_38, 0, x_4);
lean_closure_set(x_38, 1, x_5);
lean_closure_set(x_38, 2, x_6);
lean_closure_set(x_38, 3, x_7);
lean_closure_set(x_38, 4, x_37);
lean_closure_set(x_38, 5, x_27);
lean_closure_set(x_38, 6, x_2);
lean_closure_set(x_38, 7, x_8);
lean_closure_set(x_38, 8, x_9);
x_39 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_2, x_10, x_37);
x_40 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_40, 0, lean_box(0));
lean_closure_set(x_40, 1, lean_box(0));
lean_closure_set(x_40, 2, x_11);
lean_closure_set(x_40, 3, lean_box(0));
lean_closure_set(x_40, 4, lean_box(0));
lean_closure_set(x_40, 5, x_39);
lean_closure_set(x_40, 6, x_38);
x_41 = lp_aesop_Aesop_withPPAnalyze___redArg(x_12, x_40);
x_42 = lean_apply_9(x_41, x_28, x_29, x_30, x_31, x_32, x_33, x_34, x_35, lean_box(0));
return x_42;
}
}
else
{
lean_object* x_77; lean_object* x_78; lean_object* x_79; 
lean_dec(x_26);
lean_dec_ref(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_77 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__19;
x_78 = l_Lean_throwError___redArg(x_2, x_14, x_77);
x_79 = lean_apply_9(x_78, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, lean_box(0));
return x_79;
}
}
else
{
uint8_t x_80; 
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec_ref(x_20);
lean_dec(x_19);
lean_dec(x_18);
lean_dec(x_17);
lean_dec_ref(x_16);
lean_dec_ref(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_80 = !lean_is_exclusive(x_25);
if (x_80 == 0)
{
return x_25;
}
else
{
lean_object* x_81; lean_object* x_82; 
x_81 = lean_ctor_get(x_25, 0);
lean_inc(x_81);
lean_dec(x_25);
x_82 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_82, 0, x_81);
return x_82;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
lean_object* x_19 = _args[18];
lean_object* x_20 = _args[19];
lean_object* x_21 = _args[20];
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
lean_object* x_24 = _args[23];
_start:
{
lean_object* x_25; 
x_25 = lp_aesop_Aesop_finalizeProof___redArg___lam__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23);
lean_dec_ref(x_1);
return x_25;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_11 = lp_aesop_Aesop_finalizeProof___redArg___closed__0;
x_12 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1;
x_13 = lean_ctor_get(x_12, 0);
lean_inc_ref(x_13);
x_14 = !lean_is_exclusive(x_13);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; uint8_t x_39; 
x_15 = lean_ctor_get(x_13, 0);
x_16 = lean_ctor_get(x_13, 2);
x_17 = lean_ctor_get(x_13, 3);
x_18 = lean_ctor_get(x_13, 4);
x_19 = lean_ctor_get(x_13, 1);
lean_dec(x_19);
x_20 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_21 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
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
lean_ctor_set(x_13, 4, x_25);
lean_ctor_set(x_13, 3, x_26);
lean_ctor_set(x_13, 2, x_27);
lean_ctor_set(x_13, 1, x_20);
lean_ctor_set(x_13, 0, x_24);
x_28 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_28, 0, x_13);
lean_ctor_set(x_28, 1, x_21);
x_29 = l_ReaderT_instMonad___redArg(x_28);
x_30 = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(x_30, 0, lean_box(0));
lean_closure_set(x_30, 1, lean_box(0));
lean_closure_set(x_30, 2, x_29);
x_31 = l_instMonadControlTOfPure___redArg(x_30);
lean_inc_ref(x_31);
x_32 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_32, 0, x_11);
lean_closure_set(x_32, 1, x_31);
x_33 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_33, 0, x_11);
lean_closure_set(x_33, 1, x_31);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_32);
lean_ctor_set(x_34, 1, x_33);
lean_inc_ref(x_34);
x_35 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_35, 0, x_11);
lean_closure_set(x_35, 1, x_34);
x_36 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_36, 0, x_11);
lean_closure_set(x_36, 1, x_34);
x_37 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_37, 0, x_35);
lean_ctor_set(x_37, 1, x_36);
x_38 = l_Lean_Meta_instMonadMCtxMetaM;
x_39 = !lean_is_exclusive(x_38);
if (x_39 == 0)
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; uint8_t x_43; 
x_40 = lean_ctor_get(x_38, 0);
x_41 = lean_ctor_get(x_38, 1);
x_42 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
x_43 = !lean_is_exclusive(x_12);
if (x_43 == 0)
{
lean_object* x_44; lean_object* x_45; uint8_t x_46; 
x_44 = lean_ctor_get(x_12, 0);
x_45 = lean_ctor_get(x_12, 1);
lean_dec(x_45);
x_46 = !lean_is_exclusive(x_44);
if (x_46 == 0)
{
lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; uint8_t x_59; 
x_47 = lean_ctor_get(x_44, 0);
x_48 = lean_ctor_get(x_44, 2);
x_49 = lean_ctor_get(x_44, 3);
x_50 = lean_ctor_get(x_44, 4);
x_51 = lean_ctor_get(x_44, 1);
lean_dec(x_51);
lean_inc_ref(x_47);
x_52 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_52, 0, x_47);
x_53 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_53, 0, x_47);
x_54 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_54, 0, x_52);
lean_ctor_set(x_54, 1, x_53);
x_55 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_55, 0, x_50);
x_56 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_56, 0, x_49);
x_57 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_57, 0, x_48);
lean_ctor_set(x_44, 4, x_55);
lean_ctor_set(x_44, 3, x_56);
lean_ctor_set(x_44, 2, x_57);
lean_ctor_set(x_44, 1, x_20);
lean_ctor_set(x_44, 0, x_54);
lean_ctor_set(x_12, 1, x_21);
x_58 = l_ReaderT_instMonad___redArg(x_12);
x_59 = !lean_is_exclusive(x_58);
if (x_59 == 0)
{
lean_object* x_60; lean_object* x_61; uint8_t x_62; 
x_60 = lean_ctor_get(x_58, 0);
x_61 = lean_ctor_get(x_58, 1);
lean_dec(x_61);
x_62 = !lean_is_exclusive(x_60);
if (x_62 == 0)
{
lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; 
x_63 = lean_ctor_get(x_60, 0);
x_64 = lean_ctor_get(x_60, 2);
x_65 = lean_ctor_get(x_60, 3);
x_66 = lean_ctor_get(x_60, 4);
x_67 = lean_ctor_get(x_60, 1);
lean_dec(x_67);
x_68 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_69 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_70 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_70, 0, x_41);
lean_closure_set(x_70, 1, x_69);
x_71 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_71, 0, lean_box(0));
lean_closure_set(x_71, 1, lean_box(0));
lean_closure_set(x_71, 2, lean_box(0));
lean_closure_set(x_71, 3, lean_box(0));
lean_closure_set(x_71, 4, x_40);
x_72 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_72, 0, x_70);
lean_closure_set(x_72, 1, x_69);
x_73 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_73, 0, lean_box(0));
lean_closure_set(x_73, 1, lean_box(0));
lean_closure_set(x_73, 2, lean_box(0));
lean_closure_set(x_73, 3, lean_box(0));
lean_closure_set(x_73, 4, x_71);
x_74 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_74, 0, x_72);
lean_closure_set(x_74, 1, x_68);
x_75 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_75, 0, lean_box(0));
lean_closure_set(x_75, 1, x_73);
x_76 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_77 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_63);
x_78 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_78, 0, x_63);
x_79 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_79, 0, x_63);
x_80 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_80, 0, x_78);
lean_ctor_set(x_80, 1, x_79);
x_81 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_81, 0, x_66);
x_82 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_82, 0, x_65);
x_83 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_83, 0, x_64);
lean_ctor_set(x_60, 4, x_81);
lean_ctor_set(x_60, 3, x_82);
lean_ctor_set(x_60, 2, x_83);
lean_ctor_set(x_60, 1, x_76);
lean_ctor_set(x_60, 0, x_80);
lean_ctor_set(x_58, 1, x_77);
x_84 = l_ReaderT_instMonad___redArg(x_58);
x_85 = l_ReaderT_instMonad___redArg(x_84);
x_86 = l_ReaderT_instMonad___redArg(x_85);
x_87 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
lean_inc_ref(x_37);
x_88 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_88, 0, x_11);
lean_closure_set(x_88, 1, x_37);
x_89 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_89, 0, x_11);
lean_closure_set(x_89, 1, x_37);
x_90 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_90, 0, x_88);
lean_ctor_set(x_90, 1, x_89);
lean_inc_ref(x_90);
x_91 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_91, 0, x_11);
lean_closure_set(x_91, 1, x_90);
x_92 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_92, 0, x_11);
lean_closure_set(x_92, 1, x_90);
x_93 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_93, 0, x_91);
lean_ctor_set(x_93, 1, x_92);
x_94 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_94, 0, x_74);
lean_closure_set(x_94, 1, x_87);
x_95 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 2);
lean_closure_set(x_95, 0, lean_box(0));
lean_closure_set(x_95, 1, x_75);
lean_ctor_set(x_38, 1, x_94);
lean_ctor_set(x_38, 0, x_95);
x_96 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_97 = lean_ctor_get(x_96, 0);
lean_inc(x_97);
x_98 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_99 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_100 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_42);
x_101 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_100, x_42);
lean_inc_ref(x_99);
x_102 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_102, 0, x_98);
lean_ctor_set(x_102, 1, x_99);
lean_ctor_set(x_102, 2, x_101);
x_103 = lean_st_ref_get(x_3);
x_104 = lean_ctor_get(x_103, 0);
lean_inc(x_104);
lean_dec(x_103);
x_105 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_105);
x_106 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_106, 0, x_104);
lean_ctor_set(x_106, 1, x_105);
x_107 = lp_aesop_Aesop_getRootMVarId(x_106, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_106);
if (lean_obj_tag(x_107) == 0)
{
lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; 
x_108 = lean_ctor_get(x_107, 0);
lean_inc(x_108);
lean_dec_ref(x_107);
x_109 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed), 9, 0);
x_110 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed), 1, 0);
x_111 = lp_aesop_Aesop_finalizeProof___redArg___closed__6;
lean_inc_ref(x_86);
lean_inc_ref(x_42);
x_112 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed), 24, 14);
lean_closure_set(x_112, 0, x_1);
lean_closure_set(x_112, 1, x_42);
lean_closure_set(x_112, 2, x_38);
lean_closure_set(x_112, 3, x_69);
lean_closure_set(x_112, 4, x_69);
lean_closure_set(x_112, 5, x_68);
lean_closure_set(x_112, 6, x_87);
lean_closure_set(x_112, 7, x_99);
lean_closure_set(x_112, 8, x_100);
lean_closure_set(x_112, 9, x_97);
lean_closure_set(x_112, 10, x_86);
lean_closure_set(x_112, 11, x_111);
lean_closure_set(x_112, 12, x_110);
lean_closure_set(x_112, 13, x_102);
x_113 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_113, 0, lean_box(0));
lean_closure_set(x_113, 1, lean_box(0));
lean_closure_set(x_113, 2, x_86);
lean_closure_set(x_113, 3, lean_box(0));
lean_closure_set(x_113, 4, lean_box(0));
lean_closure_set(x_113, 5, x_109);
lean_closure_set(x_113, 6, x_112);
x_114 = l_Lean_MVarId_withContext___redArg(x_93, x_42, x_108, x_113);
x_115 = lean_apply_9(x_114, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_115;
}
else
{
uint8_t x_116; 
lean_dec_ref(x_102);
lean_dec_ref(x_99);
lean_dec(x_97);
lean_dec_ref(x_38);
lean_dec_ref(x_93);
lean_dec_ref(x_86);
lean_dec_ref(x_42);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_116 = !lean_is_exclusive(x_107);
if (x_116 == 0)
{
return x_107;
}
else
{
lean_object* x_117; lean_object* x_118; 
x_117 = lean_ctor_get(x_107, 0);
lean_inc(x_117);
lean_dec(x_107);
x_118 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_118, 0, x_117);
return x_118;
}
}
}
else
{
lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; 
x_119 = lean_ctor_get(x_60, 0);
x_120 = lean_ctor_get(x_60, 2);
x_121 = lean_ctor_get(x_60, 3);
x_122 = lean_ctor_get(x_60, 4);
lean_inc(x_122);
lean_inc(x_121);
lean_inc(x_120);
lean_inc(x_119);
lean_dec(x_60);
x_123 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_124 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_125 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_125, 0, x_41);
lean_closure_set(x_125, 1, x_124);
x_126 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_126, 0, lean_box(0));
lean_closure_set(x_126, 1, lean_box(0));
lean_closure_set(x_126, 2, lean_box(0));
lean_closure_set(x_126, 3, lean_box(0));
lean_closure_set(x_126, 4, x_40);
x_127 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_127, 0, x_125);
lean_closure_set(x_127, 1, x_124);
x_128 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_128, 0, lean_box(0));
lean_closure_set(x_128, 1, lean_box(0));
lean_closure_set(x_128, 2, lean_box(0));
lean_closure_set(x_128, 3, lean_box(0));
lean_closure_set(x_128, 4, x_126);
x_129 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_129, 0, x_127);
lean_closure_set(x_129, 1, x_123);
x_130 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_130, 0, lean_box(0));
lean_closure_set(x_130, 1, x_128);
x_131 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_132 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_119);
x_133 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_133, 0, x_119);
x_134 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_134, 0, x_119);
x_135 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_135, 0, x_133);
lean_ctor_set(x_135, 1, x_134);
x_136 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_136, 0, x_122);
x_137 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_137, 0, x_121);
x_138 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_138, 0, x_120);
x_139 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_139, 0, x_135);
lean_ctor_set(x_139, 1, x_131);
lean_ctor_set(x_139, 2, x_138);
lean_ctor_set(x_139, 3, x_137);
lean_ctor_set(x_139, 4, x_136);
lean_ctor_set(x_58, 1, x_132);
lean_ctor_set(x_58, 0, x_139);
x_140 = l_ReaderT_instMonad___redArg(x_58);
x_141 = l_ReaderT_instMonad___redArg(x_140);
x_142 = l_ReaderT_instMonad___redArg(x_141);
x_143 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
lean_inc_ref(x_37);
x_144 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_144, 0, x_11);
lean_closure_set(x_144, 1, x_37);
x_145 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_145, 0, x_11);
lean_closure_set(x_145, 1, x_37);
x_146 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_146, 0, x_144);
lean_ctor_set(x_146, 1, x_145);
lean_inc_ref(x_146);
x_147 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_147, 0, x_11);
lean_closure_set(x_147, 1, x_146);
x_148 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_148, 0, x_11);
lean_closure_set(x_148, 1, x_146);
x_149 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_149, 0, x_147);
lean_ctor_set(x_149, 1, x_148);
x_150 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_150, 0, x_129);
lean_closure_set(x_150, 1, x_143);
x_151 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 2);
lean_closure_set(x_151, 0, lean_box(0));
lean_closure_set(x_151, 1, x_130);
lean_ctor_set(x_38, 1, x_150);
lean_ctor_set(x_38, 0, x_151);
x_152 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_153 = lean_ctor_get(x_152, 0);
lean_inc(x_153);
x_154 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_155 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_156 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_42);
x_157 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_156, x_42);
lean_inc_ref(x_155);
x_158 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_158, 0, x_154);
lean_ctor_set(x_158, 1, x_155);
lean_ctor_set(x_158, 2, x_157);
x_159 = lean_st_ref_get(x_3);
x_160 = lean_ctor_get(x_159, 0);
lean_inc(x_160);
lean_dec(x_159);
x_161 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_161);
x_162 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_162, 0, x_160);
lean_ctor_set(x_162, 1, x_161);
x_163 = lp_aesop_Aesop_getRootMVarId(x_162, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_162);
if (lean_obj_tag(x_163) == 0)
{
lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; 
x_164 = lean_ctor_get(x_163, 0);
lean_inc(x_164);
lean_dec_ref(x_163);
x_165 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed), 9, 0);
x_166 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed), 1, 0);
x_167 = lp_aesop_Aesop_finalizeProof___redArg___closed__6;
lean_inc_ref(x_142);
lean_inc_ref(x_42);
x_168 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed), 24, 14);
lean_closure_set(x_168, 0, x_1);
lean_closure_set(x_168, 1, x_42);
lean_closure_set(x_168, 2, x_38);
lean_closure_set(x_168, 3, x_124);
lean_closure_set(x_168, 4, x_124);
lean_closure_set(x_168, 5, x_123);
lean_closure_set(x_168, 6, x_143);
lean_closure_set(x_168, 7, x_155);
lean_closure_set(x_168, 8, x_156);
lean_closure_set(x_168, 9, x_153);
lean_closure_set(x_168, 10, x_142);
lean_closure_set(x_168, 11, x_167);
lean_closure_set(x_168, 12, x_166);
lean_closure_set(x_168, 13, x_158);
x_169 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_169, 0, lean_box(0));
lean_closure_set(x_169, 1, lean_box(0));
lean_closure_set(x_169, 2, x_142);
lean_closure_set(x_169, 3, lean_box(0));
lean_closure_set(x_169, 4, lean_box(0));
lean_closure_set(x_169, 5, x_165);
lean_closure_set(x_169, 6, x_168);
x_170 = l_Lean_MVarId_withContext___redArg(x_149, x_42, x_164, x_169);
x_171 = lean_apply_9(x_170, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_171;
}
else
{
lean_object* x_172; lean_object* x_173; lean_object* x_174; 
lean_dec_ref(x_158);
lean_dec_ref(x_155);
lean_dec(x_153);
lean_dec_ref(x_38);
lean_dec_ref(x_149);
lean_dec_ref(x_142);
lean_dec_ref(x_42);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_172 = lean_ctor_get(x_163, 0);
lean_inc(x_172);
if (lean_is_exclusive(x_163)) {
 lean_ctor_release(x_163, 0);
 x_173 = x_163;
} else {
 lean_dec_ref(x_163);
 x_173 = lean_box(0);
}
if (lean_is_scalar(x_173)) {
 x_174 = lean_alloc_ctor(1, 1, 0);
} else {
 x_174 = x_173;
}
lean_ctor_set(x_174, 0, x_172);
return x_174;
}
}
}
else
{
lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; 
x_175 = lean_ctor_get(x_58, 0);
lean_inc(x_175);
lean_dec(x_58);
x_176 = lean_ctor_get(x_175, 0);
lean_inc_ref(x_176);
x_177 = lean_ctor_get(x_175, 2);
lean_inc(x_177);
x_178 = lean_ctor_get(x_175, 3);
lean_inc(x_178);
x_179 = lean_ctor_get(x_175, 4);
lean_inc(x_179);
if (lean_is_exclusive(x_175)) {
 lean_ctor_release(x_175, 0);
 lean_ctor_release(x_175, 1);
 lean_ctor_release(x_175, 2);
 lean_ctor_release(x_175, 3);
 lean_ctor_release(x_175, 4);
 x_180 = x_175;
} else {
 lean_dec_ref(x_175);
 x_180 = lean_box(0);
}
x_181 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_182 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_183 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_183, 0, x_41);
lean_closure_set(x_183, 1, x_182);
x_184 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_184, 0, lean_box(0));
lean_closure_set(x_184, 1, lean_box(0));
lean_closure_set(x_184, 2, lean_box(0));
lean_closure_set(x_184, 3, lean_box(0));
lean_closure_set(x_184, 4, x_40);
x_185 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_185, 0, x_183);
lean_closure_set(x_185, 1, x_182);
x_186 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_186, 0, lean_box(0));
lean_closure_set(x_186, 1, lean_box(0));
lean_closure_set(x_186, 2, lean_box(0));
lean_closure_set(x_186, 3, lean_box(0));
lean_closure_set(x_186, 4, x_184);
x_187 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_187, 0, x_185);
lean_closure_set(x_187, 1, x_181);
x_188 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_188, 0, lean_box(0));
lean_closure_set(x_188, 1, x_186);
x_189 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_190 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_176);
x_191 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_191, 0, x_176);
x_192 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_192, 0, x_176);
x_193 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_193, 0, x_191);
lean_ctor_set(x_193, 1, x_192);
x_194 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_194, 0, x_179);
x_195 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_195, 0, x_178);
x_196 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_196, 0, x_177);
if (lean_is_scalar(x_180)) {
 x_197 = lean_alloc_ctor(0, 5, 0);
} else {
 x_197 = x_180;
}
lean_ctor_set(x_197, 0, x_193);
lean_ctor_set(x_197, 1, x_189);
lean_ctor_set(x_197, 2, x_196);
lean_ctor_set(x_197, 3, x_195);
lean_ctor_set(x_197, 4, x_194);
x_198 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_198, 0, x_197);
lean_ctor_set(x_198, 1, x_190);
x_199 = l_ReaderT_instMonad___redArg(x_198);
x_200 = l_ReaderT_instMonad___redArg(x_199);
x_201 = l_ReaderT_instMonad___redArg(x_200);
x_202 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
lean_inc_ref(x_37);
x_203 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_203, 0, x_11);
lean_closure_set(x_203, 1, x_37);
x_204 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_204, 0, x_11);
lean_closure_set(x_204, 1, x_37);
x_205 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_205, 0, x_203);
lean_ctor_set(x_205, 1, x_204);
lean_inc_ref(x_205);
x_206 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_206, 0, x_11);
lean_closure_set(x_206, 1, x_205);
x_207 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_207, 0, x_11);
lean_closure_set(x_207, 1, x_205);
x_208 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_208, 0, x_206);
lean_ctor_set(x_208, 1, x_207);
x_209 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_209, 0, x_187);
lean_closure_set(x_209, 1, x_202);
x_210 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 2);
lean_closure_set(x_210, 0, lean_box(0));
lean_closure_set(x_210, 1, x_188);
lean_ctor_set(x_38, 1, x_209);
lean_ctor_set(x_38, 0, x_210);
x_211 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_212 = lean_ctor_get(x_211, 0);
lean_inc(x_212);
x_213 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_214 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_215 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_42);
x_216 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_215, x_42);
lean_inc_ref(x_214);
x_217 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_217, 0, x_213);
lean_ctor_set(x_217, 1, x_214);
lean_ctor_set(x_217, 2, x_216);
x_218 = lean_st_ref_get(x_3);
x_219 = lean_ctor_get(x_218, 0);
lean_inc(x_219);
lean_dec(x_218);
x_220 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_220);
x_221 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_221, 0, x_219);
lean_ctor_set(x_221, 1, x_220);
x_222 = lp_aesop_Aesop_getRootMVarId(x_221, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_221);
if (lean_obj_tag(x_222) == 0)
{
lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; 
x_223 = lean_ctor_get(x_222, 0);
lean_inc(x_223);
lean_dec_ref(x_222);
x_224 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed), 9, 0);
x_225 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed), 1, 0);
x_226 = lp_aesop_Aesop_finalizeProof___redArg___closed__6;
lean_inc_ref(x_201);
lean_inc_ref(x_42);
x_227 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed), 24, 14);
lean_closure_set(x_227, 0, x_1);
lean_closure_set(x_227, 1, x_42);
lean_closure_set(x_227, 2, x_38);
lean_closure_set(x_227, 3, x_182);
lean_closure_set(x_227, 4, x_182);
lean_closure_set(x_227, 5, x_181);
lean_closure_set(x_227, 6, x_202);
lean_closure_set(x_227, 7, x_214);
lean_closure_set(x_227, 8, x_215);
lean_closure_set(x_227, 9, x_212);
lean_closure_set(x_227, 10, x_201);
lean_closure_set(x_227, 11, x_226);
lean_closure_set(x_227, 12, x_225);
lean_closure_set(x_227, 13, x_217);
x_228 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_228, 0, lean_box(0));
lean_closure_set(x_228, 1, lean_box(0));
lean_closure_set(x_228, 2, x_201);
lean_closure_set(x_228, 3, lean_box(0));
lean_closure_set(x_228, 4, lean_box(0));
lean_closure_set(x_228, 5, x_224);
lean_closure_set(x_228, 6, x_227);
x_229 = l_Lean_MVarId_withContext___redArg(x_208, x_42, x_223, x_228);
x_230 = lean_apply_9(x_229, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_230;
}
else
{
lean_object* x_231; lean_object* x_232; lean_object* x_233; 
lean_dec_ref(x_217);
lean_dec_ref(x_214);
lean_dec(x_212);
lean_dec_ref(x_38);
lean_dec_ref(x_208);
lean_dec_ref(x_201);
lean_dec_ref(x_42);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_231 = lean_ctor_get(x_222, 0);
lean_inc(x_231);
if (lean_is_exclusive(x_222)) {
 lean_ctor_release(x_222, 0);
 x_232 = x_222;
} else {
 lean_dec_ref(x_222);
 x_232 = lean_box(0);
}
if (lean_is_scalar(x_232)) {
 x_233 = lean_alloc_ctor(1, 1, 0);
} else {
 x_233 = x_232;
}
lean_ctor_set(x_233, 0, x_231);
return x_233;
}
}
}
else
{
lean_object* x_234; lean_object* x_235; lean_object* x_236; lean_object* x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; lean_object* x_241; lean_object* x_242; lean_object* x_243; lean_object* x_244; lean_object* x_245; lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; lean_object* x_259; lean_object* x_260; lean_object* x_261; lean_object* x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; lean_object* x_269; lean_object* x_270; lean_object* x_271; lean_object* x_272; lean_object* x_273; lean_object* x_274; lean_object* x_275; lean_object* x_276; lean_object* x_277; lean_object* x_278; lean_object* x_279; lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; lean_object* x_291; lean_object* x_292; lean_object* x_293; lean_object* x_294; 
x_234 = lean_ctor_get(x_44, 0);
x_235 = lean_ctor_get(x_44, 2);
x_236 = lean_ctor_get(x_44, 3);
x_237 = lean_ctor_get(x_44, 4);
lean_inc(x_237);
lean_inc(x_236);
lean_inc(x_235);
lean_inc(x_234);
lean_dec(x_44);
lean_inc_ref(x_234);
x_238 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_238, 0, x_234);
x_239 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_239, 0, x_234);
x_240 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_240, 0, x_238);
lean_ctor_set(x_240, 1, x_239);
x_241 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_241, 0, x_237);
x_242 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_242, 0, x_236);
x_243 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_243, 0, x_235);
x_244 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_244, 0, x_240);
lean_ctor_set(x_244, 1, x_20);
lean_ctor_set(x_244, 2, x_243);
lean_ctor_set(x_244, 3, x_242);
lean_ctor_set(x_244, 4, x_241);
lean_ctor_set(x_12, 1, x_21);
lean_ctor_set(x_12, 0, x_244);
x_245 = l_ReaderT_instMonad___redArg(x_12);
x_246 = lean_ctor_get(x_245, 0);
lean_inc_ref(x_246);
if (lean_is_exclusive(x_245)) {
 lean_ctor_release(x_245, 0);
 lean_ctor_release(x_245, 1);
 x_247 = x_245;
} else {
 lean_dec_ref(x_245);
 x_247 = lean_box(0);
}
x_248 = lean_ctor_get(x_246, 0);
lean_inc_ref(x_248);
x_249 = lean_ctor_get(x_246, 2);
lean_inc(x_249);
x_250 = lean_ctor_get(x_246, 3);
lean_inc(x_250);
x_251 = lean_ctor_get(x_246, 4);
lean_inc(x_251);
if (lean_is_exclusive(x_246)) {
 lean_ctor_release(x_246, 0);
 lean_ctor_release(x_246, 1);
 lean_ctor_release(x_246, 2);
 lean_ctor_release(x_246, 3);
 lean_ctor_release(x_246, 4);
 x_252 = x_246;
} else {
 lean_dec_ref(x_246);
 x_252 = lean_box(0);
}
x_253 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_254 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_255 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_255, 0, x_41);
lean_closure_set(x_255, 1, x_254);
x_256 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_256, 0, lean_box(0));
lean_closure_set(x_256, 1, lean_box(0));
lean_closure_set(x_256, 2, lean_box(0));
lean_closure_set(x_256, 3, lean_box(0));
lean_closure_set(x_256, 4, x_40);
x_257 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_257, 0, x_255);
lean_closure_set(x_257, 1, x_254);
x_258 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_258, 0, lean_box(0));
lean_closure_set(x_258, 1, lean_box(0));
lean_closure_set(x_258, 2, lean_box(0));
lean_closure_set(x_258, 3, lean_box(0));
lean_closure_set(x_258, 4, x_256);
x_259 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_259, 0, x_257);
lean_closure_set(x_259, 1, x_253);
x_260 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_260, 0, lean_box(0));
lean_closure_set(x_260, 1, x_258);
x_261 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_262 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_248);
x_263 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_263, 0, x_248);
x_264 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_264, 0, x_248);
x_265 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_265, 0, x_263);
lean_ctor_set(x_265, 1, x_264);
x_266 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_266, 0, x_251);
x_267 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_267, 0, x_250);
x_268 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_268, 0, x_249);
if (lean_is_scalar(x_252)) {
 x_269 = lean_alloc_ctor(0, 5, 0);
} else {
 x_269 = x_252;
}
lean_ctor_set(x_269, 0, x_265);
lean_ctor_set(x_269, 1, x_261);
lean_ctor_set(x_269, 2, x_268);
lean_ctor_set(x_269, 3, x_267);
lean_ctor_set(x_269, 4, x_266);
if (lean_is_scalar(x_247)) {
 x_270 = lean_alloc_ctor(0, 2, 0);
} else {
 x_270 = x_247;
}
lean_ctor_set(x_270, 0, x_269);
lean_ctor_set(x_270, 1, x_262);
x_271 = l_ReaderT_instMonad___redArg(x_270);
x_272 = l_ReaderT_instMonad___redArg(x_271);
x_273 = l_ReaderT_instMonad___redArg(x_272);
x_274 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
lean_inc_ref(x_37);
x_275 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_275, 0, x_11);
lean_closure_set(x_275, 1, x_37);
x_276 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_276, 0, x_11);
lean_closure_set(x_276, 1, x_37);
x_277 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_277, 0, x_275);
lean_ctor_set(x_277, 1, x_276);
lean_inc_ref(x_277);
x_278 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_278, 0, x_11);
lean_closure_set(x_278, 1, x_277);
x_279 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_279, 0, x_11);
lean_closure_set(x_279, 1, x_277);
x_280 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_280, 0, x_278);
lean_ctor_set(x_280, 1, x_279);
x_281 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_281, 0, x_259);
lean_closure_set(x_281, 1, x_274);
x_282 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 2);
lean_closure_set(x_282, 0, lean_box(0));
lean_closure_set(x_282, 1, x_260);
lean_ctor_set(x_38, 1, x_281);
lean_ctor_set(x_38, 0, x_282);
x_283 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_284 = lean_ctor_get(x_283, 0);
lean_inc(x_284);
x_285 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_286 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_287 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_42);
x_288 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_287, x_42);
lean_inc_ref(x_286);
x_289 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_289, 0, x_285);
lean_ctor_set(x_289, 1, x_286);
lean_ctor_set(x_289, 2, x_288);
x_290 = lean_st_ref_get(x_3);
x_291 = lean_ctor_get(x_290, 0);
lean_inc(x_291);
lean_dec(x_290);
x_292 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_292);
x_293 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_293, 0, x_291);
lean_ctor_set(x_293, 1, x_292);
x_294 = lp_aesop_Aesop_getRootMVarId(x_293, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_293);
if (lean_obj_tag(x_294) == 0)
{
lean_object* x_295; lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; lean_object* x_301; lean_object* x_302; 
x_295 = lean_ctor_get(x_294, 0);
lean_inc(x_295);
lean_dec_ref(x_294);
x_296 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed), 9, 0);
x_297 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed), 1, 0);
x_298 = lp_aesop_Aesop_finalizeProof___redArg___closed__6;
lean_inc_ref(x_273);
lean_inc_ref(x_42);
x_299 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed), 24, 14);
lean_closure_set(x_299, 0, x_1);
lean_closure_set(x_299, 1, x_42);
lean_closure_set(x_299, 2, x_38);
lean_closure_set(x_299, 3, x_254);
lean_closure_set(x_299, 4, x_254);
lean_closure_set(x_299, 5, x_253);
lean_closure_set(x_299, 6, x_274);
lean_closure_set(x_299, 7, x_286);
lean_closure_set(x_299, 8, x_287);
lean_closure_set(x_299, 9, x_284);
lean_closure_set(x_299, 10, x_273);
lean_closure_set(x_299, 11, x_298);
lean_closure_set(x_299, 12, x_297);
lean_closure_set(x_299, 13, x_289);
x_300 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_300, 0, lean_box(0));
lean_closure_set(x_300, 1, lean_box(0));
lean_closure_set(x_300, 2, x_273);
lean_closure_set(x_300, 3, lean_box(0));
lean_closure_set(x_300, 4, lean_box(0));
lean_closure_set(x_300, 5, x_296);
lean_closure_set(x_300, 6, x_299);
x_301 = l_Lean_MVarId_withContext___redArg(x_280, x_42, x_295, x_300);
x_302 = lean_apply_9(x_301, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_302;
}
else
{
lean_object* x_303; lean_object* x_304; lean_object* x_305; 
lean_dec_ref(x_289);
lean_dec_ref(x_286);
lean_dec(x_284);
lean_dec_ref(x_38);
lean_dec_ref(x_280);
lean_dec_ref(x_273);
lean_dec_ref(x_42);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_303 = lean_ctor_get(x_294, 0);
lean_inc(x_303);
if (lean_is_exclusive(x_294)) {
 lean_ctor_release(x_294, 0);
 x_304 = x_294;
} else {
 lean_dec_ref(x_294);
 x_304 = lean_box(0);
}
if (lean_is_scalar(x_304)) {
 x_305 = lean_alloc_ctor(1, 1, 0);
} else {
 x_305 = x_304;
}
lean_ctor_set(x_305, 0, x_303);
return x_305;
}
}
}
else
{
lean_object* x_306; lean_object* x_307; lean_object* x_308; lean_object* x_309; lean_object* x_310; lean_object* x_311; lean_object* x_312; lean_object* x_313; lean_object* x_314; lean_object* x_315; lean_object* x_316; lean_object* x_317; lean_object* x_318; lean_object* x_319; lean_object* x_320; lean_object* x_321; lean_object* x_322; lean_object* x_323; lean_object* x_324; lean_object* x_325; lean_object* x_326; lean_object* x_327; lean_object* x_328; lean_object* x_329; lean_object* x_330; lean_object* x_331; lean_object* x_332; lean_object* x_333; lean_object* x_334; lean_object* x_335; lean_object* x_336; lean_object* x_337; lean_object* x_338; lean_object* x_339; lean_object* x_340; lean_object* x_341; lean_object* x_342; lean_object* x_343; lean_object* x_344; lean_object* x_345; lean_object* x_346; lean_object* x_347; lean_object* x_348; lean_object* x_349; lean_object* x_350; lean_object* x_351; lean_object* x_352; lean_object* x_353; lean_object* x_354; lean_object* x_355; lean_object* x_356; lean_object* x_357; lean_object* x_358; lean_object* x_359; lean_object* x_360; lean_object* x_361; lean_object* x_362; lean_object* x_363; lean_object* x_364; lean_object* x_365; lean_object* x_366; lean_object* x_367; lean_object* x_368; lean_object* x_369; 
x_306 = lean_ctor_get(x_12, 0);
lean_inc(x_306);
lean_dec(x_12);
x_307 = lean_ctor_get(x_306, 0);
lean_inc_ref(x_307);
x_308 = lean_ctor_get(x_306, 2);
lean_inc(x_308);
x_309 = lean_ctor_get(x_306, 3);
lean_inc(x_309);
x_310 = lean_ctor_get(x_306, 4);
lean_inc(x_310);
if (lean_is_exclusive(x_306)) {
 lean_ctor_release(x_306, 0);
 lean_ctor_release(x_306, 1);
 lean_ctor_release(x_306, 2);
 lean_ctor_release(x_306, 3);
 lean_ctor_release(x_306, 4);
 x_311 = x_306;
} else {
 lean_dec_ref(x_306);
 x_311 = lean_box(0);
}
lean_inc_ref(x_307);
x_312 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_312, 0, x_307);
x_313 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_313, 0, x_307);
x_314 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_314, 0, x_312);
lean_ctor_set(x_314, 1, x_313);
x_315 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_315, 0, x_310);
x_316 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_316, 0, x_309);
x_317 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_317, 0, x_308);
if (lean_is_scalar(x_311)) {
 x_318 = lean_alloc_ctor(0, 5, 0);
} else {
 x_318 = x_311;
}
lean_ctor_set(x_318, 0, x_314);
lean_ctor_set(x_318, 1, x_20);
lean_ctor_set(x_318, 2, x_317);
lean_ctor_set(x_318, 3, x_316);
lean_ctor_set(x_318, 4, x_315);
x_319 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_319, 0, x_318);
lean_ctor_set(x_319, 1, x_21);
x_320 = l_ReaderT_instMonad___redArg(x_319);
x_321 = lean_ctor_get(x_320, 0);
lean_inc_ref(x_321);
if (lean_is_exclusive(x_320)) {
 lean_ctor_release(x_320, 0);
 lean_ctor_release(x_320, 1);
 x_322 = x_320;
} else {
 lean_dec_ref(x_320);
 x_322 = lean_box(0);
}
x_323 = lean_ctor_get(x_321, 0);
lean_inc_ref(x_323);
x_324 = lean_ctor_get(x_321, 2);
lean_inc(x_324);
x_325 = lean_ctor_get(x_321, 3);
lean_inc(x_325);
x_326 = lean_ctor_get(x_321, 4);
lean_inc(x_326);
if (lean_is_exclusive(x_321)) {
 lean_ctor_release(x_321, 0);
 lean_ctor_release(x_321, 1);
 lean_ctor_release(x_321, 2);
 lean_ctor_release(x_321, 3);
 lean_ctor_release(x_321, 4);
 x_327 = x_321;
} else {
 lean_dec_ref(x_321);
 x_327 = lean_box(0);
}
x_328 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_329 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_330 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_330, 0, x_41);
lean_closure_set(x_330, 1, x_329);
x_331 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_331, 0, lean_box(0));
lean_closure_set(x_331, 1, lean_box(0));
lean_closure_set(x_331, 2, lean_box(0));
lean_closure_set(x_331, 3, lean_box(0));
lean_closure_set(x_331, 4, x_40);
x_332 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_332, 0, x_330);
lean_closure_set(x_332, 1, x_329);
x_333 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_333, 0, lean_box(0));
lean_closure_set(x_333, 1, lean_box(0));
lean_closure_set(x_333, 2, lean_box(0));
lean_closure_set(x_333, 3, lean_box(0));
lean_closure_set(x_333, 4, x_331);
x_334 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_334, 0, x_332);
lean_closure_set(x_334, 1, x_328);
x_335 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_335, 0, lean_box(0));
lean_closure_set(x_335, 1, x_333);
x_336 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_337 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_323);
x_338 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_338, 0, x_323);
x_339 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_339, 0, x_323);
x_340 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_340, 0, x_338);
lean_ctor_set(x_340, 1, x_339);
x_341 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_341, 0, x_326);
x_342 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_342, 0, x_325);
x_343 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_343, 0, x_324);
if (lean_is_scalar(x_327)) {
 x_344 = lean_alloc_ctor(0, 5, 0);
} else {
 x_344 = x_327;
}
lean_ctor_set(x_344, 0, x_340);
lean_ctor_set(x_344, 1, x_336);
lean_ctor_set(x_344, 2, x_343);
lean_ctor_set(x_344, 3, x_342);
lean_ctor_set(x_344, 4, x_341);
if (lean_is_scalar(x_322)) {
 x_345 = lean_alloc_ctor(0, 2, 0);
} else {
 x_345 = x_322;
}
lean_ctor_set(x_345, 0, x_344);
lean_ctor_set(x_345, 1, x_337);
x_346 = l_ReaderT_instMonad___redArg(x_345);
x_347 = l_ReaderT_instMonad___redArg(x_346);
x_348 = l_ReaderT_instMonad___redArg(x_347);
x_349 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
lean_inc_ref(x_37);
x_350 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_350, 0, x_11);
lean_closure_set(x_350, 1, x_37);
x_351 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_351, 0, x_11);
lean_closure_set(x_351, 1, x_37);
x_352 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_352, 0, x_350);
lean_ctor_set(x_352, 1, x_351);
lean_inc_ref(x_352);
x_353 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_353, 0, x_11);
lean_closure_set(x_353, 1, x_352);
x_354 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_354, 0, x_11);
lean_closure_set(x_354, 1, x_352);
x_355 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_355, 0, x_353);
lean_ctor_set(x_355, 1, x_354);
x_356 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_356, 0, x_334);
lean_closure_set(x_356, 1, x_349);
x_357 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 2);
lean_closure_set(x_357, 0, lean_box(0));
lean_closure_set(x_357, 1, x_335);
lean_ctor_set(x_38, 1, x_356);
lean_ctor_set(x_38, 0, x_357);
x_358 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_359 = lean_ctor_get(x_358, 0);
lean_inc(x_359);
x_360 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_361 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_362 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_42);
x_363 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_362, x_42);
lean_inc_ref(x_361);
x_364 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_364, 0, x_360);
lean_ctor_set(x_364, 1, x_361);
lean_ctor_set(x_364, 2, x_363);
x_365 = lean_st_ref_get(x_3);
x_366 = lean_ctor_get(x_365, 0);
lean_inc(x_366);
lean_dec(x_365);
x_367 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_367);
x_368 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_368, 0, x_366);
lean_ctor_set(x_368, 1, x_367);
x_369 = lp_aesop_Aesop_getRootMVarId(x_368, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_368);
if (lean_obj_tag(x_369) == 0)
{
lean_object* x_370; lean_object* x_371; lean_object* x_372; lean_object* x_373; lean_object* x_374; lean_object* x_375; lean_object* x_376; lean_object* x_377; 
x_370 = lean_ctor_get(x_369, 0);
lean_inc(x_370);
lean_dec_ref(x_369);
x_371 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed), 9, 0);
x_372 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed), 1, 0);
x_373 = lp_aesop_Aesop_finalizeProof___redArg___closed__6;
lean_inc_ref(x_348);
lean_inc_ref(x_42);
x_374 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed), 24, 14);
lean_closure_set(x_374, 0, x_1);
lean_closure_set(x_374, 1, x_42);
lean_closure_set(x_374, 2, x_38);
lean_closure_set(x_374, 3, x_329);
lean_closure_set(x_374, 4, x_329);
lean_closure_set(x_374, 5, x_328);
lean_closure_set(x_374, 6, x_349);
lean_closure_set(x_374, 7, x_361);
lean_closure_set(x_374, 8, x_362);
lean_closure_set(x_374, 9, x_359);
lean_closure_set(x_374, 10, x_348);
lean_closure_set(x_374, 11, x_373);
lean_closure_set(x_374, 12, x_372);
lean_closure_set(x_374, 13, x_364);
x_375 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_375, 0, lean_box(0));
lean_closure_set(x_375, 1, lean_box(0));
lean_closure_set(x_375, 2, x_348);
lean_closure_set(x_375, 3, lean_box(0));
lean_closure_set(x_375, 4, lean_box(0));
lean_closure_set(x_375, 5, x_371);
lean_closure_set(x_375, 6, x_374);
x_376 = l_Lean_MVarId_withContext___redArg(x_355, x_42, x_370, x_375);
x_377 = lean_apply_9(x_376, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_377;
}
else
{
lean_object* x_378; lean_object* x_379; lean_object* x_380; 
lean_dec_ref(x_364);
lean_dec_ref(x_361);
lean_dec(x_359);
lean_dec_ref(x_38);
lean_dec_ref(x_355);
lean_dec_ref(x_348);
lean_dec_ref(x_42);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_378 = lean_ctor_get(x_369, 0);
lean_inc(x_378);
if (lean_is_exclusive(x_369)) {
 lean_ctor_release(x_369, 0);
 x_379 = x_369;
} else {
 lean_dec_ref(x_369);
 x_379 = lean_box(0);
}
if (lean_is_scalar(x_379)) {
 x_380 = lean_alloc_ctor(1, 1, 0);
} else {
 x_380 = x_379;
}
lean_ctor_set(x_380, 0, x_378);
return x_380;
}
}
}
else
{
lean_object* x_381; lean_object* x_382; lean_object* x_383; lean_object* x_384; lean_object* x_385; lean_object* x_386; lean_object* x_387; lean_object* x_388; lean_object* x_389; lean_object* x_390; lean_object* x_391; lean_object* x_392; lean_object* x_393; lean_object* x_394; lean_object* x_395; lean_object* x_396; lean_object* x_397; lean_object* x_398; lean_object* x_399; lean_object* x_400; lean_object* x_401; lean_object* x_402; lean_object* x_403; lean_object* x_404; lean_object* x_405; lean_object* x_406; lean_object* x_407; lean_object* x_408; lean_object* x_409; lean_object* x_410; lean_object* x_411; lean_object* x_412; lean_object* x_413; lean_object* x_414; lean_object* x_415; lean_object* x_416; lean_object* x_417; lean_object* x_418; lean_object* x_419; lean_object* x_420; lean_object* x_421; lean_object* x_422; lean_object* x_423; lean_object* x_424; lean_object* x_425; lean_object* x_426; lean_object* x_427; lean_object* x_428; lean_object* x_429; lean_object* x_430; lean_object* x_431; lean_object* x_432; lean_object* x_433; lean_object* x_434; lean_object* x_435; lean_object* x_436; lean_object* x_437; lean_object* x_438; lean_object* x_439; lean_object* x_440; lean_object* x_441; lean_object* x_442; lean_object* x_443; lean_object* x_444; lean_object* x_445; lean_object* x_446; lean_object* x_447; lean_object* x_448; lean_object* x_449; 
x_381 = lean_ctor_get(x_38, 0);
x_382 = lean_ctor_get(x_38, 1);
lean_inc(x_382);
lean_inc(x_381);
lean_dec(x_38);
x_383 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
x_384 = lean_ctor_get(x_12, 0);
lean_inc_ref(x_384);
if (lean_is_exclusive(x_12)) {
 lean_ctor_release(x_12, 0);
 lean_ctor_release(x_12, 1);
 x_385 = x_12;
} else {
 lean_dec_ref(x_12);
 x_385 = lean_box(0);
}
x_386 = lean_ctor_get(x_384, 0);
lean_inc_ref(x_386);
x_387 = lean_ctor_get(x_384, 2);
lean_inc(x_387);
x_388 = lean_ctor_get(x_384, 3);
lean_inc(x_388);
x_389 = lean_ctor_get(x_384, 4);
lean_inc(x_389);
if (lean_is_exclusive(x_384)) {
 lean_ctor_release(x_384, 0);
 lean_ctor_release(x_384, 1);
 lean_ctor_release(x_384, 2);
 lean_ctor_release(x_384, 3);
 lean_ctor_release(x_384, 4);
 x_390 = x_384;
} else {
 lean_dec_ref(x_384);
 x_390 = lean_box(0);
}
lean_inc_ref(x_386);
x_391 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_391, 0, x_386);
x_392 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_392, 0, x_386);
x_393 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_393, 0, x_391);
lean_ctor_set(x_393, 1, x_392);
x_394 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_394, 0, x_389);
x_395 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_395, 0, x_388);
x_396 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_396, 0, x_387);
if (lean_is_scalar(x_390)) {
 x_397 = lean_alloc_ctor(0, 5, 0);
} else {
 x_397 = x_390;
}
lean_ctor_set(x_397, 0, x_393);
lean_ctor_set(x_397, 1, x_20);
lean_ctor_set(x_397, 2, x_396);
lean_ctor_set(x_397, 3, x_395);
lean_ctor_set(x_397, 4, x_394);
if (lean_is_scalar(x_385)) {
 x_398 = lean_alloc_ctor(0, 2, 0);
} else {
 x_398 = x_385;
}
lean_ctor_set(x_398, 0, x_397);
lean_ctor_set(x_398, 1, x_21);
x_399 = l_ReaderT_instMonad___redArg(x_398);
x_400 = lean_ctor_get(x_399, 0);
lean_inc_ref(x_400);
if (lean_is_exclusive(x_399)) {
 lean_ctor_release(x_399, 0);
 lean_ctor_release(x_399, 1);
 x_401 = x_399;
} else {
 lean_dec_ref(x_399);
 x_401 = lean_box(0);
}
x_402 = lean_ctor_get(x_400, 0);
lean_inc_ref(x_402);
x_403 = lean_ctor_get(x_400, 2);
lean_inc(x_403);
x_404 = lean_ctor_get(x_400, 3);
lean_inc(x_404);
x_405 = lean_ctor_get(x_400, 4);
lean_inc(x_405);
if (lean_is_exclusive(x_400)) {
 lean_ctor_release(x_400, 0);
 lean_ctor_release(x_400, 1);
 lean_ctor_release(x_400, 2);
 lean_ctor_release(x_400, 3);
 lean_ctor_release(x_400, 4);
 x_406 = x_400;
} else {
 lean_dec_ref(x_400);
 x_406 = lean_box(0);
}
x_407 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_408 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_409 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_409, 0, x_382);
lean_closure_set(x_409, 1, x_408);
x_410 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_410, 0, lean_box(0));
lean_closure_set(x_410, 1, lean_box(0));
lean_closure_set(x_410, 2, lean_box(0));
lean_closure_set(x_410, 3, lean_box(0));
lean_closure_set(x_410, 4, x_381);
x_411 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_411, 0, x_409);
lean_closure_set(x_411, 1, x_408);
x_412 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_412, 0, lean_box(0));
lean_closure_set(x_412, 1, lean_box(0));
lean_closure_set(x_412, 2, lean_box(0));
lean_closure_set(x_412, 3, lean_box(0));
lean_closure_set(x_412, 4, x_410);
x_413 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_413, 0, x_411);
lean_closure_set(x_413, 1, x_407);
x_414 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_414, 0, lean_box(0));
lean_closure_set(x_414, 1, x_412);
x_415 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_416 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_402);
x_417 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_417, 0, x_402);
x_418 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_418, 0, x_402);
x_419 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_419, 0, x_417);
lean_ctor_set(x_419, 1, x_418);
x_420 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_420, 0, x_405);
x_421 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_421, 0, x_404);
x_422 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_422, 0, x_403);
if (lean_is_scalar(x_406)) {
 x_423 = lean_alloc_ctor(0, 5, 0);
} else {
 x_423 = x_406;
}
lean_ctor_set(x_423, 0, x_419);
lean_ctor_set(x_423, 1, x_415);
lean_ctor_set(x_423, 2, x_422);
lean_ctor_set(x_423, 3, x_421);
lean_ctor_set(x_423, 4, x_420);
if (lean_is_scalar(x_401)) {
 x_424 = lean_alloc_ctor(0, 2, 0);
} else {
 x_424 = x_401;
}
lean_ctor_set(x_424, 0, x_423);
lean_ctor_set(x_424, 1, x_416);
x_425 = l_ReaderT_instMonad___redArg(x_424);
x_426 = l_ReaderT_instMonad___redArg(x_425);
x_427 = l_ReaderT_instMonad___redArg(x_426);
x_428 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
lean_inc_ref(x_37);
x_429 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_429, 0, x_11);
lean_closure_set(x_429, 1, x_37);
x_430 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_430, 0, x_11);
lean_closure_set(x_430, 1, x_37);
x_431 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_431, 0, x_429);
lean_ctor_set(x_431, 1, x_430);
lean_inc_ref(x_431);
x_432 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_432, 0, x_11);
lean_closure_set(x_432, 1, x_431);
x_433 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_433, 0, x_11);
lean_closure_set(x_433, 1, x_431);
x_434 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_434, 0, x_432);
lean_ctor_set(x_434, 1, x_433);
x_435 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_435, 0, x_413);
lean_closure_set(x_435, 1, x_428);
x_436 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 2);
lean_closure_set(x_436, 0, lean_box(0));
lean_closure_set(x_436, 1, x_414);
x_437 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_437, 0, x_436);
lean_ctor_set(x_437, 1, x_435);
x_438 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_439 = lean_ctor_get(x_438, 0);
lean_inc(x_439);
x_440 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_441 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_442 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_383);
x_443 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_442, x_383);
lean_inc_ref(x_441);
x_444 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_444, 0, x_440);
lean_ctor_set(x_444, 1, x_441);
lean_ctor_set(x_444, 2, x_443);
x_445 = lean_st_ref_get(x_3);
x_446 = lean_ctor_get(x_445, 0);
lean_inc(x_446);
lean_dec(x_445);
x_447 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_447);
x_448 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_448, 0, x_446);
lean_ctor_set(x_448, 1, x_447);
x_449 = lp_aesop_Aesop_getRootMVarId(x_448, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_448);
if (lean_obj_tag(x_449) == 0)
{
lean_object* x_450; lean_object* x_451; lean_object* x_452; lean_object* x_453; lean_object* x_454; lean_object* x_455; lean_object* x_456; lean_object* x_457; 
x_450 = lean_ctor_get(x_449, 0);
lean_inc(x_450);
lean_dec_ref(x_449);
x_451 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed), 9, 0);
x_452 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed), 1, 0);
x_453 = lp_aesop_Aesop_finalizeProof___redArg___closed__6;
lean_inc_ref(x_427);
lean_inc_ref(x_383);
x_454 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed), 24, 14);
lean_closure_set(x_454, 0, x_1);
lean_closure_set(x_454, 1, x_383);
lean_closure_set(x_454, 2, x_437);
lean_closure_set(x_454, 3, x_408);
lean_closure_set(x_454, 4, x_408);
lean_closure_set(x_454, 5, x_407);
lean_closure_set(x_454, 6, x_428);
lean_closure_set(x_454, 7, x_441);
lean_closure_set(x_454, 8, x_442);
lean_closure_set(x_454, 9, x_439);
lean_closure_set(x_454, 10, x_427);
lean_closure_set(x_454, 11, x_453);
lean_closure_set(x_454, 12, x_452);
lean_closure_set(x_454, 13, x_444);
x_455 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_455, 0, lean_box(0));
lean_closure_set(x_455, 1, lean_box(0));
lean_closure_set(x_455, 2, x_427);
lean_closure_set(x_455, 3, lean_box(0));
lean_closure_set(x_455, 4, lean_box(0));
lean_closure_set(x_455, 5, x_451);
lean_closure_set(x_455, 6, x_454);
x_456 = l_Lean_MVarId_withContext___redArg(x_434, x_383, x_450, x_455);
x_457 = lean_apply_9(x_456, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_457;
}
else
{
lean_object* x_458; lean_object* x_459; lean_object* x_460; 
lean_dec_ref(x_444);
lean_dec_ref(x_441);
lean_dec(x_439);
lean_dec_ref(x_437);
lean_dec_ref(x_434);
lean_dec_ref(x_427);
lean_dec_ref(x_383);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_458 = lean_ctor_get(x_449, 0);
lean_inc(x_458);
if (lean_is_exclusive(x_449)) {
 lean_ctor_release(x_449, 0);
 x_459 = x_449;
} else {
 lean_dec_ref(x_449);
 x_459 = lean_box(0);
}
if (lean_is_scalar(x_459)) {
 x_460 = lean_alloc_ctor(1, 1, 0);
} else {
 x_460 = x_459;
}
lean_ctor_set(x_460, 0, x_458);
return x_460;
}
}
}
else
{
lean_object* x_461; lean_object* x_462; lean_object* x_463; lean_object* x_464; lean_object* x_465; lean_object* x_466; lean_object* x_467; lean_object* x_468; lean_object* x_469; lean_object* x_470; lean_object* x_471; lean_object* x_472; lean_object* x_473; lean_object* x_474; lean_object* x_475; lean_object* x_476; lean_object* x_477; lean_object* x_478; lean_object* x_479; lean_object* x_480; lean_object* x_481; lean_object* x_482; lean_object* x_483; lean_object* x_484; lean_object* x_485; lean_object* x_486; lean_object* x_487; lean_object* x_488; lean_object* x_489; lean_object* x_490; lean_object* x_491; lean_object* x_492; lean_object* x_493; lean_object* x_494; lean_object* x_495; lean_object* x_496; lean_object* x_497; lean_object* x_498; lean_object* x_499; lean_object* x_500; lean_object* x_501; lean_object* x_502; lean_object* x_503; lean_object* x_504; lean_object* x_505; lean_object* x_506; lean_object* x_507; lean_object* x_508; lean_object* x_509; lean_object* x_510; lean_object* x_511; lean_object* x_512; lean_object* x_513; lean_object* x_514; lean_object* x_515; lean_object* x_516; lean_object* x_517; lean_object* x_518; lean_object* x_519; lean_object* x_520; lean_object* x_521; lean_object* x_522; lean_object* x_523; lean_object* x_524; lean_object* x_525; lean_object* x_526; lean_object* x_527; lean_object* x_528; lean_object* x_529; lean_object* x_530; lean_object* x_531; lean_object* x_532; lean_object* x_533; lean_object* x_534; lean_object* x_535; lean_object* x_536; lean_object* x_537; lean_object* x_538; lean_object* x_539; lean_object* x_540; lean_object* x_541; lean_object* x_542; lean_object* x_543; lean_object* x_544; lean_object* x_545; lean_object* x_546; lean_object* x_547; lean_object* x_548; lean_object* x_549; lean_object* x_550; lean_object* x_551; lean_object* x_552; lean_object* x_553; lean_object* x_554; 
x_461 = lean_ctor_get(x_13, 0);
x_462 = lean_ctor_get(x_13, 2);
x_463 = lean_ctor_get(x_13, 3);
x_464 = lean_ctor_get(x_13, 4);
lean_inc(x_464);
lean_inc(x_463);
lean_inc(x_462);
lean_inc(x_461);
lean_dec(x_13);
x_465 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_466 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_461);
x_467 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_467, 0, x_461);
x_468 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_468, 0, x_461);
x_469 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_469, 0, x_467);
lean_ctor_set(x_469, 1, x_468);
x_470 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_470, 0, x_464);
x_471 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_471, 0, x_463);
x_472 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_472, 0, x_462);
x_473 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_473, 0, x_469);
lean_ctor_set(x_473, 1, x_465);
lean_ctor_set(x_473, 2, x_472);
lean_ctor_set(x_473, 3, x_471);
lean_ctor_set(x_473, 4, x_470);
x_474 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_474, 0, x_473);
lean_ctor_set(x_474, 1, x_466);
x_475 = l_ReaderT_instMonad___redArg(x_474);
x_476 = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(x_476, 0, lean_box(0));
lean_closure_set(x_476, 1, lean_box(0));
lean_closure_set(x_476, 2, x_475);
x_477 = l_instMonadControlTOfPure___redArg(x_476);
lean_inc_ref(x_477);
x_478 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_478, 0, x_11);
lean_closure_set(x_478, 1, x_477);
x_479 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_479, 0, x_11);
lean_closure_set(x_479, 1, x_477);
x_480 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_480, 0, x_478);
lean_ctor_set(x_480, 1, x_479);
lean_inc_ref(x_480);
x_481 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_481, 0, x_11);
lean_closure_set(x_481, 1, x_480);
x_482 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_482, 0, x_11);
lean_closure_set(x_482, 1, x_480);
x_483 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_483, 0, x_481);
lean_ctor_set(x_483, 1, x_482);
x_484 = l_Lean_Meta_instMonadMCtxMetaM;
x_485 = lean_ctor_get(x_484, 0);
lean_inc(x_485);
x_486 = lean_ctor_get(x_484, 1);
lean_inc(x_486);
if (lean_is_exclusive(x_484)) {
 lean_ctor_release(x_484, 0);
 lean_ctor_release(x_484, 1);
 x_487 = x_484;
} else {
 lean_dec_ref(x_484);
 x_487 = lean_box(0);
}
x_488 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
x_489 = lean_ctor_get(x_12, 0);
lean_inc_ref(x_489);
if (lean_is_exclusive(x_12)) {
 lean_ctor_release(x_12, 0);
 lean_ctor_release(x_12, 1);
 x_490 = x_12;
} else {
 lean_dec_ref(x_12);
 x_490 = lean_box(0);
}
x_491 = lean_ctor_get(x_489, 0);
lean_inc_ref(x_491);
x_492 = lean_ctor_get(x_489, 2);
lean_inc(x_492);
x_493 = lean_ctor_get(x_489, 3);
lean_inc(x_493);
x_494 = lean_ctor_get(x_489, 4);
lean_inc(x_494);
if (lean_is_exclusive(x_489)) {
 lean_ctor_release(x_489, 0);
 lean_ctor_release(x_489, 1);
 lean_ctor_release(x_489, 2);
 lean_ctor_release(x_489, 3);
 lean_ctor_release(x_489, 4);
 x_495 = x_489;
} else {
 lean_dec_ref(x_489);
 x_495 = lean_box(0);
}
lean_inc_ref(x_491);
x_496 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_496, 0, x_491);
x_497 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_497, 0, x_491);
x_498 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_498, 0, x_496);
lean_ctor_set(x_498, 1, x_497);
x_499 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_499, 0, x_494);
x_500 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_500, 0, x_493);
x_501 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_501, 0, x_492);
if (lean_is_scalar(x_495)) {
 x_502 = lean_alloc_ctor(0, 5, 0);
} else {
 x_502 = x_495;
}
lean_ctor_set(x_502, 0, x_498);
lean_ctor_set(x_502, 1, x_465);
lean_ctor_set(x_502, 2, x_501);
lean_ctor_set(x_502, 3, x_500);
lean_ctor_set(x_502, 4, x_499);
if (lean_is_scalar(x_490)) {
 x_503 = lean_alloc_ctor(0, 2, 0);
} else {
 x_503 = x_490;
}
lean_ctor_set(x_503, 0, x_502);
lean_ctor_set(x_503, 1, x_466);
x_504 = l_ReaderT_instMonad___redArg(x_503);
x_505 = lean_ctor_get(x_504, 0);
lean_inc_ref(x_505);
if (lean_is_exclusive(x_504)) {
 lean_ctor_release(x_504, 0);
 lean_ctor_release(x_504, 1);
 x_506 = x_504;
} else {
 lean_dec_ref(x_504);
 x_506 = lean_box(0);
}
x_507 = lean_ctor_get(x_505, 0);
lean_inc_ref(x_507);
x_508 = lean_ctor_get(x_505, 2);
lean_inc(x_508);
x_509 = lean_ctor_get(x_505, 3);
lean_inc(x_509);
x_510 = lean_ctor_get(x_505, 4);
lean_inc(x_510);
if (lean_is_exclusive(x_505)) {
 lean_ctor_release(x_505, 0);
 lean_ctor_release(x_505, 1);
 lean_ctor_release(x_505, 2);
 lean_ctor_release(x_505, 3);
 lean_ctor_release(x_505, 4);
 x_511 = x_505;
} else {
 lean_dec_ref(x_505);
 x_511 = lean_box(0);
}
x_512 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_513 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_514 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_514, 0, x_486);
lean_closure_set(x_514, 1, x_513);
x_515 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_515, 0, lean_box(0));
lean_closure_set(x_515, 1, lean_box(0));
lean_closure_set(x_515, 2, lean_box(0));
lean_closure_set(x_515, 3, lean_box(0));
lean_closure_set(x_515, 4, x_485);
x_516 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_516, 0, x_514);
lean_closure_set(x_516, 1, x_513);
x_517 = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(x_517, 0, lean_box(0));
lean_closure_set(x_517, 1, lean_box(0));
lean_closure_set(x_517, 2, lean_box(0));
lean_closure_set(x_517, 3, lean_box(0));
lean_closure_set(x_517, 4, x_515);
x_518 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_518, 0, x_516);
lean_closure_set(x_518, 1, x_512);
x_519 = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(x_519, 0, lean_box(0));
lean_closure_set(x_519, 1, x_517);
x_520 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_521 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_507);
x_522 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_522, 0, x_507);
x_523 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_523, 0, x_507);
x_524 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_524, 0, x_522);
lean_ctor_set(x_524, 1, x_523);
x_525 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_525, 0, x_510);
x_526 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_526, 0, x_509);
x_527 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_527, 0, x_508);
if (lean_is_scalar(x_511)) {
 x_528 = lean_alloc_ctor(0, 5, 0);
} else {
 x_528 = x_511;
}
lean_ctor_set(x_528, 0, x_524);
lean_ctor_set(x_528, 1, x_520);
lean_ctor_set(x_528, 2, x_527);
lean_ctor_set(x_528, 3, x_526);
lean_ctor_set(x_528, 4, x_525);
if (lean_is_scalar(x_506)) {
 x_529 = lean_alloc_ctor(0, 2, 0);
} else {
 x_529 = x_506;
}
lean_ctor_set(x_529, 0, x_528);
lean_ctor_set(x_529, 1, x_521);
x_530 = l_ReaderT_instMonad___redArg(x_529);
x_531 = l_ReaderT_instMonad___redArg(x_530);
x_532 = l_ReaderT_instMonad___redArg(x_531);
x_533 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
lean_inc_ref(x_483);
x_534 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_534, 0, x_11);
lean_closure_set(x_534, 1, x_483);
x_535 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_535, 0, x_11);
lean_closure_set(x_535, 1, x_483);
x_536 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_536, 0, x_534);
lean_ctor_set(x_536, 1, x_535);
lean_inc_ref(x_536);
x_537 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_537, 0, x_11);
lean_closure_set(x_537, 1, x_536);
x_538 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_538, 0, x_11);
lean_closure_set(x_538, 1, x_536);
x_539 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_539, 0, x_537);
lean_ctor_set(x_539, 1, x_538);
x_540 = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(x_540, 0, x_518);
lean_closure_set(x_540, 1, x_533);
x_541 = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 2);
lean_closure_set(x_541, 0, lean_box(0));
lean_closure_set(x_541, 1, x_519);
if (lean_is_scalar(x_487)) {
 x_542 = lean_alloc_ctor(0, 2, 0);
} else {
 x_542 = x_487;
}
lean_ctor_set(x_542, 0, x_541);
lean_ctor_set(x_542, 1, x_540);
x_543 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_544 = lean_ctor_get(x_543, 0);
lean_inc(x_544);
x_545 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_546 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_547 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_488);
x_548 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_547, x_488);
lean_inc_ref(x_546);
x_549 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_549, 0, x_545);
lean_ctor_set(x_549, 1, x_546);
lean_ctor_set(x_549, 2, x_548);
x_550 = lean_st_ref_get(x_3);
x_551 = lean_ctor_get(x_550, 0);
lean_inc(x_551);
lean_dec(x_550);
x_552 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_552);
x_553 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_553, 0, x_551);
lean_ctor_set(x_553, 1, x_552);
x_554 = lp_aesop_Aesop_getRootMVarId(x_553, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_553);
if (lean_obj_tag(x_554) == 0)
{
lean_object* x_555; lean_object* x_556; lean_object* x_557; lean_object* x_558; lean_object* x_559; lean_object* x_560; lean_object* x_561; lean_object* x_562; 
x_555 = lean_ctor_get(x_554, 0);
lean_inc(x_555);
lean_dec_ref(x_554);
x_556 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed), 9, 0);
x_557 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed), 1, 0);
x_558 = lp_aesop_Aesop_finalizeProof___redArg___closed__6;
lean_inc_ref(x_532);
lean_inc_ref(x_488);
x_559 = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed), 24, 14);
lean_closure_set(x_559, 0, x_1);
lean_closure_set(x_559, 1, x_488);
lean_closure_set(x_559, 2, x_542);
lean_closure_set(x_559, 3, x_513);
lean_closure_set(x_559, 4, x_513);
lean_closure_set(x_559, 5, x_512);
lean_closure_set(x_559, 6, x_533);
lean_closure_set(x_559, 7, x_546);
lean_closure_set(x_559, 8, x_547);
lean_closure_set(x_559, 9, x_544);
lean_closure_set(x_559, 10, x_532);
lean_closure_set(x_559, 11, x_558);
lean_closure_set(x_559, 12, x_557);
lean_closure_set(x_559, 13, x_549);
x_560 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_560, 0, lean_box(0));
lean_closure_set(x_560, 1, lean_box(0));
lean_closure_set(x_560, 2, x_532);
lean_closure_set(x_560, 3, lean_box(0));
lean_closure_set(x_560, 4, lean_box(0));
lean_closure_set(x_560, 5, x_556);
lean_closure_set(x_560, 6, x_559);
x_561 = l_Lean_MVarId_withContext___redArg(x_539, x_488, x_555, x_560);
x_562 = lean_apply_9(x_561, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_562;
}
else
{
lean_object* x_563; lean_object* x_564; lean_object* x_565; 
lean_dec_ref(x_549);
lean_dec_ref(x_546);
lean_dec(x_544);
lean_dec_ref(x_542);
lean_dec_ref(x_539);
lean_dec_ref(x_532);
lean_dec_ref(x_488);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_563 = lean_ctor_get(x_554, 0);
lean_inc(x_563);
if (lean_is_exclusive(x_554)) {
 lean_ctor_release(x_554, 0);
 x_564 = x_554;
} else {
 lean_dec_ref(x_554);
 x_564 = lean_box(0);
}
if (lean_is_scalar(x_564)) {
 x_565 = lean_alloc_ctor(1, 1, 0);
} else {
 x_565 = x_564;
}
lean_ctor_set(x_565, 0, x_563);
return x_565;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_finalizeProof___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_finalizeProof(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_finalizeProof___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = l_Lean_Core_instMonadLogCoreM;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_3 = l_Lean_instMonadLogOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_traceScript___redArg___closed__0;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_3 = l_Lean_instMonadLogOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_traceScript___redArg___closed__1;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_3 = l_Lean_instMonadLogOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_traceScript___redArg___closed__2;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_3 = l_Lean_instMonadLogOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_traceScript___redArg___closed__3;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_3 = l_Lean_instMonadLogOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_traceScript___redArg___closed__4;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
x_3 = l_Lean_instMonadLogOfMonadLift___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractScriptCore___boxed), 10, 1);
lean_closure_set(x_10, 0, x_1);
x_11 = lp_aesop_Aesop_ExtractScriptM_run___redArg(x_10, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_traceScript___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" Extract script", 15, 15);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___lam__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = l_Lean_exceptEmoji___redArg(x_1);
x_11 = l_Lean_stringToMessageData(x_10);
x_12 = lp_aesop_Aesop_traceScript___redArg___lam__1___closed__1;
x_13 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
x_14 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_traceScript___redArg___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_2 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__12;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_2 = lp_aesop_Aesop_traceScript___redArg___closed__6;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_2 = lp_aesop_Aesop_traceScript___redArg___closed__7;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
x_2 = lp_aesop_Aesop_traceScript___redArg___closed__8;
x_3 = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___lam__2___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_TraceOption_script;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___lam__2___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Unstructured script:", 20, 20);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___lam__2___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__2;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, uint8_t x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22, lean_object* x_23, lean_object* x_24, lean_object* x_25) {
_start:
{
uint8_t x_27; 
x_27 = !lean_is_exclusive(x_17);
if (x_27 == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_28 = lean_ctor_get(x_17, 0);
x_29 = lean_ctor_get(x_17, 1);
x_30 = lean_st_ref_get(x_19);
lean_dec(x_30);
lean_inc(x_25);
lean_inc_ref(x_24);
lean_inc(x_23);
lean_inc_ref(x_22);
x_31 = lp_aesop_Aesop_Script_UScript_checkIfEnabled(x_28, x_22, x_23, x_24, x_25);
if (lean_obj_tag(x_31) == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
lean_dec_ref(x_31);
x_32 = lean_st_ref_get(x_19);
x_33 = lean_ctor_get(x_32, 0);
lean_inc(x_33);
lean_dec(x_32);
x_34 = lean_ctor_get(x_18, 0);
lean_inc_ref(x_34);
lean_ctor_set(x_17, 1, x_34);
lean_ctor_set(x_17, 0, x_33);
x_35 = lp_aesop_Aesop_getRootMVarId(x_17, x_20, x_21, x_22, x_23, x_24, x_25);
lean_dec_ref(x_17);
if (lean_obj_tag(x_35) == 0)
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_36 = lean_ctor_get(x_35, 0);
lean_inc(x_36);
lean_dec_ref(x_35);
x_37 = lean_st_ref_get(x_19);
lean_dec(x_37);
x_38 = lp_aesop_Aesop_getRootMetaState___redArg(x_20);
if (lean_obj_tag(x_38) == 0)
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_60; lean_object* x_61; lean_object* x_62; 
x_39 = lean_ctor_get(x_38, 0);
lean_inc(x_39);
lean_dec_ref(x_38);
x_60 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
lean_inc_ref(x_1);
x_61 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_1, x_2, x_60);
lean_inc(x_25);
lean_inc_ref(x_24);
lean_inc(x_23);
lean_inc_ref(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc_ref(x_18);
x_62 = lean_apply_9(x_61, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, lean_box(0));
if (lean_obj_tag(x_62) == 0)
{
lean_object* x_63; uint8_t x_64; 
x_63 = lean_ctor_get(x_62, 0);
lean_inc(x_63);
lean_dec_ref(x_62);
x_64 = lean_unbox(x_63);
lean_dec(x_63);
if (x_64 == 0)
{
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
x_40 = x_18;
x_41 = x_19;
x_42 = x_20;
x_43 = x_21;
x_44 = x_22;
x_45 = x_23;
x_46 = x_24;
x_47 = x_25;
x_48 = lean_box(0);
goto block_59;
}
else
{
lean_object* x_65; lean_object* x_66; 
x_65 = lean_st_ref_get(x_19);
lean_dec(x_65);
lean_inc(x_25);
lean_inc_ref(x_24);
lean_inc(x_23);
lean_inc_ref(x_22);
lean_inc(x_36);
lean_inc(x_39);
x_66 = lp_aesop_Aesop_Script_UScript_renderTacticSeq(x_28, x_39, x_36, x_22, x_23, x_24, x_25);
if (lean_obj_tag(x_66) == 0)
{
lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; uint8_t x_75; 
x_67 = lean_ctor_get(x_66, 0);
lean_inc(x_67);
lean_dec_ref(x_66);
x_68 = l_Lean_Core_instMonadTraceCoreM;
x_69 = l_Lean_instMonadTraceOfMonadLift___redArg(x_11, x_68);
x_70 = l_Lean_instMonadTraceOfMonadLift___redArg(x_12, x_69);
x_71 = l_Lean_instMonadTraceOfMonadLift___redArg(x_13, x_70);
x_72 = l_Lean_instMonadTraceOfMonadLift___redArg(x_14, x_71);
x_73 = l_Lean_instMonadTraceOfMonadLift___redArg(x_15, x_72);
x_74 = l_Lean_instMonadTraceOfMonadLift___redArg(x_16, x_73);
x_75 = !lean_is_exclusive(x_60);
if (x_75 == 0)
{
lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; 
x_76 = lean_ctor_get(x_60, 0);
x_77 = lean_ctor_get(x_60, 1);
lean_dec(x_77);
x_78 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__3;
x_79 = l_Lean_MessageData_ofSyntax(x_67);
x_80 = l_Lean_indentD(x_79);
lean_ctor_set_tag(x_60, 7);
lean_ctor_set(x_60, 1, x_80);
lean_ctor_set(x_60, 0, x_78);
lean_inc_ref(x_6);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
x_81 = l_Lean_addTrace___redArg(x_1, x_74, x_4, x_6, x_76, x_60);
lean_inc(x_25);
lean_inc_ref(x_24);
lean_inc(x_23);
lean_inc_ref(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc_ref(x_18);
x_82 = lean_apply_9(x_81, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, lean_box(0));
if (lean_obj_tag(x_82) == 0)
{
lean_dec_ref(x_82);
x_40 = x_18;
x_41 = x_19;
x_42 = x_20;
x_43 = x_21;
x_44 = x_22;
x_45 = x_23;
x_46 = x_24;
x_47 = x_25;
x_48 = lean_box(0);
goto block_59;
}
else
{
lean_dec(x_39);
lean_dec(x_36);
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_82;
}
}
else
{
lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; 
x_83 = lean_ctor_get(x_60, 0);
lean_inc(x_83);
lean_dec(x_60);
x_84 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__3;
x_85 = l_Lean_MessageData_ofSyntax(x_67);
x_86 = l_Lean_indentD(x_85);
x_87 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_87, 0, x_84);
lean_ctor_set(x_87, 1, x_86);
lean_inc_ref(x_6);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
x_88 = l_Lean_addTrace___redArg(x_1, x_74, x_4, x_6, x_83, x_87);
lean_inc(x_25);
lean_inc_ref(x_24);
lean_inc(x_23);
lean_inc_ref(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc_ref(x_18);
x_89 = lean_apply_9(x_88, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, lean_box(0));
if (lean_obj_tag(x_89) == 0)
{
lean_dec_ref(x_89);
x_40 = x_18;
x_41 = x_19;
x_42 = x_20;
x_43 = x_21;
x_44 = x_22;
x_45 = x_23;
x_46 = x_24;
x_47 = x_25;
x_48 = lean_box(0);
goto block_59;
}
else
{
lean_dec(x_39);
lean_dec(x_36);
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_89;
}
}
}
else
{
uint8_t x_90; 
lean_dec(x_39);
lean_dec(x_36);
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_90 = !lean_is_exclusive(x_66);
if (x_90 == 0)
{
return x_66;
}
else
{
lean_object* x_91; lean_object* x_92; 
x_91 = lean_ctor_get(x_66, 0);
lean_inc(x_91);
lean_dec(x_66);
x_92 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_92, 0, x_91);
return x_92;
}
}
}
}
else
{
uint8_t x_93; 
lean_dec(x_39);
lean_dec(x_36);
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_93 = !lean_is_exclusive(x_62);
if (x_93 == 0)
{
return x_62;
}
else
{
lean_object* x_94; lean_object* x_95; 
x_94 = lean_ctor_get(x_62, 0);
lean_inc(x_94);
lean_dec(x_62);
x_95 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_95, 0, x_94);
return x_95;
}
}
block_59:
{
lean_object* x_49; uint8_t x_50; lean_object* x_51; 
x_49 = lean_st_ref_get(x_41);
lean_dec(x_49);
x_50 = lean_unbox(x_29);
lean_dec(x_29);
lean_inc(x_47);
lean_inc_ref(x_46);
lean_inc(x_45);
lean_inc_ref(x_44);
lean_inc(x_36);
lean_inc(x_39);
lean_inc(x_28);
x_51 = lp_aesop_Aesop_Script_UScript_optimize(x_28, x_50, x_39, x_36, x_44, x_45, x_46, x_47);
if (lean_obj_tag(x_51) == 0)
{
lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; 
x_52 = lean_ctor_get(x_51, 0);
lean_inc(x_52);
lean_dec_ref(x_51);
x_53 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__0;
x_54 = lp_aesop_Aesop_checkAndTraceScript___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_28, x_52, x_39, x_36, x_9, x_10, x_53);
x_55 = lean_apply_9(x_54, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, lean_box(0));
return x_55;
}
else
{
uint8_t x_56; 
lean_dec(x_47);
lean_dec_ref(x_46);
lean_dec(x_45);
lean_dec_ref(x_44);
lean_dec(x_43);
lean_dec(x_42);
lean_dec(x_41);
lean_dec_ref(x_40);
lean_dec(x_39);
lean_dec(x_36);
lean_dec(x_28);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_56 = !lean_is_exclusive(x_51);
if (x_56 == 0)
{
return x_51;
}
else
{
lean_object* x_57; lean_object* x_58; 
x_57 = lean_ctor_get(x_51, 0);
lean_inc(x_57);
lean_dec(x_51);
x_58 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_58, 0, x_57);
return x_58;
}
}
}
}
else
{
uint8_t x_96; 
lean_dec(x_36);
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_96 = !lean_is_exclusive(x_38);
if (x_96 == 0)
{
return x_38;
}
else
{
lean_object* x_97; lean_object* x_98; 
x_97 = lean_ctor_get(x_38, 0);
lean_inc(x_97);
lean_dec(x_38);
x_98 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_98, 0, x_97);
return x_98;
}
}
}
else
{
uint8_t x_99; 
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_99 = !lean_is_exclusive(x_35);
if (x_99 == 0)
{
return x_35;
}
else
{
lean_object* x_100; lean_object* x_101; 
x_100 = lean_ctor_get(x_35, 0);
lean_inc(x_100);
lean_dec(x_35);
x_101 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_101, 0, x_100);
return x_101;
}
}
}
else
{
lean_free_object(x_17);
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_31;
}
}
else
{
lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; 
x_102 = lean_ctor_get(x_17, 0);
x_103 = lean_ctor_get(x_17, 1);
lean_inc(x_103);
lean_inc(x_102);
lean_dec(x_17);
x_104 = lean_st_ref_get(x_19);
lean_dec(x_104);
lean_inc(x_25);
lean_inc_ref(x_24);
lean_inc(x_23);
lean_inc_ref(x_22);
x_105 = lp_aesop_Aesop_Script_UScript_checkIfEnabled(x_102, x_22, x_23, x_24, x_25);
if (lean_obj_tag(x_105) == 0)
{
lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; 
lean_dec_ref(x_105);
x_106 = lean_st_ref_get(x_19);
x_107 = lean_ctor_get(x_106, 0);
lean_inc(x_107);
lean_dec(x_106);
x_108 = lean_ctor_get(x_18, 0);
lean_inc_ref(x_108);
x_109 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_109, 0, x_107);
lean_ctor_set(x_109, 1, x_108);
x_110 = lp_aesop_Aesop_getRootMVarId(x_109, x_20, x_21, x_22, x_23, x_24, x_25);
lean_dec_ref(x_109);
if (lean_obj_tag(x_110) == 0)
{
lean_object* x_111; lean_object* x_112; lean_object* x_113; 
x_111 = lean_ctor_get(x_110, 0);
lean_inc(x_111);
lean_dec_ref(x_110);
x_112 = lean_st_ref_get(x_19);
lean_dec(x_112);
x_113 = lp_aesop_Aesop_getRootMetaState___redArg(x_20);
if (lean_obj_tag(x_113) == 0)
{
lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_135; lean_object* x_136; lean_object* x_137; 
x_114 = lean_ctor_get(x_113, 0);
lean_inc(x_114);
lean_dec_ref(x_113);
x_135 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
lean_inc_ref(x_1);
x_136 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_1, x_2, x_135);
lean_inc(x_25);
lean_inc_ref(x_24);
lean_inc(x_23);
lean_inc_ref(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc_ref(x_18);
x_137 = lean_apply_9(x_136, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, lean_box(0));
if (lean_obj_tag(x_137) == 0)
{
lean_object* x_138; uint8_t x_139; 
x_138 = lean_ctor_get(x_137, 0);
lean_inc(x_138);
lean_dec_ref(x_137);
x_139 = lean_unbox(x_138);
lean_dec(x_138);
if (x_139 == 0)
{
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
x_115 = x_18;
x_116 = x_19;
x_117 = x_20;
x_118 = x_21;
x_119 = x_22;
x_120 = x_23;
x_121 = x_24;
x_122 = x_25;
x_123 = lean_box(0);
goto block_134;
}
else
{
lean_object* x_140; lean_object* x_141; 
x_140 = lean_st_ref_get(x_19);
lean_dec(x_140);
lean_inc(x_25);
lean_inc_ref(x_24);
lean_inc(x_23);
lean_inc_ref(x_22);
lean_inc(x_111);
lean_inc(x_114);
x_141 = lp_aesop_Aesop_Script_UScript_renderTacticSeq(x_102, x_114, x_111, x_22, x_23, x_24, x_25);
if (lean_obj_tag(x_141) == 0)
{
lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; 
x_142 = lean_ctor_get(x_141, 0);
lean_inc(x_142);
lean_dec_ref(x_141);
x_143 = l_Lean_Core_instMonadTraceCoreM;
x_144 = l_Lean_instMonadTraceOfMonadLift___redArg(x_11, x_143);
x_145 = l_Lean_instMonadTraceOfMonadLift___redArg(x_12, x_144);
x_146 = l_Lean_instMonadTraceOfMonadLift___redArg(x_13, x_145);
x_147 = l_Lean_instMonadTraceOfMonadLift___redArg(x_14, x_146);
x_148 = l_Lean_instMonadTraceOfMonadLift___redArg(x_15, x_147);
x_149 = l_Lean_instMonadTraceOfMonadLift___redArg(x_16, x_148);
x_150 = lean_ctor_get(x_135, 0);
lean_inc(x_150);
if (lean_is_exclusive(x_135)) {
 lean_ctor_release(x_135, 0);
 lean_ctor_release(x_135, 1);
 x_151 = x_135;
} else {
 lean_dec_ref(x_135);
 x_151 = lean_box(0);
}
x_152 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__3;
x_153 = l_Lean_MessageData_ofSyntax(x_142);
x_154 = l_Lean_indentD(x_153);
if (lean_is_scalar(x_151)) {
 x_155 = lean_alloc_ctor(7, 2, 0);
} else {
 x_155 = x_151;
 lean_ctor_set_tag(x_155, 7);
}
lean_ctor_set(x_155, 0, x_152);
lean_ctor_set(x_155, 1, x_154);
lean_inc_ref(x_6);
lean_inc_ref(x_4);
lean_inc_ref(x_1);
x_156 = l_Lean_addTrace___redArg(x_1, x_149, x_4, x_6, x_150, x_155);
lean_inc(x_25);
lean_inc_ref(x_24);
lean_inc(x_23);
lean_inc_ref(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc_ref(x_18);
x_157 = lean_apply_9(x_156, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, lean_box(0));
if (lean_obj_tag(x_157) == 0)
{
lean_dec_ref(x_157);
x_115 = x_18;
x_116 = x_19;
x_117 = x_20;
x_118 = x_21;
x_119 = x_22;
x_120 = x_23;
x_121 = x_24;
x_122 = x_25;
x_123 = lean_box(0);
goto block_134;
}
else
{
lean_dec(x_114);
lean_dec(x_111);
lean_dec(x_103);
lean_dec(x_102);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_157;
}
}
else
{
lean_object* x_158; lean_object* x_159; lean_object* x_160; 
lean_dec(x_114);
lean_dec(x_111);
lean_dec(x_103);
lean_dec(x_102);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_158 = lean_ctor_get(x_141, 0);
lean_inc(x_158);
if (lean_is_exclusive(x_141)) {
 lean_ctor_release(x_141, 0);
 x_159 = x_141;
} else {
 lean_dec_ref(x_141);
 x_159 = lean_box(0);
}
if (lean_is_scalar(x_159)) {
 x_160 = lean_alloc_ctor(1, 1, 0);
} else {
 x_160 = x_159;
}
lean_ctor_set(x_160, 0, x_158);
return x_160;
}
}
}
else
{
lean_object* x_161; lean_object* x_162; lean_object* x_163; 
lean_dec(x_114);
lean_dec(x_111);
lean_dec(x_103);
lean_dec(x_102);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_161 = lean_ctor_get(x_137, 0);
lean_inc(x_161);
if (lean_is_exclusive(x_137)) {
 lean_ctor_release(x_137, 0);
 x_162 = x_137;
} else {
 lean_dec_ref(x_137);
 x_162 = lean_box(0);
}
if (lean_is_scalar(x_162)) {
 x_163 = lean_alloc_ctor(1, 1, 0);
} else {
 x_163 = x_162;
}
lean_ctor_set(x_163, 0, x_161);
return x_163;
}
block_134:
{
lean_object* x_124; uint8_t x_125; lean_object* x_126; 
x_124 = lean_st_ref_get(x_116);
lean_dec(x_124);
x_125 = lean_unbox(x_103);
lean_dec(x_103);
lean_inc(x_122);
lean_inc_ref(x_121);
lean_inc(x_120);
lean_inc_ref(x_119);
lean_inc(x_111);
lean_inc(x_114);
lean_inc(x_102);
x_126 = lp_aesop_Aesop_Script_UScript_optimize(x_102, x_125, x_114, x_111, x_119, x_120, x_121, x_122);
if (lean_obj_tag(x_126) == 0)
{
lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; 
x_127 = lean_ctor_get(x_126, 0);
lean_inc(x_127);
lean_dec_ref(x_126);
x_128 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__0;
x_129 = lp_aesop_Aesop_checkAndTraceScript___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_102, x_127, x_114, x_111, x_9, x_10, x_128);
x_130 = lean_apply_9(x_129, x_115, x_116, x_117, x_118, x_119, x_120, x_121, x_122, lean_box(0));
return x_130;
}
else
{
lean_object* x_131; lean_object* x_132; lean_object* x_133; 
lean_dec(x_122);
lean_dec_ref(x_121);
lean_dec(x_120);
lean_dec_ref(x_119);
lean_dec(x_118);
lean_dec(x_117);
lean_dec(x_116);
lean_dec_ref(x_115);
lean_dec(x_114);
lean_dec(x_111);
lean_dec(x_102);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_131 = lean_ctor_get(x_126, 0);
lean_inc(x_131);
if (lean_is_exclusive(x_126)) {
 lean_ctor_release(x_126, 0);
 x_132 = x_126;
} else {
 lean_dec_ref(x_126);
 x_132 = lean_box(0);
}
if (lean_is_scalar(x_132)) {
 x_133 = lean_alloc_ctor(1, 1, 0);
} else {
 x_133 = x_132;
}
lean_ctor_set(x_133, 0, x_131);
return x_133;
}
}
}
else
{
lean_object* x_164; lean_object* x_165; lean_object* x_166; 
lean_dec(x_111);
lean_dec(x_103);
lean_dec(x_102);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_164 = lean_ctor_get(x_113, 0);
lean_inc(x_164);
if (lean_is_exclusive(x_113)) {
 lean_ctor_release(x_113, 0);
 x_165 = x_113;
} else {
 lean_dec_ref(x_113);
 x_165 = lean_box(0);
}
if (lean_is_scalar(x_165)) {
 x_166 = lean_alloc_ctor(1, 1, 0);
} else {
 x_166 = x_165;
}
lean_ctor_set(x_166, 0, x_164);
return x_166;
}
}
else
{
lean_object* x_167; lean_object* x_168; lean_object* x_169; 
lean_dec(x_103);
lean_dec(x_102);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_167 = lean_ctor_get(x_110, 0);
lean_inc(x_167);
if (lean_is_exclusive(x_110)) {
 lean_ctor_release(x_110, 0);
 x_168 = x_110;
} else {
 lean_dec_ref(x_110);
 x_168 = lean_box(0);
}
if (lean_is_scalar(x_168)) {
 x_169 = lean_alloc_ctor(1, 1, 0);
} else {
 x_169 = x_168;
}
lean_ctor_set(x_169, 0, x_167);
return x_169;
}
}
else
{
lean_dec(x_103);
lean_dec(x_102);
lean_dec(x_25);
lean_dec_ref(x_24);
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec(x_11);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_105;
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__4;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_3 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop_Aesop_traceScript___redArg___closed__10;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_3 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_aesop_Aesop_traceScript___redArg___closed__11;
x_2 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_3 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2;
x_4 = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0;
x_2 = lp_aesop_Aesop_instMonadStatsReaderT___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_expandNextGoal___redArg___closed__4;
x_2 = l_Lean_instMonadAlwaysExceptReaderT___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_getRootGoal___boxed), 8, 0);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__16() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_aesop_collectStats;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__17() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_TraceOption_stats;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__18() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_aesop_stats_file;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_56; lean_object* x_57; lean_object* x_60; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; uint8_t x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_214; lean_object* x_469; lean_object* x_470; uint8_t x_471; 
x_64 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__0;
x_65 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__1;
x_66 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
x_67 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_68 = lean_ctor_get(x_67, 0);
lean_inc(x_68);
x_69 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__16;
x_70 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__17;
x_71 = lp_aesop_Aesop_traceScript___redArg___closed__5;
x_72 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_73 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_74 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_66);
x_75 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_74, x_66);
lean_inc_ref(x_72);
x_76 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_76, 0, x_73);
lean_ctor_set(x_76, 1, x_72);
lean_ctor_set(x_76, 2, x_75);
x_77 = l_Lean_KVMap_instValueBool;
x_78 = lean_ctor_get(x_9, 2);
x_79 = lean_alloc_closure((void*)(lp_aesop_Aesop_traceScript___redArg___lam__0___boxed), 9, 0);
x_80 = lean_alloc_closure((void*)(lp_aesop_Aesop_traceScript___redArg___lam__1___boxed), 9, 0);
x_81 = lp_aesop_Aesop_traceScript___redArg___closed__9;
x_469 = lp_aesop_Aesop_traceScript___redArg___closed__16;
x_470 = l_Lean_Option_get___redArg(x_77, x_78, x_469);
x_471 = lean_unbox(x_470);
lean_dec(x_470);
if (x_471 == 0)
{
uint8_t x_472; lean_object* x_473; lean_object* x_718; lean_object* x_726; lean_object* x_727; lean_object* x_728; 
x_726 = lp_aesop_Aesop_traceScript___redArg___closed__17;
lean_inc(x_68);
lean_inc_ref(x_66);
x_727 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_66, x_68, x_726);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_728 = lean_apply_9(x_727, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_728) == 0)
{
lean_object* x_729; uint8_t x_730; 
x_729 = lean_ctor_get(x_728, 0);
lean_inc(x_729);
x_730 = lean_unbox(x_729);
if (x_730 == 0)
{
lean_object* x_731; lean_object* x_732; lean_object* x_733; lean_object* x_734; uint8_t x_735; 
lean_dec_ref(x_728);
x_731 = l_Lean_KVMap_instValueString;
x_732 = lp_aesop_Aesop_traceScript___redArg___closed__18;
x_733 = l_Lean_Option_get___redArg(x_731, x_78, x_732);
x_734 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_735 = lean_string_dec_eq(x_733, x_734);
lean_dec(x_733);
if (x_735 == 0)
{
lean_dec(x_729);
x_214 = lean_box(0);
goto block_468;
}
else
{
uint8_t x_736; 
x_736 = lean_unbox(x_729);
lean_dec(x_729);
x_472 = x_736;
x_473 = lean_box(0);
goto block_717;
}
}
else
{
lean_dec(x_729);
x_718 = x_728;
goto block_725;
}
}
else
{
x_718 = x_728;
goto block_725;
}
block_717:
{
lean_object* x_474; uint8_t x_475; 
x_474 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_474);
x_475 = lean_ctor_get_uint8(x_474, sizeof(void*)*2);
if (x_475 == 0)
{
lean_dec_ref(x_474);
lean_dec_ref(x_80);
lean_dec_ref(x_79);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_60 = lean_box(0);
goto block_63;
}
else
{
if (x_472 == 0)
{
if (x_2 == 0)
{
lean_object* x_476; lean_object* x_477; lean_object* x_478; lean_object* x_479; lean_object* x_480; 
lean_dec_ref(x_80);
lean_dec_ref(x_79);
x_476 = lean_ctor_get(x_3, 0);
x_477 = lean_st_ref_get(x_4);
x_478 = lean_ctor_get(x_477, 0);
lean_inc(x_478);
lean_dec(x_477);
lean_inc_ref(x_476);
x_479 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_479, 0, x_478);
lean_ctor_set(x_479, 1, x_476);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_480 = lp_aesop_Aesop_extractSafePrefixScript(x_479, x_5, x_6, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_480) == 0)
{
lean_object* x_481; 
x_481 = lean_ctor_get(x_480, 0);
lean_inc(x_481);
lean_dec_ref(x_480);
x_106 = x_474;
x_107 = x_481;
x_108 = x_3;
x_109 = x_4;
x_110 = x_5;
x_111 = x_6;
x_112 = x_7;
x_113 = x_8;
x_114 = x_9;
x_115 = x_10;
x_116 = lean_box(0);
goto block_213;
}
else
{
uint8_t x_482; 
lean_dec_ref(x_474);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_482 = !lean_is_exclusive(x_480);
if (x_482 == 0)
{
return x_480;
}
else
{
lean_object* x_483; lean_object* x_484; 
x_483 = lean_ctor_get(x_480, 0);
lean_inc(x_483);
lean_dec(x_480);
x_484 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_484, 0, x_483);
return x_484;
}
}
}
else
{
lean_object* x_485; lean_object* x_486; lean_object* x_487; lean_object* x_488; lean_object* x_489; lean_object* x_490; lean_object* x_491; lean_object* x_492; lean_object* x_493; lean_object* x_494; lean_object* x_495; uint8_t x_496; 
x_485 = lean_ctor_get(x_3, 0);
x_486 = lean_st_ref_get(x_4);
x_487 = lean_ctor_get(x_486, 0);
lean_inc(x_487);
lean_dec(x_486);
x_488 = lp_aesop_Aesop_TreeM_instMonad;
x_489 = lp_aesop_Aesop_expandNextGoal___redArg___closed__2;
x_490 = lp_aesop_Aesop_traceScript___redArg___closed__12;
x_491 = lean_ctor_get(x_490, 0);
lean_inc_ref(x_491);
x_492 = lp_aesop_Aesop_traceScript___redArg___closed__13;
x_493 = lean_ctor_get(x_492, 0);
lean_inc(x_493);
x_494 = lp_aesop_Aesop_traceScript___redArg___closed__14;
x_495 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1;
x_496 = !lean_is_exclusive(x_495);
if (x_496 == 0)
{
lean_object* x_497; lean_object* x_498; uint8_t x_499; 
x_497 = lean_ctor_get(x_495, 0);
x_498 = lean_ctor_get(x_495, 1);
lean_dec(x_498);
x_499 = !lean_is_exclusive(x_497);
if (x_499 == 0)
{
lean_object* x_500; lean_object* x_501; lean_object* x_502; lean_object* x_503; lean_object* x_504; lean_object* x_505; lean_object* x_506; lean_object* x_507; lean_object* x_508; lean_object* x_509; lean_object* x_510; lean_object* x_511; lean_object* x_512; lean_object* x_513; uint8_t x_514; 
x_500 = lean_ctor_get(x_497, 0);
x_501 = lean_ctor_get(x_497, 2);
x_502 = lean_ctor_get(x_497, 3);
x_503 = lean_ctor_get(x_497, 4);
x_504 = lean_ctor_get(x_497, 1);
lean_dec(x_504);
x_505 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_506 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_500);
x_507 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_507, 0, x_500);
x_508 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_508, 0, x_500);
x_509 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_509, 0, x_507);
lean_ctor_set(x_509, 1, x_508);
x_510 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_510, 0, x_503);
x_511 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_511, 0, x_502);
x_512 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_512, 0, x_501);
lean_ctor_set(x_497, 4, x_510);
lean_ctor_set(x_497, 3, x_511);
lean_ctor_set(x_497, 2, x_512);
lean_ctor_set(x_497, 1, x_505);
lean_ctor_set(x_497, 0, x_509);
lean_ctor_set(x_495, 1, x_506);
x_513 = l_ReaderT_instMonad___redArg(x_495);
x_514 = !lean_is_exclusive(x_513);
if (x_514 == 0)
{
lean_object* x_515; lean_object* x_516; uint8_t x_517; 
x_515 = lean_ctor_get(x_513, 0);
x_516 = lean_ctor_get(x_513, 1);
lean_dec(x_516);
x_517 = !lean_is_exclusive(x_515);
if (x_517 == 0)
{
lean_object* x_518; lean_object* x_519; lean_object* x_520; lean_object* x_521; lean_object* x_522; lean_object* x_523; lean_object* x_524; lean_object* x_525; lean_object* x_526; lean_object* x_527; lean_object* x_528; lean_object* x_529; lean_object* x_530; lean_object* x_531; lean_object* x_532; lean_object* x_533; uint8_t x_534; 
x_518 = lean_ctor_get(x_515, 0);
x_519 = lean_ctor_get(x_515, 2);
x_520 = lean_ctor_get(x_515, 3);
x_521 = lean_ctor_get(x_515, 4);
x_522 = lean_ctor_get(x_515, 1);
lean_dec(x_522);
x_523 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
x_524 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_525 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_518);
x_526 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_526, 0, x_518);
x_527 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_527, 0, x_518);
x_528 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_528, 0, x_526);
lean_ctor_set(x_528, 1, x_527);
x_529 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_529, 0, x_521);
x_530 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_530, 0, x_520);
x_531 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_531, 0, x_519);
lean_ctor_set(x_515, 4, x_529);
lean_ctor_set(x_515, 3, x_530);
lean_ctor_set(x_515, 2, x_531);
lean_ctor_set(x_515, 1, x_524);
lean_ctor_set(x_515, 0, x_528);
lean_ctor_set(x_513, 1, x_525);
x_532 = l_ReaderT_instMonad___redArg(x_513);
x_533 = l_ReaderT_instMonad___redArg(x_532);
x_534 = !lean_is_exclusive(x_523);
if (x_534 == 0)
{
lean_object* x_535; lean_object* x_536; lean_object* x_537; lean_object* x_538; lean_object* x_539; lean_object* x_540; lean_object* x_541; lean_object* x_542; 
x_535 = lean_ctor_get(x_523, 0);
x_536 = lean_ctor_get(x_523, 1);
lean_dec(x_536);
lean_inc_ref(x_485);
lean_ctor_set(x_523, 1, x_485);
lean_ctor_set(x_523, 0, x_487);
x_537 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_538 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_539 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_539, 0, lean_box(0));
lean_closure_set(x_539, 1, lean_box(0));
lean_closure_set(x_539, 2, x_533);
lean_closure_set(x_539, 3, lean_box(0));
lean_closure_set(x_539, 4, lean_box(0));
lean_closure_set(x_539, 5, x_538);
lean_closure_set(x_539, 6, x_79);
x_540 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_541 = l_Lean_withTraceNode___redArg(x_488, x_489, x_491, x_69, x_493, x_494, x_537, x_535, x_80, x_539, x_2, x_540);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_542 = lean_apply_8(x_541, x_523, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_542) == 0)
{
lean_object* x_543; 
x_543 = lean_ctor_get(x_542, 0);
lean_inc(x_543);
lean_dec_ref(x_542);
x_106 = x_474;
x_107 = x_543;
x_108 = x_3;
x_109 = x_4;
x_110 = x_5;
x_111 = x_6;
x_112 = x_7;
x_113 = x_8;
x_114 = x_9;
x_115 = x_10;
x_116 = lean_box(0);
goto block_213;
}
else
{
uint8_t x_544; 
lean_dec_ref(x_474);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_544 = !lean_is_exclusive(x_542);
if (x_544 == 0)
{
return x_542;
}
else
{
lean_object* x_545; lean_object* x_546; 
x_545 = lean_ctor_get(x_542, 0);
lean_inc(x_545);
lean_dec(x_542);
x_546 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_546, 0, x_545);
return x_546;
}
}
}
else
{
lean_object* x_547; lean_object* x_548; lean_object* x_549; lean_object* x_550; lean_object* x_551; lean_object* x_552; lean_object* x_553; lean_object* x_554; 
x_547 = lean_ctor_get(x_523, 0);
lean_inc(x_547);
lean_dec(x_523);
lean_inc_ref(x_485);
x_548 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_548, 0, x_487);
lean_ctor_set(x_548, 1, x_485);
x_549 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_550 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_551 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_551, 0, lean_box(0));
lean_closure_set(x_551, 1, lean_box(0));
lean_closure_set(x_551, 2, x_533);
lean_closure_set(x_551, 3, lean_box(0));
lean_closure_set(x_551, 4, lean_box(0));
lean_closure_set(x_551, 5, x_550);
lean_closure_set(x_551, 6, x_79);
x_552 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_553 = l_Lean_withTraceNode___redArg(x_488, x_489, x_491, x_69, x_493, x_494, x_549, x_547, x_80, x_551, x_2, x_552);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_554 = lean_apply_8(x_553, x_548, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_554) == 0)
{
lean_object* x_555; 
x_555 = lean_ctor_get(x_554, 0);
lean_inc(x_555);
lean_dec_ref(x_554);
x_106 = x_474;
x_107 = x_555;
x_108 = x_3;
x_109 = x_4;
x_110 = x_5;
x_111 = x_6;
x_112 = x_7;
x_113 = x_8;
x_114 = x_9;
x_115 = x_10;
x_116 = lean_box(0);
goto block_213;
}
else
{
lean_object* x_556; lean_object* x_557; lean_object* x_558; 
lean_dec_ref(x_474);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_556 = lean_ctor_get(x_554, 0);
lean_inc(x_556);
if (lean_is_exclusive(x_554)) {
 lean_ctor_release(x_554, 0);
 x_557 = x_554;
} else {
 lean_dec_ref(x_554);
 x_557 = lean_box(0);
}
if (lean_is_scalar(x_557)) {
 x_558 = lean_alloc_ctor(1, 1, 0);
} else {
 x_558 = x_557;
}
lean_ctor_set(x_558, 0, x_556);
return x_558;
}
}
}
else
{
lean_object* x_559; lean_object* x_560; lean_object* x_561; lean_object* x_562; lean_object* x_563; lean_object* x_564; lean_object* x_565; lean_object* x_566; lean_object* x_567; lean_object* x_568; lean_object* x_569; lean_object* x_570; lean_object* x_571; lean_object* x_572; lean_object* x_573; lean_object* x_574; lean_object* x_575; lean_object* x_576; lean_object* x_577; lean_object* x_578; lean_object* x_579; lean_object* x_580; lean_object* x_581; lean_object* x_582; lean_object* x_583; 
x_559 = lean_ctor_get(x_515, 0);
x_560 = lean_ctor_get(x_515, 2);
x_561 = lean_ctor_get(x_515, 3);
x_562 = lean_ctor_get(x_515, 4);
lean_inc(x_562);
lean_inc(x_561);
lean_inc(x_560);
lean_inc(x_559);
lean_dec(x_515);
x_563 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
x_564 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_565 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_559);
x_566 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_566, 0, x_559);
x_567 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_567, 0, x_559);
x_568 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_568, 0, x_566);
lean_ctor_set(x_568, 1, x_567);
x_569 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_569, 0, x_562);
x_570 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_570, 0, x_561);
x_571 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_571, 0, x_560);
x_572 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_572, 0, x_568);
lean_ctor_set(x_572, 1, x_564);
lean_ctor_set(x_572, 2, x_571);
lean_ctor_set(x_572, 3, x_570);
lean_ctor_set(x_572, 4, x_569);
lean_ctor_set(x_513, 1, x_565);
lean_ctor_set(x_513, 0, x_572);
x_573 = l_ReaderT_instMonad___redArg(x_513);
x_574 = l_ReaderT_instMonad___redArg(x_573);
x_575 = lean_ctor_get(x_563, 0);
lean_inc(x_575);
if (lean_is_exclusive(x_563)) {
 lean_ctor_release(x_563, 0);
 lean_ctor_release(x_563, 1);
 x_576 = x_563;
} else {
 lean_dec_ref(x_563);
 x_576 = lean_box(0);
}
lean_inc_ref(x_485);
if (lean_is_scalar(x_576)) {
 x_577 = lean_alloc_ctor(0, 2, 0);
} else {
 x_577 = x_576;
}
lean_ctor_set(x_577, 0, x_487);
lean_ctor_set(x_577, 1, x_485);
x_578 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_579 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_580 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_580, 0, lean_box(0));
lean_closure_set(x_580, 1, lean_box(0));
lean_closure_set(x_580, 2, x_574);
lean_closure_set(x_580, 3, lean_box(0));
lean_closure_set(x_580, 4, lean_box(0));
lean_closure_set(x_580, 5, x_579);
lean_closure_set(x_580, 6, x_79);
x_581 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_582 = l_Lean_withTraceNode___redArg(x_488, x_489, x_491, x_69, x_493, x_494, x_578, x_575, x_80, x_580, x_2, x_581);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_583 = lean_apply_8(x_582, x_577, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_583) == 0)
{
lean_object* x_584; 
x_584 = lean_ctor_get(x_583, 0);
lean_inc(x_584);
lean_dec_ref(x_583);
x_106 = x_474;
x_107 = x_584;
x_108 = x_3;
x_109 = x_4;
x_110 = x_5;
x_111 = x_6;
x_112 = x_7;
x_113 = x_8;
x_114 = x_9;
x_115 = x_10;
x_116 = lean_box(0);
goto block_213;
}
else
{
lean_object* x_585; lean_object* x_586; lean_object* x_587; 
lean_dec_ref(x_474);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_585 = lean_ctor_get(x_583, 0);
lean_inc(x_585);
if (lean_is_exclusive(x_583)) {
 lean_ctor_release(x_583, 0);
 x_586 = x_583;
} else {
 lean_dec_ref(x_583);
 x_586 = lean_box(0);
}
if (lean_is_scalar(x_586)) {
 x_587 = lean_alloc_ctor(1, 1, 0);
} else {
 x_587 = x_586;
}
lean_ctor_set(x_587, 0, x_585);
return x_587;
}
}
}
else
{
lean_object* x_588; lean_object* x_589; lean_object* x_590; lean_object* x_591; lean_object* x_592; lean_object* x_593; lean_object* x_594; lean_object* x_595; lean_object* x_596; lean_object* x_597; lean_object* x_598; lean_object* x_599; lean_object* x_600; lean_object* x_601; lean_object* x_602; lean_object* x_603; lean_object* x_604; lean_object* x_605; lean_object* x_606; lean_object* x_607; lean_object* x_608; lean_object* x_609; lean_object* x_610; lean_object* x_611; lean_object* x_612; lean_object* x_613; lean_object* x_614; lean_object* x_615; 
x_588 = lean_ctor_get(x_513, 0);
lean_inc(x_588);
lean_dec(x_513);
x_589 = lean_ctor_get(x_588, 0);
lean_inc_ref(x_589);
x_590 = lean_ctor_get(x_588, 2);
lean_inc(x_590);
x_591 = lean_ctor_get(x_588, 3);
lean_inc(x_591);
x_592 = lean_ctor_get(x_588, 4);
lean_inc(x_592);
if (lean_is_exclusive(x_588)) {
 lean_ctor_release(x_588, 0);
 lean_ctor_release(x_588, 1);
 lean_ctor_release(x_588, 2);
 lean_ctor_release(x_588, 3);
 lean_ctor_release(x_588, 4);
 x_593 = x_588;
} else {
 lean_dec_ref(x_588);
 x_593 = lean_box(0);
}
x_594 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
x_595 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_596 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_589);
x_597 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_597, 0, x_589);
x_598 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_598, 0, x_589);
x_599 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_599, 0, x_597);
lean_ctor_set(x_599, 1, x_598);
x_600 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_600, 0, x_592);
x_601 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_601, 0, x_591);
x_602 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_602, 0, x_590);
if (lean_is_scalar(x_593)) {
 x_603 = lean_alloc_ctor(0, 5, 0);
} else {
 x_603 = x_593;
}
lean_ctor_set(x_603, 0, x_599);
lean_ctor_set(x_603, 1, x_595);
lean_ctor_set(x_603, 2, x_602);
lean_ctor_set(x_603, 3, x_601);
lean_ctor_set(x_603, 4, x_600);
x_604 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_604, 0, x_603);
lean_ctor_set(x_604, 1, x_596);
x_605 = l_ReaderT_instMonad___redArg(x_604);
x_606 = l_ReaderT_instMonad___redArg(x_605);
x_607 = lean_ctor_get(x_594, 0);
lean_inc(x_607);
if (lean_is_exclusive(x_594)) {
 lean_ctor_release(x_594, 0);
 lean_ctor_release(x_594, 1);
 x_608 = x_594;
} else {
 lean_dec_ref(x_594);
 x_608 = lean_box(0);
}
lean_inc_ref(x_485);
if (lean_is_scalar(x_608)) {
 x_609 = lean_alloc_ctor(0, 2, 0);
} else {
 x_609 = x_608;
}
lean_ctor_set(x_609, 0, x_487);
lean_ctor_set(x_609, 1, x_485);
x_610 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_611 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_612 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_612, 0, lean_box(0));
lean_closure_set(x_612, 1, lean_box(0));
lean_closure_set(x_612, 2, x_606);
lean_closure_set(x_612, 3, lean_box(0));
lean_closure_set(x_612, 4, lean_box(0));
lean_closure_set(x_612, 5, x_611);
lean_closure_set(x_612, 6, x_79);
x_613 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_614 = l_Lean_withTraceNode___redArg(x_488, x_489, x_491, x_69, x_493, x_494, x_610, x_607, x_80, x_612, x_2, x_613);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_615 = lean_apply_8(x_614, x_609, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_615) == 0)
{
lean_object* x_616; 
x_616 = lean_ctor_get(x_615, 0);
lean_inc(x_616);
lean_dec_ref(x_615);
x_106 = x_474;
x_107 = x_616;
x_108 = x_3;
x_109 = x_4;
x_110 = x_5;
x_111 = x_6;
x_112 = x_7;
x_113 = x_8;
x_114 = x_9;
x_115 = x_10;
x_116 = lean_box(0);
goto block_213;
}
else
{
lean_object* x_617; lean_object* x_618; lean_object* x_619; 
lean_dec_ref(x_474);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_617 = lean_ctor_get(x_615, 0);
lean_inc(x_617);
if (lean_is_exclusive(x_615)) {
 lean_ctor_release(x_615, 0);
 x_618 = x_615;
} else {
 lean_dec_ref(x_615);
 x_618 = lean_box(0);
}
if (lean_is_scalar(x_618)) {
 x_619 = lean_alloc_ctor(1, 1, 0);
} else {
 x_619 = x_618;
}
lean_ctor_set(x_619, 0, x_617);
return x_619;
}
}
}
else
{
lean_object* x_620; lean_object* x_621; lean_object* x_622; lean_object* x_623; lean_object* x_624; lean_object* x_625; lean_object* x_626; lean_object* x_627; lean_object* x_628; lean_object* x_629; lean_object* x_630; lean_object* x_631; lean_object* x_632; lean_object* x_633; lean_object* x_634; lean_object* x_635; lean_object* x_636; lean_object* x_637; lean_object* x_638; lean_object* x_639; lean_object* x_640; lean_object* x_641; lean_object* x_642; lean_object* x_643; lean_object* x_644; lean_object* x_645; lean_object* x_646; lean_object* x_647; lean_object* x_648; lean_object* x_649; lean_object* x_650; lean_object* x_651; lean_object* x_652; lean_object* x_653; lean_object* x_654; lean_object* x_655; lean_object* x_656; lean_object* x_657; lean_object* x_658; lean_object* x_659; lean_object* x_660; lean_object* x_661; lean_object* x_662; 
x_620 = lean_ctor_get(x_497, 0);
x_621 = lean_ctor_get(x_497, 2);
x_622 = lean_ctor_get(x_497, 3);
x_623 = lean_ctor_get(x_497, 4);
lean_inc(x_623);
lean_inc(x_622);
lean_inc(x_621);
lean_inc(x_620);
lean_dec(x_497);
x_624 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_625 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_620);
x_626 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_626, 0, x_620);
x_627 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_627, 0, x_620);
x_628 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_628, 0, x_626);
lean_ctor_set(x_628, 1, x_627);
x_629 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_629, 0, x_623);
x_630 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_630, 0, x_622);
x_631 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_631, 0, x_621);
x_632 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_632, 0, x_628);
lean_ctor_set(x_632, 1, x_624);
lean_ctor_set(x_632, 2, x_631);
lean_ctor_set(x_632, 3, x_630);
lean_ctor_set(x_632, 4, x_629);
lean_ctor_set(x_495, 1, x_625);
lean_ctor_set(x_495, 0, x_632);
x_633 = l_ReaderT_instMonad___redArg(x_495);
x_634 = lean_ctor_get(x_633, 0);
lean_inc_ref(x_634);
if (lean_is_exclusive(x_633)) {
 lean_ctor_release(x_633, 0);
 lean_ctor_release(x_633, 1);
 x_635 = x_633;
} else {
 lean_dec_ref(x_633);
 x_635 = lean_box(0);
}
x_636 = lean_ctor_get(x_634, 0);
lean_inc_ref(x_636);
x_637 = lean_ctor_get(x_634, 2);
lean_inc(x_637);
x_638 = lean_ctor_get(x_634, 3);
lean_inc(x_638);
x_639 = lean_ctor_get(x_634, 4);
lean_inc(x_639);
if (lean_is_exclusive(x_634)) {
 lean_ctor_release(x_634, 0);
 lean_ctor_release(x_634, 1);
 lean_ctor_release(x_634, 2);
 lean_ctor_release(x_634, 3);
 lean_ctor_release(x_634, 4);
 x_640 = x_634;
} else {
 lean_dec_ref(x_634);
 x_640 = lean_box(0);
}
x_641 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
x_642 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_643 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_636);
x_644 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_644, 0, x_636);
x_645 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_645, 0, x_636);
x_646 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_646, 0, x_644);
lean_ctor_set(x_646, 1, x_645);
x_647 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_647, 0, x_639);
x_648 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_648, 0, x_638);
x_649 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_649, 0, x_637);
if (lean_is_scalar(x_640)) {
 x_650 = lean_alloc_ctor(0, 5, 0);
} else {
 x_650 = x_640;
}
lean_ctor_set(x_650, 0, x_646);
lean_ctor_set(x_650, 1, x_642);
lean_ctor_set(x_650, 2, x_649);
lean_ctor_set(x_650, 3, x_648);
lean_ctor_set(x_650, 4, x_647);
if (lean_is_scalar(x_635)) {
 x_651 = lean_alloc_ctor(0, 2, 0);
} else {
 x_651 = x_635;
}
lean_ctor_set(x_651, 0, x_650);
lean_ctor_set(x_651, 1, x_643);
x_652 = l_ReaderT_instMonad___redArg(x_651);
x_653 = l_ReaderT_instMonad___redArg(x_652);
x_654 = lean_ctor_get(x_641, 0);
lean_inc(x_654);
if (lean_is_exclusive(x_641)) {
 lean_ctor_release(x_641, 0);
 lean_ctor_release(x_641, 1);
 x_655 = x_641;
} else {
 lean_dec_ref(x_641);
 x_655 = lean_box(0);
}
lean_inc_ref(x_485);
if (lean_is_scalar(x_655)) {
 x_656 = lean_alloc_ctor(0, 2, 0);
} else {
 x_656 = x_655;
}
lean_ctor_set(x_656, 0, x_487);
lean_ctor_set(x_656, 1, x_485);
x_657 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_658 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_659 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_659, 0, lean_box(0));
lean_closure_set(x_659, 1, lean_box(0));
lean_closure_set(x_659, 2, x_653);
lean_closure_set(x_659, 3, lean_box(0));
lean_closure_set(x_659, 4, lean_box(0));
lean_closure_set(x_659, 5, x_658);
lean_closure_set(x_659, 6, x_79);
x_660 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_661 = l_Lean_withTraceNode___redArg(x_488, x_489, x_491, x_69, x_493, x_494, x_657, x_654, x_80, x_659, x_2, x_660);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_662 = lean_apply_8(x_661, x_656, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_662) == 0)
{
lean_object* x_663; 
x_663 = lean_ctor_get(x_662, 0);
lean_inc(x_663);
lean_dec_ref(x_662);
x_106 = x_474;
x_107 = x_663;
x_108 = x_3;
x_109 = x_4;
x_110 = x_5;
x_111 = x_6;
x_112 = x_7;
x_113 = x_8;
x_114 = x_9;
x_115 = x_10;
x_116 = lean_box(0);
goto block_213;
}
else
{
lean_object* x_664; lean_object* x_665; lean_object* x_666; 
lean_dec_ref(x_474);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_664 = lean_ctor_get(x_662, 0);
lean_inc(x_664);
if (lean_is_exclusive(x_662)) {
 lean_ctor_release(x_662, 0);
 x_665 = x_662;
} else {
 lean_dec_ref(x_662);
 x_665 = lean_box(0);
}
if (lean_is_scalar(x_665)) {
 x_666 = lean_alloc_ctor(1, 1, 0);
} else {
 x_666 = x_665;
}
lean_ctor_set(x_666, 0, x_664);
return x_666;
}
}
}
else
{
lean_object* x_667; lean_object* x_668; lean_object* x_669; lean_object* x_670; lean_object* x_671; lean_object* x_672; lean_object* x_673; lean_object* x_674; lean_object* x_675; lean_object* x_676; lean_object* x_677; lean_object* x_678; lean_object* x_679; lean_object* x_680; lean_object* x_681; lean_object* x_682; lean_object* x_683; lean_object* x_684; lean_object* x_685; lean_object* x_686; lean_object* x_687; lean_object* x_688; lean_object* x_689; lean_object* x_690; lean_object* x_691; lean_object* x_692; lean_object* x_693; lean_object* x_694; lean_object* x_695; lean_object* x_696; lean_object* x_697; lean_object* x_698; lean_object* x_699; lean_object* x_700; lean_object* x_701; lean_object* x_702; lean_object* x_703; lean_object* x_704; lean_object* x_705; lean_object* x_706; lean_object* x_707; lean_object* x_708; lean_object* x_709; lean_object* x_710; lean_object* x_711; lean_object* x_712; 
x_667 = lean_ctor_get(x_495, 0);
lean_inc(x_667);
lean_dec(x_495);
x_668 = lean_ctor_get(x_667, 0);
lean_inc_ref(x_668);
x_669 = lean_ctor_get(x_667, 2);
lean_inc(x_669);
x_670 = lean_ctor_get(x_667, 3);
lean_inc(x_670);
x_671 = lean_ctor_get(x_667, 4);
lean_inc(x_671);
if (lean_is_exclusive(x_667)) {
 lean_ctor_release(x_667, 0);
 lean_ctor_release(x_667, 1);
 lean_ctor_release(x_667, 2);
 lean_ctor_release(x_667, 3);
 lean_ctor_release(x_667, 4);
 x_672 = x_667;
} else {
 lean_dec_ref(x_667);
 x_672 = lean_box(0);
}
x_673 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_674 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_668);
x_675 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_675, 0, x_668);
x_676 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_676, 0, x_668);
x_677 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_677, 0, x_675);
lean_ctor_set(x_677, 1, x_676);
x_678 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_678, 0, x_671);
x_679 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_679, 0, x_670);
x_680 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_680, 0, x_669);
if (lean_is_scalar(x_672)) {
 x_681 = lean_alloc_ctor(0, 5, 0);
} else {
 x_681 = x_672;
}
lean_ctor_set(x_681, 0, x_677);
lean_ctor_set(x_681, 1, x_673);
lean_ctor_set(x_681, 2, x_680);
lean_ctor_set(x_681, 3, x_679);
lean_ctor_set(x_681, 4, x_678);
x_682 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_682, 0, x_681);
lean_ctor_set(x_682, 1, x_674);
x_683 = l_ReaderT_instMonad___redArg(x_682);
x_684 = lean_ctor_get(x_683, 0);
lean_inc_ref(x_684);
if (lean_is_exclusive(x_683)) {
 lean_ctor_release(x_683, 0);
 lean_ctor_release(x_683, 1);
 x_685 = x_683;
} else {
 lean_dec_ref(x_683);
 x_685 = lean_box(0);
}
x_686 = lean_ctor_get(x_684, 0);
lean_inc_ref(x_686);
x_687 = lean_ctor_get(x_684, 2);
lean_inc(x_687);
x_688 = lean_ctor_get(x_684, 3);
lean_inc(x_688);
x_689 = lean_ctor_get(x_684, 4);
lean_inc(x_689);
if (lean_is_exclusive(x_684)) {
 lean_ctor_release(x_684, 0);
 lean_ctor_release(x_684, 1);
 lean_ctor_release(x_684, 2);
 lean_ctor_release(x_684, 3);
 lean_ctor_release(x_684, 4);
 x_690 = x_684;
} else {
 lean_dec_ref(x_684);
 x_690 = lean_box(0);
}
x_691 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
x_692 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_693 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_686);
x_694 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_694, 0, x_686);
x_695 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_695, 0, x_686);
x_696 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_696, 0, x_694);
lean_ctor_set(x_696, 1, x_695);
x_697 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_697, 0, x_689);
x_698 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_698, 0, x_688);
x_699 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_699, 0, x_687);
if (lean_is_scalar(x_690)) {
 x_700 = lean_alloc_ctor(0, 5, 0);
} else {
 x_700 = x_690;
}
lean_ctor_set(x_700, 0, x_696);
lean_ctor_set(x_700, 1, x_692);
lean_ctor_set(x_700, 2, x_699);
lean_ctor_set(x_700, 3, x_698);
lean_ctor_set(x_700, 4, x_697);
if (lean_is_scalar(x_685)) {
 x_701 = lean_alloc_ctor(0, 2, 0);
} else {
 x_701 = x_685;
}
lean_ctor_set(x_701, 0, x_700);
lean_ctor_set(x_701, 1, x_693);
x_702 = l_ReaderT_instMonad___redArg(x_701);
x_703 = l_ReaderT_instMonad___redArg(x_702);
x_704 = lean_ctor_get(x_691, 0);
lean_inc(x_704);
if (lean_is_exclusive(x_691)) {
 lean_ctor_release(x_691, 0);
 lean_ctor_release(x_691, 1);
 x_705 = x_691;
} else {
 lean_dec_ref(x_691);
 x_705 = lean_box(0);
}
lean_inc_ref(x_485);
if (lean_is_scalar(x_705)) {
 x_706 = lean_alloc_ctor(0, 2, 0);
} else {
 x_706 = x_705;
}
lean_ctor_set(x_706, 0, x_487);
lean_ctor_set(x_706, 1, x_485);
x_707 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_708 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_709 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_709, 0, lean_box(0));
lean_closure_set(x_709, 1, lean_box(0));
lean_closure_set(x_709, 2, x_703);
lean_closure_set(x_709, 3, lean_box(0));
lean_closure_set(x_709, 4, lean_box(0));
lean_closure_set(x_709, 5, x_708);
lean_closure_set(x_709, 6, x_79);
x_710 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_711 = l_Lean_withTraceNode___redArg(x_488, x_489, x_491, x_69, x_493, x_494, x_707, x_704, x_80, x_709, x_2, x_710);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_712 = lean_apply_8(x_711, x_706, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_712) == 0)
{
lean_object* x_713; 
x_713 = lean_ctor_get(x_712, 0);
lean_inc(x_713);
lean_dec_ref(x_712);
x_106 = x_474;
x_107 = x_713;
x_108 = x_3;
x_109 = x_4;
x_110 = x_5;
x_111 = x_6;
x_112 = x_7;
x_113 = x_8;
x_114 = x_9;
x_115 = x_10;
x_116 = lean_box(0);
goto block_213;
}
else
{
lean_object* x_714; lean_object* x_715; lean_object* x_716; 
lean_dec_ref(x_474);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_714 = lean_ctor_get(x_712, 0);
lean_inc(x_714);
if (lean_is_exclusive(x_712)) {
 lean_ctor_release(x_712, 0);
 x_715 = x_712;
} else {
 lean_dec_ref(x_712);
 x_715 = lean_box(0);
}
if (lean_is_scalar(x_715)) {
 x_716 = lean_alloc_ctor(1, 1, 0);
} else {
 x_716 = x_715;
}
lean_ctor_set(x_716, 0, x_714);
return x_716;
}
}
}
}
else
{
lean_dec_ref(x_474);
lean_dec_ref(x_80);
lean_dec_ref(x_79);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_60 = lean_box(0);
goto block_63;
}
}
}
block_725:
{
if (lean_obj_tag(x_718) == 0)
{
lean_object* x_719; uint8_t x_720; 
x_719 = lean_ctor_get(x_718, 0);
lean_inc(x_719);
lean_dec_ref(x_718);
x_720 = lean_unbox(x_719);
if (x_720 == 0)
{
uint8_t x_721; 
x_721 = lean_unbox(x_719);
lean_dec(x_719);
x_472 = x_721;
x_473 = lean_box(0);
goto block_717;
}
else
{
lean_dec(x_719);
x_214 = lean_box(0);
goto block_468;
}
}
else
{
uint8_t x_722; 
lean_dec_ref(x_80);
lean_dec_ref(x_79);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_722 = !lean_is_exclusive(x_718);
if (x_722 == 0)
{
return x_718;
}
else
{
lean_object* x_723; lean_object* x_724; 
x_723 = lean_ctor_get(x_718, 0);
lean_inc(x_723);
lean_dec(x_718);
x_724 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_724, 0, x_723);
return x_724;
}
}
}
}
else
{
x_214 = lean_box(0);
goto block_468;
}
block_55:
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; uint8_t x_18; 
x_15 = lean_st_ref_get(x_4);
lean_dec(x_4);
lean_dec(x_15);
x_16 = lean_io_mono_nanos_now();
x_17 = lean_st_ref_take(x_6);
x_18 = !lean_is_exclusive(x_17);
if (x_18 == 0)
{
lean_object* x_19; uint8_t x_20; 
x_19 = lean_ctor_get(x_17, 1);
x_20 = !lean_is_exclusive(x_19);
if (x_20 == 0)
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_21 = lean_ctor_get(x_19, 5);
lean_dec(x_21);
x_22 = lean_nat_sub(x_16, x_12);
lean_dec(x_12);
lean_dec(x_16);
lean_ctor_set(x_19, 5, x_22);
x_23 = lean_st_ref_set(x_6, x_17);
lean_dec(x_6);
x_24 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_24, 0, x_13);
return x_24;
}
else
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_25 = lean_ctor_get(x_19, 0);
x_26 = lean_ctor_get(x_19, 1);
x_27 = lean_ctor_get(x_19, 2);
x_28 = lean_ctor_get(x_19, 3);
x_29 = lean_ctor_get(x_19, 4);
x_30 = lean_ctor_get(x_19, 6);
x_31 = lean_ctor_get(x_19, 7);
x_32 = lean_ctor_get(x_19, 8);
x_33 = lean_ctor_get(x_19, 9);
lean_inc(x_33);
lean_inc(x_32);
lean_inc(x_31);
lean_inc(x_30);
lean_inc(x_29);
lean_inc(x_28);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_25);
lean_dec(x_19);
x_34 = lean_nat_sub(x_16, x_12);
lean_dec(x_12);
lean_dec(x_16);
x_35 = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(x_35, 0, x_25);
lean_ctor_set(x_35, 1, x_26);
lean_ctor_set(x_35, 2, x_27);
lean_ctor_set(x_35, 3, x_28);
lean_ctor_set(x_35, 4, x_29);
lean_ctor_set(x_35, 5, x_34);
lean_ctor_set(x_35, 6, x_30);
lean_ctor_set(x_35, 7, x_31);
lean_ctor_set(x_35, 8, x_32);
lean_ctor_set(x_35, 9, x_33);
lean_ctor_set(x_17, 1, x_35);
x_36 = lean_st_ref_set(x_6, x_17);
lean_dec(x_6);
x_37 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_37, 0, x_13);
return x_37;
}
}
else
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; 
x_38 = lean_ctor_get(x_17, 1);
x_39 = lean_ctor_get(x_17, 0);
lean_inc(x_38);
lean_inc(x_39);
lean_dec(x_17);
x_40 = lean_ctor_get(x_38, 0);
lean_inc(x_40);
x_41 = lean_ctor_get(x_38, 1);
lean_inc(x_41);
x_42 = lean_ctor_get(x_38, 2);
lean_inc(x_42);
x_43 = lean_ctor_get(x_38, 3);
lean_inc(x_43);
x_44 = lean_ctor_get(x_38, 4);
lean_inc(x_44);
x_45 = lean_ctor_get(x_38, 6);
lean_inc(x_45);
x_46 = lean_ctor_get(x_38, 7);
lean_inc(x_46);
x_47 = lean_ctor_get(x_38, 8);
lean_inc_ref(x_47);
x_48 = lean_ctor_get(x_38, 9);
lean_inc_ref(x_48);
if (lean_is_exclusive(x_38)) {
 lean_ctor_release(x_38, 0);
 lean_ctor_release(x_38, 1);
 lean_ctor_release(x_38, 2);
 lean_ctor_release(x_38, 3);
 lean_ctor_release(x_38, 4);
 lean_ctor_release(x_38, 5);
 lean_ctor_release(x_38, 6);
 lean_ctor_release(x_38, 7);
 lean_ctor_release(x_38, 8);
 lean_ctor_release(x_38, 9);
 x_49 = x_38;
} else {
 lean_dec_ref(x_38);
 x_49 = lean_box(0);
}
x_50 = lean_nat_sub(x_16, x_12);
lean_dec(x_12);
lean_dec(x_16);
if (lean_is_scalar(x_49)) {
 x_51 = lean_alloc_ctor(0, 10, 0);
} else {
 x_51 = x_49;
}
lean_ctor_set(x_51, 0, x_40);
lean_ctor_set(x_51, 1, x_41);
lean_ctor_set(x_51, 2, x_42);
lean_ctor_set(x_51, 3, x_43);
lean_ctor_set(x_51, 4, x_44);
lean_ctor_set(x_51, 5, x_50);
lean_ctor_set(x_51, 6, x_45);
lean_ctor_set(x_51, 7, x_46);
lean_ctor_set(x_51, 8, x_47);
lean_ctor_set(x_51, 9, x_48);
x_52 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_52, 0, x_39);
lean_ctor_set(x_52, 1, x_51);
x_53 = lean_st_ref_set(x_6, x_52);
lean_dec(x_6);
x_54 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_54, 0, x_13);
return x_54;
}
}
block_59:
{
if (lean_obj_tag(x_57) == 0)
{
lean_object* x_58; 
x_58 = lean_ctor_get(x_57, 0);
lean_inc(x_58);
lean_dec_ref(x_57);
x_12 = x_56;
x_13 = x_58;
x_14 = lean_box(0);
goto block_55;
}
else
{
lean_dec(x_56);
lean_dec(x_6);
lean_dec(x_4);
return x_57;
}
}
block_63:
{
lean_object* x_61; lean_object* x_62; 
x_61 = lean_box(0);
x_62 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_62, 0, x_61);
return x_62;
}
block_105:
{
lean_object* x_96; lean_object* x_97; 
x_96 = lean_st_ref_get(x_88);
lean_dec(x_96);
lean_inc(x_94);
lean_inc_ref(x_93);
lean_inc(x_92);
lean_inc_ref(x_91);
lean_inc(x_84);
lean_inc_ref(x_82);
lean_inc_ref(x_85);
x_97 = lp_aesop_Aesop_Script_UScript_optimize(x_85, x_86, x_82, x_84, x_91, x_92, x_93, x_94);
if (lean_obj_tag(x_97) == 0)
{
lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; 
x_98 = lean_ctor_get(x_97, 0);
lean_inc(x_98);
lean_dec_ref(x_97);
x_99 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__0;
x_100 = lp_aesop_Aesop_checkAndTraceScript___redArg(x_66, x_71, x_72, x_76, x_74, x_67, x_81, x_85, x_98, x_82, x_84, x_83, x_2, x_99);
x_101 = lean_apply_9(x_100, x_87, x_88, x_89, x_90, x_91, x_92, x_93, x_94, lean_box(0));
return x_101;
}
else
{
uint8_t x_102; 
lean_dec(x_94);
lean_dec_ref(x_93);
lean_dec(x_92);
lean_dec_ref(x_91);
lean_dec(x_90);
lean_dec(x_89);
lean_dec(x_88);
lean_dec_ref(x_87);
lean_dec_ref(x_85);
lean_dec(x_84);
lean_dec_ref(x_83);
lean_dec_ref(x_82);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec_ref(x_66);
x_102 = !lean_is_exclusive(x_97);
if (x_102 == 0)
{
return x_97;
}
else
{
lean_object* x_103; lean_object* x_104; 
x_103 = lean_ctor_get(x_97, 0);
lean_inc(x_103);
lean_dec(x_97);
x_104 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_104, 0, x_103);
return x_104;
}
}
}
block_213:
{
uint8_t x_117; 
x_117 = !lean_is_exclusive(x_107);
if (x_117 == 0)
{
lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; 
x_118 = lean_ctor_get(x_107, 0);
x_119 = lean_ctor_get(x_107, 1);
x_120 = lean_st_ref_get(x_109);
lean_dec(x_120);
lean_inc(x_115);
lean_inc_ref(x_114);
lean_inc(x_113);
lean_inc_ref(x_112);
x_121 = lp_aesop_Aesop_Script_UScript_checkIfEnabled(x_118, x_112, x_113, x_114, x_115);
if (lean_obj_tag(x_121) == 0)
{
lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; 
lean_dec_ref(x_121);
x_122 = lean_st_ref_get(x_109);
x_123 = lean_ctor_get(x_122, 0);
lean_inc(x_123);
lean_dec(x_122);
x_124 = lean_ctor_get(x_108, 0);
lean_inc_ref(x_124);
lean_ctor_set(x_107, 1, x_124);
lean_ctor_set(x_107, 0, x_123);
x_125 = lp_aesop_Aesop_getRootMVarId(x_107, x_110, x_111, x_112, x_113, x_114, x_115);
lean_dec_ref(x_107);
if (lean_obj_tag(x_125) == 0)
{
lean_object* x_126; lean_object* x_127; lean_object* x_128; 
x_126 = lean_ctor_get(x_125, 0);
lean_inc(x_126);
lean_dec_ref(x_125);
x_127 = lean_st_ref_get(x_109);
lean_dec(x_127);
x_128 = lp_aesop_Aesop_getRootMetaState___redArg(x_110);
if (lean_obj_tag(x_128) == 0)
{
lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; 
x_129 = lean_ctor_get(x_128, 0);
lean_inc(x_129);
lean_dec_ref(x_128);
x_130 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
lean_inc_ref(x_66);
x_131 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_66, x_68, x_130);
lean_inc(x_115);
lean_inc_ref(x_114);
lean_inc(x_113);
lean_inc_ref(x_112);
lean_inc(x_111);
lean_inc(x_110);
lean_inc(x_109);
lean_inc_ref(x_108);
x_132 = lean_apply_9(x_131, x_108, x_109, x_110, x_111, x_112, x_113, x_114, x_115, lean_box(0));
if (lean_obj_tag(x_132) == 0)
{
lean_object* x_133; uint8_t x_134; 
x_133 = lean_ctor_get(x_132, 0);
lean_inc(x_133);
lean_dec_ref(x_132);
x_134 = lean_unbox(x_133);
lean_dec(x_133);
if (x_134 == 0)
{
uint8_t x_135; 
x_135 = lean_unbox(x_119);
lean_dec(x_119);
x_82 = x_129;
x_83 = x_106;
x_84 = x_126;
x_85 = x_118;
x_86 = x_135;
x_87 = x_108;
x_88 = x_109;
x_89 = x_110;
x_90 = x_111;
x_91 = x_112;
x_92 = x_113;
x_93 = x_114;
x_94 = x_115;
x_95 = lean_box(0);
goto block_105;
}
else
{
lean_object* x_136; lean_object* x_137; 
x_136 = lean_st_ref_get(x_109);
lean_dec(x_136);
lean_inc(x_115);
lean_inc_ref(x_114);
lean_inc(x_113);
lean_inc_ref(x_112);
lean_inc(x_126);
lean_inc(x_129);
x_137 = lp_aesop_Aesop_Script_UScript_renderTacticSeq(x_118, x_129, x_126, x_112, x_113, x_114, x_115);
if (lean_obj_tag(x_137) == 0)
{
lean_object* x_138; lean_object* x_139; uint8_t x_140; 
x_138 = lean_ctor_get(x_137, 0);
lean_inc(x_138);
lean_dec_ref(x_137);
x_139 = lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
x_140 = !lean_is_exclusive(x_130);
if (x_140 == 0)
{
lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; 
x_141 = lean_ctor_get(x_130, 0);
x_142 = lean_ctor_get(x_130, 1);
lean_dec(x_142);
x_143 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__3;
x_144 = l_Lean_MessageData_ofSyntax(x_138);
x_145 = l_Lean_indentD(x_144);
lean_ctor_set_tag(x_130, 7);
lean_ctor_set(x_130, 1, x_145);
lean_ctor_set(x_130, 0, x_143);
lean_inc_ref(x_72);
lean_inc_ref(x_66);
x_146 = l_Lean_addTrace___redArg(x_66, x_139, x_72, x_74, x_141, x_130);
lean_inc(x_115);
lean_inc_ref(x_114);
lean_inc(x_113);
lean_inc_ref(x_112);
lean_inc(x_111);
lean_inc(x_110);
lean_inc(x_109);
lean_inc_ref(x_108);
x_147 = lean_apply_9(x_146, x_108, x_109, x_110, x_111, x_112, x_113, x_114, x_115, lean_box(0));
if (lean_obj_tag(x_147) == 0)
{
uint8_t x_148; 
lean_dec_ref(x_147);
x_148 = lean_unbox(x_119);
lean_dec(x_119);
x_82 = x_129;
x_83 = x_106;
x_84 = x_126;
x_85 = x_118;
x_86 = x_148;
x_87 = x_108;
x_88 = x_109;
x_89 = x_110;
x_90 = x_111;
x_91 = x_112;
x_92 = x_113;
x_93 = x_114;
x_94 = x_115;
x_95 = lean_box(0);
goto block_105;
}
else
{
lean_dec(x_129);
lean_dec(x_126);
lean_dec(x_119);
lean_dec(x_118);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec_ref(x_66);
return x_147;
}
}
else
{
lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; 
x_149 = lean_ctor_get(x_130, 0);
lean_inc(x_149);
lean_dec(x_130);
x_150 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__3;
x_151 = l_Lean_MessageData_ofSyntax(x_138);
x_152 = l_Lean_indentD(x_151);
x_153 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_153, 0, x_150);
lean_ctor_set(x_153, 1, x_152);
lean_inc_ref(x_72);
lean_inc_ref(x_66);
x_154 = l_Lean_addTrace___redArg(x_66, x_139, x_72, x_74, x_149, x_153);
lean_inc(x_115);
lean_inc_ref(x_114);
lean_inc(x_113);
lean_inc_ref(x_112);
lean_inc(x_111);
lean_inc(x_110);
lean_inc(x_109);
lean_inc_ref(x_108);
x_155 = lean_apply_9(x_154, x_108, x_109, x_110, x_111, x_112, x_113, x_114, x_115, lean_box(0));
if (lean_obj_tag(x_155) == 0)
{
uint8_t x_156; 
lean_dec_ref(x_155);
x_156 = lean_unbox(x_119);
lean_dec(x_119);
x_82 = x_129;
x_83 = x_106;
x_84 = x_126;
x_85 = x_118;
x_86 = x_156;
x_87 = x_108;
x_88 = x_109;
x_89 = x_110;
x_90 = x_111;
x_91 = x_112;
x_92 = x_113;
x_93 = x_114;
x_94 = x_115;
x_95 = lean_box(0);
goto block_105;
}
else
{
lean_dec(x_129);
lean_dec(x_126);
lean_dec(x_119);
lean_dec(x_118);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec_ref(x_66);
return x_155;
}
}
}
else
{
uint8_t x_157; 
lean_dec(x_129);
lean_dec(x_126);
lean_dec(x_119);
lean_dec(x_118);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec_ref(x_66);
x_157 = !lean_is_exclusive(x_137);
if (x_157 == 0)
{
return x_137;
}
else
{
lean_object* x_158; lean_object* x_159; 
x_158 = lean_ctor_get(x_137, 0);
lean_inc(x_158);
lean_dec(x_137);
x_159 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_159, 0, x_158);
return x_159;
}
}
}
}
else
{
uint8_t x_160; 
lean_dec(x_129);
lean_dec(x_126);
lean_dec(x_119);
lean_dec(x_118);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec_ref(x_66);
x_160 = !lean_is_exclusive(x_132);
if (x_160 == 0)
{
return x_132;
}
else
{
lean_object* x_161; lean_object* x_162; 
x_161 = lean_ctor_get(x_132, 0);
lean_inc(x_161);
lean_dec(x_132);
x_162 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_162, 0, x_161);
return x_162;
}
}
}
else
{
uint8_t x_163; 
lean_dec(x_126);
lean_dec(x_119);
lean_dec(x_118);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
x_163 = !lean_is_exclusive(x_128);
if (x_163 == 0)
{
return x_128;
}
else
{
lean_object* x_164; lean_object* x_165; 
x_164 = lean_ctor_get(x_128, 0);
lean_inc(x_164);
lean_dec(x_128);
x_165 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_165, 0, x_164);
return x_165;
}
}
}
else
{
uint8_t x_166; 
lean_dec(x_119);
lean_dec(x_118);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
x_166 = !lean_is_exclusive(x_125);
if (x_166 == 0)
{
return x_125;
}
else
{
lean_object* x_167; lean_object* x_168; 
x_167 = lean_ctor_get(x_125, 0);
lean_inc(x_167);
lean_dec(x_125);
x_168 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_168, 0, x_167);
return x_168;
}
}
}
else
{
lean_free_object(x_107);
lean_dec(x_119);
lean_dec(x_118);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
return x_121;
}
}
else
{
lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; 
x_169 = lean_ctor_get(x_107, 0);
x_170 = lean_ctor_get(x_107, 1);
lean_inc(x_170);
lean_inc(x_169);
lean_dec(x_107);
x_171 = lean_st_ref_get(x_109);
lean_dec(x_171);
lean_inc(x_115);
lean_inc_ref(x_114);
lean_inc(x_113);
lean_inc_ref(x_112);
x_172 = lp_aesop_Aesop_Script_UScript_checkIfEnabled(x_169, x_112, x_113, x_114, x_115);
if (lean_obj_tag(x_172) == 0)
{
lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; 
lean_dec_ref(x_172);
x_173 = lean_st_ref_get(x_109);
x_174 = lean_ctor_get(x_173, 0);
lean_inc(x_174);
lean_dec(x_173);
x_175 = lean_ctor_get(x_108, 0);
lean_inc_ref(x_175);
x_176 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_176, 0, x_174);
lean_ctor_set(x_176, 1, x_175);
x_177 = lp_aesop_Aesop_getRootMVarId(x_176, x_110, x_111, x_112, x_113, x_114, x_115);
lean_dec_ref(x_176);
if (lean_obj_tag(x_177) == 0)
{
lean_object* x_178; lean_object* x_179; lean_object* x_180; 
x_178 = lean_ctor_get(x_177, 0);
lean_inc(x_178);
lean_dec_ref(x_177);
x_179 = lean_st_ref_get(x_109);
lean_dec(x_179);
x_180 = lp_aesop_Aesop_getRootMetaState___redArg(x_110);
if (lean_obj_tag(x_180) == 0)
{
lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; 
x_181 = lean_ctor_get(x_180, 0);
lean_inc(x_181);
lean_dec_ref(x_180);
x_182 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
lean_inc_ref(x_66);
x_183 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_66, x_68, x_182);
lean_inc(x_115);
lean_inc_ref(x_114);
lean_inc(x_113);
lean_inc_ref(x_112);
lean_inc(x_111);
lean_inc(x_110);
lean_inc(x_109);
lean_inc_ref(x_108);
x_184 = lean_apply_9(x_183, x_108, x_109, x_110, x_111, x_112, x_113, x_114, x_115, lean_box(0));
if (lean_obj_tag(x_184) == 0)
{
lean_object* x_185; uint8_t x_186; 
x_185 = lean_ctor_get(x_184, 0);
lean_inc(x_185);
lean_dec_ref(x_184);
x_186 = lean_unbox(x_185);
lean_dec(x_185);
if (x_186 == 0)
{
uint8_t x_187; 
x_187 = lean_unbox(x_170);
lean_dec(x_170);
x_82 = x_181;
x_83 = x_106;
x_84 = x_178;
x_85 = x_169;
x_86 = x_187;
x_87 = x_108;
x_88 = x_109;
x_89 = x_110;
x_90 = x_111;
x_91 = x_112;
x_92 = x_113;
x_93 = x_114;
x_94 = x_115;
x_95 = lean_box(0);
goto block_105;
}
else
{
lean_object* x_188; lean_object* x_189; 
x_188 = lean_st_ref_get(x_109);
lean_dec(x_188);
lean_inc(x_115);
lean_inc_ref(x_114);
lean_inc(x_113);
lean_inc_ref(x_112);
lean_inc(x_178);
lean_inc(x_181);
x_189 = lp_aesop_Aesop_Script_UScript_renderTacticSeq(x_169, x_181, x_178, x_112, x_113, x_114, x_115);
if (lean_obj_tag(x_189) == 0)
{
lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; 
x_190 = lean_ctor_get(x_189, 0);
lean_inc(x_190);
lean_dec_ref(x_189);
x_191 = lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
x_192 = lean_ctor_get(x_182, 0);
lean_inc(x_192);
if (lean_is_exclusive(x_182)) {
 lean_ctor_release(x_182, 0);
 lean_ctor_release(x_182, 1);
 x_193 = x_182;
} else {
 lean_dec_ref(x_182);
 x_193 = lean_box(0);
}
x_194 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__3;
x_195 = l_Lean_MessageData_ofSyntax(x_190);
x_196 = l_Lean_indentD(x_195);
if (lean_is_scalar(x_193)) {
 x_197 = lean_alloc_ctor(7, 2, 0);
} else {
 x_197 = x_193;
 lean_ctor_set_tag(x_197, 7);
}
lean_ctor_set(x_197, 0, x_194);
lean_ctor_set(x_197, 1, x_196);
lean_inc_ref(x_72);
lean_inc_ref(x_66);
x_198 = l_Lean_addTrace___redArg(x_66, x_191, x_72, x_74, x_192, x_197);
lean_inc(x_115);
lean_inc_ref(x_114);
lean_inc(x_113);
lean_inc_ref(x_112);
lean_inc(x_111);
lean_inc(x_110);
lean_inc(x_109);
lean_inc_ref(x_108);
x_199 = lean_apply_9(x_198, x_108, x_109, x_110, x_111, x_112, x_113, x_114, x_115, lean_box(0));
if (lean_obj_tag(x_199) == 0)
{
uint8_t x_200; 
lean_dec_ref(x_199);
x_200 = lean_unbox(x_170);
lean_dec(x_170);
x_82 = x_181;
x_83 = x_106;
x_84 = x_178;
x_85 = x_169;
x_86 = x_200;
x_87 = x_108;
x_88 = x_109;
x_89 = x_110;
x_90 = x_111;
x_91 = x_112;
x_92 = x_113;
x_93 = x_114;
x_94 = x_115;
x_95 = lean_box(0);
goto block_105;
}
else
{
lean_dec(x_181);
lean_dec(x_178);
lean_dec(x_170);
lean_dec(x_169);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec_ref(x_66);
return x_199;
}
}
else
{
lean_object* x_201; lean_object* x_202; lean_object* x_203; 
lean_dec(x_181);
lean_dec(x_178);
lean_dec(x_170);
lean_dec(x_169);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec_ref(x_66);
x_201 = lean_ctor_get(x_189, 0);
lean_inc(x_201);
if (lean_is_exclusive(x_189)) {
 lean_ctor_release(x_189, 0);
 x_202 = x_189;
} else {
 lean_dec_ref(x_189);
 x_202 = lean_box(0);
}
if (lean_is_scalar(x_202)) {
 x_203 = lean_alloc_ctor(1, 1, 0);
} else {
 x_203 = x_202;
}
lean_ctor_set(x_203, 0, x_201);
return x_203;
}
}
}
else
{
lean_object* x_204; lean_object* x_205; lean_object* x_206; 
lean_dec(x_181);
lean_dec(x_178);
lean_dec(x_170);
lean_dec(x_169);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec_ref(x_66);
x_204 = lean_ctor_get(x_184, 0);
lean_inc(x_204);
if (lean_is_exclusive(x_184)) {
 lean_ctor_release(x_184, 0);
 x_205 = x_184;
} else {
 lean_dec_ref(x_184);
 x_205 = lean_box(0);
}
if (lean_is_scalar(x_205)) {
 x_206 = lean_alloc_ctor(1, 1, 0);
} else {
 x_206 = x_205;
}
lean_ctor_set(x_206, 0, x_204);
return x_206;
}
}
else
{
lean_object* x_207; lean_object* x_208; lean_object* x_209; 
lean_dec(x_178);
lean_dec(x_170);
lean_dec(x_169);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
x_207 = lean_ctor_get(x_180, 0);
lean_inc(x_207);
if (lean_is_exclusive(x_180)) {
 lean_ctor_release(x_180, 0);
 x_208 = x_180;
} else {
 lean_dec_ref(x_180);
 x_208 = lean_box(0);
}
if (lean_is_scalar(x_208)) {
 x_209 = lean_alloc_ctor(1, 1, 0);
} else {
 x_209 = x_208;
}
lean_ctor_set(x_209, 0, x_207);
return x_209;
}
}
else
{
lean_object* x_210; lean_object* x_211; lean_object* x_212; 
lean_dec(x_170);
lean_dec(x_169);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
x_210 = lean_ctor_get(x_177, 0);
lean_inc(x_210);
if (lean_is_exclusive(x_177)) {
 lean_ctor_release(x_177, 0);
 x_211 = x_177;
} else {
 lean_dec_ref(x_177);
 x_211 = lean_box(0);
}
if (lean_is_scalar(x_211)) {
 x_212 = lean_alloc_ctor(1, 1, 0);
} else {
 x_212 = x_211;
}
lean_ctor_set(x_212, 0, x_210);
return x_212;
}
}
else
{
lean_dec(x_170);
lean_dec(x_169);
lean_dec(x_115);
lean_dec_ref(x_114);
lean_dec(x_113);
lean_dec_ref(x_112);
lean_dec(x_111);
lean_dec(x_110);
lean_dec(x_109);
lean_dec_ref(x_108);
lean_dec_ref(x_106);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
return x_172;
}
}
}
block_468:
{
lean_object* x_215; lean_object* x_216; lean_object* x_217; uint8_t x_218; 
x_215 = lean_st_ref_get(x_4);
lean_dec(x_215);
x_216 = lean_io_mono_nanos_now();
x_217 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_217);
x_218 = lean_ctor_get_uint8(x_217, sizeof(void*)*2);
if (x_218 == 0)
{
lean_object* x_219; 
lean_dec_ref(x_217);
lean_dec_ref(x_80);
lean_dec_ref(x_79);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
x_219 = lean_box(0);
x_12 = x_216;
x_13 = x_219;
x_14 = lean_box(0);
goto block_55;
}
else
{
if (x_2 == 0)
{
lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; 
lean_dec_ref(x_80);
lean_dec_ref(x_79);
x_220 = lean_ctor_get(x_3, 0);
x_221 = lean_st_ref_get(x_4);
x_222 = lean_ctor_get(x_221, 0);
lean_inc(x_222);
lean_dec(x_221);
lean_inc_ref(x_220);
x_223 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_223, 0, x_222);
lean_ctor_set(x_223, 1, x_220);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_224 = lp_aesop_Aesop_extractSafePrefixScript(x_223, x_5, x_6, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_224) == 0)
{
lean_object* x_225; lean_object* x_226; 
x_225 = lean_ctor_get(x_224, 0);
lean_inc(x_225);
lean_dec_ref(x_224);
lean_inc(x_6);
lean_inc(x_4);
x_226 = lp_aesop_Aesop_traceScript___redArg___lam__2(x_66, x_68, x_71, x_72, x_76, x_74, x_67, x_81, x_217, x_2, x_65, x_64, x_65, x_65, x_64, x_70, x_225, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
x_56 = x_216;
x_57 = x_226;
goto block_59;
}
else
{
uint8_t x_227; 
lean_dec_ref(x_217);
lean_dec(x_216);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_227 = !lean_is_exclusive(x_224);
if (x_227 == 0)
{
return x_224;
}
else
{
lean_object* x_228; lean_object* x_229; 
x_228 = lean_ctor_get(x_224, 0);
lean_inc(x_228);
lean_dec(x_224);
x_229 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_229, 0, x_228);
return x_229;
}
}
}
else
{
lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; lean_object* x_236; lean_object* x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; uint8_t x_241; 
x_230 = lean_ctor_get(x_3, 0);
x_231 = lean_st_ref_get(x_4);
x_232 = lean_ctor_get(x_231, 0);
lean_inc(x_232);
lean_dec(x_231);
x_233 = lp_aesop_Aesop_TreeM_instMonad;
x_234 = lp_aesop_Aesop_expandNextGoal___redArg___closed__2;
x_235 = lp_aesop_Aesop_traceScript___redArg___closed__12;
x_236 = lean_ctor_get(x_235, 0);
lean_inc_ref(x_236);
x_237 = lp_aesop_Aesop_traceScript___redArg___closed__13;
x_238 = lean_ctor_get(x_237, 0);
lean_inc(x_238);
x_239 = lp_aesop_Aesop_traceScript___redArg___closed__14;
x_240 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1;
x_241 = !lean_is_exclusive(x_240);
if (x_241 == 0)
{
lean_object* x_242; lean_object* x_243; uint8_t x_244; 
x_242 = lean_ctor_get(x_240, 0);
x_243 = lean_ctor_get(x_240, 1);
lean_dec(x_243);
x_244 = !lean_is_exclusive(x_242);
if (x_244 == 0)
{
lean_object* x_245; lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; uint8_t x_259; 
x_245 = lean_ctor_get(x_242, 0);
x_246 = lean_ctor_get(x_242, 2);
x_247 = lean_ctor_get(x_242, 3);
x_248 = lean_ctor_get(x_242, 4);
x_249 = lean_ctor_get(x_242, 1);
lean_dec(x_249);
x_250 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_251 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_245);
x_252 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_252, 0, x_245);
x_253 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_253, 0, x_245);
x_254 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_254, 0, x_252);
lean_ctor_set(x_254, 1, x_253);
x_255 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_255, 0, x_248);
x_256 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_256, 0, x_247);
x_257 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_257, 0, x_246);
lean_ctor_set(x_242, 4, x_255);
lean_ctor_set(x_242, 3, x_256);
lean_ctor_set(x_242, 2, x_257);
lean_ctor_set(x_242, 1, x_250);
lean_ctor_set(x_242, 0, x_254);
lean_ctor_set(x_240, 1, x_251);
x_258 = l_ReaderT_instMonad___redArg(x_240);
x_259 = !lean_is_exclusive(x_258);
if (x_259 == 0)
{
lean_object* x_260; lean_object* x_261; uint8_t x_262; 
x_260 = lean_ctor_get(x_258, 0);
x_261 = lean_ctor_get(x_258, 1);
lean_dec(x_261);
x_262 = !lean_is_exclusive(x_260);
if (x_262 == 0)
{
lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; lean_object* x_269; lean_object* x_270; lean_object* x_271; lean_object* x_272; lean_object* x_273; lean_object* x_274; lean_object* x_275; lean_object* x_276; lean_object* x_277; lean_object* x_278; uint8_t x_279; 
x_263 = lean_ctor_get(x_260, 0);
x_264 = lean_ctor_get(x_260, 2);
x_265 = lean_ctor_get(x_260, 3);
x_266 = lean_ctor_get(x_260, 4);
x_267 = lean_ctor_get(x_260, 1);
lean_dec(x_267);
x_268 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
x_269 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_270 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_263);
x_271 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_271, 0, x_263);
x_272 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_272, 0, x_263);
x_273 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_273, 0, x_271);
lean_ctor_set(x_273, 1, x_272);
x_274 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_274, 0, x_266);
x_275 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_275, 0, x_265);
x_276 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_276, 0, x_264);
lean_ctor_set(x_260, 4, x_274);
lean_ctor_set(x_260, 3, x_275);
lean_ctor_set(x_260, 2, x_276);
lean_ctor_set(x_260, 1, x_269);
lean_ctor_set(x_260, 0, x_273);
lean_ctor_set(x_258, 1, x_270);
x_277 = l_ReaderT_instMonad___redArg(x_258);
x_278 = l_ReaderT_instMonad___redArg(x_277);
x_279 = !lean_is_exclusive(x_268);
if (x_279 == 0)
{
lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; 
x_280 = lean_ctor_get(x_268, 0);
x_281 = lean_ctor_get(x_268, 1);
lean_dec(x_281);
lean_inc_ref(x_230);
lean_ctor_set(x_268, 1, x_230);
lean_ctor_set(x_268, 0, x_232);
x_282 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_283 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_284 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_284, 0, lean_box(0));
lean_closure_set(x_284, 1, lean_box(0));
lean_closure_set(x_284, 2, x_278);
lean_closure_set(x_284, 3, lean_box(0));
lean_closure_set(x_284, 4, lean_box(0));
lean_closure_set(x_284, 5, x_283);
lean_closure_set(x_284, 6, x_79);
x_285 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_286 = l_Lean_withTraceNode___redArg(x_233, x_234, x_236, x_69, x_238, x_239, x_282, x_280, x_80, x_284, x_2, x_285);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_287 = lean_apply_8(x_286, x_268, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_287) == 0)
{
lean_object* x_288; lean_object* x_289; 
x_288 = lean_ctor_get(x_287, 0);
lean_inc(x_288);
lean_dec_ref(x_287);
lean_inc(x_6);
lean_inc(x_4);
x_289 = lp_aesop_Aesop_traceScript___redArg___lam__2(x_66, x_68, x_71, x_72, x_76, x_74, x_67, x_81, x_217, x_2, x_65, x_64, x_65, x_65, x_64, x_70, x_288, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
x_56 = x_216;
x_57 = x_289;
goto block_59;
}
else
{
uint8_t x_290; 
lean_dec_ref(x_217);
lean_dec(x_216);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_290 = !lean_is_exclusive(x_287);
if (x_290 == 0)
{
return x_287;
}
else
{
lean_object* x_291; lean_object* x_292; 
x_291 = lean_ctor_get(x_287, 0);
lean_inc(x_291);
lean_dec(x_287);
x_292 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_292, 0, x_291);
return x_292;
}
}
}
else
{
lean_object* x_293; lean_object* x_294; lean_object* x_295; lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; 
x_293 = lean_ctor_get(x_268, 0);
lean_inc(x_293);
lean_dec(x_268);
lean_inc_ref(x_230);
x_294 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_294, 0, x_232);
lean_ctor_set(x_294, 1, x_230);
x_295 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_296 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_297 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_297, 0, lean_box(0));
lean_closure_set(x_297, 1, lean_box(0));
lean_closure_set(x_297, 2, x_278);
lean_closure_set(x_297, 3, lean_box(0));
lean_closure_set(x_297, 4, lean_box(0));
lean_closure_set(x_297, 5, x_296);
lean_closure_set(x_297, 6, x_79);
x_298 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_299 = l_Lean_withTraceNode___redArg(x_233, x_234, x_236, x_69, x_238, x_239, x_295, x_293, x_80, x_297, x_2, x_298);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_300 = lean_apply_8(x_299, x_294, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_300) == 0)
{
lean_object* x_301; lean_object* x_302; 
x_301 = lean_ctor_get(x_300, 0);
lean_inc(x_301);
lean_dec_ref(x_300);
lean_inc(x_6);
lean_inc(x_4);
x_302 = lp_aesop_Aesop_traceScript___redArg___lam__2(x_66, x_68, x_71, x_72, x_76, x_74, x_67, x_81, x_217, x_2, x_65, x_64, x_65, x_65, x_64, x_70, x_301, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
x_56 = x_216;
x_57 = x_302;
goto block_59;
}
else
{
lean_object* x_303; lean_object* x_304; lean_object* x_305; 
lean_dec_ref(x_217);
lean_dec(x_216);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_303 = lean_ctor_get(x_300, 0);
lean_inc(x_303);
if (lean_is_exclusive(x_300)) {
 lean_ctor_release(x_300, 0);
 x_304 = x_300;
} else {
 lean_dec_ref(x_300);
 x_304 = lean_box(0);
}
if (lean_is_scalar(x_304)) {
 x_305 = lean_alloc_ctor(1, 1, 0);
} else {
 x_305 = x_304;
}
lean_ctor_set(x_305, 0, x_303);
return x_305;
}
}
}
else
{
lean_object* x_306; lean_object* x_307; lean_object* x_308; lean_object* x_309; lean_object* x_310; lean_object* x_311; lean_object* x_312; lean_object* x_313; lean_object* x_314; lean_object* x_315; lean_object* x_316; lean_object* x_317; lean_object* x_318; lean_object* x_319; lean_object* x_320; lean_object* x_321; lean_object* x_322; lean_object* x_323; lean_object* x_324; lean_object* x_325; lean_object* x_326; lean_object* x_327; lean_object* x_328; lean_object* x_329; lean_object* x_330; 
x_306 = lean_ctor_get(x_260, 0);
x_307 = lean_ctor_get(x_260, 2);
x_308 = lean_ctor_get(x_260, 3);
x_309 = lean_ctor_get(x_260, 4);
lean_inc(x_309);
lean_inc(x_308);
lean_inc(x_307);
lean_inc(x_306);
lean_dec(x_260);
x_310 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
x_311 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_312 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_306);
x_313 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_313, 0, x_306);
x_314 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_314, 0, x_306);
x_315 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_315, 0, x_313);
lean_ctor_set(x_315, 1, x_314);
x_316 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_316, 0, x_309);
x_317 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_317, 0, x_308);
x_318 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_318, 0, x_307);
x_319 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_319, 0, x_315);
lean_ctor_set(x_319, 1, x_311);
lean_ctor_set(x_319, 2, x_318);
lean_ctor_set(x_319, 3, x_317);
lean_ctor_set(x_319, 4, x_316);
lean_ctor_set(x_258, 1, x_312);
lean_ctor_set(x_258, 0, x_319);
x_320 = l_ReaderT_instMonad___redArg(x_258);
x_321 = l_ReaderT_instMonad___redArg(x_320);
x_322 = lean_ctor_get(x_310, 0);
lean_inc(x_322);
if (lean_is_exclusive(x_310)) {
 lean_ctor_release(x_310, 0);
 lean_ctor_release(x_310, 1);
 x_323 = x_310;
} else {
 lean_dec_ref(x_310);
 x_323 = lean_box(0);
}
lean_inc_ref(x_230);
if (lean_is_scalar(x_323)) {
 x_324 = lean_alloc_ctor(0, 2, 0);
} else {
 x_324 = x_323;
}
lean_ctor_set(x_324, 0, x_232);
lean_ctor_set(x_324, 1, x_230);
x_325 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_326 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_327 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_327, 0, lean_box(0));
lean_closure_set(x_327, 1, lean_box(0));
lean_closure_set(x_327, 2, x_321);
lean_closure_set(x_327, 3, lean_box(0));
lean_closure_set(x_327, 4, lean_box(0));
lean_closure_set(x_327, 5, x_326);
lean_closure_set(x_327, 6, x_79);
x_328 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_329 = l_Lean_withTraceNode___redArg(x_233, x_234, x_236, x_69, x_238, x_239, x_325, x_322, x_80, x_327, x_2, x_328);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_330 = lean_apply_8(x_329, x_324, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_330) == 0)
{
lean_object* x_331; lean_object* x_332; 
x_331 = lean_ctor_get(x_330, 0);
lean_inc(x_331);
lean_dec_ref(x_330);
lean_inc(x_6);
lean_inc(x_4);
x_332 = lp_aesop_Aesop_traceScript___redArg___lam__2(x_66, x_68, x_71, x_72, x_76, x_74, x_67, x_81, x_217, x_2, x_65, x_64, x_65, x_65, x_64, x_70, x_331, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
x_56 = x_216;
x_57 = x_332;
goto block_59;
}
else
{
lean_object* x_333; lean_object* x_334; lean_object* x_335; 
lean_dec_ref(x_217);
lean_dec(x_216);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_333 = lean_ctor_get(x_330, 0);
lean_inc(x_333);
if (lean_is_exclusive(x_330)) {
 lean_ctor_release(x_330, 0);
 x_334 = x_330;
} else {
 lean_dec_ref(x_330);
 x_334 = lean_box(0);
}
if (lean_is_scalar(x_334)) {
 x_335 = lean_alloc_ctor(1, 1, 0);
} else {
 x_335 = x_334;
}
lean_ctor_set(x_335, 0, x_333);
return x_335;
}
}
}
else
{
lean_object* x_336; lean_object* x_337; lean_object* x_338; lean_object* x_339; lean_object* x_340; lean_object* x_341; lean_object* x_342; lean_object* x_343; lean_object* x_344; lean_object* x_345; lean_object* x_346; lean_object* x_347; lean_object* x_348; lean_object* x_349; lean_object* x_350; lean_object* x_351; lean_object* x_352; lean_object* x_353; lean_object* x_354; lean_object* x_355; lean_object* x_356; lean_object* x_357; lean_object* x_358; lean_object* x_359; lean_object* x_360; lean_object* x_361; lean_object* x_362; lean_object* x_363; 
x_336 = lean_ctor_get(x_258, 0);
lean_inc(x_336);
lean_dec(x_258);
x_337 = lean_ctor_get(x_336, 0);
lean_inc_ref(x_337);
x_338 = lean_ctor_get(x_336, 2);
lean_inc(x_338);
x_339 = lean_ctor_get(x_336, 3);
lean_inc(x_339);
x_340 = lean_ctor_get(x_336, 4);
lean_inc(x_340);
if (lean_is_exclusive(x_336)) {
 lean_ctor_release(x_336, 0);
 lean_ctor_release(x_336, 1);
 lean_ctor_release(x_336, 2);
 lean_ctor_release(x_336, 3);
 lean_ctor_release(x_336, 4);
 x_341 = x_336;
} else {
 lean_dec_ref(x_336);
 x_341 = lean_box(0);
}
x_342 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
x_343 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_344 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_337);
x_345 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_345, 0, x_337);
x_346 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_346, 0, x_337);
x_347 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_347, 0, x_345);
lean_ctor_set(x_347, 1, x_346);
x_348 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_348, 0, x_340);
x_349 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_349, 0, x_339);
x_350 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_350, 0, x_338);
if (lean_is_scalar(x_341)) {
 x_351 = lean_alloc_ctor(0, 5, 0);
} else {
 x_351 = x_341;
}
lean_ctor_set(x_351, 0, x_347);
lean_ctor_set(x_351, 1, x_343);
lean_ctor_set(x_351, 2, x_350);
lean_ctor_set(x_351, 3, x_349);
lean_ctor_set(x_351, 4, x_348);
x_352 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_352, 0, x_351);
lean_ctor_set(x_352, 1, x_344);
x_353 = l_ReaderT_instMonad___redArg(x_352);
x_354 = l_ReaderT_instMonad___redArg(x_353);
x_355 = lean_ctor_get(x_342, 0);
lean_inc(x_355);
if (lean_is_exclusive(x_342)) {
 lean_ctor_release(x_342, 0);
 lean_ctor_release(x_342, 1);
 x_356 = x_342;
} else {
 lean_dec_ref(x_342);
 x_356 = lean_box(0);
}
lean_inc_ref(x_230);
if (lean_is_scalar(x_356)) {
 x_357 = lean_alloc_ctor(0, 2, 0);
} else {
 x_357 = x_356;
}
lean_ctor_set(x_357, 0, x_232);
lean_ctor_set(x_357, 1, x_230);
x_358 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_359 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_360 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_360, 0, lean_box(0));
lean_closure_set(x_360, 1, lean_box(0));
lean_closure_set(x_360, 2, x_354);
lean_closure_set(x_360, 3, lean_box(0));
lean_closure_set(x_360, 4, lean_box(0));
lean_closure_set(x_360, 5, x_359);
lean_closure_set(x_360, 6, x_79);
x_361 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_362 = l_Lean_withTraceNode___redArg(x_233, x_234, x_236, x_69, x_238, x_239, x_358, x_355, x_80, x_360, x_2, x_361);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_363 = lean_apply_8(x_362, x_357, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_363) == 0)
{
lean_object* x_364; lean_object* x_365; 
x_364 = lean_ctor_get(x_363, 0);
lean_inc(x_364);
lean_dec_ref(x_363);
lean_inc(x_6);
lean_inc(x_4);
x_365 = lp_aesop_Aesop_traceScript___redArg___lam__2(x_66, x_68, x_71, x_72, x_76, x_74, x_67, x_81, x_217, x_2, x_65, x_64, x_65, x_65, x_64, x_70, x_364, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
x_56 = x_216;
x_57 = x_365;
goto block_59;
}
else
{
lean_object* x_366; lean_object* x_367; lean_object* x_368; 
lean_dec_ref(x_217);
lean_dec(x_216);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_366 = lean_ctor_get(x_363, 0);
lean_inc(x_366);
if (lean_is_exclusive(x_363)) {
 lean_ctor_release(x_363, 0);
 x_367 = x_363;
} else {
 lean_dec_ref(x_363);
 x_367 = lean_box(0);
}
if (lean_is_scalar(x_367)) {
 x_368 = lean_alloc_ctor(1, 1, 0);
} else {
 x_368 = x_367;
}
lean_ctor_set(x_368, 0, x_366);
return x_368;
}
}
}
else
{
lean_object* x_369; lean_object* x_370; lean_object* x_371; lean_object* x_372; lean_object* x_373; lean_object* x_374; lean_object* x_375; lean_object* x_376; lean_object* x_377; lean_object* x_378; lean_object* x_379; lean_object* x_380; lean_object* x_381; lean_object* x_382; lean_object* x_383; lean_object* x_384; lean_object* x_385; lean_object* x_386; lean_object* x_387; lean_object* x_388; lean_object* x_389; lean_object* x_390; lean_object* x_391; lean_object* x_392; lean_object* x_393; lean_object* x_394; lean_object* x_395; lean_object* x_396; lean_object* x_397; lean_object* x_398; lean_object* x_399; lean_object* x_400; lean_object* x_401; lean_object* x_402; lean_object* x_403; lean_object* x_404; lean_object* x_405; lean_object* x_406; lean_object* x_407; lean_object* x_408; lean_object* x_409; lean_object* x_410; lean_object* x_411; 
x_369 = lean_ctor_get(x_242, 0);
x_370 = lean_ctor_get(x_242, 2);
x_371 = lean_ctor_get(x_242, 3);
x_372 = lean_ctor_get(x_242, 4);
lean_inc(x_372);
lean_inc(x_371);
lean_inc(x_370);
lean_inc(x_369);
lean_dec(x_242);
x_373 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_374 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_369);
x_375 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_375, 0, x_369);
x_376 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_376, 0, x_369);
x_377 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_377, 0, x_375);
lean_ctor_set(x_377, 1, x_376);
x_378 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_378, 0, x_372);
x_379 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_379, 0, x_371);
x_380 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_380, 0, x_370);
x_381 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_381, 0, x_377);
lean_ctor_set(x_381, 1, x_373);
lean_ctor_set(x_381, 2, x_380);
lean_ctor_set(x_381, 3, x_379);
lean_ctor_set(x_381, 4, x_378);
lean_ctor_set(x_240, 1, x_374);
lean_ctor_set(x_240, 0, x_381);
x_382 = l_ReaderT_instMonad___redArg(x_240);
x_383 = lean_ctor_get(x_382, 0);
lean_inc_ref(x_383);
if (lean_is_exclusive(x_382)) {
 lean_ctor_release(x_382, 0);
 lean_ctor_release(x_382, 1);
 x_384 = x_382;
} else {
 lean_dec_ref(x_382);
 x_384 = lean_box(0);
}
x_385 = lean_ctor_get(x_383, 0);
lean_inc_ref(x_385);
x_386 = lean_ctor_get(x_383, 2);
lean_inc(x_386);
x_387 = lean_ctor_get(x_383, 3);
lean_inc(x_387);
x_388 = lean_ctor_get(x_383, 4);
lean_inc(x_388);
if (lean_is_exclusive(x_383)) {
 lean_ctor_release(x_383, 0);
 lean_ctor_release(x_383, 1);
 lean_ctor_release(x_383, 2);
 lean_ctor_release(x_383, 3);
 lean_ctor_release(x_383, 4);
 x_389 = x_383;
} else {
 lean_dec_ref(x_383);
 x_389 = lean_box(0);
}
x_390 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
x_391 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_392 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_385);
x_393 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_393, 0, x_385);
x_394 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_394, 0, x_385);
x_395 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_395, 0, x_393);
lean_ctor_set(x_395, 1, x_394);
x_396 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_396, 0, x_388);
x_397 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_397, 0, x_387);
x_398 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_398, 0, x_386);
if (lean_is_scalar(x_389)) {
 x_399 = lean_alloc_ctor(0, 5, 0);
} else {
 x_399 = x_389;
}
lean_ctor_set(x_399, 0, x_395);
lean_ctor_set(x_399, 1, x_391);
lean_ctor_set(x_399, 2, x_398);
lean_ctor_set(x_399, 3, x_397);
lean_ctor_set(x_399, 4, x_396);
if (lean_is_scalar(x_384)) {
 x_400 = lean_alloc_ctor(0, 2, 0);
} else {
 x_400 = x_384;
}
lean_ctor_set(x_400, 0, x_399);
lean_ctor_set(x_400, 1, x_392);
x_401 = l_ReaderT_instMonad___redArg(x_400);
x_402 = l_ReaderT_instMonad___redArg(x_401);
x_403 = lean_ctor_get(x_390, 0);
lean_inc(x_403);
if (lean_is_exclusive(x_390)) {
 lean_ctor_release(x_390, 0);
 lean_ctor_release(x_390, 1);
 x_404 = x_390;
} else {
 lean_dec_ref(x_390);
 x_404 = lean_box(0);
}
lean_inc_ref(x_230);
if (lean_is_scalar(x_404)) {
 x_405 = lean_alloc_ctor(0, 2, 0);
} else {
 x_405 = x_404;
}
lean_ctor_set(x_405, 0, x_232);
lean_ctor_set(x_405, 1, x_230);
x_406 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_407 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_408 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_408, 0, lean_box(0));
lean_closure_set(x_408, 1, lean_box(0));
lean_closure_set(x_408, 2, x_402);
lean_closure_set(x_408, 3, lean_box(0));
lean_closure_set(x_408, 4, lean_box(0));
lean_closure_set(x_408, 5, x_407);
lean_closure_set(x_408, 6, x_79);
x_409 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_410 = l_Lean_withTraceNode___redArg(x_233, x_234, x_236, x_69, x_238, x_239, x_406, x_403, x_80, x_408, x_2, x_409);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_411 = lean_apply_8(x_410, x_405, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_411) == 0)
{
lean_object* x_412; lean_object* x_413; 
x_412 = lean_ctor_get(x_411, 0);
lean_inc(x_412);
lean_dec_ref(x_411);
lean_inc(x_6);
lean_inc(x_4);
x_413 = lp_aesop_Aesop_traceScript___redArg___lam__2(x_66, x_68, x_71, x_72, x_76, x_74, x_67, x_81, x_217, x_2, x_65, x_64, x_65, x_65, x_64, x_70, x_412, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
x_56 = x_216;
x_57 = x_413;
goto block_59;
}
else
{
lean_object* x_414; lean_object* x_415; lean_object* x_416; 
lean_dec_ref(x_217);
lean_dec(x_216);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_414 = lean_ctor_get(x_411, 0);
lean_inc(x_414);
if (lean_is_exclusive(x_411)) {
 lean_ctor_release(x_411, 0);
 x_415 = x_411;
} else {
 lean_dec_ref(x_411);
 x_415 = lean_box(0);
}
if (lean_is_scalar(x_415)) {
 x_416 = lean_alloc_ctor(1, 1, 0);
} else {
 x_416 = x_415;
}
lean_ctor_set(x_416, 0, x_414);
return x_416;
}
}
}
else
{
lean_object* x_417; lean_object* x_418; lean_object* x_419; lean_object* x_420; lean_object* x_421; lean_object* x_422; lean_object* x_423; lean_object* x_424; lean_object* x_425; lean_object* x_426; lean_object* x_427; lean_object* x_428; lean_object* x_429; lean_object* x_430; lean_object* x_431; lean_object* x_432; lean_object* x_433; lean_object* x_434; lean_object* x_435; lean_object* x_436; lean_object* x_437; lean_object* x_438; lean_object* x_439; lean_object* x_440; lean_object* x_441; lean_object* x_442; lean_object* x_443; lean_object* x_444; lean_object* x_445; lean_object* x_446; lean_object* x_447; lean_object* x_448; lean_object* x_449; lean_object* x_450; lean_object* x_451; lean_object* x_452; lean_object* x_453; lean_object* x_454; lean_object* x_455; lean_object* x_456; lean_object* x_457; lean_object* x_458; lean_object* x_459; lean_object* x_460; lean_object* x_461; lean_object* x_462; 
x_417 = lean_ctor_get(x_240, 0);
lean_inc(x_417);
lean_dec(x_240);
x_418 = lean_ctor_get(x_417, 0);
lean_inc_ref(x_418);
x_419 = lean_ctor_get(x_417, 2);
lean_inc(x_419);
x_420 = lean_ctor_get(x_417, 3);
lean_inc(x_420);
x_421 = lean_ctor_get(x_417, 4);
lean_inc(x_421);
if (lean_is_exclusive(x_417)) {
 lean_ctor_release(x_417, 0);
 lean_ctor_release(x_417, 1);
 lean_ctor_release(x_417, 2);
 lean_ctor_release(x_417, 3);
 lean_ctor_release(x_417, 4);
 x_422 = x_417;
} else {
 lean_dec_ref(x_417);
 x_422 = lean_box(0);
}
x_423 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_424 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_418);
x_425 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_425, 0, x_418);
x_426 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_426, 0, x_418);
x_427 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_427, 0, x_425);
lean_ctor_set(x_427, 1, x_426);
x_428 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_428, 0, x_421);
x_429 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_429, 0, x_420);
x_430 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_430, 0, x_419);
if (lean_is_scalar(x_422)) {
 x_431 = lean_alloc_ctor(0, 5, 0);
} else {
 x_431 = x_422;
}
lean_ctor_set(x_431, 0, x_427);
lean_ctor_set(x_431, 1, x_423);
lean_ctor_set(x_431, 2, x_430);
lean_ctor_set(x_431, 3, x_429);
lean_ctor_set(x_431, 4, x_428);
x_432 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_432, 0, x_431);
lean_ctor_set(x_432, 1, x_424);
x_433 = l_ReaderT_instMonad___redArg(x_432);
x_434 = lean_ctor_get(x_433, 0);
lean_inc_ref(x_434);
if (lean_is_exclusive(x_433)) {
 lean_ctor_release(x_433, 0);
 lean_ctor_release(x_433, 1);
 x_435 = x_433;
} else {
 lean_dec_ref(x_433);
 x_435 = lean_box(0);
}
x_436 = lean_ctor_get(x_434, 0);
lean_inc_ref(x_436);
x_437 = lean_ctor_get(x_434, 2);
lean_inc(x_437);
x_438 = lean_ctor_get(x_434, 3);
lean_inc(x_438);
x_439 = lean_ctor_get(x_434, 4);
lean_inc(x_439);
if (lean_is_exclusive(x_434)) {
 lean_ctor_release(x_434, 0);
 lean_ctor_release(x_434, 1);
 lean_ctor_release(x_434, 2);
 lean_ctor_release(x_434, 3);
 lean_ctor_release(x_434, 4);
 x_440 = x_434;
} else {
 lean_dec_ref(x_434);
 x_440 = lean_box(0);
}
x_441 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1;
x_442 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4;
x_443 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5;
lean_inc_ref(x_436);
x_444 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_444, 0, x_436);
x_445 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_445, 0, x_436);
x_446 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_446, 0, x_444);
lean_ctor_set(x_446, 1, x_445);
x_447 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_447, 0, x_439);
x_448 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_448, 0, x_438);
x_449 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_449, 0, x_437);
if (lean_is_scalar(x_440)) {
 x_450 = lean_alloc_ctor(0, 5, 0);
} else {
 x_450 = x_440;
}
lean_ctor_set(x_450, 0, x_446);
lean_ctor_set(x_450, 1, x_442);
lean_ctor_set(x_450, 2, x_449);
lean_ctor_set(x_450, 3, x_448);
lean_ctor_set(x_450, 4, x_447);
if (lean_is_scalar(x_435)) {
 x_451 = lean_alloc_ctor(0, 2, 0);
} else {
 x_451 = x_435;
}
lean_ctor_set(x_451, 0, x_450);
lean_ctor_set(x_451, 1, x_443);
x_452 = l_ReaderT_instMonad___redArg(x_451);
x_453 = l_ReaderT_instMonad___redArg(x_452);
x_454 = lean_ctor_get(x_441, 0);
lean_inc(x_454);
if (lean_is_exclusive(x_441)) {
 lean_ctor_release(x_441, 0);
 lean_ctor_release(x_441, 1);
 x_455 = x_441;
} else {
 lean_dec_ref(x_441);
 x_455 = lean_box(0);
}
lean_inc_ref(x_230);
if (lean_is_scalar(x_455)) {
 x_456 = lean_alloc_ctor(0, 2, 0);
} else {
 x_456 = x_455;
}
lean_ctor_set(x_456, 0, x_232);
lean_ctor_set(x_456, 1, x_230);
x_457 = lp_aesop_Aesop_expandNextGoal___redArg___closed__10;
x_458 = lp_aesop_Aesop_traceScript___redArg___closed__15;
x_459 = lean_alloc_closure((void*)(l_ReaderT_bind), 8, 7);
lean_closure_set(x_459, 0, lean_box(0));
lean_closure_set(x_459, 1, lean_box(0));
lean_closure_set(x_459, 2, x_453);
lean_closure_set(x_459, 3, lean_box(0));
lean_closure_set(x_459, 4, lean_box(0));
lean_closure_set(x_459, 5, x_458);
lean_closure_set(x_459, 6, x_79);
x_460 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_461 = l_Lean_withTraceNode___redArg(x_233, x_234, x_236, x_69, x_238, x_239, x_457, x_454, x_80, x_459, x_2, x_460);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
x_462 = lean_apply_8(x_461, x_456, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_462) == 0)
{
lean_object* x_463; lean_object* x_464; 
x_463 = lean_ctor_get(x_462, 0);
lean_inc(x_463);
lean_dec_ref(x_462);
lean_inc(x_6);
lean_inc(x_4);
x_464 = lp_aesop_Aesop_traceScript___redArg___lam__2(x_66, x_68, x_71, x_72, x_76, x_74, x_67, x_81, x_217, x_2, x_65, x_64, x_65, x_65, x_64, x_70, x_463, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
x_56 = x_216;
x_57 = x_464;
goto block_59;
}
else
{
lean_object* x_465; lean_object* x_466; lean_object* x_467; 
lean_dec_ref(x_217);
lean_dec(x_216);
lean_dec_ref(x_76);
lean_dec_ref(x_72);
lean_dec(x_68);
lean_dec_ref(x_66);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_465 = lean_ctor_get(x_462, 0);
lean_inc(x_465);
if (lean_is_exclusive(x_462)) {
 lean_ctor_release(x_462, 0);
 x_466 = x_462;
} else {
 lean_dec_ref(x_462);
 x_466 = lean_box(0);
}
if (lean_is_scalar(x_466)) {
 x_467 = lean_alloc_ctor(1, 1, 0);
} else {
 x_467 = x_466;
}
lean_ctor_set(x_467, 0, x_465);
return x_467;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_traceScript___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
uint8_t x_13; lean_object* x_14; 
x_13 = lean_unbox(x_3);
x_14 = lp_aesop_Aesop_traceScript(x_1, x_2, x_13, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_2);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__2___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
lean_object* x_19 = _args[18];
lean_object* x_20 = _args[19];
lean_object* x_21 = _args[20];
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
lean_object* x_24 = _args[23];
lean_object* x_25 = _args[24];
lean_object* x_26 = _args[25];
_start:
{
uint8_t x_27; lean_object* x_28; 
x_27 = lean_unbox(x_10);
x_28 = lp_aesop_Aesop_traceScript___redArg___lam__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_27, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25);
return x_28;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; lean_object* x_13; 
x_12 = lean_unbox(x_2);
x_13 = lp_aesop_Aesop_traceScript___redArg(x_1, x_12, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_1);
return x_13;
}
}
static lean_object* _init_lp_aesop_Aesop_traceTree___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_TraceOption_tree;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_st_ref_get(x_2);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec(x_10);
x_12 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_12);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
x_14 = lp_aesop_Aesop_getRootGoal(x_13, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_13);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_st_ref_get(x_2);
lean_dec(x_16);
x_17 = lean_st_ref_get(x_15);
lean_dec(x_15);
x_18 = lean_st_ref_get(x_2);
lean_dec(x_18);
x_19 = lp_aesop_Aesop_traceTree___redArg___closed__0;
x_20 = lp_aesop_Aesop_Goal_traceTree(x_17, x_19, x_5, x_6, x_7, x_8);
return x_20;
}
else
{
uint8_t x_21; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
x_21 = !lean_is_exclusive(x_14);
if (x_21 == 0)
{
return x_14;
}
else
{
lean_object* x_22; lean_object* x_23; 
x_22 = lean_ctor_get(x_14, 0);
lean_inc(x_22);
lean_dec(x_14);
x_23 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_23, 0, x_22);
return x_23;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_traceTree___redArg(x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_traceTree(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_traceTree___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_st_ref_get(x_3);
lean_dec(x_11);
x_12 = lp_aesop_Aesop_getRootMVarCluster___redArg(x_4);
if (lean_obj_tag(x_12) == 0)
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; uint8_t x_20; uint8_t x_21; 
x_14 = lean_ctor_get(x_12, 0);
x_15 = lean_st_ref_get(x_3);
lean_dec(x_15);
x_16 = lean_st_ref_get(x_14);
lean_dec(x_14);
x_17 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_18 = lean_ctor_get(x_17, 5);
lean_inc_ref(x_18);
x_19 = lean_apply_1(x_18, x_16);
x_20 = lean_ctor_get_uint8(x_19, sizeof(void*)*2 + 1);
lean_dec_ref(x_19);
x_21 = lp_aesop_Aesop_NodeState_isProven(x_20);
if (x_21 == 0)
{
lean_object* x_22; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_22 = lean_box(x_21);
lean_ctor_set(x_12, 0, x_22);
return x_12;
}
else
{
lean_object* x_23; 
lean_free_object(x_12);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_23 = lp_aesop_Aesop_finalizeProof___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_23) == 0)
{
lean_object* x_24; 
lean_dec_ref(x_23);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_24 = lp_aesop_Aesop_traceScript___redArg(x_1, x_21, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_1);
if (lean_obj_tag(x_24) == 0)
{
lean_object* x_25; 
lean_dec_ref(x_24);
x_25 = lp_aesop_Aesop_traceTree___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
if (lean_obj_tag(x_25) == 0)
{
uint8_t x_26; 
x_26 = !lean_is_exclusive(x_25);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; 
x_27 = lean_ctor_get(x_25, 0);
lean_dec(x_27);
x_28 = lean_box(x_21);
lean_ctor_set(x_25, 0, x_28);
return x_25;
}
else
{
lean_object* x_29; lean_object* x_30; 
lean_dec(x_25);
x_29 = lean_box(x_21);
x_30 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_30, 0, x_29);
return x_30;
}
}
else
{
uint8_t x_31; 
x_31 = !lean_is_exclusive(x_25);
if (x_31 == 0)
{
return x_25;
}
else
{
lean_object* x_32; lean_object* x_33; 
x_32 = lean_ctor_get(x_25, 0);
lean_inc(x_32);
lean_dec(x_25);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_32);
return x_33;
}
}
}
else
{
uint8_t x_34; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_34 = !lean_is_exclusive(x_24);
if (x_34 == 0)
{
return x_24;
}
else
{
lean_object* x_35; lean_object* x_36; 
x_35 = lean_ctor_get(x_24, 0);
lean_inc(x_35);
lean_dec(x_24);
x_36 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_36, 0, x_35);
return x_36;
}
}
}
else
{
uint8_t x_37; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_37 = !lean_is_exclusive(x_23);
if (x_37 == 0)
{
return x_23;
}
else
{
lean_object* x_38; lean_object* x_39; 
x_38 = lean_ctor_get(x_23, 0);
lean_inc(x_38);
lean_dec(x_23);
x_39 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_39, 0, x_38);
return x_39;
}
}
}
}
else
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; uint8_t x_46; uint8_t x_47; 
x_40 = lean_ctor_get(x_12, 0);
lean_inc(x_40);
lean_dec(x_12);
x_41 = lean_st_ref_get(x_3);
lean_dec(x_41);
x_42 = lean_st_ref_get(x_40);
lean_dec(x_40);
x_43 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_44 = lean_ctor_get(x_43, 5);
lean_inc_ref(x_44);
x_45 = lean_apply_1(x_44, x_42);
x_46 = lean_ctor_get_uint8(x_45, sizeof(void*)*2 + 1);
lean_dec_ref(x_45);
x_47 = lp_aesop_Aesop_NodeState_isProven(x_46);
if (x_47 == 0)
{
lean_object* x_48; lean_object* x_49; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_48 = lean_box(x_47);
x_49 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_49, 0, x_48);
return x_49;
}
else
{
lean_object* x_50; 
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_50 = lp_aesop_Aesop_finalizeProof___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_50) == 0)
{
lean_object* x_51; 
lean_dec_ref(x_50);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_51 = lp_aesop_Aesop_traceScript___redArg(x_1, x_47, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_1);
if (lean_obj_tag(x_51) == 0)
{
lean_object* x_52; 
lean_dec_ref(x_51);
x_52 = lp_aesop_Aesop_traceTree___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
if (lean_obj_tag(x_52) == 0)
{
lean_object* x_53; lean_object* x_54; lean_object* x_55; 
if (lean_is_exclusive(x_52)) {
 lean_ctor_release(x_52, 0);
 x_53 = x_52;
} else {
 lean_dec_ref(x_52);
 x_53 = lean_box(0);
}
x_54 = lean_box(x_47);
if (lean_is_scalar(x_53)) {
 x_55 = lean_alloc_ctor(0, 1, 0);
} else {
 x_55 = x_53;
}
lean_ctor_set(x_55, 0, x_54);
return x_55;
}
else
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; 
x_56 = lean_ctor_get(x_52, 0);
lean_inc(x_56);
if (lean_is_exclusive(x_52)) {
 lean_ctor_release(x_52, 0);
 x_57 = x_52;
} else {
 lean_dec_ref(x_52);
 x_57 = lean_box(0);
}
if (lean_is_scalar(x_57)) {
 x_58 = lean_alloc_ctor(1, 1, 0);
} else {
 x_58 = x_57;
}
lean_ctor_set(x_58, 0, x_56);
return x_58;
}
}
else
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_59 = lean_ctor_get(x_51, 0);
lean_inc(x_59);
if (lean_is_exclusive(x_51)) {
 lean_ctor_release(x_51, 0);
 x_60 = x_51;
} else {
 lean_dec_ref(x_51);
 x_60 = lean_box(0);
}
if (lean_is_scalar(x_60)) {
 x_61 = lean_alloc_ctor(1, 1, 0);
} else {
 x_61 = x_60;
}
lean_ctor_set(x_61, 0, x_59);
return x_61;
}
}
else
{
lean_object* x_62; lean_object* x_63; lean_object* x_64; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_62 = lean_ctor_get(x_50, 0);
lean_inc(x_62);
if (lean_is_exclusive(x_50)) {
 lean_ctor_release(x_50, 0);
 x_63 = x_50;
} else {
 lean_dec_ref(x_50);
 x_63 = lean_box(0);
}
if (lean_is_scalar(x_63)) {
 x_64 = lean_alloc_ctor(1, 1, 0);
} else {
 x_64 = x_63;
}
lean_ctor_set(x_64, 0, x_62);
return x_64;
}
}
}
}
else
{
uint8_t x_65; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_65 = !lean_is_exclusive(x_12);
if (x_65 == 0)
{
return x_12;
}
else
{
lean_object* x_66; lean_object* x_67; 
x_66 = lean_ctor_get(x_12, 0);
lean_inc(x_66);
lean_dec(x_12);
x_67 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_67, 0, x_66);
return x_67;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_finishIfProven___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_finishIfProven(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_finishIfProven___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1(lean_object* x_1, uint8_t x_2, lean_object* x_3, size_t x_4, size_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
uint8_t x_15; 
x_15 = lean_usize_dec_eq(x_4, x_5);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lean_array_uget(x_3, x_4);
x_17 = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(x_17, 0, x_16);
x_18 = lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(x_1, x_2, x_17, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; size_t x_20; size_t x_21; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = 1;
x_21 = lean_usize_add(x_4, x_20);
x_4 = x_21;
x_6 = x_19;
goto _start;
}
else
{
return x_18;
}
}
else
{
lean_object* x_23; 
x_23 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_23, 0, x_6);
return x_23;
}
}
}
static lean_object* _init_lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_preprocessRule;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2(lean_object* x_1, uint8_t x_2, lean_object* x_3, size_t x_4, size_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
uint8_t x_15; 
x_15 = lean_usize_dec_eq(x_4, x_5);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lean_array_uget(x_3, x_4);
x_17 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_17, 0, x_16);
x_18 = lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(x_1, x_2, x_17, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; size_t x_20; size_t x_21; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = 1;
x_21 = lean_usize_add(x_4, x_20);
x_4 = x_21;
x_6 = x_19;
goto _start;
}
else
{
return x_18;
}
}
else
{
lean_object* x_23; 
x_23 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_23, 0, x_6);
return x_23;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_16; lean_object* x_20; 
switch (lean_obj_tag(x_3)) {
case 0:
{
lean_object* x_24; lean_object* x_25; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_24 = lean_ctor_get(x_3, 0);
lean_inc(x_24);
lean_dec_ref(x_3);
x_40 = lean_st_ref_get(x_24);
x_41 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_42 = lean_ctor_get(x_41, 1);
lean_inc_ref(x_42);
x_43 = lean_apply_1(x_42, x_40);
x_44 = lean_ctor_get(x_43, 5);
lean_inc(x_44);
x_45 = lean_ctor_get(x_43, 6);
lean_inc(x_45);
lean_dec_ref(x_43);
x_46 = lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f(x_45);
lean_dec(x_45);
if (lean_obj_tag(x_46) == 1)
{
uint8_t x_47; 
x_47 = !lean_is_exclusive(x_46);
if (x_47 == 0)
{
lean_object* x_48; uint8_t x_49; 
x_48 = lean_ctor_get(x_46, 0);
x_49 = l_Lean_instBEqMVarId_beq(x_48, x_44);
lean_dec(x_44);
lean_dec(x_48);
if (x_49 == 0)
{
uint8_t x_50; lean_object* x_51; lean_object* x_52; 
x_50 = 1;
x_51 = lean_box(x_50);
x_52 = lean_st_ref_set(x_1, x_51);
if (x_2 == 0)
{
lean_object* x_53; 
lean_dec(x_24);
x_53 = lean_box(0);
lean_ctor_set_tag(x_46, 0);
lean_ctor_set(x_46, 0, x_53);
return x_46;
}
else
{
lean_free_object(x_46);
x_25 = lean_box(0);
goto block_39;
}
}
else
{
lean_free_object(x_46);
x_25 = lean_box(0);
goto block_39;
}
}
else
{
lean_object* x_54; uint8_t x_55; 
x_54 = lean_ctor_get(x_46, 0);
lean_inc(x_54);
lean_dec(x_46);
x_55 = l_Lean_instBEqMVarId_beq(x_54, x_44);
lean_dec(x_44);
lean_dec(x_54);
if (x_55 == 0)
{
uint8_t x_56; lean_object* x_57; lean_object* x_58; 
x_56 = 1;
x_57 = lean_box(x_56);
x_58 = lean_st_ref_set(x_1, x_57);
if (x_2 == 0)
{
lean_object* x_59; lean_object* x_60; 
lean_dec(x_24);
x_59 = lean_box(0);
x_60 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_60, 0, x_59);
return x_60;
}
else
{
x_25 = lean_box(0);
goto block_39;
}
}
else
{
x_25 = lean_box(0);
goto block_39;
}
}
}
else
{
lean_dec(x_46);
lean_dec(x_44);
x_25 = lean_box(0);
goto block_39;
}
block_39:
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_26 = lean_st_ref_get(x_24);
lean_dec(x_24);
x_27 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_28 = lean_ctor_get(x_27, 1);
lean_inc_ref(x_28);
x_29 = lean_apply_1(x_28, x_26);
x_30 = lean_ctor_get(x_29, 2);
lean_inc_ref(x_30);
lean_dec_ref(x_29);
x_31 = lean_unsigned_to_nat(0u);
x_32 = lean_array_get_size(x_30);
x_33 = lean_nat_dec_lt(x_31, x_32);
if (x_33 == 0)
{
lean_dec_ref(x_30);
x_16 = lean_box(0);
goto block_19;
}
else
{
uint8_t x_34; 
x_34 = lean_nat_dec_le(x_32, x_32);
if (x_34 == 0)
{
lean_dec_ref(x_30);
x_16 = lean_box(0);
goto block_19;
}
else
{
lean_object* x_35; size_t x_36; size_t x_37; lean_object* x_38; 
x_35 = lean_box(0);
x_36 = 0;
x_37 = lean_usize_of_nat(x_32);
x_38 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0(x_1, x_2, x_30, x_36, x_37, x_35, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_30);
if (lean_obj_tag(x_38) == 0)
{
lean_dec_ref(x_38);
x_16 = lean_box(0);
goto block_19;
}
else
{
return x_38;
}
}
}
}
}
case 1:
{
lean_object* x_61; lean_object* x_62; lean_object* x_63; uint8_t x_78; lean_object* x_79; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; uint8_t x_93; lean_object* x_95; lean_object* x_96; lean_object* x_97; uint8_t x_98; uint64_t x_99; lean_object* x_100; uint8_t x_101; uint8_t x_102; uint8_t x_103; uint64_t x_104; uint8_t x_105; uint8_t x_113; 
x_61 = lean_ctor_get(x_3, 0);
lean_inc(x_61);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 x_62 = x_3;
} else {
 lean_dec_ref(x_3);
 x_62 = lean_box(0);
}
x_83 = lean_st_ref_get(x_61);
x_84 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_85 = lean_ctor_get(x_84, 3);
lean_inc_ref(x_85);
x_86 = lean_apply_1(x_85, x_83);
x_87 = lean_ctor_get(x_86, 3);
lean_inc_ref(x_87);
lean_dec_ref(x_86);
x_95 = lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0___closed__0;
x_96 = lean_ctor_get(x_95, 0);
lean_inc_ref(x_96);
x_97 = lp_aesop_Aesop_RegularRule_name(x_87);
x_98 = lean_ctor_get_uint8(x_97, sizeof(void*)*1 + 8);
x_99 = lean_ctor_get_uint64(x_97, sizeof(void*)*1);
x_100 = lean_ctor_get(x_96, 0);
lean_inc(x_100);
x_101 = lean_ctor_get_uint8(x_96, sizeof(void*)*1 + 8);
x_102 = lean_ctor_get_uint8(x_96, sizeof(void*)*1 + 9);
x_103 = lean_ctor_get_uint8(x_96, sizeof(void*)*1 + 10);
x_104 = lean_ctor_get_uint64(x_96, sizeof(void*)*1);
lean_dec_ref(x_96);
x_113 = lean_uint64_dec_eq(x_99, x_104);
if (x_113 == 0)
{
x_105 = x_113;
goto block_112;
}
else
{
uint8_t x_114; 
x_114 = lp_aesop_Aesop_instBEqBuilderName_beq(x_98, x_101);
x_105 = x_114;
goto block_112;
}
block_77:
{
lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; uint8_t x_71; 
x_64 = lean_st_ref_get(x_61);
lean_dec(x_61);
x_65 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_66 = lean_ctor_get(x_65, 3);
lean_inc_ref(x_66);
x_67 = lean_apply_1(x_66, x_64);
x_68 = lean_ctor_get(x_67, 2);
lean_inc_ref(x_68);
lean_dec_ref(x_67);
x_69 = lean_unsigned_to_nat(0u);
x_70 = lean_array_get_size(x_68);
x_71 = lean_nat_dec_lt(x_69, x_70);
if (x_71 == 0)
{
lean_dec_ref(x_68);
x_12 = lean_box(0);
goto block_15;
}
else
{
uint8_t x_72; 
x_72 = lean_nat_dec_le(x_70, x_70);
if (x_72 == 0)
{
lean_dec_ref(x_68);
x_12 = lean_box(0);
goto block_15;
}
else
{
lean_object* x_73; size_t x_74; size_t x_75; lean_object* x_76; 
x_73 = lean_box(0);
x_74 = 0;
x_75 = lean_usize_of_nat(x_70);
x_76 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1(x_1, x_2, x_68, x_74, x_75, x_73, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_68);
if (lean_obj_tag(x_76) == 0)
{
lean_dec_ref(x_76);
x_12 = lean_box(0);
goto block_15;
}
else
{
return x_76;
}
}
}
}
block_82:
{
if (x_78 == 0)
{
lean_object* x_80; lean_object* x_81; 
lean_dec(x_61);
x_80 = lean_box(0);
if (lean_is_scalar(x_62)) {
 x_81 = lean_alloc_ctor(0, 1, 0);
} else {
 x_81 = x_62;
 lean_ctor_set_tag(x_81, 0);
}
lean_ctor_set(x_81, 0, x_80);
return x_81;
}
else
{
lean_dec(x_62);
x_63 = lean_box(0);
goto block_77;
}
}
block_92:
{
uint8_t x_88; 
x_88 = lp_aesop_Aesop_RegularRule_isUnsafe(x_87);
lean_dec_ref(x_87);
if (x_88 == 0)
{
uint8_t x_89; lean_object* x_90; lean_object* x_91; 
x_89 = 1;
x_90 = lean_box(x_89);
x_91 = lean_st_ref_set(x_1, x_90);
x_78 = x_2;
x_79 = lean_box(0);
goto block_82;
}
else
{
x_78 = x_2;
x_79 = lean_box(0);
goto block_82;
}
}
block_94:
{
if (x_93 == 0)
{
goto block_92;
}
else
{
lean_dec_ref(x_87);
lean_dec(x_62);
x_63 = lean_box(0);
goto block_77;
}
}
block_112:
{
if (x_105 == 0)
{
lean_dec(x_100);
lean_dec_ref(x_97);
goto block_92;
}
else
{
lean_object* x_106; uint8_t x_107; uint8_t x_108; uint8_t x_109; 
x_106 = lean_ctor_get(x_97, 0);
lean_inc(x_106);
x_107 = lean_ctor_get_uint8(x_97, sizeof(void*)*1 + 9);
x_108 = lean_ctor_get_uint8(x_97, sizeof(void*)*1 + 10);
lean_dec_ref(x_97);
x_109 = lp_aesop_Aesop_instBEqPhaseName_beq(x_107, x_102);
if (x_109 == 0)
{
lean_dec(x_106);
lean_dec(x_100);
x_93 = x_109;
goto block_94;
}
else
{
uint8_t x_110; 
x_110 = lp_aesop_Aesop_instBEqScopeName_beq(x_108, x_103);
if (x_110 == 0)
{
lean_dec(x_106);
lean_dec(x_100);
x_93 = x_110;
goto block_94;
}
else
{
uint8_t x_111; 
x_111 = lean_name_eq(x_106, x_100);
lean_dec(x_100);
lean_dec(x_106);
x_93 = x_111;
goto block_94;
}
}
}
}
}
default: 
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; uint8_t x_123; 
x_115 = lean_ctor_get(x_3, 0);
lean_inc(x_115);
lean_dec_ref(x_3);
x_116 = lean_st_ref_get(x_115);
lean_dec(x_115);
x_117 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0;
x_118 = lean_ctor_get(x_117, 5);
lean_inc_ref(x_118);
x_119 = lean_apply_1(x_118, x_116);
x_120 = lean_ctor_get(x_119, 1);
lean_inc_ref(x_120);
lean_dec_ref(x_119);
x_121 = lean_unsigned_to_nat(0u);
x_122 = lean_array_get_size(x_120);
x_123 = lean_nat_dec_lt(x_121, x_122);
if (x_123 == 0)
{
lean_dec_ref(x_120);
x_20 = lean_box(0);
goto block_23;
}
else
{
uint8_t x_124; 
x_124 = lean_nat_dec_le(x_122, x_122);
if (x_124 == 0)
{
lean_dec_ref(x_120);
x_20 = lean_box(0);
goto block_23;
}
else
{
lean_object* x_125; size_t x_126; size_t x_127; lean_object* x_128; 
x_125 = lean_box(0);
x_126 = 0;
x_127 = lean_usize_of_nat(x_122);
x_128 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2(x_1, x_2, x_120, x_126, x_127, x_125, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_120);
if (lean_obj_tag(x_128) == 0)
{
lean_dec_ref(x_128);
x_20 = lean_box(0);
goto block_23;
}
else
{
return x_128;
}
}
}
}
}
block_15:
{
lean_object* x_13; lean_object* x_14; 
x_13 = lean_box(0);
x_14 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
block_19:
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_box(0);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
block_23:
{
lean_object* x_21; lean_object* x_22; 
x_21 = lean_box(0);
x_22 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_22, 0, x_21);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0(lean_object* x_1, uint8_t x_2, lean_object* x_3, size_t x_4, size_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
uint8_t x_15; 
x_15 = lean_usize_dec_eq(x_4, x_5);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lean_array_uget(x_3, x_4);
x_17 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_17, 0, x_16);
x_18 = lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(x_1, x_2, x_17, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; size_t x_20; size_t x_21; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = 1;
x_21 = lean_usize_add(x_4, x_20);
x_4 = x_21;
x_6 = x_19;
goto _start;
}
else
{
return x_18;
}
}
else
{
lean_object* x_23; 
x_23 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_23, 0, x_6);
return x_23;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeHasProgress(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_9 = 0;
x_10 = lean_box(x_9);
x_11 = lean_st_mk_ref(x_10);
x_12 = lean_st_ref_get(x_2);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec(x_12);
x_14 = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(x_14, 0, x_13);
x_15 = lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(x_11, x_9, x_14, x_1, x_2, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_15) == 0)
{
uint8_t x_16; 
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_ctor_get(x_15, 0);
lean_dec(x_17);
x_18 = lean_st_ref_get(x_11);
lean_dec(x_11);
lean_ctor_set(x_15, 0, x_18);
return x_15;
}
else
{
lean_object* x_19; lean_object* x_20; 
lean_dec(x_15);
x_19 = lean_st_ref_get(x_11);
lean_dec(x_11);
x_20 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_20, 0, x_19);
return x_20;
}
}
else
{
uint8_t x_21; 
lean_dec(x_11);
x_21 = !lean_is_exclusive(x_15);
if (x_21 == 0)
{
return x_15;
}
else
{
lean_object* x_22; lean_object* x_23; 
x_22 = lean_ctor_get(x_15, 0);
lean_inc(x_22);
lean_dec(x_15);
x_23 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_23, 0, x_22);
return x_23;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
uint8_t x_15; size_t x_16; size_t x_17; lean_object* x_18; 
x_15 = lean_unbox(x_2);
x_16 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_17 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_18 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0(x_1, x_15, x_3, x_16, x_17, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
lean_dec(x_1);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
uint8_t x_15; size_t x_16; size_t x_17; lean_object* x_18; 
x_15 = lean_unbox(x_2);
x_16 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_17 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_18 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1(x_1, x_15, x_3, x_16, x_17, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
lean_dec(x_1);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
uint8_t x_15; size_t x_16; size_t x_17; lean_object* x_18; 
x_15 = lean_unbox(x_2);
x_16 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_17 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_18 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2(x_1, x_15, x_3, x_16, x_17, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_3);
lean_dec(x_1);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeHasProgress___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_aesop_Aesop_treeHasProgress(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; lean_object* x_13; 
x_12 = lean_unbox(x_2);
x_13 = lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(x_1, x_12, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_1);
return x_13;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tactic 'aesop' failed\nInitial goal:", 35, 35);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_throwAesopEx___redArg___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tactic 'aesop' failed, ", 23, 23);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_throwAesopEx___redArg___closed__2;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("\nInitial goal:", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_throwAesopEx___redArg___closed__4;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_aesop_smallErrorMessages;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("\n\n", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_throwAesopEx___redArg___closed__7;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_throwAesopEx___redArg___closed__8;
x_2 = l_Lean_MessageData_ofFormat(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("\nRemaining goals after safe rules:", 34, 34);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_throwAesopEx___redArg___closed__10;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("\nThe safe prefix was not fully expanded because the maximum number of rule applications (", 89, 89);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_throwAesopEx___redArg___closed__12;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(") was reached.", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_throwAesopEx___redArg___closed__14;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__17() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tactic 'aesop' failed", 21, 21);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_throwAesopEx___redArg___closed__17;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_63; lean_object* x_64; lean_object* x_65; uint8_t x_66; 
x_15 = l_Lean_KVMap_instValueBool;
x_16 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
x_63 = lean_ctor_get(x_12, 2);
x_64 = lp_aesop_Aesop_throwAesopEx___redArg___closed__6;
x_65 = l_Lean_Option_get___redArg(x_15, x_63, x_64);
x_66 = lean_unbox(x_65);
lean_dec(x_65);
if (x_66 == 0)
{
lean_object* x_67; lean_object* x_68; lean_object* x_69; uint8_t x_70; 
x_67 = lean_ctor_get(x_6, 2);
x_68 = lean_ctor_get(x_67, 0);
x_69 = lean_ctor_get(x_68, 4);
x_70 = l_Array_isEmpty___redArg(x_3);
if (x_70 == 0)
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; 
x_71 = lean_alloc_closure((void*)(lp_aesop_Aesop_throwAesopEx___redArg___lam__0), 1, 0);
x_72 = lean_array_to_list(x_3);
x_73 = lean_box(0);
x_74 = l_List_mapTR_loop___redArg(x_71, x_72, x_73);
x_75 = lp_aesop_Aesop_throwAesopEx___redArg___closed__9;
x_76 = l_Lean_MessageData_joinSep(x_74, x_75);
if (x_4 == 0)
{
lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; 
x_83 = lp_aesop_Aesop_throwAesopEx___redArg___closed__13;
lean_inc(x_69);
x_84 = l_Nat_reprFast(x_69);
x_85 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_85, 0, x_84);
x_86 = l_Lean_MessageData_ofFormat(x_85);
x_87 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_87, 0, x_83);
lean_ctor_set(x_87, 1, x_86);
x_88 = lp_aesop_Aesop_throwAesopEx___redArg___closed__15;
x_89 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_89, 0, x_87);
lean_ctor_set(x_89, 1, x_88);
x_77 = x_89;
goto block_82;
}
else
{
lean_object* x_90; 
x_90 = lp_aesop_Aesop_throwAesopEx___redArg___closed__16;
x_77 = x_90;
goto block_82;
}
block_82:
{
lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; 
x_78 = lp_aesop_Aesop_throwAesopEx___redArg___closed__11;
x_79 = l_Lean_indentD(x_76);
x_80 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_80, 0, x_78);
lean_ctor_set(x_80, 1, x_79);
x_81 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_81, 0, x_80);
lean_ctor_set(x_81, 1, x_77);
x_17 = x_81;
goto block_62;
}
}
else
{
lean_object* x_91; 
lean_dec_ref(x_3);
x_91 = lp_aesop_Aesop_throwAesopEx___redArg___closed__16;
x_17 = x_91;
goto block_62;
}
}
else
{
lean_dec_ref(x_3);
lean_dec(x_2);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; 
x_92 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_93 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_94 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_16);
x_95 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_94, x_16);
x_96 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_96, 0, x_92);
lean_ctor_set(x_96, 1, x_93);
lean_ctor_set(x_96, 2, x_95);
x_97 = lp_aesop_Aesop_throwAesopEx___redArg___closed__18;
x_98 = l_Lean_throwError___redArg(x_16, x_96, x_97);
x_99 = lean_apply_9(x_98, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, lean_box(0));
return x_99;
}
else
{
lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; 
x_100 = lean_ctor_get(x_5, 0);
lean_inc(x_100);
lean_dec_ref(x_5);
x_101 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_102 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_103 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_16);
x_104 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_103, x_16);
x_105 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_105, 0, x_101);
lean_ctor_set(x_105, 1, x_102);
lean_ctor_set(x_105, 2, x_104);
x_106 = lp_aesop_Aesop_throwAesopEx___redArg___closed__3;
x_107 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_107, 0, x_106);
lean_ctor_set(x_107, 1, x_100);
x_108 = l_Lean_throwError___redArg(x_16, x_105, x_107);
x_109 = lean_apply_9(x_108, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, lean_box(0));
return x_109;
}
}
block_62:
{
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_18 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_19 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_20 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_16);
x_21 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_20, x_16);
x_22 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_22, 0, x_18);
lean_ctor_set(x_22, 1, x_19);
lean_ctor_set(x_22, 2, x_21);
x_23 = lp_aesop_Aesop_throwAesopEx___redArg___closed__1;
x_24 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_24, 0, x_2);
x_25 = l_Lean_indentD(x_24);
x_26 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_26, 0, x_23);
lean_ctor_set(x_26, 1, x_25);
x_27 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_17);
x_28 = l_Lean_throwError___redArg(x_16, x_22, x_27);
x_29 = lean_apply_9(x_28, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, lean_box(0));
return x_29;
}
else
{
uint8_t x_30; 
x_30 = !lean_is_exclusive(x_5);
if (x_30 == 0)
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; 
x_31 = lean_ctor_get(x_5, 0);
x_32 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_33 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_34 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_16);
x_35 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_34, x_16);
x_36 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_36, 0, x_32);
lean_ctor_set(x_36, 1, x_33);
lean_ctor_set(x_36, 2, x_35);
x_37 = lp_aesop_Aesop_throwAesopEx___redArg___closed__3;
x_38 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_31);
x_39 = lp_aesop_Aesop_throwAesopEx___redArg___closed__5;
x_40 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_40, 0, x_38);
lean_ctor_set(x_40, 1, x_39);
lean_ctor_set(x_5, 0, x_2);
x_41 = l_Lean_indentD(x_5);
x_42 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_42, 0, x_40);
lean_ctor_set(x_42, 1, x_41);
x_43 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_43, 0, x_42);
lean_ctor_set(x_43, 1, x_17);
x_44 = l_Lean_throwError___redArg(x_16, x_36, x_43);
x_45 = lean_apply_9(x_44, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, lean_box(0));
return x_45;
}
else
{
lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; 
x_46 = lean_ctor_get(x_5, 0);
lean_inc(x_46);
lean_dec(x_5);
x_47 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_48 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_49 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
lean_inc_ref(x_16);
x_50 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_49, x_16);
x_51 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_51, 0, x_47);
lean_ctor_set(x_51, 1, x_48);
lean_ctor_set(x_51, 2, x_50);
x_52 = lp_aesop_Aesop_throwAesopEx___redArg___closed__3;
x_53 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_53, 0, x_52);
lean_ctor_set(x_53, 1, x_46);
x_54 = lp_aesop_Aesop_throwAesopEx___redArg___closed__5;
x_55 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_55, 0, x_53);
lean_ctor_set(x_55, 1, x_54);
x_56 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_56, 0, x_2);
x_57 = l_Lean_indentD(x_56);
x_58 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_58, 0, x_55);
lean_ctor_set(x_58, 1, x_57);
x_59 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_59, 0, x_58);
lean_ctor_set(x_59, 1, x_17);
x_60 = l_Lean_throwError___redArg(x_16, x_51, x_59);
x_61 = lean_apply_9(x_60, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, lean_box(0));
return x_61;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_17; 
x_17 = lp_aesop_Aesop_throwAesopEx___redArg(x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
uint8_t x_17; lean_object* x_18; 
x_17 = lean_unbox(x_6);
x_18 = lp_aesop_Aesop_throwAesopEx(x_1, x_2, x_3, x_4, x_5, x_17, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec_ref(x_2);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
uint8_t x_15; lean_object* x_16; 
x_15 = lean_unbox(x_4);
x_16 = lp_aesop_Aesop_throwAesopEx___redArg(x_1, x_2, x_3, x_15, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_1);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_st_ref_get(x_3);
lean_dec(x_11);
x_12 = lp_aesop_Aesop_clearForwardImplDetailHyps(x_1, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_handleNonfatalError___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_11;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop: safe prefix was not fully expanded because the maximum number of rule applications (", 91, 91);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_aesop_warn_nonterminal;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop: ", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__3;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("made no progress", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__6;
x_2 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__7;
x_2 = l_Lean_MessageData_ofFormat(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__8;
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("<no proof>", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__10;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_12 = l_Lean_KVMap_instValueBool;
x_13 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
x_14 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_16 = lp_aesop_Aesop_expandSafePrefix___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lean_st_ref_get(x_4);
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec(x_18);
x_20 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_20);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_20);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
x_22 = lp_aesop_Aesop_extractSafePrefix(x_21, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_21);
if (lean_obj_tag(x_22) == 0)
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
if (lean_is_exclusive(x_22)) {
 lean_ctor_release(x_22, 0);
 x_24 = x_22;
} else {
 lean_dec_ref(x_22);
 x_24 = lean_box(0);
}
x_25 = lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__0;
lean_inc(x_15);
lean_inc_ref(x_13);
x_26 = lp_aesop_Aesop_TraceOption_isEnabled___redArg(x_13, x_15, x_25);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_27 = lean_apply_9(x_26, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_27) == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; uint8_t x_171; 
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
lean_dec_ref(x_27);
x_29 = lean_alloc_closure((void*)(lp_aesop_Aesop_handleNonfatalError___redArg___lam__0___boxed), 10, 0);
x_171 = lean_unbox(x_28);
lean_dec(x_28);
if (x_171 == 0)
{
x_134 = x_3;
x_135 = x_4;
x_136 = x_5;
x_137 = x_6;
x_138 = x_7;
x_139 = x_8;
x_140 = x_9;
x_141 = x_10;
x_142 = lean_box(0);
goto block_170;
}
else
{
lean_object* x_172; 
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_172 = lp_aesop_Aesop_getProof_x3f___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_172) == 0)
{
lean_object* x_173; 
x_173 = lean_ctor_get(x_172, 0);
lean_inc(x_173);
lean_dec_ref(x_172);
if (lean_obj_tag(x_173) == 0)
{
lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; 
x_174 = lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
x_175 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_176 = lean_ctor_get(x_25, 0);
lean_inc(x_176);
x_177 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
x_178 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__11;
lean_inc_ref(x_13);
x_179 = l_Lean_addTrace___redArg(x_13, x_174, x_175, x_177, x_176, x_178);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_180 = lean_apply_9(x_179, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_180) == 0)
{
lean_dec_ref(x_180);
x_134 = x_3;
x_135 = x_4;
x_136 = x_5;
x_137 = x_6;
x_138 = x_7;
x_139 = x_8;
x_140 = x_9;
x_141 = x_10;
x_142 = lean_box(0);
goto block_170;
}
else
{
uint8_t x_181; 
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_181 = !lean_is_exclusive(x_180);
if (x_181 == 0)
{
return x_180;
}
else
{
lean_object* x_182; lean_object* x_183; 
x_182 = lean_ctor_get(x_180, 0);
lean_inc(x_182);
lean_dec(x_180);
x_183 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_183, 0, x_182);
return x_183;
}
}
}
else
{
lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; 
x_184 = lean_ctor_get(x_173, 0);
lean_inc(x_184);
lean_dec_ref(x_173);
x_185 = lean_st_ref_get(x_4);
x_186 = lean_ctor_get(x_185, 0);
lean_inc(x_186);
lean_dec(x_185);
lean_inc_ref(x_20);
x_187 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_187, 0, x_186);
lean_ctor_set(x_187, 1, x_20);
x_188 = lp_aesop_Aesop_getRootMVarId(x_187, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_187);
if (lean_obj_tag(x_188) == 0)
{
lean_object* x_189; lean_object* x_190; lean_object* x_191; uint8_t x_192; 
x_189 = lean_ctor_get(x_188, 0);
lean_inc(x_189);
lean_dec_ref(x_188);
x_190 = lp_aesop_Aesop_finalizeProof___redArg___closed__0;
x_191 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1;
x_192 = !lean_is_exclusive(x_191);
if (x_192 == 0)
{
lean_object* x_193; lean_object* x_194; uint8_t x_195; 
x_193 = lean_ctor_get(x_191, 0);
x_194 = lean_ctor_get(x_191, 1);
lean_dec(x_194);
x_195 = !lean_is_exclusive(x_193);
if (x_195 == 0)
{
lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; lean_object* x_231; 
x_196 = lean_ctor_get(x_193, 0);
x_197 = lean_ctor_get(x_193, 2);
x_198 = lean_ctor_get(x_193, 3);
x_199 = lean_ctor_get(x_193, 4);
x_200 = lean_ctor_get(x_193, 1);
lean_dec(x_200);
x_201 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_202 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_196);
x_203 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_203, 0, x_196);
x_204 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_204, 0, x_196);
x_205 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_205, 0, x_203);
lean_ctor_set(x_205, 1, x_204);
x_206 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_206, 0, x_199);
x_207 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_207, 0, x_198);
x_208 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_208, 0, x_197);
lean_ctor_set(x_193, 4, x_206);
lean_ctor_set(x_193, 3, x_207);
lean_ctor_set(x_193, 2, x_208);
lean_ctor_set(x_193, 1, x_201);
lean_ctor_set(x_193, 0, x_205);
lean_ctor_set(x_191, 1, x_202);
x_209 = l_ReaderT_instMonad___redArg(x_191);
x_210 = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(x_210, 0, lean_box(0));
lean_closure_set(x_210, 1, lean_box(0));
lean_closure_set(x_210, 2, x_209);
x_211 = l_instMonadControlTOfPure___redArg(x_210);
lean_inc_ref(x_211);
x_212 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_212, 0, x_190);
lean_closure_set(x_212, 1, x_211);
x_213 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_213, 0, x_190);
lean_closure_set(x_213, 1, x_211);
x_214 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_214, 0, x_212);
lean_ctor_set(x_214, 1, x_213);
lean_inc_ref(x_214);
x_215 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_215, 0, x_190);
lean_closure_set(x_215, 1, x_214);
x_216 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_216, 0, x_190);
lean_closure_set(x_216, 1, x_214);
x_217 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_217, 0, x_215);
lean_ctor_set(x_217, 1, x_216);
lean_inc_ref(x_217);
x_218 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_218, 0, x_190);
lean_closure_set(x_218, 1, x_217);
x_219 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_219, 0, x_190);
lean_closure_set(x_219, 1, x_217);
x_220 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_220, 0, x_218);
lean_ctor_set(x_220, 1, x_219);
lean_inc_ref(x_220);
x_221 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_221, 0, x_190);
lean_closure_set(x_221, 1, x_220);
x_222 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_222, 0, x_190);
lean_closure_set(x_222, 1, x_220);
x_223 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_223, 0, x_221);
lean_ctor_set(x_223, 1, x_222);
x_224 = lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
x_225 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_226 = lean_ctor_get(x_25, 0);
lean_inc(x_226);
x_227 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
x_228 = l_Lean_MessageData_ofExpr(x_184);
lean_inc_ref(x_13);
x_229 = l_Lean_addTrace___redArg(x_13, x_224, x_225, x_227, x_226, x_228);
lean_inc_ref(x_13);
x_230 = l_Lean_MVarId_withContext___redArg(x_223, x_13, x_189, x_229);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_231 = lean_apply_9(x_230, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_231) == 0)
{
lean_dec_ref(x_231);
x_134 = x_3;
x_135 = x_4;
x_136 = x_5;
x_137 = x_6;
x_138 = x_7;
x_139 = x_8;
x_140 = x_9;
x_141 = x_10;
x_142 = lean_box(0);
goto block_170;
}
else
{
uint8_t x_232; 
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_232 = !lean_is_exclusive(x_231);
if (x_232 == 0)
{
return x_231;
}
else
{
lean_object* x_233; lean_object* x_234; 
x_233 = lean_ctor_get(x_231, 0);
lean_inc(x_233);
lean_dec(x_231);
x_234 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_234, 0, x_233);
return x_234;
}
}
}
else
{
lean_object* x_235; lean_object* x_236; lean_object* x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; lean_object* x_241; lean_object* x_242; lean_object* x_243; lean_object* x_244; lean_object* x_245; lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; lean_object* x_259; lean_object* x_260; lean_object* x_261; lean_object* x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; lean_object* x_269; lean_object* x_270; 
x_235 = lean_ctor_get(x_193, 0);
x_236 = lean_ctor_get(x_193, 2);
x_237 = lean_ctor_get(x_193, 3);
x_238 = lean_ctor_get(x_193, 4);
lean_inc(x_238);
lean_inc(x_237);
lean_inc(x_236);
lean_inc(x_235);
lean_dec(x_193);
x_239 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_240 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_235);
x_241 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_241, 0, x_235);
x_242 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_242, 0, x_235);
x_243 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_243, 0, x_241);
lean_ctor_set(x_243, 1, x_242);
x_244 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_244, 0, x_238);
x_245 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_245, 0, x_237);
x_246 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_246, 0, x_236);
x_247 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_247, 0, x_243);
lean_ctor_set(x_247, 1, x_239);
lean_ctor_set(x_247, 2, x_246);
lean_ctor_set(x_247, 3, x_245);
lean_ctor_set(x_247, 4, x_244);
lean_ctor_set(x_191, 1, x_240);
lean_ctor_set(x_191, 0, x_247);
x_248 = l_ReaderT_instMonad___redArg(x_191);
x_249 = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(x_249, 0, lean_box(0));
lean_closure_set(x_249, 1, lean_box(0));
lean_closure_set(x_249, 2, x_248);
x_250 = l_instMonadControlTOfPure___redArg(x_249);
lean_inc_ref(x_250);
x_251 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_251, 0, x_190);
lean_closure_set(x_251, 1, x_250);
x_252 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_252, 0, x_190);
lean_closure_set(x_252, 1, x_250);
x_253 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_253, 0, x_251);
lean_ctor_set(x_253, 1, x_252);
lean_inc_ref(x_253);
x_254 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_254, 0, x_190);
lean_closure_set(x_254, 1, x_253);
x_255 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_255, 0, x_190);
lean_closure_set(x_255, 1, x_253);
x_256 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_256, 0, x_254);
lean_ctor_set(x_256, 1, x_255);
lean_inc_ref(x_256);
x_257 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_257, 0, x_190);
lean_closure_set(x_257, 1, x_256);
x_258 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_258, 0, x_190);
lean_closure_set(x_258, 1, x_256);
x_259 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_259, 0, x_257);
lean_ctor_set(x_259, 1, x_258);
lean_inc_ref(x_259);
x_260 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_260, 0, x_190);
lean_closure_set(x_260, 1, x_259);
x_261 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_261, 0, x_190);
lean_closure_set(x_261, 1, x_259);
x_262 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_262, 0, x_260);
lean_ctor_set(x_262, 1, x_261);
x_263 = lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
x_264 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_265 = lean_ctor_get(x_25, 0);
lean_inc(x_265);
x_266 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
x_267 = l_Lean_MessageData_ofExpr(x_184);
lean_inc_ref(x_13);
x_268 = l_Lean_addTrace___redArg(x_13, x_263, x_264, x_266, x_265, x_267);
lean_inc_ref(x_13);
x_269 = l_Lean_MVarId_withContext___redArg(x_262, x_13, x_189, x_268);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_270 = lean_apply_9(x_269, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_270) == 0)
{
lean_dec_ref(x_270);
x_134 = x_3;
x_135 = x_4;
x_136 = x_5;
x_137 = x_6;
x_138 = x_7;
x_139 = x_8;
x_140 = x_9;
x_141 = x_10;
x_142 = lean_box(0);
goto block_170;
}
else
{
lean_object* x_271; lean_object* x_272; lean_object* x_273; 
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_271 = lean_ctor_get(x_270, 0);
lean_inc(x_271);
if (lean_is_exclusive(x_270)) {
 lean_ctor_release(x_270, 0);
 x_272 = x_270;
} else {
 lean_dec_ref(x_270);
 x_272 = lean_box(0);
}
if (lean_is_scalar(x_272)) {
 x_273 = lean_alloc_ctor(1, 1, 0);
} else {
 x_273 = x_272;
}
lean_ctor_set(x_273, 0, x_271);
return x_273;
}
}
}
else
{
lean_object* x_274; lean_object* x_275; lean_object* x_276; lean_object* x_277; lean_object* x_278; lean_object* x_279; lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; lean_object* x_291; lean_object* x_292; lean_object* x_293; lean_object* x_294; lean_object* x_295; lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; lean_object* x_301; lean_object* x_302; lean_object* x_303; lean_object* x_304; lean_object* x_305; lean_object* x_306; lean_object* x_307; lean_object* x_308; lean_object* x_309; lean_object* x_310; lean_object* x_311; lean_object* x_312; 
x_274 = lean_ctor_get(x_191, 0);
lean_inc(x_274);
lean_dec(x_191);
x_275 = lean_ctor_get(x_274, 0);
lean_inc_ref(x_275);
x_276 = lean_ctor_get(x_274, 2);
lean_inc(x_276);
x_277 = lean_ctor_get(x_274, 3);
lean_inc(x_277);
x_278 = lean_ctor_get(x_274, 4);
lean_inc(x_278);
if (lean_is_exclusive(x_274)) {
 lean_ctor_release(x_274, 0);
 lean_ctor_release(x_274, 1);
 lean_ctor_release(x_274, 2);
 lean_ctor_release(x_274, 3);
 lean_ctor_release(x_274, 4);
 x_279 = x_274;
} else {
 lean_dec_ref(x_274);
 x_279 = lean_box(0);
}
x_280 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2;
x_281 = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3;
lean_inc_ref(x_275);
x_282 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_282, 0, x_275);
x_283 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_283, 0, x_275);
x_284 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_284, 0, x_282);
lean_ctor_set(x_284, 1, x_283);
x_285 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_285, 0, x_278);
x_286 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_286, 0, x_277);
x_287 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_287, 0, x_276);
if (lean_is_scalar(x_279)) {
 x_288 = lean_alloc_ctor(0, 5, 0);
} else {
 x_288 = x_279;
}
lean_ctor_set(x_288, 0, x_284);
lean_ctor_set(x_288, 1, x_280);
lean_ctor_set(x_288, 2, x_287);
lean_ctor_set(x_288, 3, x_286);
lean_ctor_set(x_288, 4, x_285);
x_289 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_289, 0, x_288);
lean_ctor_set(x_289, 1, x_281);
x_290 = l_ReaderT_instMonad___redArg(x_289);
x_291 = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(x_291, 0, lean_box(0));
lean_closure_set(x_291, 1, lean_box(0));
lean_closure_set(x_291, 2, x_290);
x_292 = l_instMonadControlTOfPure___redArg(x_291);
lean_inc_ref(x_292);
x_293 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_293, 0, x_190);
lean_closure_set(x_293, 1, x_292);
x_294 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_294, 0, x_190);
lean_closure_set(x_294, 1, x_292);
x_295 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_295, 0, x_293);
lean_ctor_set(x_295, 1, x_294);
lean_inc_ref(x_295);
x_296 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_296, 0, x_190);
lean_closure_set(x_296, 1, x_295);
x_297 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_297, 0, x_190);
lean_closure_set(x_297, 1, x_295);
x_298 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_298, 0, x_296);
lean_ctor_set(x_298, 1, x_297);
lean_inc_ref(x_298);
x_299 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_299, 0, x_190);
lean_closure_set(x_299, 1, x_298);
x_300 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_300, 0, x_190);
lean_closure_set(x_300, 1, x_298);
x_301 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_301, 0, x_299);
lean_ctor_set(x_301, 1, x_300);
lean_inc_ref(x_301);
x_302 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(x_302, 0, x_190);
lean_closure_set(x_302, 1, x_301);
x_303 = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(x_303, 0, x_190);
lean_closure_set(x_303, 1, x_301);
x_304 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_304, 0, x_302);
lean_ctor_set(x_304, 1, x_303);
x_305 = lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
x_306 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_307 = lean_ctor_get(x_25, 0);
lean_inc(x_307);
x_308 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
x_309 = l_Lean_MessageData_ofExpr(x_184);
lean_inc_ref(x_13);
x_310 = l_Lean_addTrace___redArg(x_13, x_305, x_306, x_308, x_307, x_309);
lean_inc_ref(x_13);
x_311 = l_Lean_MVarId_withContext___redArg(x_304, x_13, x_189, x_310);
lean_inc(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_312 = lean_apply_9(x_311, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, lean_box(0));
if (lean_obj_tag(x_312) == 0)
{
lean_dec_ref(x_312);
x_134 = x_3;
x_135 = x_4;
x_136 = x_5;
x_137 = x_6;
x_138 = x_7;
x_139 = x_8;
x_140 = x_9;
x_141 = x_10;
x_142 = lean_box(0);
goto block_170;
}
else
{
lean_object* x_313; lean_object* x_314; lean_object* x_315; 
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_313 = lean_ctor_get(x_312, 0);
lean_inc(x_313);
if (lean_is_exclusive(x_312)) {
 lean_ctor_release(x_312, 0);
 x_314 = x_312;
} else {
 lean_dec_ref(x_312);
 x_314 = lean_box(0);
}
if (lean_is_scalar(x_314)) {
 x_315 = lean_alloc_ctor(1, 1, 0);
} else {
 x_315 = x_314;
}
lean_ctor_set(x_315, 0, x_313);
return x_315;
}
}
}
else
{
uint8_t x_316; 
lean_dec(x_184);
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_316 = !lean_is_exclusive(x_188);
if (x_316 == 0)
{
return x_188;
}
else
{
lean_object* x_317; lean_object* x_318; 
x_317 = lean_ctor_get(x_188, 0);
lean_inc(x_317);
lean_dec(x_188);
x_318 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_318, 0, x_317);
return x_318;
}
}
}
}
else
{
uint8_t x_319; 
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_319 = !lean_is_exclusive(x_172);
if (x_319 == 0)
{
return x_172;
}
else
{
lean_object* x_320; lean_object* x_321; 
x_320 = lean_ctor_get(x_172, 0);
lean_inc(x_320);
lean_dec(x_172);
x_321 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_321, 0, x_320);
return x_321;
}
}
}
block_43:
{
size_t x_39; size_t x_40; lean_object* x_41; lean_object* x_42; 
x_39 = lean_array_size(x_23);
x_40 = 0;
x_41 = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), x_13, x_29, x_39, x_40, x_23);
x_42 = lean_apply_9(x_41, x_30, x_31, x_32, x_33, x_34, x_35, x_36, x_37, lean_box(0));
return x_42;
}
block_71:
{
uint8_t x_53; 
x_53 = lean_unbox(x_17);
lean_dec(x_17);
if (x_53 == 0)
{
lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; 
x_54 = lp_aesop_Aesop_traceScript___redArg___closed__5;
x_55 = lean_ctor_get(x_44, 2);
x_56 = lean_ctor_get(x_55, 0);
x_57 = lean_ctor_get(x_56, 4);
x_58 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
x_59 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__1;
lean_inc(x_57);
x_60 = l_Nat_reprFast(x_57);
if (lean_is_scalar(x_24)) {
 x_61 = lean_alloc_ctor(3, 1, 0);
} else {
 x_61 = x_24;
 lean_ctor_set_tag(x_61, 3);
}
lean_ctor_set(x_61, 0, x_60);
x_62 = l_Lean_MessageData_ofFormat(x_61);
x_63 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_63, 0, x_59);
lean_ctor_set(x_63, 1, x_62);
x_64 = lp_aesop_Aesop_throwAesopEx___redArg___closed__15;
x_65 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_65, 0, x_63);
lean_ctor_set(x_65, 1, x_64);
lean_inc_ref(x_13);
x_66 = l_Lean_logWarning___redArg(x_13, x_54, x_58, x_15, x_65);
lean_inc(x_51);
lean_inc_ref(x_50);
lean_inc(x_49);
lean_inc_ref(x_48);
lean_inc(x_47);
lean_inc(x_46);
lean_inc(x_45);
lean_inc_ref(x_44);
x_67 = lean_apply_9(x_66, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, lean_box(0));
if (lean_obj_tag(x_67) == 0)
{
lean_dec_ref(x_67);
x_30 = x_44;
x_31 = x_45;
x_32 = x_46;
x_33 = x_47;
x_34 = x_48;
x_35 = x_49;
x_36 = x_50;
x_37 = x_51;
x_38 = lean_box(0);
goto block_43;
}
else
{
uint8_t x_68; 
lean_dec(x_51);
lean_dec_ref(x_50);
lean_dec(x_49);
lean_dec_ref(x_48);
lean_dec(x_47);
lean_dec(x_46);
lean_dec(x_45);
lean_dec_ref(x_44);
lean_dec_ref(x_29);
lean_dec(x_23);
lean_dec_ref(x_13);
x_68 = !lean_is_exclusive(x_67);
if (x_68 == 0)
{
return x_67;
}
else
{
lean_object* x_69; lean_object* x_70; 
x_69 = lean_ctor_get(x_67, 0);
lean_inc(x_69);
lean_dec(x_67);
x_70 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_70, 0, x_69);
return x_70;
}
}
}
else
{
lean_dec(x_24);
lean_dec(x_15);
x_30 = x_44;
x_31 = x_45;
x_32 = x_46;
x_33 = x_47;
x_34 = x_48;
x_35 = x_49;
x_36 = x_50;
x_37 = x_51;
x_38 = lean_box(0);
goto block_43;
}
}
block_97:
{
lean_object* x_82; uint8_t x_83; 
x_82 = lean_ctor_get(x_72, 0);
lean_inc_ref(x_82);
lean_dec_ref(x_72);
x_83 = lean_ctor_get_uint8(x_82, sizeof(void*)*6 + 5);
lean_dec_ref(x_82);
if (x_83 == 0)
{
lean_dec_ref(x_2);
x_44 = x_73;
x_45 = x_74;
x_46 = x_75;
x_47 = x_76;
x_48 = x_77;
x_49 = x_78;
x_50 = x_79;
x_51 = x_80;
x_52 = lean_box(0);
goto block_71;
}
else
{
lean_object* x_84; lean_object* x_85; lean_object* x_86; uint8_t x_87; 
x_84 = lean_ctor_get(x_79, 2);
x_85 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__2;
x_86 = l_Lean_Option_get___redArg(x_12, x_84, x_85);
x_87 = lean_unbox(x_86);
lean_dec(x_86);
if (x_87 == 0)
{
lean_dec_ref(x_2);
x_44 = x_73;
x_45 = x_74;
x_46 = x_75;
x_47 = x_76;
x_48 = x_77;
x_49 = x_78;
x_50 = x_79;
x_51 = x_80;
x_52 = lean_box(0);
goto block_71;
}
else
{
lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; 
x_88 = lp_aesop_Aesop_traceScript___redArg___closed__5;
x_89 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
x_90 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__4;
x_91 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_91, 0, x_90);
lean_ctor_set(x_91, 1, x_2);
lean_inc(x_15);
lean_inc_ref(x_13);
x_92 = l_Lean_logWarning___redArg(x_13, x_88, x_89, x_15, x_91);
lean_inc(x_80);
lean_inc_ref(x_79);
lean_inc(x_78);
lean_inc_ref(x_77);
lean_inc(x_76);
lean_inc(x_75);
lean_inc(x_74);
lean_inc_ref(x_73);
x_93 = lean_apply_9(x_92, x_73, x_74, x_75, x_76, x_77, x_78, x_79, x_80, lean_box(0));
if (lean_obj_tag(x_93) == 0)
{
lean_dec_ref(x_93);
x_44 = x_73;
x_45 = x_74;
x_46 = x_75;
x_47 = x_76;
x_48 = x_77;
x_49 = x_78;
x_50 = x_79;
x_51 = x_80;
x_52 = lean_box(0);
goto block_71;
}
else
{
uint8_t x_94; 
lean_dec(x_80);
lean_dec_ref(x_79);
lean_dec(x_78);
lean_dec_ref(x_77);
lean_dec(x_76);
lean_dec(x_75);
lean_dec(x_74);
lean_dec_ref(x_73);
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
x_94 = !lean_is_exclusive(x_93);
if (x_94 == 0)
{
return x_93;
}
else
{
lean_object* x_95; lean_object* x_96; 
x_95 = lean_ctor_get(x_93, 0);
lean_inc(x_95);
lean_dec(x_93);
x_96 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_96, 0, x_95);
return x_96;
}
}
}
}
}
block_133:
{
lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; 
x_108 = lean_st_ref_get(x_100);
x_109 = lean_ctor_get(x_108, 0);
lean_inc(x_109);
lean_dec(x_108);
x_110 = lean_ctor_get(x_99, 0);
lean_inc_ref(x_110);
x_111 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_111, 0, x_109);
lean_ctor_set(x_111, 1, x_110);
x_112 = lp_aesop_Aesop_treeHasProgress(x_111, x_101, x_102, x_103, x_104, x_105, x_106);
lean_dec_ref(x_111);
if (lean_obj_tag(x_112) == 0)
{
lean_object* x_113; uint8_t x_114; 
x_113 = lean_ctor_get(x_112, 0);
lean_inc(x_113);
lean_dec_ref(x_112);
x_114 = lean_unbox(x_113);
lean_dec(x_113);
if (x_114 == 0)
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; 
x_115 = lean_st_ref_get(x_100);
x_116 = lean_ctor_get(x_115, 0);
lean_inc(x_116);
lean_dec(x_115);
lean_inc_ref(x_110);
x_117 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_117, 0, x_116);
lean_ctor_set(x_117, 1, x_110);
x_118 = lp_aesop_Aesop_getRootMVarId(x_117, x_101, x_102, x_103, x_104, x_105, x_106);
lean_dec_ref(x_117);
if (lean_obj_tag(x_118) == 0)
{
lean_object* x_119; lean_object* x_120; lean_object* x_121; uint8_t x_122; lean_object* x_123; 
x_119 = lean_ctor_get(x_118, 0);
lean_inc(x_119);
lean_dec_ref(x_118);
x_120 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__5;
x_121 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__9;
x_122 = lean_unbox(x_17);
lean_inc(x_106);
lean_inc_ref(x_105);
lean_inc(x_104);
lean_inc_ref(x_103);
lean_inc(x_102);
lean_inc(x_101);
lean_inc(x_100);
lean_inc_ref(x_99);
x_123 = lp_aesop_Aesop_throwAesopEx___redArg(x_1, x_119, x_120, x_122, x_121, x_99, x_100, x_101, x_102, x_103, x_104, x_105, x_106);
lean_dec_ref(x_1);
if (lean_obj_tag(x_123) == 0)
{
lean_dec_ref(x_123);
x_72 = x_98;
x_73 = x_99;
x_74 = x_100;
x_75 = x_101;
x_76 = x_102;
x_77 = x_103;
x_78 = x_104;
x_79 = x_105;
x_80 = x_106;
x_81 = lean_box(0);
goto block_97;
}
else
{
uint8_t x_124; 
lean_dec(x_106);
lean_dec_ref(x_105);
lean_dec(x_104);
lean_dec_ref(x_103);
lean_dec(x_102);
lean_dec(x_101);
lean_dec(x_100);
lean_dec_ref(x_99);
lean_dec_ref(x_98);
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec_ref(x_2);
x_124 = !lean_is_exclusive(x_123);
if (x_124 == 0)
{
return x_123;
}
else
{
lean_object* x_125; lean_object* x_126; 
x_125 = lean_ctor_get(x_123, 0);
lean_inc(x_125);
lean_dec(x_123);
x_126 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_126, 0, x_125);
return x_126;
}
}
}
else
{
uint8_t x_127; 
lean_dec(x_106);
lean_dec_ref(x_105);
lean_dec(x_104);
lean_dec_ref(x_103);
lean_dec(x_102);
lean_dec(x_101);
lean_dec(x_100);
lean_dec_ref(x_99);
lean_dec_ref(x_98);
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_127 = !lean_is_exclusive(x_118);
if (x_127 == 0)
{
return x_118;
}
else
{
lean_object* x_128; lean_object* x_129; 
x_128 = lean_ctor_get(x_118, 0);
lean_inc(x_128);
lean_dec(x_118);
x_129 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_129, 0, x_128);
return x_129;
}
}
}
else
{
lean_dec_ref(x_1);
x_72 = x_98;
x_73 = x_99;
x_74 = x_100;
x_75 = x_101;
x_76 = x_102;
x_77 = x_103;
x_78 = x_104;
x_79 = x_105;
x_80 = x_106;
x_81 = lean_box(0);
goto block_97;
}
}
else
{
uint8_t x_130; 
lean_dec(x_106);
lean_dec_ref(x_105);
lean_dec(x_104);
lean_dec_ref(x_103);
lean_dec(x_102);
lean_dec(x_101);
lean_dec(x_100);
lean_dec_ref(x_99);
lean_dec_ref(x_98);
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_130 = !lean_is_exclusive(x_112);
if (x_130 == 0)
{
return x_112;
}
else
{
lean_object* x_131; lean_object* x_132; 
x_131 = lean_ctor_get(x_112, 0);
lean_inc(x_131);
lean_dec(x_112);
x_132 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_132, 0, x_131);
return x_132;
}
}
}
block_170:
{
lean_object* x_143; 
lean_inc(x_141);
lean_inc_ref(x_140);
lean_inc(x_139);
lean_inc_ref(x_138);
x_143 = lp_aesop_Aesop_traceTree___redArg(x_134, x_135, x_136, x_137, x_138, x_139, x_140, x_141);
if (lean_obj_tag(x_143) == 0)
{
uint8_t x_144; lean_object* x_145; 
lean_dec_ref(x_143);
x_144 = 0;
lean_inc(x_141);
lean_inc_ref(x_140);
lean_inc(x_139);
lean_inc_ref(x_138);
lean_inc(x_137);
lean_inc(x_136);
lean_inc(x_135);
lean_inc_ref(x_134);
x_145 = lp_aesop_Aesop_traceScript___redArg(x_1, x_144, x_134, x_135, x_136, x_137, x_138, x_139, x_140, x_141);
if (lean_obj_tag(x_145) == 0)
{
lean_object* x_146; lean_object* x_147; uint8_t x_148; 
lean_dec_ref(x_145);
x_146 = lean_ctor_get(x_134, 2);
lean_inc_ref(x_146);
x_147 = lean_ctor_get(x_146, 0);
x_148 = lean_ctor_get_uint8(x_147, sizeof(void*)*6 + 4);
if (x_148 == 0)
{
x_98 = x_146;
x_99 = x_134;
x_100 = x_135;
x_101 = x_136;
x_102 = x_137;
x_103 = x_138;
x_104 = x_139;
x_105 = x_140;
x_106 = x_141;
x_107 = lean_box(0);
goto block_133;
}
else
{
lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; 
x_149 = lean_ctor_get(x_134, 0);
x_150 = lean_st_ref_get(x_135);
x_151 = lean_ctor_get(x_150, 0);
lean_inc(x_151);
lean_dec(x_150);
lean_inc_ref(x_149);
x_152 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_152, 0, x_151);
lean_ctor_set(x_152, 1, x_149);
x_153 = lp_aesop_Aesop_getRootMVarId(x_152, x_136, x_137, x_138, x_139, x_140, x_141);
lean_dec_ref(x_152);
if (lean_obj_tag(x_153) == 0)
{
lean_object* x_154; lean_object* x_155; uint8_t x_156; lean_object* x_157; 
x_154 = lean_ctor_get(x_153, 0);
lean_inc(x_154);
lean_dec_ref(x_153);
lean_inc_ref(x_2);
x_155 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_155, 0, x_2);
x_156 = lean_unbox(x_17);
lean_inc(x_141);
lean_inc_ref(x_140);
lean_inc(x_139);
lean_inc_ref(x_138);
lean_inc(x_137);
lean_inc(x_136);
lean_inc(x_135);
lean_inc_ref(x_134);
lean_inc(x_23);
x_157 = lp_aesop_Aesop_throwAesopEx___redArg(x_1, x_154, x_23, x_156, x_155, x_134, x_135, x_136, x_137, x_138, x_139, x_140, x_141);
if (lean_obj_tag(x_157) == 0)
{
lean_dec_ref(x_157);
x_98 = x_146;
x_99 = x_134;
x_100 = x_135;
x_101 = x_136;
x_102 = x_137;
x_103 = x_138;
x_104 = x_139;
x_105 = x_140;
x_106 = x_141;
x_107 = lean_box(0);
goto block_133;
}
else
{
uint8_t x_158; 
lean_dec_ref(x_146);
lean_dec(x_141);
lean_dec_ref(x_140);
lean_dec(x_139);
lean_dec_ref(x_138);
lean_dec(x_137);
lean_dec(x_136);
lean_dec(x_135);
lean_dec_ref(x_134);
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_158 = !lean_is_exclusive(x_157);
if (x_158 == 0)
{
return x_157;
}
else
{
lean_object* x_159; lean_object* x_160; 
x_159 = lean_ctor_get(x_157, 0);
lean_inc(x_159);
lean_dec(x_157);
x_160 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_160, 0, x_159);
return x_160;
}
}
}
else
{
uint8_t x_161; 
lean_dec_ref(x_146);
lean_dec(x_141);
lean_dec_ref(x_140);
lean_dec(x_139);
lean_dec_ref(x_138);
lean_dec(x_137);
lean_dec(x_136);
lean_dec(x_135);
lean_dec_ref(x_134);
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_161 = !lean_is_exclusive(x_153);
if (x_161 == 0)
{
return x_153;
}
else
{
lean_object* x_162; lean_object* x_163; 
x_162 = lean_ctor_get(x_153, 0);
lean_inc(x_162);
lean_dec(x_153);
x_163 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_163, 0, x_162);
return x_163;
}
}
}
}
else
{
uint8_t x_164; 
lean_dec(x_141);
lean_dec_ref(x_140);
lean_dec(x_139);
lean_dec_ref(x_138);
lean_dec(x_137);
lean_dec(x_136);
lean_dec(x_135);
lean_dec_ref(x_134);
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_164 = !lean_is_exclusive(x_145);
if (x_164 == 0)
{
return x_145;
}
else
{
lean_object* x_165; lean_object* x_166; 
x_165 = lean_ctor_get(x_145, 0);
lean_inc(x_165);
lean_dec(x_145);
x_166 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_166, 0, x_165);
return x_166;
}
}
}
else
{
uint8_t x_167; 
lean_dec(x_141);
lean_dec_ref(x_140);
lean_dec(x_139);
lean_dec_ref(x_138);
lean_dec(x_137);
lean_dec(x_136);
lean_dec(x_135);
lean_dec_ref(x_134);
lean_dec_ref(x_29);
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_167 = !lean_is_exclusive(x_143);
if (x_167 == 0)
{
return x_143;
}
else
{
lean_object* x_168; lean_object* x_169; 
x_168 = lean_ctor_get(x_143, 0);
lean_inc(x_168);
lean_dec(x_143);
x_169 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_169, 0, x_168);
return x_169;
}
}
}
}
else
{
uint8_t x_322; 
lean_dec(x_24);
lean_dec(x_23);
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_322 = !lean_is_exclusive(x_27);
if (x_322 == 0)
{
return x_27;
}
else
{
lean_object* x_323; lean_object* x_324; 
x_323 = lean_ctor_get(x_27, 0);
lean_inc(x_323);
lean_dec(x_27);
x_324 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_324, 0, x_323);
return x_324;
}
}
}
else
{
lean_dec(x_17);
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_22;
}
}
else
{
uint8_t x_325; 
lean_dec(x_15);
lean_dec_ref(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_325 = !lean_is_exclusive(x_16);
if (x_325 == 0)
{
return x_16;
}
else
{
lean_object* x_326; lean_object* x_327; 
x_326 = lean_ctor_get(x_16, 0);
lean_inc(x_326);
lean_dec(x_16);
x_327 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_327, 0, x_326);
return x_327;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_handleNonfatalError___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_aesop_Aesop_handleNonfatalError(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_handleNonfatalError___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; lean_object* x_30; uint8_t x_31; lean_object* x_32; uint8_t x_33; 
x_11 = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), x_1);
x_12 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
x_13 = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), x_1);
x_14 = lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
x_15 = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(x_14, x_11);
x_16 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_16, 1, x_13);
lean_ctor_set(x_16, 2, x_15);
x_17 = lean_ctor_get(x_8, 0);
x_18 = lean_ctor_get(x_8, 1);
x_19 = lean_ctor_get(x_8, 2);
x_20 = lean_ctor_get(x_8, 3);
x_21 = lean_ctor_get(x_8, 4);
x_22 = lean_ctor_get(x_8, 5);
x_23 = lean_ctor_get(x_8, 6);
x_24 = lean_ctor_get(x_8, 7);
x_25 = lean_ctor_get(x_8, 8);
x_26 = lean_ctor_get(x_8, 9);
x_27 = lean_ctor_get(x_8, 10);
x_28 = lean_ctor_get(x_8, 11);
x_29 = lean_ctor_get_uint8(x_8, sizeof(void*)*14);
x_30 = lean_ctor_get(x_8, 12);
x_31 = lean_ctor_get_uint8(x_8, sizeof(void*)*14 + 1);
x_32 = lean_ctor_get(x_8, 13);
x_33 = lean_nat_dec_eq(x_20, x_21);
if (x_33 == 0)
{
uint8_t x_34; 
lean_inc_ref(x_32);
lean_inc(x_30);
lean_inc(x_28);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_25);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc_ref(x_18);
lean_inc_ref(x_17);
lean_dec_ref(x_16);
x_34 = !lean_is_exclusive(x_8);
if (x_34 == 0)
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; 
x_35 = lean_ctor_get(x_8, 13);
lean_dec(x_35);
x_36 = lean_ctor_get(x_8, 12);
lean_dec(x_36);
x_37 = lean_ctor_get(x_8, 11);
lean_dec(x_37);
x_38 = lean_ctor_get(x_8, 10);
lean_dec(x_38);
x_39 = lean_ctor_get(x_8, 9);
lean_dec(x_39);
x_40 = lean_ctor_get(x_8, 8);
lean_dec(x_40);
x_41 = lean_ctor_get(x_8, 7);
lean_dec(x_41);
x_42 = lean_ctor_get(x_8, 6);
lean_dec(x_42);
x_43 = lean_ctor_get(x_8, 5);
lean_dec(x_43);
x_44 = lean_ctor_get(x_8, 4);
lean_dec(x_44);
x_45 = lean_ctor_get(x_8, 3);
lean_dec(x_45);
x_46 = lean_ctor_get(x_8, 2);
lean_dec(x_46);
x_47 = lean_ctor_get(x_8, 1);
lean_dec(x_47);
x_48 = lean_ctor_get(x_8, 0);
lean_dec(x_48);
x_49 = lean_st_ref_get(x_3);
lean_dec(x_49);
x_50 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__0;
x_51 = lean_unsigned_to_nat(1u);
x_52 = lean_nat_add(x_20, x_51);
lean_dec(x_20);
lean_ctor_set(x_8, 3, x_52);
x_53 = l_Lean_Core_checkSystem(x_50, x_8, x_9);
if (lean_obj_tag(x_53) == 0)
{
lean_object* x_54; 
lean_dec_ref(x_53);
lean_inc_ref(x_2);
x_54 = lp_aesop_Aesop_checkRootUnprovable___redArg(x_2, x_3, x_4);
if (lean_obj_tag(x_54) == 0)
{
lean_object* x_55; 
x_55 = lean_ctor_get(x_54, 0);
lean_inc(x_55);
lean_dec_ref(x_54);
if (lean_obj_tag(x_55) == 1)
{
lean_object* x_56; lean_object* x_57; 
x_56 = lean_ctor_get(x_55, 0);
lean_inc(x_56);
lean_dec_ref(x_55);
x_57 = lp_aesop_Aesop_handleNonfatalError___redArg(x_1, x_56, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_57;
}
else
{
lean_object* x_58; 
lean_dec(x_55);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_58 = lp_aesop_Aesop_finishIfProven___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_58) == 0)
{
uint8_t x_59; 
x_59 = !lean_is_exclusive(x_58);
if (x_59 == 0)
{
lean_object* x_60; uint8_t x_61; 
x_60 = lean_ctor_get(x_58, 0);
x_61 = lean_unbox(x_60);
lean_dec(x_60);
if (x_61 == 0)
{
lean_object* x_62; 
lean_free_object(x_58);
lean_inc_ref(x_2);
x_62 = lp_aesop_Aesop_checkGoalLimit___redArg(x_2, x_3, x_4);
if (lean_obj_tag(x_62) == 0)
{
lean_object* x_63; 
x_63 = lean_ctor_get(x_62, 0);
lean_inc(x_63);
lean_dec_ref(x_62);
if (lean_obj_tag(x_63) == 1)
{
lean_object* x_64; lean_object* x_65; 
x_64 = lean_ctor_get(x_63, 0);
lean_inc(x_64);
lean_dec_ref(x_63);
x_65 = lp_aesop_Aesop_handleNonfatalError___redArg(x_1, x_64, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_65;
}
else
{
lean_object* x_66; 
lean_dec(x_63);
lean_inc_ref(x_2);
x_66 = lp_aesop_Aesop_checkRappLimit___redArg(x_2, x_3, x_4);
if (lean_obj_tag(x_66) == 0)
{
lean_object* x_67; 
x_67 = lean_ctor_get(x_66, 0);
lean_inc(x_67);
lean_dec_ref(x_66);
if (lean_obj_tag(x_67) == 1)
{
lean_object* x_68; lean_object* x_69; 
x_68 = lean_ctor_get(x_67, 0);
lean_inc(x_68);
lean_dec_ref(x_67);
x_69 = lp_aesop_Aesop_handleNonfatalError___redArg(x_1, x_68, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_69;
}
else
{
lean_object* x_70; 
lean_dec(x_67);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_70 = lp_aesop_Aesop_expandNextGoal___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_70) == 0)
{
lean_object* x_71; lean_object* x_72; 
lean_dec_ref(x_70);
x_71 = lean_st_ref_get(x_3);
lean_dec(x_71);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_72 = lp_aesop_Aesop_checkInvariantsIfEnabled___redArg(x_4, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_72) == 0)
{
lean_object* x_73; 
lean_dec_ref(x_72);
x_73 = lp_aesop_Aesop_incrementIteration___redArg(x_3);
if (lean_obj_tag(x_73) == 0)
{
lean_dec_ref(x_73);
goto _start;
}
else
{
uint8_t x_75; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_75 = !lean_is_exclusive(x_73);
if (x_75 == 0)
{
return x_73;
}
else
{
lean_object* x_76; lean_object* x_77; 
x_76 = lean_ctor_get(x_73, 0);
lean_inc(x_76);
lean_dec(x_73);
x_77 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_77, 0, x_76);
return x_77;
}
}
}
else
{
uint8_t x_78; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_78 = !lean_is_exclusive(x_72);
if (x_78 == 0)
{
return x_72;
}
else
{
lean_object* x_79; lean_object* x_80; 
x_79 = lean_ctor_get(x_72, 0);
lean_inc(x_79);
lean_dec(x_72);
x_80 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_80, 0, x_79);
return x_80;
}
}
}
else
{
uint8_t x_81; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_81 = !lean_is_exclusive(x_70);
if (x_81 == 0)
{
return x_70;
}
else
{
lean_object* x_82; lean_object* x_83; 
x_82 = lean_ctor_get(x_70, 0);
lean_inc(x_82);
lean_dec(x_70);
x_83 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_83, 0, x_82);
return x_83;
}
}
}
}
else
{
uint8_t x_84; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_84 = !lean_is_exclusive(x_66);
if (x_84 == 0)
{
return x_66;
}
else
{
lean_object* x_85; lean_object* x_86; 
x_85 = lean_ctor_get(x_66, 0);
lean_inc(x_85);
lean_dec(x_66);
x_86 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_86, 0, x_85);
return x_86;
}
}
}
}
else
{
uint8_t x_87; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_87 = !lean_is_exclusive(x_62);
if (x_87 == 0)
{
return x_62;
}
else
{
lean_object* x_88; lean_object* x_89; 
x_88 = lean_ctor_get(x_62, 0);
lean_inc(x_88);
lean_dec(x_62);
x_89 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_89, 0, x_88);
return x_89;
}
}
}
else
{
lean_object* x_90; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_90 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__5;
lean_ctor_set(x_58, 0, x_90);
return x_58;
}
}
else
{
lean_object* x_91; uint8_t x_92; 
x_91 = lean_ctor_get(x_58, 0);
lean_inc(x_91);
lean_dec(x_58);
x_92 = lean_unbox(x_91);
lean_dec(x_91);
if (x_92 == 0)
{
lean_object* x_93; 
lean_inc_ref(x_2);
x_93 = lp_aesop_Aesop_checkGoalLimit___redArg(x_2, x_3, x_4);
if (lean_obj_tag(x_93) == 0)
{
lean_object* x_94; 
x_94 = lean_ctor_get(x_93, 0);
lean_inc(x_94);
lean_dec_ref(x_93);
if (lean_obj_tag(x_94) == 1)
{
lean_object* x_95; lean_object* x_96; 
x_95 = lean_ctor_get(x_94, 0);
lean_inc(x_95);
lean_dec_ref(x_94);
x_96 = lp_aesop_Aesop_handleNonfatalError___redArg(x_1, x_95, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_96;
}
else
{
lean_object* x_97; 
lean_dec(x_94);
lean_inc_ref(x_2);
x_97 = lp_aesop_Aesop_checkRappLimit___redArg(x_2, x_3, x_4);
if (lean_obj_tag(x_97) == 0)
{
lean_object* x_98; 
x_98 = lean_ctor_get(x_97, 0);
lean_inc(x_98);
lean_dec_ref(x_97);
if (lean_obj_tag(x_98) == 1)
{
lean_object* x_99; lean_object* x_100; 
x_99 = lean_ctor_get(x_98, 0);
lean_inc(x_99);
lean_dec_ref(x_98);
x_100 = lp_aesop_Aesop_handleNonfatalError___redArg(x_1, x_99, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_100;
}
else
{
lean_object* x_101; 
lean_dec(x_98);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_101 = lp_aesop_Aesop_expandNextGoal___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_101) == 0)
{
lean_object* x_102; lean_object* x_103; 
lean_dec_ref(x_101);
x_102 = lean_st_ref_get(x_3);
lean_dec(x_102);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_103 = lp_aesop_Aesop_checkInvariantsIfEnabled___redArg(x_4, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_103) == 0)
{
lean_object* x_104; 
lean_dec_ref(x_103);
x_104 = lp_aesop_Aesop_incrementIteration___redArg(x_3);
if (lean_obj_tag(x_104) == 0)
{
lean_dec_ref(x_104);
goto _start;
}
else
{
lean_object* x_106; lean_object* x_107; lean_object* x_108; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_106 = lean_ctor_get(x_104, 0);
lean_inc(x_106);
if (lean_is_exclusive(x_104)) {
 lean_ctor_release(x_104, 0);
 x_107 = x_104;
} else {
 lean_dec_ref(x_104);
 x_107 = lean_box(0);
}
if (lean_is_scalar(x_107)) {
 x_108 = lean_alloc_ctor(1, 1, 0);
} else {
 x_108 = x_107;
}
lean_ctor_set(x_108, 0, x_106);
return x_108;
}
}
else
{
lean_object* x_109; lean_object* x_110; lean_object* x_111; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_109 = lean_ctor_get(x_103, 0);
lean_inc(x_109);
if (lean_is_exclusive(x_103)) {
 lean_ctor_release(x_103, 0);
 x_110 = x_103;
} else {
 lean_dec_ref(x_103);
 x_110 = lean_box(0);
}
if (lean_is_scalar(x_110)) {
 x_111 = lean_alloc_ctor(1, 1, 0);
} else {
 x_111 = x_110;
}
lean_ctor_set(x_111, 0, x_109);
return x_111;
}
}
else
{
lean_object* x_112; lean_object* x_113; lean_object* x_114; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_112 = lean_ctor_get(x_101, 0);
lean_inc(x_112);
if (lean_is_exclusive(x_101)) {
 lean_ctor_release(x_101, 0);
 x_113 = x_101;
} else {
 lean_dec_ref(x_101);
 x_113 = lean_box(0);
}
if (lean_is_scalar(x_113)) {
 x_114 = lean_alloc_ctor(1, 1, 0);
} else {
 x_114 = x_113;
}
lean_ctor_set(x_114, 0, x_112);
return x_114;
}
}
}
else
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_115 = lean_ctor_get(x_97, 0);
lean_inc(x_115);
if (lean_is_exclusive(x_97)) {
 lean_ctor_release(x_97, 0);
 x_116 = x_97;
} else {
 lean_dec_ref(x_97);
 x_116 = lean_box(0);
}
if (lean_is_scalar(x_116)) {
 x_117 = lean_alloc_ctor(1, 1, 0);
} else {
 x_117 = x_116;
}
lean_ctor_set(x_117, 0, x_115);
return x_117;
}
}
}
else
{
lean_object* x_118; lean_object* x_119; lean_object* x_120; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_118 = lean_ctor_get(x_93, 0);
lean_inc(x_118);
if (lean_is_exclusive(x_93)) {
 lean_ctor_release(x_93, 0);
 x_119 = x_93;
} else {
 lean_dec_ref(x_93);
 x_119 = lean_box(0);
}
if (lean_is_scalar(x_119)) {
 x_120 = lean_alloc_ctor(1, 1, 0);
} else {
 x_120 = x_119;
}
lean_ctor_set(x_120, 0, x_118);
return x_120;
}
}
else
{
lean_object* x_121; lean_object* x_122; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_121 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__5;
x_122 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_122, 0, x_121);
return x_122;
}
}
}
else
{
uint8_t x_123; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_123 = !lean_is_exclusive(x_58);
if (x_123 == 0)
{
return x_58;
}
else
{
lean_object* x_124; lean_object* x_125; 
x_124 = lean_ctor_get(x_58, 0);
lean_inc(x_124);
lean_dec(x_58);
x_125 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_125, 0, x_124);
return x_125;
}
}
}
}
else
{
uint8_t x_126; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_126 = !lean_is_exclusive(x_54);
if (x_126 == 0)
{
return x_54;
}
else
{
lean_object* x_127; lean_object* x_128; 
x_127 = lean_ctor_get(x_54, 0);
lean_inc(x_127);
lean_dec(x_54);
x_128 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_128, 0, x_127);
return x_128;
}
}
}
else
{
uint8_t x_129; 
lean_dec_ref(x_8);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_129 = !lean_is_exclusive(x_53);
if (x_129 == 0)
{
return x_53;
}
else
{
lean_object* x_130; lean_object* x_131; 
x_130 = lean_ctor_get(x_53, 0);
lean_inc(x_130);
lean_dec(x_53);
x_131 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_131, 0, x_130);
return x_131;
}
}
}
else
{
lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; 
lean_dec(x_8);
x_132 = lean_st_ref_get(x_3);
lean_dec(x_132);
x_133 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__0;
x_134 = lean_unsigned_to_nat(1u);
x_135 = lean_nat_add(x_20, x_134);
lean_dec(x_20);
x_136 = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(x_136, 0, x_17);
lean_ctor_set(x_136, 1, x_18);
lean_ctor_set(x_136, 2, x_19);
lean_ctor_set(x_136, 3, x_135);
lean_ctor_set(x_136, 4, x_21);
lean_ctor_set(x_136, 5, x_22);
lean_ctor_set(x_136, 6, x_23);
lean_ctor_set(x_136, 7, x_24);
lean_ctor_set(x_136, 8, x_25);
lean_ctor_set(x_136, 9, x_26);
lean_ctor_set(x_136, 10, x_27);
lean_ctor_set(x_136, 11, x_28);
lean_ctor_set(x_136, 12, x_30);
lean_ctor_set(x_136, 13, x_32);
lean_ctor_set_uint8(x_136, sizeof(void*)*14, x_29);
lean_ctor_set_uint8(x_136, sizeof(void*)*14 + 1, x_31);
x_137 = l_Lean_Core_checkSystem(x_133, x_136, x_9);
if (lean_obj_tag(x_137) == 0)
{
lean_object* x_138; 
lean_dec_ref(x_137);
lean_inc_ref(x_2);
x_138 = lp_aesop_Aesop_checkRootUnprovable___redArg(x_2, x_3, x_4);
if (lean_obj_tag(x_138) == 0)
{
lean_object* x_139; 
x_139 = lean_ctor_get(x_138, 0);
lean_inc(x_139);
lean_dec_ref(x_138);
if (lean_obj_tag(x_139) == 1)
{
lean_object* x_140; lean_object* x_141; 
x_140 = lean_ctor_get(x_139, 0);
lean_inc(x_140);
lean_dec_ref(x_139);
x_141 = lp_aesop_Aesop_handleNonfatalError___redArg(x_1, x_140, x_2, x_3, x_4, x_5, x_6, x_7, x_136, x_9);
return x_141;
}
else
{
lean_object* x_142; 
lean_dec(x_139);
lean_inc(x_9);
lean_inc_ref(x_136);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_142 = lp_aesop_Aesop_finishIfProven___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_136, x_9);
if (lean_obj_tag(x_142) == 0)
{
lean_object* x_143; lean_object* x_144; uint8_t x_145; 
x_143 = lean_ctor_get(x_142, 0);
lean_inc(x_143);
if (lean_is_exclusive(x_142)) {
 lean_ctor_release(x_142, 0);
 x_144 = x_142;
} else {
 lean_dec_ref(x_142);
 x_144 = lean_box(0);
}
x_145 = lean_unbox(x_143);
lean_dec(x_143);
if (x_145 == 0)
{
lean_object* x_146; 
lean_dec(x_144);
lean_inc_ref(x_2);
x_146 = lp_aesop_Aesop_checkGoalLimit___redArg(x_2, x_3, x_4);
if (lean_obj_tag(x_146) == 0)
{
lean_object* x_147; 
x_147 = lean_ctor_get(x_146, 0);
lean_inc(x_147);
lean_dec_ref(x_146);
if (lean_obj_tag(x_147) == 1)
{
lean_object* x_148; lean_object* x_149; 
x_148 = lean_ctor_get(x_147, 0);
lean_inc(x_148);
lean_dec_ref(x_147);
x_149 = lp_aesop_Aesop_handleNonfatalError___redArg(x_1, x_148, x_2, x_3, x_4, x_5, x_6, x_7, x_136, x_9);
return x_149;
}
else
{
lean_object* x_150; 
lean_dec(x_147);
lean_inc_ref(x_2);
x_150 = lp_aesop_Aesop_checkRappLimit___redArg(x_2, x_3, x_4);
if (lean_obj_tag(x_150) == 0)
{
lean_object* x_151; 
x_151 = lean_ctor_get(x_150, 0);
lean_inc(x_151);
lean_dec_ref(x_150);
if (lean_obj_tag(x_151) == 1)
{
lean_object* x_152; lean_object* x_153; 
x_152 = lean_ctor_get(x_151, 0);
lean_inc(x_152);
lean_dec_ref(x_151);
x_153 = lp_aesop_Aesop_handleNonfatalError___redArg(x_1, x_152, x_2, x_3, x_4, x_5, x_6, x_7, x_136, x_9);
return x_153;
}
else
{
lean_object* x_154; 
lean_dec(x_151);
lean_inc(x_9);
lean_inc_ref(x_136);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_154 = lp_aesop_Aesop_expandNextGoal___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_136, x_9);
if (lean_obj_tag(x_154) == 0)
{
lean_object* x_155; lean_object* x_156; 
lean_dec_ref(x_154);
x_155 = lean_st_ref_get(x_3);
lean_dec(x_155);
lean_inc(x_9);
lean_inc_ref(x_136);
lean_inc(x_7);
lean_inc_ref(x_6);
x_156 = lp_aesop_Aesop_checkInvariantsIfEnabled___redArg(x_4, x_6, x_7, x_136, x_9);
if (lean_obj_tag(x_156) == 0)
{
lean_object* x_157; 
lean_dec_ref(x_156);
x_157 = lp_aesop_Aesop_incrementIteration___redArg(x_3);
if (lean_obj_tag(x_157) == 0)
{
lean_dec_ref(x_157);
x_8 = x_136;
goto _start;
}
else
{
lean_object* x_159; lean_object* x_160; lean_object* x_161; 
lean_dec_ref(x_136);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_159 = lean_ctor_get(x_157, 0);
lean_inc(x_159);
if (lean_is_exclusive(x_157)) {
 lean_ctor_release(x_157, 0);
 x_160 = x_157;
} else {
 lean_dec_ref(x_157);
 x_160 = lean_box(0);
}
if (lean_is_scalar(x_160)) {
 x_161 = lean_alloc_ctor(1, 1, 0);
} else {
 x_161 = x_160;
}
lean_ctor_set(x_161, 0, x_159);
return x_161;
}
}
else
{
lean_object* x_162; lean_object* x_163; lean_object* x_164; 
lean_dec_ref(x_136);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_162 = lean_ctor_get(x_156, 0);
lean_inc(x_162);
if (lean_is_exclusive(x_156)) {
 lean_ctor_release(x_156, 0);
 x_163 = x_156;
} else {
 lean_dec_ref(x_156);
 x_163 = lean_box(0);
}
if (lean_is_scalar(x_163)) {
 x_164 = lean_alloc_ctor(1, 1, 0);
} else {
 x_164 = x_163;
}
lean_ctor_set(x_164, 0, x_162);
return x_164;
}
}
else
{
lean_object* x_165; lean_object* x_166; lean_object* x_167; 
lean_dec_ref(x_136);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_165 = lean_ctor_get(x_154, 0);
lean_inc(x_165);
if (lean_is_exclusive(x_154)) {
 lean_ctor_release(x_154, 0);
 x_166 = x_154;
} else {
 lean_dec_ref(x_154);
 x_166 = lean_box(0);
}
if (lean_is_scalar(x_166)) {
 x_167 = lean_alloc_ctor(1, 1, 0);
} else {
 x_167 = x_166;
}
lean_ctor_set(x_167, 0, x_165);
return x_167;
}
}
}
else
{
lean_object* x_168; lean_object* x_169; lean_object* x_170; 
lean_dec_ref(x_136);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_168 = lean_ctor_get(x_150, 0);
lean_inc(x_168);
if (lean_is_exclusive(x_150)) {
 lean_ctor_release(x_150, 0);
 x_169 = x_150;
} else {
 lean_dec_ref(x_150);
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
}
else
{
lean_object* x_171; lean_object* x_172; lean_object* x_173; 
lean_dec_ref(x_136);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_171 = lean_ctor_get(x_146, 0);
lean_inc(x_171);
if (lean_is_exclusive(x_146)) {
 lean_ctor_release(x_146, 0);
 x_172 = x_146;
} else {
 lean_dec_ref(x_146);
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
lean_object* x_174; lean_object* x_175; 
lean_dec_ref(x_136);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_174 = lp_aesop_Aesop_handleNonfatalError___redArg___closed__5;
if (lean_is_scalar(x_144)) {
 x_175 = lean_alloc_ctor(0, 1, 0);
} else {
 x_175 = x_144;
}
lean_ctor_set(x_175, 0, x_174);
return x_175;
}
}
else
{
lean_object* x_176; lean_object* x_177; lean_object* x_178; 
lean_dec_ref(x_136);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_176 = lean_ctor_get(x_142, 0);
lean_inc(x_176);
if (lean_is_exclusive(x_142)) {
 lean_ctor_release(x_142, 0);
 x_177 = x_142;
} else {
 lean_dec_ref(x_142);
 x_177 = lean_box(0);
}
if (lean_is_scalar(x_177)) {
 x_178 = lean_alloc_ctor(1, 1, 0);
} else {
 x_178 = x_177;
}
lean_ctor_set(x_178, 0, x_176);
return x_178;
}
}
}
else
{
lean_object* x_179; lean_object* x_180; lean_object* x_181; 
lean_dec_ref(x_136);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_179 = lean_ctor_get(x_138, 0);
lean_inc(x_179);
if (lean_is_exclusive(x_138)) {
 lean_ctor_release(x_138, 0);
 x_180 = x_138;
} else {
 lean_dec_ref(x_138);
 x_180 = lean_box(0);
}
if (lean_is_scalar(x_180)) {
 x_181 = lean_alloc_ctor(1, 1, 0);
} else {
 x_181 = x_180;
}
lean_ctor_set(x_181, 0, x_179);
return x_181;
}
}
else
{
lean_object* x_182; lean_object* x_183; lean_object* x_184; 
lean_dec_ref(x_136);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_182 = lean_ctor_get(x_137, 0);
lean_inc(x_182);
if (lean_is_exclusive(x_137)) {
 lean_ctor_release(x_137, 0);
 x_183 = x_137;
} else {
 lean_dec_ref(x_137);
 x_183 = lean_box(0);
}
if (lean_is_scalar(x_183)) {
 x_184 = lean_alloc_ctor(1, 1, 0);
} else {
 x_184 = x_183;
}
lean_ctor_set(x_184, 0, x_182);
return x_184;
}
}
}
else
{
lean_object* x_185; lean_object* x_186; 
lean_dec_ref(x_1);
lean_inc(x_22);
x_185 = l_Lean_throwMaxRecDepthAt___redArg(x_16, x_22);
x_186 = lean_apply_9(x_185, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
return x_186;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_searchLoop___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_searchLoop(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_searchLoop___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; uint8_t x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 2);
x_5 = lp_aesop_Aesop_Check_get(x_4, x_1);
x_6 = lean_box(x_5);
x_7 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(x_1, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; 
x_14 = lp_aesop_Aesop_SearchM_run___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_14;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = l_Lean_KVMap_find(x_1, x_3);
if (lean_obj_tag(x_5) == 0)
{
uint8_t x_6; 
x_6 = lean_unbox(x_4);
return x_6;
}
else
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
lean_dec_ref(x_5);
if (lean_obj_tag(x_7) == 1)
{
uint8_t x_8; 
x_8 = lean_ctor_get_uint8(x_7, 0);
lean_dec_ref(x_7);
return x_8;
}
else
{
uint8_t x_9; 
lean_dec(x_7);
x_9 = lean_unbox(x_4);
return x_9;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_st_ref_get(x_1);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec(x_11);
x_13 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_13);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_15 = lp_aesop_Aesop_collectGoalStatsIfEnabled(x_14, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_14);
if (lean_obj_tag(x_15) == 0)
{
lean_object* x_16; lean_object* x_17; 
lean_dec_ref(x_15);
x_16 = lean_st_ref_get(x_1);
lean_dec(x_16);
x_17 = lp_aesop_Aesop_freeTree___redArg(x_3);
return x_17;
}
else
{
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_11 = lp_aesop_Aesop_searchLoop___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_11) == 0)
{
uint8_t x_12; 
x_12 = !lean_is_exclusive(x_11);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; 
x_13 = lean_ctor_get(x_11, 0);
lean_inc(x_13);
lean_ctor_set_tag(x_11, 1);
x_14 = lp_aesop_Aesop_search___lam__0(x_3, x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_11);
lean_dec_ref(x_11);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec(x_3);
if (lean_obj_tag(x_14) == 0)
{
uint8_t x_15; 
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; 
x_16 = lean_ctor_get(x_14, 0);
lean_dec(x_16);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
else
{
lean_object* x_17; 
lean_dec(x_14);
x_17 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_17, 0, x_13);
return x_17;
}
}
else
{
uint8_t x_18; 
lean_dec(x_13);
x_18 = !lean_is_exclusive(x_14);
if (x_18 == 0)
{
return x_14;
}
else
{
lean_object* x_19; lean_object* x_20; 
x_19 = lean_ctor_get(x_14, 0);
lean_inc(x_19);
lean_dec(x_14);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
return x_20;
}
}
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_21 = lean_ctor_get(x_11, 0);
lean_inc(x_21);
lean_dec(x_11);
lean_inc(x_21);
x_22 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_22, 0, x_21);
x_23 = lp_aesop_Aesop_search___lam__0(x_3, x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_22);
lean_dec_ref(x_22);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec(x_3);
if (lean_obj_tag(x_23) == 0)
{
lean_object* x_24; lean_object* x_25; 
if (lean_is_exclusive(x_23)) {
 lean_ctor_release(x_23, 0);
 x_24 = x_23;
} else {
 lean_dec_ref(x_23);
 x_24 = lean_box(0);
}
if (lean_is_scalar(x_24)) {
 x_25 = lean_alloc_ctor(0, 1, 0);
} else {
 x_25 = x_24;
}
lean_ctor_set(x_25, 0, x_21);
return x_25;
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; 
lean_dec(x_21);
x_26 = lean_ctor_get(x_23, 0);
lean_inc(x_26);
if (lean_is_exclusive(x_23)) {
 lean_ctor_release(x_23, 0);
 x_27 = x_23;
} else {
 lean_dec_ref(x_23);
 x_27 = lean_box(0);
}
if (lean_is_scalar(x_27)) {
 x_28 = lean_alloc_ctor(1, 1, 0);
} else {
 x_28 = x_27;
}
lean_ctor_set(x_28, 0, x_26);
return x_28;
}
}
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_29 = lean_ctor_get(x_11, 0);
lean_inc(x_29);
lean_dec_ref(x_11);
x_30 = lean_box(0);
x_31 = lp_aesop_Aesop_search___lam__0(x_3, x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_30);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec(x_3);
if (lean_obj_tag(x_31) == 0)
{
uint8_t x_32; 
x_32 = !lean_is_exclusive(x_31);
if (x_32 == 0)
{
lean_object* x_33; 
x_33 = lean_ctor_get(x_31, 0);
lean_dec(x_33);
lean_ctor_set_tag(x_31, 1);
lean_ctor_set(x_31, 0, x_29);
return x_31;
}
else
{
lean_object* x_34; 
lean_dec(x_31);
x_34 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_34, 0, x_29);
return x_34;
}
}
else
{
uint8_t x_35; 
lean_dec(x_29);
x_35 = !lean_is_exclusive(x_31);
if (x_35 == 0)
{
return x_31;
}
else
{
lean_object* x_36; lean_object* x_37; 
x_36 = lean_ctor_get(x_31, 0);
lean_inc(x_36);
lean_dec(x_31);
x_37 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_37, 0, x_36);
return x_37;
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_aesop_dev_generateScript;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_Check_script;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_Check_script_steps;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_8; lean_object* x_9; lean_object* x_13; lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_17 = lean_ctor_get(x_5, 2);
x_18 = lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__0;
x_19 = lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0(x_17, x_18);
if (x_19 == 0)
{
uint8_t x_20; 
x_20 = lean_ctor_get_uint8(x_1, sizeof(void*)*6 + 6);
if (x_20 == 0)
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; 
x_21 = lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__1;
x_22 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(x_21, x_5);
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
x_24 = lean_unbox(x_23);
lean_dec(x_23);
if (x_24 == 0)
{
lean_object* x_25; lean_object* x_26; 
lean_dec_ref(x_22);
x_25 = lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__2;
x_26 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(x_25, x_5);
x_13 = x_26;
goto block_16;
}
else
{
x_13 = x_22;
goto block_16;
}
}
else
{
x_8 = x_20;
x_9 = lean_box(0);
goto block_12;
}
}
else
{
x_8 = x_19;
x_9 = lean_box(0);
goto block_12;
}
block_12:
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_10, 0, x_1);
lean_ctor_set(x_10, 1, x_2);
lean_ctor_set_uint8(x_10, sizeof(void*)*2, x_8);
x_11 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
block_16:
{
lean_object* x_14; uint8_t x_15; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lean_unbox(x_14);
lean_dec(x_14);
x_8 = x_15;
x_9 = lean_box(0);
goto block_12;
}
}
}
static lean_object* _init_lp_aesop_Aesop_search___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_traceScript___redArg___lam__2___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_search___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_aesop_Aesop_search___lam__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; 
x_12 = lp_aesop_Aesop_search___closed__0;
lean_inc(x_1);
x_13 = l_Lean_MVarId_checkNotAssigned(x_1, x_12, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; lean_object* x_15; 
lean_dec_ref(x_13);
x_14 = lean_box(0);
x_15 = lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0(x_3, x_14, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_15) == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_48; 
x_48 = lp_aesop_Aesop_Frontend_getDefaultGlobalRuleSets(x_9, x_10);
if (lean_obj_tag(x_48) == 0)
{
lean_object* x_49; lean_object* x_50; 
x_49 = lean_ctor_get(x_48, 0);
lean_inc(x_49);
lean_dec_ref(x_48);
x_50 = lp_aesop_Aesop_mkLocalRuleSet(x_49, x_16, x_9, x_10);
lean_dec(x_49);
if (lean_obj_tag(x_50) == 0)
{
lean_object* x_51; 
x_51 = lean_ctor_get(x_50, 0);
lean_inc(x_51);
lean_dec_ref(x_50);
x_17 = x_51;
x_18 = x_7;
x_19 = x_8;
x_20 = x_9;
x_21 = x_10;
x_22 = lean_box(0);
goto block_47;
}
else
{
uint8_t x_52; 
lean_dec(x_16);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_1);
x_52 = !lean_is_exclusive(x_50);
if (x_52 == 0)
{
return x_50;
}
else
{
lean_object* x_53; lean_object* x_54; 
x_53 = lean_ctor_get(x_50, 0);
lean_inc(x_53);
lean_dec(x_50);
x_54 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_54, 0, x_53);
return x_54;
}
}
}
else
{
uint8_t x_55; 
lean_dec(x_16);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_1);
x_55 = !lean_is_exclusive(x_48);
if (x_55 == 0)
{
return x_48;
}
else
{
lean_object* x_56; lean_object* x_57; 
x_56 = lean_ctor_get(x_48, 0);
lean_inc(x_56);
lean_dec(x_48);
x_57 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_57, 0, x_56);
return x_57;
}
}
}
else
{
lean_object* x_58; 
x_58 = lean_ctor_get(x_2, 0);
lean_inc(x_58);
lean_dec_ref(x_2);
x_17 = x_58;
x_18 = x_7;
x_19 = x_8;
x_20 = x_9;
x_21 = x_10;
x_22 = lean_box(0);
goto block_47;
}
block_47:
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_23 = lean_ctor_get(x_16, 0);
x_24 = lp_aesop_Aesop_Options_queue(x_23);
x_25 = lean_ctor_get(x_24, 1);
lean_inc(x_25);
lean_dec_ref(x_24);
lean_inc(x_25);
x_26 = lean_alloc_closure((void*)(lp_aesop_Aesop_search___lam__1___boxed), 10, 1);
lean_closure_set(x_26, 0, x_25);
x_27 = lean_alloc_closure((void*)(lp_aesop_Aesop_search___lam__2___boxed), 13, 7);
lean_closure_set(x_27, 0, x_25);
lean_closure_set(x_27, 1, x_17);
lean_closure_set(x_27, 2, x_16);
lean_closure_set(x_27, 3, x_4);
lean_closure_set(x_27, 4, x_5);
lean_closure_set(x_27, 5, x_1);
lean_closure_set(x_27, 6, x_26);
x_28 = lp_aesop_Aesop_BaseM_run___redArg(x_27, x_6, x_18, x_19, x_20, x_21);
if (lean_obj_tag(x_28) == 0)
{
uint8_t x_29; 
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_30 = lean_ctor_get(x_28, 0);
x_31 = lean_ctor_get(x_30, 0);
lean_inc(x_31);
x_32 = lean_ctor_get(x_30, 1);
lean_inc(x_32);
lean_dec(x_30);
x_33 = !lean_is_exclusive(x_31);
if (x_33 == 0)
{
lean_object* x_34; 
x_34 = lean_ctor_get(x_31, 1);
lean_dec(x_34);
lean_ctor_set(x_31, 1, x_32);
lean_ctor_set(x_28, 0, x_31);
return x_28;
}
else
{
lean_object* x_35; lean_object* x_36; 
x_35 = lean_ctor_get(x_31, 0);
lean_inc(x_35);
lean_dec(x_31);
x_36 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_36, 0, x_35);
lean_ctor_set(x_36, 1, x_32);
lean_ctor_set(x_28, 0, x_36);
return x_28;
}
}
else
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; 
x_37 = lean_ctor_get(x_28, 0);
lean_inc(x_37);
lean_dec(x_28);
x_38 = lean_ctor_get(x_37, 0);
lean_inc(x_38);
x_39 = lean_ctor_get(x_37, 1);
lean_inc(x_39);
lean_dec(x_37);
x_40 = lean_ctor_get(x_38, 0);
lean_inc(x_40);
if (lean_is_exclusive(x_38)) {
 lean_ctor_release(x_38, 0);
 lean_ctor_release(x_38, 1);
 x_41 = x_38;
} else {
 lean_dec_ref(x_38);
 x_41 = lean_box(0);
}
if (lean_is_scalar(x_41)) {
 x_42 = lean_alloc_ctor(0, 2, 0);
} else {
 x_42 = x_41;
}
lean_ctor_set(x_42, 0, x_40);
lean_ctor_set(x_42, 1, x_39);
x_43 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_43, 0, x_42);
return x_43;
}
}
else
{
uint8_t x_44; 
x_44 = !lean_is_exclusive(x_28);
if (x_44 == 0)
{
return x_28;
}
else
{
lean_object* x_45; lean_object* x_46; 
x_45 = lean_ctor_get(x_28, 0);
lean_inc(x_45);
lean_dec(x_28);
x_46 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_46, 0, x_45);
return x_46;
}
}
}
}
else
{
uint8_t x_59; 
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_2);
lean_dec(x_1);
x_59 = !lean_is_exclusive(x_15);
if (x_59 == 0)
{
return x_15;
}
else
{
lean_object* x_60; lean_object* x_61; 
x_60 = lean_ctor_get(x_15, 0);
lean_inc(x_60);
lean_dec(x_15);
x_61 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_61, 0, x_60);
return x_61;
}
}
}
else
{
uint8_t x_62; 
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_62 = !lean_is_exclusive(x_13);
if (x_62 == 0)
{
return x_13;
}
else
{
lean_object* x_63; lean_object* x_64; 
x_63 = lean_ctor_get(x_13, 0);
lean_inc(x_63);
lean_dec(x_13);
x_64 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_64, 0, x_63);
return x_64;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0(x_1, x_2);
lean_dec_ref(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Aesop_search___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_aesop_Aesop_search(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Check(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Options(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleSet(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_Check(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_Main(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_Expansion(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_ExpandSafePrefix(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_Queue(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Stats(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_Main(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Options(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Expansion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_ExpandSafePrefix(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Queue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Stats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__3 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__3);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__2 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__2);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__4 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__4();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__4);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__6 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__6();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__6);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__5 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__5();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__5);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__7 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__7();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__7);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__9 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__9();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__9);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__8 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__8();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__8);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__10 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__10();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__10);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__12 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__12();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__12);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__11 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__11();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__11);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__13 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__13();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__13);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__19 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__19();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__19);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__18 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__18();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__18);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__20 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__20();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__20);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__22 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__22();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__22);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__21 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__21();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__21);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__23 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__23);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__1 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__14 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__14();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__14);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__15 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__15();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__15);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__0 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__16 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__16();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__16);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__17 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__17();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__17);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__24 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__25 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__25();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__25);
lp_aesop_Aesop_nextActiveGoal___redArg___closed__26 = _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__26();
lean_mark_persistent(lp_aesop_Aesop_nextActiveGoal___redArg___closed__26);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__7 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__7();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__7);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__8 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__8();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__8);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__1 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__1();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__1);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__3 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__3();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__3);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__5 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__5();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__5);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__3 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__3();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__3);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__0 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__0);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__1 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__1();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__1);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__4 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__4();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__4);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__1 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__1();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__1);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__0 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__0);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__1);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__2);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__3 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__3();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__3);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__4 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__4();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__4);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__5 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__5();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__5);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__6 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__6();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__6);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__7 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__7();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__7);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__8 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__8();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__8);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__9 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__9();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__9);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__10 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__10();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__10);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__11 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__11();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__11);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__12 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__12();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__12);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__13 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__13();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__13);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__14 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__14();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__14);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__15 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__15();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__15);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__16 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__16();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__16);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__17 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__17();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__17);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__18 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__18();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__18);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__19 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__19();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__19);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__20 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__20();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__20);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__21 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__21();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__21);
lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__22 = _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__22();
lean_mark_persistent(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___closed__22);
lp_aesop_Aesop_expandNextGoal___redArg___closed__0 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__0);
lp_aesop_Aesop_expandNextGoal___redArg___closed__1 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__1);
lp_aesop_Aesop_expandNextGoal___redArg___closed__2 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__2);
lp_aesop_Aesop_expandNextGoal___redArg___closed__5 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__5();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__5);
lp_aesop_Aesop_expandNextGoal___redArg___closed__3 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__3);
lp_aesop_Aesop_expandNextGoal___redArg___closed__4 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__4();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__4);
lp_aesop_Aesop_expandNextGoal___redArg___closed__6 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__6();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__6);
lp_aesop_Aesop_expandNextGoal___redArg___closed__7 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__7();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__7);
lp_aesop_Aesop_expandNextGoal___redArg___closed__8 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__8();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__8);
lp_aesop_Aesop_expandNextGoal___redArg___closed__9 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__9();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__9);
lp_aesop_Aesop_expandNextGoal___redArg___closed__10 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__10();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__10);
lp_aesop_Aesop_expandNextGoal___redArg___closed__11 = _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__11();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___closed__11);
lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__0 = _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__0();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__0);
lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1 = _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1);
lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0 = _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0);
lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1 = _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1);
lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__2 = _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__2();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__2);
lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3 = _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3);
lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__4 = _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__4();
lean_mark_persistent(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__4);
lp_aesop_Aesop_checkGoalLimit___redArg___closed__0 = _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_checkGoalLimit___redArg___closed__0);
lp_aesop_Aesop_checkGoalLimit___redArg___closed__1 = _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_checkGoalLimit___redArg___closed__1);
lp_aesop_Aesop_checkGoalLimit___redArg___closed__2 = _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_checkGoalLimit___redArg___closed__2);
lp_aesop_Aesop_checkGoalLimit___redArg___closed__3 = _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_checkGoalLimit___redArg___closed__3);
lp_aesop_Aesop_checkRappLimit___redArg___closed__0 = _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_checkRappLimit___redArg___closed__0);
lp_aesop_Aesop_checkRappLimit___redArg___closed__1 = _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_checkRappLimit___redArg___closed__1);
lp_aesop_Aesop_checkRappLimit___redArg___closed__2 = _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_checkRappLimit___redArg___closed__2);
lp_aesop_Aesop_checkRappLimit___redArg___closed__3 = _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_checkRappLimit___redArg___closed__3);
lp_aesop_Aesop_checkRootUnprovable___redArg___closed__0 = _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_checkRootUnprovable___redArg___closed__0);
lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1 = _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1);
lp_aesop_Aesop_checkRootUnprovable___redArg___closed__2 = _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_checkRootUnprovable___redArg___closed__2);
lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3 = _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3);
lp_aesop_Aesop_checkRootUnprovable___redArg___closed__4 = _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__4();
lean_mark_persistent(lp_aesop_Aesop_checkRootUnprovable___redArg___closed__4);
lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5 = _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5();
lean_mark_persistent(lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5);
lp_aesop_Aesop_finalizeProof___redArg___closed__0 = _init_lp_aesop_Aesop_finalizeProof___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___closed__0);
lp_aesop_Aesop_finalizeProof___redArg___closed__1 = _init_lp_aesop_Aesop_finalizeProof___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___closed__1);
lp_aesop_Aesop_finalizeProof___redArg___closed__2 = _init_lp_aesop_Aesop_finalizeProof___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___closed__2);
lp_aesop_Aesop_finalizeProof___redArg___closed__3 = _init_lp_aesop_Aesop_finalizeProof___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___closed__3);
lp_aesop_Aesop_finalizeProof___redArg___closed__4 = _init_lp_aesop_Aesop_finalizeProof___redArg___closed__4();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___closed__4);
lp_aesop_Aesop_finalizeProof___redArg___closed__5 = _init_lp_aesop_Aesop_finalizeProof___redArg___closed__5();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___closed__5);
lp_aesop_Aesop_finalizeProof___redArg___closed__6 = _init_lp_aesop_Aesop_finalizeProof___redArg___closed__6();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___closed__6);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__0 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__0();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__0);
lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__0 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__0();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__0);
lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__2 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__2();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__2);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__4 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__4();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__4);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__11 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__11();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__11);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__10 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__10();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__10);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__9 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__9();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__9);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__8 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__8();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__8);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__7 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__7();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__7);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__6 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__6();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__6);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__5 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__5();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__5);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__12 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__12();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__12);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__13 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__13();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__13);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__14 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__14();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__14);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__15 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__15();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__15);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__17 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__17();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__17);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18);
lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__19 = _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__19();
lean_mark_persistent(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__19);
lp_aesop_Aesop_traceScript___redArg___closed__0 = _init_lp_aesop_Aesop_traceScript___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__0);
lp_aesop_Aesop_traceScript___redArg___closed__1 = _init_lp_aesop_Aesop_traceScript___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__1);
lp_aesop_Aesop_traceScript___redArg___closed__2 = _init_lp_aesop_Aesop_traceScript___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__2);
lp_aesop_Aesop_traceScript___redArg___closed__3 = _init_lp_aesop_Aesop_traceScript___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__3);
lp_aesop_Aesop_traceScript___redArg___closed__4 = _init_lp_aesop_Aesop_traceScript___redArg___closed__4();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__4);
lp_aesop_Aesop_traceScript___redArg___closed__5 = _init_lp_aesop_Aesop_traceScript___redArg___closed__5();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__5);
lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0 = _init_lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0);
lp_aesop_Aesop_traceScript___redArg___lam__1___closed__1 = _init_lp_aesop_Aesop_traceScript___redArg___lam__1___closed__1();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___lam__1___closed__1);
lp_aesop_Aesop_traceScript___redArg___closed__6 = _init_lp_aesop_Aesop_traceScript___redArg___closed__6();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__6);
lp_aesop_Aesop_traceScript___redArg___closed__7 = _init_lp_aesop_Aesop_traceScript___redArg___closed__7();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__7);
lp_aesop_Aesop_traceScript___redArg___closed__8 = _init_lp_aesop_Aesop_traceScript___redArg___closed__8();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__8);
lp_aesop_Aesop_traceScript___redArg___closed__9 = _init_lp_aesop_Aesop_traceScript___redArg___closed__9();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__9);
lp_aesop_Aesop_traceScript___redArg___lam__2___closed__0 = _init_lp_aesop_Aesop_traceScript___redArg___lam__2___closed__0();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___lam__2___closed__0);
lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1 = _init_lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___lam__2___closed__1);
lp_aesop_Aesop_traceScript___redArg___lam__2___closed__2 = _init_lp_aesop_Aesop_traceScript___redArg___lam__2___closed__2();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___lam__2___closed__2);
lp_aesop_Aesop_traceScript___redArg___lam__2___closed__3 = _init_lp_aesop_Aesop_traceScript___redArg___lam__2___closed__3();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___lam__2___closed__3);
lp_aesop_Aesop_traceScript___redArg___closed__10 = _init_lp_aesop_Aesop_traceScript___redArg___closed__10();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__10);
lp_aesop_Aesop_traceScript___redArg___closed__11 = _init_lp_aesop_Aesop_traceScript___redArg___closed__11();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__11);
lp_aesop_Aesop_traceScript___redArg___closed__12 = _init_lp_aesop_Aesop_traceScript___redArg___closed__12();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__12);
lp_aesop_Aesop_traceScript___redArg___closed__13 = _init_lp_aesop_Aesop_traceScript___redArg___closed__13();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__13);
lp_aesop_Aesop_traceScript___redArg___closed__14 = _init_lp_aesop_Aesop_traceScript___redArg___closed__14();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__14);
lp_aesop_Aesop_traceScript___redArg___closed__15 = _init_lp_aesop_Aesop_traceScript___redArg___closed__15();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__15);
lp_aesop_Aesop_traceScript___redArg___closed__16 = _init_lp_aesop_Aesop_traceScript___redArg___closed__16();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__16);
lp_aesop_Aesop_traceScript___redArg___closed__17 = _init_lp_aesop_Aesop_traceScript___redArg___closed__17();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__17);
lp_aesop_Aesop_traceScript___redArg___closed__18 = _init_lp_aesop_Aesop_traceScript___redArg___closed__18();
lean_mark_persistent(lp_aesop_Aesop_traceScript___redArg___closed__18);
lp_aesop_Aesop_traceTree___redArg___closed__0 = _init_lp_aesop_Aesop_traceTree___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_traceTree___redArg___closed__0);
lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0___closed__0 = _init_lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0___closed__0();
lean_mark_persistent(lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0___closed__0);
lp_aesop_Aesop_throwAesopEx___redArg___closed__0 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__0);
lp_aesop_Aesop_throwAesopEx___redArg___closed__1 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__1);
lp_aesop_Aesop_throwAesopEx___redArg___closed__2 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__2);
lp_aesop_Aesop_throwAesopEx___redArg___closed__3 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__3);
lp_aesop_Aesop_throwAesopEx___redArg___closed__4 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__4();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__4);
lp_aesop_Aesop_throwAesopEx___redArg___closed__5 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__5();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__5);
lp_aesop_Aesop_throwAesopEx___redArg___closed__6 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__6();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__6);
lp_aesop_Aesop_throwAesopEx___redArg___closed__7 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__7();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__7);
lp_aesop_Aesop_throwAesopEx___redArg___closed__8 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__8();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__8);
lp_aesop_Aesop_throwAesopEx___redArg___closed__9 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__9();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__9);
lp_aesop_Aesop_throwAesopEx___redArg___closed__10 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__10();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__10);
lp_aesop_Aesop_throwAesopEx___redArg___closed__11 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__11();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__11);
lp_aesop_Aesop_throwAesopEx___redArg___closed__12 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__12();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__12);
lp_aesop_Aesop_throwAesopEx___redArg___closed__13 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__13();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__13);
lp_aesop_Aesop_throwAesopEx___redArg___closed__14 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__14();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__14);
lp_aesop_Aesop_throwAesopEx___redArg___closed__15 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__15();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__15);
lp_aesop_Aesop_throwAesopEx___redArg___closed__16 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__16();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__16);
lp_aesop_Aesop_throwAesopEx___redArg___closed__17 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__17();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__17);
lp_aesop_Aesop_throwAesopEx___redArg___closed__18 = _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__18();
lean_mark_persistent(lp_aesop_Aesop_throwAesopEx___redArg___closed__18);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__0 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__0();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__0);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__1 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__1();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__1);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__2 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__2();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__2);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__3 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__3();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__3);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__4 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__4();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__4);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__5 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__5();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__5);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__6 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__6();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__6);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__7 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__7();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__7);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__8 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__8();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__8);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__9 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__9();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__9);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__10 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__10();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__10);
lp_aesop_Aesop_handleNonfatalError___redArg___closed__11 = _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__11();
lean_mark_persistent(lp_aesop_Aesop_handleNonfatalError___redArg___closed__11);
lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__0 = _init_lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__0();
lean_mark_persistent(lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__0);
lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__1 = _init_lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__1();
lean_mark_persistent(lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__1);
lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__2 = _init_lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__2();
lean_mark_persistent(lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___closed__2);
lp_aesop_Aesop_search___closed__0 = _init_lp_aesop_Aesop_search___closed__0();
lean_mark_persistent(lp_aesop_Aesop_search___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
