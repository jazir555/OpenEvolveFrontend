// Lean compiler output
// Module: runLinter
// Imports: public import Init public import Batteries.Tactic.Lint public import Batteries.Data.Array.Basic public import Lake.CLI.Main
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
static lean_object* lp_batteries_runLinterOnModule___closed__2;
static lean_object* lp_batteries_runLinterOnModule___closed__23;
lean_object* lp_batteries_Batteries_Tactic_Lint_getChecks(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__12(lean_object*, size_t, size_t, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__3;
static lean_object* lp_batteries_runLinterOnModule___closed__8;
static lean_object* lp_batteries_runLinterOnModule___closed__19;
lean_object* l_Lean_Core_getMaxHeartbeats(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* _lean_main(lean_object*);
static lean_object* lp_batteries_determineModulesToLint___closed__0;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_fromJson_x3f(lean_object*);
lean_object* l_instMonadEST___lam__2___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lake_AnsiMode_isEnabled(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0___closed__0;
static lean_object* lp_batteries_runLinterOnModule___closed__0;
lean_object* l_System_FilePath_normalize(lean_object*);
lean_object* l___private_Lake_Load_Resolve_0__Lake_Workspace_runResolveT___at___00Lake_Workspace_materializeDeps_spec__4(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__0;
lean_object* l_Lean_MessageData_toString(lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__13;
extern lean_object* l_Lean_instInhabitedFileMap_default;
uint8_t l_Array_isEmpty___redArg(lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__16;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Array_qpartition___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0(lean_object*, uint8_t, uint8_t, lean_object*);
static lean_object* lp_batteries_resolveDefaultRootModules___closed__2;
uint8_t lean_usize_dec_eq(size_t, size_t);
static lean_object* lp_batteries_runLinterOnModule___closed__31;
LEAN_EXPORT lean_object* lp_batteries_Prod_toJson___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__6(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lake_Workspace_updateAndMaterialize_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6(lean_object*);
static lean_object* lp_batteries_resolveDefaultRootModules___closed__4;
lean_object* lean_io_getenv(lean_object*);
lean_object* l___private_Lake_Load_Resolve_0__Lake_validateManifest(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_resolveDefaultRootModules___closed__3;
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_List_toString___at___00Lean_Environment_AddConstAsyncResult_commitConst_spec__1(lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__37;
lean_object* lean_io_get_num_heartbeats();
static lean_object* lp_batteries_runLinterOnModule___closed__7;
extern lean_object* l_Lean_maxRecDepth;
static lean_object* lp_batteries_runLinterOnModule___closed__18;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_main___boxed(lean_object*, lean_object*);
uint8_t l_System_FilePath_pathExists(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_IO_print___at___00IO_println___at___00__private_Lake_CLI_Main_0__Lake_verifyInstall_spec__1_spec__1(lean_object*);
lean_object* l_Lake_Manifest_load_x3f(lean_object*);
static lean_object* lp_batteries_parseLinterArgs___closed__1;
LEAN_EXPORT lean_object* lp_batteries_parseLinterArgs(lean_object*);
lean_object* l_Lake_OutStream_get(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_parseLinterArgs___lam__0(lean_object*, lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
lean_object* l_Lake_Package_findTargetDecl_x3f(lean_object*, lean_object*);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* lean_io_process_child_wait(lean_object*, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__5;
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule(uint8_t, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__28;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_initSearchPath(lean_object*, lean_object*);
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__0;
size_t lean_usize_of_nat(lean_object*);
static lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__5;
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule_unsafe__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_findSysroot(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__15(size_t, size_t, lean_object*);
static lean_object* lp_batteries_main___closed__0;
lean_object* lean_st_ref_take(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__33;
static lean_object* lp_batteries_runLinterOnModule___closed__29;
static lean_object* lp_batteries_parseLinterArgs___closed__2;
lean_object* l___private_Lake_Load_Resolve_0__Lake_Workspace_writeManifest(lean_object*, lean_object*);
static lean_object* lp_batteries_readJsonFile___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__6;
lean_object* l_Array_empty(lean_object*);
lean_object* l_Lean_Option_get___at___00Lake_Package_mkConfigString_spec__1(lean_object*, lean_object*);
uint8_t l_Lean_Option_get___at___00Lake_Package_mkConfigString_spec__0(lean_object*, lean_object*);
lean_object* l_Lake_Workspace_addPackage(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Prod_fromJson_x3f___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__14(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___at___00runLinterOnModule_spec__13___boxed(lean_object*, lean_object*);
static lean_object* lp_batteries_main___closed__1;
static lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__3;
static lean_object* lp_batteries_runLinterOnModule___closed__34;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_enable_initializer_execution();
static lean_object* lp_batteries_runLinterOnModule___closed__32;
lean_object* l_liftExcept___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lake_logToStream(lean_object*, lean_object*, uint8_t, uint8_t);
static lean_object* lp_batteries_runLinterOnModule___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile___at___00runLinterOnModule_spec__6(lean_object*, lean_object*);
static lean_object* lp_batteries_parseLinterArgs___closed__4;
lean_object* lean_st_ref_get(lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__35;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lake_CliError_toString(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_array_to_list(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13___redArg(lean_object*);
lean_object* lean_io_process_spawn(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___at___00runLinterOnModule_spec__13(lean_object*);
lean_object* l_Lean_Name_getRoot(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_determineModulesToLint(lean_object*);
lean_object* l_Lake_LakeOptions_mkLoadConfig(lean_object*);
lean_object* l_IO_println___at___00__private_Lake_CLI_Main_0__Lake_verifyInstall_spec__1(lean_object*);
lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage___redArg(lean_object*, lean_object*);
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11___closed__0;
static lean_object* lp_batteries_runLinterOnModule___closed__15;
uint8_t lean_name_eq(lean_object*, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__17;
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile___at___00runLinterOnModule_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__20;
static lean_object* lp_batteries_determineModulesToLint___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_updateAndMaterialize___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__1(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_updateAndMaterialize___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_parseLinterArgs___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14(lean_object*);
uint8_t l_Option_instBEq_beq___at___00Lake_Workspace_materializeDeps_spec__10(lean_object*, lean_object*);
extern lean_object* l_Lean_diagnostics;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00main_spec__0(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_parseLinterArgs___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_determineModulesToLint___boxed(lean_object*, lean_object*);
static lean_object* lp_batteries_resolveDefaultRootModules___closed__5;
static lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__2;
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Prod_toJson___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__6___closed__0;
lean_object* l___private_Lake_Load_Resolve_0__Lake_Workspace_updateAndMaterializeCore___at___00Lake_Workspace_updateAndMaterialize_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
static lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__4;
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg___lam__0(lean_object*, lean_object*);
extern lean_object* l_Lean_inheritedTraceOptions;
static lean_object* lp_batteries_runLinterOnModule___closed__9;
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
static lean_object* lp_batteries_runLinterOnModule___closed__24;
lean_object* l_IO_FS_readFile(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__12;
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___lam__0(lean_object*, uint8_t, uint8_t, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__2;
lean_object* lean_array_fget(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00runLinterOnModule_spec__5(lean_object*, size_t, size_t);
lean_object* l_Lake_Manifest_tryLoadEntries(lean_object*);
static lean_object* lp_batteries_determineModulesToLint___closed__3;
static lean_object* lp_batteries_runLinterOnModule___closed__36;
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_resolveDefaultRootModules();
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__15___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_IO_eprintln___at___00Lake_serve_spec__0(lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__11;
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg___lam__0(uint8_t, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_findOLean(lean_object*);
static lean_object* lp_batteries_Prod_fromJson_x3f___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__14___closed__0;
static lean_object* lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__1;
static lean_object* lp_batteries_runLinterOnModule___closed__21;
LEAN_EXPORT lean_object* lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13___redArg___boxed(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lake_loadWorkspace_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*);
lean_object* lean_io_exit(uint8_t);
static lean_object* lp_batteries_runLinterOnModule___closed__22;
lean_object* l_String_toName(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lake_Workspace_materializeDeps_spec__8(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lake_mkRelPathString(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
static lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__1;
static lean_object* lp_batteries_runLinterOnModule___closed__26;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00main_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_resolveDefaultRootModules___closed__0;
static lean_object* lp_batteries_runLinterOnModule___closed__38;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__1;
lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
LEAN_EXPORT lean_object* lp_batteries_resolveDefaultRootModules___boxed(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
size_t lean_array_size(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__10;
lean_object* l_Lean_Name_mkStr1(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00runLinterOnModule_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_IO_FS_writeFile(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_erase___at___00Lean_LocalContext_findFromUserNames_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00runLinterOnModule_spec__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Batteries_Tactic_Lint_lintCore(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule_unsafe__1();
lean_object* lean_array_get_size(lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__14;
lean_object* l_Lake_findInstall_x3f();
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00runLinterOnModule_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_readJsonFile(lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_resolveDefaultRootModules___closed__1;
lean_object* l_Lean_Json_parse(lean_object*);
lean_object* l_Lake_joinRelative(lean_object*, lean_object*);
extern lean_object* l_Lake_LeanExe_keyword;
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l___private_Lake_Load_Workspace_0__Lake_loadWorkspaceRoot(lean_object*, lean_object*);
lean_object* l_Lean_importModules(lean_object*, lean_object*, uint32_t, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*);
static lean_object* lp_batteries_parseLinterArgs___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Json_pretty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__30;
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
static lean_object* lp_batteries_determineModulesToLint___closed__2;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__7(size_t, size_t, lean_object*);
uint8_t l_Lean_Name_lt(lean_object*, lean_object*);
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
static lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__0;
static lean_object* lp_batteries_runLinterOnModule___closed__1;
static lean_object* lp_batteries_runLinterOnModule___closed__4;
static uint8_t lp_batteries_runLinterOnModule___closed__25;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_runLinterOnModule___closed__27;
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_ctor(18, 1, 0);
lean_ctor_set(x_4, 0, x_2);
x_5 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_readJsonFile___redArg___lam__0(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_1(x_2, lean_box(0));
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_readJsonFile___redArg___lam__1(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_5;
}
}
static lean_object* _init_lp_batteries_readJsonFile___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instMonadEST___lam__2___boxed), 3, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_alloc_closure((void*)(lp_batteries_readJsonFile___redArg___lam__0___boxed), 3, 0);
x_5 = lean_alloc_closure((void*)(lp_batteries_readJsonFile___redArg___lam__1___boxed), 4, 0);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
x_7 = l_IO_FS_readFile(x_2);
if (lean_obj_tag(x_7) == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lp_batteries_readJsonFile___redArg___closed__0;
x_10 = l_Lean_Json_parse(x_8);
lean_inc_ref(x_6);
x_11 = l_liftExcept___redArg(x_6, x_9, x_10);
x_12 = lean_apply_1(x_11, lean_box(0));
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lean_apply_1(x_1, x_13);
x_15 = l_liftExcept___redArg(x_6, x_9, x_14);
x_16 = lean_apply_1(x_15, lean_box(0));
return x_16;
}
else
{
uint8_t x_17; 
lean_dec_ref(x_6);
lean_dec_ref(x_1);
x_17 = !lean_is_exclusive(x_12);
if (x_17 == 0)
{
return x_12;
}
else
{
lean_object* x_18; lean_object* x_19; 
x_18 = lean_ctor_get(x_12, 0);
lean_inc(x_18);
lean_dec(x_12);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
}
else
{
uint8_t x_20; 
lean_dec_ref(x_6);
lean_dec_ref(x_1);
x_20 = !lean_is_exclusive(x_7);
if (x_20 == 0)
{
return x_7;
}
else
{
lean_object* x_21; lean_object* x_22; 
x_21 = lean_ctor_get(x_7, 0);
lean_inc(x_21);
lean_dec(x_7);
x_22 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_22, 0, x_21);
return x_22;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_readJsonFile(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_readJsonFile___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_readJsonFile(x_1, x_2, x_3);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_readJsonFile___redArg(x_1, x_2);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; uint32_t x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_apply_1(x_1, x_3);
x_6 = lean_unsigned_to_nat(80u);
x_7 = l_Lean_Json_pretty(x_5, x_6);
x_8 = 10;
x_9 = lean_string_push(x_7, x_8);
x_10 = l_IO_FS_writeFile(x_2, x_9);
lean_dec_ref(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_writeJsonFile___redArg(x_2, x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_writeJsonFile(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_writeJsonFile___redArg(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___lam__0(lean_object* x_1, uint8_t x_2, uint8_t x_3, lean_object* x_4) {
_start:
{
lean_object* x_6; 
x_6 = l_Lake_logToStream(x_4, x_1, x_2, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; uint8_t x_7; lean_object* x_8; 
x_6 = lean_unbox(x_2);
x_7 = lean_unbox(x_3);
x_8 = lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___lam__0(x_1, x_6, x_7, x_4);
lean_dec_ref(x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_updateAndMaterialize___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__1(lean_object* x_1, uint8_t x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, uint8_t x_7) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_box(x_2);
x_10 = lean_box(x_3);
lean_inc_ref(x_1);
x_11 = lean_alloc_closure((void*)(lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___lam__0___boxed), 5, 3);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_9);
lean_closure_set(x_11, 2, x_10);
lean_inc_ref(x_11);
x_12 = l___private_Lake_Load_Resolve_0__Lake_Workspace_updateAndMaterializeCore___at___00Lake_Workspace_updateAndMaterialize_spec__0(x_11, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_13, 1);
lean_inc(x_15);
lean_dec(x_13);
lean_inc(x_14);
x_16 = l___private_Lake_Load_Resolve_0__Lake_Workspace_writeManifest(x_14, x_15);
lean_dec(x_15);
if (lean_obj_tag(x_16) == 0)
{
uint8_t x_17; 
lean_dec_ref(x_1);
x_17 = !lean_is_exclusive(x_16);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; uint8_t x_22; 
x_18 = lean_ctor_get(x_16, 0);
lean_dec(x_18);
x_19 = lean_ctor_get(x_14, 4);
x_20 = lean_unsigned_to_nat(0u);
x_21 = lean_array_get_size(x_19);
x_22 = lean_nat_dec_lt(x_20, x_21);
if (x_22 == 0)
{
lean_dec_ref(x_11);
lean_ctor_set(x_16, 0, x_14);
return x_16;
}
else
{
uint8_t x_23; 
x_23 = lean_nat_dec_le(x_21, x_21);
if (x_23 == 0)
{
lean_dec_ref(x_11);
lean_ctor_set(x_16, 0, x_14);
return x_16;
}
else
{
lean_object* x_24; size_t x_25; size_t x_26; lean_object* x_27; 
lean_free_object(x_16);
x_24 = lean_box(0);
x_25 = 0;
x_26 = lean_usize_of_nat(x_21);
lean_inc(x_14);
x_27 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lake_Workspace_updateAndMaterialize_spec__1(x_19, x_25, x_26, x_24, x_14, x_11);
if (lean_obj_tag(x_27) == 0)
{
uint8_t x_28; 
x_28 = !lean_is_exclusive(x_27);
if (x_28 == 0)
{
lean_object* x_29; 
x_29 = lean_ctor_get(x_27, 0);
lean_dec(x_29);
lean_ctor_set(x_27, 0, x_14);
return x_27;
}
else
{
lean_object* x_30; 
lean_dec(x_27);
x_30 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_30, 0, x_14);
return x_30;
}
}
else
{
uint8_t x_31; 
lean_dec(x_14);
x_31 = !lean_is_exclusive(x_27);
if (x_31 == 0)
{
return x_27;
}
else
{
lean_object* x_32; lean_object* x_33; 
x_32 = lean_ctor_get(x_27, 0);
lean_inc(x_32);
lean_dec(x_27);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_32);
return x_33;
}
}
}
}
}
else
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; uint8_t x_37; 
lean_dec(x_16);
x_34 = lean_ctor_get(x_14, 4);
x_35 = lean_unsigned_to_nat(0u);
x_36 = lean_array_get_size(x_34);
x_37 = lean_nat_dec_lt(x_35, x_36);
if (x_37 == 0)
{
lean_object* x_38; 
lean_dec_ref(x_11);
x_38 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_38, 0, x_14);
return x_38;
}
else
{
uint8_t x_39; 
x_39 = lean_nat_dec_le(x_36, x_36);
if (x_39 == 0)
{
lean_object* x_40; 
lean_dec_ref(x_11);
x_40 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_40, 0, x_14);
return x_40;
}
else
{
lean_object* x_41; size_t x_42; size_t x_43; lean_object* x_44; 
x_41 = lean_box(0);
x_42 = 0;
x_43 = lean_usize_of_nat(x_36);
lean_inc(x_14);
x_44 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lake_Workspace_updateAndMaterialize_spec__1(x_34, x_42, x_43, x_41, x_14, x_11);
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
lean_ctor_set(x_46, 0, x_14);
return x_46;
}
else
{
lean_object* x_47; lean_object* x_48; lean_object* x_49; 
lean_dec(x_14);
x_47 = lean_ctor_get(x_44, 0);
lean_inc(x_47);
if (lean_is_exclusive(x_44)) {
 lean_ctor_release(x_44, 0);
 x_48 = x_44;
} else {
 lean_dec_ref(x_44);
 x_48 = lean_box(0);
}
if (lean_is_scalar(x_48)) {
 x_49 = lean_alloc_ctor(1, 1, 0);
} else {
 x_49 = x_48;
}
lean_ctor_set(x_49, 0, x_47);
return x_49;
}
}
}
}
}
else
{
uint8_t x_50; 
lean_dec(x_14);
lean_dec_ref(x_11);
x_50 = !lean_is_exclusive(x_16);
if (x_50 == 0)
{
lean_object* x_51; lean_object* x_52; uint8_t x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; 
x_51 = lean_ctor_get(x_16, 0);
x_52 = lean_io_error_to_string(x_51);
x_53 = 3;
x_54 = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(x_54, 0, x_52);
lean_ctor_set_uint8(x_54, sizeof(void*)*1, x_53);
x_55 = l_Lake_logToStream(x_54, x_1, x_2, x_3);
lean_dec_ref(x_54);
x_56 = lean_box(0);
lean_ctor_set(x_16, 0, x_56);
return x_16;
}
else
{
lean_object* x_57; lean_object* x_58; uint8_t x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; 
x_57 = lean_ctor_get(x_16, 0);
lean_inc(x_57);
lean_dec(x_16);
x_58 = lean_io_error_to_string(x_57);
x_59 = 3;
x_60 = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(x_60, 0, x_58);
lean_ctor_set_uint8(x_60, sizeof(void*)*1, x_59);
x_61 = l_Lake_logToStream(x_60, x_1, x_2, x_3);
lean_dec_ref(x_60);
x_62 = lean_box(0);
x_63 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_63, 0, x_62);
return x_63;
}
}
}
else
{
uint8_t x_64; 
lean_dec_ref(x_11);
lean_dec_ref(x_1);
x_64 = !lean_is_exclusive(x_12);
if (x_64 == 0)
{
return x_12;
}
else
{
lean_object* x_65; lean_object* x_66; 
x_65 = lean_ctor_get(x_12, 0);
lean_inc(x_65);
lean_dec(x_12);
x_66 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_66, 0, x_65);
return x_66;
}
}
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("lean_lib", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__2;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3(lean_object* x_1, lean_object* x_2, size_t x_3, size_t x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_14; 
x_14 = lean_usize_dec_eq(x_3, x_4);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_28; 
x_15 = lean_ctor_get(x_1, 0);
x_16 = lean_array_uget(x_2, x_3);
x_28 = l_Lake_Package_findTargetDecl_x3f(x_16, x_15);
if (lean_obj_tag(x_28) == 0)
{
goto block_27;
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_29 = lean_ctor_get(x_28, 0);
lean_inc(x_29);
lean_dec_ref(x_28);
x_30 = lean_ctor_get(x_29, 2);
lean_inc(x_30);
x_31 = lean_ctor_get(x_29, 3);
lean_inc(x_31);
lean_dec(x_29);
x_32 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__3;
x_33 = lean_name_eq(x_30, x_32);
lean_dec(x_30);
if (x_33 == 0)
{
lean_dec(x_31);
goto block_27;
}
else
{
lean_object* x_34; lean_object* x_35; 
lean_dec(x_16);
x_34 = lean_ctor_get(x_31, 2);
lean_inc_ref(x_34);
lean_dec(x_31);
x_35 = l_Array_append___redArg(x_5, x_34);
lean_dec_ref(x_34);
x_6 = x_35;
goto block_10;
}
}
block_27:
{
lean_object* x_17; 
x_17 = l_Lake_Package_findTargetDecl_x3f(x_16, x_15);
lean_dec(x_16);
if (lean_obj_tag(x_17) == 0)
{
goto block_13;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; uint8_t x_22; 
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lean_ctor_get(x_18, 2);
lean_inc(x_19);
x_20 = lean_ctor_get(x_18, 3);
lean_inc(x_20);
lean_dec(x_18);
x_21 = l_Lake_LeanExe_keyword;
x_22 = lean_name_eq(x_19, x_21);
lean_dec(x_19);
if (x_22 == 0)
{
lean_dec(x_20);
goto block_13;
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_23 = lean_ctor_get(x_20, 2);
lean_inc(x_23);
lean_dec(x_20);
x_24 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__1;
x_25 = lean_array_push(x_24, x_23);
x_26 = l_Array_append___redArg(x_5, x_25);
lean_dec_ref(x_25);
x_6 = x_26;
goto block_10;
}
}
}
}
else
{
return x_5;
}
block_10:
{
size_t x_7; size_t x_8; 
x_7 = 1;
x_8 = lean_usize_add(x_3, x_7);
x_3 = x_8;
x_5 = x_6;
goto _start;
}
block_13:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__0;
x_12 = l_Array_append___redArg(x_5, x_11);
x_6 = x_12;
goto block_10;
}
}
}
static lean_object* _init_lp_batteries_resolveDefaultRootModules___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(".", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_batteries_resolveDefaultRootModules___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("lakefile", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_batteries_resolveDefaultRootModules___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_resolveDefaultRootModules___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_batteries_resolveDefaultRootModules___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("failed to load Lake workspace", 29, 29);
return x_1;
}
}
static lean_object* _init_lp_batteries_resolveDefaultRootModules___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries_resolveDefaultRootModules___closed__4;
x_2 = lean_alloc_ctor(18, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("missing manifest; use `lake update` to generate one", 51, 51);
return x_1;
}
}
static lean_object* _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__1() {
_start:
{
lean_object* x_1; uint8_t x_2; lean_object* x_3; 
x_1 = lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__0;
x_2 = 3;
x_3 = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set_uint8(x_3, sizeof(void*)*1, x_2);
return x_3;
}
}
static lean_object* _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(".lake", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("package-overrides.json", 22, 22);
return x_1;
}
}
static lean_object* _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("manifest out of date: packages directory changed; use `lake update` to rebuild the manifest (warning: this will update ALL workspace dependencies)", 146, 146);
return x_1;
}
}
static lean_object* _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__5() {
_start:
{
lean_object* x_1; uint8_t x_2; lean_object* x_3; 
x_1 = lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__4;
x_2 = 2;
x_3 = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set_uint8(x_3, sizeof(void*)*1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0(lean_object* x_1, uint8_t x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, uint8_t x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; uint8_t x_109; 
x_85 = lean_ctor_get(x_5, 2);
lean_inc(x_85);
x_86 = lean_ctor_get(x_5, 3);
lean_inc_ref(x_86);
lean_dec_ref(x_5);
x_99 = lean_box(x_2);
x_100 = lean_box(x_3);
lean_inc_ref(x_1);
x_101 = lean_alloc_closure((void*)(lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___lam__0___boxed), 5, 3);
lean_closure_set(x_101, 0, x_1);
lean_closure_set(x_101, 1, x_99);
lean_closure_set(x_101, 2, x_100);
x_109 = l_Array_isEmpty___redArg(x_86);
if (x_109 == 0)
{
lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; uint8_t x_116; 
x_110 = lean_ctor_get(x_4, 0);
x_111 = lean_ctor_get(x_110, 6);
x_112 = lean_ctor_get(x_111, 0);
lean_inc_ref(x_112);
x_113 = l_System_FilePath_normalize(x_112);
x_114 = l_Lake_mkRelPathString(x_113);
x_115 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_115, 0, x_114);
x_116 = l_Option_instBEq_beq___at___00Lake_Workspace_materializeDeps_spec__10(x_85, x_115);
lean_dec_ref(x_115);
if (x_116 == 0)
{
lean_object* x_117; lean_object* x_118; 
x_117 = lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__5;
x_118 = l_Lake_logToStream(x_117, x_1, x_2, x_3);
x_102 = lean_box(0);
goto block_108;
}
else
{
lean_dec_ref(x_1);
x_102 = lean_box(0);
goto block_108;
}
}
else
{
lean_dec_ref(x_1);
x_102 = lean_box(0);
goto block_108;
}
block_19:
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_15 = l_Lake_Workspace_addPackage(x_12, x_4);
x_16 = lean_ctor_get(x_15, 0);
lean_inc_ref(x_16);
x_17 = lean_box(0);
x_18 = l___private_Lake_Load_Resolve_0__Lake_Workspace_runResolveT___at___00Lake_Workspace_materializeDeps_spec__4(x_10, x_11, x_6, x_7, x_15, x_16, x_17, x_13);
return x_18;
}
block_31:
{
if (lean_obj_tag(x_25) == 0)
{
lean_dec_ref(x_22);
x_10 = x_25;
x_11 = x_21;
x_12 = x_20;
x_13 = x_23;
x_14 = lean_box(0);
goto block_19;
}
else
{
uint8_t x_26; 
x_26 = l_Array_isEmpty___redArg(x_22);
lean_dec_ref(x_22);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
lean_dec_ref(x_21);
lean_dec_ref(x_20);
lean_dec(x_6);
lean_dec_ref(x_4);
x_27 = lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__1;
x_28 = lean_apply_2(x_23, x_27, lean_box(0));
x_29 = lean_box(0);
x_30 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_30, 0, x_29);
return x_30;
}
else
{
x_10 = x_25;
x_11 = x_21;
x_12 = x_20;
x_13 = x_23;
x_14 = lean_box(0);
goto block_19;
}
}
}
block_45:
{
lean_object* x_39; uint8_t x_40; 
x_39 = lean_array_get_size(x_8);
x_40 = lean_nat_dec_lt(x_37, x_39);
if (x_40 == 0)
{
x_20 = x_33;
x_21 = x_32;
x_22 = x_35;
x_23 = x_34;
x_24 = lean_box(0);
x_25 = x_38;
goto block_31;
}
else
{
uint8_t x_41; 
x_41 = lean_nat_dec_le(x_39, x_39);
if (x_41 == 0)
{
x_20 = x_33;
x_21 = x_32;
x_22 = x_35;
x_23 = x_34;
x_24 = lean_box(0);
x_25 = x_38;
goto block_31;
}
else
{
size_t x_42; size_t x_43; lean_object* x_44; 
x_42 = 0;
x_43 = lean_usize_of_nat(x_39);
x_44 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lake_Workspace_materializeDeps_spec__8(x_8, x_42, x_43, x_38);
x_20 = x_33;
x_21 = x_32;
x_22 = x_35;
x_23 = x_34;
x_24 = lean_box(0);
x_25 = x_44;
goto block_31;
}
}
}
block_84:
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; 
x_51 = lean_ctor_get(x_4, 0);
x_52 = lean_ctor_get(x_51, 4);
x_53 = lean_ctor_get(x_51, 12);
lean_inc_ref(x_47);
x_54 = l___private_Lake_Load_Resolve_0__Lake_validateManifest(x_50, x_53, x_47);
if (lean_obj_tag(x_54) == 0)
{
lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; 
lean_dec_ref(x_54);
x_55 = lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__2;
lean_inc_ref(x_52);
x_56 = l_Lake_joinRelative(x_52, x_55);
x_57 = lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__3;
x_58 = l_Lake_joinRelative(x_56, x_57);
x_59 = l_Lake_Manifest_tryLoadEntries(x_58);
if (lean_obj_tag(x_59) == 0)
{
lean_object* x_60; lean_object* x_61; uint8_t x_62; 
x_60 = lean_ctor_get(x_59, 0);
lean_inc(x_60);
lean_dec_ref(x_59);
x_61 = lean_array_get_size(x_60);
x_62 = lean_nat_dec_lt(x_49, x_61);
if (x_62 == 0)
{
lean_dec(x_60);
lean_inc_ref(x_53);
lean_inc_ref(x_51);
x_32 = x_46;
x_33 = x_51;
x_34 = x_47;
x_35 = x_53;
x_36 = lean_box(0);
x_37 = x_49;
x_38 = x_50;
goto block_45;
}
else
{
uint8_t x_63; 
x_63 = lean_nat_dec_le(x_61, x_61);
if (x_63 == 0)
{
lean_dec(x_60);
lean_inc_ref(x_53);
lean_inc_ref(x_51);
x_32 = x_46;
x_33 = x_51;
x_34 = x_47;
x_35 = x_53;
x_36 = lean_box(0);
x_37 = x_49;
x_38 = x_50;
goto block_45;
}
else
{
size_t x_64; size_t x_65; lean_object* x_66; 
x_64 = 0;
x_65 = lean_usize_of_nat(x_61);
x_66 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lake_Workspace_materializeDeps_spec__8(x_60, x_64, x_65, x_50);
lean_dec(x_60);
lean_inc_ref(x_53);
lean_inc_ref(x_51);
x_32 = x_46;
x_33 = x_51;
x_34 = x_47;
x_35 = x_53;
x_36 = lean_box(0);
x_37 = x_49;
x_38 = x_66;
goto block_45;
}
}
}
else
{
uint8_t x_67; 
lean_dec(x_50);
lean_dec_ref(x_46);
lean_dec(x_6);
lean_dec_ref(x_4);
x_67 = !lean_is_exclusive(x_59);
if (x_67 == 0)
{
lean_object* x_68; lean_object* x_69; uint8_t x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; 
x_68 = lean_ctor_get(x_59, 0);
x_69 = lean_io_error_to_string(x_68);
x_70 = 3;
x_71 = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(x_71, 0, x_69);
lean_ctor_set_uint8(x_71, sizeof(void*)*1, x_70);
x_72 = lean_apply_2(x_47, x_71, lean_box(0));
x_73 = lean_box(0);
lean_ctor_set(x_59, 0, x_73);
return x_59;
}
else
{
lean_object* x_74; lean_object* x_75; uint8_t x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; 
x_74 = lean_ctor_get(x_59, 0);
lean_inc(x_74);
lean_dec(x_59);
x_75 = lean_io_error_to_string(x_74);
x_76 = 3;
x_77 = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(x_77, 0, x_75);
lean_ctor_set_uint8(x_77, sizeof(void*)*1, x_76);
x_78 = lean_apply_2(x_47, x_77, lean_box(0));
x_79 = lean_box(0);
x_80 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_80, 0, x_79);
return x_80;
}
}
}
else
{
uint8_t x_81; 
lean_dec(x_50);
lean_dec_ref(x_47);
lean_dec_ref(x_46);
lean_dec(x_6);
lean_dec_ref(x_4);
x_81 = !lean_is_exclusive(x_54);
if (x_81 == 0)
{
return x_54;
}
else
{
lean_object* x_82; lean_object* x_83; 
x_82 = lean_ctor_get(x_54, 0);
lean_inc(x_82);
lean_dec(x_54);
x_83 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_83, 0, x_82);
return x_83;
}
}
}
block_98:
{
lean_object* x_90; lean_object* x_91; lean_object* x_92; uint8_t x_93; 
x_90 = lean_box(1);
x_91 = lean_unsigned_to_nat(0u);
x_92 = lean_array_get_size(x_86);
x_93 = lean_nat_dec_lt(x_91, x_92);
if (x_93 == 0)
{
lean_dec_ref(x_86);
x_46 = x_89;
x_47 = x_87;
x_48 = lean_box(0);
x_49 = x_91;
x_50 = x_90;
goto block_84;
}
else
{
uint8_t x_94; 
x_94 = lean_nat_dec_le(x_92, x_92);
if (x_94 == 0)
{
lean_dec_ref(x_86);
x_46 = x_89;
x_47 = x_87;
x_48 = lean_box(0);
x_49 = x_91;
x_50 = x_90;
goto block_84;
}
else
{
size_t x_95; size_t x_96; lean_object* x_97; 
x_95 = 0;
x_96 = lean_usize_of_nat(x_92);
x_97 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lake_Workspace_materializeDeps_spec__8(x_86, x_95, x_96, x_90);
lean_dec_ref(x_86);
x_46 = x_89;
x_47 = x_87;
x_48 = lean_box(0);
x_49 = x_91;
x_50 = x_97;
goto block_84;
}
}
}
block_108:
{
if (lean_obj_tag(x_85) == 0)
{
lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; 
x_103 = lean_ctor_get(x_4, 0);
x_104 = lean_ctor_get(x_103, 6);
x_105 = lean_ctor_get(x_104, 0);
lean_inc_ref(x_105);
x_106 = l_System_FilePath_normalize(x_105);
x_87 = x_101;
x_88 = lean_box(0);
x_89 = x_106;
goto block_98;
}
else
{
lean_object* x_107; 
x_107 = lean_ctor_get(x_85, 0);
lean_inc(x_107);
lean_dec_ref(x_85);
x_87 = x_101;
x_88 = lean_box(0);
x_89 = x_107;
goto block_98;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0(lean_object* x_1, uint8_t x_2, uint8_t x_3, lean_object* x_4) {
_start:
{
lean_object* x_6; lean_object* x_10; lean_object* x_11; uint8_t x_12; uint8_t x_13; uint8_t x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_10 = lean_ctor_get(x_4, 9);
lean_inc_ref(x_10);
x_11 = lean_ctor_get(x_4, 11);
lean_inc(x_11);
x_12 = lean_ctor_get_uint8(x_4, sizeof(void*)*14);
x_13 = lean_ctor_get_uint8(x_4, sizeof(void*)*14 + 1);
x_14 = lean_ctor_get_uint8(x_4, sizeof(void*)*14 + 2);
x_15 = lean_unsigned_to_nat(0u);
x_16 = lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0___closed__0;
x_17 = l___private_Lake_Load_Workspace_0__Lake_loadWorkspaceRoot(x_4, x_16);
x_18 = lean_box(x_2);
x_19 = lean_box(x_3);
lean_inc_ref(x_1);
x_20 = lean_alloc_closure((void*)(lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___lam__0___boxed), 5, 3);
lean_closure_set(x_20, 0, x_1);
lean_closure_set(x_20, 1, x_18);
lean_closure_set(x_20, 2, x_19);
if (lean_obj_tag(x_17) == 0)
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_51; uint8_t x_52; 
x_21 = lean_ctor_get(x_17, 0);
lean_inc(x_21);
x_22 = lean_ctor_get(x_17, 1);
lean_inc(x_22);
lean_dec_ref(x_17);
x_51 = lean_array_get_size(x_22);
x_52 = lean_nat_dec_lt(x_15, x_51);
if (x_52 == 0)
{
lean_dec(x_22);
lean_dec_ref(x_20);
x_23 = lean_box(0);
goto block_50;
}
else
{
uint8_t x_53; 
x_53 = lean_nat_dec_le(x_51, x_51);
if (x_53 == 0)
{
lean_dec(x_22);
lean_dec_ref(x_20);
x_23 = lean_box(0);
goto block_50;
}
else
{
lean_object* x_54; size_t x_55; size_t x_56; lean_object* x_57; 
x_54 = lean_box(0);
x_55 = 0;
x_56 = lean_usize_of_nat(x_51);
x_57 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lake_loadWorkspace_spec__0(x_22, x_55, x_56, x_54, x_20);
lean_dec(x_22);
if (lean_obj_tag(x_57) == 0)
{
lean_dec_ref(x_57);
x_23 = lean_box(0);
goto block_50;
}
else
{
uint8_t x_58; 
lean_dec(x_21);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_1);
x_58 = !lean_is_exclusive(x_57);
if (x_58 == 0)
{
return x_57;
}
else
{
lean_object* x_59; lean_object* x_60; 
x_59 = lean_ctor_get(x_57, 0);
lean_inc(x_59);
lean_dec(x_57);
x_60 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_60, 0, x_59);
return x_60;
}
}
}
}
block_50:
{
if (x_13 == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_24 = lean_ctor_get(x_21, 0);
x_25 = lean_ctor_get(x_24, 4);
x_26 = lean_ctor_get(x_24, 9);
lean_inc_ref(x_26);
lean_inc_ref(x_25);
x_27 = l_Lake_joinRelative(x_25, x_26);
x_28 = l_Lake_Manifest_load_x3f(x_27);
if (lean_obj_tag(x_28) == 0)
{
lean_object* x_29; 
x_29 = lean_ctor_get(x_28, 0);
lean_inc(x_29);
lean_dec_ref(x_28);
if (lean_obj_tag(x_29) == 1)
{
lean_object* x_30; lean_object* x_31; 
x_30 = lean_ctor_get(x_29, 0);
lean_inc(x_30);
lean_dec_ref(x_29);
x_31 = lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0(x_1, x_2, x_3, x_21, x_30, x_11, x_12, x_10);
lean_dec_ref(x_10);
return x_31;
}
else
{
lean_object* x_32; lean_object* x_33; 
lean_dec(x_29);
lean_dec_ref(x_10);
x_32 = lean_box(1);
x_33 = lp_batteries_Lake_Workspace_updateAndMaterialize___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__1(x_1, x_2, x_3, x_21, x_32, x_11, x_14);
return x_33;
}
}
else
{
uint8_t x_34; 
lean_dec(x_21);
lean_dec(x_11);
lean_dec_ref(x_10);
x_34 = !lean_is_exclusive(x_28);
if (x_34 == 0)
{
lean_object* x_35; lean_object* x_36; uint8_t x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_35 = lean_ctor_get(x_28, 0);
x_36 = lean_io_error_to_string(x_35);
x_37 = 3;
x_38 = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(x_38, 0, x_36);
lean_ctor_set_uint8(x_38, sizeof(void*)*1, x_37);
x_39 = l_Lake_logToStream(x_38, x_1, x_2, x_3);
lean_dec_ref(x_38);
x_40 = lean_box(0);
lean_ctor_set(x_28, 0, x_40);
return x_28;
}
else
{
lean_object* x_41; lean_object* x_42; uint8_t x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; 
x_41 = lean_ctor_get(x_28, 0);
lean_inc(x_41);
lean_dec(x_28);
x_42 = lean_io_error_to_string(x_41);
x_43 = 3;
x_44 = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(x_44, 0, x_42);
lean_ctor_set_uint8(x_44, sizeof(void*)*1, x_43);
x_45 = l_Lake_logToStream(x_44, x_1, x_2, x_3);
lean_dec_ref(x_44);
x_46 = lean_box(0);
x_47 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_47, 0, x_46);
return x_47;
}
}
}
else
{
lean_object* x_48; lean_object* x_49; 
lean_dec_ref(x_10);
x_48 = lean_box(1);
x_49 = lp_batteries_Lake_Workspace_updateAndMaterialize___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__1(x_1, x_2, x_3, x_21, x_48, x_11, x_14);
return x_49;
}
}
}
else
{
lean_object* x_61; lean_object* x_62; uint8_t x_63; 
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_1);
x_61 = lean_ctor_get(x_17, 1);
lean_inc(x_61);
lean_dec_ref(x_17);
x_62 = lean_array_get_size(x_61);
x_63 = lean_nat_dec_lt(x_15, x_62);
if (x_63 == 0)
{
lean_object* x_64; lean_object* x_65; 
lean_dec(x_61);
lean_dec_ref(x_20);
x_64 = lean_box(0);
x_65 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_65, 0, x_64);
return x_65;
}
else
{
uint8_t x_66; 
x_66 = lean_nat_dec_le(x_62, x_62);
if (x_66 == 0)
{
lean_dec(x_61);
lean_dec_ref(x_20);
x_6 = lean_box(0);
goto block_9;
}
else
{
lean_object* x_67; size_t x_68; size_t x_69; lean_object* x_70; 
x_67 = lean_box(0);
x_68 = 0;
x_69 = lean_usize_of_nat(x_62);
x_70 = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lake_loadWorkspace_spec__0(x_61, x_68, x_69, x_67, x_20);
lean_dec(x_61);
if (lean_obj_tag(x_70) == 0)
{
lean_dec_ref(x_70);
x_6 = lean_box(0);
goto block_9;
}
else
{
uint8_t x_71; 
x_71 = !lean_is_exclusive(x_70);
if (x_71 == 0)
{
return x_70;
}
else
{
lean_object* x_72; lean_object* x_73; 
x_72 = lean_ctor_get(x_70, 0);
lean_inc(x_72);
lean_dec(x_70);
x_73 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_73, 0, x_72);
return x_73;
}
}
}
}
}
block_9:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_box(0);
x_8 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_resolveDefaultRootModules() {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; uint8_t x_14; uint8_t x_15; lean_object* x_16; uint8_t x_17; uint8_t x_18; uint8_t x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_2 = l_Lake_findInstall_x3f();
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
lean_dec(x_3);
x_7 = lean_box(0);
x_8 = lp_batteries_resolveDefaultRootModules___closed__0;
x_9 = lp_batteries_resolveDefaultRootModules___closed__1;
x_10 = lean_box(1);
x_11 = lean_unsigned_to_nat(0u);
x_12 = lp_batteries_resolveDefaultRootModules___closed__2;
x_13 = 0;
x_14 = 1;
x_15 = 1;
x_16 = lean_box(0);
x_17 = 3;
x_18 = 0;
x_19 = 0;
x_20 = lean_unsigned_to_nat(100u);
x_21 = lean_alloc_ctor(0, 17, 14);
lean_ctor_set(x_21, 0, x_7);
lean_ctor_set(x_21, 1, x_8);
lean_ctor_set(x_21, 2, x_9);
lean_ctor_set(x_21, 3, x_4);
lean_ctor_set(x_21, 4, x_5);
lean_ctor_set(x_21, 5, x_6);
lean_ctor_set(x_21, 6, x_10);
lean_ctor_set(x_21, 7, x_12);
lean_ctor_set(x_21, 8, x_7);
lean_ctor_set(x_21, 9, x_16);
lean_ctor_set(x_21, 10, x_16);
lean_ctor_set(x_21, 11, x_16);
lean_ctor_set(x_21, 12, x_16);
lean_ctor_set(x_21, 13, x_16);
lean_ctor_set(x_21, 14, x_16);
lean_ctor_set(x_21, 15, x_16);
lean_ctor_set(x_21, 16, x_20);
lean_ctor_set_uint8(x_21, sizeof(void*)*17, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 1, x_14);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 2, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 3, x_15);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 4, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 5, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 6, x_15);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 7, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 8, x_17);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 9, x_18);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 10, x_19);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 11, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 12, x_13);
lean_ctor_set_uint8(x_21, sizeof(void*)*17 + 13, x_13);
x_22 = l_Lake_LakeOptions_mkLoadConfig(x_21);
if (lean_obj_tag(x_22) == 0)
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; uint8_t x_27; lean_object* x_28; 
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec_ref(x_22);
x_24 = lean_box(1);
x_25 = l_Lake_OutStream_get(x_24);
lean_inc_ref(x_25);
x_26 = l_Lake_AnsiMode_isEnabled(x_25, x_18);
x_27 = 1;
x_28 = lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0(x_25, x_27, x_26, x_23);
if (lean_obj_tag(x_28) == 0)
{
uint8_t x_29; 
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; uint8_t x_35; 
x_30 = lean_ctor_get(x_28, 0);
x_31 = lean_ctor_get(x_30, 0);
x_32 = lean_ctor_get(x_31, 15);
lean_inc_ref(x_32);
x_33 = lp_batteries_resolveDefaultRootModules___closed__3;
x_34 = lean_array_get_size(x_32);
x_35 = lean_nat_dec_lt(x_11, x_34);
if (x_35 == 0)
{
lean_dec_ref(x_32);
lean_dec(x_30);
lean_ctor_set(x_28, 0, x_33);
return x_28;
}
else
{
uint8_t x_36; 
x_36 = lean_nat_dec_le(x_34, x_34);
if (x_36 == 0)
{
lean_dec_ref(x_32);
lean_dec(x_30);
lean_ctor_set(x_28, 0, x_33);
return x_28;
}
else
{
size_t x_37; size_t x_38; lean_object* x_39; 
x_37 = 0;
x_38 = lean_usize_of_nat(x_34);
x_39 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3(x_30, x_32, x_37, x_38, x_33);
lean_dec_ref(x_32);
lean_dec(x_30);
lean_ctor_set(x_28, 0, x_39);
return x_28;
}
}
}
else
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; uint8_t x_45; 
x_40 = lean_ctor_get(x_28, 0);
lean_inc(x_40);
lean_dec(x_28);
x_41 = lean_ctor_get(x_40, 0);
x_42 = lean_ctor_get(x_41, 15);
lean_inc_ref(x_42);
x_43 = lp_batteries_resolveDefaultRootModules___closed__3;
x_44 = lean_array_get_size(x_42);
x_45 = lean_nat_dec_lt(x_11, x_44);
if (x_45 == 0)
{
lean_object* x_46; 
lean_dec_ref(x_42);
lean_dec(x_40);
x_46 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_46, 0, x_43);
return x_46;
}
else
{
uint8_t x_47; 
x_47 = lean_nat_dec_le(x_44, x_44);
if (x_47 == 0)
{
lean_object* x_48; 
lean_dec_ref(x_42);
lean_dec(x_40);
x_48 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_48, 0, x_43);
return x_48;
}
else
{
size_t x_49; size_t x_50; lean_object* x_51; lean_object* x_52; 
x_49 = 0;
x_50 = lean_usize_of_nat(x_44);
x_51 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3(x_40, x_42, x_49, x_50, x_43);
lean_dec_ref(x_42);
lean_dec(x_40);
x_52 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_52, 0, x_51);
return x_52;
}
}
}
}
else
{
uint8_t x_53; 
x_53 = !lean_is_exclusive(x_28);
if (x_53 == 0)
{
lean_object* x_54; lean_object* x_55; 
x_54 = lean_ctor_get(x_28, 0);
lean_dec(x_54);
x_55 = lp_batteries_resolveDefaultRootModules___closed__5;
lean_ctor_set(x_28, 0, x_55);
return x_28;
}
else
{
lean_object* x_56; lean_object* x_57; 
lean_dec(x_28);
x_56 = lp_batteries_resolveDefaultRootModules___closed__5;
x_57 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_57, 0, x_56);
return x_57;
}
}
}
else
{
uint8_t x_58; 
x_58 = !lean_is_exclusive(x_22);
if (x_58 == 0)
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; 
x_59 = lean_ctor_get(x_22, 0);
x_60 = l_Lake_CliError_toString(x_59);
x_61 = lean_alloc_ctor(18, 1, 0);
lean_ctor_set(x_61, 0, x_60);
lean_ctor_set(x_22, 0, x_61);
return x_22;
}
else
{
lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; 
x_62 = lean_ctor_get(x_22, 0);
lean_inc(x_62);
lean_dec(x_22);
x_63 = l_Lake_CliError_toString(x_62);
x_64 = lean_alloc_ctor(18, 1, 0);
lean_ctor_set(x_64, 0, x_63);
x_65 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_65, 0, x_64);
return x_65;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_updateAndMaterialize___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; uint8_t x_10; uint8_t x_11; lean_object* x_12; 
x_9 = lean_unbox(x_2);
x_10 = lean_unbox(x_3);
x_11 = lean_unbox(x_7);
x_12 = lp_batteries_Lake_Workspace_updateAndMaterialize___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__1(x_1, x_9, x_10, x_4, x_5, x_6, x_11);
lean_dec(x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
size_t x_6; size_t x_7; lean_object* x_8; 
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_8 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3(x_1, x_2, x_6, x_7, x_5);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_resolveDefaultRootModules___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_resolveDefaultRootModules();
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; uint8_t x_7; lean_object* x_8; 
x_6 = lean_unbox(x_2);
x_7 = lean_unbox(x_3);
x_8 = lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0(x_1, x_6, x_7, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; uint8_t x_11; uint8_t x_12; lean_object* x_13; 
x_10 = lean_unbox(x_2);
x_11 = lean_unbox(x_3);
x_12 = lean_unbox(x_7);
x_13 = lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0(x_1, x_10, x_11, x_4, x_5, x_6, x_12, x_8);
lean_dec_ref(x_8);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_batteries_parseLinterArgs___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; lean_object* x_5; 
x_3 = 0;
x_4 = lean_box(x_3);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_1);
return x_5;
}
}
static lean_object* _init_lp_batteries_parseLinterArgs___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("cannot convert module to Name", 29, 29);
return x_1;
}
}
static lean_object* _init_lp_batteries_parseLinterArgs___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries_parseLinterArgs___closed__0;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_parseLinterArgs___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("cannot parse arguments", 22, 22);
return x_1;
}
}
static lean_object* _init_lp_batteries_parseLinterArgs___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries_parseLinterArgs___closed__2;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_parseLinterArgs___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("--update", 8, 8);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_parseLinterArgs(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; lean_object* x_26; 
if (lean_obj_tag(x_1) == 1)
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; uint8_t x_34; 
x_31 = lean_ctor_get(x_1, 0);
x_32 = lean_ctor_get(x_1, 1);
x_33 = lp_batteries_parseLinterArgs___closed__4;
x_34 = lean_string_dec_eq(x_31, x_33);
if (x_34 == 0)
{
lean_object* x_35; 
lean_inc_ref(x_1);
x_35 = lp_batteries_parseLinterArgs___lam__0(x_1, x_1);
lean_dec_ref(x_1);
x_26 = x_35;
goto block_30;
}
else
{
lean_inc(x_32);
lean_dec_ref(x_1);
x_2 = x_34;
x_3 = x_32;
goto block_25;
}
}
else
{
lean_object* x_36; 
lean_inc(x_1);
x_36 = lp_batteries_parseLinterArgs___lam__0(x_1, x_1);
lean_dec(x_1);
x_26 = x_36;
goto block_30;
}
block_25:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_box(0);
x_5 = lean_box(x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
else
{
lean_object* x_8; 
x_8 = lean_ctor_get(x_3, 1);
if (lean_obj_tag(x_8) == 0)
{
uint8_t x_9; 
x_9 = !lean_is_exclusive(x_3);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_10 = lean_ctor_get(x_3, 0);
x_11 = lean_ctor_get(x_3, 1);
lean_dec(x_11);
x_12 = l_String_toName(x_10);
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; 
lean_free_object(x_3);
x_13 = lp_batteries_parseLinterArgs___closed__1;
return x_13;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_12);
x_15 = lean_box(x_2);
lean_ctor_set_tag(x_3, 0);
lean_ctor_set(x_3, 1, x_14);
lean_ctor_set(x_3, 0, x_15);
x_16 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_16, 0, x_3);
return x_16;
}
}
else
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_ctor_get(x_3, 0);
lean_inc(x_17);
lean_dec(x_3);
x_18 = l_String_toName(x_17);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; 
x_19 = lp_batteries_parseLinterArgs___closed__1;
return x_19;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_18);
x_21 = lean_box(x_2);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_21);
lean_ctor_set(x_22, 1, x_20);
x_23 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_23, 0, x_22);
return x_23;
}
}
}
else
{
lean_object* x_24; 
lean_dec_ref(x_3);
x_24 = lp_batteries_parseLinterArgs___closed__3;
return x_24;
}
}
}
block_30:
{
lean_object* x_27; lean_object* x_28; uint8_t x_29; 
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
x_28 = lean_ctor_get(x_26, 1);
lean_inc(x_28);
lean_dec_ref(x_26);
x_29 = lean_unbox(x_27);
lean_dec(x_27);
x_2 = x_29;
x_3 = x_28;
goto block_25;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_parseLinterArgs___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_parseLinterArgs___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
static lean_object* _init_lp_batteries_determineModulesToLint___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Automatically detecting modules to lint", 39, 39);
return x_1;
}
}
static lean_object* _init_lp_batteries_determineModulesToLint___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Default modules: ", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_batteries_determineModulesToLint___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("#", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_batteries_determineModulesToLint___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Running linter on specified module: ", 36, 36);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_determineModulesToLint(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_batteries_determineModulesToLint___closed__0;
x_4 = l_IO_println___at___00__private_Lake_CLI_Main_0__Lake_verifyInstall_spec__1(x_3);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; 
lean_dec_ref(x_4);
x_5 = lp_batteries_resolveDefaultRootModules();
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lp_batteries_determineModulesToLint___closed__1;
x_8 = lp_batteries_determineModulesToLint___closed__2;
lean_inc(x_6);
x_9 = lean_array_to_list(x_6);
x_10 = l_List_toString___at___00Lean_Environment_AddConstAsyncResult_commitConst_spec__1(x_9);
x_11 = lean_string_append(x_8, x_10);
lean_dec_ref(x_10);
x_12 = lean_string_append(x_7, x_11);
lean_dec_ref(x_11);
x_13 = l_IO_println___at___00__private_Lake_CLI_Main_0__Lake_verifyInstall_spec__1(x_12);
if (lean_obj_tag(x_13) == 0)
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_13);
if (x_14 == 0)
{
lean_object* x_15; 
x_15 = lean_ctor_get(x_13, 0);
lean_dec(x_15);
lean_ctor_set(x_13, 0, x_6);
return x_13;
}
else
{
lean_object* x_16; 
lean_dec(x_13);
x_16 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_16, 0, x_6);
return x_16;
}
}
else
{
uint8_t x_17; 
lean_dec(x_6);
x_17 = !lean_is_exclusive(x_13);
if (x_17 == 0)
{
return x_13;
}
else
{
lean_object* x_18; lean_object* x_19; 
x_18 = lean_ctor_get(x_13, 0);
lean_inc(x_18);
lean_dec(x_13);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
}
else
{
return x_5;
}
}
else
{
uint8_t x_20; 
x_20 = !lean_is_exclusive(x_4);
if (x_20 == 0)
{
return x_4;
}
else
{
lean_object* x_21; lean_object* x_22; 
x_21 = lean_ctor_get(x_4, 0);
lean_inc(x_21);
lean_dec(x_4);
x_22 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_22, 0, x_21);
return x_22;
}
}
}
else
{
lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_23 = lean_ctor_get(x_1, 0);
lean_inc(x_23);
lean_dec_ref(x_1);
x_24 = lp_batteries_determineModulesToLint___closed__3;
x_25 = 1;
lean_inc(x_23);
x_26 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_23, x_25);
x_27 = lean_string_append(x_24, x_26);
lean_dec_ref(x_26);
x_28 = l_IO_println___at___00__private_Lake_CLI_Main_0__Lake_verifyInstall_spec__1(x_27);
if (lean_obj_tag(x_28) == 0)
{
uint8_t x_29; 
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_30 = lean_ctor_get(x_28, 0);
lean_dec(x_30);
x_31 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__1;
x_32 = lean_array_push(x_31, x_23);
lean_ctor_set(x_28, 0, x_32);
return x_28;
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; 
lean_dec(x_28);
x_33 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__1;
x_34 = lean_array_push(x_33, x_23);
x_35 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_35, 0, x_34);
return x_35;
}
}
else
{
uint8_t x_36; 
lean_dec(x_23);
x_36 = !lean_is_exclusive(x_28);
if (x_36 == 0)
{
return x_28;
}
else
{
lean_object* x_37; lean_object* x_38; 
x_37 = lean_ctor_get(x_28, 0);
lean_inc(x_37);
lean_dec(x_28);
x_38 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_38, 0, x_37);
return x_38;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_determineModulesToLint___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_determineModulesToLint(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule_unsafe__1() {
_start:
{
lean_object* x_2; 
x_2 = lean_enable_initializer_execution();
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule_unsafe__1___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_runLinterOnModule_unsafe__1();
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___redArg(lean_object* x_1, lean_object* x_2, size_t x_3, size_t x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_11; 
x_11 = lean_usize_dec_eq(x_3, x_4);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_12 = lean_array_uget(x_2, x_3);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_12, 1);
lean_inc(x_14);
lean_dec(x_12);
x_15 = lean_ctor_get(x_1, 1);
x_16 = lean_name_eq(x_15, x_13);
lean_dec(x_13);
if (x_16 == 0)
{
lean_dec(x_14);
x_6 = x_5;
goto block_10;
}
else
{
lean_object* x_17; 
x_17 = l_Std_DHashMap_Internal_Raw_u2080_erase___at___00Lean_LocalContext_findFromUserNames_spec__2___redArg(x_5, x_14);
lean_dec(x_14);
x_6 = x_17;
goto block_10;
}
}
else
{
return x_5;
}
block_10:
{
size_t x_7; size_t x_8; 
x_7 = 1;
x_8 = lean_usize_add(x_3, x_7);
x_3 = x_8;
x_5 = x_6;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, size_t x_5, size_t x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___redArg(x_1, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg___lam__0(uint8_t x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
x_7 = lean_ctor_get(x_4, 0);
x_8 = lean_ctor_get(x_4, 1);
x_9 = l_Lean_Name_lt(x_5, x_7);
if (x_9 == 0)
{
uint8_t x_10; 
x_10 = lean_name_eq(x_5, x_7);
if (x_10 == 0)
{
return x_1;
}
else
{
uint8_t x_11; 
x_11 = l_Lean_Name_lt(x_6, x_8);
return x_11;
}
}
else
{
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; uint8_t x_6; uint8_t x_7; lean_object* x_8; 
x_5 = lean_unbox(x_1);
x_6 = lean_unbox(x_2);
x_7 = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg___lam__0(x_5, x_6, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_8 = lean_box(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg(uint8_t x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = lean_nat_dec_lt(x_4, x_5);
if (x_6 == 0)
{
lean_dec(x_4);
return x_3;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_7 = lean_box(x_1);
x_8 = lean_box(x_2);
x_9 = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_9, 0, x_7);
lean_closure_set(x_9, 1, x_8);
lean_inc(x_4);
x_10 = l_Array_qpartition___redArg(x_3, x_9, x_4, x_5);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_10, 1);
lean_inc(x_12);
lean_dec_ref(x_10);
x_13 = lean_nat_dec_le(x_5, x_11);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg(x_1, x_2, x_12, x_4, x_11);
x_15 = lean_unsigned_to_nat(1u);
x_16 = lean_nat_add(x_11, x_15);
lean_dec(x_11);
x_3 = x_14;
x_4 = x_16;
goto _start;
}
else
{
lean_dec(x_11);
lean_dec(x_4);
return x_12;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10(uint8_t x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg(x_1, x_2, x_4, x_5, x_6);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; 
lean_ctor_set_tag(x_1, 18);
x_4 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_4, 0, x_1);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec(x_1);
x_6 = lean_alloc_ctor(18, 1, 0);
lean_ctor_set(x_6, 0, x_5);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
}
else
{
uint8_t x_8; 
x_8 = !lean_is_exclusive(x_1);
if (x_8 == 0)
{
lean_ctor_set_tag(x_1, 0);
return x_1;
}
else
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_ctor_get(x_1, 0);
lean_inc(x_9);
lean_dec(x_1);
x_10 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13___redArg(x_2);
return x_4;
}
}
static lean_object* _init_lp_batteries_Prod_toJson___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__6___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Prod_toJson___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__6(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = 1;
x_5 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_2, x_4);
x_6 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_6, 0, x_5);
x_7 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_3, x_4);
x_8 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_8, 0, x_7);
x_9 = lp_batteries_Prod_toJson___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__6___closed__0;
x_10 = lean_array_push(x_9, x_6);
x_11 = lean_array_push(x_10, x_8);
x_12 = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__7(size_t x_1, size_t x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_usize_dec_lt(x_2, x_1);
if (x_4 == 0)
{
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; size_t x_9; size_t x_10; lean_object* x_11; 
x_5 = lean_array_uget(x_3, x_2);
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_array_uset(x_3, x_2, x_6);
x_8 = lp_batteries_Prod_toJson___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__6(x_5);
x_9 = 1;
x_10 = lean_usize_add(x_2, x_9);
x_11 = lean_array_uset(x_7, x_2, x_8);
x_2 = x_10;
x_3 = x_11;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6(lean_object* x_1) {
_start:
{
size_t x_2; size_t x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_array_size(x_1);
x_3 = 0;
x_4 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__7(x_2, x_3, x_1);
x_5 = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile___at___00runLinterOnModule_spec__6(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint32_t x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lp_batteries_Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6(x_2);
x_5 = lean_unsigned_to_nat(80u);
x_6 = l_Lean_Json_pretty(x_4, x_5);
x_7 = 10;
x_8 = lean_string_push(x_6, x_7);
x_9 = l_IO_FS_writeFile(x_1, x_8);
lean_dec_ref(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00runLinterOnModule_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
return x_2;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_3, 0);
x_5 = lean_ctor_get(x_3, 2);
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_inc(x_6);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
x_8 = lean_array_push(x_2, x_7);
x_2 = x_8;
x_3 = x_5;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__1(lean_object* x_1, lean_object* x_2, size_t x_3, size_t x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = lean_usize_dec_eq(x_3, x_4);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; size_t x_9; size_t x_10; 
x_7 = lean_array_uget(x_2, x_3);
x_8 = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00runLinterOnModule_spec__0(x_1, x_5, x_7);
lean_dec(x_7);
x_9 = 1;
x_10 = lean_usize_add(x_3, x_9);
x_3 = x_10;
x_5 = x_8;
goto _start;
}
else
{
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__12(lean_object* x_1, size_t x_2, size_t x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = lean_usize_dec_eq(x_2, x_3);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; size_t x_8; size_t x_9; 
x_6 = lean_array_uget(x_1, x_2);
x_7 = l_Array_append___redArg(x_4, x_6);
lean_dec(x_6);
x_8 = 1;
x_9 = lean_usize_add(x_2, x_8);
x_2 = x_9;
x_4 = x_7;
goto _start;
}
else
{
return x_4;
}
}
}
static lean_object* _init_lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("expected JSON array, got '", 26, 26);
return x_1;
}
}
static lean_object* _init_lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("'", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_batteries_Prod_fromJson_x3f___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__14___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("expected pair, got '", 20, 20);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_Prod_fromJson_x3f___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__14(lean_object* x_1) {
_start:
{
lean_object* x_2; 
if (lean_obj_tag(x_1) == 4)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_11 = lean_ctor_get(x_1, 0);
x_12 = lean_array_get_size(x_11);
x_13 = lean_unsigned_to_nat(2u);
x_14 = lean_nat_dec_eq(x_12, x_13);
if (x_14 == 0)
{
x_2 = x_1;
goto block_10;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_inc_ref(x_11);
lean_dec_ref(x_1);
x_15 = lean_unsigned_to_nat(0u);
x_16 = lean_array_fget_borrowed(x_11, x_15);
lean_inc(x_16);
x_17 = l_Lean_Name_fromJson_x3f(x_16);
if (lean_obj_tag(x_17) == 0)
{
uint8_t x_18; 
lean_dec_ref(x_11);
x_18 = !lean_is_exclusive(x_17);
if (x_18 == 0)
{
return x_17;
}
else
{
lean_object* x_19; lean_object* x_20; 
x_19 = lean_ctor_get(x_17, 0);
lean_inc(x_19);
lean_dec(x_17);
x_20 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_20, 0, x_19);
return x_20;
}
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_21 = lean_ctor_get(x_17, 0);
lean_inc(x_21);
lean_dec_ref(x_17);
x_22 = lean_unsigned_to_nat(1u);
x_23 = lean_array_fget(x_11, x_22);
lean_dec_ref(x_11);
x_24 = l_Lean_Name_fromJson_x3f(x_23);
if (lean_obj_tag(x_24) == 0)
{
uint8_t x_25; 
lean_dec(x_21);
x_25 = !lean_is_exclusive(x_24);
if (x_25 == 0)
{
return x_24;
}
else
{
lean_object* x_26; lean_object* x_27; 
x_26 = lean_ctor_get(x_24, 0);
lean_inc(x_26);
lean_dec(x_24);
x_27 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_27, 0, x_26);
return x_27;
}
}
else
{
uint8_t x_28; 
x_28 = !lean_is_exclusive(x_24);
if (x_28 == 0)
{
lean_object* x_29; lean_object* x_30; 
x_29 = lean_ctor_get(x_24, 0);
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_21);
lean_ctor_set(x_30, 1, x_29);
lean_ctor_set(x_24, 0, x_30);
return x_24;
}
else
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_31 = lean_ctor_get(x_24, 0);
lean_inc(x_31);
lean_dec(x_24);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_21);
lean_ctor_set(x_32, 1, x_31);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_32);
return x_33;
}
}
}
}
}
else
{
x_2 = x_1;
goto block_10;
}
block_10:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lp_batteries_Prod_fromJson_x3f___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__14___closed__0;
x_4 = lean_unsigned_to_nat(80u);
x_5 = l_Lean_Json_pretty(x_2, x_4);
x_6 = lean_string_append(x_3, x_5);
lean_dec_ref(x_5);
x_7 = lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__1;
x_8 = lean_string_append(x_6, x_7);
x_9 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__15(size_t x_1, size_t x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_usize_dec_lt(x_2, x_1);
if (x_4 == 0)
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_5, 0, x_3);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_array_uget(x_3, x_2);
x_7 = lp_batteries_Prod_fromJson_x3f___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__14(x_6);
if (lean_obj_tag(x_7) == 0)
{
uint8_t x_8; 
lean_dec_ref(x_3);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
return x_7;
}
else
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_ctor_get(x_7, 0);
lean_inc(x_9);
lean_dec(x_7);
x_10 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; size_t x_14; size_t x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_7, 0);
lean_inc(x_11);
lean_dec_ref(x_7);
x_12 = lean_unsigned_to_nat(0u);
x_13 = lean_array_uset(x_3, x_2, x_12);
x_14 = 1;
x_15 = lean_usize_add(x_2, x_14);
x_16 = lean_array_uset(x_13, x_2, x_11);
x_2 = x_15;
x_3 = x_16;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 4)
{
lean_object* x_2; size_t x_3; size_t x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lean_array_size(x_2);
x_4 = 0;
x_5 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__15(x_3, x_4, x_2);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__0;
x_7 = lean_unsigned_to_nat(80u);
x_8 = l_Lean_Json_pretty(x_1, x_7);
x_9 = lean_string_append(x_6, x_8);
lean_dec_ref(x_8);
x_10 = lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__1;
x_11 = lean_string_append(x_9, x_10);
x_12 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___at___00runLinterOnModule_spec__13(lean_object* x_1) {
_start:
{
lean_object* x_3; 
x_3 = l_IO_FS_readFile(x_1);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = l_Lean_Json_parse(x_4);
x_6 = lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13___redArg(x_5);
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14(x_7);
x_9 = lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13___redArg(x_8);
return x_9;
}
else
{
uint8_t x_10; 
x_10 = !lean_is_exclusive(x_6);
if (x_10 == 0)
{
return x_6;
}
else
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_ctor_get(x_6, 0);
lean_inc(x_11);
lean_dec(x_6);
x_12 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
}
}
else
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_3);
if (x_13 == 0)
{
return x_3;
}
else
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_3, 0);
lean_inc(x_14);
lean_dec(x_3);
x_15 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00runLinterOnModule_spec__5(lean_object* x_1, size_t x_2, size_t x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_usize_dec_eq(x_2, x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; lean_object* x_9; uint8_t x_10; 
x_5 = lean_array_uget(x_1, x_2);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec(x_6);
x_8 = 1;
x_9 = lean_unsigned_to_nat(0u);
x_10 = lean_nat_dec_eq(x_7, x_9);
lean_dec(x_7);
if (x_10 == 0)
{
return x_8;
}
else
{
if (x_4 == 0)
{
size_t x_11; size_t x_12; 
x_11 = 1;
x_12 = lean_usize_add(x_2, x_11);
x_2 = x_12;
goto _start;
}
else
{
return x_8;
}
}
}
else
{
uint8_t x_14; 
x_14 = 0;
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, size_t x_5, size_t x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; 
x_8 = lean_usize_dec_lt(x_6, x_5);
if (x_8 == 0)
{
return x_7;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_20; uint8_t x_21; 
x_9 = lean_array_uget(x_7, x_6);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_9, 1);
lean_inc(x_11);
x_12 = lean_unsigned_to_nat(0u);
x_13 = lean_array_uset(x_7, x_6, x_12);
x_20 = lean_array_get_size(x_1);
x_21 = lean_nat_dec_lt(x_2, x_20);
if (x_21 == 0)
{
lean_dec(x_11);
lean_dec(x_10);
x_14 = x_9;
goto block_19;
}
else
{
uint8_t x_22; 
x_22 = lean_nat_dec_le(x_20, x_20);
if (x_22 == 0)
{
lean_dec(x_11);
lean_dec(x_10);
x_14 = x_9;
goto block_19;
}
else
{
uint8_t x_23; 
x_23 = !lean_is_exclusive(x_9);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; size_t x_26; size_t x_27; lean_object* x_28; 
x_24 = lean_ctor_get(x_9, 1);
lean_dec(x_24);
x_25 = lean_ctor_get(x_9, 0);
lean_dec(x_25);
x_26 = 0;
x_27 = lean_usize_of_nat(x_20);
x_28 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___redArg(x_10, x_1, x_26, x_27, x_11);
lean_ctor_set(x_9, 1, x_28);
x_14 = x_9;
goto block_19;
}
else
{
size_t x_29; size_t x_30; lean_object* x_31; lean_object* x_32; 
lean_dec(x_9);
x_29 = 0;
x_30 = lean_usize_of_nat(x_20);
x_31 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___redArg(x_10, x_1, x_29, x_30, x_11);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_10);
lean_ctor_set(x_32, 1, x_31);
x_14 = x_32;
goto block_19;
}
}
}
block_19:
{
size_t x_15; size_t x_16; lean_object* x_17; 
x_15 = 1;
x_16 = lean_usize_add(x_6, x_15);
x_17 = lean_array_uset(x_13, x_6, x_14);
x_6 = x_16;
x_7 = x_17;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, size_t x_5, size_t x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; 
x_8 = lean_usize_dec_lt(x_6, x_5);
if (x_8 == 0)
{
return x_7;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_20; uint8_t x_21; 
x_9 = lean_array_uget(x_7, x_6);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_9, 1);
lean_inc(x_11);
x_12 = lean_unsigned_to_nat(0u);
x_13 = lean_array_uset(x_7, x_6, x_12);
x_20 = lean_array_get_size(x_3);
x_21 = lean_nat_dec_lt(x_4, x_20);
if (x_21 == 0)
{
lean_dec(x_11);
lean_dec(x_10);
x_14 = x_9;
goto block_19;
}
else
{
uint8_t x_22; 
x_22 = lean_nat_dec_le(x_20, x_20);
if (x_22 == 0)
{
lean_dec(x_11);
lean_dec(x_10);
x_14 = x_9;
goto block_19;
}
else
{
uint8_t x_23; 
x_23 = !lean_is_exclusive(x_9);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; size_t x_26; size_t x_27; lean_object* x_28; 
x_24 = lean_ctor_get(x_9, 1);
lean_dec(x_24);
x_25 = lean_ctor_get(x_9, 0);
lean_dec(x_25);
x_26 = 0;
x_27 = lean_usize_of_nat(x_20);
x_28 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___redArg(x_10, x_3, x_26, x_27, x_11);
lean_ctor_set(x_9, 1, x_28);
x_14 = x_9;
goto block_19;
}
else
{
size_t x_29; size_t x_30; lean_object* x_31; lean_object* x_32; 
lean_dec(x_9);
x_29 = 0;
x_30 = lean_usize_of_nat(x_20);
x_31 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___redArg(x_10, x_3, x_29, x_30, x_11);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_10);
lean_ctor_set(x_32, 1, x_31);
x_14 = x_32;
goto block_19;
}
}
}
block_19:
{
size_t x_15; size_t x_16; lean_object* x_17; lean_object* x_18; 
x_15 = 1;
x_16 = lean_usize_add(x_6, x_15);
x_17 = lean_array_uset(x_13, x_6, x_14);
x_18 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3_spec__3(x_3, x_4, x_1, x_2, x_5, x_16, x_17);
return x_18;
}
}
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11(size_t x_1, size_t x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lean_usize_dec_lt(x_2, x_1);
if (x_4 == 0)
{
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_5 = lean_array_uget(x_3, x_2);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
lean_dec(x_5);
x_8 = lean_ctor_get(x_6, 1);
lean_inc_ref(x_8);
lean_dec(x_6);
x_9 = lean_unsigned_to_nat(0u);
x_10 = lean_array_uset(x_3, x_2, x_9);
x_17 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11___closed__0;
x_18 = lean_array_get_size(x_8);
x_19 = lean_nat_dec_lt(x_9, x_18);
if (x_19 == 0)
{
lean_dec_ref(x_8);
lean_dec(x_7);
x_11 = x_17;
goto block_16;
}
else
{
uint8_t x_20; 
x_20 = lean_nat_dec_le(x_18, x_18);
if (x_20 == 0)
{
lean_dec_ref(x_8);
lean_dec(x_7);
x_11 = x_17;
goto block_16;
}
else
{
size_t x_21; size_t x_22; lean_object* x_23; 
x_21 = 0;
x_22 = lean_usize_of_nat(x_18);
x_23 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__1(x_7, x_8, x_21, x_22, x_17);
lean_dec_ref(x_8);
lean_dec(x_7);
x_11 = x_23;
goto block_16;
}
}
block_16:
{
size_t x_12; size_t x_13; lean_object* x_14; 
x_12 = 1;
x_13 = lean_usize_add(x_2, x_12);
x_14 = lean_array_uset(x_10, x_2, x_11);
x_2 = x_13;
x_3 = x_14;
goto _start;
}
}
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("-- Linting passed for ", 22, 22);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("in ", 3, 3);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule___lam__0(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
size_t x_38; size_t x_39; lean_object* x_40; lean_object* x_41; uint8_t x_42; 
x_38 = lean_array_size(x_3);
x_39 = 0;
x_40 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3(x_4, x_5, x_6, x_7, x_38, x_39, x_3);
x_41 = lean_array_get_size(x_40);
x_42 = lean_nat_dec_lt(x_7, x_41);
if (x_42 == 0)
{
lean_dec_ref(x_40);
goto block_37;
}
else
{
if (x_42 == 0)
{
lean_dec_ref(x_40);
goto block_37;
}
else
{
size_t x_43; uint8_t x_44; 
x_43 = lean_usize_of_nat(x_41);
x_44 = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00runLinterOnModule_spec__5(x_40, x_39, x_43);
if (x_44 == 0)
{
lean_dec_ref(x_40);
goto block_37;
}
else
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; uint8_t x_48; lean_object* x_49; lean_object* x_50; 
x_45 = lp_batteries_runLinterOnModule___lam__0___closed__1;
x_46 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_1, x_44);
x_47 = lean_string_append(x_45, x_46);
lean_dec_ref(x_46);
x_48 = 1;
x_49 = lean_array_get_size(x_8);
lean_inc_ref(x_11);
x_50 = lp_batteries_Batteries_Tactic_Lint_formatLinterResults(x_40, x_9, x_2, x_47, x_2, x_48, x_49, x_2, x_11, x_12);
if (lean_obj_tag(x_50) == 0)
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; 
x_51 = lean_ctor_get(x_50, 0);
lean_inc(x_51);
lean_dec_ref(x_50);
x_52 = l_Lean_MessageData_toString(x_51);
x_53 = l_IO_print___at___00IO_println___at___00__private_Lake_CLI_Main_0__Lake_verifyInstall_spec__1_spec__1(x_52);
if (lean_obj_tag(x_53) == 0)
{
uint8_t x_54; lean_object* x_55; 
lean_dec_ref(x_53);
x_54 = 1;
x_55 = lean_io_exit(x_54);
if (lean_obj_tag(x_55) == 0)
{
uint8_t x_56; 
lean_dec_ref(x_11);
x_56 = !lean_is_exclusive(x_55);
if (x_56 == 0)
{
return x_55;
}
else
{
lean_object* x_57; lean_object* x_58; 
x_57 = lean_ctor_get(x_55, 0);
lean_inc(x_57);
lean_dec(x_55);
x_58 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_58, 0, x_57);
return x_58;
}
}
else
{
uint8_t x_59; 
x_59 = !lean_is_exclusive(x_55);
if (x_59 == 0)
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; 
x_60 = lean_ctor_get(x_55, 0);
x_61 = lean_ctor_get(x_11, 5);
lean_inc(x_61);
lean_dec_ref(x_11);
x_62 = lean_io_error_to_string(x_60);
x_63 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_63, 0, x_62);
x_64 = l_Lean_MessageData_ofFormat(x_63);
x_65 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_65, 0, x_61);
lean_ctor_set(x_65, 1, x_64);
lean_ctor_set(x_55, 0, x_65);
return x_55;
}
else
{
lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; 
x_66 = lean_ctor_get(x_55, 0);
lean_inc(x_66);
lean_dec(x_55);
x_67 = lean_ctor_get(x_11, 5);
lean_inc(x_67);
lean_dec_ref(x_11);
x_68 = lean_io_error_to_string(x_66);
x_69 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_69, 0, x_68);
x_70 = l_Lean_MessageData_ofFormat(x_69);
x_71 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_71, 0, x_67);
lean_ctor_set(x_71, 1, x_70);
x_72 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_72, 0, x_71);
return x_72;
}
}
}
else
{
uint8_t x_73; 
x_73 = !lean_is_exclusive(x_53);
if (x_73 == 0)
{
lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; 
x_74 = lean_ctor_get(x_53, 0);
x_75 = lean_ctor_get(x_11, 5);
lean_inc(x_75);
lean_dec_ref(x_11);
x_76 = lean_io_error_to_string(x_74);
x_77 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_77, 0, x_76);
x_78 = l_Lean_MessageData_ofFormat(x_77);
x_79 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_79, 0, x_75);
lean_ctor_set(x_79, 1, x_78);
lean_ctor_set(x_53, 0, x_79);
return x_53;
}
else
{
lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; 
x_80 = lean_ctor_get(x_53, 0);
lean_inc(x_80);
lean_dec(x_53);
x_81 = lean_ctor_get(x_11, 5);
lean_inc(x_81);
lean_dec_ref(x_11);
x_82 = lean_io_error_to_string(x_80);
x_83 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_83, 0, x_82);
x_84 = l_Lean_MessageData_ofFormat(x_83);
x_85 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_85, 0, x_81);
lean_ctor_set(x_85, 1, x_84);
x_86 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_86, 0, x_85);
return x_86;
}
}
}
else
{
uint8_t x_87; 
lean_dec_ref(x_11);
x_87 = !lean_is_exclusive(x_50);
if (x_87 == 0)
{
return x_50;
}
else
{
lean_object* x_88; lean_object* x_89; 
x_88 = lean_ctor_get(x_50, 0);
lean_inc(x_88);
lean_dec(x_50);
x_89 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_89, 0, x_88);
return x_89;
}
}
}
}
}
block_37:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_14 = lp_batteries_runLinterOnModule___lam__0___closed__0;
x_15 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_1, x_2);
x_16 = lean_string_append(x_14, x_15);
lean_dec_ref(x_15);
x_17 = lp_batteries_resolveDefaultRootModules___closed__0;
x_18 = lean_string_append(x_16, x_17);
x_19 = l_IO_println___at___00__private_Lake_CLI_Main_0__Lake_verifyInstall_spec__1(x_18);
if (lean_obj_tag(x_19) == 0)
{
uint8_t x_20; 
lean_dec_ref(x_11);
x_20 = !lean_is_exclusive(x_19);
if (x_20 == 0)
{
return x_19;
}
else
{
lean_object* x_21; lean_object* x_22; 
x_21 = lean_ctor_get(x_19, 0);
lean_inc(x_21);
lean_dec(x_19);
x_22 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_22, 0, x_21);
return x_22;
}
}
else
{
uint8_t x_23; 
x_23 = !lean_is_exclusive(x_19);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_24 = lean_ctor_get(x_19, 0);
x_25 = lean_ctor_get(x_11, 5);
lean_inc(x_25);
lean_dec_ref(x_11);
x_26 = lean_io_error_to_string(x_24);
x_27 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_27, 0, x_26);
x_28 = l_Lean_MessageData_ofFormat(x_27);
x_29 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_29, 0, x_25);
lean_ctor_set(x_29, 1, x_28);
lean_ctor_set(x_19, 0, x_29);
return x_19;
}
else
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_30 = lean_ctor_get(x_19, 0);
lean_inc(x_30);
lean_dec(x_19);
x_31 = lean_ctor_get(x_11, 5);
lean_inc(x_31);
lean_dec_ref(x_11);
x_32 = lean_io_error_to_string(x_30);
x_33 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_33, 0, x_32);
x_34 = l_Lean_MessageData_ofFormat(x_33);
x_35 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_35, 0, x_31);
lean_ctor_set(x_35, 1, x_34);
x_36 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_36, 0, x_35);
return x_36;
}
}
}
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("internal exception #", 20, 20);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("lean", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Name_beq___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Name_hash___override___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_maxRecDepth;
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("_uniq", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries_runLinterOnModule___closed__8;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lp_batteries_runLinterOnModule___closed__9;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_box(0);
x_2 = lean_unsigned_to_nat(1u);
x_3 = lean_box(0);
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(32u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries_runLinterOnModule___closed__12;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__14() {
_start:
{
size_t x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = 5;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_batteries_runLinterOnModule___closed__12;
x_4 = lp_batteries_runLinterOnModule___closed__13;
x_5 = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_2);
lean_ctor_set(x_5, 3, x_2);
lean_ctor_set_usize(x_5, 4, x_1);
return x_5;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__15() {
_start:
{
lean_object* x_1; uint64_t x_2; lean_object* x_3; 
x_1 = lp_batteries_runLinterOnModule___closed__14;
x_2 = 0;
x_3 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set_uint64(x_3, sizeof(void*)*1, x_2);
return x_3;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__16() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries_runLinterOnModule___closed__16;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_batteries_runLinterOnModule___closed__17;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(1);
x_2 = lp_batteries_runLinterOnModule___closed__14;
x_3 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_2);
lean_ctor_set(x_3, 2, x_1);
return x_3;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; uint8_t x_3; lean_object* x_4; 
x_1 = lp_batteries_runLinterOnModule___closed__14;
x_2 = lp_batteries_runLinterOnModule___closed__17;
x_3 = 1;
x_4 = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
lean_ctor_set_uint8(x_4, sizeof(void*)*3, x_3);
return x_4;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__21() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_inheritedTraceOptions;
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__22() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("", 0, 0);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__23() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = l_Lean_Core_getMaxHeartbeats(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__24() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_diagnostics;
return x_1;
}
}
static uint8_t _init_lp_batteries_runLinterOnModule___closed__25() {
_start:
{
lean_object* x_1; lean_object* x_2; uint8_t x_3; 
x_1 = lp_batteries_runLinterOnModule___closed__24;
x_2 = lean_box(0);
x_3 = l_Lean_Option_get___at___00Lake_Package_mkConfigString_spec__0(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__26() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("scripts/nolints.json", 20, 20);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__27() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("build", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__28() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("+", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__29() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__30() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_batteries_runLinterOnModule___closed__27;
x_2 = lp_batteries_runLinterOnModule___closed__29;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__31() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__32() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Batteries", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__33() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__34() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lint", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__35() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_batteries_runLinterOnModule___closed__34;
x_2 = lp_batteries_runLinterOnModule___closed__33;
x_3 = lp_batteries_runLinterOnModule___closed__32;
x_4 = l_Lean_Name_mkStr3(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__36() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("LAKE", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__37() {
_start:
{
uint8_t x_1; uint8_t x_2; lean_object* x_3; 
x_1 = 1;
x_2 = 2;
x_3 = lean_alloc_ctor(0, 0, 3);
lean_ctor_set_uint8(x_3, 0, x_2);
lean_ctor_set_uint8(x_3, 1, x_1);
lean_ctor_set_uint8(x_3, 2, x_1);
return x_3;
}
}
static lean_object* _init_lp_batteries_runLinterOnModule___closed__38() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("lake", 4, 4);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
uint8_t x_14; lean_object* x_15; 
x_14 = lean_unbox(x_2);
x_15 = lp_batteries_runLinterOnModule___lam__0(x_1, x_14, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_12);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule(uint8_t x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_10; lean_object* x_11; lean_object* x_20; lean_object* x_21; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_48; lean_object* x_49; uint8_t x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; uint8_t x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_61; lean_object* x_62; uint8_t x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; uint8_t x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_74; uint8_t x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; uint8_t x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_90; lean_object* x_91; 
x_90 = lp_batteries_runLinterOnModule___closed__1;
x_91 = l_Lean_findSysroot(x_90);
if (lean_obj_tag(x_91) == 0)
{
lean_object* x_92; lean_object* x_93; lean_object* x_94; 
x_92 = lean_ctor_get(x_91, 0);
lean_inc(x_92);
lean_dec_ref(x_91);
x_93 = lean_box(0);
x_94 = l_Lean_initSearchPath(x_92, x_93);
if (lean_obj_tag(x_94) == 0)
{
lean_object* x_95; 
lean_dec_ref(x_94);
lean_inc(x_2);
x_95 = l_Lean_findOLean(x_2);
if (lean_obj_tag(x_95) == 0)
{
lean_object* x_96; uint8_t x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; uint8_t x_102; uint8_t x_103; uint8_t x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; uint8_t x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; uint8_t x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_154; lean_object* x_155; uint8_t x_156; uint8_t x_157; uint8_t x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; uint8_t x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_183; lean_object* x_184; uint8_t x_185; uint8_t x_186; uint8_t x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; uint8_t x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; uint8_t x_198; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_266; lean_object* x_267; uint8_t x_277; uint8_t x_278; lean_object* x_279; lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_301; 
x_96 = lean_ctor_get(x_95, 0);
lean_inc(x_96);
lean_dec_ref(x_95);
x_97 = l_System_FilePath_pathExists(x_96);
lean_dec(x_96);
x_98 = lp_batteries_runLinterOnModule___closed__2;
x_99 = lp_batteries_runLinterOnModule___closed__3;
if (x_97 == 0)
{
lean_object* x_316; lean_object* x_317; uint8_t x_318; lean_object* x_319; lean_object* x_320; 
x_316 = lp_batteries_runLinterOnModule___closed__36;
x_317 = lean_io_getenv(x_316);
x_318 = 1;
x_319 = lp_batteries_runLinterOnModule___closed__37;
if (lean_obj_tag(x_317) == 0)
{
lean_object* x_339; 
x_339 = lp_batteries_runLinterOnModule___closed__38;
x_320 = x_339;
goto block_338;
}
else
{
lean_object* x_340; 
x_340 = lean_ctor_get(x_317, 0);
lean_inc(x_340);
lean_dec_ref(x_317);
x_320 = x_340;
goto block_338;
}
block_338:
{
lean_object* x_321; lean_object* x_322; lean_object* x_323; lean_object* x_324; lean_object* x_325; lean_object* x_326; lean_object* x_327; lean_object* x_328; lean_object* x_329; 
x_321 = lp_batteries_runLinterOnModule___closed__28;
lean_inc(x_2);
x_322 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_2, x_318);
x_323 = lean_string_append(x_321, x_322);
lean_dec_ref(x_322);
x_324 = lp_batteries_runLinterOnModule___closed__30;
x_325 = lean_array_push(x_324, x_323);
x_326 = lean_box(0);
x_327 = lp_batteries_runLinterOnModule___closed__31;
x_328 = lean_alloc_ctor(0, 5, 2);
lean_ctor_set(x_328, 0, x_319);
lean_ctor_set(x_328, 1, x_320);
lean_ctor_set(x_328, 2, x_325);
lean_ctor_set(x_328, 3, x_326);
lean_ctor_set(x_328, 4, x_327);
lean_ctor_set_uint8(x_328, sizeof(void*)*5, x_318);
lean_ctor_set_uint8(x_328, sizeof(void*)*5 + 1, x_97);
x_329 = lean_io_process_spawn(x_328);
if (lean_obj_tag(x_329) == 0)
{
lean_object* x_330; lean_object* x_331; 
x_330 = lean_ctor_get(x_329, 0);
lean_inc(x_330);
lean_dec_ref(x_329);
x_331 = lean_io_process_child_wait(x_319, x_330);
lean_dec(x_330);
if (lean_obj_tag(x_331) == 0)
{
lean_dec_ref(x_331);
x_301 = lean_box(0);
goto block_315;
}
else
{
uint8_t x_332; 
lean_dec(x_2);
x_332 = !lean_is_exclusive(x_331);
if (x_332 == 0)
{
return x_331;
}
else
{
lean_object* x_333; lean_object* x_334; 
x_333 = lean_ctor_get(x_331, 0);
lean_inc(x_333);
lean_dec(x_331);
x_334 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_334, 0, x_333);
return x_334;
}
}
}
else
{
uint8_t x_335; 
lean_dec(x_2);
x_335 = !lean_is_exclusive(x_329);
if (x_335 == 0)
{
return x_329;
}
else
{
lean_object* x_336; lean_object* x_337; 
x_336 = lean_ctor_get(x_329, 0);
lean_inc(x_336);
lean_dec(x_329);
x_337 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_337, 0, x_336);
return x_337;
}
}
}
}
else
{
x_301 = lean_box(0);
goto block_315;
}
block_153:
{
lean_object* x_127; 
x_127 = lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage___redArg(x_111, x_125);
lean_dec(x_111);
if (lean_obj_tag(x_127) == 0)
{
lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; 
x_128 = lean_ctor_get(x_127, 0);
lean_inc(x_128);
lean_dec_ref(x_127);
x_129 = lp_batteries_runLinterOnModule___closed__4;
x_130 = l_Lean_Option_get___at___00Lake_Package_mkConfigString_spec__1(x_106, x_129);
x_131 = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(x_131, 0, x_112);
lean_ctor_set(x_131, 1, x_113);
lean_ctor_set(x_131, 2, x_106);
lean_ctor_set(x_131, 3, x_114);
lean_ctor_set(x_131, 4, x_130);
lean_ctor_set(x_131, 5, x_115);
lean_ctor_set(x_131, 6, x_116);
lean_ctor_set(x_131, 7, x_117);
lean_ctor_set(x_131, 8, x_118);
lean_ctor_set(x_131, 9, x_119);
lean_ctor_set(x_131, 10, x_120);
lean_ctor_set(x_131, 11, x_121);
lean_ctor_set(x_131, 12, x_122);
lean_ctor_set(x_131, 13, x_124);
lean_ctor_set_uint8(x_131, sizeof(void*)*14, x_103);
lean_ctor_set_uint8(x_131, sizeof(void*)*14 + 1, x_123);
x_132 = lean_box(0);
x_133 = lp_batteries_Batteries_Tactic_Lint_getChecks(x_108, x_132, x_132, x_131, x_125);
if (lean_obj_tag(x_133) == 0)
{
lean_object* x_134; lean_object* x_135; 
x_134 = lean_ctor_get(x_133, 0);
lean_inc(x_134);
lean_dec_ref(x_133);
lean_inc_ref(x_131);
lean_inc(x_134);
x_135 = lp_batteries_Batteries_Tactic_Lint_lintCore(x_128, x_134, x_131, x_125);
if (lean_obj_tag(x_135) == 0)
{
lean_object* x_136; lean_object* x_137; lean_object* x_138; 
x_136 = lean_ctor_get(x_135, 0);
lean_inc(x_136);
lean_dec_ref(x_135);
x_137 = lean_box(x_102);
lean_inc(x_128);
lean_inc(x_134);
lean_inc(x_101);
lean_inc_ref(x_100);
lean_inc(x_136);
lean_inc(x_2);
x_138 = lean_alloc_closure((void*)(lp_batteries_runLinterOnModule___lam__0___boxed), 13, 9);
lean_closure_set(x_138, 0, x_2);
lean_closure_set(x_138, 1, x_137);
lean_closure_set(x_138, 2, x_136);
lean_closure_set(x_138, 3, x_98);
lean_closure_set(x_138, 4, x_99);
lean_closure_set(x_138, 5, x_100);
lean_closure_set(x_138, 6, x_101);
lean_closure_set(x_138, 7, x_134);
lean_closure_set(x_138, 8, x_128);
if (x_1 == 0)
{
lean_object* x_139; lean_object* x_140; 
lean_dec_ref(x_138);
lean_dec(x_109);
lean_dec(x_107);
lean_dec_ref(x_105);
x_139 = lean_box(0);
x_140 = lp_batteries_runLinterOnModule___lam__0(x_2, x_102, x_136, x_98, x_99, x_100, x_101, x_134, x_128, x_139, x_131, x_125);
lean_dec(x_125);
lean_dec(x_128);
lean_dec(x_134);
lean_dec(x_101);
lean_dec_ref(x_100);
x_20 = x_110;
x_21 = x_140;
goto block_28;
}
else
{
size_t x_141; size_t x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; uint8_t x_146; 
lean_dec(x_134);
lean_dec(x_128);
lean_dec(x_101);
lean_dec_ref(x_100);
lean_dec(x_2);
x_141 = lean_array_size(x_136);
x_142 = 0;
x_143 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11(x_141, x_142, x_136);
x_144 = lp_batteries_runLinterOnModule___closed__5;
x_145 = lean_array_get_size(x_143);
x_146 = lean_nat_dec_lt(x_107, x_145);
if (x_146 == 0)
{
lean_dec_ref(x_143);
x_74 = x_125;
x_75 = x_104;
x_76 = lean_box(0);
x_77 = x_105;
x_78 = x_138;
x_79 = x_107;
x_80 = x_108;
x_81 = x_110;
x_82 = x_109;
x_83 = x_131;
x_84 = x_144;
goto block_89;
}
else
{
uint8_t x_147; 
x_147 = lean_nat_dec_le(x_145, x_145);
if (x_147 == 0)
{
lean_dec_ref(x_143);
x_74 = x_125;
x_75 = x_104;
x_76 = lean_box(0);
x_77 = x_105;
x_78 = x_138;
x_79 = x_107;
x_80 = x_108;
x_81 = x_110;
x_82 = x_109;
x_83 = x_131;
x_84 = x_144;
goto block_89;
}
else
{
size_t x_148; lean_object* x_149; 
x_148 = lean_usize_of_nat(x_145);
x_149 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__12(x_143, x_142, x_148, x_144);
lean_dec_ref(x_143);
x_74 = x_125;
x_75 = x_104;
x_76 = lean_box(0);
x_77 = x_105;
x_78 = x_138;
x_79 = x_107;
x_80 = x_108;
x_81 = x_110;
x_82 = x_109;
x_83 = x_131;
x_84 = x_149;
goto block_89;
}
}
}
}
else
{
lean_object* x_150; 
lean_dec(x_134);
lean_dec_ref(x_131);
lean_dec(x_128);
lean_dec(x_125);
lean_dec(x_110);
lean_dec(x_109);
lean_dec(x_107);
lean_dec_ref(x_105);
lean_dec(x_101);
lean_dec_ref(x_100);
lean_dec(x_2);
x_150 = lean_ctor_get(x_135, 0);
lean_inc(x_150);
lean_dec_ref(x_135);
x_10 = x_150;
x_11 = lean_box(0);
goto block_19;
}
}
else
{
lean_object* x_151; 
lean_dec_ref(x_131);
lean_dec(x_128);
lean_dec(x_125);
lean_dec(x_110);
lean_dec(x_109);
lean_dec(x_107);
lean_dec_ref(x_105);
lean_dec(x_101);
lean_dec_ref(x_100);
lean_dec(x_2);
x_151 = lean_ctor_get(x_133, 0);
lean_inc(x_151);
lean_dec_ref(x_133);
x_10 = x_151;
x_11 = lean_box(0);
goto block_19;
}
}
else
{
lean_object* x_152; 
lean_dec(x_125);
lean_dec_ref(x_124);
lean_dec(x_122);
lean_dec(x_121);
lean_dec(x_120);
lean_dec(x_119);
lean_dec(x_118);
lean_dec(x_117);
lean_dec(x_116);
lean_dec(x_115);
lean_dec(x_114);
lean_dec_ref(x_113);
lean_dec_ref(x_112);
lean_dec(x_110);
lean_dec(x_109);
lean_dec(x_107);
lean_dec(x_106);
lean_dec_ref(x_105);
lean_dec(x_101);
lean_dec_ref(x_100);
lean_dec(x_2);
x_152 = lean_ctor_get(x_127, 0);
lean_inc(x_152);
lean_dec_ref(x_127);
x_10 = x_152;
x_11 = lean_box(0);
goto block_19;
}
}
block_182:
{
lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; uint8_t x_180; lean_object* x_181; 
x_169 = lean_ctor_get(x_166, 0);
lean_inc_ref(x_169);
x_170 = lean_ctor_get(x_166, 1);
lean_inc_ref(x_170);
x_171 = lean_ctor_get(x_166, 3);
lean_inc(x_171);
x_172 = lean_ctor_get(x_166, 5);
lean_inc(x_172);
x_173 = lean_ctor_get(x_166, 6);
lean_inc(x_173);
x_174 = lean_ctor_get(x_166, 7);
lean_inc(x_174);
x_175 = lean_ctor_get(x_166, 8);
lean_inc(x_175);
x_176 = lean_ctor_get(x_166, 9);
lean_inc(x_176);
x_177 = lean_ctor_get(x_166, 10);
lean_inc(x_177);
x_178 = lean_ctor_get(x_166, 11);
lean_inc(x_178);
x_179 = lean_ctor_get(x_166, 12);
lean_inc(x_179);
x_180 = lean_ctor_get_uint8(x_166, sizeof(void*)*14 + 1);
x_181 = lean_ctor_get(x_166, 13);
lean_inc_ref(x_181);
lean_dec_ref(x_166);
x_100 = x_154;
x_101 = x_155;
x_102 = x_156;
x_103 = x_157;
x_104 = x_158;
x_105 = x_159;
x_106 = x_160;
x_107 = x_161;
x_108 = x_162;
x_109 = x_163;
x_110 = x_164;
x_111 = x_165;
x_112 = x_169;
x_113 = x_170;
x_114 = x_171;
x_115 = x_172;
x_116 = x_173;
x_117 = x_174;
x_118 = x_175;
x_119 = x_176;
x_120 = x_177;
x_121 = x_178;
x_122 = x_179;
x_123 = x_180;
x_124 = x_181;
x_125 = x_167;
x_126 = lean_box(0);
goto block_153;
}
block_216:
{
if (x_198 == 0)
{
lean_object* x_199; uint8_t x_200; 
x_199 = lean_st_ref_take(x_195);
x_200 = !lean_is_exclusive(x_199);
if (x_200 == 0)
{
lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; 
x_201 = lean_ctor_get(x_199, 0);
x_202 = lean_ctor_get(x_199, 5);
lean_dec(x_202);
x_203 = l_Lean_Kernel_enableDiag(x_201, x_186);
lean_ctor_set(x_199, 5, x_188);
lean_ctor_set(x_199, 0, x_203);
x_204 = lean_st_ref_set(x_195, x_199);
lean_inc(x_195);
x_154 = x_183;
x_155 = x_184;
x_156 = x_185;
x_157 = x_186;
x_158 = x_187;
x_159 = x_190;
x_160 = x_189;
x_161 = x_191;
x_162 = x_192;
x_163 = x_193;
x_164 = x_195;
x_165 = x_196;
x_166 = x_194;
x_167 = x_195;
x_168 = lean_box(0);
goto block_182;
}
else
{
lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; 
x_205 = lean_ctor_get(x_199, 0);
x_206 = lean_ctor_get(x_199, 1);
x_207 = lean_ctor_get(x_199, 2);
x_208 = lean_ctor_get(x_199, 3);
x_209 = lean_ctor_get(x_199, 4);
x_210 = lean_ctor_get(x_199, 6);
x_211 = lean_ctor_get(x_199, 7);
x_212 = lean_ctor_get(x_199, 8);
lean_inc(x_212);
lean_inc(x_211);
lean_inc(x_210);
lean_inc(x_209);
lean_inc(x_208);
lean_inc(x_207);
lean_inc(x_206);
lean_inc(x_205);
lean_dec(x_199);
x_213 = l_Lean_Kernel_enableDiag(x_205, x_186);
x_214 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_214, 0, x_213);
lean_ctor_set(x_214, 1, x_206);
lean_ctor_set(x_214, 2, x_207);
lean_ctor_set(x_214, 3, x_208);
lean_ctor_set(x_214, 4, x_209);
lean_ctor_set(x_214, 5, x_188);
lean_ctor_set(x_214, 6, x_210);
lean_ctor_set(x_214, 7, x_211);
lean_ctor_set(x_214, 8, x_212);
x_215 = lean_st_ref_set(x_195, x_214);
lean_inc(x_195);
x_154 = x_183;
x_155 = x_184;
x_156 = x_185;
x_157 = x_186;
x_158 = x_187;
x_159 = x_190;
x_160 = x_189;
x_161 = x_191;
x_162 = x_192;
x_163 = x_193;
x_164 = x_195;
x_165 = x_196;
x_166 = x_194;
x_167 = x_195;
x_168 = lean_box(0);
goto block_182;
}
}
else
{
lean_dec_ref(x_188);
lean_inc(x_195);
x_154 = x_183;
x_155 = x_184;
x_156 = x_185;
x_157 = x_186;
x_158 = x_187;
x_159 = x_190;
x_160 = x_189;
x_161 = x_191;
x_162 = x_192;
x_163 = x_193;
x_164 = x_195;
x_165 = x_196;
x_166 = x_194;
x_167 = x_195;
x_168 = lean_box(0);
goto block_182;
}
}
block_265:
{
lean_object* x_221; 
x_221 = lean_enable_initializer_execution();
if (lean_obj_tag(x_221) == 0)
{
uint8_t x_222; uint8_t x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; uint32_t x_230; lean_object* x_231; lean_object* x_232; uint8_t x_233; lean_object* x_234; lean_object* x_235; 
lean_dec_ref(x_221);
x_222 = 0;
x_223 = 1;
lean_inc(x_2);
x_224 = lean_alloc_ctor(0, 1, 3);
lean_ctor_set(x_224, 0, x_2);
lean_ctor_set_uint8(x_224, sizeof(void*)*1, x_222);
lean_ctor_set_uint8(x_224, sizeof(void*)*1 + 1, x_223);
lean_ctor_set_uint8(x_224, sizeof(void*)*1 + 2, x_222);
x_225 = lean_alloc_ctor(0, 1, 3);
lean_ctor_set(x_225, 0, x_217);
lean_ctor_set_uint8(x_225, sizeof(void*)*1, x_222);
lean_ctor_set_uint8(x_225, sizeof(void*)*1 + 1, x_223);
lean_ctor_set_uint8(x_225, sizeof(void*)*1 + 2, x_222);
x_226 = lean_unsigned_to_nat(2u);
x_227 = lp_batteries_runLinterOnModule___closed__6;
x_228 = lean_array_push(x_227, x_224);
x_229 = lean_array_push(x_228, x_225);
x_230 = 1024;
x_231 = lean_unsigned_to_nat(0u);
x_232 = lp_batteries_runLinterOnModule___closed__7;
x_233 = 2;
x_234 = lean_box(1);
x_235 = l_Lean_importModules(x_229, x_93, x_230, x_232, x_222, x_223, x_233, x_234);
if (lean_obj_tag(x_235) == 0)
{
lean_object* x_236; lean_object* x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; lean_object* x_241; lean_object* x_242; lean_object* x_243; lean_object* x_244; lean_object* x_245; lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; lean_object* x_259; uint8_t x_260; uint8_t x_261; 
x_236 = lean_ctor_get(x_235, 0);
lean_inc(x_236);
lean_dec_ref(x_235);
x_237 = lean_io_get_num_heartbeats();
x_238 = lean_box(0);
x_239 = lean_unsigned_to_nat(1u);
x_240 = lp_batteries_runLinterOnModule___closed__10;
x_241 = lp_batteries_runLinterOnModule___closed__11;
x_242 = lp_batteries_runLinterOnModule___closed__15;
x_243 = lp_batteries_runLinterOnModule___closed__18;
x_244 = lp_batteries_runLinterOnModule___closed__19;
x_245 = lp_batteries_runLinterOnModule___closed__20;
x_246 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_246, 0, x_236);
lean_ctor_set(x_246, 1, x_226);
lean_ctor_set(x_246, 2, x_240);
lean_ctor_set(x_246, 3, x_241);
lean_ctor_set(x_246, 4, x_242);
lean_ctor_set(x_246, 5, x_243);
lean_ctor_set(x_246, 6, x_244);
lean_ctor_set(x_246, 7, x_245);
lean_ctor_set(x_246, 8, x_232);
x_247 = lean_st_mk_ref(x_246);
x_248 = lp_batteries_runLinterOnModule___closed__21;
x_249 = lean_st_ref_get(x_248);
x_250 = lean_st_ref_get(x_247);
x_251 = lean_ctor_get(x_250, 0);
lean_inc_ref(x_251);
lean_dec(x_250);
x_252 = lp_batteries_runLinterOnModule___closed__22;
x_253 = l_Lean_instInhabitedFileMap_default;
x_254 = lean_unsigned_to_nat(1000u);
x_255 = lean_box(0);
x_256 = lp_batteries_runLinterOnModule___closed__23;
x_257 = lean_box(0);
x_258 = l_Lean_Name_getRoot(x_2);
lean_inc(x_249);
lean_inc(x_237);
x_259 = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(x_259, 0, x_252);
lean_ctor_set(x_259, 1, x_253);
lean_ctor_set(x_259, 2, x_93);
lean_ctor_set(x_259, 3, x_231);
lean_ctor_set(x_259, 4, x_254);
lean_ctor_set(x_259, 5, x_255);
lean_ctor_set(x_259, 6, x_238);
lean_ctor_set(x_259, 7, x_93);
lean_ctor_set(x_259, 8, x_237);
lean_ctor_set(x_259, 9, x_256);
lean_ctor_set(x_259, 10, x_238);
lean_ctor_set(x_259, 11, x_239);
lean_ctor_set(x_259, 12, x_257);
lean_ctor_set(x_259, 13, x_249);
lean_ctor_set_uint8(x_259, sizeof(void*)*14, x_222);
lean_ctor_set_uint8(x_259, sizeof(void*)*14 + 1, x_222);
x_260 = lp_batteries_runLinterOnModule___closed__25;
x_261 = l_Lean_Kernel_isDiagnosticsEnabled(x_251);
lean_dec_ref(x_251);
if (x_261 == 0)
{
if (x_260 == 0)
{
lean_dec_ref(x_259);
lean_inc(x_247);
x_100 = x_219;
x_101 = x_231;
x_102 = x_223;
x_103 = x_260;
x_104 = x_222;
x_105 = x_218;
x_106 = x_93;
x_107 = x_231;
x_108 = x_223;
x_109 = x_239;
x_110 = x_247;
x_111 = x_258;
x_112 = x_252;
x_113 = x_253;
x_114 = x_231;
x_115 = x_255;
x_116 = x_238;
x_117 = x_93;
x_118 = x_237;
x_119 = x_256;
x_120 = x_238;
x_121 = x_239;
x_122 = x_257;
x_123 = x_222;
x_124 = x_249;
x_125 = x_247;
x_126 = lean_box(0);
goto block_153;
}
else
{
lean_dec(x_249);
lean_dec(x_237);
x_183 = x_219;
x_184 = x_231;
x_185 = x_223;
x_186 = x_260;
x_187 = x_222;
x_188 = x_243;
x_189 = x_93;
x_190 = x_218;
x_191 = x_231;
x_192 = x_223;
x_193 = x_239;
x_194 = x_259;
x_195 = x_247;
x_196 = x_258;
x_197 = lean_box(0);
x_198 = x_261;
goto block_216;
}
}
else
{
lean_dec(x_249);
lean_dec(x_237);
x_183 = x_219;
x_184 = x_231;
x_185 = x_223;
x_186 = x_260;
x_187 = x_222;
x_188 = x_243;
x_189 = x_93;
x_190 = x_218;
x_191 = x_231;
x_192 = x_223;
x_193 = x_239;
x_194 = x_259;
x_195 = x_247;
x_196 = x_258;
x_197 = lean_box(0);
x_198 = x_260;
goto block_216;
}
}
else
{
uint8_t x_262; 
lean_dec_ref(x_219);
lean_dec_ref(x_218);
lean_dec(x_2);
x_262 = !lean_is_exclusive(x_235);
if (x_262 == 0)
{
return x_235;
}
else
{
lean_object* x_263; lean_object* x_264; 
x_263 = lean_ctor_get(x_235, 0);
lean_inc(x_263);
lean_dec(x_235);
x_264 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_264, 0, x_263);
return x_264;
}
}
}
else
{
lean_dec_ref(x_219);
lean_dec_ref(x_218);
lean_dec(x_217);
lean_dec(x_2);
return x_221;
}
}
block_276:
{
lean_object* x_268; uint8_t x_269; 
x_268 = lp_batteries_runLinterOnModule___closed__26;
x_269 = l_System_FilePath_pathExists(x_268);
if (x_269 == 0)
{
lean_object* x_270; 
x_270 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11___closed__0;
x_217 = x_266;
x_218 = x_268;
x_219 = x_270;
x_220 = lean_box(0);
goto block_265;
}
else
{
lean_object* x_271; 
x_271 = lp_batteries_readJsonFile___at___00runLinterOnModule_spec__13(x_268);
if (lean_obj_tag(x_271) == 0)
{
lean_object* x_272; 
x_272 = lean_ctor_get(x_271, 0);
lean_inc(x_272);
lean_dec_ref(x_271);
x_217 = x_266;
x_218 = x_268;
x_219 = x_272;
x_220 = lean_box(0);
goto block_265;
}
else
{
uint8_t x_273; 
lean_dec(x_266);
lean_dec(x_2);
x_273 = !lean_is_exclusive(x_271);
if (x_273 == 0)
{
return x_271;
}
else
{
lean_object* x_274; lean_object* x_275; 
x_274 = lean_ctor_get(x_271, 0);
lean_inc(x_274);
lean_dec(x_271);
x_275 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_275, 0, x_274);
return x_275;
}
}
}
}
block_300:
{
lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; lean_object* x_291; 
x_283 = lp_batteries_runLinterOnModule___closed__28;
lean_inc(x_281);
x_284 = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(x_281, x_278);
x_285 = lean_string_append(x_283, x_284);
lean_dec_ref(x_284);
x_286 = lp_batteries_runLinterOnModule___closed__30;
x_287 = lean_array_push(x_286, x_285);
x_288 = lean_box(0);
x_289 = lp_batteries_runLinterOnModule___closed__31;
lean_inc_ref(x_280);
x_290 = lean_alloc_ctor(0, 5, 2);
lean_ctor_set(x_290, 0, x_280);
lean_ctor_set(x_290, 1, x_282);
lean_ctor_set(x_290, 2, x_287);
lean_ctor_set(x_290, 3, x_288);
lean_ctor_set(x_290, 4, x_289);
lean_ctor_set_uint8(x_290, sizeof(void*)*5, x_278);
lean_ctor_set_uint8(x_290, sizeof(void*)*5 + 1, x_277);
x_291 = lean_io_process_spawn(x_290);
if (lean_obj_tag(x_291) == 0)
{
lean_object* x_292; lean_object* x_293; 
x_292 = lean_ctor_get(x_291, 0);
lean_inc(x_292);
lean_dec_ref(x_291);
x_293 = lean_io_process_child_wait(x_280, x_292);
lean_dec(x_292);
lean_dec_ref(x_280);
if (lean_obj_tag(x_293) == 0)
{
lean_dec_ref(x_293);
x_266 = x_281;
x_267 = lean_box(0);
goto block_276;
}
else
{
uint8_t x_294; 
lean_dec(x_281);
lean_dec(x_2);
x_294 = !lean_is_exclusive(x_293);
if (x_294 == 0)
{
return x_293;
}
else
{
lean_object* x_295; lean_object* x_296; 
x_295 = lean_ctor_get(x_293, 0);
lean_inc(x_295);
lean_dec(x_293);
x_296 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_296, 0, x_295);
return x_296;
}
}
}
else
{
uint8_t x_297; 
lean_dec(x_281);
lean_dec_ref(x_280);
lean_dec(x_2);
x_297 = !lean_is_exclusive(x_291);
if (x_297 == 0)
{
return x_291;
}
else
{
lean_object* x_298; lean_object* x_299; 
x_298 = lean_ctor_get(x_291, 0);
lean_inc(x_298);
lean_dec(x_291);
x_299 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_299, 0, x_298);
return x_299;
}
}
}
block_315:
{
lean_object* x_302; lean_object* x_303; 
x_302 = lp_batteries_runLinterOnModule___closed__35;
x_303 = l_Lean_findOLean(x_302);
if (lean_obj_tag(x_303) == 0)
{
lean_object* x_304; uint8_t x_305; 
x_304 = lean_ctor_get(x_303, 0);
lean_inc(x_304);
lean_dec_ref(x_303);
x_305 = l_System_FilePath_pathExists(x_304);
lean_dec(x_304);
if (x_305 == 0)
{
lean_object* x_306; lean_object* x_307; uint8_t x_308; lean_object* x_309; 
x_306 = lp_batteries_runLinterOnModule___closed__36;
x_307 = lean_io_getenv(x_306);
x_308 = 1;
x_309 = lp_batteries_runLinterOnModule___closed__37;
if (lean_obj_tag(x_307) == 0)
{
lean_object* x_310; 
x_310 = lp_batteries_runLinterOnModule___closed__38;
x_277 = x_305;
x_278 = x_308;
x_279 = lean_box(0);
x_280 = x_309;
x_281 = x_302;
x_282 = x_310;
goto block_300;
}
else
{
lean_object* x_311; 
x_311 = lean_ctor_get(x_307, 0);
lean_inc(x_311);
lean_dec_ref(x_307);
x_277 = x_305;
x_278 = x_308;
x_279 = lean_box(0);
x_280 = x_309;
x_281 = x_302;
x_282 = x_311;
goto block_300;
}
}
else
{
x_266 = x_302;
x_267 = lean_box(0);
goto block_276;
}
}
else
{
uint8_t x_312; 
lean_dec(x_2);
x_312 = !lean_is_exclusive(x_303);
if (x_312 == 0)
{
return x_303;
}
else
{
lean_object* x_313; lean_object* x_314; 
x_313 = lean_ctor_get(x_303, 0);
lean_inc(x_313);
lean_dec(x_303);
x_314 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_314, 0, x_313);
return x_314;
}
}
}
}
else
{
uint8_t x_341; 
lean_dec(x_2);
x_341 = !lean_is_exclusive(x_95);
if (x_341 == 0)
{
return x_95;
}
else
{
lean_object* x_342; lean_object* x_343; 
x_342 = lean_ctor_get(x_95, 0);
lean_inc(x_342);
lean_dec(x_95);
x_343 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_343, 0, x_342);
return x_343;
}
}
}
else
{
lean_dec(x_2);
return x_94;
}
}
else
{
uint8_t x_344; 
lean_dec(x_2);
x_344 = !lean_is_exclusive(x_91);
if (x_344 == 0)
{
return x_91;
}
else
{
lean_object* x_345; lean_object* x_346; 
x_345 = lean_ctor_get(x_91, 0);
lean_inc(x_345);
lean_dec(x_91);
x_346 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_346, 0, x_345);
return x_346;
}
}
block_9:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = l_Lean_MessageData_toString(x_4);
x_7 = lean_alloc_ctor(18, 1, 0);
lean_ctor_set(x_7, 0, x_6);
x_8 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
block_19:
{
if (lean_obj_tag(x_10) == 0)
{
lean_object* x_12; 
x_12 = lean_ctor_get(x_10, 1);
lean_inc_ref(x_12);
lean_dec_ref(x_10);
x_4 = x_12;
x_5 = lean_box(0);
goto block_9;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_13 = lean_ctor_get(x_10, 0);
lean_inc(x_13);
lean_dec_ref(x_10);
x_14 = lp_batteries_runLinterOnModule___closed__0;
x_15 = l_Nat_reprFast(x_13);
x_16 = lean_string_append(x_14, x_15);
lean_dec_ref(x_15);
x_17 = lean_alloc_ctor(18, 1, 0);
lean_ctor_set(x_17, 0, x_16);
x_18 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
}
block_28:
{
if (lean_obj_tag(x_21) == 0)
{
uint8_t x_22; 
x_22 = !lean_is_exclusive(x_21);
if (x_22 == 0)
{
lean_object* x_23; 
x_23 = lean_st_ref_get(x_20);
lean_dec(x_20);
lean_dec(x_23);
return x_21;
}
else
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_24 = lean_ctor_get(x_21, 0);
lean_inc(x_24);
lean_dec(x_21);
x_25 = lean_st_ref_get(x_20);
lean_dec(x_20);
lean_dec(x_25);
x_26 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_26, 0, x_24);
return x_26;
}
}
else
{
lean_object* x_27; 
lean_dec(x_20);
x_27 = lean_ctor_get(x_21, 0);
lean_inc(x_27);
lean_dec_ref(x_21);
x_10 = x_27;
x_11 = lean_box(0);
goto block_19;
}
}
block_47:
{
lean_object* x_36; 
x_36 = lp_batteries_writeJsonFile___at___00runLinterOnModule_spec__6(x_32, x_35);
lean_dec_ref(x_32);
if (lean_obj_tag(x_36) == 0)
{
lean_object* x_37; lean_object* x_38; 
x_37 = lean_ctor_get(x_36, 0);
lean_inc(x_37);
lean_dec_ref(x_36);
x_38 = lean_apply_4(x_31, x_37, x_34, x_29, lean_box(0));
x_20 = x_33;
x_21 = x_38;
goto block_28;
}
else
{
uint8_t x_39; 
lean_dec_ref(x_34);
lean_dec(x_33);
lean_dec_ref(x_31);
lean_dec(x_29);
x_39 = !lean_is_exclusive(x_36);
if (x_39 == 0)
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_40 = lean_ctor_get(x_36, 0);
x_41 = lean_io_error_to_string(x_40);
lean_ctor_set_tag(x_36, 3);
lean_ctor_set(x_36, 0, x_41);
x_42 = l_Lean_MessageData_ofFormat(x_36);
x_4 = x_42;
x_5 = lean_box(0);
goto block_9;
}
else
{
lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_43 = lean_ctor_get(x_36, 0);
lean_inc(x_43);
lean_dec(x_36);
x_44 = lean_io_error_to_string(x_43);
x_45 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_45, 0, x_44);
x_46 = l_Lean_MessageData_ofFormat(x_45);
x_4 = x_46;
x_5 = lean_box(0);
goto block_9;
}
}
}
block_60:
{
lean_object* x_59; 
x_59 = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg(x_50, x_55, x_49, x_54, x_58);
lean_dec(x_58);
x_29 = x_48;
x_30 = lean_box(0);
x_31 = x_53;
x_32 = x_52;
x_33 = x_56;
x_34 = x_57;
x_35 = x_59;
goto block_47;
}
block_73:
{
uint8_t x_72; 
x_72 = lean_nat_dec_le(x_71, x_62);
if (x_72 == 0)
{
lean_dec(x_62);
lean_inc(x_71);
x_48 = x_61;
x_49 = x_64;
x_50 = x_63;
x_51 = lean_box(0);
x_52 = x_67;
x_53 = x_66;
x_54 = x_71;
x_55 = x_68;
x_56 = x_69;
x_57 = x_70;
x_58 = x_71;
goto block_60;
}
else
{
x_48 = x_61;
x_49 = x_64;
x_50 = x_63;
x_51 = lean_box(0);
x_52 = x_67;
x_53 = x_66;
x_54 = x_71;
x_55 = x_68;
x_56 = x_69;
x_57 = x_70;
x_58 = x_62;
goto block_60;
}
}
block_89:
{
lean_object* x_85; uint8_t x_86; 
x_85 = lean_array_get_size(x_84);
x_86 = lean_nat_dec_eq(x_85, x_79);
if (x_86 == 0)
{
lean_object* x_87; uint8_t x_88; 
x_87 = lean_nat_sub(x_85, x_82);
lean_dec(x_82);
x_88 = lean_nat_dec_le(x_79, x_87);
if (x_88 == 0)
{
lean_dec(x_79);
lean_inc(x_87);
x_61 = x_74;
x_62 = x_87;
x_63 = x_75;
x_64 = x_84;
x_65 = lean_box(0);
x_66 = x_78;
x_67 = x_77;
x_68 = x_80;
x_69 = x_81;
x_70 = x_83;
x_71 = x_87;
goto block_73;
}
else
{
x_61 = x_74;
x_62 = x_87;
x_63 = x_75;
x_64 = x_84;
x_65 = lean_box(0);
x_66 = x_78;
x_67 = x_77;
x_68 = x_80;
x_69 = x_81;
x_70 = x_83;
x_71 = x_79;
goto block_73;
}
}
else
{
lean_dec(x_82);
lean_dec(x_79);
x_29 = x_74;
x_30 = lean_box(0);
x_31 = x_78;
x_32 = x_77;
x_33 = x_81;
x_34 = x_83;
x_35 = x_84;
goto block_47;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
size_t x_8; size_t x_9; lean_object* x_10; 
x_8 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_9 = lean_unbox_usize(x_6);
lean_dec(x_6);
x_10 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2(x_1, x_2, x_3, x_4, x_8, x_9, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; uint8_t x_11; lean_object* x_12; 
x_10 = lean_unbox(x_1);
x_11 = lean_unbox(x_2);
x_12 = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10(x_10, x_11, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_6);
lean_dec(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_liftExcept___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__13___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_writeJsonFile___at___00runLinterOnModule_spec__6___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_writeJsonFile___at___00runLinterOnModule_spec__6(x_1, x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00runLinterOnModule_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00runLinterOnModule_spec__0(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
size_t x_6; size_t x_7; lean_object* x_8; 
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_8 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__1(x_1, x_2, x_6, x_7, x_5);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__12___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
size_t x_5; size_t x_6; lean_object* x_7; 
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__12(x_1, x_5, x_6, x_4);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__7___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; size_t x_5; lean_object* x_6; 
x_4 = lean_unbox_usize(x_1);
lean_dec(x_1);
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__7(x_4, x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_readJsonFile___at___00runLinterOnModule_spec__13___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_readJsonFile___at___00runLinterOnModule_spec__13(x_1);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; uint8_t x_7; lean_object* x_8; 
x_6 = lean_unbox(x_1);
x_7 = lean_unbox(x_2);
x_8 = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00runLinterOnModule_spec__10___redArg(x_6, x_7, x_3, x_4, x_5);
lean_dec(x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__15___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; size_t x_5; lean_object* x_6; 
x_4 = lean_unbox_usize(x_1);
lean_dec(x_1);
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__15(x_4, x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
size_t x_6; size_t x_7; lean_object* x_8; 
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_8 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00runLinterOnModule_spec__2___redArg(x_1, x_2, x_6, x_7, x_5);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00runLinterOnModule_spec__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; size_t x_5; uint8_t x_6; lean_object* x_7; 
x_4 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_5 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_6 = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00runLinterOnModule_spec__5(x_1, x_4, x_5);
lean_dec_ref(x_1);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
size_t x_8; size_t x_9; lean_object* x_10; 
x_8 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_9 = lean_unbox_usize(x_6);
lean_dec(x_6);
x_10 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3_spec__3(x_1, x_2, x_3, x_4, x_8, x_9, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
size_t x_8; size_t x_9; lean_object* x_10; 
x_8 = lean_unbox_usize(x_5);
lean_dec(x_5);
x_9 = lean_unbox_usize(x_6);
lean_dec(x_6);
x_10 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__3(x_1, x_2, x_3, x_4, x_8, x_9, x_7);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
size_t x_4; size_t x_5; lean_object* x_6; 
x_4 = lean_unbox_usize(x_1);
lean_dec(x_1);
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11(x_4, x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_runLinterOnModule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lean_unbox(x_1);
x_5 = lp_batteries_runLinterOnModule(x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00main_spec__0(uint8_t x_1, lean_object* x_2, size_t x_3, size_t x_4, lean_object* x_5) {
_start:
{
uint8_t x_7; 
x_7 = lean_usize_dec_eq(x_3, x_4);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_array_uget(x_2, x_3);
x_9 = lp_batteries_runLinterOnModule(x_1, x_8);
if (lean_obj_tag(x_9) == 0)
{
lean_object* x_10; size_t x_11; size_t x_12; 
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = 1;
x_12 = lean_usize_add(x_3, x_11);
x_3 = x_12;
x_5 = x_10;
goto _start;
}
else
{
return x_9;
}
}
else
{
lean_object* x_14; 
x_14 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_14, 0, x_5);
return x_14;
}
}
}
static lean_object* _init_lp_batteries_main___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Error parsing args: ", 20, 20);
return x_1;
}
}
static lean_object* _init_lp_batteries_main___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Usage: runLinter [--update] [Batteries.Data.Nat.Basic]", 54, 54);
return x_1;
}
}
LEAN_EXPORT lean_object* _lean_main(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_35; 
x_35 = lp_batteries_parseLinterArgs(x_1);
if (lean_obj_tag(x_35) == 0)
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_36 = lean_ctor_get(x_35, 0);
lean_inc(x_36);
lean_dec_ref(x_35);
x_37 = lp_batteries_main___closed__0;
x_38 = lean_string_append(x_37, x_36);
lean_dec(x_36);
x_39 = l_IO_eprintln___at___00Lake_serve_spec__0(x_38);
if (lean_obj_tag(x_39) == 0)
{
lean_object* x_40; lean_object* x_41; 
lean_dec_ref(x_39);
x_40 = lp_batteries_main___closed__1;
x_41 = l_IO_eprintln___at___00Lake_serve_spec__0(x_40);
if (lean_obj_tag(x_41) == 0)
{
uint8_t x_42; lean_object* x_43; 
lean_dec_ref(x_41);
x_42 = 1;
x_43 = lean_io_exit(x_42);
if (lean_obj_tag(x_43) == 0)
{
lean_object* x_44; 
x_44 = lean_ctor_get(x_43, 0);
lean_inc(x_44);
lean_dec_ref(x_43);
x_3 = x_44;
x_4 = lean_box(0);
goto block_34;
}
else
{
uint8_t x_45; 
x_45 = !lean_is_exclusive(x_43);
if (x_45 == 0)
{
return x_43;
}
else
{
lean_object* x_46; lean_object* x_47; 
x_46 = lean_ctor_get(x_43, 0);
lean_inc(x_46);
lean_dec(x_43);
x_47 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_47, 0, x_46);
return x_47;
}
}
}
else
{
return x_41;
}
}
else
{
return x_39;
}
}
else
{
lean_object* x_48; 
x_48 = lean_ctor_get(x_35, 0);
lean_inc(x_48);
lean_dec_ref(x_35);
x_3 = x_48;
x_4 = lean_box(0);
goto block_34;
}
block_34:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = lp_batteries_determineModulesToLint(x_6);
if (lean_obj_tag(x_7) == 0)
{
uint8_t x_8; 
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_unsigned_to_nat(0u);
x_11 = lean_array_get_size(x_9);
x_12 = lean_box(0);
x_13 = lean_nat_dec_lt(x_10, x_11);
if (x_13 == 0)
{
lean_dec(x_9);
lean_dec(x_5);
lean_ctor_set(x_7, 0, x_12);
return x_7;
}
else
{
uint8_t x_14; 
x_14 = lean_nat_dec_le(x_11, x_11);
if (x_14 == 0)
{
lean_dec(x_9);
lean_dec(x_5);
lean_ctor_set(x_7, 0, x_12);
return x_7;
}
else
{
size_t x_15; size_t x_16; uint8_t x_17; lean_object* x_18; 
lean_free_object(x_7);
x_15 = 0;
x_16 = lean_usize_of_nat(x_11);
x_17 = lean_unbox(x_5);
lean_dec(x_5);
x_18 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00main_spec__0(x_17, x_9, x_15, x_16, x_12);
lean_dec(x_9);
return x_18;
}
}
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; 
x_19 = lean_ctor_get(x_7, 0);
lean_inc(x_19);
lean_dec(x_7);
x_20 = lean_unsigned_to_nat(0u);
x_21 = lean_array_get_size(x_19);
x_22 = lean_box(0);
x_23 = lean_nat_dec_lt(x_20, x_21);
if (x_23 == 0)
{
lean_object* x_24; 
lean_dec(x_19);
lean_dec(x_5);
x_24 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_24, 0, x_22);
return x_24;
}
else
{
uint8_t x_25; 
x_25 = lean_nat_dec_le(x_21, x_21);
if (x_25 == 0)
{
lean_object* x_26; 
lean_dec(x_19);
lean_dec(x_5);
x_26 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_26, 0, x_22);
return x_26;
}
else
{
size_t x_27; size_t x_28; uint8_t x_29; lean_object* x_30; 
x_27 = 0;
x_28 = lean_usize_of_nat(x_21);
x_29 = lean_unbox(x_5);
lean_dec(x_5);
x_30 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00main_spec__0(x_29, x_19, x_27, x_28, x_22);
lean_dec(x_19);
return x_30;
}
}
}
}
else
{
uint8_t x_31; 
lean_dec(x_5);
x_31 = !lean_is_exclusive(x_7);
if (x_31 == 0)
{
return x_7;
}
else
{
lean_object* x_32; lean_object* x_33; 
x_32 = lean_ctor_get(x_7, 0);
lean_inc(x_32);
lean_dec(x_7);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_32);
return x_33;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00main_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; size_t x_8; size_t x_9; lean_object* x_10; 
x_7 = lean_unbox(x_1);
x_8 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_9 = lean_unbox_usize(x_4);
lean_dec(x_4);
x_10 = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00main_spec__0(x_7, x_2, x_8, x_9, x_5);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_main___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = _lean_main(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Lint(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Array_Basic(uint8_t builtin);
lean_object* initialize_Lake_CLI_Main(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_runLinter(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Lint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Array_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lake_CLI_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_readJsonFile___redArg___closed__0 = _init_lp_batteries_readJsonFile___redArg___closed__0();
lean_mark_persistent(lp_batteries_readJsonFile___redArg___closed__0);
lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__0 = _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__0();
lean_mark_persistent(lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__0);
lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__1 = _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__1();
lean_mark_persistent(lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__1);
lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__2 = _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__2();
lean_mark_persistent(lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__2);
lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__3 = _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__3();
lean_mark_persistent(lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00resolveDefaultRootModules_spec__3___closed__3);
lp_batteries_resolveDefaultRootModules___closed__0 = _init_lp_batteries_resolveDefaultRootModules___closed__0();
lean_mark_persistent(lp_batteries_resolveDefaultRootModules___closed__0);
lp_batteries_resolveDefaultRootModules___closed__1 = _init_lp_batteries_resolveDefaultRootModules___closed__1();
lean_mark_persistent(lp_batteries_resolveDefaultRootModules___closed__1);
lp_batteries_resolveDefaultRootModules___closed__2 = _init_lp_batteries_resolveDefaultRootModules___closed__2();
lean_mark_persistent(lp_batteries_resolveDefaultRootModules___closed__2);
lp_batteries_resolveDefaultRootModules___closed__3 = _init_lp_batteries_resolveDefaultRootModules___closed__3();
lean_mark_persistent(lp_batteries_resolveDefaultRootModules___closed__3);
lp_batteries_resolveDefaultRootModules___closed__4 = _init_lp_batteries_resolveDefaultRootModules___closed__4();
lean_mark_persistent(lp_batteries_resolveDefaultRootModules___closed__4);
lp_batteries_resolveDefaultRootModules___closed__5 = _init_lp_batteries_resolveDefaultRootModules___closed__5();
lean_mark_persistent(lp_batteries_resolveDefaultRootModules___closed__5);
lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0___closed__0 = _init_lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0___closed__0();
lean_mark_persistent(lp_batteries_Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0___closed__0);
lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__0 = _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__0();
lean_mark_persistent(lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__0);
lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__1 = _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__1();
lean_mark_persistent(lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__1);
lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__2 = _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__2();
lean_mark_persistent(lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__2);
lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__3 = _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__3();
lean_mark_persistent(lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__3);
lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__4 = _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__4();
lean_mark_persistent(lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__4);
lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__5 = _init_lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__5();
lean_mark_persistent(lp_batteries_Lake_Workspace_materializeDeps___at___00Lake_loadWorkspace___at___00resolveDefaultRootModules_spec__0_spec__0___closed__5);
lp_batteries_parseLinterArgs___closed__0 = _init_lp_batteries_parseLinterArgs___closed__0();
lean_mark_persistent(lp_batteries_parseLinterArgs___closed__0);
lp_batteries_parseLinterArgs___closed__1 = _init_lp_batteries_parseLinterArgs___closed__1();
lean_mark_persistent(lp_batteries_parseLinterArgs___closed__1);
lp_batteries_parseLinterArgs___closed__2 = _init_lp_batteries_parseLinterArgs___closed__2();
lean_mark_persistent(lp_batteries_parseLinterArgs___closed__2);
lp_batteries_parseLinterArgs___closed__3 = _init_lp_batteries_parseLinterArgs___closed__3();
lean_mark_persistent(lp_batteries_parseLinterArgs___closed__3);
lp_batteries_parseLinterArgs___closed__4 = _init_lp_batteries_parseLinterArgs___closed__4();
lean_mark_persistent(lp_batteries_parseLinterArgs___closed__4);
lp_batteries_determineModulesToLint___closed__0 = _init_lp_batteries_determineModulesToLint___closed__0();
lean_mark_persistent(lp_batteries_determineModulesToLint___closed__0);
lp_batteries_determineModulesToLint___closed__1 = _init_lp_batteries_determineModulesToLint___closed__1();
lean_mark_persistent(lp_batteries_determineModulesToLint___closed__1);
lp_batteries_determineModulesToLint___closed__2 = _init_lp_batteries_determineModulesToLint___closed__2();
lean_mark_persistent(lp_batteries_determineModulesToLint___closed__2);
lp_batteries_determineModulesToLint___closed__3 = _init_lp_batteries_determineModulesToLint___closed__3();
lean_mark_persistent(lp_batteries_determineModulesToLint___closed__3);
lp_batteries_Prod_toJson___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__6___closed__0 = _init_lp_batteries_Prod_toJson___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__6___closed__0();
lean_mark_persistent(lp_batteries_Prod_toJson___at___00Array_toJson___at___00writeJsonFile___at___00runLinterOnModule_spec__6_spec__6_spec__6___closed__0);
lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__0 = _init_lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__0();
lean_mark_persistent(lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__0);
lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__1 = _init_lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__1();
lean_mark_persistent(lp_batteries_Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14___closed__1);
lp_batteries_Prod_fromJson_x3f___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__14___closed__0 = _init_lp_batteries_Prod_fromJson_x3f___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__14___closed__0();
lean_mark_persistent(lp_batteries_Prod_fromJson_x3f___at___00Array_fromJson_x3f___at___00readJsonFile___at___00runLinterOnModule_spec__13_spec__14_spec__14___closed__0);
lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11___closed__0 = _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11___closed__0();
lean_mark_persistent(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00runLinterOnModule_spec__11___closed__0);
lp_batteries_runLinterOnModule___lam__0___closed__0 = _init_lp_batteries_runLinterOnModule___lam__0___closed__0();
lean_mark_persistent(lp_batteries_runLinterOnModule___lam__0___closed__0);
lp_batteries_runLinterOnModule___lam__0___closed__1 = _init_lp_batteries_runLinterOnModule___lam__0___closed__1();
lean_mark_persistent(lp_batteries_runLinterOnModule___lam__0___closed__1);
lp_batteries_runLinterOnModule___closed__0 = _init_lp_batteries_runLinterOnModule___closed__0();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__0);
lp_batteries_runLinterOnModule___closed__1 = _init_lp_batteries_runLinterOnModule___closed__1();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__1);
lp_batteries_runLinterOnModule___closed__2 = _init_lp_batteries_runLinterOnModule___closed__2();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__2);
lp_batteries_runLinterOnModule___closed__3 = _init_lp_batteries_runLinterOnModule___closed__3();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__3);
lp_batteries_runLinterOnModule___closed__4 = _init_lp_batteries_runLinterOnModule___closed__4();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__4);
lp_batteries_runLinterOnModule___closed__5 = _init_lp_batteries_runLinterOnModule___closed__5();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__5);
lp_batteries_runLinterOnModule___closed__6 = _init_lp_batteries_runLinterOnModule___closed__6();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__6);
lp_batteries_runLinterOnModule___closed__7 = _init_lp_batteries_runLinterOnModule___closed__7();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__7);
lp_batteries_runLinterOnModule___closed__8 = _init_lp_batteries_runLinterOnModule___closed__8();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__8);
lp_batteries_runLinterOnModule___closed__9 = _init_lp_batteries_runLinterOnModule___closed__9();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__9);
lp_batteries_runLinterOnModule___closed__10 = _init_lp_batteries_runLinterOnModule___closed__10();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__10);
lp_batteries_runLinterOnModule___closed__11 = _init_lp_batteries_runLinterOnModule___closed__11();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__11);
lp_batteries_runLinterOnModule___closed__12 = _init_lp_batteries_runLinterOnModule___closed__12();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__12);
lp_batteries_runLinterOnModule___closed__13 = _init_lp_batteries_runLinterOnModule___closed__13();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__13);
lp_batteries_runLinterOnModule___closed__14 = _init_lp_batteries_runLinterOnModule___closed__14();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__14);
lp_batteries_runLinterOnModule___closed__15 = _init_lp_batteries_runLinterOnModule___closed__15();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__15);
lp_batteries_runLinterOnModule___closed__16 = _init_lp_batteries_runLinterOnModule___closed__16();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__16);
lp_batteries_runLinterOnModule___closed__17 = _init_lp_batteries_runLinterOnModule___closed__17();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__17);
lp_batteries_runLinterOnModule___closed__18 = _init_lp_batteries_runLinterOnModule___closed__18();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__18);
lp_batteries_runLinterOnModule___closed__19 = _init_lp_batteries_runLinterOnModule___closed__19();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__19);
lp_batteries_runLinterOnModule___closed__20 = _init_lp_batteries_runLinterOnModule___closed__20();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__20);
lp_batteries_runLinterOnModule___closed__21 = _init_lp_batteries_runLinterOnModule___closed__21();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__21);
lp_batteries_runLinterOnModule___closed__22 = _init_lp_batteries_runLinterOnModule___closed__22();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__22);
lp_batteries_runLinterOnModule___closed__23 = _init_lp_batteries_runLinterOnModule___closed__23();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__23);
lp_batteries_runLinterOnModule___closed__24 = _init_lp_batteries_runLinterOnModule___closed__24();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__24);
lp_batteries_runLinterOnModule___closed__25 = _init_lp_batteries_runLinterOnModule___closed__25();
lp_batteries_runLinterOnModule___closed__26 = _init_lp_batteries_runLinterOnModule___closed__26();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__26);
lp_batteries_runLinterOnModule___closed__27 = _init_lp_batteries_runLinterOnModule___closed__27();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__27);
lp_batteries_runLinterOnModule___closed__28 = _init_lp_batteries_runLinterOnModule___closed__28();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__28);
lp_batteries_runLinterOnModule___closed__29 = _init_lp_batteries_runLinterOnModule___closed__29();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__29);
lp_batteries_runLinterOnModule___closed__30 = _init_lp_batteries_runLinterOnModule___closed__30();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__30);
lp_batteries_runLinterOnModule___closed__31 = _init_lp_batteries_runLinterOnModule___closed__31();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__31);
lp_batteries_runLinterOnModule___closed__32 = _init_lp_batteries_runLinterOnModule___closed__32();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__32);
lp_batteries_runLinterOnModule___closed__33 = _init_lp_batteries_runLinterOnModule___closed__33();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__33);
lp_batteries_runLinterOnModule___closed__34 = _init_lp_batteries_runLinterOnModule___closed__34();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__34);
lp_batteries_runLinterOnModule___closed__35 = _init_lp_batteries_runLinterOnModule___closed__35();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__35);
lp_batteries_runLinterOnModule___closed__36 = _init_lp_batteries_runLinterOnModule___closed__36();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__36);
lp_batteries_runLinterOnModule___closed__37 = _init_lp_batteries_runLinterOnModule___closed__37();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__37);
lp_batteries_runLinterOnModule___closed__38 = _init_lp_batteries_runLinterOnModule___closed__38();
lean_mark_persistent(lp_batteries_runLinterOnModule___closed__38);
lp_batteries_main___closed__0 = _init_lp_batteries_main___closed__0();
lean_mark_persistent(lp_batteries_main___closed__0);
lp_batteries_main___closed__1 = _init_lp_batteries_main___closed__1();
lean_mark_persistent(lp_batteries_main___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
char ** lean_setup_args(int argc, char ** argv);
void lean_initialize();

  #if defined(WIN32) || defined(_WIN32)
  #include <windows.h>
  #endif

  int main(int argc, char ** argv) {
  #if defined(WIN32) || defined(_WIN32)
  SetErrorMode(SEM_FAILCRITICALERRORS);
  SetConsoleOutputCP(CP_UTF8);
  #endif
  lean_object* in; lean_object* res;
argv = lean_setup_args(argc, argv);
lean_initialize();
lean_set_panic_messages(false);
res = initialize_batteries_runLinter(1 /* builtin */);
lean_set_panic_messages(true);
lean_io_mark_end_initialization();
if (lean_io_result_is_ok(res)) {
lean_dec_ref(res);
lean_init_task_manager();
in = lean_box(0);
int i = argc;
while (i > 1) {
 lean_object* n;
 i--;
 n = lean_alloc_ctor(1,2,0); lean_ctor_set(n, 0, lean_mk_string(argv[i])); lean_ctor_set(n, 1, in);
 in = n;
}
res = _lean_main(in);
}
lean_finalize_task_manager();
if (lean_io_result_is_ok(res)) {
  int ret = 0;
  lean_dec_ref(res);
  return ret;
} else {
  lean_io_result_show_error(res);
  lean_dec_ref(res);
  return 1;
}
}
#ifdef __cplusplus
}
#endif
