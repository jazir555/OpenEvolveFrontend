# God Class Refactoring Analysis Report

**Generated:** 2025-12-29
**Total Lines Analyzed:** 6,683 lines across 3 files

## Executive Summary

Three god classes have been identified and analyzed. This report provides detailed refactoring plans for splitting each class into focused, single-responsibility classes.

---

## 1. CleanupStaticHelper.php (1,714 lines)

### Current Issues
- **All methods are static** - prevents dependency injection and testing
- **Mixed responsibilities** - file operations, database operations, cron scheduling, memory management
- **Tight coupling** to WordPress globals and direct database access

### Identified Responsibilities

1. **File System Operations** (~400 lines)
   - `delete_directory()`
   - `get_temp_files()`
   - `cleanup_resources()`
   - `initialize_filesystem_helper()`

2. **Database Operations** (~500 lines)
   - `cleanup_stale_queue_items()`
   - `cleanup_orphaned_tasks()`
   - `fetch_orphaned_task_ids()`
   - `process_orphaned_tasks()`
   - `update_orphaned_task_status()`

3. **Task-Specific Cleanup** (~400 lines)
   - `cleanup_failed_task()`
   - `cleanup_task_resources_static()`
   - `cleanup_task_asset_enqueue()`
   - `cleanup_task_cron_hooks()`
   - `cleanup_task_transients()`

4. **Scheduling & Coordination** (~200 lines)
   - `maybe_schedule_cleanup()`
   - `perform_global_cleanup()`
   - `unschedule_cleanup_cron()`

5. **Memory & Data Management** (~214 lines)
   - `check_and_cleanup_memory()`
   - `perform_periodic_cleanup()`
   - `clear_temporary_data()`

### Proposed Split

```php
// 1. CleanupFileOperator (instance-based)
class CleanupFileOperator {
    private \LHA\Interfaces\LoggerInterface $logger;
    private \LHA\Interfaces\LockInterface $lock;

    public function __construct(LoggerInterface, LockInterface);
    public function delete_directory(string $dir): bool;
    public function get_temp_files(): array;
    public function cleanup_resources(): CleanupResult;
    public function cleanup_failed_enqueue(string $handle, string $asset_type): void;
}

// 2. CleanupRepository (database operations)
class CleanupRepository {
    private \wpdb $wpdb;
    private \LHA\Interfaces\LoggerInterface $logger;

    public function __construct(\wpdb, LoggerInterface);
    public function cleanup_stale_queue_items(): int;
    public function fetch_orphaned_task_ids(int $hours, array $statuses): array;
    public function cleanup_orphaned_tasks(array $task_ids): CleanupResult;
    public function get_task_details(int $task_id): ?array;
}

// 3. CleanupTaskManager (task-specific operations)
class CleanupTaskManager {
    private CleanupRepository $repository;
    private CleanupFileOperator $fileOperator;
    private \LHA\Interfaces\GenerateInterface $generate;

    public function __construct(CleanupRepository, CleanupFileOperator, GenerateInterface);
    public function cleanup_failed_task(int $task_id): void;
    public function cleanup_task_resources(int $task_id): void;
    public function cleanup_existing_task(int $task_id, string $hook_name): void;
}

// 4. CleanupScheduler (coordination & scheduling)
class CleanupScheduler {
    private \LHA\Interfaces\TaskQueueInterface $tasks;
    private \LHA\Interfaces\LoggerInterface $logger;

    public function __construct(TaskQueueInterface, LoggerInterface);
    public function maybe_schedule_cleanup(): void;
    public function unschedule_cleanup_cron(): void;
    public function perform_global_cleanup(): void;
}

// 5. MemoryManager (memory & data cleanup)
class MemoryManager {
    private \LHA\Interfaces\GetdataInterface $getdata;
    private \LHA\Interfaces\NormalizeInterface $normalize;
    private \LHA\Interfaces\LoggerInterface $logger;

    public function __construct(GetdataInterface, NormalizeInterface, LoggerInterface);
    public function check_and_cleanup_memory(): void;
    public function perform_periodic_cleanup(): void;
    public function clear_temporary_data(): void;
}

// 6. CleanupStaticHelper (facade for backward compatibility)
class CleanupStaticHelper {
    private static ?CleanupFileOperator $fileOperator = null;
    private static ?CleanupRepository $repository = null;
    private static ?CleanupTaskManager $taskManager = null;
    private static ?CleanupScheduler $scheduler = null;
    private static ?MemoryManager $memoryManager = null;

    // Initialize all helpers with dependencies
    public static function initialize(
        \wpdb $wpdb,
        LoggerInterface $logger,
        // ... other dependencies
    ): void;

    // Static facade methods delegate to instance methods
    public static function delete_directory(string $dir): bool {
        return self::$fileOperator->delete_directory($dir);
    }
    // ... other facade methods
}
```

---

## 2. RetryOperationHelper.php (2,658 lines)

### Current Issues
- **2,658 lines** - massive class with too many responsibilities
- **Mixed concerns** - scheduling, execution, validation, database operations, policy management
- **Static properties** - `$executors`, `$signalShutdown` create global state

### Identified Responsibilities

1. **Queue Operations** (~600 lines)
   - `enqueue_retry()`
   - `add_to_retry_queue()`
   - `store_retry_job()`
   - `retrieve_and_lock_ready_jobs()`
   - `remove_retry_operation()`

2. **Job Execution** (~800 lines)
   - `process_retry()`
   - `process_all_retries()`
   - `execute_retry_operation()`
   - `retry_failed_job()`
   - `retry_failed_jobs_bulk()`

3. **Job Scheduling & Rescheduling** (~500 lines)
   - `reschedule_failed_job()`
   - `calculate_delay()`
   - `schedule_retry_processor_event()`
   - `unschedule_retry_processor_event()`

4. **Dead Letter Queue (DLQ)** (~300 lines)
   - `move_to_dlq()`
   - `get_dlq_reason_description()`

5. **Dependency Management** (~200 lines)
   - `promote_dependent_jobs()`
   - `promote_waiting_jobs()`

6. **Job State Management** (~400 lines)
   - `mark_as_failed()`
   - `mark_as_completed()`
   - `update_heartbeat()`
   - `get_active_job_count_for_group()`

7. **History & Logging** (~300 lines)
   - `log_history()`
   - `get_recently_locked_cache_key()`

8. **Validation & Policy** (~300 lines)
   - `should_retry_exception()`
   - `check_poison_pill()`
   - `get_retry_config()`
   - `process_expired_jobs()`

### Proposed Split

```php
// 1. RetryQueue (queue management)
class RetryQueue {
    private \wpdb $wpdb;
    private \LHA\Interfaces\LoggerInterface $logger;
    private \LHA\Interfaces\LockInterface $lock;

    public function __construct(\wpdb, LoggerInterface, LockInterface);
    public function enqueue(array $data): int|false;
    public function store_job(string $category, string $operation_type, array $data, array $options): int|false;
    public function retrieve_and_lock_ready_jobs(int $batch_size, string $processor_id): array;
    public function remove_job(int $job_id): bool;
    public function get_job(int $job_id): ?array;
}

// 2. RetryExecutor (job execution)
class RetryExecutor {
    private RetryQueue $queue;
    private \LHA\Interfaces\LoggerInterface $logger;
    private array $executors = [];

    public function __construct(RetryQueue, LoggerInterface);
    public function register_executor(string $operation_type, callable $executor): void;
    public function process_retry(int $retry_id): bool;
    public function process_all_retries(int $batch_size): void;
    public function execute_retry_operation(array $job): bool;
    public function retry_failed_job(int $asset_id): bool;
    public function retry_failed_jobs_bulk(array $asset_ids): array;
}

// 3. RetryScheduler (scheduling & rescheduling)
class RetryScheduler {
    private \wpdb $wpdb;
    private \LHA\Interfaces\LoggerInterface $logger;
    private RetryPolicyManager $policyManager;

    public function __construct(\wpdb, LoggerInterface, RetryPolicyManager);
    public function reschedule_failed_job(array $job, string $error_msg, string $error_code, string $lock_token): bool;
    public function calculate_delay(array $job): int;
    public function schedule_processor_event(): void;
    public function unschedule_processor_event(): void;
}

// 4. RetryPolicyManager (policy & validation)
class RetryPolicyManager {
    private \LHA\Interfaces\LoggerInterface $logger;

    public function __construct(LoggerInterface);
    public function get_retry_config(): array;
    public function should_retry_exception(\Throwable $exception): bool;
    public function check_poison_pill(int $job_id, string $error_code): bool;
    public function get_max_attempts(): int;
    public function get_backoff_strategy(): string;
}

// 5. RetryDeadLetterQueue (DLQ management)
class RetryDeadLetterQueue {
    private \wpdb $wpdb;
    private \LHA\Interfaces\LoggerInterface $logger;

    public function __construct(\wpdb, LoggerInterface);
    public function move_to_dlq(array $job, string $reason, string $dlq_reason_code): bool;
    public function get_dlq_reason_description(string $code): string;
}

// 6. RetryStateManager (state transitions)
class RetryStateManager {
    private \wpdb $wpdb;
    private \LHA\Interfaces\LoggerInterface $logger;

    public function __construct(\wpdb, LoggerInterface);
    public function mark_as_failed(int $job_id, string $error_msg, string $error_code, ?string $lock_token): bool;
    public function mark_as_completed(int $job_id, ?string $lock_token): bool;
    public function update_heartbeat(int $job_id, string $lock_token): bool;
    public function get_active_job_count(string $group): int;
    public function process_expired_jobs(): int;
}

// 7. RetryDependencyManager (dependency handling)
class RetryDependencyManager {
    private \wpdb $wpdb;
    private \LHA\Interfaces\LoggerInterface $logger;

    public function __construct(\wpdb, LoggerInterface);
    public function promote_dependent_jobs(int $job_id, bool $success): void;
    public function promote_waiting_jobs(): void;
}

// 8. RetryHistoryLogger (history tracking)
class RetryHistoryLogger {
    private \wpdb $wpdb;
    private \LHA\Interfaces\LoggerInterface $logger;

    public function __construct(\wpdb, LoggerInterface);
    public function log_history(array $job, string $status, ?int $duration_ms, ?string $error_msg, ?string $error_code, string $processor_id, ?string $stack_trace, ?array $state): void;
    public function get_recently_locked_cache_key(): string;
}

// 9. RetryOperationHelper (facade/orchestrator)
class RetryOperationHelper {
    private RetryQueue $queue;
    private RetryExecutor $executor;
    private RetryScheduler $scheduler;
    private RetryPolicyManager $policyManager;
    private RetryDeadLetterQueue $dlq;
    private RetryStateManager $stateManager;
    private RetryDependencyManager $dependencyManager;
    private RetryHistoryLogger $historyLogger;

    public function __construct(
        \wpdb $wpdb,
        LoggerInterface $logger,
        LockInterface $lock
    ) {
        // Initialize all components
        $this->queue = new RetryQueue($wpdb, $logger, $lock);
        $this->policyManager = new RetryPolicyManager($logger);
        $this->scheduler = new RetryScheduler($wpdb, $logger, $this->policyManager);
        $this->dlq = new RetryDeadLetterQueue($wpdb, $logger);
        $this->stateManager = new RetryStateManager($wpdb, $logger);
        $this->dependencyManager = new RetryDependencyManager($wpdb, $logger);
        $this->historyLogger = new RetryHistoryLogger($wpdb, $logger);
        $this->executor = new RetryExecutor($this->queue, $logger);
    }

    // Public API methods delegate to components
    public function enqueue_retry(array $data): int|false {
        return $this->queue->enqueue($data);
    }

    public function process_retry(int $retry_id): bool {
        return $this->executor->process_retry($retry_id);
    }

    // ... other facade methods
}
```

---

## 3. ProcessTaskHelper.php (2,311 lines)

### Current Issues
- **2,311 lines** - enormous class
- **14 constructor dependencies** - excessive constructor parameter count
- **Already uses helper classes** but still too large
- **Mixed UI processing and business logic**

### Identified Responsibilities

1. **Admin Form Processing** (~600 lines)
   - `process_add_asset()`
   - `process_edit_asset()`
   - `process_delete_asset()`
   - `process_bulk_actions()`
   - `process_manual_validation()`
   - `process_force_refresh_cache()`
   - `process_remediate_order_table()`

2. **Task Processing Orchestration** (~400 lines)
   - `process_scheduled_task()`
   - `process_task()`

3. **Asset Type Processors** (~300 lines)
   - `process_css_task()`
   - `process_js_task()`
   - `process_svg_task()`
   - `process_generic_asset_task()`

4. **Content Processing** (~900 lines)
   - `process_inline_scripts()`
   - `process_css_content()`
   - `process_js_content()`
   - `process_svg_content()`

5. **Batch Operations** (~111 lines)
   - `process_all_css()`

### Proposed Split

```php
// 1. AssetFormProcessor (admin form handling)
class AssetFormProcessor {
    private \LHA\Interfaces\LoggerInterface $logger;
    private \LHA\Interfaces\DatabaseInterface $database;
    private \LHA\Interfaces\AssetValidatorInterface $validator;
    private AssetTaskScheduler $taskScheduler;

    public function __construct(LoggerInterface, DatabaseInterface, AssetValidatorInterface, AssetTaskScheduler);
    public function process_add_asset(): void;
    public function process_edit_asset(): void;
    public function process_delete_asset(): void;
    public function process_bulk_actions(): void;
}

// 2. AssetTaskScheduler (task scheduling)
class AssetTaskScheduler {
    private \LHA\Interfaces\TaskQueueInterface $tasks;
    private \LHA\Interfaces\LoggerInterface $logger;

    public function __construct(TaskQueueInterface, LoggerInterface);
    public function schedule_asset_task(int $asset_id, array $options): bool;
    public function schedule_batch_tasks(array $asset_ids, string $action): array;
}

// 3. TaskOrchestrator (task execution coordination)
class TaskOrchestrator {
    private \LHA\Interfaces\LoggerInterface $logger;
    private AssetProcessorFactory $processorFactory;
    private \LHA\ProcessHelpers\ProcessValidationHelper $validationHelper;

    public function __construct(LoggerInterface, AssetProcessorFactory, ProcessValidationHelper);
    public function process_scheduled_task(string $serialized_task_data): void;
    public function process_task(int $task_id, int $priority, array $task): void;
}

// 4. AssetProcessorFactory (create type-specific processors)
class AssetProcessorFactory {
    private \LHA\Interfaces\LoggerInterface $logger;
    private \LHA\Interfaces\DatabaseInterface $database;
    // ... other dependencies

    public function __construct(/* dependencies */);
    public function create_css_processor(): CssAssetProcessor;
    public function create_js_processor(): JsAssetProcessor;
    public function create_svg_processor(): SvgAssetProcessor;
    public function create_generic_processor(): GenericAssetProcessor;
}

// 5. CssAssetProcessor (CSS-specific processing)
class CssAssetProcessor {
    private \LHA\Interfaces\ExtractInterface $extract;
    private \LHA\Interfaces\SelfHostInterface $selfHost;
    private \LHA\Interfaces\CacheInterface $cache;

    public function __construct(ExtractInterface, SelfHostInterface, CacheInterface);
    public function process_task(array $task): void;
    public function process_content(string $css_content, string $css_url, array $options): string|false;
}

// 6. JsAssetProcessor (JS-specific processing)
class JsAssetProcessor {
    private \LHA\Interfaces\ExtractInterface $extract;
    private \LHA\Interfaces\SelfHostInterface $selfHost;
    private \LHA\Interfaces\CacheInterface $cache;

    public function __construct(ExtractInterface, SelfHostInterface, CacheInterface);
    public function process_task(array $task): void;
    public function process_content(string $js_content, string $js_url, array $options): string;
}

// 7. SvgAssetProcessor (SVG-specific processing)
class SvgAssetProcessor {
    private \LHA\Interfaces\ExtractInterface $extract;
    private \LHA\Interfaces\SelfHostInterface $selfHost;
    private \LHA\Interfaces\SVGInterface $svg;

    public function __construct(ExtractInterface, SelfHostInterface, SVGInterface);
    public function process_task(array $task): void;
    public function process_content(string $svg_content, string $svg_url, array $options): string|false;
}

// 8. InlineScriptProcessor (inline script processing)
class InlineScriptProcessor {
    private \LHA\Interfaces\ExtractInterface $extract;
    private CssAssetProcessor $cssProcessor;
    private JsAssetProcessor $jsProcessor;

    public function __construct(ExtractInterface, CssAssetProcessor, JsAssetProcessor);
    public function process_inline_scripts(string $html_content, string $page_url, bool $force_refresh): string;
}

// 9. BatchAssetProcessor (batch operations)
class BatchAssetProcessor {
    private CssAssetProcessor $cssProcessor;
    private \LHA\Interfaces\LoggerInterface $logger;

    public function __construct(CssAssetProcessor, LoggerInterface);
    public function process_all_css(bool $force_refresh, int $cache_expiration_days, int $offset, int $limit): void;
}

// 10. ProcessTaskHelper (facade/orchestrator)
class ProcessTaskHelper {
    private AssetFormProcessor $formProcessor;
    private TaskOrchestrator $taskOrchestrator;
    private InlineScriptProcessor $inlineProcessor;
    private BatchAssetProcessor $batchProcessor;

    public function __construct(
        LoggerInterface $logger,
        DatabaseInterface $database,
        // ... all dependencies
    ) {
        // Initialize components
        $processorFactory = new AssetProcessorFactory(/* dependencies */);
        $this->formProcessor = new AssetFormProcessor(/* dependencies */);
        $this->taskOrchestrator = new TaskOrchestrator($logger, $processorFactory, $validationHelper);
        $this->inlineProcessor = new InlineScriptProcessor(/* dependencies */);
        $this->batchProcessor = new BatchAssetProcessor(/* dependencies */);
    }

    // Public API delegates to components
    public function process_add_asset(): void {
        $this->formProcessor->process_add_asset();
    }

    public function process_task(int $task_id, int $priority, array $task): void {
        $this->taskOrchestrator->process_task($task_id, $priority, $task);
    }

    // ... other facade methods
}
```

---

## Refactoring Strategy

### Phase 1: Preparation (1-2 days)
1. **Create interfaces** for all new classes
2. **Set up dependency injection** container updates
3. **Write integration tests** for existing behavior
4. **Create feature branch** for refactoring

### Phase 2: CleanupStaticHelper (2-3 days)
1. Create new classes:
   - `CleanupFileOperator`
   - `CleanupRepository`
   - `CleanupTaskManager`
   - `CleanupScheduler`
   - `MemoryManager`
2. Convert static methods to instance methods
3. Update `CleanupStaticHelper` to be a facade
4. Update all usages
5. Run tests

### Phase 3: RetryOperationHelper (3-4 days)
1. Create new classes:
   - `RetryQueue`
   - `RetryExecutor`
   - `RetryScheduler`
   - `RetryPolicyManager`
   - `RetryDeadLetterQueue`
   - `RetryStateManager`
   - `RetryDependencyManager`
   - `RetryHistoryLogger`
2. Move methods to appropriate classes
3. Update `RetryOperationHelper` to be an orchestrator
4. Update all usages
5. Run tests

### Phase 4: ProcessTaskHelper (4-5 days)
1. Create new classes:
   - `AssetFormProcessor`
   - `AssetTaskScheduler`
   - `TaskOrchestrator`
   - `AssetProcessorFactory`
   - `CssAssetProcessor`
   - `JsAssetProcessor`
   - `SvgAssetProcessor`
   - `InlineScriptProcessor`
   - `BatchAssetProcessor`
2. Move methods to appropriate classes
3. Update `ProcessTaskHelper` to be a facade
4. Update all usages
5. Run tests

### Phase 5: Cleanup & Testing (2-3 days)
1. **Remove dead code**
2. **Update documentation**
3. **Run full test suite**
4. **Performance testing**
5. **Code review**

---

## Benefits of Refactoring

### Maintainability
- **Single Responsibility Principle** - each class has one clear purpose
- **Easier to understand** - smaller classes are easier to comprehend
- **Easier to modify** - changes are localized to specific classes

### Testability
- **Dependency Injection** - all dependencies are injectable
- **Mockable components** - each component can be mocked independently
- **Focused unit tests** - test each component in isolation

### Extensibility
- **Open/Closed Principle** - open for extension, closed for modification
- **Plugin architecture** - new executors/processors can be registered
- **Strategy pattern** - policies can be swapped without changing code

### Performance
- **Lazy loading** - components can be instantiated only when needed
- **Better caching** - focused classes can implement targeted caching
- **Reduced memory footprint** - smaller classes load faster

---

## Risk Mitigation

### Backward Compatibility
- Keep original classes as **facades** that delegate to new classes
- Maintain **existing public API**
- Use **semantic versioning** - this is a major version change

### Testing Strategy
- **Characterization tests** - capture current behavior before refactoring
- **Integration tests** - ensure components work together
- **Regression tests** - verify no functionality is broken

### Incremental Approach
- Refactor **one class at a time**
- Run **tests after each change**
- **Commit frequently** with atomic changes

---

## Recommended Next Steps

1. **Start with CleanupStaticHelper** - it's the smallest and simplest
2. **Create a proof of concept** - refactor one method completely
3. **Get team feedback** - review the approach before proceeding
4. **Schedule dedicated time** - this refactoring requires focus
5. **Consider pair programming** - complex refactoring benefits from collaboration

---

## Metrics for Success

- **Lines per class** - target < 500 lines per class
- **Constructor parameters** - target < 7 parameters per class
- **Cyclomatic complexity** - target < 10 per method
- **Test coverage** - maintain or improve current coverage
- **Performance** - no regression in processing speed

---

## Conclusion

This refactoring will transform three massive, difficult-to-maintain god classes into approximately 30 focused, single-responsibility classes. The result will be:

- **More maintainable codebase**
- **Better testability**
- **Easier to extend**
- **Clearer architecture**
- **Reduced technical debt**

The estimated effort is **12-17 days** of focused development work, with the recommended approach being to tackle one class at a time, starting with the smallest (CleanupStaticHelper) and working up to the largest (ProcessTaskHelper).
