<?php

declare(strict_types=1);

namespace LHA;

use LHA\CleanupHelpers\CleanupOperationHelper;
use LHA\CleanupHelpers\CleanupDeleteHelper;
use LHA\CleanupHelpers\CleanupClearHelper;
use LHA\CleanupHelpers\CleanupQueryHelper;
use LHA\CleanupHelpers\CleanupScheduleHelper;
use LHA\CleanupHelpers\CleanupUtilityHelper;
use LHA\CleanupHelpers\CleanupStaticHelper;
use LHA\Interfaces\LoggerInterface;
use LHA\GetData;
use LHA\Interfaces\DatabaseInterface;
use LHA\FileLock;

/**
 * Class Cleanup
 *
 * Facade for cleanup operations. Routes to specialized helper classes.
 *
 * Production Ready: Yes
 */
class Cleanup implements \LHA\Interfaces\CleanupInterface
{
    // ============================================================
    // Stub methods to satisfy CleanupInterface
    // These delegate to __call() for actual implementation
    // ============================================================

    public function cleanup_temp_files(int $age_hours = 24): int {
        return $this->__call('cleanup_temp_files', func_get_args());
    }

    public function cleanup_old_logs(int $age_days = 30): int {
        return $this->__call('cleanup_old_logs', func_get_args());
    }

    public function cleanup_cache(int $age_hours = 24): int {
        return $this->__call('cleanup_cache', func_get_args());
    }

    public function cleanup_database(int $age_days = 30): int {
        return $this->__call('cleanup_database', func_get_args());
    }

    public function cleanup_orphaned_files(): int {
        return $this->__call('cleanup_orphaned_files', func_get_args());
    }

    public function cleanup_expired_assets(): int {
        return $this->__call('cleanup_expired_assets', func_get_args());
    }

    public function cleanup_retry_queue(): int {
        return $this->__call('cleanup_retry_queue', func_get_args());
    }

    public function cleanup_task_queue(): int {
        return $this->__call('cleanup_task_queue', func_get_args());
    }

    public function cleanup_mapping_table(): int {
        return $this->__call('cleanup_mapping_table', func_get_args());
    }

    public function cleanup_order_table(): int {
        return $this->__call('cleanup_order_table', func_get_args());
    }

    public function cleanup_options(): int {
        return $this->__call('cleanup_options', func_get_args());
    }

    public function cleanup_transients(): int {
        return $this->__call('cleanup_transients', func_get_args());
    }

    public function cleanup_cache_directory(string $subdir = ''): int {
        return $this->__call('cleanup_cache_directory', func_get_args());
    }

    public function cleanup_old_asset_versions(int $keep_versions = 1): int {
        return $this->__call('cleanup_old_asset_versions', func_get_args());
    }

    public function cleanup_unlinked_files(): int {
        return $this->__call('cleanup_unlinked_files', func_get_args());
    }

    public function cleanup_unused_assets(): int {
        return $this->__call('cleanup_unused_assets', func_get_args());
    }

    public function cleanup_failed_downloads(): int {
        return $this->__call('cleanup_failed_downloads', func_get_args());
    }

    public function cleanup_stale_locks(): int {
        return $this->__call('cleanup_stale_locks', func_get_args());
    }

    public function cleanup_old_backups(): int {
        return $this->__call('cleanup_old_backups', func_get_args());
    }

    public function cleanup_security_logs(): int {
        return $this->__call('cleanup_security_logs', func_get_args());
    }

    public function get_cleanup_statistics(): array {
        return $this->__call('get_cleanup_statistics', func_get_args());
    }

    public function schedule_cleanup(string $schedule = 'daily'): bool {
        return $this->__call('schedule_cleanup', func_get_args());
    }

    public function run_cleanup_routine(): bool {
        return $this->__call('run_cleanup_routine', func_get_args());
    }

    public function check_cleanup_needed(): bool {
        return $this->__call('check_cleanup_needed', func_get_args());
    }

    public function get_cleanup_schedule(): array {
        return $this->__call('get_cleanup_schedule', func_get_args());
    }

    public function cancel_cleanup_schedule(): bool {
        return $this->__call('cancel_cleanup_schedule', func_get_args());
    }

    public function set_cleanup_config(array $config): bool {
        return $this->__call('set_cleanup_config', func_get_args());
    }

    public function get_cleanup_config(): array {
        return $this->__call('get_cleanup_config', func_get_args());
    }

    public function perform_aggressive_cleanup(): bool {
        return $this->__call('perform_aggressive_cleanup', func_get_args());
    }

    public function perform_light_cleanup(): bool {
        return $this->__call('perform_light_cleanup', func_get_args());
    }

    public function delete_asset_file(int $asset_id, bool $force = false): bool {
        return $this->__call('delete_asset_file', func_get_args());
    }

    public function delete_asset_files(array $asset_ids, bool $force = false): array {
        return $this->__call('delete_asset_files', func_get_args());
    }

    public function clear_asset_transients(): bool {
        return $this->__call('clear_asset_transients', func_get_args());
    }

    public function cleanup_task_resources(int $task_id): bool {
        return $this->__call('cleanup_task_resources', func_get_args());
    }


    // Dependencies
    private LoggerInterface $logger;
    private GetData $getdata;
    private DatabaseInterface $database;
    private FileLock $fileLock;

    // Lazy-loaded helpers
    private ?CleanupOperationHelper $operationHelper = null;
    private ?CleanupDeleteHelper $deleteHelper = null;
    private ?CleanupClearHelper $clearHelper = null;
    private ?CleanupQueryHelper $queryHelper = null;
    private ?CleanupScheduleHelper $scheduleHelper = null;
    private ?CleanupUtilityHelper $utilityHelper = null;

    /**
     * Constructor
     */
    public function __construct(
        LoggerInterface $logger,
        GetData $getdata,
        DatabaseInterface $database,
        FileLock $fileLock
    ) {
        $this->logger = $logger;
        $this->getdata = $getdata;
        $this->database = $database;
        $this->fileLock = $fileLock;
    }

    /**
     * Magic call method to route to helper classes
     */
    public function __call(string $name, array $arguments)
    {
        static $methodMap = [
            'cleanup_task_resources' => 'CleanupOperationHelper',
            'cleanup_temp_files' => 'CleanupOperationHelper',
            'cleanup_old_logs' => 'CleanupOperationHelper',
            'cleanup_cache' => 'CleanupOperationHelper',
            'cleanup_database' => 'CleanupOperationHelper',
            'cleanup_orphaned_files' => 'CleanupOperationHelper',
            'cleanup_expired_assets' => 'CleanupOperationHelper',
            'cleanup_retry_queue' => 'CleanupOperationHelper',
            'cleanup_task_queue' => 'CleanupOperationHelper',
            'cleanup_mapping_table' => 'CleanupOperationHelper',
            'cleanup_order_table' => 'CleanupOperationHelper',
            'cleanup_options' => 'CleanupOperationHelper',
            'cleanup_transients' => 'CleanupOperationHelper',
            'cleanup_cache_directory' => 'CleanupOperationHelper',
            'cleanup_old_asset_versions' => 'CleanupOperationHelper',
            'cleanup_unlinked_files' => 'CleanupOperationHelper',
            'cleanup_unused_assets' => 'CleanupOperationHelper',
            'cleanup_failed_downloads' => 'CleanupOperationHelper',
            'cleanup_stale_locks' => 'CleanupOperationHelper',
            'cleanup_old_backups' => 'CleanupOperationHelper',
            'cleanup_security_logs' => 'CleanupOperationHelper',
            'delete_asset_file' => 'CleanupDeleteHelper',
            'delete_asset_files' => 'CleanupDeleteHelper',
            'clear_asset_transients' => 'CleanupClearHelper',
            'get_cleanup_statistics' => 'CleanupQueryHelper',
            'get_cleanup_schedule' => 'CleanupQueryHelper',
            'get_cleanup_config' => 'CleanupQueryHelper',
            'schedule_cleanup' => 'CleanupScheduleHelper',
            'run_cleanup_routine' => 'CleanupScheduleHelper',
            'check_cleanup_needed' => 'CleanupScheduleHelper',
            'cancel_cleanup_schedule' => 'CleanupScheduleHelper',
            'set_cleanup_config' => 'CleanupUtilityHelper',
            'perform_aggressive_cleanup' => 'CleanupUtilityHelper',
            'perform_light_cleanup' => 'CleanupUtilityHelper',
        ];

        if (!isset($methodMap[$name])) {
            throw new \BadMethodCallException("Method $name does not exist");
        }

        $helperClass = $methodMap[$name];
        $helper = $this->getHelper($helperClass);
        return $helper->$name(...$arguments);
    }

    /**
     * Static proxy using __callStatic magic method
     */
    public static function __callStatic(string $method, array $arguments)
    {
        // Static methods - use static variable for performance
        static $staticMethods = [
            'cleanup_resources',
            'delete_directory',
            'maybe_schedule_cleanup',
            'perform_global_cleanup',
            'check_and_cleanup_memory',
            'perform_periodic_cleanup',
            'clear_temporary_data',
            'get_temp_files',
            'cleanup_orphaned_tasks',
            'cleanup_stale_queue_items',
            'cleanup_task_resources_static',
            'cleanup_failed_task',
            'cleanup_existing_task',
            'cleanup_failed_enqueue',
            'cleanup_on_failure',
            'unschedule_cleanup_cron',
        ];

        // Check if it's a static method in CleanupStaticHelper
        if (in_array($method, $staticMethods)) {
            $helperClass = '\\LHA\\CleanupHelpers\\CleanupStaticHelper';
            return call_user_func_array([$helperClass, $method], $arguments);
        }

        // For other static calls, try to get an instance from container
        global $lha_container;

        if (isset($lha_container) && $lha_container instanceof \LHA\ServiceContainer) {
            try {
                $instance = $lha_container->get(Cleanup::class);
                if ($instance !== null) {
                    return $instance->$method(...$arguments);
                }
            } catch (\Throwable $e) {
                // Fall through
            }
        }

        throw new \BadMethodCallException("Static method $method does not exist or could not be routed to instance");
    }

    /**
     * Get helper instance (lazy loading)
     */
    private function getHelper(string $helperClass): object
    {
        return match($helperClass) {
            'CleanupOperationHelper' => $this->getOperationHelper(),
            'CleanupDeleteHelper' => $this->getDeleteHelper(),
            'CleanupClearHelper' => $this->getClearHelper(),
            'CleanupQueryHelper' => $this->getQueryHelper(),
            'CleanupScheduleHelper' => $this->getScheduleHelper(),
            'CleanupUtilityHelper' => $this->getUtilityHelper(),
            default => throw new \InvalidArgumentException("Unknown helper: $helperClass"),
        };
    }

    private function getOperationHelper(): CleanupOperationHelper
    {
        if ($this->operationHelper === null) {
            $this->operationHelper = new CleanupOperationHelper(
                $this->logger,
                $this->getdata,
                $this->database,
                $this->fileLock
            );
        }
        return $this->operationHelper;
    }

    private function getDeleteHelper(): CleanupDeleteHelper
    {
        if ($this->deleteHelper === null) {
            $this->deleteHelper = new CleanupDeleteHelper(
                $this->logger,
                $this->database
            );
        }
        return $this->deleteHelper;
    }

    private function getClearHelper(): CleanupClearHelper
    {
        if ($this->clearHelper === null) {
            $this->clearHelper = new CleanupClearHelper(
                $this->logger,
                $this->database
            );
        }
        return $this->clearHelper;
    }

    private function getQueryHelper(): CleanupQueryHelper
    {
        if ($this->queryHelper === null) {
            $this->queryHelper = new CleanupQueryHelper(
                $this->logger,
                $this->database
            );
        }
        return $this->queryHelper;
    }

    private function getScheduleHelper(): CleanupScheduleHelper
    {
        if ($this->scheduleHelper === null) {
            $this->scheduleHelper = new CleanupScheduleHelper(
                $this->logger
            );
        }
        return $this->scheduleHelper;
    }

    private function getUtilityHelper(): CleanupUtilityHelper
    {
        if ($this->utilityHelper === null) {
            $this->utilityHelper = new CleanupUtilityHelper(
                $this->logger,
                $this->getdata,
                $this->database
            );
        }
        return $this->utilityHelper;
    }
}
