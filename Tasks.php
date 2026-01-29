<?php

namespace LHA;

use LHA\TaskHelpers\TaskEnqueueHelper;
use LHA\TaskHelpers\TaskProcessingHelper;
use LHA\TaskHelpers\TaskStatusHelper;
use LHA\TaskHelpers\TaskQueryHelper;
use LHA\TaskHelpers\TaskScheduleHelper;
use LHA\TaskHelpers\TaskMaintenanceHelper;
use LHA\TaskHelpers\TaskUtilityHelper;
use LHA\TaskHelpers\TaskCacheHelper;
use LHA\TaskHelpers\TaskSchedulerHelper;
use LHA\TaskHelpers\TaskValidationHelper;
use LHA\TaskHelpers\TaskCronHelper;

/**
 * Class Tasks
 *
 * Manages task queueing, processing, and related cron events for the Self-Host Assets plugin.
 * REFACTORED: Now acts as a facade that delegates to specialized helper classes
 * for better separation of concerns, maintainability, and performance.
 * Designed for enterprise production environments with high standards for performance, security, and robustness.
 */
class Tasks implements \LHA\Interfaces\TaskQueueInterface, \LHA\Interfaces\TasksInterface
{
    public const STATUS_PENDING    = 'pending';
    public const STATUS_PROCESSING = 'processing';
    public const STATUS_PROCESSED  = 'processed';
    public const STATUS_FAILED     = 'failed';
    public const CACHE_GROUP = 'lha_tasks';
    // Helper instances - each handles a specific domain of task operations
    private TaskEnqueueHelper $enqueueHelper;
    private TaskProcessingHelper $processingHelper;
    private TaskStatusHelper $statusHelper;
    private TaskQueryHelper $queryHelper;
    private TaskScheduleHelper $scheduleHelper;
    private TaskMaintenanceHelper $maintenanceHelper;
    private TaskUtilityHelper $utilityHelper;
    private TaskCacheHelper $cacheHelper;
    private TaskSchedulerHelper $schedulerHelper;
    private TaskValidationHelper $validationHelper;
    private TaskCronHelper $cronHelper;

    // Keep original dependencies for backward compatibility
    private \LHA\Interfaces\LoggerInterface $logger;
    private \LHA\Interfaces\DatabaseInterface $database;
    private \wpdb $wpdb;
    private \LHA\Interfaces\LockInterface $lock;
    private ?\LHA\Interfaces\CleanupInterface $cleanup = null;
    private ?\LHA\TaskProcessorManager $processor_manager = null;
    private ?\LHA\Interfaces\AssetDataInterface $assetData = null;
    private ?\LHA\Interfaces\AssetValidatorInterface $assetValidator = null;
    private ?\LHA\Interfaces\NormalizeInterface $normalize = null;
    private ?\LHA\Interfaces\CacheInterface $cache = null;
    private ?\LHA\Interfaces\UrlProcessorInterface $urlProcessor = null;
    private ?\LHA\Interfaces\GetdataInterface $getdata = null;
    private ?\LHA\Interfaces\ProcessInterface $process = null;
    private ?\LHA\Interfaces\ServiceContainerInterface $container = null;
    private ?\LHA\Interfaces\AssetUtilsInterface $assetUtils = null;

    public function __construct(
        \LHA\Interfaces\LoggerInterface $logger,
        \LHA\Interfaces\DatabaseInterface $database,
        \wpdb $wpdb,
        \LHA\Interfaces\LockInterface $lock,
        ?\LHA\Interfaces\CleanupInterface $cleanup = null,
        ?\LHA\TaskProcessorManager $processor_manager = null,
        ?\LHA\Interfaces\AssetDataInterface $assetData = null,
        ?\LHA\Interfaces\AssetValidatorInterface $assetValidator = null,
        ?\LHA\Interfaces\NormalizeInterface $normalize = null,
        ?\LHA\Interfaces\CacheInterface $cache = null,
        ?\LHA\Interfaces\UrlProcessorInterface $urlProcessor = null,
        ?\LHA\Interfaces\GetdataInterface $getdata = null,
        ?\LHA\Interfaces\ProcessInterface $process = null,
        ?\LHA\Interfaces\ServiceContainerInterface $container = null,
        ?\LHA\Interfaces\AssetUtilsInterface $assetUtils = null
    ) {
        $this->logger = $logger;
        $this->database = $database;
        $this->wpdb = $wpdb;
        $this->lock = $lock;
        $this->cleanup = $cleanup;
        $this->processor_manager = $processor_manager;
        $this->assetData = $assetData;
        $this->assetValidator = $assetValidator;
        $this->normalize = $normalize;
        $this->cache = $cache;
        $this->urlProcessor = $urlProcessor;
        $this->getdata = $getdata;
        $this->process = $process;
        $this->container = $container;
        $this->assetUtils = $assetUtils;

        // Initialize processor manager if provided
        if ( $this->processor_manager !== null ) {
            $this->processor_manager->set_task_callback( [ $this, 'process_task' ] );
            $this->processor_manager->set_batch_callback( [ $this, 'process_task_batch' ] );
        }

        // Instantiate helper classes after dependencies are set
        $this->enqueueHelper = new \LHA\TaskHelpers\TaskEnqueueHelper(
            $this->logger,
            $this->cache,
            $this->database,
            $this->processor_manager
        );

        $this->processingHelper = new \LHA\TaskHelpers\TaskProcessingHelper(
            $this->logger,
            $this->cache,
            $this->database,
            null, // Settings
            null, // Initialize
            $this->processor_manager
        );

        $this->statusHelper = new \LHA\TaskHelpers\TaskStatusHelper(
            $this->logger,
            $this->cache,
            $this->database
        );

        $this->queryHelper = new \LHA\TaskHelpers\TaskQueryHelper(
            $this->logger,
            $this->cache,
            $this->database
        );

        $this->scheduleHelper = new \LHA\TaskHelpers\TaskScheduleHelper(
            $this->logger,
            $this->cache,
            null, // Settings
            null, // Options
            $this->processor_manager
        );

        $this->maintenanceHelper = new \LHA\TaskHelpers\TaskMaintenanceHelper(
            $this->logger,
            $this->cache,
            $this->database,
            null // Retry
        );

        $this->utilityHelper = new \LHA\TaskHelpers\TaskUtilityHelper(
            $this->logger,
            $this->cache,
            null, // Settings
            null, // Options
            $this->process,
            $this->processor_manager
        );

        $this->cacheHelper = new \LHA\TaskHelpers\TaskCacheHelper(
            $this->logger,
            $this->cache,
            null // Options
        );

        $this->schedulerHelper = new \LHA\TaskHelpers\TaskSchedulerHelper(
            $this->logger,
            $this->cache,
            $this->database,
            null // Settings
        );

        $this->validationHelper = new \LHA\TaskHelpers\TaskValidationHelper(
            $this->logger
        );

        $this->cronHelper = new \LHA\TaskHelpers\TaskCronHelper(
            $this->logger,
            $this->cache,
            $this->database
        );
    }

    /**
     * Facade method for enqueue_task
     * Delegates to enqueueHelper
     */
    public function enqueue_task(...$args) {
        return $this->enqueueHelper->enqueue_task(...$args);
    }

    /**
     * Facade method for enqueue_asset_task
     * Delegates to enqueueHelper
     */
    public function enqueue_asset_task(...$args) {
        return $this->enqueueHelper->enqueue_asset_task(...$args);
    }

    /**
     * Facade method for enqueue_asset_task_by_id
     * Delegates to enqueueHelper
     */
    public function enqueue_asset_task_by_id(...$args) {
        return $this->enqueueHelper->enqueue_asset_task_by_id(...$args);
    }

    /**
     * Facade method for enqueue_asset_tasks_bulk
     * Delegates to enqueueHelper
     */
    public function enqueue_asset_tasks_bulk(...$args) {
        return $this->enqueueHelper->enqueue_asset_tasks_bulk(...$args);
    }

    /**
     * Facade method for enqueue_reprocess_task
     * Delegates to enqueueHelper
     */
    public function enqueue_reprocess_task(...$args) {
        return $this->enqueueHelper->enqueue_reprocess_task(...$args);
    }

    /**
     * Facade method for enqueue_reprocess_tasks_bulk
     * Delegates to enqueueHelper
     */
    public function enqueue_reprocess_tasks_bulk(...$args) {
        return $this->enqueueHelper->enqueue_reprocess_tasks_bulk(...$args);
    }

    /**
     * Facade method for enqueue_svg_processing_task
     * Delegates to enqueueHelper
     */
    public function enqueue_svg_processing_task(...$args) {
        return $this->enqueueHelper->enqueue_svg_processing_task(...$args);
    }

    /**
     * Facade method for enqueue_task_immediately
     * Delegates to enqueueHelper
     */
    public function enqueue_task_immediately(...$args) {
        return $this->enqueueHelper->enqueue_task_immediately(...$args);
    }

    /**
     * Facade method for batch_enqueue_tasks
     * Delegates to enqueueHelper
     */
    public function batch_enqueue_tasks(...$args) {
        return $this->enqueueHelper->batch_enqueue_tasks(...$args);
    }

    /**
     * Facade method for add_task
     * Delegates to enqueueHelper
     */
    public function add_task(...$args) {
        return $this->enqueueHelper->add_task(...$args);
    }

    /**
     * Facade method for process_task
     * Delegates to processingHelper
     */
    public function process_task(...$args) {
        return $this->processingHelper->process_task(...$args);
    }

    /**
     * Facade method for process_task_batch
     * Delegates to processingHelper
     */
    public function process_task_batch(...$args) {
        return $this->processingHelper->process_task_batch(...$args);
    }

    /**
     * Facade method for process_scheduled_task
     * Delegates to processingHelper
     */
    public function process_scheduled_task(...$args) {
        return $this->processingHelper->process_scheduled_task(...$args);
    }

    /**
     * Facade method for execute_delayed_task
     * Delegates to processingHelper
     */
    public function execute_delayed_task(...$args) {
        return $this->processingHelper->execute_delayed_task(...$args);
    }

    /**
     * Facade method for execute_cron_tasks
     * Delegates to processingHelper
     */
    public function execute_cron_tasks(...$args) {
        return $this->processingHelper->execute_cron_tasks(...$args);
    }

    /**
     * Facade method for handle_delayed_js_task
     * Delegates to processingHelper
     */
    public function handle_delayed_js_task(...$args) {
        return $this->processingHelper->handle_delayed_js_task(...$args);
    }

    /**
     * Facade method for daily_maintenance_callback
     * Delegates to processingHelper
     */
    public function daily_maintenance_callback(...$args) {
        return $this->processingHelper->daily_maintenance_callback(...$args);
    }

    /**
     * Facade method for update_task_status
     * Delegates to statusHelper
     */
    public function update_task_status(...$args) {
        return $this->statusHelper->update_task_status(...$args);
    }

    /**
     * Facade method for update_task_fields
     * Delegates to statusHelper
     */
    public function update_task_fields(...$args) {
        return $this->statusHelper->update_task_fields(...$args);
    }

    /**
     * Facade method for batch_update_task_status
     * Delegates to statusHelper
     */
    public function batch_update_task_status(...$args) {
        return $this->statusHelper->batch_update_task_status(...$args);
    }

    /**
     * Facade method for check_task_timeout
     * Delegates to statusHelper
     */
    public function check_task_timeout(...$args) {
        return $this->statusHelper->check_task_timeout(...$args);
    }

    /**
     * Facade method for map_task_status_to_human_readable
     * Delegates to statusHelper
     */
    public function map_task_status_to_human_readable(...$args) {
        return $this->statusHelper->map_task_status_to_human_readable(...$args);
    }

    /**
     * Facade method for get_task_by_id
     * Delegates to statusHelper
     */
    public function get_task_by_id(...$args) {
        return $this->statusHelper->get_task_by_id(...$args);
    }

    /**
     * Facade method for get_tasks_by_ids
     * Delegates to statusHelper
     */
    public function get_tasks_by_ids(...$args) {
        return $this->statusHelper->get_tasks_by_ids(...$args);
    }

    /**
     * Facade method for get_last_task_id
     * Delegates to statusHelper
     */
    public function get_last_task_id(...$args) {
        return $this->statusHelper->get_last_task_id(...$args);
    }

    /**
     * Facade method for get_pending_tasks
     * Delegates to queryHelper
     */
    public function get_pending_tasks(...$args) {
        return $this->queryHelper->get_pending_tasks(...$args);
    }

    /**
     * Facade method for get_pending_asset_tasks
     * Delegates to queryHelper
     */
    public function get_pending_asset_tasks(...$args) {
        return $this->queryHelper->get_pending_asset_tasks(...$args);
    }

    /**
     * Facade method for get_pending_tasks_count
     * Delegates to queryHelper
     */
    public function get_pending_tasks_count(...$args) {
        return $this->queryHelper->get_pending_tasks_count(...$args);
    }

    /**
     * Facade method for get_pending_tasks_batch
     * Delegates to queryHelper
     */
    public function get_pending_tasks_batch(...$args) {
        return $this->queryHelper->get_pending_tasks_batch(...$args);
    }

    /**
     * Facade method for get_pending_tasks_optimized
     * Delegates to queryHelper
     */
    public function get_pending_tasks_optimized(...$args) {
        return $this->queryHelper->get_pending_tasks_optimized(...$args);
    }

    /**
     * Facade method for get_stuck_tasks_optimized
     * Delegates to queryHelper
     */
    public function get_stuck_tasks_optimized(...$args) {
        return $this->queryHelper->get_stuck_tasks_optimized(...$args);
    }

    /**
     * Facade method for has_pending_tasks
     * Delegates to queryHelper
     */
    public function has_pending_tasks(...$args) {
        return $this->queryHelper->has_pending_tasks(...$args);
    }

    /**
     * Facade method for are_tasks_in_progress
     * Delegates to queryHelper
     */
    public function are_tasks_in_progress(...$args) {
        return $this->queryHelper->are_tasks_in_progress(...$args);
    }

    /**
     * Facade method for schedule_task_processing
     * Delegates to scheduleHelper
     */
    public function schedule_task_processing(...$args) {
        return $this->scheduleHelper->schedule_task_processing(...$args);
    }

    /**
     * Facade method for schedule_task_processing_via_cron
     * Delegates to scheduleHelper
     */
    public function schedule_task_processing_via_cron(...$args) {
        return $this->scheduleHelper->schedule_task_processing_via_cron(...$args);
    }

    /**
     * Facade method for schedule_cron_event
     * Delegates to scheduleHelper
     */
    public function schedule_cron_event(...$args) {
        return $this->scheduleHelper->schedule_cron_event(...$args);
    }

    /**
     * Facade method for reschedule_cron_event
     * Delegates to scheduleHelper
     */
    public function reschedule_cron_event(...$args) {
        return $this->scheduleHelper->reschedule_cron_event(...$args);
    }

    /**
     * Facade method for unschedule_cron_event
     * Delegates to scheduleHelper
     */
    public function unschedule_cron_event(...$args) {
        return $this->scheduleHelper->unschedule_cron_event(...$args);
    }

    /**
     * Facade method for manage_cron_events
     * Delegates to scheduleHelper
     */
    public function manage_cron_events(...$args) {
        return $this->scheduleHelper->manage_cron_events(...$args);
    }

    /**
     * Facade method for clear_scheduled_cron_events
     * Delegates to scheduleHelper
     */
    public function clear_scheduled_cron_events(...$args) {
        return $this->scheduleHelper->clear_scheduled_cron_events(...$args);
    }

    /**
     * Facade method for handle_schedule_change
     * Delegates to scheduleHelper
     */
    public function handle_schedule_change(...$args) {
        return $this->scheduleHelper->handle_schedule_change(...$args);
    }

    /**
     * Facade method for ensure_batch_processor_scheduled
     * Delegates to scheduleHelper
     */
    public function ensure_batch_processor_scheduled(...$args) {
        return $this->scheduleHelper->ensure_batch_processor_scheduled(...$args);
    }

    /**
     * Facade method for ensure_batch_processor_scheduled_public
     * Delegates to scheduleHelper
     */
    public function ensure_batch_processor_scheduled_public(...$args) {
        return $this->scheduleHelper->ensure_batch_processor_scheduled_public(...$args);
    }

    /**
     * Facade method for refresh_asset_caches
     * Delegates to scheduleHelper
     */
    public function refresh_asset_caches(...$args) {
        return $this->scheduleHelper->refresh_asset_caches(...$args);
    }

    /**
     * Facade method for schedule_database_retry
     * Delegates to maintenanceHelper
     */
    public function schedule_database_retry(...$args) {
        return $this->maintenanceHelper->schedule_database_retry(...$args);
    }

    /**
     * Facade method for batch_delete_old_tasks
     * Delegates to maintenanceHelper
     */
    public function batch_delete_old_tasks(...$args) {
        return $this->maintenanceHelper->batch_delete_old_tasks(...$args);
    }

    /**
     * Facade method for optimize_database_tables
     * Delegates to maintenanceHelper
     */
    public function optimize_database_tables(...$args) {
        return $this->maintenanceHelper->optimize_database_tables(...$args);
    }

    /**
     * Facade method for verify_task_indexes
     * Delegates to maintenanceHelper
     */
    public function verify_task_indexes(...$args) {
        return $this->maintenanceHelper->verify_task_indexes(...$args);
    }

    /**
     * Facade method for schedule_daily_maintenance
     * Delegates to maintenanceHelper
     */
    public function schedule_daily_maintenance(...$args) {
        return $this->maintenanceHelper->schedule_daily_maintenance(...$args);
    }

    /**
     * Facade method for cleanup_failed_task_files
     * Delegates to maintenanceHelper
     */
    public function cleanup_failed_task_files(...$args) {
        return $this->maintenanceHelper->cleanup_failed_task_files(...$args);
    }

    /**
     * Facade method for cleanup_individual_task_crons
     * Delegates to maintenanceHelper
     */
    public function cleanup_individual_task_crons(...$args) {
        return $this->maintenanceHelper->cleanup_individual_task_crons(...$args);
    }

    /**
     * Facade method for get_task_table_name
     * Delegates to utilityHelper
     */
    public function get_task_table_name(...$args) {
        return $this->utilityHelper->get_task_table_name(...$args);
    }

    /**
     * Facade method for get_config_value
     * Delegates to utilityHelper
     */
    public function get_config_value(...$args) {
        return $this->utilityHelper->get_config_value(...$args);
    }

    /**
     * Facade method for get_transient_via_cache
     * Delegates to utilityHelper
     */
    public function get_transient_via_cache(...$args) {
        return $this->utilityHelper->get_transient_via_cache(...$args);
    }

    /**
     * Facade method for set_transient_via_cache
     * Delegates to utilityHelper
     */
    public function set_transient_via_cache(...$args) {
        return $this->utilityHelper->set_transient_via_cache(...$args);
    }

    /**
     * Facade method for delete_transient_via_cache
     * Delegates to utilityHelper
     */
    public function delete_transient_via_cache(...$args) {
        return $this->utilityHelper->delete_transient_via_cache(...$args);
    }

    /**
     * Facade method for get_process
     * Delegates to utilityHelper
     */
    public function get_process(...$args) {
        return $this->utilityHelper->get_process(...$args);
    }

    /**
     * Facade method for get_processor_manager
     * Delegates to utilityHelper
     */
    public function get_processor_manager(...$args) {
        return $this->utilityHelper->get_processor_manager(...$args);
    }

    /**
     * Facade method for get_processor_status
     * Delegates to utilityHelper
     */
    public function get_processor_status(...$args) {
        return $this->utilityHelper->get_processor_status(...$args);
    }

    /**
     * Facade method for is_using_action_scheduler
     * Delegates to utilityHelper
     */
    public function is_using_action_scheduler(...$args) {
        return $this->utilityHelper->is_using_action_scheduler(...$args);
    }

    /**
     * Facade method for should_use_external_retry
     * Delegates to utilityHelper
     */
    public function should_use_external_retry(...$args) {
        return $this->utilityHelper->should_use_external_retry(...$args);
    }

    /**
     * Facade method for has_native_retry
     * Delegates to utilityHelper
     */
    public function has_native_retry(...$args) {
        return $this->utilityHelper->has_native_retry(...$args);
    }

    /**
     * Facade method for delete_cron_lock
     * Delegates to utilityHelper
     */
    public function delete_cron_lock(...$args) {
        return $this->utilityHelper->delete_cron_lock(...$args);
    }

    /**
     * Facade method for track_query_performance
     * Delegates to cacheHelper
     */
    public function track_query_performance(...$args) {
        return $this->cacheHelper->track_query_performance(...$args);
    }

    /**
     * Facade method for track_batch_metrics
     * Delegates to cacheHelper
     */
    public function track_batch_metrics(...$args) {
        return $this->cacheHelper->track_batch_metrics(...$args);
    }

    /**
     * Facade method for increment_completed_tasks
     * Delegates to cacheHelper
     */
    public function increment_completed_tasks(...$args) {
        return $this->cacheHelper->increment_completed_tasks(...$args);
    }

    /**
     * Facade method for invalidate_task_count_cache
     * Delegates to cacheHelper
     */
    public function invalidate_task_count_cache(...$args) {
        return $this->cacheHelper->invalidate_task_count_cache(...$args);
    }

    /**
     * Facade method for warm_caches
     * Delegates to cacheHelper
     */
    public function warm_caches(...$args) {
        return $this->cacheHelper->warm_caches(...$args);
    }

    /**
     * Facade method for topological_sort_tasks
     * Delegates to schedulerHelper
     */
    public function topological_sort_tasks(...$args) {
        return $this->schedulerHelper->topological_sort_tasks(...$args);
    }

    /**
     * Facade method for calculate_task_priority
     * Delegates to schedulerHelper
     */
    public function calculate_task_priority(...$args) {
        return $this->schedulerHelper->calculate_task_priority(...$args);
    }

    /**
     * Facade method for store_task_metadata
     * Delegates to schedulerHelper
     */
    public function store_task_metadata(...$args) {
        return $this->schedulerHelper->store_task_metadata(...$args);
    }

    /**
     * Facade method for is_task_enqueued
     * Delegates to schedulerHelper
     */
    public function is_task_enqueued(...$args) {
        return $this->schedulerHelper->is_task_enqueued(...$args);
    }

    /**
     * Facade method for get_cron_hook
     * Delegates to schedulerHelper
     */
    public function get_cron_hook(...$args) {
        return $this->schedulerHelper->get_cron_hook(...$args);
    }

    /**
     * Facade method for validate_task_structure
     * Delegates to validationHelper
     */
    public function validate_task_structure(...$args) {
        return $this->validationHelper->validate_task_structure(...$args);
    }

    /**
     * Facade method for safely_unserialize_task
     * Delegates to validationHelper
     */
    public function safely_unserialize_task(...$args) {
        return $this->validationHelper->safely_unserialize_task(...$args);
    }

    /**
     * Facade method for is_valid_http_url
     * Delegates to validationHelper
     */
    public function is_valid_http_url(...$args) {
        return $this->validationHelper->is_valid_http_url(...$args);
    }

    /**
     * Facade method for is_js_task_with_delay
     * Delegates to validationHelper
     */
    public function is_js_task_with_delay(...$args) {
        return $this->validationHelper->is_js_task_with_delay(...$args);
    }

    /**
     * Facade method for add_five_minute_cron_schedule
     * Delegates to cronHelper
     */
    public function add_five_minute_cron_schedule(...$args) {
        return $this->cronHelper->add_five_minute_cron_schedule(...$args);
    }

    /**
     * Facade method for reset_stuck_tasks
     * Delegates to cronHelper
     */
    public function reset_stuck_tasks(...$args) {
        return $this->cronHelper->reset_stuck_tasks(...$args);
    }



    public static function __callStatic(string $name, array $arguments)
    {
        // Static methods that are in \LHA\TaskHelpers\TasksStaticHelper
        $staticMethods = [
            'update_task_fields_static',
            'add_custom_cron_schedules',
        ];

        if (in_array($name, $staticMethods)) {
            // Route directly to \LHA\TaskHelpers\TasksStaticHelper
            return call_user_func_array(['\LHA\TaskHelpers\TasksStaticHelper', $name], $arguments);
        }

        // For other static calls, try to get an instance from container
        global $lha_container;

        if (isset($lha_container) && $lha_container instanceof \LHA\ServiceContainer) {
            try {
                $instance = $lha_container->get(\LHA\Interfaces\TaskQueueInterface::class);
                if ($instance !== null) {
                    return $instance->$name(...$arguments);
                }
            } catch (\Exception $e) {
                // Fall through
            }
        }

        throw new \BadMethodCallException("Static method $name does not exist");
    }

}
