<?php

namespace LHA;

use LHA\Interfaces\DatabaseInterface;
use LHA\DatabaseHelpers\DatabaseTableHelper;
use LHA\DatabaseHelpers\DatabaseTransactionHelper;
use LHA\DatabaseHelpers\DatabaseQueryHelper;
use LHA\DatabaseHelpers\DatabaseMappingHelper;
use LHA\DatabaseHelpers\DatabaseAssetHelper;
use LHA\DatabaseHelpers\DatabaseStatsHelper;
use LHA\DatabaseHelpers\DatabaseTaskHelper;
use LHA\DatabaseHelpers\DatabaseOptionHelper;
use LHA\DatabaseHelpers\DatabaseProgressHelper;
use LHA\DatabaseHelpers\DatabaseCacheHelper;
use LHA\DatabaseHelpers\DatabaseIndexHelper;

/**
 * Database facade - delegates to specialized helper classes
 *
 * @package LHA
 * @since   1.0.0
 */
class Database implements \LHA\Interfaces\DatabaseInterface
{
    // ============================================================
    // Stub methods to satisfy DatabaseInterface
    // These delegate to __call() for actual implementation
    // ============================================================

    public function update_mapping_entry(string $url, string $type, string $hashed_filename, string $status, array $dependencies = [], array $order_details = []): bool {
        return $this->__call('update_mapping_entry', func_get_args());
    }

    public function update_mapping_status(...$args) {
        return $this->__call('update_mapping_status', $args);
    }

    public function insert_mapping_entry(...$args) {
        return $this->__call('insert_mapping_entry', $args);
    }

    public function get_asset_by_id(...$args) {
        return $this->__call('get_asset_by_id', $args);
    }

    public function get_asset_by_original_url(...$args) {
        return $this->__call('get_asset_by_original_url', $args);
    }

    public function cleanup_stale_mappings(...$args) {
        return $this->__call('cleanup_stale_mappings', $args);
    }

    public function get_table_definitions(...$args) {
        return $this->__call('get_table_definitions', $args);
    }

    public function get_table_name(...$args) {
        return $this->__call('get_table_name', $args);
    }

    public function table_exists(...$args) {
        return $this->__call('table_exists', $args);
    }

    public function install(...$args) {
        return $this->__call('install', $args);
    }

    public function create_table(...$args) {
        return $this->__call('create_table', $args);
    }

    public function mapping_entry_exists_by_id(...$args) {
        return $this->__call('mapping_entry_exists_by_id', $args);
    }

    public function add_asset_entry(...$args) {
        return $this->__call('add_asset_entry', $args);
    }

    public function delete_asset_entry(...$args) {
        return $this->__call('delete_asset_entry', $args);
    }

    public function update_asset_dependencies(...$args) {
        return $this->__call('update_asset_dependencies', $args);
    }

    public function batch_get_assets(...$args) {
        return $this->__call('batch_get_assets', $args);
    }

    public function batch_insert_assets(...$args) {
        return $this->__call('batch_insert_assets', $args);
    }

    public function get_query_stats(...$args) {
        return $this->__call('get_query_stats', $args);
    }

    public function get_performance_report(...$args) {
        return $this->__call('get_performance_report', $args);
    }

    public function execute_in_transaction(...$args) {
        return $this->__call('execute_in_transaction', $args);
    }

    public function stream_assets(...$args) {
        return $this->__call('stream_assets', $args);
    }

    public function update_mapping_status_by_url(...$args) {
        return $this->__call('update_mapping_status_by_url', $args);
    }

    public function batch_update_mapping_status(...$args) {
        return $this->__call('batch_update_mapping_status', $args);
    }

    public function batch_update_mapping_status_by_url(...$args) {
        return $this->__call('batch_update_mapping_status_by_url', $args);
    }

    public function get_dashboard_stats(...$args) {
        return $this->__call('get_dashboard_stats', $args);
    }

    public function get_assets_with_tasks(...$args) {
        return $this->__call('get_assets_with_tasks', $args);
    }

    public function batch_delete_assets(...$args) {
        return $this->__call('batch_delete_assets', $args);
    }

    public function verify_tasks_table(...$args) {
        return $this->__call('verify_tasks_table', $args);
    }

    public function remediate_plugin_tables(...$args) {
        return $this->__call('remediate_plugin_tables', $args);
    }

    public function validate_order_table(...$args) {
        return $this->__call('validate_order_table', $args);
    }

    public function verify_plugin_tables(...$args) {
        return $this->__call('verify_plugin_tables', $args);
    }

    public function rollback_transactions(...$args) {
        return $this->__call('rollback_transactions', $args);
    }

    public function generate_create_table_sql(...$args) {
        return $this->__call('generate_create_table_sql', $args);
    }

    public function create_retry_table(...$args) {
        return $this->__call('create_retry_table', $args);
    }

    public function create_tasks_table(...$args) {
        return $this->__call('create_tasks_table', $args);
    }

    public function create_mapping_table(...$args) {
        return $this->__call('create_mapping_table', $args);
    }

    public function create_order_table(...$args) {
        return $this->__call('create_order_table', $args);
    }

    public function create_locks_table(...$args) {
        return $this->__call('create_locks_table', $args);
    }

    public function verify_single_table(...$args) {
        return $this->__call('verify_single_table', $args);
    }

    public function commit_transaction(...$args) {
        return $this->__call('commit_transaction', $args);
    }

    public function start_transaction(...$args) {
        return $this->__call('start_transaction', $args);
    }

    public function is_transaction_active(...$args) {
        return $this->__call('is_transaction_active', $args);
    }

    public function rollback_transactions_on_shutdown(...$args) {
        return $this->__call('rollback_transactions_on_shutdown', $args);
    }

    public function get_table_names(...$args) {
        return $this->__call('get_table_names', $args);
    }

    public function get_asset_count(...$args) {
        return $this->__call('get_asset_count', $args);
    }

    public function get_cached_count(...$args) {
        return $this->__call('get_cached_count', $args);
    }

    public function get_count_by_status(...$args) {
        return $this->__call('get_count_by_status', $args);
    }

    public function has_pending_tasks(...$args) {
        return $this->__call('has_pending_tasks', $args);
    }

    public function get_pending_tasks_count(...$args) {
        return $this->__call('get_pending_tasks_count', $args);
    }

    public function batch_update_asset_status(...$args) {
        return $this->__call('batch_update_asset_status', $args);
    }

    public function get_assets_by_statuses(...$args) {
        return $this->__call('get_assets_by_statuses', $args);
    }

    public function get_assets_by_type(...$args) {
        return $this->__call('get_assets_by_type', $args);
    }

    public function get_failed_assets_with_retry_info(...$args) {
        return $this->__call('get_failed_assets_with_retry_info', $args);
    }

    public function get_assets_by_status_optimized(...$args) {
        return $this->__call('get_assets_by_status_optimized', $args);
    }

    public function create_plugin_tables(...$args) {
        return $this->__call('create_plugin_tables', $args);
    }

    public function verify_mapping_table(...$args) {
        return $this->__call('verify_mapping_table', $args);
    }

    public function get_mapping(...$args) {
        return $this->__call('get_mapping', $args);
    }

    public function get_assets_by_ids(...$args) {
        return $this->__call('get_assets_by_ids', $args);
    }

    public function get_assets_by_urls(...$args) {
        return $this->__call('get_assets_by_urls', $args);
    }

    public function batch_update_mapping_statuses(...$args) {
        return $this->__call('batch_update_mapping_statuses', $args);
    }

    public function get_assets_with_order(...$args) {
        return $this->__call('get_assets_with_order', $args);
    }

    public function get_table_definitions_cached(...$args) {
        return $this->__call('get_table_definitions_cached', $args);
    }

    public function clear_table_definitions_cache(...$args) {
        return $this->__call('clear_table_definitions_cache', $args);
    }

    public function batch_insert_mappings(...$args) {
        return $this->__call('batch_insert_mappings', $args);
    }

    public function get_pending_tasks_with_assets(...$args) {
        return $this->__call('get_pending_tasks_with_assets', $args);
    }

    public function insert_mapping(...$args) {
        return $this->__call('insert_mapping', $args);
    }

    public function delete_mapping(...$args) {
        return $this->__call('delete_mapping', $args);
    }

    public function get_pending_tasks(...$args) {
        return $this->__call('get_pending_tasks', $args);
    }

    public function begin_transaction(...$args) {
        return $this->__call('begin_transaction', $args);
    }

    public function rollback_transaction(...$args) {
        return $this->__call('rollback_transaction', $args);
    }

    public function get_default_main_options(...$args) {
        return $this->__call('get_default_main_options', $args);
    }

    public function get_default_tool_options(...$args) {
        return $this->__call('get_default_tool_options', $args);
    }

    public function initialize_plugin_options(...$args) {
        return $this->__call('initialize_plugin_options', $args);
    }

    public function update_plugin_version(...$args) {
        return $this->__call('update_plugin_version', $args);
    }

    public function update_db_version(...$args) {
        return $this->__call('update_db_version', $args);
    }

    public function get_plugin_version(...$args) {
        return $this->__call('get_plugin_version', $args);
    }

    public function get_db_version(...$args) {
        return $this->__call('get_db_version', $args);
    }

    public function delete_all_plugin_options(...$args) {
        return $this->__call('delete_all_plugin_options', $args);
    }

    public function check_plugin_options(...$args) {
        return $this->__call('check_plugin_options', $args);
    }

    public function repair_plugin_options(...$args) {
        return $this->__call('repair_plugin_options', $args);
    }

    public function get_cache_expiration(...$args) {
        return $this->__call('get_cache_expiration', $args);
    }

    public function set_cache_expiration(...$args) {
        return $this->__call('set_cache_expiration', $args);
    }

    public function get_progress(...$args) {
        return $this->__call('get_progress', $args);
    }

    public function update_progress(...$args) {
        return $this->__call('update_progress', $args);
    }

    public function reset_progress(...$args) {
        return $this->__call('reset_progress', $args);
    }

    public function increment_completed_tasks(...$args) {
        return $this->__call('increment_completed_tasks', $args);
    }

    public function get_cleanup_config(...$args) {
        return $this->__call('get_cleanup_config', $args);
    }

    public function set_cleanup_config(...$args) {
        return $this->__call('set_cleanup_config', $args);
    }

    public function get_all_option_names(...$args) {
        return $this->__call('get_all_option_names', $args);
    }

    public function get_assets_with_tasks_keyset(...$args) {
        return $this->__call('get_assets_with_tasks_keyset', $args);
    }

    public function warm_asset_cache(...$args) {
        return $this->__call('warm_asset_cache', $args);
    }

    public function get_assets_lightweight(...$args) {
        return $this->__call('get_assets_lightweight', $args);
    }

    public function is_connection_healthy(...$args) {
        return $this->__call('is_connection_healthy', $args);
    }

    public function reconnect_if_needed(...$args) {
        return $this->__call('reconnect_if_needed', $args);
    }

    public function get_performance_warnings(...$args) {
        return $this->__call('get_performance_warnings', $args);
    }

    public function profile_operation(...$args) {
        return $this->__call('profile_operation', $args);
    }

    public function get_cache_stats(...$args) {
        return $this->__call('get_cache_stats', $args);
    }

    public function clear_all_caches(...$args) {
        return $this->__call('clear_all_caches', $args);
    }

    public function add_ajax_performance_indexes(...$args) {
        return $this->__call('add_ajax_performance_indexes', $args);
    }

    public function get_ajax_performance_indexes(...$args) {
        return $this->__call('get_ajax_performance_indexes', $args);
    }

    public function get_mapping_by_url(...$args) {
        return $this->__call('get_mapping_by_url', $args);
    }

    public function get_mapping_id_by_url(...$args) {
        return $this->__call('get_mapping_id_by_url', $args);
    }

    public function optimize_database_tables(...$args) {
        return $this->__call('optimize_database_tables', $args);
    }


public const TABLE_MAPPINGS = 'lha_mappings';
    public const TABLE_TASKS = 'lha_tasks';
    public const TABLE_ORDER = 'lha_order';
    public const TABLE_RETRY_QUEUE = 'lha_retry_queue';
    public const TABLE_RETRY_HISTORY = 'lha_retry_history';
    public const TABLE_RETRY_DLQ = 'lha_retry_dlq';
    public const TABLE_LOCKS = 'lha_locks';

    /**
     * Common column name constants to avoid duplication
     */
    public const COLUMN_ID = 'id';
    public const COLUMN_ORIGINAL_URL = 'original_url';
    public const COLUMN_TYPE = 'type';
    public const COLUMN_STATUS = 'status';
    public const COLUMN_HANDLE = 'handle';
    public const COLUMN_HASHED_FILENAME = 'hashed_filename';
    public const COLUMN_LOCAL_URL = 'local_url';
    public const COLUMN_DEPENDENCIES = 'dependencies';
    public const COLUMN_LAST_ERROR = 'last_error';
    public const COLUMN_METADATA = 'metadata';
    public const COLUMN_CREATED_AT = 'created_at';
    public const COLUMN_UPDATED_AT = 'updated_at';
    public const COLUMN_ASSET_ID = 'asset_id';
    public const COLUMN_POST_ID = 'post_id';
    public const COLUMN_TASK_ID = 'task_id';
    public const COLUMN_RETRY_ID = 'retry_id';
    public const COLUMN_LOCK_ID = 'lock_id';
    public const COLUMN_LOCK_KEY = 'lock_key';
    public const COLUMN_LOCK_VALUE = 'lock_value';
    public const COLUMN_LOCK_TIME = 'lock_time';
    public const COLUMN_EXPIRES_AT = 'expires_at';
    public const COLUMN_PRIORITY = 'priority';
    public const COLUMN_ATTEMPTS = 'attempts';
    public const COLUMN_LAST_ATTEMPT_AT = 'last_attempt_at';
    public const COLUMN_NEXT_ATTEMPT_AT = 'next_attempt_at';
    public const COLUMN_FORCE_REFRESH = 'force_refresh';
    public const COLUMN_CACHE_EXPIRATION = 'cache_expiration';
    public const COLUMN_RELATED_ID = 'related_id';
    public const COLUMN_RELATED_TYPE = 'related_type';
    public const COLUMN_RETRY_REASON = 'retry_reason';
    public const COLUMN_RETRY_COUNT = 'retry_count';
    public const COLUMN_NEXT_RETRY_AT = 'next_retry_at';
    public const COLUMN_LAST_ERROR_DATA = 'last_error_data';
    public const COLUMN_ASSET_ORDER = 'asset_order';
    public const COLUMN_DELAY_JS = 'delay_js';
    public const COLUMN_TIMEOUT_JS = 'timeout_js';

    /**
     * Cache key prefix constants to ensure consistent naming
     */
    public const CACHE_KEY_PREFIX_ASSET = 'lha_asset_';
    public const CACHE_KEY_PREFIX_MAPPING_ID = 'lha_mapping_id_';
    public const CACHE_KEY_PREFIX_ASSET_COUNT = 'lha_asset_count_';
    public const CACHE_KEY_PREFIX_DASHBOARD_STATS = 'lha_dashboard_stats';
    public const CACHE_KEY_PREFIX_PENDING_TASKS_COUNT = 'lha_pending_tasks_count';
    public const CACHE_KEY_PREFIX_HAS_PENDING_TASKS = 'lha_has_pending_tasks';
    public const CACHE_KEY_PREFIX_COUNT = 'lha_count_';
    public const CACHE_KEY_PREFIX_STATUS_COUNT = 'lha_status_count_';
    public const CACHE_KEY_PREFIX_ASSETS_BY_URLS = 'lha_assets_by_urls_';
    public const CACHE_KEY_PREFIX_ASSETS_LIGHTWEIGHT = 'lha_assets_lightweight_';
    public const CACHE_KEY_PREFIX_FAILED_ASSETS_RETRY_INFO = 'lha_failed_assets_retry_info_';
    public const CACHE_KEY_PREFIX_ASSETS_WITH_TASKS = 'lha_assets_with_tasks_';
    public const CACHE_KEY_PREFIX_ASSETS_BY_STATUSES = 'lha_assets_by_statuses_';
    public const CACHE_KEY_PREFIX_ASSETS_BY_TYPE = 'lha_assets_by_type_';
    public const CACHE_KEY_PREFIX_QUERY_STATS = 'lha_query_stats';
    public const CACHE_KEY_PREFIX_PERFORMANCE_REPORT = 'lha_performance_report';
    public const CACHE_KEY_PREFIX_TABLE_DEFINITIONS = 'lha_table_definitions_';
    public const CACHE_KEY_PREFIX_CALCULATING = '_calculating';

    /**
     * Allowed status codes for mapping entries.
     * Used for validation in update methods.
     * @var string[]
     */
    

    /**
     * WordPress database object
     * @var \wpdb
     */
    private \wpdb $wpdb;

    /**
     * Logger instance
     * @var \LHA\Interfaces\LoggerInterface|null
     */
    private ?\LHA\Interfaces\LoggerInterface $logger;

    /**
     * File lock instance
     * @var \LHA\Interfaces\LockInterface|null
     */
    private ?\LHA\Interfaces\LockInterface $lock;

    /**
     * Asset validator instance for URL validation
     * @var \LHA\Interfaces\AssetValidatorInterface|null
     */
    private ?\LHA\Interfaces\AssetValidatorInterface $assetValidator;

    /**
     * Normalize instance for URL normalization
     * @var \LHA\Interfaces\NormalizeInterface|null
     */
    private ?\LHA\Interfaces\NormalizeInterface $normalize = null;

    /**
     * UrlProcessor instance for URL processing operations
     * @var \LHA\Interfaces\UrlProcessorInterface|null
     */
    private ?\LHA\Interfaces\UrlProcessorInterface $urlProcessor = null;

    /**
     * Transaction active flag
     * @var bool
     */
    private bool $transaction_active = false;

    /**
     * Query cache
     * @var array
     */
    private array $query_cache = [];

    /**
     * Cache version
     * @var int|null
     */
    private ?int $cache_version = null;

    /**
     * Slow query threshold
     * @var float
     */
    private float $slow_query_threshold = 1.0;

    /**
     * Query statistics
     * @var array
     */
    private array $query_stats = [
        'count' => 0,
        'time' => 0.0,
        'slow_queries' => []
    ];

    /**
     * Table definitions cache
     * @var array|null
     */
    private ?array $table_definitions_cache = null;

    /**
     * Validated tables cache
     * @var array|null
     */
    private ?array $validated_tables = null;

    /**
     * Helper instances
     */
    private ?DatabaseTableHelper $tableHelper = null;
    private ?DatabaseTransactionHelper $transactionHelper = null;
    private ?DatabaseQueryHelper $queryHelper = null;
    private ?DatabaseMappingHelper $mappingHelper = null;
    private ?DatabaseAssetHelper $assetHelper = null;
    private ?DatabaseStatsHelper $statsHelper = null;
    private ?DatabaseTaskHelper $taskHelper = null;
    private ?DatabaseOptionHelper $optionHelper = null;
    private ?DatabaseProgressHelper $progressHelper = null;
    private ?DatabaseCacheHelper $cacheHelper = null;
    private ?DatabaseIndexHelper $indexHelper = null;

    /**
     * Constructor
     */
    public function __construct(
        \wpdb $wpdb,
        ?\LHA\Interfaces\LoggerInterface $logger = null,
        ?\LHA\Interfaces\LockInterface $lock = null,
        ?\LHA\Interfaces\AssetValidatorInterface $assetValidator = null,
        ?\LHA\Interfaces\NormalizeInterface $normalize = null,
        ?\LHA\Interfaces\UrlProcessorInterface $urlProcessor = null
    ) {
        $this->wpdb = $wpdb;
        $this->logger = $logger;
        $this->lock = $lock;
        $this->assetValidator = $assetValidator;
        $this->normalize = $normalize;
        $this->urlProcessor = $urlProcessor;
    }

    /**
     * Get helper instance (lazy initialization)
     */
    private function getTableHelper(): DatabaseTableHelper
    {
        if ($this->tableHelper === null) {
            $this->tableHelper = new DatabaseTableHelper(
                $this->wpdb,
                $this->logger,
                $this->lock,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->tableHelper;
    }

    private function getTransactionHelper(): DatabaseTransactionHelper
    {
        if ($this->transactionHelper === null) {
            $this->transactionHelper = new DatabaseTransactionHelper(
                $this->wpdb,
                $this->logger,
                $this->lock,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->transactionHelper;
    }

    private function getQueryHelper(): DatabaseQueryHelper
    {
        if ($this->queryHelper === null) {
            $this->queryHelper = new DatabaseQueryHelper(
                $this->wpdb,
                $this->logger,
                $this->lock,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->queryHelper;
    }

    private function getMappingHelper(): DatabaseMappingHelper
    {
        if ($this->mappingHelper === null) {
            $this->mappingHelper = new DatabaseMappingHelper(
                $this->wpdb,
                $this->logger,
                $this->lock,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->mappingHelper;
    }

    private function getAssetHelper(): DatabaseAssetHelper
    {
        if ($this->assetHelper === null) {
            $this->assetHelper = new DatabaseAssetHelper(
                $this->wpdb,
                $this->logger,
                $this->lock,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->assetHelper;
    }

    private function getStatsHelper(): DatabaseStatsHelper
    {
        if ($this->statsHelper === null) {
            $this->statsHelper = new DatabaseStatsHelper(
                $this->wpdb,
                $this->logger,
                $this->lock,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->statsHelper;
    }

    private function getTaskHelper(): DatabaseTaskHelper
    {
        if ($this->taskHelper === null) {
            $this->taskHelper = new DatabaseTaskHelper(
                $this->wpdb,
                $this->logger,
                $this->lock,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->taskHelper;
    }

    private function getOptionHelper(): DatabaseOptionHelper
    {
        if ($this->optionHelper === null) {
            $this->optionHelper = new DatabaseOptionHelper(
                $this->wpdb,
                $this->logger,
                $this->lock,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->optionHelper;
    }

    private function getProgressHelper(): DatabaseProgressHelper
    {
        if ($this->progressHelper === null) {
            $this->progressHelper = new DatabaseProgressHelper(
                $this->wpdb,
                $this->logger,
                $this->lock,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->progressHelper;
    }

    private function getCacheHelper(): DatabaseCacheHelper
    {
        if ($this->cacheHelper === null) {
            $this->cacheHelper = new DatabaseCacheHelper(
                $this->wpdb,
                $this->logger,
                $this->lock,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->cacheHelper;
    }

    private function getIndexHelper(): DatabaseIndexHelper
    {
        if ($this->indexHelper === null) {
            $this->indexHelper = new DatabaseIndexHelper(
                $this->wpdb,
                $this->logger,
                $this->lock,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->indexHelper;
    }

    /**
     * Magic call method for delegating to helpers
     */
    public function __call(string $name, array $arguments)
    {
        static $methodMap = [
            'get_table_name' => 'DatabaseTableHelper',
            'table_exists' => 'DatabaseTableHelper',
            'get_table_names' => 'DatabaseTableHelper',
            'get_table_definitions' => 'DatabaseTableHelper',
            'get_table_definitions_cached' => 'DatabaseTableHelper',
            'clear_table_definitions_cache' => 'DatabaseTableHelper',
            'create_table' => 'DatabaseTableHelper',
            'install' => 'DatabaseTableHelper',
            'create_mapping_table' => 'DatabaseTableHelper',
            'create_tasks_table' => 'DatabaseTableHelper',
            'create_order_table' => 'DatabaseTableHelper',
            'create_locks_table' => 'DatabaseTableHelper',
            'create_retry_table' => 'DatabaseTableHelper',
            'verify_mapping_table' => 'DatabaseTableHelper',
            'verify_tasks_table' => 'DatabaseTableHelper',
            'verify_single_table' => 'DatabaseTableHelper',
            'verify_plugin_tables' => 'DatabaseTableHelper',
            'validate_order_table' => 'DatabaseTableHelper',
            'remediate_plugin_tables' => 'DatabaseTableHelper',
            'generate_create_table_sql' => 'DatabaseTableHelper',
            'create_plugin_tables' => 'DatabaseTableHelper',
            'start_transaction' => 'DatabaseTransactionHelper',
            'begin_transaction' => 'DatabaseTransactionHelper',
            'commit_transaction' => 'DatabaseTransactionHelper',
            'rollback_transaction' => 'DatabaseTransactionHelper',
            'execute_in_transaction' => 'DatabaseTransactionHelper',
            'is_transaction_active' => 'DatabaseTransactionHelper',
            'rollback_transactions' => 'DatabaseTransactionHelper',
            'batch_get_assets' => 'DatabaseQueryHelper',
            'get_asset_by_id' => 'DatabaseQueryHelper',
            'get_asset_by_original_url' => 'DatabaseQueryHelper',
            'get_assets_by_ids' => 'DatabaseQueryHelper',
            'get_assets_by_urls' => 'DatabaseQueryHelper',
            'get_assets_with_order' => 'DatabaseQueryHelper',
            'get_assets_lightweight' => 'DatabaseQueryHelper',
            'get_assets_by_type' => 'DatabaseQueryHelper',
            'get_assets_by_type_keyset' => 'DatabaseQueryHelper',
            'get_assets_by_statuses' => 'DatabaseQueryHelper',
            'get_assets_by_statuses_keyset' => 'DatabaseQueryHelper',
            'get_assets_by_status_optimized' => 'DatabaseQueryHelper',
            'get_assets_by_status_keyset' => 'DatabaseQueryHelper',
            'get_failed_assets_with_retry_info' => 'DatabaseQueryHelper',
            'stream_assets' => 'DatabaseQueryHelper',
            'profile_operation' => 'DatabaseQueryHelper',
            'insert_mapping_entry' => 'DatabaseMappingHelper',
            'update_mapping_entry' => 'DatabaseMappingHelper',
            'update_mapping_status' => 'DatabaseMappingHelper',
            'update_mapping_status_by_url' => 'DatabaseMappingHelper',
            'batch_update_mapping_status' => 'DatabaseMappingHelper',
            'batch_update_mapping_status_by_url' => 'DatabaseMappingHelper',
            'batch_update_mapping_statuses' => 'DatabaseMappingHelper',
            'get_mapping_id_by_url' => 'DatabaseMappingHelper',
            'get_mapping' => 'DatabaseMappingHelper',
            'get_mapping_by_url' => 'DatabaseMappingHelper',
            'mapping_entry_exists_by_id' => 'DatabaseMappingHelper',
            'insert_mapping' => 'DatabaseMappingHelper',
            'delete_mapping' => 'DatabaseMappingHelper',
            'batch_insert_mappings' => 'DatabaseMappingHelper',
            'cleanup_stale_mappings' => 'DatabaseMappingHelper',
            'add_asset_entry' => 'DatabaseAssetHelper',
            'delete_asset_entry' => 'DatabaseAssetHelper',
            'batch_insert_assets' => 'DatabaseAssetHelper',
            'batch_delete_assets' => 'DatabaseAssetHelper',
            'update_asset_dependencies' => 'DatabaseAssetHelper',
            'batch_update_asset_status' => 'DatabaseAssetHelper',
            'get_asset_count' => 'DatabaseStatsHelper',
            'get_cached_count' => 'DatabaseStatsHelper',
            'get_count_by_status' => 'DatabaseStatsHelper',
            'get_dashboard_stats' => 'DatabaseStatsHelper',
            'get_pending_tasks_count' => 'DatabaseStatsHelper',
            'has_pending_tasks' => 'DatabaseStatsHelper',
            'get_performance_report' => 'DatabaseStatsHelper',
            'get_query_stats' => 'DatabaseStatsHelper',
            'get_performance_warnings' => 'DatabaseStatsHelper',
            'get_cache_stats' => 'DatabaseStatsHelper',
            'reset_query_stats' => 'DatabaseStatsHelper',
            'get_slow_query_threshold' => 'DatabaseStatsHelper',
            'set_slow_query_threshold' => 'DatabaseStatsHelper',
            'get_pending_tasks' => 'DatabaseTaskHelper',
            'get_pending_tasks_with_assets' => 'DatabaseTaskHelper',
            'get_assets_with_tasks' => 'DatabaseTaskHelper',
            'get_assets_with_tasks_keyset' => 'DatabaseTaskHelper',
            'warm_asset_cache' => 'DatabaseTaskHelper',
            'increment_completed_tasks' => 'DatabaseTaskHelper',
            'get_default_main_options' => 'DatabaseOptionHelper',
            'get_default_tool_options' => 'DatabaseOptionHelper',
            'initialize_plugin_options' => 'DatabaseOptionHelper',
            'update_plugin_version' => 'DatabaseOptionHelper',
            'update_db_version' => 'DatabaseOptionHelper',
            'get_plugin_version' => 'DatabaseOptionHelper',
            'get_db_version' => 'DatabaseOptionHelper',
            'delete_all_plugin_options' => 'DatabaseOptionHelper',
            'check_plugin_options' => 'DatabaseOptionHelper',
            'repair_plugin_options' => 'DatabaseOptionHelper',
            'get_all_option_names' => 'DatabaseOptionHelper',
            'get_cache_expiration' => 'DatabaseOptionHelper',
            'set_cache_expiration' => 'DatabaseOptionHelper',
            'get_progress' => 'DatabaseProgressHelper',
            'update_progress' => 'DatabaseProgressHelper',
            'reset_progress' => 'DatabaseProgressHelper',
            'get_cleanup_config' => 'DatabaseProgressHelper',
            'set_cleanup_config' => 'DatabaseProgressHelper',
            'clear_all_caches' => 'DatabaseCacheHelper',
            'add_ajax_performance_indexes' => 'DatabaseIndexHelper',
            'get_ajax_performance_indexes' => 'DatabaseIndexHelper',
            'optimize_database_tables' => 'DatabaseIndexHelper'
        ];

        if (!isset($methodMap[$name])) {
            throw new \BadMethodCallException("Method $name does not exist in " . self::class);
        }

        $helperClass = $methodMap[$name];
        $getterMethod = "get" . str_replace("Database", "", $helperClass);

        if (!method_exists($this, $getterMethod)) {
            throw new \BadMethodCallException("Helper getter $getterMethod not found");
        }

        $helper = $this->$getterMethod();
        return $helper->$name(...$arguments);
    }

    /**
     * Static method support (for backwards compatibility)
     */
        public static function __callStatic(string $name, array $arguments)
    {
        // Static methods that are in DatabaseStaticHelper
        $staticMethods = [
            'remediate_order_table',
            'remediate_mapping_table',
            'static_add_asset_entry',
            'static_delete_asset_entry',
            'get_last_error',
            'handle_manual_validation',
            'handle_manual_remediation',
            'touch_mapping_entry',
            'update_mapping_field_by_url',
            'static_update_asset_dependencies',
            'get_mapping_table_name',
            'proxy_update_asset_entry',
        ];

        if (in_array($name, $staticMethods)) {
            // Route directly to DatabaseStaticHelper
            $helperClass = '\LHA\DatabaseHelpers\DatabaseStaticHelper';
            return call_user_func_array([$helperClass, $name], $arguments);
        }

        // For other static calls, try to get an instance
        global $lha_container;

        if (isset($lha_container) && $lha_container instanceof \LHA\ServiceContainer) {
            try {
                $db = $lha_container->get(\LHA\Interfaces\DatabaseInterface::class);
                if ($db !== null) {
                    return $db->$name(...$arguments);
                }
            } catch (\Exception $e) {
                // Fall through
            }
        }

        // Fallback: create a new instance
        global $wpdb;
        $db = new self($wpdb);
        return $db->$name(...$arguments);
    }

    /**
     * Static method: get_mapping_table_name
     */
    public static function get_table_name_static(string $table_constant): string
    {
        global $wpdb;

        $tableTypes = [
            'mappings' => self::TABLE_MAPPINGS,
            'tasks' => self::TABLE_TASKS,
            'order' => self::TABLE_ORDER,
            'retry_queue' => self::TABLE_RETRY_QUEUE,
            'retry_history' => self::TABLE_RETRY_HISTORY,
            'retry_dlq' => self::TABLE_RETRY_DLQ,
            'locks' => self::TABLE_LOCKS,
        ];

        if (!isset($tableTypes[$table_constant])) {
            return '';
        }

        return $wpdb->prefix . $tableTypes[$table_constant];
    }
}
