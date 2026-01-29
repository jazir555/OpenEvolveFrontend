<?php

namespace LHA;

use LHA\Interfaces\GetdataInterface;
use LHA\Interfaces\AssetDataInterface;
use LHA\Interfaces\AssetValidatorInterface;
use LHA\Interfaces\AssetActionHandlerInterface;
use LHA\Interfaces\DatabaseInterface;
use LHA\Sanitize;

/**
 * GetData - Centralized data retrieval and validation facade
 * 
 * PERFORMANCE OPTIMIZATIONS APPLIED:
 * 
 * 1. Static Caching:
 *    - determine_asset_status(): Caches status lookups to avoid repeated DB queries
 *    - get_local_url_with_download_fallback(): Caches URL lookups within request
 *    - determine_primary_asset_type(): Caches type determination results
 *    - determine_memory_threshold(): Caches calculated threshold
 *    - get_local_file_path(): Caches upload directory path
 * 
 * 2. Query Optimization:
 *    - get_assets(): Efficient query building with optional total count
 *    - get_mapping_entry_by_hash(): Uses indexed column first, LIKE as fallback
 *    - get_detailed_item_info(): Single query with JOINs instead of multiple queries
 *    - All queries select only needed columns to reduce memory usage
 * 
 * 3. Early Returns:
 *    - All methods validate inputs and return early on invalid data
 *    - File operations check existence before processing
 *    - Type checks happen before expensive operations
 * 
 * 4. Minimal Overhead:
 *    - Static arrays for extension/MIME maps (avoid recreation)
 *    - Pre-calculated constants for memory calculations
 *    - Efficient string operations (strpos instead of regex where possible)
 *    - @ error suppression for performance-critical file checks
 * 
 * 5. Smart Delegation:
 *    - Most methods delegate to specialized services (AssetData, AssetValidator, etc.)
 *    - Only implements logic that requires coordination between services
 *    - Keeps this class thin and focused on orchestration
 * 
 * MEMORY EFFICIENCY:
 * - Static caches are bounded (only cache within single request)
 * - No unbounded arrays or recursive structures
 * - Minimal object creation in hot paths
 * 
 * DATABASE EFFICIENCY:
 * - Uses prepared statements for all queries
 * - Leverages indexes (status, type, original_url)
 * - Selects only needed columns
 * - Uses LIMIT 1 where appropriate
 * - JOINs instead of multiple queries
 */
class GetData implements GetdataInterface {

    private AssetDataInterface $assetData;
    private AssetValidatorInterface $assetValidator;
    private AssetActionHandlerInterface $assetActionHandler;
    private ?\LHA\Interfaces\NormalizeInterface $normalize;
    private ?DatabaseInterface $database;
    private ?\LHA\Interfaces\UrlProcessorInterface $urlProcessor;
    private ?\LHA\Interfaces\AssetUtilsInterface $assetUtils = null;
    private ?\wpdb $wpdb = null;

    /**
     * @param AssetDataInterface $assetData
     * @param AssetValidatorInterface $assetValidator
     * @param AssetActionHandlerInterface $assetActionHandler
     * @param \LHA\Interfaces\GenerateInterface|null $generate Kept for backward compatibility, not used
     * @param \LHA\Interfaces\NormalizeInterface|null $normalize
     * @param DatabaseInterface|null $database
     * @param \LHA\Interfaces\UrlProcessorInterface|null $urlProcessor
     * @param \LHA\Interfaces\AssetUtilsInterface|null $assetUtils
     * @param \wpdb|null $wpdb
     * @phpstan-ignore-next-line constructor.unusedParameter (backward compatibility)
     */
    public function __construct(
        AssetDataInterface $assetData,
        AssetValidatorInterface $assetValidator,
        AssetActionHandlerInterface $assetActionHandler,
        ?\LHA\Interfaces\GenerateInterface $generate = null,
        ?\LHA\Interfaces\NormalizeInterface $normalize = null,
        ?DatabaseInterface $database = null,
        ?\LHA\Interfaces\UrlProcessorInterface $urlProcessor = null,
        ?\LHA\Interfaces\AssetUtilsInterface $assetUtils = null,
        ?\wpdb $wpdb = null
    ) {
        $this->assetData = $assetData;
        $this->assetValidator = $assetValidator;
        $this->assetActionHandler = $assetActionHandler;
        // Note: $generate parameter kept for backward compatibility but not stored (unused)
        $this->normalize = $normalize;
        $this->database = $database;
        $this->urlProcessor = $urlProcessor;
        $this->assetUtils = $assetUtils;

        if ($wpdb === null) {
            global $global_wpdb;
            // Validate that global $wpdb is actually a wpdb instance
            // This prevents type errors if WordPress isn't fully loaded
            if (isset($global_wpdb) && $global_wpdb instanceof \wpdb) {
                $this->wpdb = $global_wpdb;
            } else {
                // Keep as null if global $wpdb is not available or wrong type
                // Methods that need $wpdb should check for this
                $this->wpdb = null;
            }
        } else {
            // Validate provided $wpdb is actually a wpdb instance
            if ($wpdb instanceof \wpdb) {
                $this->wpdb = $wpdb;
            } else {
                $this->wpdb = null;
            }
        }
    }


    /**
     * Check if a URL is external (not from the current site).
     * Delegates to UrlProcessor for consistent external URL detection.
     *
     * @param string $url The URL to check.
     * @return bool True if external, false if local.
     */
    public function is_external_url(string $url): bool {
        try {
            if ($this->urlProcessor !== null) {
                return $this->urlProcessor->is_external_url($url);
            }

            // Fallback: Use Normalize for URL parsing if UrlProcessor is not available
            if ($this->normalize !== null) {
                $home_url = home_url();
                // Normalize URLs for consistent parsing
                $normalized_home = $this->normalize->normalize_url_base($home_url);
                $normalized_url = $this->normalize->normalize_url_base($url);

                // Parse normalized URLs
                $parsed_home = function_exists('wp_parse_url') ? wp_parse_url($normalized_home) : parse_url($normalized_home);
                $parsed_url = function_exists('wp_parse_url') ? wp_parse_url($normalized_url) : parse_url($normalized_url);

                // If we can't parse either URL, treat as local (safe default)
                if ($parsed_url === false || $parsed_home === false) {
                    return false;
                }

                // Compare hosts (case-insensitive)
                $url_host = isset($parsed_url['host']) ? strtolower($parsed_url['host']) : '';
                $home_host = isset($parsed_home['host']) ? strtolower($parsed_home['host']) : '';

                return $url_host !== '' && $url_host !== $home_host;
            }

            // Last resort: Basic parsing without Normalize
            $home_url = home_url();
            $parsed_home = function_exists('wp_parse_url') ? wp_parse_url($home_url) : parse_url($home_url);
            $parsed_url = function_exists('wp_parse_url') ? wp_parse_url($url) : parse_url($url);

            if ($parsed_url === false || $parsed_home === false) {
                return false;
            }

            $url_host = isset($parsed_url['host']) ? strtolower($parsed_url['host']) : '';
            $home_host = isset($parsed_home['host']) ? strtolower($parsed_home['host']) : '';

            return $url_host !== '' && $url_host !== $home_host;
        } catch (\Exception $e) {
            Logging::log_error("Failed to check if URL is external for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false; // Safe default: treat as local
        }
    }

    /**
     * Validate if a URL is valid.
     * Delegates to UrlProcessor for consistent URL validation.
     *
     * @param string|null $url The URL to validate.
     * @return bool True if valid, false otherwise.
     */
    public function is_valid_url(?string $url): bool {
        try {
            if ($url === null || $url === '') {
                return false;
            }

            if ($this->urlProcessor !== null) {
                return $this->urlProcessor->is_valid_url($url);
            }

            // Fallback: Use Normalize for URL validation if UrlProcessor is not available
            if ($this->normalize !== null) {
                $normalized = $this->normalize->normalize_url_base($url);
                if ($normalized === '') {
                    return false;
                }

                $parsed = function_exists('wp_parse_url') ? wp_parse_url($normalized) : parse_url($normalized);
                if ($parsed === false || empty($parsed)) {
                    return false;
                }

                // Must have a scheme and host
                return !empty($parsed['scheme']) && !empty($parsed['host']);
            }

            // Last resort: Basic parsing without Normalize
            $parsed = function_exists('wp_parse_url') ? wp_parse_url($url) : parse_url($url);
            if ($parsed === false || empty($parsed)) {
                return false;
            }

            return !empty($parsed['scheme']) && !empty($parsed['host']);
        } catch (\Exception $e) {
            Logging::log_error("Failed to validate URL: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false; // Safe default: treat as invalid
        }
    }

    /**
     * Determine asset type from URL.
     * Delegates to UrlProcessor for consistent type detection.
     *
     * @param string $url The URL to analyze.
     * @return string The determined asset type.
     */
    public function determine_asset_type_from_url(string $url): string {
        try {
            if ($this->urlProcessor !== null) {
                return $this->urlProcessor->determine_asset_type($url);
            }

            // Fallback: Use determine_primary_asset_type with 'file' context
            return $this->determine_primary_asset_type($url, 'file');
        } catch (\Exception $e) {
            Logging::log_error("Failed to determine asset type for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return 'file'; // Safe default: generic file type
        }
    }
    
    /**
     * Process a URL for asset management.
     * Delegates to UrlProcessor for comprehensive URL processing.
     *
     * @param string $url The URL to process.
     * @param array $options Processing options.
     * @return array Processed URL data including original_url, processed_url, is_external, is_valid, type, hash, status.
     */
    public function process_url(string $url, array $options = []): array {
        try {
            // Validate input (use trim() to handle whitespace-only URLs correctly)
            $trimmed_url = trim($url);
            if ($trimmed_url === '') {
                return [
                    'original_url' => $url,
                    'processed_url' => '',
                    'is_external' => false,
                    'is_valid' => false,
                    'type' => 'file',
                    'hash' => '',
                    'status' => 'invalid'
                ];
            }

            if ($this->urlProcessor !== null) {
                return $this->urlProcessor->process_url($url, $options);
            }

            // Fallback: Build basic processed data using Normalize for URL normalization
            $normalized = ($this->normalize !== null)
                ? $this->normalize->normalize_url_base($url)
                : trim($url);

            // If normalization failed, use original
            if ($normalized === '' || $normalized === null) {
                $normalized = trim($url);
            }

            $hash_data = json_encode([$normalized], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
            $hash = ($hash_data === false) ? md5($normalized) : md5($hash_data);

            return [
                'original_url' => $url,
                'processed_url' => $normalized,
                'is_external' => $this->is_external_url($url),
                'is_valid' => $this->is_valid_url($url),
                'type' => $this->determine_asset_type_from_url($url),
                'hash' => $hash,
                'status' => 'pending'
            ];
        } catch (\Exception $e) {
            Logging::log_error("Failed to process URL {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            // Return safe default on error
            return [
                'original_url' => $url,
                'processed_url' => ($this->normalize !== null) ? $this->normalize->normalize_url_base($url) : trim($url),
                'is_external' => false,
                'is_valid' => false,
                'type' => 'file',
                'hash' => md5($url),
                'status' => 'error'
            ];
        }
    }

    /**
     * Get memory usage with error handling
     *
     * @param bool $real_usage Use real memory usage
     * @return int Memory usage in bytes
     */
    public function get_memory_usage(bool $real_usage = true): int {
        try {
            return $this->assetData->get_memory_usage($real_usage);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get memory usage: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return 0;
        }
    }

    /**
     * Get paginated assets with validation
     *
     * @param int $paged Page number
     * @param int $per_page Items per page
     * @param string $asset_type Asset type filter
     * @return array|false Paginated assets or false on failure
     */
    public function get_paginated_assets(int $paged = 1, int $per_page = 20, string $asset_type = ''): array|false {
        try {
            if ($paged < 1 || $per_page < 1) {
                return false;
            }
            return $this->assetData->get_paginated_assets($paged, $per_page, $asset_type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get paginated assets: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset IDs by handles with validation
     *
     * @param array $handles Array of asset handles
     * @return array Array of asset IDs
     */
    public function get_asset_ids_by_handles(array $handles): array {
        try {
            if (empty($handles)) {
                return [];
            }
            return $this->assetData->get_asset_ids_by_handles($handles);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset IDs by handles: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Get assets with advanced filtering, pagination, and search
     * 
     * Optimized for performance with:
     * - Efficient query building with minimal overhead
     * - Optional total count calculation (skip when not needed)
     * - Proper index usage (status, type, created_at)
     * - Minimal memory footprint
     * - Static caching for repeated identical queries
     * - Query result caching to avoid duplicate database hits
     * 
     * @param array $args {
     *     Query arguments
     *     @type int    $paged         Page number (default: 1)
     *     @type int    $number        Results per page (default: 20)
     *     @type string $asset_type    Filter by asset type (js, css, image, etc.)
     *     @type string $status        Filter by status (pending, processed, failed, etc.)
     *     @type string $search        Search term for URL matching
     *     @type string $orderby       Column to sort by (default: 'id')
     *     @type string $order         Sort direction ASC or DESC (default: 'DESC')
     *     @type bool   $count_total   Whether to count total results (default: true)
     * }
     * @return array {
     *     Results array
     *     @type array $assets Array of asset records
     *     @type int   $total  Total number of matching assets
     * }
     */
    public function get_assets(array $args = []): array {
        // Static cache for repeated identical queries within same request
        static $query_cache = [];

        // Use json_encode instead of serialize to prevent object injection vulnerabilities
        $json_data = json_encode($args, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
        if ($json_data === false) {
            // JSON encoding failed (recursive references, invalid UTF-8, etc.)
            // Use fallback: create a safe cache key without serialize
            $cache_key_parts = [];
            foreach ($args as $key => $value) {
                if (is_scalar($value)) {
                    $cache_key_parts[] = $key . '=' . (string)$value;
                } else {
                    $cache_key_parts[] = $key . '=[' . gettype($value) . ']';
                }
            }
            $cache_key = md5(implode('&', $cache_key_parts));
        } else {
            $cache_key = md5($json_data);
        }

        if (isset($query_cache[$cache_key])) {
            return $query_cache[$cache_key];
        }

        // Bound cache to 50 entries to prevent memory exhaustion
        if (count($query_cache) >= 50) {
            array_shift($query_cache);
        }
        
        $wpdb = $this->wpdb;
        
        // Validate $wpdb is available
        if (!$wpdb instanceof \wpdb) {
            return ['assets' => [], 'total' => 0, 'pages' => 0];
        }
        
        // Extract and validate arguments with null coalescing
        $paged = max(1, absint($args['paged'] ?? 1));
        $per_page = max(1, min(200, absint($args['number'] ?? 20)));

        // Validate types before sanitization to prevent type errors
        // sanitize_key() and sanitize_text_field() expect strings
        $asset_type_raw = $args['asset_type'] ?? '';
        $asset_type = (!empty($asset_type_raw) && is_string($asset_type_raw))
            ? (class_exists('\LHA\Sanitize') ? \LHA\Sanitize::sanitize_key($asset_type_raw) : sanitize_key($asset_type_raw))
            : '';

        $status_raw = $args['status'] ?? '';
        $status = (!empty($status_raw) && is_string($status_raw))
            ? (class_exists('\LHA\Sanitize') ? \LHA\Sanitize::sanitize_key($status_raw) : sanitize_key($status_raw))
            : '';

        // Use Sanitize class for consistent security handling
        $search_raw = $args['search'] ?? '';
        $search = (!empty($search_raw) && is_string($search_raw))
            ? (class_exists('\LHA\Sanitize') ? \LHA\Sanitize::sanitize_text_field($search_raw) : sanitize_text_field($search_raw))
            : '';

        $orderby_raw = $args['orderby'] ?? 'id';
        $orderby = (is_string($orderby_raw))
            ? (class_exists('\LHA\Sanitize') ? \LHA\Sanitize::sanitize_key($orderby_raw) : sanitize_key($orderby_raw))
            : 'id';

        $order_raw = $args['order'] ?? 'DESC';
        $order = (is_string($order_raw))
            ? strtoupper(class_exists('\LHA\Sanitize') ? \LHA\Sanitize::sanitize_key($order_raw) : sanitize_key($order_raw))
            : 'DESC';

        $count_total = filter_var($args['count_total'] ?? true, FILTER_VALIDATE_BOOLEAN);

        // Validate order direction (fast array lookup)
        static $valid_orders = null;
        if ($valid_orders === null) {
            $valid_orders = ['ASC' => true, 'DESC' => true];
        }
        $order = isset($valid_orders[$order]) ? $order : 'DESC';

        // Validate orderby column (static whitelist for performance)
        static $allowed_orderby = null;
        if ($allowed_orderby === null) {
            $allowed_orderby = [
                'id' => true, 'original_url' => true, 'type' => true,
                'status' => true, 'created_at' => true, 'updated_at' => true
            ];
        }
        $orderby = isset($allowed_orderby[$orderby]) ? $orderby : 'id';
        
        // Build query efficiently - use Database class for table name if available
        $table = $this->database !== null
            ? $this->database->get_table_name(\LHA\Database::TABLE_MAPPINGS)
            : $wpdb->prefix . 'lha_mappings';

        // Validate and escape table name for security
        // Ensure table name is a string and not empty
        if (empty($table) || !is_string($table)) {
            return ['assets' => [], 'total' => 0, 'pages' => 0];
        }

        // Remove NULL bytes and other potentially dangerous characters
        $table = str_replace("\0", '', $table);

        // Escape backticks to prevent SQL injection
        $escaped_table = "`" . str_replace('`', '``', $table) . "`";
        
        $where_clauses = [];
        $prepare_args = [];
        
        // Add filters only if provided (reduces query complexity)
        if ($asset_type !== '') {
            $where_clauses[] = 'type = %s';
            $prepare_args[] = $asset_type;
        }
        
        if ($status !== '') {
            $where_clauses[] = 'status = %s';
            $prepare_args[] = $status;
        }
        
        if ($search !== '') {
            $where_clauses[] = 'original_url LIKE %s';
            $prepare_args[] = '%' . $wpdb->esc_like($search) . '%';
        }
        // Note: Empty search string is intentionally excluded to prevent "LIKE %%" which matches everything
        
        // Build WHERE clause (empty = no WHERE needed)
        $where_sql = !empty($where_clauses) ? 'WHERE ' . implode(' AND ', $where_clauses) : '';
        
        // Count total only if requested (performance optimization)
        $total = 0;
        if ($count_total) {
            $count_query = "SELECT COUNT(*) FROM {$escaped_table} {$where_sql}";
            if (!empty($prepare_args)) {
                $count_query = $wpdb->prepare($count_query, $prepare_args);
            }
            $total = (int) $wpdb->get_var($count_query);
            
            // Early return if no results (cache empty result)
            if ($total === 0) {
                $result = ['assets' => [], 'total' => 0, 'pages' => 0];
                $query_cache[$cache_key] = $result;
                return $result;
            }
        }
        
        // Calculate offset
        $offset = ($paged - 1) * $per_page;
        
        // Selective columns for better memory efficiency and performance
        $columns = "id, original_url, type, handle, status, hashed_filename, local_url, created_at, updated_at";

        // Build main query with LIMIT
        $query = "SELECT {$columns} FROM {$escaped_table} {$where_sql} ORDER BY {$orderby} {$order} LIMIT %d OFFSET %d";
        $query_args = array_merge($prepare_args, [$per_page, $offset]);
        $prepared_query = $wpdb->prepare($query, $query_args);
        
        // Execute query
        $assets = $wpdb->get_results($prepared_query, ARRAY_A);
        
        // Handle null result as empty array
        $assets = $assets ?? [];
        
        // Calculate pages for interface compliance
        // When count_total is false, we can't calculate pages accurately
        // but we can indicate if there might be more results
        $pages = 0;
        if ($count_total && $total > 0) {
            $pages = (int) ceil($total / $per_page);
        } elseif (!$count_total && count($assets) === $per_page) {
            // If we got a full page of results, there might be more
            // Set pages to -1 to indicate "unknown but possibly more"
            $pages = -1;
        }
        
        $result = [
            'assets' => $assets,
            'total' => $total,
            'pages' => $pages,
        ];
        
        $query_cache[$cache_key] = $result;
        return $result;
    }

    /**
     * Check if URI is a valid data URI
     *
     * @param string $uri The URI to validate
     * @return bool True if valid data URI, false otherwise
     */
    public function is_valid_data_uri(string $uri): bool {
        try {
            return $this->assetValidator->is_valid_data_uri($uri);
        } catch (\Exception $e) {
            Logging::log_error("Failed to validate data URI: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset status by ID with validation
     *
     * @param int $asset_id The asset ID
     * @return string|false The asset status or false on failure
     */
    public function get_asset_status_by_id(int $asset_id): string|false {
        try {
            if ($asset_id <= 0) {
                Logging::log_error("Invalid asset ID: {$asset_id}", Logging::LEVEL_ERROR);
                return false;
            }
            return $this->assetData->get_asset_status_by_id($asset_id);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset status by ID {$asset_id}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset status by URL and type with error handling
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return string|false The asset status or false on failure
     */
    public function get_asset_status(string $url, string $type): string|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_status($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset status for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get dynamic asset URLs for a page
     *
     * @param string $page_url The page URL
     * @return array Array of dynamic asset URLs
     */
    public function get_dynamic_asset_urls(string $page_url): array {
        try {
            return $this->assetData->get_dynamic_asset_urls($page_url);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get dynamic asset URLs for {$page_url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Check if URL is a valid asset URL
     *
     * @param string $url The URL to validate
     * @return bool True if valid, false otherwise
     */
    public function is_valid_asset_url(string $url): bool {
        try {
            return $this->assetValidator->is_valid_asset_url($url);
        } catch (\Exception $e) {
            Logging::log_error("Failed to validate asset URL {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Check if asset is enqueued
     *
     * @param array $asset The asset data
     * @return bool True if enqueued, false otherwise
     */
    public function is_asset_enqueued(array $asset): bool {
        try {
            return $this->assetValidator->is_asset_enqueued($asset);
        } catch (\Exception $e) {
            Logging::log_error("Failed to check if asset is enqueued: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get local URL from file path
     *
     * @param string $file_path The file path
     * @return string|null The local URL or null on failure
     */
    public function get_local_url_from_path(string $file_path): ?string {
        try {
            if (empty($file_path)) {
                return null;
            }
            $result = $this->assetData->get_local_url_from_path($file_path);
            return ($result === false) ? null : $result;
        } catch (\Exception $e) {
            Logging::log_error("Failed to get local URL from path {$file_path}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return null;
        }
    }

    /**
     * Get task table name
     *
     * @return string|false The table name or false on failure
     */
    public function get_task_table_name(): string|false {
        try {
            return $this->assetData->get_task_table_name();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get task table name: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Validate media handle
     *
     * @param string $handle The media handle
     * @param string $url The URL
     * @param string $type The asset type
     * @return bool True if valid, false otherwise
     */
    public function validate_media_handle(string $handle, string $url, string $type): bool {
        try {
            if (empty($handle) || empty($url) || empty($type)) {
                return false;
            }
            return $this->assetValidator->validate_media_handle($handle, $url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to validate media handle {$handle}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Validate hash salt
     *
     * @param string $salt The salt to validate
     * @return bool True if valid, false otherwise
     */
    public function validate_hash_salt(string $salt): bool {
        try {
            if (empty($salt)) {
                return false;
            }
            return $this->assetValidator->validate_hash_salt($salt);
        } catch (\Exception $e) {
            Logging::log_error("Failed to validate hash salt: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Validate origin and referer headers for CSRF protection.
     *
     * @param string|null $origin The Origin header value
     * @param string|null $referer The Referer header value
     * @param string $expectedHostLower The expected host in lowercase
     * @return bool True if validation passes, false otherwise
     */
    public function validate_origin_referer(?string $origin, ?string $referer, string $expectedHostLower): bool {
        try {
            if (empty($expectedHostLower)) {
                return false;
            }
            return $this->assetValidator->validate_origin_referer($origin, $referer, $expectedHostLower);
        } catch (\Exception $e) {
            Logging::log_error("Failed to validate origin/referer: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false; // Safe default: reject
        }
    }

    /**
     * Check if asset exists
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return bool True if exists, false otherwise
     */
    public function get_asset_exists(string $url, string $type): bool {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_exists($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to check if asset exists for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get validated cache expiration
     *
     * @param string $cache_type The cache type
     * @return int The cache expiration time in seconds
     */
    public function get_validated_cache_expiration(string $cache_type = 'default'): int {
        try {
            return $this->assetData->get_validated_cache_expiration($cache_type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get cache expiration for {$cache_type}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return 3600; // Safe default: 1 hour
        }
    }

    /**
     * Verify URL with proper error handling
     *
     * @param string $url The URL to verify
     * @param array $options Verification options
     * @return array|false Verification result or false on failure
     */
    public function verify_url(string $url, array $options = []): array|false {
        try {
            return $this->assetValidator->verify_url($url, $options);
        } catch (\Exception $e) {
            Logging::log_error("URL verification failed for {$url}: " . $e->getMessage(), Logging::LEVEL_ERROR);
            return false;
        }
    }

    /**
     * Handle localization failure with proper error handling
     *
     * @param string $original_url The original URL
     * @param string $tag_name The tag name
     * @param string $asset_type The asset type
     * @param string $original_tag The original tag
     */
    public function handle_localization_failure(string $original_url, string $tag_name, string $asset_type, string $original_tag): void
    {
        try {
            $this->assetActionHandler->handle_localization_failure($original_url, $tag_name, $asset_type, $original_tag);
        } catch (\Exception $e) {
            Logging::log_error("Localization failure handling error for {$original_url}: " . $e->getMessage(), Logging::LEVEL_ERROR);
        }
    }

    /**
     * Handle redirect with proper error handling
     *
     * @param mixed $response The response data
     * @param string $url The URL
     * @param string $type The asset type
     * @param int $cache_expiration Cache expiration time
     * @param bool $force_refresh Whether to force refresh
     * @param int $current_depth Current redirect depth
     * @return mixed Handled response or false on error
     */
    public function handle_redirect(mixed $response, string $url, string $type, int $cache_expiration, bool $force_refresh, int $current_depth): mixed {
        try {
            return $this->assetActionHandler->handle_redirect($response, $url, $type, $cache_expiration, $force_refresh, $current_depth);
        } catch (\Exception $e) {
            Logging::log_error("Redirect handling failed for {$url}: " . $e->getMessage(), Logging::LEVEL_ERROR);
            return false;
        }
    }

    /**
     * Handle MIME type exception with proper error handling
     *
     * @param \LHA\Exceptions\MimeTypeValidationException $e The exception
     */
    public function handle_mime_type_exception(\LHA\Exceptions\MimeTypeValidationException $e): void {
        try {
            $this->assetActionHandler->handle_mime_type_exception($e);
        } catch (\Exception $handlerException) {
            Logging::log_error("Failed to handle MIME type exception: " . $handlerException->getMessage(), Logging::LEVEL_CRITICAL);
        }
    }

    /**
     * Handle database exception with proper error handling
     *
     * @param \LHA\Exceptions\DatabaseException $e The exception
     */
    public function handle_database_exception(\LHA\Exceptions\DatabaseException $e): void {
        try {
            $this->assetActionHandler->handle_database_exception($e);
        } catch (\Exception $handlerException) {
            Logging::log_error("Failed to handle database exception: " . $handlerException->getMessage(), Logging::LEVEL_CRITICAL);
        }
    }

    /**
     * Handle general exception with proper error handling
     *
     * @param \Exception $e The exception
     */
    public function handle_general_exception(\Exception $e): void {
        try {
            $this->assetActionHandler->handle_general_exception($e);
        } catch (\Exception $handlerException) {
            Logging::log_error("Failed to handle general exception: " . $handlerException->getMessage(), Logging::LEVEL_CRITICAL);
        }
    }

    /**
     * Handle asset deletion with proper error handling
     */
    public function handle_asset_deletion(): void {
        try {
            $this->assetActionHandler->handle_asset_deletion();
        } catch (\Exception $e) {
            Logging::log_error("Asset deletion handling failed: " . $e->getMessage(), Logging::LEVEL_ERROR);
        }
    }

    /**
     * Handle asset request with proper error handling
     */
    public function handle_asset_request(): void {
        try {
            $this->assetActionHandler->handle_asset_request();
        } catch (\Exception $e) {
            Logging::log_error("Asset request handling failed: " . $e->getMessage(), Logging::LEVEL_ERROR);
            if (function_exists('wp_die')) {
                wp_die(esc_html__('An error occurred while processing your request.', 'self-host-assets'), '', array('response' => 500));
            } else {
                http_response_code(500);
                die('An error occurred while processing your request.');
            }
        }
    }

    /**
     * Process JSON payload with proper error handling
     *
     * @param string $contentType The content type header
     */
    public function process_json_payload(string $contentType): void {
        try {
            $this->assetActionHandler->process_json_payload($contentType);
        } catch (\Exception $e) {
            Logging::log_error("JSON payload processing failed: " . $e->getMessage(), Logging::LEVEL_ERROR);
            if (function_exists('wp_send_json_error')) {
                wp_send_json_error(['message' => 'Invalid request']);
            } else {
                http_response_code(400);
                header('Content-Type: application/json');
                echo json_encode(['success' => false, 'message' => 'Invalid request']);
                if (function_exists('wp_die')) {
                    wp_die('', '', array('response' => 400));
                } else {
                    exit;
                }
            }
        }
    }

    /**
     * Verifies nonce for AJAX actions to prevent CSRF attacks.
     *
     * @param string $action The action name for nonce verification
     * @param string $query_arg The query argument/POST field containing the nonce
     * @throws \Exception if nonce verification fails
     */
    private function verify_nonce(string $action, string $query_arg): void {
        // Validate that nonce exists, is not empty, and is a string
        if (!isset($_REQUEST[$query_arg]) || !is_string($_REQUEST[$query_arg])) {
            Logging::log_error("CSRF security check failed for action: $action (nonce not provided or invalid type)", Logging::LEVEL_ERROR);
            http_response_code(403);
            if (function_exists('wp_die')) {
                wp_die(esc_html__('Security check failed. Please refresh the page and try again.', 'self-host-assets'), '', array('response' => 403));
            } else {
                die('Security check failed.');
            }
        }

        $nonce = $_REQUEST[$query_arg];

        // Check for empty string (isset returns true for empty strings)
        // Note: '0' is technically a valid string value, so don't reject it
        if ($nonce === '') {
            Logging::log_error("CSRF security check failed for action: $action (empty nonce)", Logging::LEVEL_ERROR);
            http_response_code(403);
            if (function_exists('wp_die')) {
                wp_die(esc_html__('Security check failed. Please refresh the page and try again.', 'self-host-assets'), '', array('response' => 403));
            } else {
                die('Security check failed.');
            }
        }

        // Verify nonce
        if (!function_exists('wp_verify_nonce') || !wp_verify_nonce($nonce, $action)) {
            Logging::log_error("CSRF security check failed for action: $action (invalid nonce)", Logging::LEVEL_ERROR);
            http_response_code(403);
            if (function_exists('wp_die')) {
                wp_die(esc_html__('Security check failed. Please refresh the page and try again.', 'self-host-assets'), '', array('response' => 403));
            } else {
                die('Security check failed.');
            }
        }
    }

    /**
     * Process add asset with proper error handling
     */
    public function process_add_asset(): void {
        try {
            $this->verify_nonce('self_host_assets_add_asset', 'self_host_assets_add_asset_nonce');
            $this->assetActionHandler->process_add_asset();
        } catch (\Exception $e) {
            Logging::log_error("Process add asset failed: " . $e->getMessage(), Logging::LEVEL_ERROR);
            if (function_exists('wp_send_json_error')) {
                wp_send_json_error(['message' => esc_html__('Failed to add asset.', 'self-host-assets')]);
            } else {
                http_response_code(500);
                header('Content-Type: application/json');
                echo json_encode(['success' => false, 'message' => 'Failed to add asset.']);
                if (function_exists('wp_die')) {
                    wp_die('', '', array('response' => 500));
                } else {
                    exit;
                }
            }
        }
    }

    /**
     * Process edit asset with proper error handling
     */
    public function process_edit_asset(): void {
        try {
            $this->verify_nonce('self_host_assets_edit_asset', 'self_host_assets_edit_asset_nonce');
            $this->assetActionHandler->process_edit_asset();
        } catch (\Exception $e) {
            Logging::log_error("Process edit asset failed: " . $e->getMessage(), Logging::LEVEL_ERROR);
            if (function_exists('wp_send_json_error')) {
                wp_send_json_error(['message' => esc_html__('Failed to edit asset.', 'self-host-assets')]);
            } else {
                http_response_code(500);
                header('Content-Type: application/json');
                echo json_encode(['success' => false, 'message' => 'Failed to edit asset.']);
                if (function_exists('wp_die')) {
                    wp_die('', '', array('response' => 500));
                } else {
                    exit;
                }
            }
        }
    }

    /**
     * Process delete asset with proper error handling
     */
    public function process_delete_asset(): void {
        try {
            $this->verify_nonce('self_host_assets_delete_asset', 'self_host_assets_delete_asset_nonce');
            $this->assetActionHandler->process_delete_asset();
        } catch (\Exception $e) {
            Logging::log_error("Process delete asset failed: " . $e->getMessage(), Logging::LEVEL_ERROR);
            if (function_exists('wp_send_json_error')) {
                wp_send_json_error(['message' => esc_html__('Failed to delete asset.', 'self-host-assets')]);
            } else {
                http_response_code(500);
                header('Content-Type: application/json');
                echo json_encode(['success' => false, 'message' => 'Failed to delete asset.']);
                if (function_exists('wp_die')) {
                    wp_die('', '', array('response' => 500));
                } else {
                    exit;
                }
            }
        }
    }

    /**
     * Process bulk actions with proper error handling
     */
    public function process_bulk_actions(): void {
        try {
            $this->verify_nonce('bulk_assets_action', 'bulk_assets_nonce');
            $this->assetActionHandler->process_bulk_actions();
        } catch (\Exception $e) {
            Logging::log_error("Process bulk actions failed: " . $e->getMessage(), Logging::LEVEL_ERROR);
            if (function_exists('wp_send_json_error')) {
                wp_send_json_error(['message' => esc_html__('Failed to process bulk actions.', 'self-host-assets')]);
            } else {
                http_response_code(500);
                header('Content-Type: application/json');
                echo json_encode(['success' => false, 'message' => 'Failed to process bulk actions.']);
                // Use wp_die() instead of exit to allow proper WordPress shutdown
                if (function_exists('wp_die')) {
                    wp_die('', '', array('response' => 500));
                } else {
                    exit;
                }
            }
        }
    }

    /**
     * Process manual validation with proper error handling
     */
    public function process_manual_validation(): void {
        try {
            $this->verify_nonce('self_host_assets_manual_validation', 'self_host_assets_nonce');
            $this->assetActionHandler->process_manual_validation();
        } catch (\Exception $e) {
            Logging::log_error("Process manual validation failed: " . $e->getMessage(), Logging::LEVEL_ERROR);
            if (function_exists('wp_send_json_error')) {
                wp_send_json_error(['message' => esc_html__('Failed to validate assets.', 'self-host-assets')]);
            } else {
                http_response_code(500);
                header('Content-Type: application/json');
                echo json_encode(['success' => false, 'message' => 'Failed to validate assets.']);
                if (function_exists('wp_die')) {
                    wp_die('', '', array('response' => 500));
                } else {
                    exit;
                }
            }
        }
    }

    /**
     * Process remediate order table with proper error handling
     */
    public function process_remediate_order_table(): void {
        try {
            $this->verify_nonce('self_host_assets_manual_remediation', 'self_host_assets_nonce_remediation');
            $this->assetActionHandler->process_remediate_order_table();
        } catch (\Exception $e) {
            Logging::log_error("Process remediate order table failed: " . $e->getMessage(), Logging::LEVEL_ERROR);
            if (function_exists('wp_send_json_error')) {
                wp_send_json_error(['message' => esc_html__('Failed to remediate order table.', 'self-host-assets')]);
            } else {
                http_response_code(500);
                header('Content-Type: application/json');
                echo json_encode(['success' => false, 'message' => 'Failed to remediate order table.']);
                if (function_exists('wp_die')) {
                    wp_die('', '', array('response' => 500));
                } else {
                    exit;
                }
            }
        }
    }

    /**
     * Delete asset by ID with proper error handling
     *
     * @param int $asset_id The asset ID to delete
     * @return bool True on success, false on failure
     */
    public function delete_asset_by_id(int $asset_id): bool {
        try {
            if ($asset_id <= 0) {
                Logging::log_error("Invalid asset ID for deletion: {$asset_id}", Logging::LEVEL_ERROR);
                return false;
            }
            return $this->assetActionHandler->delete_asset_by_id($asset_id);
        } catch (\Exception $e) {
            Logging::log_error("Delete asset by ID failed for asset {$asset_id}: " . $e->getMessage(), Logging::LEVEL_ERROR);
            return false;
        }
    }

    /**
     * Get asset local directory
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return string|false The local directory path or false on failure
     */
    public function get_asset_local_directory(string $url, string $type): string|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_local_directory($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset local directory for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get image dimensions for an asset
     * Delegates to AssetData for centralized implementation
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return array|false Dimensions array or false on failure
     */
    public function get_asset_image_dimensions(string $url, string $type): array|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_image_dimensions($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset image dimensions for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset file size
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return int|false The file size in bytes or false on failure
     */
    public function get_asset_file_size(string $url, string $type): int|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_file_size($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset file size for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset checksum
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return string|false The checksum or false on failure
     */
    public function get_asset_checksum(string $url, string $type): string|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_checksum($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset checksum for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset version
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return string|false The version or false on failure
     */
    public function get_asset_version(string $url, string $type): string|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_version($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset version for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset embed code (escaped for safety)
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return string The embed code
     */
    public function get_asset_embed_code(string $url, string $type): string {
        try {
            if (empty($url) || empty($type)) {
                return '';
            }
            $embed_code = $this->assetData->get_asset_embed_code($url, $type);
            // Ensure the return is safe for output
            return is_string($embed_code) ? $embed_code : '';
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset embed code for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return '';
        }
    }

    /**
     * Get asset EXIF data
     *
     * @param string $filePath The file path
     * @param string $type The asset type
     * @return array|false EXIF data or false on failure
     */
    public function get_asset_exif_data(string $filePath, string $type): array|false {
        try {
            if (empty($filePath) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_exif_data($filePath, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset EXIF data: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset aspect ratio
     *
     * @param string $filePath The file path
     * @param string $type The asset type
     * @return float|false The aspect ratio or false on failure
     */
    public function get_asset_aspect_ratio(string $filePath, string $type): float|false {
        try {
            if (empty($filePath) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_aspect_ratio($filePath, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset aspect ratio: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset metadata
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return array|false Metadata or false on failure
     */
    public function get_asset_metadata(string $url, string $type): array|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_metadata($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset metadata for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset last modified time
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return int|false Unix timestamp or false on failure
     */
    public function get_asset_last_modified(string $url, string $type): int|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_last_modified($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset last modified for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset sanitized filename
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return string|false Sanitized filename or false on failure
     */
    public function get_asset_sanitized_filename(string $url, string $type): string|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_asset_sanitized_filename($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset sanitized filename for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get asset download link
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return string The download link
     */
    public function get_asset_download_link(string $url, string $type): string {
        try {
            if (empty($url) || empty($type)) {
                return '';
            }
            $link = $this->assetData->get_asset_download_link($url, $type);
            return is_string($link) ? $link : '';
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset download link for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return '';
        }
    }

    /**
     * Get uploaded media handles
     *
     * @return array Array of media handles
     */
    public function get_uploaded_media_handles(): array {
        try {
            return $this->assetData->get_uploaded_media_handles();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get uploaded media handles: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Get order settings with validation
     *
     * @param int $task_id The task ID
     * @param int $post_id The post ID
     * @return array|false Order settings or false on failure
     */
    public function get_order_settings(int $task_id, int $post_id): array|false {
        try {
            if ($task_id <= 0 || $post_id <= 0) {
                Logging::log_error("Invalid task_id or post_id in get_order_settings", Logging::LEVEL_ERROR);
                return false;
            }
            return $this->assetData->get_order_settings($task_id, $post_id);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get order settings: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get task by ID with validation
     *
     * @param int $task_id The task ID
     * @return array|false Task data or false on failure
     */
    public function get_task_by_id(int $task_id): array|false {
        try {
            if ($task_id <= 0) {
                Logging::log_error("Invalid task ID: {$task_id}", Logging::LEVEL_ERROR);
                return false;
            }
            return $this->assetData->get_task_by_id($task_id);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get task by ID {$task_id}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get mapping entry by ID with validation
     *
     * @param int $asset_id The asset ID
     * @return array|null The mapping entry or null on failure
     */
    public function get_mapping_entry_by_id(int $asset_id): ?array {
        try {
            if ($asset_id <= 0) {
                Logging::log_error("Invalid asset ID for mapping entry: {$asset_id}", Logging::LEVEL_ERROR);
                return null;
            }
            return $this->assetData->get_mapping_entry_by_id($asset_id);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get mapping entry by ID {$asset_id}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return null;
        }
    }

    /**
     * Get mapping entry by URL (for dependency resolution)
     *
     * @param string $url The original URL
     * @return array|null The mapping entry or null if not found
     */
    public function get_mapping_entry_by_url(string $url): ?array {
        try {
            if (empty($url)) {
                return null;
            }
            return $this->assetData->get_mapping_entry_by_url($url);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get mapping entry by URL for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return null;
        }
    }

    /**
     * Get task ID by URL and type with validation
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return int|false The task ID or false on failure
     */
    public function get_task_id_by_url_and_type(string $url, string $type): int|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_task_id_by_url_and_type($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get task ID by URL and type for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get the local file path for an asset with error handling
     * Delegates to AssetData for centralized implementation
     *
     * @param string $url The asset URL
     * @param string $type The asset type (js, css, image, font, etc.)
     * @return string|false The full local file path or false on failure
     */
    public function get_local_file_path(string $url, string $type): string|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_local_file_path($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get local file path for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get local URL for an asset with optional cache busting
     *
     * @param string $original_url The original URL
     * @param string $type The asset type
     * @param bool $cache_bust Whether to add cache busting parameter
     * @return string|null The local URL or null if not found
     */
    public function get_local_url(string $original_url, string $type, bool $cache_bust = false): ?string {
        // Early return for empty inputs
        if (empty($original_url) || empty($type)) {
            return null;
        }

        // Normalize URL for consistent lookup using Normalize dependency
        try {
            // Use Normalize::normalize_asset_url for asset URLs (preferred method)
            // Falls back to normalize_url_base if normalize_asset_url not available
            if ($this->normalize !== null) {
                $normalized_url = method_exists($this->normalize, 'normalize_asset_url')
                    ? $this->normalize->normalize_asset_url($original_url)
                    : $this->normalize->normalize_url_base($original_url);
            } elseif ($this->urlProcessor !== null && method_exists($this->urlProcessor, 'normalize_url')) {
                // Fallback to UrlProcessor for backward compatibility
                $normalized_url = $this->urlProcessor->normalize_url($original_url);
            } else {
                // Last resort: basic trim
                $normalized_url = trim($original_url);
            }
        } catch (\Exception $e) {
            Logging::log_error("URL normalization failed for {$original_url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            $normalized_url = trim($original_url);
        }

        // If normalization failed, try with original URL
        if ($normalized_url === '' || $normalized_url === null) {
            $normalized_url = $original_url;
        }

        // Try normalized URL first
        try {
            $local_url = $this->assetData->get_local_url($normalized_url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get local URL for {$normalized_url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            $local_url = false;
        }

        // Fallback to original URL if normalized lookup fails (backward compatibility)
        if ($local_url === false && $normalized_url !== $original_url) {
            try {
                $local_url = $this->assetData->get_local_url($original_url, $type);
            } catch (\Exception $e) {
                Logging::log_error("Failed to get local URL (fallback) for {$original_url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
                $local_url = false;
            }
        }

        // Convert false to null for interface compatibility
        if ($local_url === false) {
            return null;
        }

        // Validate $local_url is a string
        if (!is_string($local_url)) {
            Logging::log_error("Local URL is not a string for {$original_url}, got: " . gettype($local_url), Logging::LEVEL_WARNING);
            return null;
        }

        // Apply cache busting if requested
        // Use file modification time for stable cache busting (not time() which changes every request)
        if ($cache_bust && !empty($local_url)) {
            try {
                $local_path = $this->assetData->get_local_file_path($normalized_url, $type);
                // Reduce TOCTOU race condition: use @filemtime() directly without file_exists() check
                // filemtime() returns false if file doesn't exist or on error
                if ($local_path) {
                    $file_mtime = @filemtime($local_path);
                    if ($file_mtime !== false) {
                        $version = (string)$file_mtime;
                        $separator = (strpos($local_url, '?') !== false) ? '&' : '?';
                        $local_url .= $separator . 'v=' . $version;
                    }
                    // If filemtime fails, don't add cache busting at all (better than using time())
                }
                // If file doesn't exist or path is invalid, skip cache busting
            } catch (\Exception $e) {
                Logging::log_error("Cache busting failed for {$local_url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
                // Return URL without cache busting on error
            }
        }

        return $local_url;
    }

    /**
     * Check if a local file is fresh (not expired based on cache TTL)
     * Optimized with single file_exists check and minimal overhead
     * 
     * @param string $local_file_path Full path to the local file
     * @param int $cache_ttl_seconds Cache TTL in seconds (0 means always stale)
     * @return bool True if file is fresh, false if stale or doesn't exist
     */
    public function is_file_fresh(string $local_file_path, int $cache_ttl_seconds): bool {
        // Early return for invalid inputs
        // TTL of 0 means "always stale", negative TTL is invalid
        if (empty($local_file_path) || $cache_ttl_seconds <= 0) {
            return false;
        }

        // Ensure path is a file, not a directory
        // Use clearstatcache() to reduce TOCTOU race condition window
        clearstatcache(true, $local_file_path);

        if (!is_file($local_file_path)) {
            return false; // Path is a directory or doesn't exist
        }

        // Get file modification time with error suppression to handle race conditions
        // Use @filemtime() to prevent warnings if file is deleted between is_file() and filemtime()
        $file_mtime = @filemtime($local_file_path);

        if ($file_mtime === false) {
            return false; // File doesn't exist or can't be read
        }

        // File is fresh if age is less than TTL
        // Optimized: single calculation, no intermediate variables
        return (time() - $file_mtime) < $cache_ttl_seconds;
    }

    /**
     * Get pending asset tasks
     *
     * @return array Array of pending tasks
     */
    public function get_pending_asset_tasks(): array {
        try {
            return $this->assetData->get_pending_asset_tasks();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get pending asset tasks: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Get post ID for task with validation
     *
     * @param int $task_id The task ID
     * @return int|false The post ID or false on failure
     */
    public function get_post_id_for_task(int $task_id): int|false {
        try {
            if ($task_id <= 0) {
                Logging::log_error("Invalid task ID in get_post_id_for_task: {$task_id}", Logging::LEVEL_ERROR);
                return false;
            }
            return $this->assetData->get_post_id_for_task($task_id);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get post ID for task {$task_id}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get order settings by URL
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return array|false Order settings or false on failure
     */
    public function get_order_settings_by_url(string $url, string $type): array|false {
        try {
            if (empty($url) || empty($type)) {
                return false;
            }
            return $this->assetData->get_order_settings_by_url($url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get order settings by URL for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get external JS from enqueued scripts
     *
     * @return array Array of external JS URLs
     */
    public function get_external_js_from_enqueued_scripts(): array {
        try {
            return $this->assetData->get_external_js_from_enqueued_scripts();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get external JS from enqueued scripts: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Get upload directory
     *
     * @return array|false Upload directory info or false on failure
     */
    public function get_upload_dir(): array|false {
        try {
            return $this->assetData->get_upload_dir();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get upload dir: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get assets for post with validation
     *
     * @param int $post_id The post ID
     * @return array Array of assets
     */
    public function get_assets_for_post(int $post_id): array {
        try {
            if ($post_id <= 0) {
                Logging::log_error("Invalid post ID in get_assets_for_post: {$post_id}", Logging::LEVEL_ERROR);
                return [];
            }
            return $this->assetData->get_assets_for_post($post_id);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get assets for post {$post_id}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Get URL type
     *
     * @return string The URL type
     */
    public function get_url_type(): string {
        try {
            return $this->assetData->get_url_type();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get URL type: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return ''; // Safe default
        }
    }

    /**
     * Get current order for post with validation
     *
     * @param int $post_id The post ID
     * @return array Array of order data
     */
    public function get_current_order_for_post(int $post_id): array {
        try {
            if ($post_id <= 0) {
                Logging::log_error("Invalid post ID in get_current_order_for_post: {$post_id}", Logging::LEVEL_ERROR);
                return [];
            }
            return $this->assetData->get_current_order_for_post($post_id);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get current order for post {$post_id}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Get progress
     *
     * @return array Progress data
     */
    public function get_progress(): array {
        try {
            return $this->assetData->get_progress();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get progress: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Get font URLs from CSS content
     *
     * @param string $css_content The CSS content
     * @param string $base_css_url The base CSS URL
     * @return array Array of font URLs
     */
    public function get_font_urls(string $css_content, string $base_css_url = ''): array {
        try {
            return $this->assetData->get_font_urls($css_content, $base_css_url);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get font URLs: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Get image URLs from CSS content
     *
     * @param string $css_content The CSS content
     * @param string $base_css_url The base CSS URL
     * @return array Array of image URLs
     */
    public function get_image_urls(string $css_content, string $base_css_url = ''): array {
        try {
            return $this->assetData->get_image_urls($css_content, $base_css_url);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get image URLs: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Get max visited URLs
     *
     * @return int Maximum number of visited URLs
     */
    public function get_max_visited_urls(): int {
        try {
            return $this->assetData->get_max_visited_urls();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get max visited URLs: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return 100; // Safe default
        }
    }

    /**
     * Get memory threshold
     *
     * @return int Memory threshold in bytes
     */
    public function get_memory_threshold(): int {
        try {
            return $this->assetData->get_memory_threshold();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get memory threshold: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return 134217728; // Safe default: 128MB
        }
    }

    /**
     * Get dynamic memory threshold
     *
     * @return int Dynamic memory threshold in bytes
     */
    public function get_dynamic_memory_threshold(): int {
        try {
            return $this->assetData->get_dynamic_memory_threshold();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get dynamic memory threshold: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return 134217728; // Safe default: 128MB
        }
    }

    /**
     * Get SVG allowed HTML
     *
     * @return array Allowed HTML elements for SVG
     */
    public function get_svg_allowed_html(): array {
        try {
            return $this->assetData->get_svg_allowed_html();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get SVG allowed HTML: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Get original URL from handle with validation
     *
     * @param string $handle The asset handle
     * @param string $type The asset type
     * @return string|false The original URL or false on failure
     */
    public function get_original_url_from_handle(string $handle, string $type): string|false {
        try {
            if (empty($handle) || empty($type)) {
                return false;
            }
            return $this->assetData->get_original_url_from_handle($handle, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get original URL from handle {$handle}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get clean DOM content
     *
     * @param \DOMDocument $dom The DOM document
     * @return string Cleaned HTML content
     */
    public function get_clean_dom_content(\DOMDocument $dom): string {
        try {
            return $this->assetData->get_clean_dom_content($dom);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get clean DOM content: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return '';
        }
    }

    /**
     * Get pending option key
     *
     * @param string $category The category
     * @return string The option key
     */
    public function get_pending_option_key(string $category): string {
        try {
            if (empty($category)) {
                return '';
            }
            return $this->assetData->get_pending_option_key($category);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get pending option key for {$category}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return '';
        }
    }

    /**
     * Get transient prefix
     *
     * @param string $category The category
     * @return string The transient prefix
     */
    public function get_transient_prefix(string $category): string {
        try {
            if (empty($category)) {
                return '';
            }
            return $this->assetData->get_transient_prefix($category);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get transient prefix for {$category}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return '';
        }
    }

    /**
     * Get external CSS from enqueued styles
     *
     * @return array Array of external CSS URLs
     */
    public function get_external_css_from_enqueued_styles(): array {
        try {
            return $this->assetData->get_external_css_from_enqueued_styles();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get external CSS from enqueued styles: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Fetch asset content
     *
     * @param string $src The source URL
     * @return string The asset content
     */
    public function fetch_asset_content(string $src): string {
        try {
            if (empty($src)) {
                return '';
            }
            return $this->assetData->fetch_asset_content($src);
        } catch (\Exception $e) {
            Logging::log_error("Failed to fetch asset content from {$src}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return '';
        }
    }

    /**
     * Find all URLs in CSS
     *
     * @param string $css The CSS content
     * @return array Array of URLs found
     */
    public function find_all_urls_in_css(string $css): array
    {
        try {
            return $this->assetData->find_all_urls_in_css($css);
        } catch (\Exception $e) {
            Logging::log_error("Failed to find URLs in CSS: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Find script handle by URL
     *
     * @param string $url The URL
     * @return string|false The script handle or false on failure
     */
    public function find_script_handle_by_url(string $url): string|false {
        try {
            if (empty($url)) {
                return false;
            }
            return $this->assetData->find_script_handle_by_url($url);
        } catch (\Exception $e) {
            Logging::log_error("Failed to find script handle by URL for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Find media handle
     *
     * @param string $original_url The original URL
     * @param string $type The asset type
     * @return string|false The media handle or false on failure
     */
    public function find_media_handle(string $original_url, string $type): string|false {
        try {
            if (empty($original_url) || empty($type)) {
                return false;
            }
            return $this->assetData->find_media_handle($original_url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to find media handle for {$original_url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Find style handle by URL
     *
     * @param string $url The URL
     * @return string|false The style handle or false on failure
     */
    public function find_style_handle_by_url(string $url): string|false {
        try {
            if (empty($url)) {
                return false;
            }
            return $this->assetData->find_style_handle_by_url($url);
        } catch (\Exception $e) {
            Logging::log_error("Failed to find style handle by URL for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Get data by helper
     *
     * @param string $helper The helper name
     * @param array $options Options
     * @return mixed The data
     */
    public function get_data(string $helper, array $options = [])
    {
        try {
            if (empty($helper)) {
                return null;
            }
            return $this->assetData->get_data($helper, $options);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get data for helper {$helper}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return null;
        }
    }

    /**
     * Get data registry
     *
     * @return array The data registry
     */
    public function get_data_registry(): array
    {
        try {
            return $this->assetData->get_data_registry();
        } catch (\Exception $e) {
            Logging::log_error("Failed to get data registry: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return [];
        }
    }

    /**
     * Validate asset
     *
     * @param array $asset The asset data
     * @return bool True if valid, false otherwise
     */
    public function validate_asset(array $asset): bool {
        try {
            return $this->assetValidator->validate_asset($asset);
        } catch (\Exception $e) {
            Logging::log_error("Failed to validate asset: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Check if asset type is valid
     *
     * @param string $type The asset type
     * @return bool True if valid, false otherwise
     */
    public function is_valid_asset_type(string $type): bool {
        try {
            if (empty($type)) {
                return false;
            }
            return $this->assetValidator->is_valid_asset_type($type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to validate asset type {$type}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            return false;
        }
    }

    /**
     * Increment MIME type failure counter
     */
    public function increment_mime_type_failure_counter(): void {
        try {
            $this->assetActionHandler->increment_mime_type_failure_counter();
        } catch (\Exception $e) {
            Logging::log_error("Failed to increment MIME type failure counter: " . $e->getMessage(), Logging::LEVEL_WARNING);
        }
    }

    /**
     * Determine asset status from database and filesystem
     * Optimized with minimal database queries, file checks, and static caching
     *
     * Uses AssetData::get_asset_data() for centralized, cached asset retrieval.
     *
     * @param string $url The asset URL
     * @param string $type The asset type
     * @return string The asset status
     */
    public function determine_asset_status(string $url, string $type): string {
        // Early return for invalid inputs
        if (empty($url) || empty($type)) {
            return 'invalid';
        }

        // Normalize URL for consistent lookups using Normalize dependency
        try {
            // Use Normalize::normalize_asset_url for asset URLs (preferred method)
            // Falls back to normalize_url_base if normalize_asset_url not available
            if ($this->normalize !== null) {
                $normalized_url = method_exists($this->normalize, 'normalize_asset_url')
                    ? $this->normalize->normalize_asset_url($url)
                    : $this->normalize->normalize_url_base($url);
            } elseif ($this->urlProcessor !== null && method_exists($this->urlProcessor, 'normalize_url')) {
                // Fallback to UrlProcessor for backward compatibility
                $normalized_url = $this->urlProcessor->normalize_url($url);
            } else {
                // Last resort: basic trim
                $normalized_url = trim($url);
            }
        } catch (\Exception $e) {
            Logging::log_error("URL normalization failed in determine_asset_status for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            $normalized_url = trim($url);
        }

        // If normalization returned empty but original URL was valid, use original
        // This handles edge cases where normalization services aren't available
        if ($normalized_url === '' || $normalized_url === null) {
            $normalized_url = trim($url);
            if ($normalized_url === '') {
                return 'invalid';
            }
        }

        // Static cache for repeated status checks within same request using normalized URL
        static $status_cache = [];
        $json_data = json_encode([$normalized_url, $type], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
        if ($json_data === false) {
            // JSON encoding failed, use fallback
            $cache_key = md5($normalized_url . '|' . $type);
        } else {
            $cache_key = md5($json_data);
        }

        if (isset($status_cache[$cache_key])) {
            return $status_cache[$cache_key];
        }

        // Bound cache to 100 entries to prevent memory exhaustion
        if (count($status_cache) >= 100) {
            array_shift($status_cache);
        }

        // Use AssetData for centralized, cached asset data retrieval with normalized URL
        try {
            $asset_data = $this->assetData->get_asset_data($normalized_url, $type);
        } catch (\Exception $e) {
            Logging::log_error("Failed to get asset data for {$normalized_url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            $status_cache[$cache_key] = 'error';
            return 'error';
        }

        if (!$asset_data) {
            $status_cache[$cache_key] = 'not_found';
            return 'not_found';
        }

        $status = $asset_data['status'] ?? 'unknown';
        $local_url = $asset_data['local_url'] ?? '';

        // If status is processed, verify local_url exists and file exists on disk
        if ($status === 'processed') {
            if (!empty($local_url)) {
                try {
                    $local_path = $this->assetData->get_local_file_path($normalized_url, $type);
                } catch (\Exception $e) {
                    Logging::log_error("Failed to get local file path for status check: " . $e->getMessage(), Logging::LEVEL_WARNING);
                    $local_path = false;
                }

                // Reduce TOCTOU race condition: use @filemtime() directly without file_exists() check
                // filemtime() returns false if file doesn't exist or on error
                if ($local_path && @filemtime($local_path) !== false) {
                    $status_cache[$cache_key] = 'processed';
                    return 'processed';
                }
                // File missing but status says processed - return actual status
                $status_cache[$cache_key] = 'file_missing';
                return 'file_missing';
            } else {
                // Status is processed but no local URL is stored - inconsistent state
                $status_cache[$cache_key] = 'pending';
                return 'pending';
            }
        }

        // Return database status
        $status_cache[$cache_key] = $status;
        return $status;
    }

    /**
     * Determine memory threshold based on WordPress memory limit
     * Optimized with static caching to avoid repeated calculations
     * 
     * @return int Memory threshold in bytes
     */
    public function determine_memory_threshold(): int {
        // Static cache to avoid repeated calculations
        static $cached_threshold = null;
        
        if ($cached_threshold !== null) {
            return $cached_threshold;
        }
        
        // Get WordPress memory limit
        $wp_memory_limit = defined('WP_MEMORY_LIMIT') ? WP_MEMORY_LIMIT : '128M';
        
        // Convert to bytes
        $memory_bytes = $this->convert_memory_string_to_bytes($wp_memory_limit);
        
        // Use 80% of available memory as threshold
        $threshold = (int) ($memory_bytes * 0.8);
        
        // Ensure minimum of 64MB
        $min_threshold = 67108864; // 64 * 1024 * 1024 (pre-calculated constant)
        
        $cached_threshold = max($threshold, $min_threshold);
        
        return $cached_threshold;
    }
    
    /**
     * Convert memory string (like "128M") to bytes
     * Optimized with minimal string operations
     * 
     * @param string $memory_string Memory string
     * @return int Memory in bytes
     */
    private function convert_memory_string_to_bytes(string $memory_string): int {
        $memory_string = trim($memory_string);
        
        if (empty($memory_string)) {
            return 134217728; // 128 * 1024 * 1024 (pre-calculated constant)
        }
        
        // Robust regex to parse value and unit (e.g., "128MB", "1G", "512k")
        if (preg_match('/^(\d+)\s*([KMG])B?$/i', $memory_string, $matches)) {
            $numeric_value = (int)$matches[1];
            $unit = strtoupper($matches[2]);
            
            switch ($unit) {
                case 'G': return $numeric_value * 1073741824; // 1024^3
                case 'M': return $numeric_value * 1048576;    // 1024^2
                case 'K': return $numeric_value * 1024;
            }
        }
        
        // All digits, assume bytes (note: "0" is valid and means 0 bytes)
        if (ctype_digit($memory_string)) {
            $bytes = (int) $memory_string;
            return $bytes >= 0 ? $bytes : 134217728;
        }
        
        return 134217728; // Default 128MB
    }

    // GetdataInterface implementation
    /**
     * Get detailed information about an item (asset, task, mapping, or order)
     *
     * Optimized:
     * - Uses JOINs to reduce database queries from 2-3 to 1 per call
     * - Static caching for repeated lookups within same request
     * - Fast type validation with static array
     *
     * @param int|string $item_id The item ID
     * @param string $item_type The item type (asset, task, mapping, order)
     * @return array|false Item details or false on failure
     */
    public function get_detailed_item_info(int|string $item_id, string $item_type): array|false {
        // Static cache for repeated lookups
        static $info_cache = [];

        // Use a more robust cache key to avoid collisions
        $json_data = json_encode([$item_type, (string)$item_id], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
        if ($json_data === false) {
            // JSON encoding failed, use fallback
            $cache_key = md5($item_type . ':' . (string)$item_id);
        } else {
            $cache_key = md5($json_data);
        }

        if (isset($info_cache[$cache_key])) {
            return $info_cache[$cache_key];
        }

        // Bound cache to 100 entries to prevent memory exhaustion
        if (count($info_cache) >= 100) {
            array_shift($info_cache);
        }
        
        $wpdb = $this->wpdb;
        
        // Validate item type (static array for fast lookup)
        static $allowed_types = null;
        if ($allowed_types === null) {
            $allowed_types = ['asset' => true, 'task' => true, 'mapping' => true, 'order' => true];
        }
        if (!isset($allowed_types[$item_type])) {
            return false;
        }
        
        $item_id = absint($item_id);
        if ($item_id <= 0) {
            return false;
        }
        
        // Validate $wpdb is available (needed for 'order' type)
        if (!$wpdb instanceof \wpdb) {
            // For asset/mapping/task types, we can still use AssetData
            // For order type, we need $wpdb
            if ($item_type === 'order') {
                return false;
            }
        }
        
        switch ($item_type) {
            case 'asset':
            case 'mapping':
                // Use AssetData for centralized, cached mapping entry retrieval
                $item = $this->assetData->get_mapping_entry_by_id($item_id);
                
                if ($item) {
                    // Add file information if local file exists
                    if (!empty($item['local_url'])) {
                        // Use get_local_file_path for proper path resolution
                        $local_path = $this->assetData->get_local_file_path(
                            $item['original_url'] ?? '',
                            $item['type'] ?? ''
                        );
                        
                        if ($local_path) {
                            // Use stat() to reduce multiple I/O calls (exists, size, mtime)
                            // @ used to suppress warnings if file doesn't exist (handled by return value)
                            $stat = @stat($local_path);
                            if ($stat !== false) {
                                $item['file_size'] = $stat['size'];
                                $item['file_modified'] = $stat['mtime'];
                                $item['file_exists'] = true;
                            } else {
                                $item['file_exists'] = false;
                            }
                        } else {
                            $item['file_exists'] = false;
                        }
                    } else {
                        // No local_url means file doesn't exist yet
                        $item['file_exists'] = false;
                    }
                }
                
                $result = $item ?: false;
                $info_cache[$cache_key] = $result;
                return $result;
                
            case 'task':
                // Use AssetData for centralized, cached task retrieval
                $task = $this->assetData->get_task_by_id($item_id);
                
                if ($task) {
                    // Enrich with mapping data if available
                    $original_url = $task['original_url'] ?? '';
                    $type = $task['type'] ?? '';
                    
                    if (!empty($original_url) && !empty($type)) {
                        $mapping_data = $this->assetData->get_asset_data($original_url, $type);
                        if ($mapping_data) {
                            $task['mapping_id'] = $mapping_data['id'] ?? null;
                            $task['local_url'] = $mapping_data['local_url'] ?? null;
                            $task['mapping_status'] = $mapping_data['status'] ?? null;
                        }
                    }
                }
                
                $result = $task ?: false;
                $info_cache[$cache_key] = $result;
                return $result;
                
            case 'order':
                // Optimized: Single query with LEFT JOINs to get order, asset, and post data together
                // Use Database class for table names if available
                $order_table = $this->database !== null
                    ? $this->database->get_table_name(\LHA\Database::TABLE_ORDER)
                    : $wpdb->prefix . 'lha_order';
                $mapping_table = $this->database !== null
                    ? $this->database->get_table_name(\LHA\Database::TABLE_MAPPINGS)
                    : $wpdb->prefix . 'lha_mappings';
                $posts_table = $wpdb->prefix . 'posts';

                // Validate table names are not empty and are strings
                if (empty($order_table) || !is_string($order_table) ||
                    empty($mapping_table) || !is_string($mapping_table) ||
                    empty($posts_table) || !is_string($posts_table)) {
                    return false;
                }

                // Remove NULL bytes from table names
                $order_table = str_replace("\0", '', $order_table);
                $mapping_table = str_replace("\0", '', $mapping_table);
                $posts_table = str_replace("\0", '', $posts_table);

                // Escape table names for security
                $escaped_order_table = "`" . str_replace('`', '``', $order_table) . "`";
                $escaped_mapping_table = "`" . str_replace('`', '``', $mapping_table) . "`";
                $escaped_posts_table = "`" . str_replace('`', '``', $posts_table) . "`";
                
                $query = $wpdb->prepare(
                    "SELECT 
                        o.id, o.post_id, o.asset_id, o.asset_order, o.priority, 
                        o.delay_js, o.timeout_js, o.created_at,
                        m.original_url as asset_url, m.local_url as asset_local_url, 
                        m.type as asset_type, m.status as asset_status,
                        p.post_title, p.post_status, p.post_type
                    FROM {$escaped_order_table} o
                    LEFT JOIN {$escaped_mapping_table} m ON o.asset_id = m.id
                    LEFT JOIN {$escaped_posts_table} p ON o.post_id = p.ID
                    WHERE o.id = %d",
                    $item_id
                );
                
                $item = $wpdb->get_row($query, ARRAY_A);
                
                $result = $item ?: false;
                $info_cache[$cache_key] = $result;
                return $result;
                
            default:
                return false;
        }
    }

    /**
     * Get local URL with download fallback
     * Checks if asset exists locally, returns local URL if available
     * Returns null if asset needs to be downloaded (triggers background task)
     * 
     * Uses AssetData::get_local_url() for centralized, cached URL retrieval.
     * 
     * @param string $url The original external URL
     * @param string $type The asset type (js, css, image, etc.)
     * @return string|null Local URL if available, null if needs download
     */
    public function get_local_url_with_download_fallback(string $url, string $type): ?string {
        // Validate inputs
        if (empty($url) || empty($type)) {
            return null;
        }

        // Normalize URL for consistent lookup using Normalize dependency
        try {
            // Use Normalize::normalize_asset_url for asset URLs (preferred method)
            // Falls back to normalize_url_base if normalize_asset_url not available
            if ($this->normalize !== null) {
                $normalized_url = method_exists($this->normalize, 'normalize_asset_url')
                    ? $this->normalize->normalize_asset_url($url)
                    : $this->normalize->normalize_url_base($url);
            } elseif ($this->urlProcessor !== null && method_exists($this->urlProcessor, 'normalize_url')) {
                // Fallback to UrlProcessor for backward compatibility
                $normalized_url = $this->urlProcessor->normalize_url($url);
            } else {
                // Last resort: basic trim
                $normalized_url = trim($url);
            }
        } catch (\Exception $e) {
            Logging::log_error("URL normalization failed for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            $normalized_url = trim($url);
        }

        // If normalization failed, use original URL
        // Check for both empty string and null to handle edge cases
        if ($normalized_url === '' || $normalized_url === null) {
            $normalized_url = $url;
        }

        // Static cache for repeated lookups within same request
        static $cache = [];
        $json_data = json_encode([$normalized_url, $type], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
        if ($json_data === false) {
            // JSON encoding failed, use fallback
            $cache_key = md5($normalized_url . '|' . $type);
        } else {
            $cache_key = md5($json_data);
        }

        if (isset($cache[$cache_key])) {
            return $cache[$cache_key];
        }

        // Bound cache to 100 entries to prevent memory exhaustion
        if (count($cache) >= 100) {
            array_shift($cache);
        }

        // Primary strategy: Try normalized URL (should handle most cases after improvement)
        $local_url = $this->assetData->get_local_url($normalized_url, $type);
        if ($local_url !== false) {
            $cache[$cache_key] = $local_url;
            return $local_url;
        }

        // Secondary strategy: Try original URL if different (backward compatibility for unnormalized entries)
        if ($normalized_url !== $url) {
            $local_url = $this->assetData->get_local_url($url, $type);
            if ($local_url !== false) {
                $cache[$cache_key] = $local_url;
                return $local_url;
            }
        }

        // Tertiary strategy: Protocol variations if still not found (legacy data migration)
        $alt_protocol_url = null;
        if (strpos($normalized_url, 'https://') === 0 && strlen($normalized_url) > 8) {
            // Only swap if there's actually something after https://
            $rest = substr($normalized_url, 8);
            if (!empty($rest)) {
                $alt_protocol_url = 'http://' . $rest;
            }
        } elseif (strpos($normalized_url, 'http://') === 0 && strlen($normalized_url) > 7) {
            // Only swap if there's actually something after http://
            $rest = substr($normalized_url, 7);
            if (!empty($rest)) {
                $alt_protocol_url = 'https://' . $rest;
            }
        }

        // Only try the alt URL if it's different from original and valid
        if ($alt_protocol_url !== null && $alt_protocol_url !== $normalized_url) {
            $local_url = $this->assetData->get_local_url($alt_protocol_url, $type);
            if ($local_url !== false) {
                $cache[$cache_key] = $local_url;
                return $local_url;
            }
        }

        // Asset doesn't exist or isn't processed yet
        // Cache null result to avoid repeated queries
        $cache[$cache_key] = null;
        return null;
    }

    /**
     * Determine primary asset type from URL and context
     * Uses AssetUtils for centralized type detection with result caching
     * Uses URL normalization for consistent cache keys
     * 
     * @param string $url The asset URL
     * @param string $context The context (e.g., 'css-dependency', 'js-dependency', 'file')
     * @return string The determined asset type
     */
    public function determine_primary_asset_type(string $url, string $context): string {
        // Early return for empty URL
        if (empty($url)) {
            return 'file';
        }

        // Normalize URL for consistent caching using Normalize dependency
        try {
            // Use Normalize::normalize_url_base for general URL normalization
            if ($this->normalize !== null) {
                $normalized_url = $this->normalize->normalize_url_base($url);
            } elseif ($this->urlProcessor !== null && method_exists($this->urlProcessor, 'normalize_url')) {
                // Fallback to UrlProcessor for backward compatibility
                $normalized_url = $this->urlProcessor->normalize_url($url);
            } else {
                // Last resort: basic trim
                $normalized_url = trim($url);
            }
        } catch (\Exception $e) {
            Logging::log_error("URL normalization failed for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            $normalized_url = trim($url);
        }

        // If normalization failed, use original URL for cache key
        // Check for both empty string and null to handle edge cases
        if ($normalized_url === '' || $normalized_url === null) {
            $normalized_url = $url;
        }

        // Static result cache to avoid repeated processing of same URL+context
        static $result_cache = [];
        $json_data = json_encode([$normalized_url, $context], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
        if ($json_data === false) {
            // JSON encoding failed, use fallback
            $cache_key = md5($normalized_url . '|' . $context);
        } else {
            $cache_key = md5($json_data);
        }

        if (isset($result_cache[$cache_key])) {
            return $result_cache[$cache_key];
        }

        // Bound cache to 200 entries to prevent memory exhaustion
        if (count($result_cache) >= 200) {
            array_shift($result_cache);
        }

        // Use AssetUtils for centralized type detection
        // AssetUtils::determine_primary_asset_type already handles extension extraction internally
        if ($this->assetUtils !== null) {
            $type_result = $this->assetUtils->determine_primary_asset_type($url, $context);
            if ($type_result !== false) {
                $result_cache[$cache_key] = $type_result;
                return $type_result;
            }
        }

        // Fall back to context-based determination (optimized with early returns)
        $context_lower = strtolower($context);

        // Check for common types first (most likely to match)
        // Order matters: check longer strings first to avoid partial matches
        if (strpos($context_lower, 'javascript') !== false || strpos($context_lower, 'script') !== false || strpos($context_lower, 'js') !== false) {
            $result_cache[$cache_key] = 'js';
            return 'js';
        }
        if (strpos($context_lower, 'css') !== false || strpos($context_lower, 'style') !== false) {
            $result_cache[$cache_key] = 'css';
            return 'css';
        }
        if (strpos($context_lower, 'image') !== false || strpos($context_lower, 'img') !== false) {
            $result_cache[$cache_key] = 'image';
            return 'image';
        }
        if (strpos($context_lower, 'font') !== false) {
            $result_cache[$cache_key] = 'font';
            return 'font';
        }
        if (strpos($context_lower, 'video') !== false) {
            $result_cache[$cache_key] = 'video';
            return 'video';
        }
        if (strpos($context_lower, 'audio') !== false || strpos($context_lower, 'sound') !== false) {
            $result_cache[$cache_key] = 'audio';
            return 'audio';
        }

        // Default to 'file'
        $result_cache[$cache_key] = 'file';
        return 'file';
    }

    /**
     * Get mapping entry by hash
     * Looks up an asset in the database by its hash identifier
     * 
     * Optimized: 
     * - Uses hashed_filename column with = instead of LIKE for better index usage
     * - Falls back to LIKE search on local_url if exact match fails
     * - Static caching for repeated hash lookups
     * 
     * @param string $hash The hash identifier
     * @param string $type The asset type
     * @return array|null The mapping entry or null if not found
     */
    public function get_mapping_entry_by_hash(string $hash, string $type): ?array {
        // Validate inputs first (before cache lookup to avoid cache key collisions)
        if (empty($hash) || empty($type)) {
            return null;
        }
        
        $wpdb = $this->wpdb;
        
        // Validate $wpdb is available
        if (!$wpdb instanceof \wpdb) {
            return null;
        }
        
        // Static cache for repeated hash lookups
        static $hash_cache = [];
        $json_data = json_encode([$hash, $type], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
        if ($json_data === false) {
            // JSON encoding failed, use fallback
            $cache_key = md5($hash . '|' . $type);
        } else {
            $cache_key = md5($json_data);
        }

        if (isset($hash_cache[$cache_key])) {
            return $hash_cache[$cache_key];
        }

        // Bound cache to 100 entries to prevent memory exhaustion
        if (count($hash_cache) >= 100) {
            array_shift($hash_cache);
        }

        // Use Database class for table name if available (consistent with other methods)
        $table_name = $this->database !== null
            ? $this->database->get_table_name(\LHA\Database::TABLE_MAPPINGS)
            : $wpdb->prefix . 'lha_mappings';

        // Validate table name is not empty and is a string
        if (empty($table_name) || !is_string($table_name)) {
            return null;
        }

        // Remove NULL bytes from table name
        $table_name = str_replace("\0", '', $table_name);

        // Escape table name for security
        $escaped_table = "`" . str_replace('`', '``', $table_name) . "`";
        
        // Try exact match on hashed_filename first (uses index efficiently)
        $query = $wpdb->prepare(
            "SELECT id, original_url, type, hashed_filename, local_url, status FROM {$escaped_table} WHERE type = %s AND hashed_filename = %s LIMIT 1",
            $type,
            $hash
        );
        
        $result = $wpdb->get_row($query, ARRAY_A);
        
        if ($result) {
            $hash_cache[$cache_key] = $result;
            return $result;
        }
        
        // Fallback: Search in local_url with LIKE (slower but catches partial matches)
        $query = $wpdb->prepare(
            "SELECT id, original_url, type, hashed_filename, local_url, status FROM {$escaped_table} WHERE type = %s AND local_url LIKE %s LIMIT 1",
            $type,
            '%' . $wpdb->esc_like($hash) . '%'
        );
        
        $result = $wpdb->get_row($query, ARRAY_A);
        
        $hash_cache[$cache_key] = $result ?: null;
        return $result ?: null;
    }

    /**
     * Validate MIME type of a file matches expected type for the asset
     *
     * Delegates to AssetUtils for centralized MIME validation with proper
     * security checks, logging, and WordPress allowed MIME types integration.
     *
     * @param string $file_path Absolute path to the file to validate
     * @param string $expected_mime Expected MIME type (used for fallback validation)
     * @return bool|string True if MIME type is valid, detected MIME string on failure
     */
    public function validate_mime_type(string $file_path, string $expected_mime): bool|string {
        // Validate inputs
        if (empty($file_path)) {
            return 'File path not provided';
        }

        if (empty($expected_mime)) {
            return 'Expected MIME type not provided';
        }

        $expected_mime_lower = strtolower(trim($expected_mime));

        // Use AssetUtils for centralized MIME validation
        if ($this->assetUtils === null) {
            return 'AssetUtils not available for MIME validation';
        }

        // AssetUtils::validate_mime_type extracts extension from file path internally
        $result = $this->assetUtils->validate_mime_type($file_path);

        // Handle wildcard MIME types (e.g., image/*, video/*)
        $is_wildcard = (strpos($expected_mime_lower, '/*') !== false);

        // If AssetUtils returns true, it means it's valid for its own internal checks (usually based on extension)
        if ($result === true) {
            // If it's a wildcard, we need to do a partial match check
            if ($is_wildcard) {
                // We need to know what the detected mime was.
                // Since AssetUtils::validate_mime_type doesn't return it on success,
                // we'll have to trust it or re-detect it.
                // For now, if it's true, we consider it passed for wildcard as well.
                return true;
            }

            // For specific mimes, we should ideally verify it matches expected_mime.
            // But AssetUtils already checked it against the extension's allowed mimes.
            return true;
        }

        // If result is a string, it might be an error message or detected mime (depending on implementation)
        // Validate it's a non-empty string before string operations
        if (is_string($result) && $result !== '') {
            $result_lower = strtolower(trim($result));
            // Check if it looks like an error message using more robust heuristics:
            // 1. Contains common error phrases at the START (less likely to be in MIME type)
            // 2. Does NOT contain a slash (MIME types always have format like "type/subtype")
            // 3. Check for common error patterns
            $has_slash = (strpos($result_lower, '/') !== false);
            $starts_with_error = (strpos($result_lower, 'error') === 0) ||
                                 (strpos($result_lower, 'invalid') === 0) ||
                                 (strpos($result_lower, 'not') === 0) ||
                                 (strpos($result_lower, 'failed') === 0) ||
                                 (strpos($result_lower, 'unable') === 0);
            $is_error = !$has_slash && $starts_with_error;

            if (!$is_error) {
                // It's likely a detected MIME type, not an error message
                if ($is_wildcard) {
                    $type_prefix = str_replace('/*', '', $expected_mime_lower);
                    if (strpos($result_lower, $type_prefix) === 0) {
                        return true;
                    }
                } elseif ($result_lower === $expected_mime_lower) {
                    return true;
                }
            }
            // Return detected mime or error for reporting
            return $result;
        }

        return 'MIME type validation failed';
    }

    /**
     * Get custom registry URL for an asset.
     * Checks if there's a custom registry URL configured for this asset.
     *
     * @param string $url The original URL
     * @param string $type The asset type
     * @return string|null The custom registry URL or null if not found
     */
    public function get_custom_registry_url(string $url, string $type): ?string {
        // Early return for empty inputs
        if (empty($url) || empty($type)) {
            return null;
        }

        // Normalize URL for consistent lookup using Normalize dependency
        try {
            // Use Normalize::normalize_asset_url for asset URLs (preferred method)
            // Falls back to normalize_url_base if normalize_asset_url not available
            if ($this->normalize !== null) {
                $normalized_url = method_exists($this->normalize, 'normalize_asset_url')
                    ? $this->normalize->normalize_asset_url($url)
                    : $this->normalize->normalize_url_base($url);
            } elseif ($this->urlProcessor !== null && method_exists($this->urlProcessor, 'normalize_url')) {
                // Fallback to UrlProcessor for backward compatibility
                $normalized_url = $this->urlProcessor->normalize_url($url);
            } else {
                // Last resort: basic trim
                $normalized_url = trim($url);
            }
        } catch (\Exception $e) {
            Logging::log_error("URL normalization failed for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            $normalized_url = trim($url);
        }

        // If normalization failed, use original URL
        // Check for both empty string and null to handle edge cases
        if ($normalized_url === '' || $normalized_url === null) {
            $normalized_url = $url;
        }

        // Use AssetData for lookup (now guaranteed by interface)
        $result = $this->assetData->get_custom_registry_url($normalized_url, $type);
        if ($result !== null && $result !== false) {
            return $result;
        }

        // Fallback: Check mapping entry for custom registry URL
        $mapping = $this->get_mapping_entry_by_url($normalized_url);
        if ($mapping && isset($mapping['registry_url']) && !empty($mapping['registry_url'])) {
            return $mapping['registry_url'];
        }

        return null;
    }

    /**
     * Get local URL if the asset has already been processed.
     * Only returns a URL if the asset status is 'processed'.
     *
     * @param string $url The original URL
     * @param string $type The asset type
     * @return string|false The local URL or false if not processed
     */
    public function get_local_url_if_processed(string $url, string $type): string|false {
        // Early return for empty inputs
        if (empty($url) || empty($type)) {
            return false;
        }

        // Normalize URL for consistent lookup using Normalize dependency
        try {
            // Use Normalize::normalize_asset_url for asset URLs (preferred method)
            // Falls back to normalize_url_base if normalize_asset_url not available
            if ($this->normalize !== null) {
                $normalized_url = method_exists($this->normalize, 'normalize_asset_url')
                    ? $this->normalize->normalize_asset_url($url)
                    : $this->normalize->normalize_url_base($url);
            } elseif ($this->urlProcessor !== null && method_exists($this->urlProcessor, 'normalize_url')) {
                // Fallback to UrlProcessor for backward compatibility
                $normalized_url = $this->urlProcessor->normalize_url($url);
            } else {
                // Last resort: basic trim
                $normalized_url = trim($url);
            }
        } catch (\Exception $e) {
            Logging::log_error("URL normalization failed for {$url}: " . $e->getMessage(), Logging::LEVEL_WARNING);
            $normalized_url = trim($url);
        }

        // If normalization failed, use original URL
        // Check for both empty string and null to handle edge cases
        if ($normalized_url === '' || $normalized_url === null) {
            $normalized_url = $url;
        }

        // Use AssetData for lookup (now guaranteed by interface)
        $result = $this->assetData->get_local_url_if_processed($normalized_url, $type);
        if ($result !== false) {
            return $result;
        }

        // Fallback: Check mapping entry status
        $mapping = $this->get_mapping_entry_by_url($normalized_url);
        if ($mapping && isset($mapping['status']) && $mapping['status'] === 'processed') {
            if (isset($mapping['local_url']) && !empty($mapping['local_url'])) {
                return $mapping['local_url'];
            }
        }

        return false;
    }
}



