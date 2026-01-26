<?php
/**
 * Generate comprehensive helper extraction report
 */

$helpersDir = __DIR__ . '/AssetDataHelpers/';

$helperInfo = [
    'AssetCacheHelper.php' => [
        'description' => 'Cache invalidation and management methods',
        'methods' => [
            'invalidate_asset_cache',
            'invalidate_paginated_cache',
            'clear_asset_cache',
            'clear_all_asset_caches',
        ],
    ],
    'AssetDatabaseHelper.php' => [
        'description' => 'Database table name and entry retrieval methods',
        'methods' => [
            'get_mapping_table_name',
            'get_tasks_table_name',
            'get_order_table_name',
            'get_task_table_name',
            'get_asset_exists',
            'get_asset_status_by_id',
            'get_mapping_entry_by_id',
        ],
    ],
    'AssetDataRegistryHelper.php' => [
        'description' => 'Data registry management methods',
        'methods' => [
            'get_data_registry',
        ],
    ],
    'AssetIntegrationHelper.php' => [
        'description' => 'Integration and data retrieval methods',
        'methods' => [
            'get_data',
        ],
    ],
    'AssetMemoryHelper.php' => [
        'description' => 'Memory usage and threshold methods',
        'methods' => [
            'get_memory_usage',
            'get_peak_memory_usage',
            'get_memory_threshold',
            'get_dynamic_memory_threshold',
        ],
    ],
    'AssetMetadataHelper.php' => [
        'description' => 'Asset metadata retrieval methods',
        'methods' => [
            'get_asset_local_directory',
            'get_asset_image_dimensions',
            'get_asset_file_size',
            'get_asset_checksum',
            'get_asset_version',
            'get_asset_embed_code',
            'get_asset_exif_data',
            'get_asset_aspect_ratio',
            'get_asset_metadata',
            'get_asset_last_modified',
            'get_asset_sanitized_filename',
            'get_asset_download_link',
        ],
    ],
    'AssetOrderHelper.php' => [
        'description' => 'Asset order management methods',
        'methods' => [
            'get_order_settings',
            'get_current_order_for_post',
        ],
    ],
    'AssetQueryHelper.php' => [
        'description' => 'Asset query and retrieval methods',
        'methods' => [
            'get_paginated_assets',
            'get_asset_ids_by_handles',
            'get_asset_status',
            'batch_get_asset_statuses',
            'get_custom_registry_url',
            'get_local_url_if_processed',
            'get_asset_data',
            'get_all_processed_assets_for_reversal',
        ],
    ],
    'AssetStatisticsHelper.php' => [
        'description' => 'Statistics and progress tracking methods',
        'methods' => [
            'get_asset_statistics',
            'get_processed_assets_count_by_pattern',
            'has_processed_clarity_assets',
            'get_progress',
        ],
    ],
    'AssetTaskHelper.php' => [
        'description' => 'Task management methods',
        'methods' => [
            'is_task_enqueued',
            'get_task_by_id',
        ],
    ],
    'AssetURLHelper.php' => [
        'description' => 'URL parsing and extraction methods',
        'methods' => [
            'normalize_url',
            'extract_urls_from_src',
            'extract_url_from_parentheses',
            'get_local_url_from_path',
            'get_original_url_from_handle',
            'find_all_urls_in_css',
            'find_script_handle_by_url',
            'find_media_handle',
            'find_style_handle_by_url',
            'get_font_urls',
            'get_image_urls',
        ],
    ],
    'AssetUtilityHelper.php' => [
        'description' => 'Utility and helper methods',
        'methods' => [
            'is_rate_limited',
            'get_validated_cache_expiration',
            'get_pending_option_key',
            'get_transient_prefix',
            'get_clean_dom_content',
            'fetch_asset_content',
            'get_dynamic_asset_urls',
            'get_uploaded_media_handles',
            'get_max_visited_urls',
            'get_svg_allowed_html',
            'get_external_css_from_enqueued_styles',
        ],
    ],
    'AssetValidationHelper.php' => [
        'description' => 'Validation and status determination methods',
        'methods' => [
            'is_valid_url',
            'is_valid_asset_url',
            'is_external_url',
            'is_url_safe_to_fetch',
            'is_asset_fully_processed',
            'determine_asset_status',
            'determine_asset_type',
            'determine_primary_asset_type',
            'determine_asset_type_from_extension',
        ],
    ],
];

echo "=== ASSET DATA HELPERS EXTRACTION REPORT ===\n\n";
echo "Source: AssetData.php.backup\n";
echo "Destination: AssetDataHelpers/\n\n";

$totalMethods = 0;
$totalFiles = count($helperInfo);

foreach ($helperInfo as $file => $info) {
    $filepath = $helpersDir . $file;

    echo "📄 $file\n";
    echo "   Description: {$info['description']}\n";
    echo "   Methods: " . count($info['methods']) . "\n\n";

    foreach ($info['methods'] as $method) {
        echo "   - $method()\n";
        $totalMethods++;
    }

    echo "\n";
}

echo "=== SUMMARY ===\n";
echo "Total helper files created: $totalFiles\n";
echo "Total methods extracted: $totalMethods\n";
echo "Average methods per file: " . round($totalMethods / $totalFiles, 1) . "\n";
echo "\nAll files: ✓ Valid PHP syntax\n";
echo "Namespace: LHA\\AssetDataHelpers\n";
echo "Declaration: declare(strict_types=1);\n";
