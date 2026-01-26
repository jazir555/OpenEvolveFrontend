<?php

namespace LHA;

use LHA\AssetActionHandler;
use LHA\AssetHtmlParser;
use LHA\Database;
use LHA\Logging;

/**
 * Class Replace
 * Handles finding and replacing external asset URLs within content
 * (HTML, CSS, JS) with their locally hosted counterparts. Ensures robustness,
 * security, performance, and adheres to enterprise production standards.
 *
 * Relies on external classes (potentially within a specific framework like WordPress or a custom setup):
 * - LHA\Logging: For centralized logging.
 * - Urls: For checking if a URL is external and validating URLs.
 * - Getdata: For retrieving the asset map, determining asset types, and getting local URLs.
 * - normalize: For resolving relative URLs to absolute URLs.
 * - SelfHost: Potentially used for direct synchronous downloads.
 * - tasks: For enqueueing background tasks.
 * - GetOption: For retrieving configuration settings.
 */
class Replace implements \LHA\Interfaces\ReplaceInterface {

    /**
     * Logger instance
     * @var \LHA\Interfaces\LoggerInterface
     */
    private \LHA\Interfaces\LoggerInterface $logger;

    /**
     * URL Processor instance
     * @var \LHA\Interfaces\UrlProcessorInterface
     */
    private \LHA\Interfaces\UrlProcessorInterface $urlProcessor;

    /**
     * GetData instance
     * @var \LHA\Interfaces\GetdataInterface
     */
    private \LHA\Interfaces\GetdataInterface $getdata;

    /**
     * Normalize instance
     * @var \LHA\Interfaces\NormalizeInterface
     */
    private \LHA\Interfaces\NormalizeInterface $normalize;

    /**
     * SelfHost instance
     * @var \LHA\Interfaces\SelfHostInterface
     */
    private \LHA\Interfaces\SelfHostInterface $selfHost;

    /**
     * GetOption instance
     * @var \LHA\Interfaces\GetOptionInterface
     */
    private \LHA\Interfaces\GetOptionInterface $options;

    /**
     * TaskQueue instance
     * @var \LHA\Interfaces\TaskQueueInterface
     */
    private \LHA\Interfaces\TaskQueueInterface $tasks;

    /**
     * File lock instance
     * @var \LHA\Interfaces\LockInterface
     */
    private \LHA\Interfaces\LockInterface $lock;

    /**
     * AssetData instance for asset data operations
     * @var \LHA\Interfaces\AssetDataInterface
     */
    private \LHA\Interfaces\AssetDataInterface $assetData;

    /**
     * Cache instance for asset map caching
     * @var \LHA\Interfaces\CacheInterface|null
     */
    private ?\LHA\Interfaces\CacheInterface $cache = null;

    /**
     * AssetValidator instance for URL validation
     * @var \LHA\Interfaces\AssetValidatorInterface|null
     */
    private ?\LHA\Interfaces\AssetValidatorInterface $assetValidator = null;

    /**
     * Database instance for database operations
     * @var \LHA\Interfaces\DatabaseInterface|null
     */
    private ?\LHA\Interfaces\DatabaseInterface $database = null;

    /**
     * Extract instance for URL extraction operations (srcset, CSS dependencies, etc.)
     * @var \LHA\Interfaces\ExtractInterface|null
     */
    private ?\LHA\Interfaces\ExtractInterface $extract = null;

    /**
     * Cached regex patterns for performance
     * @var array<string, array>
     */
    private static array $pattern_cache = [];

    /**
     * Constructor
     * 
     * @param \LHA\Interfaces\LoggerInterface $logger
     * @param \LHA\Interfaces\UrlProcessorInterface $urlProcessor
     * @param \LHA\Interfaces\GetdataInterface $getdata
     * @param \LHA\Interfaces\NormalizeInterface $normalize
     * @param \LHA\Interfaces\SelfHostInterface $selfHost
     * @param \LHA\Interfaces\GetOptionInterface $options
     * @param \LHA\Interfaces\TaskQueueInterface $tasks
     * @param \LHA\Interfaces\LockInterface $lock
     * @param \LHA\Interfaces\AssetDataInterface $assetData
     * @param \LHA\Interfaces\CacheInterface|null $cache Optional cache service for asset map
     * @param \LHA\Interfaces\AssetValidatorInterface|null $assetValidator Optional asset validator service
     * @param \LHA\Interfaces\DatabaseInterface|null $database Optional database service
     * @param \LHA\Interfaces\ExtractInterface|null $extract Optional extract service for srcset parsing
     */
    public function __construct(
        \LHA\Interfaces\LoggerInterface $logger,
        \LHA\Interfaces\UrlProcessorInterface $urlProcessor,
        \LHA\Interfaces\GetdataInterface $getdata,
        \LHA\Interfaces\NormalizeInterface $normalize,
        \LHA\Interfaces\SelfHostInterface $selfHost,
        \LHA\Interfaces\GetOptionInterface $options,
        \LHA\Interfaces\TaskQueueInterface $tasks,
        \LHA\Interfaces\LockInterface $lock,
        \LHA\Interfaces\AssetDataInterface $assetData,
        ?\LHA\Interfaces\CacheInterface $cache = null,
        ?\LHA\Interfaces\AssetValidatorInterface $assetValidator = null,
        ?\LHA\Interfaces\DatabaseInterface $database = null,
        ?\LHA\Interfaces\ExtractInterface $extract = null
    ) {
        $this->logger = $logger;
        $this->urlProcessor = $urlProcessor;
        $this->getdata = $getdata;
        $this->normalize = $normalize;
        $this->selfHost = $selfHost;
        $this->options = $options;
        $this->tasks = $tasks;
        $this->lock = $lock;
        $this->assetData = $assetData;
        $this->cache = $cache;
        $this->assetValidator = $assetValidator;
        $this->database = $database;
        $this->extract = $extract;
    }

    /**
     * Centralized logging function. Uses injected logger.
     * Ensures log messages are consistent and context-rich.
     * Dispatches to appropriate Logging method based on level.
     *
     * @param string      $message       The log message. Cannot be empty.
     * @param string      $level         Log level (e.g., 'debug', 'info', 'notice', 'warning', 'error'). Defaults to 'notice'.
     * @param string|null $context_type  Type of context (e.g., 'img', 'dependency-resolution'). Defaults to null.
     * @param string|null $original_url  Original URL related to the log message, if applicable. Defaults to null.
     * @param string|null $local_url     Local URL related to the log message, if applicable. Defaults to null.
     */
    public function log_message(string $message, string $level = 'notice', ?string $context_type = null, ?string $original_url = null, ?string $local_url = null): void
    {
        // Early exit for empty messages
        if ($message === '' || $message === '0') {
            return;
        }

        // Sanitize message
        $safe_message = htmlspecialchars($message, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');

        // Sanitize context values
        $safe_context_type = $context_type !== null ? htmlspecialchars($context_type, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8') : null;

        // Safely normalize URLs - fall back to htmlspecialchars if normalization fails
        try {
            $safe_original_url = $original_url !== null
                ? htmlspecialchars($this->urlProcessor->normalize_url($original_url), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8')
                : null;
        } catch (\Throwable $e) {
            $safe_original_url = $original_url !== null ? htmlspecialchars($original_url, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8') : null;
        }

        try {
            $safe_local_url = $local_url !== null
                ? htmlspecialchars($this->urlProcessor->normalize_url($local_url), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8')
                : null;
        } catch (\Throwable $e) {
            $safe_local_url = $local_url !== null ? htmlspecialchars($local_url, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8') : null;
        }

        // Dispatch to LHA Logging class methods based on level
        switch (strtolower($level)) {
            case 'debug':
                Logging::log_debug($safe_message, $safe_context_type, $safe_original_url, $safe_local_url);
                break;
            case 'info':
                Logging::log_info($safe_message, $safe_context_type, $safe_original_url, $safe_local_url);
                break;
            case 'warning':
                Logging::log_warning($safe_message, $safe_context_type, $safe_original_url, $safe_local_url);
                break;
            case 'error':
                Logging::log_error($safe_message, $safe_context_type, $safe_original_url, $safe_local_url);
                break;
            case 'notice':
            default:
                if (method_exists('\LHA\Logging', 'log_notice')) {
                    Logging::log_notice($safe_message, $safe_context_type, $safe_original_url, $safe_local_url);
                } else {
                    // Fallback to info for non-critical notices if log_notice is missing
                    Logging::log_info($safe_message, $safe_context_type, $safe_original_url, $safe_local_url);
                }
                break;
        }
    }

    /**
     * Replace external <a href="..."> references with locally hosted versions for downloadable files.
     * Checks allowed file extensions before attempting replacement. Robust and secure.
     *
     * @param string $content The HTML content that may contain <a> tags. Must not be empty.
     * @return string The modified content with local anchor URLs if available. Returns original on error or no matches.
     */
    public function replace_content_anchor_urls(string $content): string
    {
        // Early exit for performance if content is empty or lacks relevant patterns.
        if (empty($content) || stripos($content, '<a') === false || stripos($content, ' href=') === false) {
            return $content;
        }
        $original_content = $content; // For fallback on critical error

        // Define specific parameters for anchor tag replacement.
        $attribute_name = 'href';
        $tag_name = 'a';
        $asset_type_context = 'file'; // Context for Getdata (should determine specific type if possible)

        // Get allowed extensions, filterable if in WordPress context.
        $default_exts = [
            'pdf', 'doc', 'docx', 'zip', 'rar', 'tar', 'gz', '7z', // Docs & Archives
            'xls', 'xlsx', 'ppt', 'pptx', 'odt', 'ods', 'odp',     // Spreadsheets & Presentations
            'txt', 'rtf', 'csv', 'xml', 'json',                    // Text & Data
            'epub', 'mobi', 'key', 'pages', 'numbers'              // Specific formats
        ];
        $allowed_file_exts = function_exists('apply_filters')
            ? apply_filters('self_host_assets_anchor_extensions', $default_exts)
            : $default_exts;

        // Ensure $allowed_file_exts is a valid array. Sanitize entries.
        if (!is_array($allowed_file_exts)) {
            $this->log_message("Invalid 'self_host_assets_anchor_extensions' filter value. Expected array, got " . gettype($allowed_file_exts) . ". Using default extensions.", 'warning', $tag_name);
            $allowed_file_exts = $default_exts;
        } else {
            // Sanitize extensions: lowercase, trim, remove empty, ensure uniqueness
            $allowed_file_exts = array_unique(array_filter(array_map(function($ext) {
                return is_string($ext) ? strtolower(trim($ext)) : null;
            }, $allowed_file_exts)));
        }
        if (empty($allowed_file_exts)) {
             $this->log_message("Anchor replacement disabled: allowed file extension list is empty after sanitization.", 'warning', $tag_name);
            return $original_content;
        }

        // Use preg_replace_callback for fine-grained control over each anchor tag.
        // Regex uses named groups for safe reconstruction. Case-insensitive tag/attribute match.
        // Improved regex to handle attributes containing '>' within quotes. Added length limits to prevent ReDoS.
        $regex_pattern = '/<(?<tag_name>' . preg_quote($tag_name, '/') . ')\b(?<before_href>(?:[^>"\']|"[^"]*"|\'[^\']*\'){0,2000}?)\s+' . preg_quote($attribute_name, '/') . '\s*=\s*(?<quote>["\'])(?<url>.+?)\k<quote>(?<after_href>(?:[^>"\']|"[^"]*"|\'[^\']*\'){0,2000})>/i';

        // Dependencies are injected, no need to check class existence

        $modified_content = preg_replace_callback($regex_pattern, function ($matches) use ($attribute_name, $tag_name, $asset_type_context, $allowed_file_exts) {
            $original_tag = $matches[0];
            // Use null coalescing operator for safety
            $original_url_encoded = $matches['url'] ?? '';
            $original_url = html_entity_decode($original_url_encoded, ENT_QUOTES | ENT_HTML5);
            $matched_tag_case = $matches['tag_name'] ?? $tag_name; // Fallback to lower case

            if (empty($original_url) || trim($original_url) === '') { return $original_tag; } // Skip empty URLs

            // 1. Validate URL Structure (allow relative URLs initially if they are processed by UrlProcessor::is_external_url appropriately)
            $is_valid_structure = $this->urlProcessor->is_valid_url($original_url) || (strpos($original_url, '/') === 0 && strpos($original_url, '//') !== 0) || (strpos($original_url, '//') === 0);
             if (!$is_valid_structure) {
                $this->log_message("Skipping potentially invalid URL structure in anchor href: {$original_url}", 'debug', $tag_name, $original_url);
                return $original_tag;
            }

            // 2. Check if External (use injected URL processor)
            try {
                $is_external = $this->urlProcessor->is_external_url($original_url);
            } catch (\Throwable $e) {
                $this->log_message("Error checking if URL is external: " . $e->getMessage(), 'error', $tag_name, $original_url);
                return $original_tag; // Skip on error
            }
            if (!$is_external) { return $original_tag; } // Skip internal URLs

            // 3. Check File Extension (only for absolute URLs or paths where extension can be determined)
            $parsed_url_path = parse_url($original_url, PHP_URL_PATH);
            $file_ext = ($parsed_url_path !== null && $parsed_url_path !== '')
                ? strtolower(pathinfo($parsed_url_path, PATHINFO_EXTENSION))
                : '';

            // Trim query strings/fragments from extension if pathinfo includes them (unlikely but possible)
            // Note: strtok() returns false if delimiter not in string, or empty string if token is empty
            $file_ext_trimmed = strtok($file_ext, '?#');
            // Need to check for both false and empty string (empty string means pathinfo returned just delimiter)
            $file_ext = ($file_ext_trimmed !== false && $file_ext_trimmed !== '') ? $file_ext_trimmed : $file_ext;

            if (empty($file_ext) || !in_array($file_ext, $allowed_file_exts, true)) {
                $this->log_message("Skipping anchor URL with disallowed/missing extension '{$file_ext}': {$original_url}", 'debug', $tag_name, $original_url);
                return $original_tag; // Skip non-files or disallowed extensions
            }

            // 4. Determine Asset Type (use injected getdata)
            $asset_type = $asset_type_context;
            try {
                $determined_type = $this->getdata->determine_primary_asset_type($original_url, $asset_type_context);
                // Use determined type only if it's a non-empty string
                $asset_type = (!empty($determined_type) && is_string($determined_type)) ? $determined_type : $asset_type_context;
            } catch (\Throwable $e) {
                $this->log_message("Error determining asset type for {$original_url}: " . $e->getMessage(), 'error', $tag_name, $original_url);
                // Continue with default asset type context
            }

            // 5. Process asset URL (check status, get local URL, handle errors)
            $local_url = $this->process_asset_url($original_url, $asset_type, 'anchor-replacement');

            if ($local_url === false) {
                // Asset is disabled via toggle, keep original external URL
                $this->log_message(
                    "Asset is disabled via toggle (status: ignored), keeping original URL: {$original_url}",
                    'debug', 'anchor-replacement', $original_url
                );
                return $original_tag;
            }

            if ($local_url === null) {
                // Error occurred or asset is pending, keep original and ensure task is enqueued
                $this->enqueue_dependency_localization_task($original_url, $asset_type);
                return $original_tag;
            }

            // Only proceed with replacement if asset is not ignored (i.e. enabled)
            if ($local_url && is_string($local_url) && trim($local_url) !== '') {
                // Sanitize thoroughly before output using Normalize for consistency
                try {
                    $sanitized_local_url = $this->urlProcessor->normalize_url($local_url);
                } catch (\Throwable $e) {
                    $this->log_message('Failed to normalize local URL for anchor replacement: ' . $e->getMessage(), 'warning', $tag_name, $original_url, $local_url);
                    return $original_tag;
                }

                // Validate the sanitized URL again to prevent filter bypass or invalid schemes
                // Allow absolute URLs OR root-relative paths (starting with /)
                $is_absolute = $this->urlProcessor->is_valid_url($sanitized_local_url);
                // Check for root-relative: starts with / but not with //
                $is_root_relative = (strpos($sanitized_local_url, '/') === 0 && strpos($sanitized_local_url, '//') === false);

                if (empty($sanitized_local_url) || (!$is_absolute && !$is_root_relative)) {
                    $this->log_message('Failed to sanitize/validate local URL for anchor replacement. Check URL scheme and format.', 'warning', $tag_name, $original_url, $local_url);
                    return $original_tag;
                }

                // Encode for safe insertion into the attribute value
                // Use esc_attr for attribute values (esc_html is for content between tags)
                $encoded_local_url = function_exists('esc_attr') ? esc_attr($sanitized_local_url) : htmlspecialchars($sanitized_local_url, ENT_QUOTES | ENT_HTML5, 'UTF-8');

                // Safe reconstruction using captured parts
                // Sanitize tag name to prevent HTML injection
                $safe_tag_case = preg_replace('/[^a-zA-Z0-9]/', '', $matched_tag_case);
                $safe_before_href = htmlspecialchars(rtrim($matches['before_href'] ?? ''), ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
                $safe_after_href = htmlspecialchars(ltrim($matches['after_href'] ?? ''), ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');

                $modified_tag = sprintf(
                    '<%s%s %s=%s%s%s%s>',
                    $safe_tag_case,
                    $safe_before_href,
                    $attribute_name,
                    $matches['quote'] ?? '"',
                    $encoded_local_url,
                    $matches['quote'] ?? '"',
                    $safe_after_href
                );

                $log_text_format = function_exists('__') ? __("Replaced external anchor URL in <%1\$s> tag. Original: %2\$s, Local: %3\$s", 'self-host-assets') : "Replaced external anchor URL in <%1\$s> tag. Original: %2\$s, Local: %3\$s";
                $this->log_message(
                    sprintf($log_text_format, $matched_tag_case, $original_url, $sanitized_local_url),
                    'notice', $tag_name, $original_url, $sanitized_local_url // Log sanitized local URL
                );
                return $modified_tag;
            } else {
                // Localization failed or pending. Log and ensure task is enqueued if necessary.
                if ($local_url !== null) { // Avoid double logging if Getdata returned null (implying pending)
                    $this->handle_localization_failure($original_url, $matched_tag_case, $asset_type, $original_tag);
                }
                // Ensure task is enqueued (safe if already done by Getdata method)
                $this->enqueue_dependency_localization_task($original_url, $asset_type);
                return $original_tag;
            }
        }, $content, -1, $count); // Use $count to potentially log total replacements if needed

        // Check for regex execution errors
        if ($modified_content === null) {
             @preg_match('/./', ''); // Clear last error reliably
             $last_error_code = preg_last_error();
             if ($last_error_code !== PREG_NO_ERROR) {
                $error_msg = function_exists('preg_last_error_msg') ? preg_last_error_msg() : "Code {" . $last_error_code . "}";
                $this->log_message("preg_replace_callback error during anchor ({$tag_name}/{$attribute_name}) replacement: " . $error_msg, 'error', $tag_name);
                return $original_content; // Return original content on critical error
             }
             // If null but no error, could be memory issue or other edge case. Still safer to return original.
             $this->log_message("preg_replace_callback returned null without explicit error during anchor replacement. Potential resource issue?", 'error', $tag_name);
             return $original_content;
        }

        return (string) $modified_content;
    }

    /**
     * Get asset data for a specific URL or the entire map (deprecated).
     *
     * @param string|null $url Optional URL to look up a single asset.
     * @return array<string, array>|array The asset data for the URL, or empty array if no URL provided (to avoid memory exhaustion).
     */
    public function get_asset_map(?string $url = null): array {
        if ($url !== null && !empty($url)) {
            return $this->get_individual_asset_data($url);
        }

        // Returning the full map is deprecated due to memory exhaustion risks.
        // Production code should use individual lookups.
        return [];
    }

    /**
     * Helper: Get individual asset data in the format expected by legacy asset_map consumers.
     * 
     * @param string $url The original URL to look up.
     * @return array Found asset data or empty array.
     */
    private function get_individual_asset_data(string $url): array {
        // Try to get processed local URL
        // We check for 'js', 'css', and 'image' types as common defaults
        $types = ['js', 'css', 'image', 'font', 'video', 'audio', 'file'];
        
        foreach ($types as $type) {
            try {
                $local_url = $this->getdata->get_local_url_if_processed($url, $type);
                if ($local_url && is_string($local_url)) {
                    // Return in the format expected by some parts of the codebase
                    // format: [original_url => local_url] (v5 format)
                    return [$url => $local_url];
                }
            } catch (\Throwable $e) {
                continue;
            }
        }
        
        return [];
    }
    
    /**
     * Invalidate the asset map cache. Call this when assets are added, updated, or deleted.
     * This ensures the cache stays fresh and URL replacements use current data.
     *
     * @return void
     */
    public function invalidate_asset_map_cache(): void {
        // Delete all cache versions to ensure clean slate
        $keys = ['lha_asset_map_v3', 'lha_asset_map_v4', 'lha_asset_map_v5'];

        if ($this->cache !== null) {
            foreach ($keys as $key) {
                $this->cache->delete($key);
            }
        } elseif (function_exists('delete_transient')) {
            foreach ($keys as $key) {
                delete_transient($key);
            }
        }
    }



    /**
     * Defines the regex patterns for dependency URL replacement in JS and CSS.
     * Patterns use '#' as the delimiter and include capture groups for reconstruction.
     * Ensures security by using properly escaped URL.
     *
     * OPTIMIZED: Added pattern caching to avoid repeated pattern generation
     *
     * @param string $escaped_dep_url The dependency URL, MUST be escaped via preg_quote($url, '#'). Cannot be empty.
     * @param string $local_url       Local URL (Not directly used in pattern, context for caller).
     * @param string $type            Asset type ('js', 'css'). Must be 'js' or 'css'.
     * @return array<string> Array of regex patterns. Returns empty array for unknown types or invalid input.
     */
    public function get_replacement_patterns(string $escaped_dep_url, string $local_url, string $type): array
    {
        // Early exit for empty URLs
        if ($escaped_dep_url === '') {
            $this->log_message('Empty escaped dependency URL provided to get_replacement_patterns. Cannot generate patterns.', 'warning', 'regex-pattern-gen', null, null);
            return [];
        }
        
        // Validate type early
        if ($type !== 'js' && $type !== 'css') {
            $this->log_message("Unsupported type '{$type}' provided to get_replacement_patterns. Must be 'js' or 'css'.", 'warning', 'regex-pattern-gen', null, null);
            return [];
        }

        // Check cache first for performance
        // Use '::' separator to avoid potential collision with URLs containing ':'
        $cache_key = $type . '::' . $escaped_dep_url;
        if (isset(self::$pattern_cache[$cache_key])) {
            return self::$pattern_cache[$cache_key];
        }

        $patterns = [];
        if ($type === 'js') {
            // Match ES6 'import ... from "URL"' or 'import ... from \'URL\'' (static)
            // Improved robustness: handles whitespace, various import forms. Case-insensitive, dotall.
            // Captures: 1=prefix (import...from), 2=quote, 3=closing quote
            $patterns[] = '#(import(?:["\']\s*[^"\']+\s*["\']|(?:(?:\s*[\w*{}\s,:[\\\]]+|\s*\*\s*(?:as\s+\w+)?)\s+from))\s*)(["\'])' . $escaped_dep_url . '(["\'])#is';
            // Match dynamic 'import("URL")' or 'import(\'URL\')'. Case-insensitive. Added whitespace tolerance.
            // Captures: 1=prefix (import()), 2=quote, 3=closing paren+quote
            $patterns[] = '#(import\s*\(\s*)(["\'])' . $escaped_dep_url . '(["\']\s*\))#i';
            // Match CommonJS 'require("URL")' or 'require(\'URL\')'. Case-insensitive. Added whitespace tolerance.
            // Captures: 1=prefix (require()), 2=quote, 3=closing paren+quote
            $patterns[] = '#(require\s*\(\s*)(["\'])' . $escaped_dep_url . '(["\']\s*\))#i';
        } else { // $type === 'css'
            // Match url(URL), url('URL'), url("URL") - excluding data: URIs. Case-insensitive. Non-greedy URL match.
            // Captures: 1=prefix(url...), 2=optional quote, 3=closing optional quote/paren
            // Ensure it doesn't match already replaced URLs if they contain the original URL string? This is tricky. Assume simple replacement for now.
            $patterns[] = '#(url\(\s*)(["\']?)(?!data:)'. $escaped_dep_url . '(["\']?\s*\))#i';
            // Match @import "URL"; or @import 'URL'; or @import url(URL); - Case-insensitive. Non-greedy URL capture.
            // Handles optional media queries after URL before semicolon. More robust ending capture.
            // Captures: 1=prefix(@import...), 2=optional url( prefix, 3=optional quote
            // Non-capturing group consumes the closing quote/paren/url-suffix.
            // 4=rest of string (media queries, semicolon)
            $patterns[] = '#(@import\s+)(?:(url\(\s*)?(["\']?)?)?' . $escaped_dep_url . '(?:(?(3)\3)(?(2)\s*\)))((?:\s*[^;]*?);)#i';
        }
        
        // Cache patterns for reuse (significant performance gain for repeated URLs)
        self::$pattern_cache[$cache_key] = $patterns;
        
        // Limit cache size to prevent memory issues (keep last 100 patterns)
        if (count(self::$pattern_cache) > 100) {
            self::$pattern_cache = array_slice(self::$pattern_cache, -100, null, true);
        }
        
        return $patterns;
    }

    /**
     * Modify style source URL if a local version exists in the asset map.
     * Hooked to 'style_loader_src'. Robust and secure.
     *
     * OPTIMIZED: Direct URL lookup with query string fallback for WordPress version strings
     *
     * @param string $src The source URL of the style. May be empty or invalid.
     * @param string $handle The style's registered handle. Must not be empty.
     * @return string Modified source URL or original URL.
     */
    public function modify_style_src(string $src, string $handle): string {
        // Bail early if handle or src is empty/invalid
        if (empty($handle) || empty($src) || !is_string($src)) {
            return $src;
        }

        // Basic check if $src looks like a plausible URL or path
        $is_valid_url = $this->urlProcessor->is_valid_url($src);
        if (!$is_valid_url && strpos($src, '//') !== 0 && strpos($src, '/') !== 0) {
            return $src;
        }

        // Check if self-hosting is enabled
        if (!$this->is_selfhosting_enabled()) {
            return $src;
        }

        // OPTIMIZED: Use individual cached lookups via GetData instead of loading the entire asset map.
        // This handles URL variations and Object Caching internally, avoiding the "memory bomb" on large sites.
        try {
            $local_src = $this->getdata->get_local_url_if_processed($src, 'css');

            if ($local_src && is_string($local_src) && trim($local_src) !== '') {
                // Ensure local URL is normalized for consistent output
                $sanitized_local_src = $this->urlProcessor->normalize_url($local_src);

                if (!empty($sanitized_local_src)) {
                    return $sanitized_local_src;
                }
            }
        } catch (\Throwable $e) {
            $this->log_message("Error during modify_style_src lookup: " . $e->getMessage(), 'debug', 'style-filter', $src, null);
        }

        // Return the original source if no valid local version found
        return $src;
    }

    /**
     * Modify script source URL if a local version exists in the asset map.
     * Hooked to 'script_loader_src'. Robust and secure.
     *
     * OPTIMIZED: Direct URL lookup with query string fallback for WordPress version strings
     *
     * @param string $src The source URL of the script. May be empty or invalid.
     * @param string $handle The script's registered handle. Must not be empty.
     * @return string Modified source URL or original URL.
     */
    public function modify_script_src(string $src, string $handle): string {
        // Bail early if handle or src is empty/invalid
        if (empty($handle) || empty($src) || !is_string($src)) {
            return $src;
        }

        // Basic check if $src looks like a plausible URL or path
        $is_valid_url = $this->urlProcessor->is_valid_url($src);
        if (!$is_valid_url && strpos($src, '//') !== 0 && strpos($src, '/') !== 0) {
            return $src;
        }

        // Check if self-hosting is enabled
        if (!$this->is_selfhosting_enabled()) {
            return $src;
        }

        // OPTIMIZED: Use individual cached lookups via GetData instead of loading the entire asset map.
        // This avoids loading thousands of assets into memory for a single script handle lookup.
        try {
            $local_src = $this->getdata->get_local_url_if_processed($src, 'js');

            if ($local_src && is_string($local_src) && trim($local_src) !== '') {
                // Ensure local URL is normalized for consistent output
                $sanitized_local_src = $this->urlProcessor->normalize_url($local_src);

                if (!empty($sanitized_local_src)) {
                    return $sanitized_local_src;
                }
            }
        } catch (\Throwable $e) {
            $this->log_message("Error during modify_script_src lookup: " . $e->getMessage(), 'debug', 'script-filter', $src, null);
        }

        // Return the original source if no valid local version found
        return $src;
    }

    /**
     * Check if self-hosting is enabled in plugin settings.
     * Uses static cache for performance within a single request.
     * Note: lha_selfhost_settings is a separate WordPress option, not part of lha_options,
     * so we use get_option directly here (GetOption only manages lha_options).
     *
     * @return bool True if self-hosting is enabled, false otherwise.
     */
    private function is_selfhosting_enabled(): bool {
        static $enabled = null;
        
        if ($enabled !== null) {
            return $enabled;
        }

        // lha_selfhost_settings is a separate option from lha_options
        // GetOption class only manages lha_options, so we use get_option directly
        $selfhost_settings = function_exists('get_option') ? get_option('lha_selfhost_settings', []) : [];
        $enabled = isset($selfhost_settings['enable_asset_hosting']) &&
                   ($selfhost_settings['enable_asset_hosting'] === '1' || 
                    $selfhost_settings['enable_asset_hosting'] === 1 || 
                    $selfhost_settings['enable_asset_hosting'] === true);
        
        return $enabled;
    }

    /**
     * Handle localization failure for assets. Delegates to AssetActionHandler for centralized handling.
     * Indicates immediate replacement failed, possibly pending async task.
     *
     * @param string $original_url The original external URL that failed immediate localization. Must be a non-empty string.
     * @param string $tag_name     HTML tag name or context type (e.g., 'img', 'css-dependency'). Must be a non-empty string.
     * @param string $asset_type   Type of asset (e.g., 'image', 'css', 'font'). Must be a non-empty string.
     * @param string $original_tag Optional original HTML tag string (for context). Defaults to empty string.
     */
    public function handle_localization_failure(string $original_url, string $tag_name, string $asset_type, string $original_tag = ''): void {
        // Input validation
        if (trim($original_url) === '' || trim($tag_name) === '' || trim($asset_type) === '') {
             $this->log_message('Invalid empty parameter provided to handle_localization_failure.', 'warning', 'localization-failure', null, null);
             return;
        }

        // Delegate to AssetActionHandler for centralized localization failure handling
        // This ensures consistent behavior across all asset processing paths
        AssetActionHandler::handle_localization_failure($original_url, $tag_name, $asset_type, $original_tag);
    }

    /**
     * Generic function to replace external URLs in specified HTML tag attributes.
     * Robust, secure, and uses Getdata for local URL retrieval with async fallback.
     *
     * @param string $content The HTML content to process. Must not be empty.
     * @param string $attribute_name The attribute name (e.g., 'src', 'href'). Must not be empty.
     * @param string $tag_name The HTML tag name (e.g., 'img', 'script'). Case-insensitive match. Must not be empty.
     * @param string $asset_type The asset type (e.g., 'image', 'js'). Must not be empty.
     * @param string|null $rel_attribute_value Optional: For <link>, required 'rel' value (e.g., 'stylesheet'). Case-insensitive.
     * @return string The modified HTML content. Returns original on critical error or no matches.
     */
    public function replace_content_url_attributes(string $content, string $attribute_name, string $tag_name, string $asset_type, ?string $rel_attribute_value = null): string
    {
        // Validate inputs
        if (empty($content) || empty($attribute_name) || empty($tag_name) || empty($asset_type)) {
            // Log if essential parameters are missing
            if (empty($attribute_name) || empty($tag_name) || empty($asset_type)) {
                $this->log_message('Missing required parameters (attribute, tag, or asset type) for replace_content_url_attributes.', 'error', 'attribute-replace-init', null, null);
            }
            return $content; // Return original if content or required params empty
        }

        // Early exit for performance if specific tags/attributes unlikely to be present.
        // Using stripos for case-insensitivity and speed.
        if (stripos($content, '<' . $tag_name) === false || stripos($content, $attribute_name . '=') === false) {
            return $content;
        }
        $original_content = $content; // For fallback on critical error

        // Robust Regex using named capture groups for safe reconstruction. Case-insensitive tag/attribute name.
        // Improved regex to handle attributes containing '>' within quotes. Added length limits to prevent ReDoS.
        $regex_pattern = '/<(?<tag_name>' . preg_quote($tag_name, '/') . ')\b(?<before_attr>(?:[^>"\']|"[^"]*"|\'[^\']*\'){0,2000}?)\s+' . preg_quote($attribute_name, '/') . '\s*=\s*(?<quote>["\'])(?<url>.+?)\k<quote>(?<after_attr>(?:[^>"\']|"[^"]*"|\'[^\']*\'){0,2000})>/i';

        // Dependencies are injected, no need to check class existence

        $modified_content = preg_replace_callback($regex_pattern, function ($matches) use ($attribute_name, $tag_name, $asset_type, $rel_attribute_value) {
            $original_tag = $matches[0];
            $original_url_encoded = $matches['url'] ?? '';
            $original_url = html_entity_decode($original_url_encoded, ENT_QUOTES | ENT_HTML5);
            $matched_tag_case = $matches['tag_name'] ?? $tag_name;

            if (empty($original_url) || trim($original_url) === '') { return $original_tag; }

            // Special handling for <link rel="stylesheet">
            if ($tag_name === 'link' && $rel_attribute_value !== null) {
                // Combine all attributes for checking 'rel'. Case-insensitive check. Trim spaces around value.
                $attributes_string = ($matches['before_attr'] ?? '') . ' ' . ($matches['after_attr'] ?? '');
                // Ensure rel value matching handles potential variations like ' stylesheet ' within quotes.
                // Match rel="stylesheet", rel=\'stylesheet\', rel=stylesheet (less common but possible)
                // Use word boundary \b around rel= ? Not reliable. Use whitespace \s checks.
                $escaped_rel_value = preg_quote(trim($rel_attribute_value), '/');
                if (!preg_match('/\srel\s*=\s*(["\']?)\s*' . $escaped_rel_value . '\s*\1/i', ' ' . $attributes_string)) { // Prepend space to ensure matching start
                    return $original_tag; // Skip if rel attribute doesn't match
                }
            }

            // 1. Validate URL Structure (basic check)
            $is_valid_structure = $this->urlProcessor->is_valid_url($original_url) || (strpos($original_url, '/') === 0 && strpos($original_url, '//') !== 0) || (strpos($original_url, '//') === 0);
             if (!$is_valid_structure) {
                $this->log_message("Skipping invalid URL structure in {$matched_tag_case} {$attribute_name}: {$original_url}", 'debug', $tag_name, $original_url, null);
                return $original_tag;
            }

            // 2. Check if External (use injected URL processor)
            try {
                $is_external = $this->urlProcessor->is_external_url($original_url);
            } catch (\Throwable $e) {
                $this->log_message("Error checking if URL is external for {$attribute_name}: " . $e->getMessage(), 'error', $tag_name, $original_url, null);
                return $original_tag;
            }
            if (!$is_external) { return $original_tag; }

            // Process the asset URL (check status, get local URL, handle errors)
            $local_url = $this->process_asset_url($original_url, $asset_type, 'attribute-replacement');

            if ($local_url === false) {
                // Asset is disabled via toggle, keep original external URL
                return $original_tag;
            }

            if ($local_url === null) {
                // Error occurred or asset is pending, keep original and ensure task is enqueued
                $this->enqueue_dependency_localization_task($original_url, $asset_type);
                return $original_tag;
            }

            // 4. Replace if Local URL Available and Valid
            if ($local_url && is_string($local_url) && trim($local_url) !== '') {
                // Sanitize thoroughly before output
                $sanitized_local_url = function_exists('esc_url_raw') ? esc_url_raw($local_url, ['http', 'https']) : filter_var($local_url, FILTER_SANITIZE_URL);
                 $is_valid_sanitized_url = $this->urlProcessor->is_valid_url($sanitized_local_url);
                 if (empty($sanitized_local_url) || !$is_valid_sanitized_url) {
                    $this->log_message('Failed to sanitize/validate local URL during attribute replacement.', 'warning', $tag_name, $original_url, $local_url);
                    return $original_tag;
                }
                 // Encode for safe insertion into the attribute value
                 // Use esc_attr for attribute values (esc_html is for content between tags)
                $encoded_local_url = function_exists('esc_attr') ? esc_attr($sanitized_local_url) : htmlspecialchars($sanitized_local_url, ENT_QUOTES | ENT_HTML5, 'UTF-8');

                // Safe reconstruction with HTML injection prevention
                $safe_tag_case = preg_replace('/[^a-zA-Z0-9]/', '', $matched_tag_case);
                $safe_before_attr = htmlspecialchars(rtrim($matches['before_attr'] ?? ''), ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
                $safe_after_attr = htmlspecialchars(ltrim($matches['after_attr'] ?? ''), ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');

                $modified_tag = sprintf(
                    '<%s%s %s=%s%s%s%s>',
                    $safe_tag_case,
                    $safe_before_attr,
                    $attribute_name,
                    $matches['quote'] ?? '"',
                    $encoded_local_url,
                    $matches['quote'] ?? '"',
                    $safe_after_attr
                );

                $log_message = sprintf(
                    "Replaced external %s URL in <%s> tag. Original: %s, Local: %s",
                    $asset_type,
                    $matched_tag_case,
                    $original_url,
                    $sanitized_local_url
                );
                $this->log_message(
                    $log_message,
                    'notice', $tag_name, $original_url, $sanitized_local_url
                );
                return $modified_tag;
            } else {
                // Localization failed or pending. Log and ensure task is enqueued if necessary.
                if ($local_url !== null) { // Avoid double logging if Getdata returned null (pending)
                    $this->handle_localization_failure($original_url, $matched_tag_case, $asset_type, $original_tag);
                }
                // Ensure task is enqueued (safe if already done by Getdata method)
                $this->enqueue_dependency_localization_task($original_url, $asset_type);
                return $original_tag;
            }
        }, $content); // Limit -1, default flags

        // Check for regex execution errors
        if ($modified_content === null) {
             @preg_match('/./', ''); // Clear last error reliably
             $last_error_code = preg_last_error();
             if ($last_error_code !== PREG_NO_ERROR) {
                $error_msg = function_exists('preg_last_error_msg') ? preg_last_error_msg() : "Code {" . $last_error_code . "}";
                $this->log_message("preg_replace_callback error during {$tag_name}/{$attribute_name} replacement: " . $error_msg, 'error', $tag_name, null, null);
                return $original_content; // Return original content on critical error
             }
              $this->log_message("preg_replace_callback returned null without explicit error during {$tag_name}/{$attribute_name} replacement.", 'error', $tag_name, null, null);
              return $original_content;
        }

        return (string) $modified_content;
    }


    /**
     * Replace external <script src="..."> URLs with locally hosted versions.
     *
     * @param string $content The HTML content that may contain <script> tags. Must not be empty.
     * @return string The modified content. Returns original on error or no matches.
     */
    public function replace_content_script_urls(string $content): string
    {
        if (empty($content)) return $content;
        return $this->replace_content_url_attributes($content, 'src', 'script', 'js');
    }

    /**
     * Replace external URLs within inline <script> tags (JavaScript code inside script tags)
     * This handles cases like Clarity loaders that dynamically create script tags
     *
     * @param string $content The HTML content that may contain inline <script> tags
     * @return string The modified content with replaced URLs
     */
    public function replace_inline_script_urls(string $content): string
    {
        if (empty($content) || stripos($content, '<script') === false) {
            return $content;
        }
        
        $original_content = $content;
        
        try {
            // Process each <script> tag that contains inline JavaScript
            $content = preg_replace_callback(
                '/<script\b(?<attrs>[^>]*)>(?<js_content>.*?)<\/script>/is',
                function($matches) {
                    $attrs = $matches['attrs'] ?? '';
                    $js_content = $matches['js_content'] ?? '';
                    
                    // Skip empty scripts or scripts with src attribute (already handled by replace_content_script_urls)
                    // Use word boundary to avoid matching data-src, srcset, etc.
                    if (empty(trim($js_content)) || preg_match('/\bsrc\s*=/i', $attrs)) {
                        return $matches[0];
                    }
                    
                    $modified_js = $js_content;
                    $was_modified = false;
                    
                    // Optimization: Find potential URL-like strings in the JS content first
                    // Matches strings starting with http://, https://, or // inside quotes
                    if (preg_match_all('/(["\'])((?:https?:)?\/\/[^"\']+?)\1/', $js_content, $url_matches)) {
                        $found_urls = array_unique($url_matches[2], SORT_STRING);

                        foreach ($found_urls as $url_found) {
                            // Check if this URL is in our asset map (using individual lookup)
                            // Individual lookup handles Object Cache internally, avoiding memory bomb.
                            $local_url = $this->getdata->get_local_url_if_processed($url_found, 'js');
                            
                            if (!$local_url || !is_string($local_url) || empty(trim($local_url))) {
                                continue;
                            }
                            
                            // Perform replacement for this specific asset
                            $escaped_original = preg_quote($url_found, '/'); // Use the found string for matching
                            
                            // Pattern 1: .src = "URL" or .src="URL" or t.src="URL"
                            $pattern1 = '/(\w*\.src\s*=\s*["\'])' . $escaped_original . '(["\'])/i';
                            if (preg_match($pattern1, $modified_js)) {
                                $escaped_replacement = addcslashes($local_url, '\\$');
                                $result = preg_replace($pattern1, '$1' . $escaped_replacement . '$2', $modified_js);
                                if ($result !== null) {
                                    $modified_js = $result;
                                    $was_modified = true;
                                    $this->log_message(
                                        "Replaced inline script URL: {$url_found} -> {$local_url}",
                                        'debug',
                                        'inline-script',
                                        $url_found,
                                        $local_url
                                    );
                                } else {
                                    // Log error but continue with other replacements
                                    $this->log_message("preg_replace failed for pattern1 in inline script replacement", 'error', 'inline-script', $url_found, null);
                                }
                            }

                            // Pattern 2: var url = "URL" or const url = "URL"
                            $pattern2 = '/((?:var|let|const)\s+\w+\s*=\s*["\'])' . $escaped_original . '(["\'])/i';
                            if (preg_match($pattern2, $modified_js)) {
                                $escaped_replacement = addcslashes($local_url, '\\$');
                                $result = preg_replace($pattern2, '$1' . $escaped_replacement . '$2', $modified_js);
                                if ($result !== null) {
                                    $modified_js = $result;
                                    $was_modified = true;
                                } else {
                                    // Log error but continue with other replacements
                                    $this->log_message("preg_replace failed for pattern2 in inline script replacement", 'error', 'inline-script', $url_found, null);
                                }
                            }
                            
                            // Pattern 3: Direct string literals "URL" - REMOVED for safety
                            // This was too aggressive and could replace non-resource strings (UI text, etc.)
                            // $pattern3 = '/(["\'])' . $escaped_original . '(["\'])/';
                            // if (preg_match($pattern3, $modified_js)) {
                            //    $modified_js = preg_replace($pattern3, '$1' . $local_url . '$2', $modified_js);
                            //    $was_modified = true;
                            // }
                        }
                    }
                    
                    if ($was_modified) {
                        // Reconstruct the script tag with modified content
                        return str_replace($js_content, $modified_js, $matches[0]);
                    }
                    
                    return $matches[0];
                },
                $content
            );
            
            // Check for regex errors
            if ($content === null) {
                $last_error = preg_last_error();
                if ($last_error !== PREG_NO_ERROR) {
                    $error_msg = function_exists('preg_last_error_msg') ? preg_last_error_msg() : "Code {" . $last_error . "}";
                    $this->log_message(
                        "preg_replace_callback error during inline script replacement: {$error_msg}",
                        'error',
                        'inline-script',
                        null,
                        null
                    );
                    return $original_content;
                }
                // If null but no explicit error, could be memory issue or other edge case
                $this->log_message(
                    "preg_replace_callback returned null without explicit error during inline script replacement. Potential resource issue?",
                    'error',
                    'inline-script',
                    null,
                    null
                );
                return $original_content;
            }
            
        } catch (\Throwable $e) {
            $this->log_message(
                "Exception during inline script URL replacement: " . $e->getMessage(),
                'error',
                'inline-script',
                null,
                null
            );
            return $original_content;
        }
        
        return $content;
    }

    /**
     * Replace external <link rel="stylesheet" href="..."> URLs with locally hosted versions.
     *
     * @param string $content The HTML content that may contain <link> tags. Must not be empty.
     * @return string The modified content. Returns original on error or no matches.
     */
    public function replace_content_stylesheet_urls(string $content): string
    {
        if (empty($content)) return $content;
        // Pass 'stylesheet' as the required 'rel' attribute value
        return $this->replace_content_url_attributes($content, 'href', 'link', 'css', 'stylesheet');
    }

    /**
     * Replace external <img src="..."> URLs with locally hosted versions.
     * Also handles srcset attribute for responsive images when Extract service is available.
     *
     * @param string $content The HTML content that may contain <img> tags. Must not be empty.
     * @return string The modified content. Returns original on error or no matches.
     */
    public function replace_content_image_urls(string $content): string
    {
        if (empty($content)) return $content;
        
        // Replace the main 'src' attribute using the generic function
        $content = $this->replace_content_url_attributes($content, 'src', 'img', 'image');
        
        // Also handle srcset attribute if Extract service is available
        if ($this->extract !== null && stripos($content, 'srcset') !== false) {
            $content = $this->replace_srcset_urls($content);
        }
        
        return $content;
    }
    
    /**
     * Replace external URLs in srcset attributes with locally hosted versions.
     * Uses Extract::parse_srcset() for robust srcset parsing.
     *
     * @param string $content The HTML content that may contain srcset attributes.
     * @return string The modified content with srcset URLs replaced.
     */
    private function replace_srcset_urls(string $content): string
    {
        if (empty($content) || $this->extract === null) {
            return $content;
        }
        
        // Match srcset attributes in img and source tags
        // Improved regex to handle attributes containing '>' within quotes
        $pattern = '/(<(?:img|source)\b(?:[^>"\']|"[^"]*"|\'[^\']*\')*?)srcset\s*=\s*(["\'])([^"\\]+?)\2((?:[^>"\']|"[^"]*"|\'[^\']*\')*>)/is';
        
        $modified_content = preg_replace_callback($pattern, function ($matches) {
            $before_srcset = $matches[1];
            $quote = $matches[2];
            $srcset_value = $matches[3];
            $after_srcset = $matches[4];
            
            // Parse srcset using Extract
            $urls = $this->extract->parse_srcset($srcset_value);
            
            if (empty($urls)) {
                return $matches[0]; // Return original if no URLs found
            }
            
            $modified_srcset = $srcset_value;
            $replacements_made = false;
            
            foreach ($urls as $original_url) {
                // Skip non-external URLs
                try {
                    if (!$this->urlProcessor->is_external_url($original_url)) {
                        continue;
                    }
                } catch (\Throwable $e) {
                    $this->log_message("Error checking externality for srcset URL '{$original_url}': " . $e->getMessage(), 'error', 'srcset-replacement', $original_url, null);
                    continue;
                }

                // Process the asset URL (check status, get local URL, handle errors)
                try {
                    $local_url = $this->process_asset_url($original_url, 'image', 'srcset-replacement');
                } catch (\Throwable $e) {
                    $this->log_message("Error processing srcset URL '{$original_url}': " . $e->getMessage(), 'error', 'srcset-replacement', $original_url, null);
                    continue;
                }

                if ($local_url === false) {
                    continue; // Asset is disabled via toggle
                }

                if ($local_url && is_string($local_url) && trim($local_url) !== '') {
                    try {
                        $sanitized_local_url = $this->urlProcessor->normalize_url($local_url);
                        $is_valid_local_url = $this->urlProcessor->is_valid_url($sanitized_local_url);
                    } catch (\Throwable $e) {
                        $this->log_message("Error normalizing srcset local URL: " . $e->getMessage(), 'error', 'srcset-replacement', $original_url, $local_url);
                        continue;
                    }

                    if (!empty($sanitized_local_url) && $is_valid_local_url) {
                        // Replace the URL in srcset, preserving the descriptor (e.g., "2x", "300w")
                        // Use preg_replace for safer replacement (avoid partial matches of similar URLs)
                        $pattern_url = '/(?<=^|\s|,)' . preg_quote($original_url, '/') . '(?=\s|,|$)/';
                        $result = preg_replace($pattern_url, $sanitized_local_url, $modified_srcset);
                        if ($result !== null) {
                            $modified_srcset = $result;
                            $replacements_made = true;
                        } else {
                            // Log error but continue with other URLs
                            $this->log_message("preg_replace failed for srcset URL replacement", 'error', 'srcset-replacement', $original_url, $sanitized_local_url);
                        }

                        $this->log_message(
                            sprintf('Replaced srcset URL: %s -> %s', $original_url, $sanitized_local_url),
                            'debug', 'srcset-replacement', $original_url, $sanitized_local_url
                        );
                    }
                } else {
                    // Enqueue for background processing if not available or error occurred
                    $this->enqueue_dependency_localization_task($original_url, 'image');
                }
            }
            
            if ($replacements_made) {
                return $before_srcset . 'srcset=' . $quote . $modified_srcset . $quote . $after_srcset;
            }
            
            return $matches[0]; // Return original if no replacements made
        }, $content);
        
        // Check for regex errors
        if ($modified_content === null) {
            @preg_match('/./', ''); // Clear last error reliably
            $last_error_code = preg_last_error();
            if ($last_error_code !== PREG_NO_ERROR) {
                $error_msg = function_exists('preg_last_error_msg') ? preg_last_error_msg() : "Code {" . $last_error_code . "}";
                $this->log_message('preg_replace_callback error during srcset replacement: ' . $error_msg, 'error', 'srcset-replacement', null, null);
                return $content;
            }
            // If null but no explicit error, could be memory issue or other edge case
            $this->log_message('preg_replace_callback returned null without explicit error during srcset replacement. Potential resource issue?', 'error', 'srcset-replacement', null, null);
            return $content;
        }
        
        return $modified_content;
    }

    /**
     * Helper function to replace src attributes on <source> tags within a block of HTML.
     * Used by video and audio replacement functions.
     *
     * @param string $inner_content The HTML content from within a <video> or <audio> tag.
     * @param string $media_type    The parent media type ('video' or 'audio').
     * @return string The modified inner content.
     */
    private function replace_media_sources(string $inner_content, string $media_type): string
    {
        if (empty($inner_content) || empty($media_type)) {
            return $inner_content;
        }
        // Use the generic attribute replacer, targeting <source src="...">
        // The asset type passed to Getdata will be the parent media type.
        return $this->replace_content_url_attributes($inner_content, 'src', 'source', $media_type);
    }

    /**
     * Replace external <video src="..."> and inner <source src="..."> URLs with locally hosted versions.
     *
     * @param string $content The HTML content that may contain <video> tags. Must not be empty.
     * @return string The modified content. Returns original on error or no matches.
     */
    public function replace_content_video_urls(string $content): string
    {
        // Early exit for performance.
        if (empty($content) || stripos($content, '<video') === false) { return $content; }

        // 1. Replace the optional src attribute on the main <video> tag itself
        // If replace_content_url_attributes fails critically, it returns original $content.
        $content = $this->replace_content_url_attributes($content, 'src', 'video', 'video');

        // 2. Handle <source> tags nested within <video> tags
        // Use preg_replace_callback on the potentially modified content. /is flag for multiline content.
        // Ensure regex doesn't break on complex attributes or self-closing tags (though <video> shouldn't be self-closing).
        $modified_content = preg_replace_callback('/<video\b(?<video_attrs>[^>]*)>(?<inner_content>.*?)<\/video>/is', function ($matches) {
            $video_attrs = $matches['video_attrs'] ?? '';
            $inner_content = $matches['inner_content'] ?? '';
            $modified_inner_content = $inner_content; // Assume no change initially

            // Optimization: Only process inner content if it likely contains <source src="...">
            if (!empty($inner_content) && stripos($inner_content, '<source') !== false) {
                // Check if there's actually a src attribute (not data-src or srcset)
                if (preg_match('/<source\b[^>]*\bsrc\s*=/i', $inner_content)) {
                    // Call the dedicated helper to replace URLs in <source> tags
                    // This helper should also return original content on critical error.
                    $modified_inner_content = $this->replace_media_sources($inner_content, 'video');
                }
            }
            // Reconstruct the video tag safely
            // The video attributes are captured from original HTML and should be safe
            // However, we strip any potential event handlers for security
            $safe_video_attrs = preg_replace('/\s+on\w+\s*=\s*(["\'])[^\1]*?\1/i', '', $video_attrs);
            $safe_video_attrs = preg_replace('/\s+on\w+\s*=\s*[^\s>]*/i', '', $safe_video_attrs);
            return '<video' . $safe_video_attrs . '>' . $modified_inner_content . '</video>';
        }, $content); // Use content already processed for main src attribute

        // Check for errors during the video source replacement callback
        if ($modified_content === null) {
            @preg_match('/./', ''); // Clear last error reliably
            $last_error_code = preg_last_error();
            if ($last_error_code !== PREG_NO_ERROR) {
                $error_msg = function_exists('preg_last_error_msg') ? preg_last_error_msg() : "Code {" . $last_error_code . "}";
                $this->log_message('preg_replace_callback error during video source replacement: ' . $error_msg, 'error', 'video-source', null, null);
                // Return content with potentially only main src replaced if inner failed.
                return $content;
            }
             $this->log_message('preg_replace_callback returned null without explicit error during video source replacement.', 'error', 'video-source', null, null);
             return $content;
        }
        return (string) $modified_content;
    }

    /**
     * Replace external <audio src="..."> and inner <source src="..."> URLs with locally hosted versions.
     *
     * @param string $content The HTML content that may contain <audio> tags. Must not be empty.
     * @return string The modified content. Returns original on error or no matches.
     */
    public function replace_content_audio_urls(string $content): string
    {
        // Early exit for performance.
        if (empty($content) || stripos($content, '<audio') === false) { return $content; }

        // 1. Replace the optional src attribute on the main <audio> tag itself
        $content = $this->replace_content_url_attributes($content, 'src', 'audio', 'audio');

        // 2. Handle <source> tags nested within <audio> tags
        $modified_content = preg_replace_callback('/<audio\b(?<audio_attrs>[^>]*)>(?<inner_content>.*?)<\/audio>/is', function ($matches) {
            $audio_attrs = $matches['audio_attrs'] ?? '';
            $inner_content = $matches['inner_content'] ?? '';
            $modified_inner_content = $inner_content;

            if (!empty($inner_content) && stripos($inner_content, '<source') !== false) {
                // Check if there's actually a src attribute (not data-src or srcset)
                if (preg_match('/<source\b[^>]*\bsrc\s*=/i', $inner_content)) {
                    $modified_inner_content = $this->replace_media_sources($inner_content, 'audio');
                }
            }
            // Reconstruct audio tag safely, stripping event handlers
            $safe_audio_attrs = preg_replace('/\s+on\w+\s*=\s*(["\'])[^\1]*?\1/i', '', $audio_attrs);
            $safe_audio_attrs = preg_replace('/\s+on\w+\s*=\s*[^\s>]*/i', '', $safe_audio_attrs);
            return '<audio' . $safe_audio_attrs . '>' . $modified_inner_content . '</audio>';
        }, $content); // Use content already processed for main src attribute

        // Check for errors during the audio source replacement callback
        if ($modified_content === null) {
            @preg_match('/./', ''); // Clear last error reliably
            $last_error_code = preg_last_error();
            if ($last_error_code !== PREG_NO_ERROR) {
                $error_msg = function_exists('preg_last_error_msg') ? preg_last_error_msg() : "Code {" . $last_error_code . "}";
                $this->log_message('preg_replace_callback error during audio source replacement: ' . $error_msg, 'error', 'audio-source', null, null);
                return $content; // Return content with potentially only main src replaced
            }
             $this->log_message('preg_replace_callback returned null without explicit error during audio source replacement.', 'error', 'audio-source', null, null);
             return $content;
        }
        return (string) $modified_content;
    }

    /**
     * Resolve dependencies recursively using Depth First Search. Detects cycles.
     * Requires Getdata instance for mapping lookup.
     *
     * @param array<string> $dependencies List of dependency identifiers (original URLs).
     * @param array<string> &$resolved    Accumulated list of resolved identifiers (e.g., local handles/hashed_filenames). Passed by reference.
     * @param array<string, bool> &$seen        Tracks dependencies currently in recursion stack (original IDs => true). Passed by reference.
     * @return bool True on success for this branch, false on failure (cycle or missing critical data).
     */
    private function resolve_dependency_recursive(array $dependencies, array &$resolved, array &$seen, int $depth = 0): bool
    {
        // Prevent stack overflow from very deep dependency trees
        $max_depth = 100; // Safety limit for recursion depth
        if ($depth > $max_depth) {
            $this->log_message("Dependency resolution depth limit ({$max_depth}) exceeded. Possible infinite dependency chain or excessively deep tree.", 'error', 'dependency-resolution', null, null);
            return false;
        }

        // Availability check moved to public wrapper for efficiency.

        foreach ($dependencies as $dependency_id) {
            // --- Input Validation ---
            if (!is_string($dependency_id) || trim($dependency_id) === '') {
                $this->log_message('Invalid dependency identifier encountered (must be non-empty string). Skipping resolution step.', 'warning', 'dependency-resolution', null, null);
                continue; // Skip this invalid entry, but don't fail the whole branch.
            }

            // --- Resolve Identifier & Get Children (using Getdata) ---
            $resolved_identifier = $dependency_id; // Default to original ID
            $child_dependencies = [];
            $entry_data = null;

            try {
                 $entry_data = $this->getdata->get_mapping_entry_by_url($dependency_id);
            } catch (\Throwable $e) {
                 $this->log_message("Error calling get_mapping_entry_by_url for dependency '{$dependency_id}': " . $e->getMessage(), 'error', 'dependency-resolution', $dependency_id, null);
                 // Treat as dependency not found, allow resolution to continue with original ID if possible.
                 $entry_data = null; // Ensure it's null on error
            }

            if (is_array($entry_data)) {
                // Use mapped identifier (e.g., hash) if available and valid non-empty string
                if (isset($entry_data['hashed_filename']) && is_string($entry_data['hashed_filename']) && trim($entry_data['hashed_filename']) !== '') {
                    $resolved_identifier = $entry_data['hashed_filename'];
                }
                // Get children if available and valid array
                if (isset($entry_data['dependencies']) && is_array($entry_data['dependencies'])) {
                    // Filter out invalid child dependency entries upfront
                    $child_dependencies = array_filter($entry_data['dependencies'], function($dep) {
                        return is_string($dep) && trim($dep) !== '';
                    });
                }
            } elseif ($entry_data === null || $entry_data === false) {
                // Dependency not found in map. Log warning, proceed with original ID and no children.
                // This might be acceptable if the dependency is optional or handled elsewhere.
                $message_format = function_exists('__') ? __("Dependency not found in map during resolution: %s. Resolution path may be incomplete.", 'self-host-assets') : "Dependency not found in map during resolution: %s. Resolution path may be incomplete.";
                $safe_url = function_exists('esc_url_raw') ? esc_url_raw($dependency_id) : filter_var($dependency_id, FILTER_SANITIZE_URL);
                $this->log_message(sprintf($message_format, $safe_url), 'warning', 'dependency-resolution', $dependency_id, null);
            }
            // If $entry_data is unexpected type, implicitly uses original ID and no children.


            // --- Skip if Already Resolved ---
            // Check using the *resolved* identifier against the final list.
            if (in_array($resolved_identifier, $resolved, true)) {
                continue;
            }

            // --- Circular Dependency Detection ---
            // Check using the *original* ID against the current recursion stack ($seen).
            if (isset($seen[$dependency_id])) {
                $message_format = function_exists('__') ? __("Circular dependency detected involving asset: %s", 'self-host-assets') : "Circular dependency detected involving asset: %s";
                $safe_url = function_exists('esc_url_raw') ? esc_url_raw($dependency_id) : filter_var($dependency_id, FILTER_SANITIZE_URL);
                $this->log_message(sprintf($message_format, $safe_url), 'error', 'dependency-resolution', $dependency_id, null);
                return false; // Failure due to cycle
            }

            // --- Mark and Recurse ---
            $seen[$dependency_id] = true; // Mark original ID as seen for this path

            $recursion_success = true;
            if (!empty($child_dependencies)) {
                // Pass dependencies down correctly with incremented depth
                if (!$this->resolve_dependency_recursive($child_dependencies, $resolved, $seen, $depth + 1)) {
                    $recursion_success = false; // Propagate failure up
                }
            }

            // --- Backtrack ---
            unset($seen[$dependency_id]); // Unmark after recursion returns

            // --- Handle Recursion Result ---
            if (!$recursion_success) {
                return false; // Propagate failure up immediately
            }

            // --- Add to Resolved List (Post-Order) ---
            // Add the *resolved* identifier after all children are processed.
            // Ensure it's not already added (double check, though initial check should cover it).
            if (!in_array($resolved_identifier, $resolved, true)) {
                $resolved[] = $resolved_identifier;
            }
        }

        return true; // Success for this level
    }

    /**
     * Public wrapper for resolving dependencies. Initializes state and calls recursive helper.
     * Requires Getdata class for mapping lookups.
     *
     * @param array<string> $dependencies List of initial dependency identifiers (original URLs).
     * @return array<string>|false Ordered list of resolved identifiers (e.g., local handles/hashes) on success, false on failure (cycle, missing Getdata).
     */
    public function resolve_dependencies(array $dependencies): array|bool {
        // Handle empty input gracefully.
        if (empty($dependencies)) {
            return [];
        }
        // Filter initial dependencies for validity
         $valid_dependencies = array_filter($dependencies, function($dep) {
            $is_valid = is_string($dep) && trim($dep) !== '';
            if (!$is_valid) {
                $context_value = is_scalar($dep) ? (string) $dep : gettype($dep);
                 $this->log_message("Invalid initial dependency identifier ('{$context_value}'). Filtering out.", 'warning', 'dependency-resolution', null, null);
            }
            return $is_valid;
        });

        if (empty($valid_dependencies)) {
            return [];
        }


        $resolved = []; // Final ordered list
        $seen = [];     // Tracks path for cycle detection

        if ($this->resolve_dependency_recursive($valid_dependencies, $resolved, $seen)) {
            return $resolved; // Success
        } else {
            // Failure logged within helper. Log summary failure here.
            $message = function_exists('__') ? __("Failed to resolve dependency tree fully. Check logs for details (e.g., circular reference or missing dependency mapping).", 'self-host-assets') : "Failed to resolve dependency tree fully. Check logs for details (e.g., circular reference or missing dependency mapping).";
            $this->log_message($message, 'error', 'dependency-resolution-summary', null, null);
            return false; // Indicate overall failure
        }
    }


    /**
     * Resolve a dependency handle (e.g., 'self-hosted-js-abcdef123') back to its original URL.
     * Uses injected Getdata dependency for hash-to-URL mapping.
     *
     * @param string $dep_handle The dependency handle string (expected format: 'prefix-[type]-[hash]'). Must not be empty.
     * @param string $type       The asset type ('js', 'css', etc.). Must not be empty.
     * @return string|false The resolved original URL string on success, or false on failure.
     */
    public function resolve_dependency_handle(string $dep_handle, string $type) {
        // Basic input validation
        if (trim($dep_handle) === '' || trim($type) === '') {
            $this->log_message('Invalid empty handle or type provided to resolve_dependency_handle.', 'warning', 'dependency-handle-resolve', null, null);
            return false;
        }

        // Define pattern: prefix + type + hash (hex chars). Case-insensitive.
        // Allow filterable prefix? For now, hardcode 'self-hosted-'. Ensure prefix is safe.
        $prefix = 'self-hosted-'; // Consider making this a class constant or configurable.
        // Ensure type is safe for regex (basic alphanumeric check)
        if (!ctype_alnum($type)) {
             $this->log_message("Invalid characters in type '{$type}' for resolve_dependency_handle.", 'warning', 'dependency-handle-resolve', null, $dep_handle);
            return false;
        }
        // Use preg_quote for prefix, allow specific types, capture hash.
        $pattern = '/^' . preg_quote($prefix, '/') . '(?:' . preg_quote($type, '/') . ')-([a-f0-9]+)$/i';

        if (preg_match($pattern, $dep_handle, $matches)) {
            $dep_hash = $matches[1] ?? null;

            // Validate extracted hash
            if ($dep_hash === null || !ctype_xdigit($dep_hash)) { // Use ctype_xdigit for hex check
                $log_text = function_exists('__') ? __("Invalid hash extracted from dependency handle: '%s'.", 'self-host-assets') : "Invalid hash extracted from dependency handle: '%s'.";
                $sanitized_handle = function_exists('esc_html') ? esc_html($dep_handle) : htmlspecialchars($dep_handle, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
                $this->log_message(sprintf($log_text, $sanitized_handle), 'warning', 'dependency-handle-resolve', null, $dep_handle);
                return false;
            }

            // Lookup using injected Getdata dependency
            try {
                $dependency_entry = $this->getdata->get_mapping_entry_by_hash($dep_hash, $type);
            } catch (\Throwable $e) {
                 $this->log_message("Error calling get_mapping_entry_by_hash for hash '{$dep_hash}': " . $e->getMessage(), 'error', 'dependency-handle-resolve', null, $dep_handle);
                 return false;
            }

            // Check result and return original_url if valid non-empty string and looks like a URL.
            if (is_array($dependency_entry) && isset($dependency_entry['original_url']) && is_string($dependency_entry['original_url']) && trim($dependency_entry['original_url']) !== '') {
                $is_valid_dep_url = $this->urlProcessor->is_valid_url($dependency_entry['original_url']);
                if ($is_valid_dep_url) {
                    return $dependency_entry['original_url']; // Success
                }
            }
            $log_text = function_exists('__') ? __("No valid mapping entry found for handle '%1\$s' (hash: %2\$s, type: %3\$s). Cannot resolve to original URL.", 'self-host-assets') : "No valid mapping entry found for handle '%1\$s' (hash: %2\$s, type: %3\$s). Cannot resolve to original URL.";
            $sanitized_handle = function_exists('esc_html') ? esc_html($dep_handle) : htmlspecialchars($dep_handle, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
            $sanitized_hash = function_exists('esc_html') ? esc_html($dep_hash) : htmlspecialchars($dep_hash, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
            $sanitized_type = function_exists('esc_html') ? esc_html($type) : htmlspecialchars($type, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
            $this->log_message(sprintf($log_text, $sanitized_handle, $sanitized_hash, $sanitized_type), 'warning', 'dependency-handle-resolve', null, $dep_handle);
            return false; // Not found or invalid entry
        } else {
            // Handle does not match expected format
            $log_text = function_exists('__') ? __("Invalid dependency handle format: '%1\$s'. Expected '%2\$s%3\$s-[hash]'.", 'self-host-assets') : "Invalid dependency handle format: '%1\$s'. Expected '%2\$s%3\$s-[hash]'.";
            $sanitized_handle = function_exists('esc_html') ? esc_html($dep_handle) : htmlspecialchars($dep_handle, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
            $safe_prefix = function_exists('esc_html') ? esc_html($prefix) : htmlspecialchars($prefix, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
            $safe_type = function_exists('esc_html') ? esc_html($type) : htmlspecialchars($type, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
            $this->log_message(sprintf($log_text, $sanitized_handle, $safe_prefix, $safe_type), 'warning', 'dependency-handle-resolve', null, $dep_handle);
            return false; // Format mismatch
        }
    }

    /**
     * Find all raw URL strings mentioned in CSS content (within url(...) and @import).
     * Uses injected Extract service when available, falls back to AssetHtmlParser.
     *
     * @param string $css_content The CSS content string. Must not be empty.
     * @return array<string> Array of unique URL strings found.
     */
    public function find_all_urls_in_css(string $css_content): array {
        if (empty(trim($css_content))) { return []; }

        // Use a reasonable base URL for relative URL resolution
        $base_url = function_exists('site_url') ? site_url('/') : 'http://localhost/';

        // Use injected Extract service when available
        if ($this->extract !== null) {
            try {
                return $this->extract->extract_css_dependencies($css_content, $base_url);
            } catch (\Throwable $e) {
                $this->log_message("Extract service failed, falling back to AssetHtmlParser: " . $e->getMessage(), 'warning', 'css-extraction', null, null);
            }
        }

        // Fallback to AssetHtmlParser static method
        return AssetHtmlParser::extract_urls_from_css($css_content, $base_url);
    }

    /**
     * Find all potential external dependency URLs in JavaScript content (import/require statements).
     * Uses injected Extract service when available, falls back to AssetHtmlParser.
     *
     * @param string $js_content The JavaScript content string. Must not be empty.
     * @return array<string> Array of unique external dependency URLs found.
     */
    public function find_all_dependencies_in_js(string $js_content): array {
        if (empty(trim($js_content))) { return []; }

        // Use a reasonable base URL for relative URL resolution
        $base_url = function_exists('site_url') ? site_url('/') : 'http://localhost/';

        // Use injected Extract service when available
        if ($this->extract !== null) {
            try {
                $extracted_assets = $this->extract->extract_assets_from_js($js_content, $base_url);
                if (!empty($extracted_assets)) {
                    // Collect all JS dependencies from the categorized results
                    $js_deps = $extracted_assets['js'] ?? [];
                    $worker_deps = $extracted_assets['worker'] ?? [];
                    $fetch_deps = $extracted_assets['fetch'] ?? [];
                    return array_merge($js_deps, $worker_deps, $fetch_deps);
                }
            } catch (\Throwable $e) {
                $this->log_message("Extract service failed, falling back to AssetHtmlParser: " . $e->getMessage(), 'warning', 'js-extraction', null, null);
            }
        }

        // Fallback to AssetHtmlParser static method
        $extracted_assets = AssetHtmlParser::extract_js_assets($js_content, $base_url);
        
        if (empty($extracted_assets)) {
            return [];
        }
        
        $dependencies = [];
        
        // Collect all JS dependencies from the categorized results
        // Primary focus is on 'js' category, but also include 'worker' and 'fetch' URLs
        $js_deps = $extracted_assets['js'] ?? [];
        $worker_deps = $extracted_assets['worker'] ?? [];
        $fetch_deps = $extracted_assets['fetch'] ?? [];
        
        // Merge all relevant dependencies
        $all_deps = array_merge($js_deps, $worker_deps, $fetch_deps);
        
        // Filter to only include external URLs
        foreach ($all_deps as $dep_url) {
            if (!empty($dep_url) && is_string($dep_url)) {
                try {
                    if ($this->urlProcessor->is_external_url($dep_url)) {
                        $dependencies[$dep_url] = true;
                    }
                } catch (\Throwable $e) {
                    $this->log_message(
                        "Error checking externality for JS dependency '{$dep_url}': " . $e->getMessage(),
                        'error', 'js-dependency-extraction', $dep_url, null
                    );
                }
            }
        }
        
        return array_keys($dependencies);
    }

    /**
     * Replace external dependencies (found URLs) within content (CSS or JS) with their local URLs.
     * Handles relative URL resolution (needs normalize class), validation, local URL retrieval, replacement.
     *
     * @param string $content      The CSS or JS content string. Must not be empty.
     * @param array<string>  $dependencies Array of dependency URL strings to replace (raw as found).
     * @param string $type         Parent asset type ('css', 'js'). Must be 'css' or 'js'.
     * @param string $base_url     Absolute base URL of the original content (required for relative CSS deps). Defaults to ''.
     * @return string|false Updated content string, or false on critical replacement error. Original if no valid dependencies/replacements.
     */
    public function replace_external_dependencies_with_local(string $content, array $dependencies, string $type, string $base_url = '')
    {
        if (empty(trim($content)) || empty($dependencies)) { return $content; }
        if ($type !== 'css' && $type !== 'js') {
            $this->log_message("Invalid type '{$type}' for replace_external_dependencies_with_local.", 'error', 'dependency-replacement', null, null);
            return $content; // Return original if type invalid
        }

        $original_content = $content; // For fallback

        // Dependencies are injected via constructor, always available
        $has_valid_base_url = !empty($base_url) && $this->urlProcessor->is_valid_url($base_url);
        $can_resolve_relative = $type === 'css' && $has_valid_base_url;

        foreach ($dependencies as $dep_url) {
            if (empty($dep_url) || !is_string($dep_url)) {
                $context_value = is_scalar($dep_url) ? (string) $dep_url : gettype($dep_url);
                $this->log_message("Skipping invalid dependency entry ('{$context_value}') in {$type}.", 'debug', 'dependency-resolution', null, null);
                continue;
            }

            $original_dep_for_matching = $dep_url; // Use the exact string found for regex matching
            $absolute_dep_url = $dep_url; // Assume absolute initially

            // 1. Resolve Relative URLs (Primarily for CSS)
            $is_relative = !preg_match('~^([a-z][a-z0-9+.-]*:|//|#|data:)~i', trim($dep_url));
            if ($is_relative) {
                if ($type === 'css') {
                    if ($can_resolve_relative) {
                        try {
                            $resolved_candidate = $this->normalize->make_absolute_url($dep_url, $base_url);
                            // Validate the resolved URL strictly
                            $is_valid_resolved = $this->urlProcessor->is_valid_url($resolved_candidate);
                            if ($is_valid_resolved) {
                                $absolute_dep_url = $resolved_candidate;
                            } else {
                                $safe_url = function_exists('esc_html') ? esc_html($dep_url) : htmlspecialchars($dep_url, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
                                $log_text = function_exists('__') ? __("Failed to resolve relative CSS dependency '%1\$s' to a valid absolute URL (base: %2\$s). Skipping replacement.", 'self-host-assets') : "Failed to resolve relative CSS dependency '%1\$s' to a valid absolute URL (base: %2\$s). Skipping replacement.";
                                $this->log_message(sprintf($log_text, $safe_url, $base_url), 'warning', 'dependency-resolution', $dep_url, null);
                                continue; // Skip this relative URL
                            }
                        } catch (\Throwable $e) {
                            $safe_url = function_exists('esc_html') ? esc_html($dep_url) : htmlspecialchars($dep_url, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
                            $this->log_message("Exception resolving relative URL '{$safe_url}': " . $e->getMessage(), 'error', 'dependency-resolution', $dep_url, null);
                            continue; // Skip on exception
                        }
                    } else {
                        // Cannot resolve relative URL. Log warning and skip.
                        $safe_url = function_exists('esc_html') ? esc_html($dep_url) : htmlspecialchars($dep_url, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
                        $log_text = function_exists('__') ? __("Cannot resolve relative CSS dependency '%s'. Missing normalize class or invalid base URL. Skipping.", 'self-host-assets') : "Cannot resolve relative CSS dependency '%s'. Missing normalize class or invalid base URL. Skipping.";
                        if ($has_valid_base_url) { // Only log if normalize was the issue
                             $this->log_message(sprintf($log_text, $safe_url), 'warning', 'dependency-resolution', $dep_url, null);
                        }
                        continue;
                    }
                } else { // For JS, relative URLs are typically module identifiers, not external URLs to replace. Skip.
                    $this->log_message("Skipping relative dependency found in JS: {$dep_url}", 'debug', 'dependency-resolution', $dep_url, null);
                    continue;
                }
            }

            // 2. Validate Resolved/Absolute URL (Must be valid & external). Use helper.
            if (!$this->is_valid_external_dependency_url($absolute_dep_url, $type . '-dependency')) {
                continue; // Skip internal, invalid, or data URIs
            }

            // 3. Determine Specific Asset Type for Getdata context
            $asset_type_for_getdata = $type; // Default to parent type (css/js)
            try {
                 $determined_type = $this->getdata->determine_primary_asset_type($absolute_dep_url, $type . '-dependency');
                 if (!empty($determined_type) && is_string($determined_type)) { $asset_type_for_getdata = $determined_type; }
             } catch (\Throwable $e) {
                 $this->log_message("Error determining asset type for dependency {$absolute_dep_url}: " . $e->getMessage(), 'error', 'dependency-resolution', $absolute_dep_url, null);
                 // Proceed with default type
             }

            // Process the asset URL (check status, get local URL, handle errors)
            $local_url = $this->process_asset_url($absolute_dep_url, $asset_type_for_getdata, 'dependency-replacement');

            if ($local_url === false) {
                // Asset is disabled via toggle, skip replacement for this dependency
                continue;
            }

            if ($local_url === null) {
                // Error occurred, task already enqueued by process_asset_url
                continue;
            }

            // 5. Validate local URL and perform replacement
            // Check for unexpected non-string types (defensive programming)
            if (!is_string($local_url) || trim($local_url) === '') {
                // Failed or unexpected type. Log info, ensure task is queued.
                $this->handle_localization_failure($absolute_dep_url, "{$type}-dependency", $asset_type_for_getdata, '');
                $this->enqueue_dependency_localization_task($absolute_dep_url, $asset_type_for_getdata); // Ensure task is enqueued
                continue; // Skip replacement for now
            }

            // Asset is enabled and has local URL, perform replacement
            // Use the helper function which handles regex generation, execution, and error checking.
            $replacement_result = $this->perform_dependency_replacement(
                $content,                   // Current content state
                $original_dep_for_matching, // The exact string found in content
                $local_url,                 // The resolved local URL
                $type                       // Parent type ('css' or 'js')
            );

            // 6. Handle Replacement Result
            if ($replacement_result === false) {
                // Critical error occurred (e.g., regex failure). Bail out immediately.
                $this->log_message("Critical error replacing dependency '{$original_dep_for_matching}'. Aborting further replacements in this content.", 'error', 'dependency-resolution', $original_dep_for_matching, null);
                return false;
            } elseif (is_string($replacement_result)) {
                // Update content whether changed or not (safer, handles no-match case).
                $content = $replacement_result;
            }
        } // End foreach dependency loop

        // Return the final content (potentially modified).
        return $content;
    }

    /**
     * Helper: Checks if a dependency URL is valid, not a data URI, and external.
     * Logs warnings for invalid structure only. Returns false silently for valid internal/data URLs.
     * Assumes URL is absolute or resolved.
     *
     * @param string $dep_url The dependency URL string.
     * @param string $context Context string for logging (e.g., 'css-dependency', 'font').
     * @return bool True if valid and external, false otherwise.
     */
    private function is_valid_external_dependency_url(string $dep_url, string $context): bool
    {
        $trimmed_url = trim($dep_url);
        // Silently skip empty strings and data URIs. Check case-insensitively.
        if ($trimmed_url === '' || stripos($trimmed_url, 'data:') === 0) {
            return false;
        }

        // Validate structure using UrlProcessor
        $is_valid_structure = $this->urlProcessor->is_valid_url($trimmed_url);

        if (!$is_valid_structure) {
            $log_text = function_exists('__') ? __("Invalid URL structure encountered in %1\$s context: %2\$s", 'self-host-assets') : "Invalid URL structure encountered in %1\$s context: %2\$s";
            // Sanitize URL minimally for logging, avoiding excessive changes.
            $safe_url_log = filter_var($trimmed_url, FILTER_SANITIZE_URL);
            $safe_context = function_exists('esc_html') ? esc_html($context) : htmlspecialchars($context, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
            $this->log_message(sprintf($log_text, $safe_context, $safe_url_log), 'warning', $context, $trimmed_url, null);
            return false;
        }

        // Check externality using injected UrlProcessor dependency
        try {
             if (!$this->urlProcessor->is_external_url($trimmed_url)) {
                // Valid but internal. Skip silently.
                $this->log_message("Skipping internal dependency URL in {$context}: {$trimmed_url}", 'debug', $context, $trimmed_url, null);
                return false;
             }
             return true; // Valid and external
         } catch (\Throwable $e) {
              $this->log_message("Error checking externality for URL '{$trimmed_url}' in context {$context}: " . $e->getMessage(), 'error', $context, $trimmed_url, null);
              return false; // Assume not external on error
         }
    }

    /**
     * Downloads a dependency file synchronously using injected SelfHost dependency.
     * WARNING: Synchronous download. Avoid in frontend requests. Use async tasks instead.
     * Kept for specific backend/CLI use cases or compatibility. Ensure URL is validated before calling.
     *
     * @param string $dep_url Absolute, validated external dependency URL.
     * @param string $asset_type Asset type ('css', 'js', 'font', etc.). Must not be empty.
     * @return bool True on success (or if file already existed), false on failure.
     */
    public function download_dependency(string $dep_url, string $asset_type): bool
    {
         // Validate asset type
        if (trim($asset_type) === '') {
             $this->log_message("Invalid asset type for synchronous download.", 'error', 'dependency-download', $dep_url, null);
            return false;
        }
        // Basic URL validation again (defense in depth)
        if (!$this->urlProcessor->is_valid_url($dep_url)) {
            $this->log_message("Invalid URL provided for synchronous download: {$dep_url}", 'error', 'dependency-download', $dep_url, null);
            return false;
        }

        // Get configuration (simplified for this less preferred method). Use safe defaults.
        // Use shorter cache duration for sync as it might bypass normal async refresh logic? Or keep consistent? Using defaults.
        $cache_duration_days = $this->get_option_value('cache_expiration_days', 30);
        $force_refresh = false; // Typically false unless explicitly needed
        $current_depth = 0; // Starting depth for dependency resolution
        $retry_count = 0; // Initial retry count (SelfHost handles max_retries internally)


        try {
            // Call the synchronous download method using injected dependency
            // Parameters: url, type, cache_expiration_days, force_refresh, current_depth, retry_count
            $download_result = $this->selfHost->download_file(
                $dep_url, $asset_type, $cache_duration_days, $force_refresh, $current_depth, $retry_count
            );

            // Assume non-false result means success (might return path or true). Check type? Be explicit.
             // Let's assume `false` is failure, anything else is success (path, true, etc.)
            if ($download_result !== false) {
                $this->log_message("Synchronous download successful/file exists for dependency: {$dep_url}", 'debug', 'dependency-download', $dep_url, null);
                return true;
            } else {
                // Log sync download failure specifically. Get potential error from SelfHost? Assume it logs internally.
                $log_text = function_exists('__') ? __("Synchronous download failed for dependency: %s", 'self-host-assets') : "Synchronous download failed for dependency: %s";
                $safe_url = function_exists('esc_url_raw') ? esc_url_raw($dep_url) : filter_var($dep_url, FILTER_SANITIZE_URL);
                $this->log_message(sprintf($log_text, $safe_url), 'warning', 'dependency-download-failure', $dep_url, null);
                return false;
            }
        } catch (\Throwable $e) {
            $log_text = function_exists('__') ? __("Exception during synchronous download for %1\$s: %2\$s", 'self-host-assets') : "Exception during synchronous download for %1\$s: %2\$s";
            $safe_url = function_exists('esc_url_raw') ? esc_url_raw($dep_url) : filter_var($dep_url, FILTER_SANITIZE_URL);
            $this->log_message(sprintf($log_text, $safe_url, $e->getMessage()), 'error', 'dependency-download-exception', $dep_url, null);
            return false; // Exception means failure
        }
    }

    /**
     * Enqueues a background task for asynchronous asset localization (download & mapping).
     * Uses 'tasks::enqueue_task'. Retrieves config via 'GetOption'. Robust and secure.
     * Ensures URL is valid before enqueueing.
     *
     * @param string $dep_url Absolute, validated external dependency URL. Must be a valid URL string.
     * @param string $asset_type Asset type ('css', 'js', 'font', etc.). Must not be empty string.
     */
    public function enqueue_dependency_localization_task(string $dep_url, string $asset_type): void
    {
        // Input validation.
        if (trim($asset_type) === '') {
            $this->log_message('Invalid parameters (empty Type) for enqueue_dependency_localization_task. Cannot enqueue.', 'error', 'dependency-task-enqueue', is_string($dep_url) ? $dep_url : '(non-string)', null);
            return;
        }
        // Strict URL validation before passing to task system.
        if (!$this->urlProcessor->is_valid_url($dep_url)) {
            $safe_url = function_exists('esc_url_raw') ? esc_url_raw($dep_url) : filter_var($dep_url, FILTER_SANITIZE_URL);
            $this->log_message('Invalid URL structure provided for enqueue_dependency_localization_task: ' . $safe_url, 'error', 'dependency-task-enqueue', $dep_url, null);
            return;
        }

        // Task system is injected, no need to check availability

        // Log scheduling attempt (info level). Use sanitized values for logging.
        $safe_url_log = function_exists('esc_url_raw') ? esc_url_raw($dep_url) : filter_var($dep_url, FILTER_SANITIZE_URL);
        $safe_type = htmlspecialchars($asset_type, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
        $log_text = function_exists('__') ? __("Scheduling background localization task for %1\$s dependency: %2\$s", 'self-host-assets') : "Scheduling background localization task for %1\$s dependency: %2\$s";
        $this->log_message(sprintf($log_text, $safe_type, $safe_url_log), 'info', 'dependency-task-enqueue', $dep_url, null);

        // --- Retrieve Task Configuration (with safe defaults and validation) ---
        $max_retries = $this->get_option_value('max_retries', 3); // Default 3 retries
        $cache_expiration = $this->get_option_value('cache_expiration_days', 30); // Default 30 days cache

        // --- Prepare and Enqueue Task ---
        // Ensure action name is specific and consistent. Make it filterable? Maybe later.
        $task_action = 'self_host_asset_localization';
        $task_data = [
            'action'         => $task_action, // Action hook for the task handler
            'type'           => $asset_type,
            'original_url'   => $dep_url,     // Pass the validated URL
            'force_refresh'  => false,       // Default: standard localization task
            'retry_count'    => 0,           // Initial attempt
            'max_retries'    => $max_retries, // Configured max retries
            'cache_duration' => $cache_expiration, // Configured cache duration (days)
            // Add unique identifier for potential deduplication if task system doesn't handle it based on args.
            // Example: 'unique_id' => md5($task_action . $dep_url . $asset_type)
        ];

        try {
            // Enqueue the task using injected task queue
            $this->tasks->enqueue_task($task_data);
            // Note: Task deduplication logic depends entirely on the task queue implementation.
            // If it doesn't deduplicate based on arguments, multiple identical tasks might be queued.
        } catch (\Throwable $e) {
            // Catch exceptions specifically during the *enqueueing* process itself.
            $safe_url = function_exists('esc_url_raw') ? esc_url_raw($dep_url) : filter_var($dep_url, FILTER_SANITIZE_URL);
            $log_text = function_exists('__') ? __("Exception while enqueueing localization task for %1\$s: %2\$s", 'self-host-assets') : "Exception while enqueueing localization task for %1\$s: %2\$s";
            $this->log_message(sprintf($log_text, $safe_url, $e->getMessage()), 'error', 'dependency-task-enqueue-exception', $dep_url, null);
        }
    }

    /**
     * Helper: Performs the replacement of a specific dependency URL string with its local URL.
     * Uses appropriate regex patterns and handles sanitization/escaping.
     * Relies on `apply_regex_replacements` for execution and error handling.
     *
     * @param string $content Original content string (CSS or JS).
     * @param string $original_dep_url Exact original dependency URL string found in content. Cannot be empty.
     * @param string $local_url Absolute, validated local URL for replacement. Cannot be empty.
     * @param string $type Parent asset type ('css', 'js'). Must be 'css' or 'js'.
     * @return string|false Modified content string, original if no replacement needed/possible, false on critical error.
     */
    private function perform_dependency_replacement(string $content, string $original_dep_url, string $local_url, string $type)
    {
         // Basic validation
        if (trim($original_dep_url) === '' || trim($local_url) === '') {
            $this->log_message("Empty original or local URL provided to perform_dependency_replacement.", 'error', 'dependency-replace', $original_dep_url, $local_url);
            return $content; // Return original content if inputs invalid
        }
        if ($type !== 'css' && $type !== 'js') {
             $this->log_message("Invalid type '{$type}' for perform_dependency_replacement.", 'error', 'dependency-replace', $original_dep_url, $local_url);
            return $content;
        }

        // 1. Escape the original URL string for safe use in regex patterns. Use '#' delimiter.
        $escaped_dep_url = preg_quote($original_dep_url, '#');
        if (empty($escaped_dep_url)) {
            // This might happen if the URL itself contains characters that preg_quote removes entirely,
            // though unlikely for valid URLs. Log and prevent potential empty pattern.
            $this->log_message("Failed to escape original dependency URL for regex (result was empty): {$original_dep_url}", 'warning', 'dependency-replace', $original_dep_url, null);
            return $content; // Return original content if escaping fails critically
        }

        // 2. Get appropriate regex patterns.
        $patterns = $this->get_replacement_patterns($escaped_dep_url, $local_url, $type);
        if (empty($patterns)) {
            // Logging handled within get_replacement_patterns or due to invalid type check above.
            return $content; // Return original content if no patterns generated.
        }

        // 3. Sanitize local URL again (defense-in-depth) and validate. Use strict validation.
        $safe_local_url = function_exists('esc_url_raw') ? esc_url_raw($local_url, ['http', 'https']) : filter_var($local_url, FILTER_SANITIZE_URL);
        if (empty($safe_local_url) || !$this->urlProcessor->is_valid_url($safe_local_url)) {
            $this->log_message('Failed final sanitization/validation of local URL for dependency replacement.', 'warning', 'dependency-replace', $original_dep_url, $local_url);
            return $content; // Return original content if sanitization fails
        }

        // 4. Get corresponding replacement strings (using backreferences).
        // These use the $safe_local_url, escaped appropriately for the replacement context.
        $replacements = $this->get_replacement_values($safe_local_url, $type);

        // 5. Sanity check: Pattern count must match replacement count.
        if (count($patterns) !== count($replacements)) {
            $safe_type = function_exists('esc_html') ? esc_html($type) : htmlspecialchars($type, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');
            $safe_url = function_exists('esc_url_raw') ? esc_url_raw($original_dep_url) : filter_var($original_dep_url, FILTER_SANITIZE_URL);
            $this->log_message("Internal error: Pattern/replacement count mismatch for type {$safe_type}, URL {$safe_url}.", 'error', 'dependency-replace', $original_dep_url, null);
            // This indicates a bug in get_replacement_patterns or get_replacement_values.
            // Returning false signals a critical error preventing replacement.
            return false;
        }

        // 6. Apply replacements using the helper function.
        // Pass context for logging within the helper.
        return $this->apply_regex_replacements($content, $patterns, $replacements, $original_dep_url, $safe_local_url, $type);
    }

    /**
     * Helper: Defines the replacement strings (using backreferences) for dependency URLs.
     * Corresponds to capture groups in `get_replacement_patterns`. Escapes local URL appropriately for preg_replace and CSS/JS contexts.
     *
     * @param string $local_url Sanitized, absolute local URL.
     * @param string $type Asset type ('js', 'css'). Must be 'js' or 'css'.
     * @return array<string> Array of replacement strings. Empty for unknown types.
     */
    private function get_replacement_values(string $local_url, string $type): array
    {
        if ($type !== 'js' && $type !== 'css') return [];

        $replacements = [];
        // Escape for preg_replace context: Backslashes (\) and Dollar signs ($) literals need escaping.
        // Backreferences ($1, $2 etc.) should NOT be escaped here.
        $escaped_for_preg = addcslashes($local_url, '\$');

        if ($type === 'js') {
            // JS Pattern 1 (import...from): $1=prefix, $2=quote, $3=closing quote
            // We need to ensure the URL is safe within JS quotes. addcslashes should handle backslashes.
            // Single/double quotes within the URL path itself are rare but could break syntax if not handled.
            // Standard URL encoding should prevent raw quotes. Rely on $local_url being correctly formed.
            $replacements[] = '$1$2' . $escaped_for_preg . '$3';
            // JS Pattern 2 (import()): $1=prefix, $2=quote, 3=closing paren+quote
            $replacements[] = '$1$2' . $escaped_for_preg . '$3';
            // JS Pattern 3 (require()): $1=prefix, $2=quote, 3=closing paren+quote
            $replacements[] = '$1$2' . $escaped_for_preg . '$3';
        } elseif ($type === 'css') {
            // Escape for CSS url('...') context: single quotes (') and backslashes (\).
            $escaped_for_css = addcslashes($escaped_for_preg, "'\\"); // Apply CSS escaping *after* preg escaping

            // CSS Pattern 1 (url()): $1=prefix(url...), $2=orig_quote, $3=original closing quote/paren
            // Force single quotes url('...') for consistency and safety.
            // Reconstruct as prefix + 'new_url' + )
            $replacements[] = '$1\'' . $escaped_for_css . '\')';

            // CSS Pattern 2 (@import): $1=prefix(@import...), $2=opt url(, $3=opt quote, $4=rest (media queries+semicolon)
            // Force url('...') syntax: @import url('...'). Use same CSS-escaped URL.
            // Capture group 4 contains the rest of the statement after the URL closing delimiter.
            // We need to reconstruct carefully: Prefix + url('new_url') + rest.
            // No extra space needed before $4 as it includes preceding whitespace if any was matched.
            $replacements[] = '$1url(\'' . $escaped_for_css . '\')$4';
        }
        // Ensure counts match patterns implicitly. Caller checks explicitly.
        return $replacements;
    }

    /**
     * Replace external dependencies found in JS content (import/require) with local URLs.
     * Wrapper for finding and replacing dependencies in JS.
     * Uses Extract service when available for comprehensive JS asset extraction including
     * ES6 modules, dynamic imports, Workers, WebSockets, AJAX calls, and CDN resources.
     *
     * @param string $js_content The JavaScript content string. Must not be empty.
     * @param string $base_url   Optional base URL (rarely needed for absolute JS deps). Defaults to ''.
     * @return string|false Modified JS content, false on critical failure, original if no changes or no external deps found.
     */
    public function replace_external_dependencies_in_js(string $js_content, string $base_url = '')
    {
        if (empty(trim($js_content))) { return $js_content; }

        $dependencies = [];

        // Use Extract service if available for comprehensive JS asset extraction
        // Extract::extract_assets_from_js handles ES6 imports, dynamic imports, require(),
        // Workers, WebSockets, AJAX calls, CDN resources, and more
        if ($this->extract !== null) {
            try {
                // Use provided base_url or fall back to site_url for relative URL resolution
                $effective_base_url = !empty($base_url) ? $base_url : (function_exists('site_url') ? site_url('/') : 'http://localhost/');
                $extracted_assets = $this->extract->extract_assets_from_js($js_content, $effective_base_url);
                
                if (!empty($extracted_assets)) {
                    // Collect all JS dependencies from the categorized results
                    // Primary focus is on 'js' category, but also include 'worker' and 'fetch' URLs
                    $js_deps = $extracted_assets['js'] ?? [];
                    $worker_deps = $extracted_assets['worker'] ?? [];
                    $fetch_deps = $extracted_assets['fetch'] ?? [];
                    
                    // Merge all relevant dependencies
                    $all_deps = array_merge($js_deps, $worker_deps, $fetch_deps);
                    
                    // Filter to only include external URLs
                    foreach ($all_deps as $dep_url) {
                        if (!empty($dep_url) && is_string($dep_url)) {
                            try {
                                if ($this->urlProcessor->is_external_url($dep_url)) {
                                    $dependencies[$dep_url] = true;
                                }
                            } catch (\Throwable $e) {
                                $this->log_message("Error checking externality for JS dependency '{$dep_url}': " . $e->getMessage(), 'error', 'js-dependency-extraction', $dep_url, null);
                            }
                        }
                    }
                    
                }
            } catch (\Throwable $e) {
                $this->log_message("Error using Extract service for JS dependencies: " . $e->getMessage() . ". Falling back to legacy method.", 'warning', 'js-dependency-extraction', null, null);
                $dependencies = []; // Reset to trigger fallback
            }
        }

        // Fallback to legacy method if Extract is not available or returned no results
        if (empty($dependencies)) {
            $dependencies = $this->find_all_dependencies_in_js($js_content);
        }

        if (empty($dependencies)) {
            return $js_content; // No external dependencies found.
        }

        // Call the main replacement function. Handles errors internally.
        return $this->replace_external_dependencies_with_local($js_content, $dependencies, 'js', $base_url);
    }

    /**
     * Replace external font URLs within CSS @font-face src declarations with local versions.
     * Handles relative URL resolution based on the CSS file's base URL. Robust and secure.
     *
     * @param string $css The CSS content string. Must not be empty.
     * @param string $base_url Absolute base URL of the original CSS file. Required and must be a valid URL string.
     * @return string Modified CSS content. Returns original on error, invalid base URL, or no matches.
     */
    public function replace_css_font_face_urls(string $css, string $base_url): string
    {
        // Early exit for performance or if base URL invalid.
        if (empty(trim($css)) || stripos($css, '@font-face') === false || stripos($css, 'src:') === false) { return $css; }
        $original_css_for_error = $css; // For fallback

        // Validate base_url strictly.
        if (empty($base_url) || !$this->urlProcessor->is_valid_url($base_url)) {
            // Only log warning if CSS likely contains URLs that might need resolving.
            if (stripos($css, 'url(') !== false || stripos($css, '@import') !== false || stripos($css, '@font-face') !== false) {
                $this->log_message('Invalid or empty base_url provided for CSS processing. Relative paths cannot be resolved. Skipping replacement.', 'warning', 'css-processing', $base_url, null);
            }
            return $original_css_for_error;
        }

        // Normalize is always available via dependency injection
        $can_resolve_relative = true;


        try {
            // Find All CSS Dependencies (@import, url(), @font-face)
            // Use Extract service if available for robust CSS dependency extraction with proper URL resolution
            // Extract::extract_css_dependencies handles relative URL resolution, validation, and returns absolute URLs
            $dependency_list = [];
            
            if ($this->extract !== null) {
                // Use Extract service for comprehensive CSS dependency extraction
                // This handles @import, url(), relative URL resolution, and validation
                $extracted_deps = $this->extract->extract_css_dependencies($css, $base_url);
                
                if (!empty($extracted_deps)) {
                    // Extract returns already-resolved absolute URLs that are valid and external
                    // Filter to only include external URLs using our validation
                    foreach ($extracted_deps as $absolute_url) {
                        if ($this->is_valid_external_dependency_url($absolute_url, 'css-dependency')) {
                            $dependency_list[] = $absolute_url;
                        }
                    }
                }
            } else {
                // Fallback to legacy method if Extract service is not available
                $raw_urls = $this->find_all_urls_in_css($css);
                $dependencies_to_replace = []; // Map: original_raw_url => absolute_external_url

                if (!empty($raw_urls)) {
                    foreach ($raw_urls as $raw_url) {
                         if (empty(trim($raw_url))) continue; // Skip empty strings

                        $absolute_url = $raw_url; // Assume absolute initially
                        $is_relative = !preg_match('~^([a-z][a-z0-9+.-]*:|//|#|data:)~i', trim($raw_url));

                        if ($is_relative) {
                            if ($can_resolve_relative) {
                                try {
                                    $resolved = $this->normalize->make_absolute_url($raw_url, $base_url);
                                    if ($this->urlProcessor->is_valid_url($resolved)) {
                                        $absolute_url = $resolved;
                                    } else {
                                        // Failed to resolve relative URL to valid absolute. Skip it.
                                        $this->log_message("Failed to resolve relative CSS URL '{$raw_url}' (base: {$base_url}). Skipping.", 'debug', 'css-processing', $raw_url, null);
                                        continue;
                                    }
                                } catch (\Throwable $e) {
                                    $this->log_message("Exception resolving relative CSS URL '{$raw_url}': " . $e->getMessage(), 'error', 'css-processing', $raw_url, null);
                                    continue; // Skip on error
                                }
                            } else {
                                // Cannot resolve relative, skip it. Warning logged earlier if applicable.
                                continue;
                            }
                        }

                        // Check if the (now absolute) URL is valid and external.
                        // Use 'css-dependency' context. Helper logs invalid structure warnings.
                        if ($this->is_valid_external_dependency_url($absolute_url, 'css-dependency')) {
                            // Store mapping: original string found => absolute external URL
                            // Use original raw string for matching later. Avoid overwriting if multiple raw URLs resolve to the same absolute.
                            // Let replace_external_dependencies_with_local handle the list of raw URLs.
                            $dependencies_to_replace[$raw_url] = true; // Just mark the raw URL for processing
                        }
                    } // End foreach raw_url
                }
                
                $dependency_list = array_keys($dependencies_to_replace);
            }


            // 3. Replace Remaining External Dependencies (@import, other url())
            if (!empty($dependency_list)) {
                $result = $this->replace_external_dependencies_with_local(
                    $css,                       // Current CSS content (potentially with fonts replaced)
                    $dependency_list,           // Unique original raw URL strings to find and replace
                    'css',                      // Type
                    $base_url                   // Base URL for context if needed by replacer
                );

                // Check for critical failure from dependency replacement.
                if ($result === false) {
                    // Log that a critical error occurred during this phase.
                    $this->log_message('Critical error during general CSS dependency replacement (@import/url()). Returning CSS processed up to this point (potentially only fonts replaced).', 'error', 'css-processing', null, null);
                    // Return the CSS as it was *before* this failed step. $css holds this state.
                    return $css;
                }
                // Update CSS content with the result (which might be unchanged if no matches found).
                $css = $result;
            }

        } catch (\Throwable $e) {
            $this->log_message('Unhandled exception during CSS content replacement: ' . $e->getMessage() . ' Trace: ' . $e->getTraceAsString(), 'error', 'css-processing-exception', null, null);
            return $original_css_for_error; // Return original on major exception
        }

        return $css; // Return fully processed CSS.
    }

    /**
     * Reverse URL replacement - restore original external URLs when local files are deleted
     * 
     * This method is called when:
     * - Local asset files are deleted via UI
     * - Assets are cleaned up automatically
     * - Plugin is deactivated
     * 
     * It ensures that when local cached versions are removed, the original external URLs
     * are restored so nothing breaks and remote files are fetched as intended.
     *
     * @param string $content HTML content with potentially replaced local URLs
     * @param array $deleted_assets Array of deleted asset data with 'original_url' and 'local_url' keys
     * @return string Content with local URLs reversed back to original external URLs
     */
    public function reverse_url_replacements(string $content, array $deleted_assets): string {
        if (empty($content) || empty($deleted_assets)) {
            return $content;
        }

        $original_content = $content;

        try {
            foreach ($deleted_assets as $asset) {
                // Validate asset data structure
                if (!is_array($asset) || !isset($asset['original_url'], $asset['local_url'])) {
                    $this->log_message(
                        'Invalid asset data structure in reverse_url_replacements. Missing required keys.',
                        'warning',
                        'url-reversal',
                        null,
                        null
                    );
                    continue;
                }

                $original_url = $asset['original_url'];
                $local_url = $asset['local_url'];
                $asset_type = $asset['type'] ?? 'unknown';

                // Skip if either URL is empty
                if (empty($original_url) || empty($local_url)) {
                    continue;
                }

                // Sanitize URLs
                $safe_original_url = function_exists('esc_url_raw') 
                    ? esc_url_raw($original_url, ['http', 'https']) 
                    : filter_var($original_url, FILTER_SANITIZE_URL);
                    
                $safe_local_url = function_exists('esc_url_raw')
                    ? esc_url_raw($local_url, ['http', 'https'])
                    : filter_var($local_url, FILTER_SANITIZE_URL);

                if (empty($safe_original_url) || empty($safe_local_url)) {
                    $this->log_message(
                        'URL sanitization failed during reversal',
                        'warning',
                        'url-reversal',
                        $original_url,
                        $local_url
                    );
                    continue;
                }

                // Escape URLs for regex
                $escaped_local_url = preg_quote($safe_local_url, '#');

                // Replace in various contexts based on asset type
                $content = $this->reverse_url_in_html_attributes($content, $escaped_local_url, $safe_original_url, $asset_type);
                $content = $this->reverse_url_in_css($content, $escaped_local_url, $safe_original_url);
                $content = $this->reverse_url_in_js($content, $escaped_local_url, $safe_original_url);

                $this->log_message(
                    sprintf(
                        'Reversed URL replacement for %s asset. Local: %s, Restored Original: %s',
                        $asset_type,
                        $safe_local_url,
                        $safe_original_url
                    ),
                    'info',
                    'url-reversal',
                    $safe_original_url,
                    $safe_local_url
                );
            }

            return $content;

        } catch (\Throwable $e) {
            $this->log_message(
                'Exception during URL reversal: ' . $e->getMessage(),
                'error',
                'url-reversal',
                null,
                null
            );
            return $original_content; // Return original on error
        }
    }

    /**
     * Reverse URL replacements in HTML attributes (src, href, etc.)
     *
     * @param string $content HTML content
     * @param string $escaped_local_url Escaped local URL for regex
     * @param string $original_url Original external URL to restore
     * @param string $asset_type Asset type for context
     * @return string Modified content
     */
    private function reverse_url_in_html_attributes(string $content, string $escaped_local_url, string $original_url, string $asset_type): string {
        // Pattern matches: attribute="local_url" or attribute='local_url'
        // Handles: src, href, data-src, data-href, etc.
        $pattern = '#((?:src|href|data-src|data-href)\s*=\s*)(["\'])' . $escaped_local_url . '\2#i';
        
        $result = preg_replace_callback($pattern, function($matches) use ($original_url) {
            // Encode the original URL for HTML attribute
            $encoded_original = function_exists('esc_html')
                ? esc_html($original_url)
                : htmlspecialchars($original_url, ENT_QUOTES | ENT_HTML5 | ENT_SUBSTITUTE, 'UTF-8');

            return $matches[1] . $matches[2] . $encoded_original . $matches[2];
        }, $content);
        
        // Check for regex errors
        if ($result === null) {
            $error = preg_last_error();
            $error_msg = 'Unknown error';
            if (function_exists('preg_last_error_msg')) {
                $error_msg = preg_last_error_msg();
            } else {
                $errors = [
                    PREG_NO_ERROR => 'No error',
                    PREG_INTERNAL_ERROR => 'Internal error',
                    PREG_BACKTRACK_LIMIT_ERROR => 'Backtrack limit exceeded',
                    PREG_RECURSION_LIMIT_ERROR => 'Recursion limit exceeded',
                    PREG_BAD_UTF8_ERROR => 'Bad UTF8',
                    PREG_BAD_UTF8_OFFSET_ERROR => 'Bad UTF8 offset'
                ];
                $error_msg = $errors[$error] ?? "Error code: {" . $error . "}";
            }
            
            $this->log_message(
                "Regex error in reverse_url_in_html_attributes: {$error_msg}",
                'error',
                'url-reversal',
                null,
                null
            );
            return $content; // Return original on error
        }

        return $result;
    }

    /**
     * Reverse URL replacements in CSS (url() and @import)
     *
     * @param string $content CSS content or HTML with inline styles
     * @param string $escaped_local_url Escaped local URL for regex
     * @param string $original_url Original external URL to restore
     * @return string Modified content
     */
    private function reverse_url_in_css(string $content, string $escaped_local_url, string $original_url): string {
        // Escape $ in original_url to prevent regex backreference interpretation
        $escaped_original_url = addcslashes($original_url, '\$');
        
        // Pattern 1: url(local_url) or url('local_url') or url("local_url")
        $pattern1 = '#(url\s*\(\s*)(["\']?)' . $escaped_local_url . '\2(\s*))#i';
        $result1 = preg_replace($pattern1, '${1}${2}' . $escaped_original_url . '${2}${3}', $content);
        
        if ($result1 === null) {
            $this->log_message('Regex error in reverse_url_in_css (pattern 1)', 'error', 'url-reversal', null, null);
            return $content;
        }
        $content = $result1;

        // Pattern 2: @import "local_url" or @import 'local_url' or @import url(local_url)
        $pattern2 = '#(@import\s+(?:url\s*\(\s*)?)(["\']?)' . $escaped_local_url . '\2#i';
        $result2 = preg_replace($pattern2, '${1}${2}' . $escaped_original_url . '${2}', $content);
        
        if ($result2 === null) {
            $this->log_message('Regex error in reverse_url_in_css (pattern 2)', 'error', 'url-reversal', null, null);
            return $content;
        }

        return $result2;
    }

    /**
     * Reverse URL replacements in JavaScript (import, require, dynamic imports)
     *
     * @param string $content JavaScript content or HTML with inline scripts
     * @param string $escaped_local_url Escaped local URL for regex
     * @param string $original_url Original external URL to restore
     * @return string Modified content
     */
    private function reverse_url_in_js(string $content, string $escaped_local_url, string $original_url): string {
        // Escape $ in original_url to prevent regex backreference interpretation
        $escaped_original_url = addcslashes($original_url, '\$');
        
        // Pattern 1: import ... from "local_url" or import ... from 'local_url'
        $pattern1 = '#(import\s+(?:[^"\']+\s+from\s+)?\s*)(["\'])' . $escaped_local_url . '\2#i';
        $result1 = preg_replace($pattern1, '${1}${2}' . $escaped_original_url . '${2}', $content);
        
        if ($result1 === null) {
            $this->log_message('Regex error in reverse_url_in_js (pattern 1)', 'error', 'url-reversal', null, null);
            return $content;
        }
        $content = $result1;

        // Pattern 2: import("local_url") or import('local_url')
        $pattern2 = '#(import\s*\(\s*)(["\'])' . $escaped_local_url . '\2(\s*))#i';
        $result2 = preg_replace($pattern2, '${1}${2}' . $escaped_original_url . '${2}${3}', $content);
        
        if ($result2 === null) {
            $this->log_message('Regex error in reverse_url_in_js (pattern 2)', 'error', 'url-reversal', null, null);
            return $content;
        }
        $content = $result2;

        // Pattern 3: require("local_url") or require('local_url')
        $pattern3 = '#(require\s*\(\s*)(["\'])' . $escaped_local_url . '\2(\s*))#i';
        $result3 = preg_replace($pattern3, '${1}${2}' . $escaped_original_url . '${2}${3}', $content);
        
        if ($result3 === null) {
            $this->log_message('Regex error in reverse_url_in_js (pattern 3)', 'error', 'url-reversal', null, null);
            return $content;
        }

        return $result3;
    }

    /**
     * Reverse all URL replacements for a specific asset across all content
     * 
     * This is a convenience method that can be called when a single asset is deleted.
     * It wraps reverse_url_replacements() for single-asset operations.
     *
     * @param string $content HTML content
     * @param string $original_url Original external URL
     * @param string $local_url Local URL to be reversed
     * @param string $asset_type Asset type (js, css, image, etc.)
     * @return string Content with URLs reversed
     */
    public function reverse_single_asset_url(string $content, string $original_url, string $local_url, string $asset_type = 'unknown'): string {
        $asset_data = [
            [
                'original_url' => $original_url,
                'local_url' => $local_url,
                'type' => $asset_type
            ]
        ];

        return $this->reverse_url_replacements($content, $asset_data);
    }

    /**
     * Get all assets that need URL reversal from database
     * 
     * This method retrieves assets that are being deleted or have been marked for cleanup.
     * It's used to prepare the asset list for bulk URL reversal operations.
     * 
     * Uses the local_url column from the database for guaranteed 1:1 URL mapping accuracy.
     *
     * OPTIMIZED: Batch database query instead of N+1 queries
     *
     * @param array $asset_ids Array of asset IDs to get reversal data for
     * @return array Array of asset data with original_url, local_url, and type
     */
    public function get_assets_for_reversal(array $asset_ids): array {
        if (empty($asset_ids)) {
            return [];
        }

        // Validate and sanitize IDs
        $valid_ids = array_filter($asset_ids, function($id) {
            return is_numeric($id) && $id > 0;
        });

        if (empty($valid_ids)) {
            return [];
        }

        $assets_for_reversal = [];

        try {
            // Use Database class batch_get_assets if available for optimized retrieval
            if ($this->database !== null) {
                $assets = $this->database->batch_get_assets($valid_ids, ['id', 'original_url', 'hashed_filename', 'type', 'local_url']);
            } else {
                global $wpdb;
                if (!$wpdb) {
                    return [];
                }

                // Fallback table name logic
                $mappings_const = defined('\LHA\Database::TABLE_MAPPINGS') ? \LHA\Database::TABLE_MAPPINGS : 'lha_mappings';
                $table_suffix = $mappings_const;
                $table = $wpdb->prefix . $table_suffix;
                
                // Validate table name for security
                if (!preg_match('/^[a-zA-Z0-9_]+$/', $table)) {
                    $table = $wpdb->prefix . 'lha_mappings';
                }
                
                // OPTIMIZED: Single batch query instead of N queries
                $placeholders = implode(',', array_fill(0, count($valid_ids), '%d'));
                $query = $wpdb->prepare(
                    "SELECT id, original_url, hashed_filename, type, local_url 
                     FROM {$table} 
                     WHERE id IN ($placeholders)",
                    ...$valid_ids
                );
                
                $assets = $wpdb->get_results($query, ARRAY_A);
            }

            if ($assets && is_array($assets)) {
                // Get upload directory info once
                $upload_dir = function_exists('wp_upload_dir') ? wp_upload_dir() : ['baseurl' => ''];
                $base_url = isset($upload_dir['baseurl']) && function_exists('trailingslashit')
                    ? trailingslashit($upload_dir['baseurl'])
                    : (isset($upload_dir['baseurl']) ? rtrim($upload_dir['baseurl'], '/') . '/' : '');
                $base_path = 'lha-assets/';

                // Subdirectory mapping - must match AssetUtils::get_subdirectory_mapping()
                // Type 'image' maps to 'images' subdirectory, 'font' to 'fonts', etc.
                $subdir_map = [
                    'js' => 'js', 'css' => 'css',
                    'image' => 'images', 'font' => 'fonts',
                    'video' => 'misc', 'audio' => 'misc',
                    'document' => 'misc', 'file' => 'misc',
                    'mjs' => 'js', 'cjs' => 'js',
                    'png' => 'images', 'jpg' => 'images', 'jpeg' => 'images', 'gif' => 'images',
                    'svg' => 'images', 'webp' => 'images', 'avif' => 'images', 'bmp' => 'images',
                    'ico' => 'images', 'tif' => 'images', 'tiff' => 'images', 'heic' => 'images', 'heif' => 'images',
                    'woff' => 'fonts', 'woff2' => 'fonts', 'ttf' => 'fonts', 'otf' => 'fonts', 'eot' => 'fonts',
                    'mp4' => 'misc', 'm4v' => 'misc', 'webm' => 'misc', 'mov' => 'misc', 'avi' => 'misc',
                    'mkv' => 'misc', 'ogv' => 'misc', 'wmv' => 'misc', 'flv' => 'misc',
                    'mp3' => 'misc', 'ogg' => 'misc', 'oga' => 'misc', 'wav' => 'misc', 'flac' => 'misc',
                    'aac' => 'misc', 'm4a' => 'misc', 'opus' => 'misc',
                    'pdf' => 'misc', 'doc' => 'misc', 'docx' => 'misc', 'xls' => 'misc', 'xlsx' => 'misc',
                    'ppt' => 'misc', 'pptx' => 'misc', 'txt' => 'misc', 'rtf' => 'misc', 'csv' => 'misc',
                    'md' => 'misc', 'json' => 'misc', 'xml' => 'misc', 'html' => 'misc', 'htm' => 'misc',
                    'zip' => 'misc', 'rar' => 'misc', '7z' => 'misc', 'tar' => 'misc', 'gz' => 'misc',
                    'bz2' => 'misc', 'dat' => 'misc',
                ];

                // Process results
                foreach ($assets as $asset_data) {
                    if (empty($asset_data['original_url'])) {
                        continue;
                    }

                    // Use local_url from database if available (guaranteed accurate 1:1 mapping)
                    $local_url = $asset_data['local_url'] ?? null;
                    
                    // Fallback: construct from hashed filename if local_url not stored
                    if (empty($local_url) && !empty($asset_data['hashed_filename']) && !empty($asset_data['type'])) {
                        // Map type to subdirectory (e.g., 'image' -> 'images', 'font' -> 'fonts')
                        $subdir = $subdir_map[$asset_data['type']] ?? $asset_data['type'];
                        $local_url = $base_url . $base_path . $subdir . '/' . $asset_data['hashed_filename'];
                    }

                    if ($local_url) {
                        $assets_for_reversal[] = [
                            'original_url' => $asset_data['original_url'],
                            'local_url' => $local_url,
                            'type' => $asset_data['type'] ?? 'unknown'
                        ];
                    }
                }
            }

        } catch (\Throwable $e) {
            $this->log_message(
                'Exception while getting assets for reversal: ' . $e->getMessage(),
                'error',
                'url-reversal',
                null,
                null
            );
        }

        return $assets_for_reversal;
    }

    /**
     * Get the status of an asset from the database
     *
     * Used to check if an asset has downloading enabled (status !== 'ignored')
     * before replacing with local URL.
     *
     * Uses AssetData::get_asset_status() for centralized, cached asset status retrieval.
     * Falls back to URL variations if exact match not found.
     *
     * @param string $url Asset URL
     * @param string $type Asset type
     * @return string|null Asset status ('pending', 'processing', 'processed', 'failed', 'ignored', 'retry') or null if not found
     */
    private function get_asset_status(string $url, string $type): ?string
    {
        // Helper function to check status using GetData (centralized, cached with URL normalization)
        $check_url = function(string $check_url) use ($type): ?string {
            $status = $this->getdata->get_asset_status($check_url, $type);
            // GetData returns false if not found, convert to null for consistency
            return $status !== false ? (string) $status : null;
        };

        // Strategy 1: Try exact match first
        $status = $check_url($url);
        if ($status !== null) {
            return $status;
        }

        // Strategy 2: Try without query string (WordPress adds ?ver=X.X to URLs)
        $query_pos = strpos($url, '?');
        $url_without_query = $query_pos !== false ? substr($url, 0, $query_pos) : $url;

        // Check without query string if query string exists
        if ($url_without_query !== $url) {
            $status = $check_url($url_without_query);
            if ($status !== null) {
                return $status;
            }
        }
        
        // Strategy 3: Try without trailing slash
        $url_no_slash = rtrim($url_without_query, '/');
        if ($url_no_slash !== $url_without_query) {
            $status = $check_url($url_no_slash);
            if ($status !== null) {
                return $status;
            }
        }
        
        // Strategy 4: Try protocol variations (http vs https)
        if (strpos($url_without_query, 'https://') === 0 && strlen($url_without_query) > 8) {
            $http_url = 'http://' . substr($url_without_query, 8);
            $status = $check_url($http_url);
            if ($status !== null) {
                return $status;
            }
        } elseif (strpos($url_without_query, 'http://') === 0 && strlen($url_without_query) > 7) {
            $https_url = 'https://' . substr($url_without_query, 7);
            $status = $check_url($https_url);
            if ($status !== null) {
                return $status;
            }
        } elseif (strpos($url_without_query, '//') === 0) {
            // Protocol-relative URL - try both
            $http_url = 'http:' . $url_without_query;
            $status = $check_url($http_url);
            if ($status !== null) {
                return $status;
            }
            
            $https_url = 'https:' . $url_without_query;
            $status = $check_url($https_url);
            if ($status !== null) {
                return $status;
            }
        }

        return null;
    }

    /**
     * Retrieves a configuration option value with safe defaults and validation.
     * Ensures the value is a non-negative integer.
     *
     * @param string $option_name The name of the option to retrieve
     * @param int $default_value The default value if option is not set or invalid
     * @return int The validated option value
     */
    private function get_option_value(string $option_name, int $default_value): int
    {
        $value = $default_value;
        try {
            $retrieved_value = $this->options->get($option_name);
            // Validate: Must be non-negative integer.
            // Check if it's an actual integer type (not just numeric string) to avoid type juggling issues
            if (is_int($retrieved_value) && $retrieved_value >= 0) {
                $value = $retrieved_value;
            } elseif (is_string($retrieved_value) && ctype_digit($retrieved_value)) {
                // String representation of integer is also acceptable
                $value = function_exists('absint') ? absint($retrieved_value) : (int) $retrieved_value;
            }
        } catch (\Throwable $e) {
            $this->log_message("Error retrieving option '{$option_name}': " . $e->getMessage(), 'warning', 'option-retrieval', null, null);
        }
        return $value;
    }

    /**
     * Processes an asset URL by checking status, retrieving local URL, and handling errors.
     * Returns the local URL if available, null if pending/failed, or false if ignored.
     *
     * @param string $original_url The original external URL
     * @param string $asset_type The asset type
     * @param string $context Context for logging
     * @return string|null|false Local URL if available, null if pending/failed, false if ignored
     */
    private function process_asset_url(string $original_url, string $asset_type, string $context = 'asset-processing'): string|null|false
    {
        // Check asset status before attempting to get local URL
        $asset_status = $this->get_asset_status($original_url, $asset_type);

        if ($asset_status === 'ignored') {
            // Asset is disabled via toggle, skip processing
            $this->log_message(
                "Asset is disabled via toggle (status: ignored), skipping: {$original_url}",
                'debug', $context, $original_url, null
            );
            return false; // Explicitly return false to indicate ignored
        }

        // Try to get local URL with download fallback
        $local_url = null;
        try {
            $local_url = $this->getdata->get_local_url_with_download_fallback($original_url, $asset_type);
        } catch (\Throwable $e) {
            $this->log_message("Error calling Getdata for {$original_url}: " . $e->getMessage(), 'error', $context, $original_url, null);
            // Enqueue task for background processing
            $this->enqueue_dependency_localization_task($original_url, $asset_type);
            return null; // Return null to indicate error/pending
        }

        return $local_url; // Could be string, null, or false
    }

    /**
     * Helper: Applies regex replacements.
     *
     * @param string $content Content to modify
     * @param array $patterns Regex patterns
     * @param array $replacements Replacement strings
     * @param string $original_url Original URL for logging
     * @param string $local_url Local URL for logging
     * @param string $type Asset type
     * @return string|false Modified content or false on error
     */
    private function apply_regex_replacements(string $content, array $patterns, array $replacements, string $original_url, string $local_url, string $type)
    {
        $result = preg_replace($patterns, $replacements, $content);
        
        if ($result === null) {
            $error_code = preg_last_error();
            $this->log_message("Regex error during dependency replacement (Code: $error_code)", 'error', $type, $original_url);
            return false;
        }
        
        // Optional: Log success if the content actually changed (avoid spamming if not found)
        // Check strict inequality to detect change.
        if ($result !== $content) {
             // Only log at debug level to avoid spamming main log
            $this->log_message("Replaced dependency: $original_url -> $local_url", 'debug', $type);
        }
        
        return $result;
    }

}