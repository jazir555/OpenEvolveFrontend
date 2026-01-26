<?php

declare(strict_types=1);

namespace LHA;

use LHA\SettingsHelpers\SettingsRenderHelper;
use LHA\SettingsHelpers\SettingsRegisterHelper;
use LHA\SettingsHelpers\SettingsSanitizeHelper;
use LHA\SettingsHelpers\SettingsQueryHelper;
use LHA\SettingsHelpers\SettingsSaveHelper;
use LHA\SettingsHelpers\SettingsUtilityHelper;
use LHA\Interfaces\LoggerInterface;
use LHA\Interfaces\GetdataInterface;
use LHA\Interfaces\InitializeInterface;
use LHA\Interfaces\DatabaseInterface;
use LHA\Interfaces\AssetValidatorInterface;
use LHA\Interfaces\NormalizeInterface;
use LHA\Interfaces\UrlProcessorInterface;

/**
 * Class Settings
 *
 * Facade for settings operations. Routes to specialized helper classes.
 *
 * Production Ready: Yes
 */
class Settings implements \LHA\Interfaces\SettingsInterface
{
    // ============================================================
    // Stub methods to satisfy SettingsInterface
    // These delegate to __call() for actual implementation
    // ============================================================

    public function init(): void {
        $this->__call('init', func_get_args());
    }

    public function get_upload_dir(): ?array {
        return $this->__call('get_upload_dir', func_get_args());
    }

    public function handle_manual_validation(): void {
        $this->__call('handle_manual_validation', func_get_args());
    }

    public function handle_manual_remediation(): void {
        $this->__call('handle_manual_remediation', func_get_args());
    }

    public function add_bulk_actions(): void {
        $this->__call('add_bulk_actions', func_get_args());
    }

    public function add_inline_checkbox_column(mixed $columns): array {
        return $this->__call('add_inline_checkbox_column', func_get_args());
    }

    public function add_admin_menu(): void {
        $this->__call('add_admin_menu', func_get_args());
    }

    public function register_settings_display(): void {
        $this->__call('register_settings_display', func_get_args());
    }

    public function sanitize_options($input): array {
        return $this->__call('sanitize_options', func_get_args());
    }

    public function sanitize_tools($input): array {
        return $this->__call('sanitize_tools', func_get_args());
    }

    public function sanitize_imported_settings(array $imported_settings): array {
        return $this->__call('sanitize_imported_settings', func_get_args());
    }

    public function default_options(): array {
        return $this->__call('default_options', func_get_args());
    }

    public function default_tools(): array {
        return $this->__call('default_tools', func_get_args());
    }

    public function get_list_option_key(string $list_type, string $item_type): string|false {
        return $this->__call('get_list_option_key', func_get_args());
    }

    public function register_save_post_handler(): void {
        $this->__call('register_save_post_handler', func_get_args());
    }

    public function register_classic_editor_scripts(): void {
        $this->__call('register_classic_editor_scripts', func_get_args());
    }

    public function fetch_assets_for_gutenberg(\WP_REST_Request $request): \WP_REST_Response | \WP_Error {
        return $this->__call('fetch_assets_for_gutenberg', func_get_args());
    }

    public function register_and_enqueue_js_asset(string $original_url,
        int $priority = 10,
        array $deps = [],
        bool $in_footer = true,
        $version = null): bool {
        return $this->__call('register_and_enqueue_js_asset', func_get_args());
    }

    public function register_and_enqueue_style_asset(string $original_url,
        int $priority = 10,
        array $deps = [],
        string $media = 'all',
        $version = null): bool {
        return $this->__call('register_and_enqueue_style_asset', func_get_args());
    }

    public function register_classic_editor_meta_box(): void {
        $this->__call('register_classic_editor_meta_box', func_get_args());
    }

    public function register_additional_rest_routes(): void {
        $this->__call('register_additional_rest_routes', func_get_args());
    }

    public function get_asset_item_schema(): array {
        return $this->__call('get_asset_item_schema', func_get_args());
    }

    public function render_settings_page(): void {
        $this->__call('render_settings_page', func_get_args());
    }

    public function render_manage_assets_page(): void {
        $this->__call('render_manage_assets_page', func_get_args());
    }

    public function render_maintenance_page(): void {
        $this->__call('render_maintenance_page', func_get_args());
    }

    public function render_error_logs_page(): void {
        $this->__call('render_error_logs_page', func_get_args());
    }

    public function render_admin_page(): void {
        $this->__call('render_admin_page', func_get_args());
    }

    public function enqueue_bulk_actions_script(string $hook_suffix): void {
        $this->__call('enqueue_bulk_actions_script', func_get_args());
    }

    public function title(string $title, $label_for_id = false, ?string $doc_url = null): string {
        return $this->__call('title', func_get_args());
    }

    public function print_input(array $args): void {
        $this->__call('print_input', func_get_args());
    }

    public function print_input_rows(array $args): void {
        $this->__call('print_input_rows', func_get_args());
    }

    public function print_purge_meta_action_control(array $args): void {
        $this->__call('print_purge_meta_action_control', func_get_args());
    }

    public function save_post_asset_order(int $post_id, \WP_Post $post): void {
        $this->__call('save_post_asset_order', func_get_args());
    }

    public function enqueue_classic_editor_scripts(string $hook_suffix): void {
        $this->__call('enqueue_classic_editor_scripts', func_get_args());
    }

    public function render_classic_editor_meta_box(\WP_Post $post): void {
        $this->__call('render_classic_editor_meta_box', func_get_args());
    }

    public function save_asset_order_rest(\WP_REST_Request $request): \WP_REST_Response | \WP_Error {
        return $this->__call('save_asset_order_rest', func_get_args());
    }

    public function rest_permissions_check(\WP_REST_Request $request): bool | \WP_Error {
        return $this->__call('rest_permissions_check', func_get_args());
    }

    public function get_option_group_for_key(string $option_key): string|false {
        return $this->__call('get_option_group_for_key', func_get_args());
    }

    public function register_hooks(): void {
        $this->__call('register_hooks', func_get_args());
    }

    public function register_admin_pages(): void {
        $this->__call('register_admin_pages', func_get_args());
    }

    public function init_admin_assets(): void {
        $this->__call('init_admin_assets', func_get_args());
    }

    public function register_meta_boxes(): void {
        $this->__call('register_meta_boxes', func_get_args());
    }

    public function render_asset_management_metabox(\WP_Post $post): void {
        $this->__call('render_asset_management_metabox', func_get_args());
    }

    public function reschedule_cron_events(): void {
        $this->__call('reschedule_cron_events', func_get_args());
    }

    public function trigger_immediate_task_processing(): void {
        $this->__call('trigger_immediate_task_processing', func_get_args());
    }

    public function trigger_automatic_asset_scan($old_value, $new_value): void {
        $this->__call('trigger_automatic_asset_scan', func_get_args());
    }

    public function get_sanitized_options(): array {
        return $this->__call('get_sanitized_options', func_get_args());
    }

    public function default_svg_options(): array {
        return $this->__call('default_svg_options', func_get_args());
    }

    public function sanitize_svg_options($input): array {
        return $this->__call('sanitize_svg_options', func_get_args());
    }

    public function is_debug_logging_enabled(): bool {
        return $this->__call('is_debug_logging_enabled', func_get_args());
    }

    public function render_scanner_page(): void {
        $this->__call('render_scanner_page', func_get_args());
    }

    public function render_section_scan_config(): void {
        $this->__call('render_section_scan_config', func_get_args());
    }

    public function render_field_scan_homepage(): void {
        $this->__call('render_field_scan_homepage', func_get_args());
    }

    public function render_field_scan_post_types(): void {
        $this->__call('render_field_scan_post_types', func_get_args());
    }

    public function render_field_max_pages_per_type(): void {
        $this->__call('render_field_max_pages_per_type', func_get_args());
    }

    public function render_field_scan_recent_only(): void {
        $this->__call('render_field_scan_recent_only', func_get_args());
    }

    public function render_field_immediate_asset_processing(): void {
        $this->__call('render_field_immediate_asset_processing', func_get_args());
    }

    public function render_field_custom_scan_urls(): void {
        $this->__call('render_field_custom_scan_urls', func_get_args());
    }

    public function render_field_specific_pages(): void {
        $this->__call('render_field_specific_pages', func_get_args());
    }

    public function sanitize_selfhost_settings(array $input): array {
        return $this->__call('sanitize_selfhost_settings', func_get_args());
    }

    public function get_default_selfhost_settings(): array {
        return $this->__call('get_default_selfhost_settings', func_get_args());
    }

    public function get_selfhost_setting(string $key, $default = null) {
        return $this->__call('get_selfhost_setting', func_get_args());
    }

    public function get_setting(string $key, $default = null) {
        return $this->__call('get_setting', func_get_args());
    }

    public function update_selfhost_setting(string $key, $value): bool {
        return $this->__call('update_selfhost_setting', func_get_args());
    }

    public function render_assets_log_page(): void {
        $this->__call('render_assets_log_page', func_get_args());
    }

    public function render_section_clarity(): void {
        $this->__call('render_section_clarity', func_get_args());
    }

    public function render_field_clarity_manual_refresh(): void {
        $this->__call('render_field_clarity_manual_refresh', func_get_args());
    }

    public function heartbeat_received(array $response, array $data): array {
        return $this->__call('heartbeat_received', func_get_args());
    }

    public function render_tools_page(): void {
        $this->__call('render_tools_page', func_get_args());
    }

    public function render_svg_settings_page(): void {
        $this->__call('render_svg_settings_page', func_get_args());
    }

    public static function get_all_exportable_options(): array {
        return self::__callStatic('get_all_exportable_options', func_get_args());
    }

    public static function validate_single_option(string $field_name, $value): array {
        return self::__callStatic('validate_single_option', func_get_args());
    }

    public static function sanitize_option_input(string $option_name, $value) {
        return self::__callStatic('sanitize_option_input', func_get_args());
    }

    public static function get_export_settings_defaults(): array {
        return self::__callStatic('get_export_settings_defaults', func_get_args());
    }

    public static function render_progress_bar(int $completed, int $total): void {
        self::__callStatic('render_progress_bar', func_get_args());
    }


    // Dependencies
    private LoggerInterface $logger;
    private \LHA\GetData $getdata;
    private \LHA\Initialize $initialize;
    private DatabaseInterface $database;
    private ?AssetValidatorInterface $assetValidator;
    private ?NormalizeInterface $normalize;
    private ?UrlProcessorInterface $urlProcessor;

    // Lazy-loaded helpers
    private ?SettingsRenderHelper $renderHelper = null;
    private ?SettingsRegisterHelper $registerHelper = null;
    private ?SettingsSanitizeHelper $sanitizeHelper = null;
    private ?SettingsQueryHelper $queryHelper = null;
    private ?SettingsSaveHelper $saveHelper = null;
    private ?SettingsUtilityHelper $utilityHelper = null;

    /**
     * Constructor
     */
    public function __construct(
        LoggerInterface $logger,
        \LHA\GetData $getdata,
        \LHA\Initialize $initialize,
        DatabaseInterface $database,
        ?AssetValidatorInterface $assetValidator = null,
        ?NormalizeInterface $normalize = null,
        ?UrlProcessorInterface $urlProcessor = null
    ) {
        $this->logger = $logger;
        $this->getdata = $getdata;
        $this->initialize = $initialize;
        $this->database = $database;
        $this->assetValidator = $assetValidator;
        $this->normalize = $normalize;
        $this->urlProcessor = $urlProcessor;
    }

    /**
     * Magic call method to route to helper classes
     */
    public function __call(string $name, array $arguments)
    {
        static $methodMap = [
            'render_asset_management_metabox' => 'SettingsRenderHelper',
            'render_settings_page' => 'SettingsRenderHelper',
            'render_manage_assets_page' => 'SettingsRenderHelper',
            'render_maintenance_page' => 'SettingsRenderHelper',
            'render_error_logs_page' => 'SettingsRenderHelper',
            'render_scanner_page' => 'SettingsRenderHelper',
            'render_admin_page' => 'SettingsRenderHelper',
            'render_classic_editor_meta_box' => 'SettingsRenderHelper',
            'render_section_scan_config' => 'SettingsRenderHelper',
            'render_field_scan_homepage' => 'SettingsRenderHelper',
            'render_field_scan_post_types' => 'SettingsRenderHelper',
            'render_field_max_pages_per_type' => 'SettingsRenderHelper',
            'render_field_scan_recent_only' => 'SettingsRenderHelper',
            'render_field_immediate_asset_processing' => 'SettingsRenderHelper',
            'render_field_custom_scan_urls' => 'SettingsRenderHelper',
            'render_field_specific_pages' => 'SettingsRenderHelper',
            'render_assets_log_page' => 'SettingsRenderHelper',
            'render_section_clarity' => 'SettingsRenderHelper',
            'render_field_clarity_manual_refresh' => 'SettingsRenderHelper',
            'render_tools_page' => 'SettingsRenderHelper',
            'render_svg_settings_page' => 'SettingsRenderHelper',
            'register_hooks' => 'SettingsRegisterHelper',
            'register_admin_pages' => 'SettingsRegisterHelper',
            'register_meta_boxes' => 'SettingsRegisterHelper',
            'register_settings_display' => 'SettingsRegisterHelper',
            'register_save_post_handler' => 'SettingsRegisterHelper',
            'register_classic_editor_scripts' => 'SettingsRegisterHelper',
            'register_and_enqueue_js_asset' => 'SettingsRegisterHelper',
            'register_and_enqueue_style_asset' => 'SettingsRegisterHelper',
            'register_classic_editor_meta_box' => 'SettingsRegisterHelper',
            'register_additional_rest_routes' => 'SettingsRegisterHelper',
            'sanitize_options' => 'SettingsSanitizeHelper',
            'sanitize_tools' => 'SettingsSanitizeHelper',
            'sanitize_imported_settings' => 'SettingsSanitizeHelper',
            'sanitize_svg_options' => 'SettingsSanitizeHelper',
            'sanitize_selfhost_settings' => 'SettingsSanitizeHelper',
            'get_upload_dir' => 'SettingsQueryHelper',
            'get_sanitized_options' => 'SettingsQueryHelper',
            'get_list_option_key' => 'SettingsQueryHelper',
            'get_asset_item_schema' => 'SettingsQueryHelper',
            'get_option_group_for_key' => 'SettingsQueryHelper',
            'get_default_selfhost_settings' => 'SettingsQueryHelper',
            'get_selfhost_setting' => 'SettingsQueryHelper',
            'get_setting' => 'SettingsQueryHelper',
            'update_selfhost_setting' => 'SettingsQueryHelper',
            'save_post_asset_order' => 'SettingsSaveHelper',
            'save_asset_order_rest' => 'SettingsSaveHelper',
            'init_admin_assets' => 'SettingsUtilityHelper',
            'init' => 'SettingsUtilityHelper',
            'handle_manual_validation' => 'SettingsUtilityHelper',
            'handle_manual_remediation' => 'SettingsUtilityHelper',
            'add_bulk_actions' => 'SettingsUtilityHelper',
            'add_inline_checkbox_column' => 'SettingsUtilityHelper',
            'add_admin_menu' => 'SettingsUtilityHelper',
            'reschedule_cron_events' => 'SettingsUtilityHelper',
            'trigger_immediate_task_processing' => 'SettingsUtilityHelper',
            'trigger_automatic_asset_scan' => 'SettingsUtilityHelper',
            'default_options' => 'SettingsUtilityHelper',
            'default_svg_options' => 'SettingsUtilityHelper',
            'default_tools' => 'SettingsUtilityHelper',
            'fetch_assets_for_gutenberg' => 'SettingsUtilityHelper',
            'is_debug_logging_enabled' => 'SettingsUtilityHelper',
            'enqueue_bulk_actions_script' => 'SettingsUtilityHelper',
            'title' => 'SettingsUtilityHelper',
            'print_input' => 'SettingsUtilityHelper',
            'print_input_rows' => 'SettingsUtilityHelper',
            'print_purge_meta_action_control' => 'SettingsUtilityHelper',
            'enqueue_classic_editor_scripts' => 'SettingsUtilityHelper',
            'rest_permissions_check' => 'SettingsUtilityHelper',
            'heartbeat_received' => 'SettingsUtilityHelper',
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
    public static function __callStatic($method, $arguments)
    {
        // Static methods
        $staticMethods = [
            'get_all_exportable_options',
            'validate_single_option',
            'sanitize_option_input',
            'render_progress_bar',
        ];

        // Check if it's a static method in SettingsValidationHelper
        if (in_array($method, $staticMethods)) {
            $helperClass = '\\LHA\\SettingsHelpers\\SettingsValidationHelper';
            return call_user_func_array([$helperClass, $method], $arguments);
        }

        // For other static calls, try to get an instance from container
        global $lha_container;

        if (isset($lha_container) && $lha_container instanceof \LHA\ServiceContainer) {
            try {
                $instance = $lha_container->get(Settings::class);
                if ($instance !== null) {
                    return $instance->$method(...$arguments);
                }
            } catch (\Exception $e) {
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
            'SettingsRenderHelper' => $this->getRenderHelper(),
            'SettingsRegisterHelper' => $this->getRegisterHelper(),
            'SettingsSanitizeHelper' => $this->getSanitizeHelper(),
            'SettingsQueryHelper' => $this->getQueryHelper(),
            'SettingsSaveHelper' => $this->getSaveHelper(),
            'SettingsUtilityHelper' => $this->getUtilityHelper(),
            default => throw new \InvalidArgumentException("Unknown helper: $helperClass"),
        };
    }

    private function getRenderHelper(): SettingsRenderHelper
    {
        if ($this->renderHelper === null) {
            $this->renderHelper = new SettingsRenderHelper(
                $this->logger,
                $this->getdata,
                $this->database,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->renderHelper;
    }

    private function getRegisterHelper(): SettingsRegisterHelper
    {
        if ($this->registerHelper === null) {
            $this->registerHelper = new SettingsRegisterHelper(
                $this->logger,
                $this->database,
                $this->initialize
            );
        }
        return $this->registerHelper;
    }

    private function getSanitizeHelper(): SettingsSanitizeHelper
    {
        if ($this->sanitizeHelper === null) {
            $this->sanitizeHelper = new SettingsSanitizeHelper(
                $this->logger,
                $this->database,
                $this->urlProcessor
            );
        }
        return $this->sanitizeHelper;
    }

    private function getQueryHelper(): SettingsQueryHelper
    {
        if ($this->queryHelper === null) {
            $this->queryHelper = new SettingsQueryHelper(
                $this->logger,
                $this->database,
                $this->normalize
            );
        }
        return $this->queryHelper;
    }

    private function getSaveHelper(): SettingsSaveHelper
    {
        if ($this->saveHelper === null) {
            $this->saveHelper = new SettingsSaveHelper(
                $this->logger,
                $this->database
            );
        }
        return $this->saveHelper;
    }

    private function getUtilityHelper(): SettingsUtilityHelper
    {
        if ($this->utilityHelper === null) {
            $this->utilityHelper = new SettingsUtilityHelper(
                $this->logger,
                $this->getdata,
                $this->database,
                $this->initialize,
                $this->assetValidator,
                $this->normalize,
                $this->urlProcessor
            );
        }
        return $this->utilityHelper;
    }
}
