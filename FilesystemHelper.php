<?php
/**
 * Filesystem Helper - Global wrapper for WP_Filesystem with Docker/native PHP fallback
 * 
 * This class provides a unified interface for filesystem operations that works in:
 * - Standard WordPress environments
 * - Docker containers
 * - Any environment where WP_Filesystem may fail
 * 
 * Usage: Replace all WP_Filesystem calls with FilesystemHelper calls
 * Example: FilesystemHelper::put_contents($path, $content) instead of $wp_filesystem->put_contents()
 */

namespace LHA;

class FilesystemHelper {
    
    /**
     * Cached filesystem instance
     * @var \WP_Filesystem_Base|null
     */
    private static $wp_filesystem = null;
    
    /**
     * Whether we've attempted to initialize WP_Filesystem
     * @var bool
     */
    private static $initialized = false;
    
    /**
     * Cached Docker detection result
     * @var bool|null
     */
    private static $is_docker = null;
    
    /**
     * Check if WP_Filesystem needs fallback support
     * 
     * @return bool True if fallback is needed
     */
    private static function needs_fallback(): bool {
        static $needs_fallback = null;
        
        if ($needs_fallback !== null) {
            return $needs_fallback;
        }
        
        // Check if WP_Filesystem is working properly
        global $wp_filesystem;
        
        if (!isset($wp_filesystem) || !is_object($wp_filesystem)) {
            $needs_fallback = true;
            return true;
        }
        
        // Check if it's using ftpsockets (Docker issue)
        $class_name = get_class($wp_filesystem);
        if (strpos($class_name, 'ftpsockets') !== false) {
            $needs_fallback = true;
            return true;
        }
        
        // WP_Filesystem is working fine
        $needs_fallback = false;
        return false;
    }
    
    /**
     * Detect if running in Docker container
     * 
     * Delegates to HostingEnvironment for centralized Docker detection.
     * Falls back to simple check if HostingEnvironment is not available.
     * 
     * @return bool True if running in Docker
     */
    public static function is_docker(): bool {
        // Return cached result if already checked
        if (self::$is_docker !== null) {
            return self::$is_docker;
        }
        
        // Delegate to HostingEnvironment for comprehensive detection
        if (class_exists('\LHA\HostingEnvironment')) {
            return self::$is_docker = HostingEnvironment::is_docker_environment();
        }
        
        // Lightweight fallback if HostingEnvironment not available
        if (@file_exists('/.dockerenv')) {
            return self::$is_docker = true;
        }
        
        return self::$is_docker = false;
    }
    
    /**
     * Initialize WP_Filesystem with Docker support
     * 
     * @return bool True if WP_Filesystem is available
     */
    private static function init(): bool {
        if (self::$initialized) {
            return self::$wp_filesystem !== null;
        }
        
        self::$initialized = true;
        
        global $wp_filesystem;
        
        // Check if already initialized
        if (isset($wp_filesystem) && is_object($wp_filesystem)) {
            self::$wp_filesystem = $wp_filesystem;
            return true;
        }
        
        // Load WP_Filesystem
        if (!function_exists('WP_Filesystem')) {
            require_once ABSPATH . 'wp-admin/includes/file.php';
        }
        
        // Only force direct method in Docker environments
        // This optimizes performance for non-Docker hosting
        if (self::is_docker() && !defined('FS_METHOD')) {
            // Check if filesystem is actually writable before forcing direct
            $upload_dir = wp_upload_dir();
            $is_writable = !empty($upload_dir['basedir']) && is_writable($upload_dir['basedir']);
            
            if ($is_writable) {
                define('FS_METHOD', 'direct');
            }
        }
        
        // Try to initialize
        @WP_Filesystem();
        
        if (isset($wp_filesystem) && is_object($wp_filesystem)) {
            self::$wp_filesystem = $wp_filesystem;
            return true;
        }
        
        return false;
    }
    
    /**
     * Check if a file or directory exists
     * 
     * @param string $path Path to check
     * @return bool True if exists
     */
    public static function exists(string $path): bool {
        self::init();
        
        // If WP_Filesystem works fine, use it exclusively
        if (self::$wp_filesystem && !self::needs_fallback()) {
            return self::$wp_filesystem->exists($path);
        }
        
        // Try WP_Filesystem first, fallback to native PHP
        if (self::$wp_filesystem && self::$wp_filesystem->exists($path)) {
            return true;
        }
        
        // Fallback to native PHP (Docker/ftpsockets environments)
        return file_exists($path);
    }
    
    /**
     * Check if path is a directory
     * 
     * @param string $path Path to check
     * @return bool True if directory
     */
    public static function is_dir(string $path): bool {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem && self::$wp_filesystem->is_dir($path)) {
            return true;
        }
        
        // Fallback to native PHP
        return is_dir($path);
    }
    
    /**
     * Check if path is a file
     * 
     * @param string $path Path to check
     * @return bool True if file
     */
    public static function is_file(string $path): bool {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem && self::$wp_filesystem->is_file($path)) {
            return true;
        }
        
        // Fallback to native PHP
        return is_file($path);
    }
    
    /**
     * Get file contents
     * 
     * @param string $path Path to file
     * @return string|false File contents or false on failure
     */
    public static function get_contents(string $path): string|false {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem) {
            $contents = self::$wp_filesystem->get_contents($path);
            if ($contents !== false) {
                return $contents;
            }
        }
        
        // Fallback to native PHP
        return @file_get_contents($path);
    }
    
    /**
     * Write contents to file
     * 
     * @param string $path Path to file
     * @param string $contents Contents to write
     * @param int|null $chmod Optional chmod value (default: 0644)
     * @return bool True on success
     */
    public static function put_contents(string $path, string $contents, ?int $chmod = null): bool {
        self::init();
        
        if ($chmod === null) {
            $chmod = defined('FS_CHMOD_FILE') ? FS_CHMOD_FILE : 0644;
        }
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem) {
            $result = self::$wp_filesystem->put_contents($path, $contents, $chmod);
            if ($result) {
                return true;
            }
        }
        
        // Fallback to native PHP if directory is writable
        $dir = dirname($path);
        if (is_writable($dir) || !file_exists($dir)) {
            $result = @file_put_contents($path, $contents);
            if ($result !== false) {
                @chmod($path, $chmod);
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * Delete a file
     * 
     * @param string $path Path to file
     * @return bool True on success
     */
    public static function delete(string $path): bool {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem && self::$wp_filesystem->delete($path)) {
            return true;
        }
        
        // Try to make file writable first in case of permission issues (e.g. .htaccess files)
        @chmod($path, 0644);
        // Fallback to native PHP
        return @unlink($path);
    }
    
    /**
     * Create a directory
     * 
     * @param string $path Path to directory
     * @param int|null $chmod Optional chmod value (default: 0755)
     * @return bool True on success
     */
    public static function mkdir(string $path, ?int $chmod = null): bool {
        self::init();
        
        if ($chmod === null) {
            $chmod = defined('FS_CHMOD_DIR') ? FS_CHMOD_DIR : 0755;
        }
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem && self::$wp_filesystem->mkdir($path, $chmod)) {
            return true;
        }
        
        // Fallback to native PHP
        return @mkdir($path, $chmod, true);
    }
    
    /**
     * Remove a directory
     * 
     * @param string $path Path to directory
     * @param bool $recursive Remove recursively
     * @return bool True on success
     */
    public static function rmdir(string $path, bool $recursive = false): bool {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem && self::$wp_filesystem->rmdir($path, $recursive)) {
            return true;
        }
        
        // Fallback to native PHP
        if ($recursive) {
            return self::rmdir_recursive($path);
        }
        
        return @rmdir($path);
    }
    
    /**
     * Recursively remove directory (native PHP fallback)
     *
     * @param string $path Path to directory
     * @return bool True on success
     */
    private static function rmdir_recursive(string $path): bool {
        if (!is_dir($path)) {
            return false;
        }

        $files = @scandir($path);
        if ($files === false) {
            return false; // Directory couldn't be read
        }

        $files = array_diff($files, ['.', '..']);

        foreach ($files as $file) {
            $file_path = $path . '/' . $file;

            if (is_dir($file_path)) {
                // Recursively delete subdirectory
                if (!self::rmdir_recursive($file_path)) {
                    return false; // If subdirectory deletion fails, return false
                }
            } else {
                // Try to make file writable first in case of permission issues (e.g. .htaccess files)
                @chmod($file_path, 0666);
                clearstatcache(); // Clear stat cache to refresh permissions info

                // Special handling for .htaccess files which may have restrictive permissions in Docker
                if (basename($file_path) === '.htaccess') {
                    // Avoid shell_exec which can cause hangs in Docker environments
                    // Instead use pure PHP approaches
                    clearstatcache(true, $file_path); // Clear stat cache for the specific file
                }

                $unlink_result = @unlink($file_path);
                if (!$unlink_result) {
                    // If unlink fails, try making the file writable first
                    @chmod($file_path, 0666);
                    @clearstatcache(true, $file_path);
                    $unlink_result = @unlink($file_path);
                }

                if (!$unlink_result) {
                    // If unlink still fails, return false
                    return false;
                }
            }
        }

        // Try to make directory writable first in case of permission issues
        @chmod($path, 0755);
        return @rmdir($path);
    }
    
    /**
     * Copy a file
     * 
     * @param string $source Source path
     * @param string $destination Destination path
     * @param bool $overwrite Overwrite if exists
     * @return bool True on success
     */
    public static function copy(string $source, string $destination, bool $overwrite = false): bool {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem && self::$wp_filesystem->copy($source, $destination, $overwrite)) {
            return true;
        }
        
        // Fallback to native PHP
        if (!$overwrite && file_exists($destination)) {
            return false;
        }
        
        return @copy($source, $destination);
    }
    
    /**
     * Move a file
     * 
     * @param string $source Source path
     * @param string $destination Destination path
     * @param bool $overwrite Overwrite if exists
     * @return bool True on success
     */
    public static function move(string $source, string $destination, bool $overwrite = false): bool {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem && self::$wp_filesystem->move($source, $destination, $overwrite)) {
            return true;
        }
        
        // Fallback to native PHP
        if (!$overwrite && file_exists($destination)) {
            return false;
        }
        
        return @rename($source, $destination);
    }
    
    /**
     * Get file modification time
     * 
     * @param string $path Path to file
     * @return int|false Modification time or false on failure
     */
    public static function mtime(string $path): int|false {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem) {
            $mtime = self::$wp_filesystem->mtime($path);
            if ($mtime !== false) {
                return $mtime;
            }
        }
        
        // Fallback to native PHP
        return @filemtime($path);
    }
    
    /**
     * Get file size
     * 
     * @param string $path Path to file
     * @return int|false File size or false on failure
     */
    public static function size(string $path): int|false {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem) {
            $size = self::$wp_filesystem->size($path);
            if ($size !== false) {
                return $size;
            }
        }
        
        // Fallback to native PHP
        return @filesize($path);
    }
    
    /**
     * Check if path is writable
     * 
     * @param string $path Path to check
     * @return bool True if writable
     */
    public static function is_writable(string $path): bool {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem && self::$wp_filesystem->is_writable($path)) {
            return true;
        }
        
        // Fallback to native PHP
        return is_writable($path);
    }
    
    /**
     * Check if path is readable
     * 
     * @param string $path Path to check
     * @return bool True if readable
     */
    public static function is_readable(string $path): bool {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem && self::$wp_filesystem->is_readable($path)) {
            return true;
        }
        
        // Fallback to native PHP
        return is_readable($path);
    }
    
    /**
     * Change file permissions
     *
     * @param string $path Path to file/directory
     * @param int $mode Permission mode
     * @param bool $recursive Apply recursively
     * @return bool True on success
     */
    public static function chmod(string $path, int $mode, bool $recursive = false): bool {
        self::init();

        // Try WP_Filesystem first (handles recursive properly)
        if (self::$wp_filesystem && self::$wp_filesystem->chmod($path, $mode, $recursive)) {
            return true;
        }

        // Fallback to native PHP
        if (!$recursive) {
            return @chmod($path, $mode);
        }

        // For recursive operations with native PHP fallback
        if (!is_dir($path)) {
            return @chmod($path, $mode);
        }

        // Recursive: chmod the directory itself
        $result = @chmod($path, $mode);
        if (!$result) {
            return false;
        }

        // Then recursively chmod contents
        $files = @scandir($path);
        if ($files === false) {
            return true; // Directory exists but can't be read, continue with success
        }

        foreach ($files as $file) {
            if ($file === '.' || $file === '..') {
                continue;
            }

            $file_path = rtrim($path, '/') . '/' . $file;
            if (is_dir($file_path)) {
                // Recursive call for subdirectory
                $sub_result = self::chmod($file_path, $mode, true);
                if (!$sub_result) {
                    $result = false; // Don't return false immediately, continue with other files
                }
            } else {
                // chmod for file
                $file_result = @chmod($file_path, $mode);
                if (!$file_result) {
                    $result = false;
                }
            }
        }

        return $result;
    }
    
    /**
     * Get directory listing
     * 
     * @param string $path Path to directory
     * @return array|false Array of files/directories or false on failure
     */
    public static function dirlist(string $path): array|false {
        self::init();
        
        // Try WP_Filesystem first
        if (self::$wp_filesystem) {
            $list = self::$wp_filesystem->dirlist($path);
            if ($list !== false) {
                return $list;
            }
        }
        
        // Fallback to native PHP
        if (!is_dir($path)) {
            return false;
        }
        
        $files = @scandir($path);
        if ($files === false) {
            return false;
        }
        
        $result = [];
        foreach ($files as $file) {
            if ($file === '.' || $file === '..') {
                continue;
            }
            
            $file_path = trailingslashit($path) . $file;
            $result[$file] = [
                'name' => $file,
                'type' => is_dir($file_path) ? 'd' : 'f',
                'size' => @filesize($file_path),
                'lastmodunix' => @filemtime($file_path),
            ];
        }
        
        return $result;
    }
}
