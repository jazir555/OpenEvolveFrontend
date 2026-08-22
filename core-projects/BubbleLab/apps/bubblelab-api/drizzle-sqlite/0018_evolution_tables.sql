CREATE TABLE `evolution_assets` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`run_id` integer NOT NULL,
	`user_id` text NOT NULL,
	`kind` text NOT NULL,
	`content_type` text NOT NULL,
	`file_path` text NOT NULL,
	`size` integer NOT NULL,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`run_id`) REFERENCES `evolution_runs`(`id`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`clerk_id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `evolution_designs` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`request_id` integer NOT NULL,
	`generation` integer NOT NULL,
	`html` text NOT NULL,
	`css` text,
	`metadata` text,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`request_id`) REFERENCES `evolution_requests`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `evolution_designs_request_generation_idx` ON `evolution_designs` (`request_id`,`generation`);--> statement-breakpoint
CREATE TABLE `evolution_judge_scores` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`design_id` integer NOT NULL,
	`agent` text NOT NULL,
	`score` real NOT NULL,
	`reasoning` text,
	`highlights` text,
	`issues` text,
	`recommendations` text,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`design_id`) REFERENCES `evolution_designs`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `evolution_judge_scores_design_id_idx` ON `evolution_judge_scores` (`design_id`);--> statement-breakpoint
CREATE TABLE `evolution_nodes` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`run_id` integer NOT NULL,
	`node_id` text NOT NULL,
	`parent_node_id` text,
	`generation` integer NOT NULL,
	`status` text NOT NULL,
	`fitness` real,
	`score` real,
	`label` text,
	`html_asset_id` integer,
	`thumbnail_asset_id` integer,
	`metadata` text,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	FOREIGN KEY (`run_id`) REFERENCES `evolution_runs`(`id`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`html_asset_id`) REFERENCES `evolution_assets`(`id`) ON UPDATE no action ON DELETE set null,
	FOREIGN KEY (`thumbnail_asset_id`) REFERENCES `evolution_assets`(`id`) ON UPDATE no action ON DELETE set null
);
--> statement-breakpoint
CREATE UNIQUE INDEX `evolution_nodes_run_id_node_id_unique` ON `evolution_nodes` (`run_id`,`node_id`);--> statement-breakpoint
CREATE TABLE `evolution_requests` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`user_id` text NOT NULL,
	`content` text NOT NULL,
	`mode` text DEFAULT 'standard' NOT NULL,
	`parameters` text,
	`constraints` text,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`clerk_id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `evolution_requests_user_id_idx` ON `evolution_requests` (`user_id`);--> statement-breakpoint
CREATE TABLE `evolution_results` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`request_id` integer NOT NULL,
	`best_design_id` integer,
	`summary` text,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`request_id`) REFERENCES `evolution_requests`(`id`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`best_design_id`) REFERENCES `evolution_designs`(`id`) ON UPDATE no action ON DELETE set null
);
--> statement-breakpoint
CREATE INDEX `evolution_results_request_id_idx` ON `evolution_results` (`request_id`);--> statement-breakpoint
CREATE TABLE `evolution_runs` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`user_id` text NOT NULL,
	`evolution_id` text NOT NULL,
	`status` text NOT NULL,
	`name` text,
	`config` text,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`clerk_id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `evolution_runs_user_id_evolution_id_unique` ON `evolution_runs` (`user_id`,`evolution_id`);--> statement-breakpoint
CREATE TABLE `evolution_screenshots` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`design_id` integer NOT NULL,
	`asset_id` integer,
	`kind` text DEFAULT 'thumbnail' NOT NULL,
	`width` integer,
	`height` integer,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`design_id`) REFERENCES `evolution_designs`(`id`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`asset_id`) REFERENCES `evolution_assets`(`id`) ON UPDATE no action ON DELETE set null
);
--> statement-breakpoint
CREATE INDEX `evolution_screenshots_design_id_idx` ON `evolution_screenshots` (`design_id`);--> statement-breakpoint
CREATE TABLE `idempotency_keys` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`key` text NOT NULL,
	`user_id` text NOT NULL,
	`endpoint` text,
	`params` text,
	`response` text,
	`status_code` integer NOT NULL,
	`expires_at` integer,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`clerk_id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `idempotency_keys_key_user_id_unique` ON `idempotency_keys` (`key`,`user_id`);--> statement-breakpoint
ALTER TABLE `user_credentials` ADD `is_default` integer DEFAULT false;