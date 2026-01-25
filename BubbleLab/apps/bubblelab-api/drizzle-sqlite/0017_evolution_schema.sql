CREATE TABLE `evolution_requests` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`user_id` text NOT NULL,
	`content` text NOT NULL,
	`mode` text DEFAULT 'standard' NOT NULL,
	`parameters` text,
	`constraints` text,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`clerk_id`) ON DELETE cascade ON UPDATE no action
);
CREATE INDEX `evolution_requests_user_id_idx` ON `evolution_requests` (`user_id`);
CREATE TABLE `evolution_designs` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`request_id` integer NOT NULL,
	`generation` integer NOT NULL,
	`html` text NOT NULL,
	`css` text,
	`metadata` text,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`request_id`) REFERENCES `evolution_requests`(`id`) ON DELETE cascade ON UPDATE no action
);
CREATE INDEX `evolution_designs_request_generation_idx` ON `evolution_designs` (`request_id`,`generation`);
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
	FOREIGN KEY (`design_id`) REFERENCES `evolution_designs`(`id`) ON DELETE cascade ON UPDATE no action
);
CREATE INDEX `evolution_judge_scores_design_id_idx` ON `evolution_judge_scores` (`design_id`);
CREATE TABLE `evolution_results` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`request_id` integer NOT NULL,
	`best_design_id` integer,
	`summary` text,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`request_id`) REFERENCES `evolution_requests`(`id`) ON DELETE cascade ON UPDATE no action,
	FOREIGN KEY (`best_design_id`) REFERENCES `evolution_designs`(`id`) ON DELETE set null ON UPDATE no action
);
CREATE INDEX `evolution_results_request_id_idx` ON `evolution_results` (`request_id`);
CREATE TABLE `evolution_screenshots` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`design_id` integer NOT NULL,
	`asset_id` integer,
	`kind` text DEFAULT 'thumbnail' NOT NULL,
	`width` integer,
	`height` integer,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`design_id`) REFERENCES `evolution_designs`(`id`) ON DELETE cascade ON UPDATE no action,
	FOREIGN KEY (`asset_id`) REFERENCES `evolution_assets`(`id`) ON DELETE set null ON UPDATE no action
);
CREATE INDEX `evolution_screenshots_design_id_idx` ON `evolution_screenshots` (`design_id`);
