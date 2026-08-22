CREATE TABLE "evolution_assets" (
	"id" serial PRIMARY KEY NOT NULL,
	"run_id" integer NOT NULL,
	"user_id" text NOT NULL,
	"kind" text NOT NULL,
	"content_type" text NOT NULL,
	"file_path" text NOT NULL,
	"size" integer NOT NULL,
	"created_at" timestamp NOT NULL
);
--> statement-breakpoint
CREATE TABLE "evolution_designs" (
	"id" serial PRIMARY KEY NOT NULL,
	"request_id" integer NOT NULL,
	"generation" integer NOT NULL,
	"html" text NOT NULL,
	"css" text,
	"metadata" jsonb,
	"created_at" timestamp NOT NULL
);
--> statement-breakpoint
CREATE TABLE "evolution_judge_scores" (
	"id" serial PRIMARY KEY NOT NULL,
	"design_id" integer NOT NULL,
	"agent" text NOT NULL,
	"score" double precision NOT NULL,
	"reasoning" text,
	"highlights" jsonb,
	"issues" jsonb,
	"recommendations" jsonb,
	"created_at" timestamp NOT NULL
);
--> statement-breakpoint
CREATE TABLE "evolution_nodes" (
	"id" serial PRIMARY KEY NOT NULL,
	"run_id" integer NOT NULL,
	"node_id" text NOT NULL,
	"parent_node_id" text,
	"generation" integer NOT NULL,
	"status" text NOT NULL,
	"fitness" double precision,
	"score" double precision,
	"label" text,
	"html_asset_id" integer,
	"thumbnail_asset_id" integer,
	"metadata" jsonb,
	"created_at" timestamp NOT NULL,
	"updated_at" timestamp NOT NULL,
	CONSTRAINT "evolution_nodes_run_id_node_id_unique" UNIQUE("run_id","node_id")
);
--> statement-breakpoint
CREATE TABLE "evolution_requests" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" text NOT NULL,
	"content" text NOT NULL,
	"mode" text DEFAULT 'standard' NOT NULL,
	"parameters" jsonb,
	"constraints" jsonb,
	"created_at" timestamp NOT NULL,
	"updated_at" timestamp NOT NULL
);
--> statement-breakpoint
CREATE TABLE "evolution_results" (
	"id" serial PRIMARY KEY NOT NULL,
	"request_id" integer NOT NULL,
	"best_design_id" integer,
	"summary" jsonb,
	"created_at" timestamp NOT NULL
);
--> statement-breakpoint
CREATE TABLE "evolution_runs" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" text NOT NULL,
	"evolution_id" text NOT NULL,
	"status" text NOT NULL,
	"name" text,
	"config" jsonb,
	"created_at" timestamp NOT NULL,
	"updated_at" timestamp NOT NULL,
	CONSTRAINT "evolution_runs_user_id_evolution_id_unique" UNIQUE("user_id","evolution_id")
);
--> statement-breakpoint
CREATE TABLE "evolution_screenshots" (
	"id" serial PRIMARY KEY NOT NULL,
	"design_id" integer NOT NULL,
	"asset_id" integer,
	"kind" text DEFAULT 'thumbnail' NOT NULL,
	"width" integer,
	"height" integer,
	"created_at" timestamp NOT NULL
);
--> statement-breakpoint
CREATE TABLE "idempotency_keys" (
	"id" serial PRIMARY KEY NOT NULL,
	"key" text NOT NULL,
	"user_id" text NOT NULL,
	"endpoint" text,
	"params" jsonb,
	"response" jsonb,
	"status_code" integer NOT NULL,
	"expires_at" timestamp,
	"created_at" timestamp NOT NULL,
	CONSTRAINT "idempotency_keys_key_user_id_unique" UNIQUE("key","user_id")
);
--> statement-breakpoint
ALTER TABLE "user_credentials" ADD COLUMN "is_default" boolean DEFAULT false;--> statement-breakpoint
ALTER TABLE "evolution_assets" ADD CONSTRAINT "evolution_assets_run_id_evolution_runs_id_fk" FOREIGN KEY ("run_id") REFERENCES "public"."evolution_runs"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_assets" ADD CONSTRAINT "evolution_assets_user_id_users_clerk_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("clerk_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_designs" ADD CONSTRAINT "evolution_designs_request_id_evolution_requests_id_fk" FOREIGN KEY ("request_id") REFERENCES "public"."evolution_requests"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_judge_scores" ADD CONSTRAINT "evolution_judge_scores_design_id_evolution_designs_id_fk" FOREIGN KEY ("design_id") REFERENCES "public"."evolution_designs"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_nodes" ADD CONSTRAINT "evolution_nodes_run_id_evolution_runs_id_fk" FOREIGN KEY ("run_id") REFERENCES "public"."evolution_runs"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_nodes" ADD CONSTRAINT "evolution_nodes_html_asset_id_evolution_assets_id_fk" FOREIGN KEY ("html_asset_id") REFERENCES "public"."evolution_assets"("id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_nodes" ADD CONSTRAINT "evolution_nodes_thumbnail_asset_id_evolution_assets_id_fk" FOREIGN KEY ("thumbnail_asset_id") REFERENCES "public"."evolution_assets"("id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_requests" ADD CONSTRAINT "evolution_requests_user_id_users_clerk_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("clerk_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_results" ADD CONSTRAINT "evolution_results_request_id_evolution_requests_id_fk" FOREIGN KEY ("request_id") REFERENCES "public"."evolution_requests"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_results" ADD CONSTRAINT "evolution_results_best_design_id_evolution_designs_id_fk" FOREIGN KEY ("best_design_id") REFERENCES "public"."evolution_designs"("id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_runs" ADD CONSTRAINT "evolution_runs_user_id_users_clerk_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("clerk_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_screenshots" ADD CONSTRAINT "evolution_screenshots_design_id_evolution_designs_id_fk" FOREIGN KEY ("design_id") REFERENCES "public"."evolution_designs"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evolution_screenshots" ADD CONSTRAINT "evolution_screenshots_asset_id_evolution_assets_id_fk" FOREIGN KEY ("asset_id") REFERENCES "public"."evolution_assets"("id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "idempotency_keys" ADD CONSTRAINT "idempotency_keys_user_id_users_clerk_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("clerk_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "evolution_designs_request_generation_idx" ON "evolution_designs" USING btree ("request_id","generation");--> statement-breakpoint
CREATE INDEX "evolution_judge_scores_design_id_idx" ON "evolution_judge_scores" USING btree ("design_id");--> statement-breakpoint
CREATE INDEX "evolution_requests_user_id_idx" ON "evolution_requests" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "evolution_results_request_id_idx" ON "evolution_results" USING btree ("request_id");--> statement-breakpoint
CREATE INDEX "evolution_screenshots_design_id_idx" ON "evolution_screenshots" USING btree ("design_id");