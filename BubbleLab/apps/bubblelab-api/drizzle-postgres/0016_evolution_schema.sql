CREATE TABLE "evolution_requests" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" text NOT NULL,
	"content" text NOT NULL,
	"mode" text DEFAULT 'standard' NOT NULL,
	"parameters" jsonb,
	"constraints" jsonb,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "evolution_designs" (
	"id" serial PRIMARY KEY NOT NULL,
	"request_id" integer NOT NULL,
	"generation" integer NOT NULL,
	"html" text NOT NULL,
	"css" text,
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now() NOT NULL
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
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "evolution_results" (
	"id" serial PRIMARY KEY NOT NULL,
	"request_id" integer NOT NULL,
	"best_design_id" integer,
	"summary" jsonb,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "evolution_screenshots" (
	"id" serial PRIMARY KEY NOT NULL,
	"design_id" integer NOT NULL,
	"asset_id" integer,
	"kind" text DEFAULT 'thumbnail' NOT NULL,
	"width" integer,
	"height" integer,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE INDEX "evolution_requests_user_id_idx" ON "evolution_requests" ("user_id");
--> statement-breakpoint
CREATE INDEX "evolution_designs_request_generation_idx" ON "evolution_designs" ("request_id","generation");
--> statement-breakpoint
CREATE INDEX "evolution_judge_scores_design_id_idx" ON "evolution_judge_scores" ("design_id");
--> statement-breakpoint
CREATE INDEX "evolution_results_request_id_idx" ON "evolution_results" ("request_id");
--> statement-breakpoint
CREATE INDEX "evolution_screenshots_design_id_idx" ON "evolution_screenshots" ("design_id");
--> statement-breakpoint
ALTER TABLE "evolution_requests" ADD CONSTRAINT "evolution_requests_user_id_users_clerk_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("clerk_id") ON DELETE cascade ON UPDATE no action;
--> statement-breakpoint
ALTER TABLE "evolution_designs" ADD CONSTRAINT "evolution_designs_request_id_evolution_requests_id_fk" FOREIGN KEY ("request_id") REFERENCES "public"."evolution_requests"("id") ON DELETE cascade ON UPDATE no action;
--> statement-breakpoint
ALTER TABLE "evolution_judge_scores" ADD CONSTRAINT "evolution_judge_scores_design_id_evolution_designs_id_fk" FOREIGN KEY ("design_id") REFERENCES "public"."evolution_designs"("id") ON DELETE cascade ON UPDATE no action;
--> statement-breakpoint
ALTER TABLE "evolution_results" ADD CONSTRAINT "evolution_results_request_id_evolution_requests_id_fk" FOREIGN KEY ("request_id") REFERENCES "public"."evolution_requests"("id") ON DELETE cascade ON UPDATE no action;
--> statement-breakpoint
ALTER TABLE "evolution_results" ADD CONSTRAINT "evolution_results_best_design_id_evolution_designs_id_fk" FOREIGN KEY ("best_design_id") REFERENCES "public"."evolution_designs"("id") ON DELETE set null ON UPDATE no action;
--> statement-breakpoint
ALTER TABLE "evolution_screenshots" ADD CONSTRAINT "evolution_screenshots_design_id_evolution_designs_id_fk" FOREIGN KEY ("design_id") REFERENCES "public"."evolution_designs"("id") ON DELETE cascade ON UPDATE no action;
--> statement-breakpoint
ALTER TABLE "evolution_screenshots" ADD CONSTRAINT "evolution_screenshots_asset_id_evolution_assets_id_fk" FOREIGN KEY ("asset_id") REFERENCES "public"."evolution_assets"("id") ON DELETE set null ON UPDATE no action;
