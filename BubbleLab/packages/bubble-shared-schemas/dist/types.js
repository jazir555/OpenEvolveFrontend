// Define CredentialType enum here to avoid circular dependencies
export var CredentialType;
(function (CredentialType) {
    // AI Credentials
    CredentialType["OPENAI_CRED"] = "OPENAI_CRED";
    CredentialType["GOOGLE_GEMINI_CRED"] = "GOOGLE_GEMINI_CRED";
    CredentialType["ANTHROPIC_CRED"] = "ANTHROPIC_CRED";
    CredentialType["OPENROUTER_CRED"] = "OPENROUTER_CRED";
    CredentialType["DEEPSEEK_CRED"] = "DEEPSEEK_CRED";
    // Search Credentials
    CredentialType["FIRECRAWL_API_KEY"] = "FIRECRAWL_API_KEY";
    // Database Credentials
    CredentialType["DATABASE_CRED"] = "DATABASE_CRED";
    // Communication Credentials
    CredentialType["SLACK_CRED"] = "SLACK_CRED";
    CredentialType["TELEGRAM_BOT_TOKEN"] = "TELEGRAM_BOT_TOKEN";
    // Email Credentials
    CredentialType["RESEND_CRED"] = "RESEND_CRED";
    // Storage Credentials
    CredentialType["CLOUDFLARE_R2_ACCESS_KEY"] = "CLOUDFLARE_R2_ACCESS_KEY";
    CredentialType["CLOUDFLARE_R2_SECRET_KEY"] = "CLOUDFLARE_R2_SECRET_KEY";
    CredentialType["CLOUDFLARE_R2_ACCOUNT_ID"] = "CLOUDFLARE_R2_ACCOUNT_ID";
    // Scraping Credentials
    CredentialType["APIFY_CRED"] = "APIFY_CRED";
    // Voice Credentials
    CredentialType["ELEVENLABS_API_KEY"] = "ELEVENLABS_API_KEY";
    // OAuth Credentials
    CredentialType["GOOGLE_DRIVE_CRED"] = "GOOGLE_DRIVE_CRED";
    CredentialType["GMAIL_CRED"] = "GMAIL_CRED";
    CredentialType["GOOGLE_SHEETS_CRED"] = "GOOGLE_SHEETS_CRED";
    CredentialType["GOOGLE_CALENDAR_CRED"] = "GOOGLE_CALENDAR_CRED";
    CredentialType["FUB_CRED"] = "FUB_CRED";
    CredentialType["NOTION_OAUTH_TOKEN"] = "NOTION_OAUTH_TOKEN";
    // Development Platform Credentials
    CredentialType["GITHUB_TOKEN"] = "GITHUB_TOKEN";
    CredentialType["GITHUB_CRED"] = "GITHUB_CRED";
    // Browser Automation Credentials
    CredentialType["AGI_API_KEY"] = "AGI_API_KEY";
    // Database/Storage Credentials
    CredentialType["AIRTABLE_CRED"] = "AIRTABLE_CRED";
    CredentialType["ELASTICSEARCH_CRED"] = "ELASTICSEARCH_CRED";
    // Payment Credentials
    CredentialType["STRIPE_CRED"] = "STRIPE_CRED";
    CredentialType["SENDGRID_CRED"] = "SENDGRID_CRED";
    CredentialType["TWILIO_CRED"] = "TWILIO_CRED";
    // InsForge Credentials
    CredentialType["INSFORGE_BASE_URL"] = "INSFORGE_BASE_URL";
    CredentialType["INSFORGE_API_KEY"] = "INSFORGE_API_KEY";
    // PostgreSQL Credentials
    CredentialType["POSTGRESQL_CRED"] = "POSTGRESQL_CRED";
    // Qdrant Credentials
    CredentialType["QDRANT_CRED"] = "QDRANT_CRED";
    // Redis Credentials
    CredentialType["REDIS_CRED"] = "REDIS_CRED";
    // OAuth Credentials
    CredentialType["OAUTH_TOKEN"] = "OAUTH_TOKEN";
    // Custom Authentication Credentials
    CredentialType["CUSTOM_AUTH_KEY"] = "CUSTOM_AUTH_KEY";
})(CredentialType || (CredentialType = {}));
//# sourceMappingURL=types.js.map