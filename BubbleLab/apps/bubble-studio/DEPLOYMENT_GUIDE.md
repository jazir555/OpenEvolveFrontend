# Deployment Guide

OpenEvolve Frontend - BubbleLab React UI

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Building for Production](#building-for-production)
- [Deployment Options](#deployment-options)
- [Post-Deployment](#post-deployment)
- [Monitoring](#monitoring)

---

## Prerequisites

### Required:
- Node.js 18+ and npm
- Python 3.9+
- Git
- Domain name (for production)

### Accounts:
- GitHub (for repository)
- Vercel/Netlify/AWS (for hosting)
- Clerk (for authentication)
- OpenAI/LLM provider (for API access)

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/openevolve-frontend.git
cd openevolve-frontend/BubbleLab
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Environment Variables

Create `.env.production`:

```env
# API Configuration
VITE_API_BASE_URL=https://api.your-domain.com

# Clerk Authentication
VITE_CLERK_PUBLISHABLE_KEY=pk_live_...

# Optional: Analytics
VITE_GA_TRACKING_ID=G-XXXXXXXXXX

# Optional: Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_BENCHMARKS=true
```

### 4. Python Backend Setup

```bash
# Install Python dependencies
pip install -r api_bridge_requirements.txt

# Configure environment
export OPENAI_API_KEY="sk-..."
export CLERK_SECRET_KEY="sk_live_..."
```

---

## Building for Production

### Development Build

```bash
npm run dev
```

Runs on `http://localhost:5173`

### Production Build

```bash
npm run build
```

Creates optimized build in `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

---

## Deployment Options

### Option 1: Vercel (Recommended)

Vercel provides zero-config deployment for React apps.

#### Deploy to Vercel:

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Login:**
   ```bash
   vercel login
   ```

3. **Deploy:**
   ```bash
   cd apps/bubble-studio
   vercel --prod
   ```

4. **Configure Environment Variables:**
   - Go to Vercel Dashboard
   - Project Settings → Environment Variables
   - Add all variables from `.env.production`

#### Automatic Deployments:

Vercel automatically deploys when you push to `main` branch:
```bash
git push origin main
```

---

### Option 2: Netlify

#### Deploy to Netlify:

1. **Install Netlify CLI:**
   ```bash
   npm i -g netlify-cli
   ```

2. **Build:**
   ```bash
   npm run build
   ```

3. **Deploy:**
   ```bash
   netlify deploy --prod --dir=apps/bubble-studio/dist
   ```

#### Netlify Configuration:

Create `netlify.toml`:
```toml
[build]
  command = "npm run build"
  publish = "apps/bubble-studio/dist"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

---

### Option 3: AWS S3 + CloudFront

#### Deploy to AWS:

1. **Build the app:**
   ```bash
   npm run build
   ```

2. **Sync to S3:**
   ```bash
   aws s3 sync apps/bubble-studio/dist s3://your-bucket --delete
   ```

3. **Invalidate CloudFront cache:**
   ```bash
   aws cloudfront create-invalidation --distribution-id YOUR_ID --paths "/*"
   ```

---

### Option 4: Docker

#### Create Dockerfile:

```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
COPY apps/bubble-studio ./apps/bubble-studio
RUN npm ci
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/apps/bubble-studio/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### Build and Run:

```bash
docker build -t openevolve-frontend .
docker run -p 80:80 openevolve-frontend
```

---

## Backend Deployment

### Deploy API Bridge

#### Using Systemd (Linux):

Create `/etc/systemd/system/openevolve-api.service`:

```ini
[Unit]
Description=OpenEvolve API Bridge
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/openevolve-frontend
Environment="PATH=/var/www/openevolve-frontend/venv/bin"
ExecStart=/var/www/openevolve-frontend/venv/bin/uvicorn api_bridge:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl start openevolve-api
sudo systemctl enable openevolve-api
```

---

### Using Gunicorn (Production):

```bash
pip install gunicorn

gunicorn api_bridge:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8001 \
  --access-logfile - \
  --error-logfile -
```

---

### Using Supervisor:

Create `/etc/supervisor/conf.d/openevolve-api.conf`:

```ini
[program:openevolve-api]
command=/var/www/openevolve-frontend/venv/bin/uvicorn api_bridge:app --host 0.0.0.0 --port 8001
directory=/var/www/openevolve-frontend
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/openevolve-api.err.log
stdout_logfile=/var/log/openevolve-api.out.log
```

---

## Reverse Proxy Configuration

### Nginx Configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/your-domain.com.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.com.key;

    # React app
    location / {
        root /var/www/openevolve-frontend/BubbleLab/apps/bubble-studio/dist;
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api/ {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # SSE streaming
    location /stream/ {
        proxy_pass http://localhost:8001;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_cache off;
    }
}
```

---

## Post-Deployment

### 1. Health Checks

Verify API is running:
```bash
curl https://your-domain.com/api/health
```

### 2. Test Authentication

1. Navigate to `https://your-domain.com`
2. Click "Sign In"
3. Complete Clerk authentication flow
4. Verify you're logged in

### 3. Test Workflow Creation

1. Click "Create Workflow"
2. Fill out the form
3. Submit
4. Verify workflow appears in list

### 4. Test Real-time Updates

1. Start a workflow execution
2. Verify SSE connection establishes
3. Watch real-time updates
4. Check browser console for errors

---

## Monitoring

### Application Monitoring

#### Add Sentry (Error Tracking):

```bash
npm install @sentry/react
```

```tsx
// main.tsx
import * as Sentry from "@sentry/react";

Sentry.init({
  dsn: import.meta.env.VITE_SENTRY_DSN,
  environment: import.meta.env.MODE,
});
```

#### Add Analytics (Google Analytics):

```tsx
// index.tsx
import { useEffect } from 'react';
import { GoogleAnalytics } from './components/GoogleAnalytics';

function App() {
  useEffect(() => {
    if (import.meta.env.VITE_GA_TRACKING_ID) {
      // Initialize GA
    }
  }, []);

  return <YourApp />;
}
```

### Performance Monitoring

#### Vercel Analytics:

```bash
npm install @vercel/analytics
```

```tsx
// App.tsx
import { Analytics } from '@vercel/analytics/react';

export function App() {
  return (
    <>
      <YourApp />
      <Analytics />
    </>
  );
}
```

---

## SSL/TLS Setup

### Let's Encrypt (Free SSL):

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

---

## CI/CD Pipeline

### GitHub Actions:

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Build
        run: npm run build

      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          vercel-args: '--prod'
```

---

## Rollback Procedure

### Vercel:

```bash
# List deployments
vercel ls

# Rollback to specific deployment
vercel rollback <deployment-url>
```

### Manual Rollback:

1. Rebuild previous version:
   ```bash
   git checkout <previous-commit>
   npm run build
   ```

2. Deploy:
   ```bash
   vercel --prod
   ```

---

## Troubleshooting

### Build Failures

**Problem:** Build fails with "Cannot find module"

**Solution:**
```bash
rm -rf node_modules package-lock.json
npm install
```

---

### API Connection Issues

**Problem:** Frontend can't connect to backend

**Solution:**
1. Check API is running: `curl http://localhost:8001/api/health`
2. Check CORS configuration in `api_bridge.py`
3. Verify `VITE_API_BASE_URL` is correct

---

### SSE Streaming Issues

**Problem:** Real-time updates not working

**Solution:**
1. Check browser console for errors
2. Verify no proxy is blocking SSE
3. Test SSE endpoint: `curl http://localhost:8001/stream/workflow/test`

---

### Performance Issues

**Problem:** Slow page loads

**Solution:**
1. Enable compression in Nginx
2. Add CDN caching
3. Optimize images
4. Enable code splitting

---

## Scaling

### Horizontal Scaling

Run multiple API bridge instances behind load balancer:

```bash
# Instance 1
uvicorn api_bridge:app --port 8001

# Instance 2
uvicorn api_bridge:app --port 8002

# Instance 3
uvicorn api_bridge:app --port 8003
```

Configure Nginx upstream:
```nginx
upstream api_backend {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    location /api/ {
        proxy_pass http://api_backend;
    }
}
```

---

## Backup Strategy

### Database Backups:

```bash
# Daily backup
0 2 * * * pg_dump -U user dbname > /backups/db_$(date +\%Y\%m\%d).sql
```

### File Backups:

```bash
# Backup uploaded files
rsync -av /var/www/uploads/ /backups/files/
```

---

## Security Checklist

- [ ] SSL/TLS enabled
- [ ] CORS properly configured
- [ ] Rate limiting enabled
- [ ] Input validation on all forms
- [ ] SQL injection prevention
- [ ] XSS protection
- [ ] CSRF tokens
- [ ] Security headers (CSP, HSTS, etc.)
- [ ] Regular dependency updates
- [ ] Environment variables secured
- [ ] API keys in backend only
- [ ] Database encryption at rest
- [ ] Audit logging enabled

---

## Support

For deployment issues:
1. Check this guide first
2. Review error logs
3. Check GitHub Issues
4. Contact DevOps team

---

**Last Updated:** 2025-01-26
**Version:** 1.0.0
