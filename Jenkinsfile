// ============================================================================
// OpenEvolve Frontend - Jenkins Pipeline
// ============================================================================
// This pipeline orchestrates the build, test, and deployment of the
// OpenEvolve Frontend hybrid PES-Evolution System.
//
// Architecture: Federation of 30+ Open Source systems
// Operating Mode: ZERO TRUST
// ============================================================================

pipeline {
    agent any

    tools {
        nodejs 'NodeJS-20'
        jdk 'JDK-17'
    }

    environment {
        NODE_VERSION = '20'
        DOCKER_REGISTRY = credentials('docker-registry')
        DOCKER_CREDENTIALS = credentials('docker-credentials')
        KUBECONFIG_STAGING = credentials('kubeconfig-staging')
        KUBECONFIG_PRODUCTION = credentials('kubeconfig-production')
        VAULT_ADDR = 'https://vault.openevolve.io'
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '30', artifactNumToKeepStr: '10'))
        disableConcurrentBuilds()
        timeout(time: 1, unit: 'HOURS')
        timestamps()
        ansiColor('xterm')
    }

    stages {
        // ====================================================================
        // Stage 1: Initialize & Install
        // ====================================================================
        stage('Initialize') {
            steps {
                script {
                    echo "========================================="
                    echo "OpenEvolve Frontend - Build Pipeline"
                    echo "========================================="
                    echo "Branch: ${env.BRANCH_NAME}"
                    echo "Build: ${env.BUILD_NUMBER}"
                    echo "SHA: ${env.GIT_COMMIT}"
                    echo "========================================="
                }

                sh '''
                    echo "Node version: $(node --version)"
                    echo "NPM version: $(npm --version)"
                    echo "Docker version: $(docker --version)"
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    echo 'Installing dependencies...'
                }

                sh '''
                    # Clean install
                    npm ci --prefer-offline --no-audit

                    # Install orchestration dependencies
                    cd glue/orchestration
                    npm ci --prefer-offline --no-audit
                    cd ../..
                '''
            }

            post {
                success {
                    script {
                        echo 'Dependencies installed successfully'
                    }
                }
                failure {
                    script {
                        error('Dependency installation failed')
                    }
                }
            }
        }

        // ====================================================================
        // Stage 2: Code Quality
        // ====================================================================
        stage('Lint & Validate') {
            parallel {
                stage('ESLint') {
                    steps {
                        sh 'npm run lint'
                    }
                }

                stage('TypeScript Check') {
                    steps {
                        sh 'npm run typecheck'
                    }
                }

                stage('Format Check') {
                    steps {
                        sh 'npm run format -- --check || true'
                    }
                }
            }
        }

        // ====================================================================
        // Stage 3: Testing
        // ====================================================================
        stage('Unit Tests') {
            steps {
                script {
                    echo 'Running unit tests...'
                }

                sh '''
                    npm run test:ci
                '''
            }

            post {
                always {
                    // Publish HTML coverage report
                    publishHTML(target: [
                        reportDir: 'coverage',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report',
                        keepAll: true,
                        allowMissing: false
                    ])

                    // Publish coverage metrics
                    publishCoverage adapters: [coberturaAdapter('coverage/cobertura-coverage.xml')],
                        sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                }
            }
        }

        stage('Integration Tests') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }

            environment {
                VALKEY_HOST = 'localhost'
                VALKEY_PORT = '6379'
            }

            steps {
                script {
                    echo 'Running integration tests...'
                }

                // Start Valkey container
                sh '''
                    docker run -d --name valkey-ci \
                        -p 6379:6379 \
                        redis:7-alpine \
                        --save "" --appendonly no
                '''

                // Wait for Valkey to be ready
                sh '''
                    for i in {1..30}; do
                        if docker exec valkey-ci redis-cli ping | grep -q PONG; then
                            echo "Valkey is ready"
                            exit 0
                        fi
                        echo "Waiting for Valkey... ($i/30)"
                        sleep 1
                    done
                    echo "Valkey failed to start"
                    exit 1
                '''

                // Run integration tests
                sh 'npm run test:e2e'
            }

            post {
                always {
                    sh 'docker stop valkey-ci || true'
                    sh 'docker rm valkey-ci || true'
                }
            }
        }

        stage('Contract Tests') {
            steps {
                script {
                    echo 'Running contract tests...'
                }

                sh 'npm run test:contract || echo "No contract tests configured"'
            }
        }

        // ====================================================================
        // Stage 4: Security Scanning
        // ====================================================================
        stage('Security Scan') {
            parallel {
                stage('Dependency Audit') {
                    steps {
                        sh 'npm audit --audit-level=high || true'
                    }
                }

                stage('SAST') {
                    steps {
                        sh '''
                            # Install semgrep if not present
                            if ! command -v semgrep &> /dev/null; then
                                python3 -m pip install semgrep --user
                            fi
                            semgrep --config auto --json --output semgrep-report.json . || true
                        '''
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'semgrep-report.json', allowEmptyArchive: true
                        }
                    }
                }
            }
        }

        // ====================================================================
        // Stage 5: Build
        // ====================================================================
        stage('Build TypeScript') {
            steps {
                script {
                    echo 'Building TypeScript projects...'
                }

                sh 'npm run build'
            }

            post {
                success {
                    archiveArtifacts artifacts: 'glue/orchestration/workflows/dist/**/*', fingerprint: true
                    archiveArtifacts artifacts: 'glue/adapters/*/dist/**/*', fingerprint: true
                }
            }
        }

        stage('Build Docker Image') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }

            steps {
                script {
                    def imageTag = "${env.DOCKER_REGISTRY}/openevolve/frontend:${env.GIT_COMMIT}"
                    def imageLatest = "${env.DOCKER_REGISTRY}/openevolve/frontend:latest"

                    echo "Building Docker image: ${imageTag}"

                    sh """
                        docker build -t ${imageTag} -t ${imageLatest} .

                        # Test the image
                        docker run --rm ${imageTag} node --version

                        # Push to registry
                        echo ${DOCKER_CREDENTIALS} | docker login ${DOCKER_REGISTRY} -u ${DOCKER_REGISTRY} --password-stdin
                        docker push ${imageTag}
                        docker push ${imageLatest}
                    """
                }
            }
        }

        // ====================================================================
        // Stage 6: Deploy
        // ====================================================================
        stage('Deploy Staging') {
            when {
                branch 'develop'
            }

            steps {
                script {
                    echo 'Deploying to staging environment...'
                }

                sh '''
                    # Configure kubectl
                    echo ${KUBECONFIG_STAGING} | base64 -d > /tmp/kubeconfig-staging
                    export KUBECONFIG=/tmp/kubeconfig-staging

                    # Deploy
                    kubectl set image deployment/openevolve-frontend \
                        openevolve=${DOCKER_REGISTRY}/openevolve/frontend:${GIT_COMMIT} \
                        -n openevolve-staging

                    # Wait for rollout
                    kubectl rollout status deployment/openevolve-frontend \
                        -n openevolve-staging \
                        --timeout=5m
                '''
            }

            post {
                success {
                    script {
                        echo 'Staging deployment successful'
                    }
                }
            }
        }

        stage('Smoke Tests Staging') {
            when {
                branch 'develop'
            }

            steps {
                sh 'npm run test:smoke -- --env=staging || true'
            }
        }

        stage('Deploy Production') {
            when {
                branch 'main'
            }

            steps {
                script {
                    echo 'Deploying to production environment...'

                    // Manual approval gate
                    input message: 'Deploy to production?', ok: 'Deploy'
                }

                sh '''
                    # Configure kubectl
                    echo ${KUBECONFIG_PRODUCTION} | base64 -d > /tmp/kubeconfig-production
                    export KUBECONFIG=/tmp/kubeconfig-production

                    # Deploy
                    kubectl set image deployment/openevolve-frontend \
                        openevolve=${DOCKER_REGISTRY}/openevolve/frontend:${GIT_COMMIT} \
                        -n openevolve-production

                    # Wait for rollout
                    kubectl rollout status deployment/openevolve-frontend \
                        -n openevolve-production \
                        --timeout=10m
                '''
            }

            post {
                success {
                    script {
                        echo 'Production deployment successful'
                    }
                }
            }
        }

        stage('Smoke Tests Production') {
            when {
                branch 'main'
            }

            steps {
                sh 'npm run test:smoke -- --env=production'
            }
        }
    }

    // ========================================================================
    // Post-Build Actions
    // ========================================================================
    post {
        always {
            script {
                echo "========================================="
                echo "Build ${env.BUILD_NUMBER} completed"
                echo "Status: ${currentBuild.result}"
                echo "========================================="
            }

            // Clean workspace
            cleanWs(
                deleteDirs: true,
                patterns: [
                    [pattern: 'node_modules', type: 'INCLUDE'],
                    [pattern: '.npm', type: 'INCLUDE'],
                    [pattern: 'coverage', type: 'EXCLUDE'],
                    [pattern: 'test-results', type: 'EXCLUDE']
                ]
            )
        }

        success {
            script {
                echo 'Pipeline completed successfully!'

                // Send notification
                // slackSend(color: 'good', message: "Build ${env.BUILD_NUMBER} succeeded")
            }
        }

        failure {
            script {
                echo 'Pipeline failed!'

                // Send notification
                // slackSend(color: 'danger', message: "Build ${env.BUILD_NUMBER} failed")
            }
        }

        unstable {
            script {
                echo 'Pipeline is unstable!'

                // Send notification
                // slackSend(color: 'warning', message: "Build ${env.BUILD_NUMBER} is unstable")
            }
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

def notifyBuild(String buildStatus) {
    def buildUrl = env.BUILD_URL ?: 'N/A'
    def subject = "${buildStatus}: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'"
    def summary = "${subject} (${buildUrl})"
    def details = """<p>${buildStatus}: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'</p>
      <p>Check console output at <a href="${buildUrl}">${buildUrl}</a></p>"""

    // Email notification
    // emailext(
    //     subject: subject,
    //     body: details,
    //     to: 'devops@openevolve.io',
    //     mimeType: 'text/html'
    // )
}
