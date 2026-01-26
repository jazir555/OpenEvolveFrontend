"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.BackendClient = void 0;
exports.createBackendClient = createBackendClient;
const axios_1 = __importDefault(require("axios"));
const socket_io_client_1 = require("socket.io-client");
const errors_1 = require("./errors");
class BackendClient {
    constructor(config) {
        this.socket = null;
        this.globalAbortController = new AbortController();
        this.config = config;
        this.debug = config.debug || false;
        this.httpClient = axios_1.default.create({
            baseURL: config.baseUrl,
            timeout: config.timeout || 30000,
            headers: {
                'Content-Type': 'application/json',
                ...(config.apiKey && { Authorization: `Bearer ${config.apiKey}` }),
                ...config.headers,
            },
        });
        this.httpClient.defaults.signal = this.globalAbortController.signal;
        this.setupInterceptors();
        const sanitizedBaseUrl = config.baseUrl.endsWith('/')
            ? config.baseUrl.slice(0, -1)
            : config.baseUrl;
        this.wsUrl = sanitizedBaseUrl
            .replace('http://', 'ws://')
            .replace('https://', 'wss://');
        this.log('Backend client initialized', { baseUrl: sanitizedBaseUrl });
    }
    setupInterceptors() {
        this.httpClient.interceptors.request.use((config) => {
            this.log('Request:', {
                method: config.method?.toUpperCase(),
                url: config.url,
                data: config.data,
            });
            if (this.config.requestTransform && config.data) {
                config.data = this.config.requestTransform(config.data);
            }
            return config;
        }, (error) => {
            this.log('Request error:', error);
            return Promise.reject(error);
        });
        this.httpClient.interceptors.response.use((response) => {
            this.log('Response:', {
                status: response.status,
                data: response.data,
            });
            if (this.config.responseTransform && response.data) {
                response.data = this.config.responseTransform(response.data);
            }
            return response;
        }, (error) => {
            this.log('Response error:', error);
            return Promise.reject(this.handleAxiosError(error));
        });
    }
    handleAxiosError(error) {
        return (0, errors_1.createIntegrationError)('backend', error);
    }
    async post(endpoint, data, config) {
        try {
            const response = await this.httpClient.post(endpoint, data, config);
            return response.data;
        }
        catch (error) {
            throw this.handleAxiosError(error);
        }
    }
    async get(endpoint, config) {
        try {
            const response = await this.httpClient.get(endpoint, config);
            return response.data;
        }
        catch (error) {
            throw this.handleAxiosError(error);
        }
    }
    async put(endpoint, data, config) {
        try {
            const response = await this.httpClient.put(endpoint, data, config);
            return response.data;
        }
        catch (error) {
            throw this.handleAxiosError(error);
        }
    }
    async delete(endpoint, config) {
        try {
            const response = await this.httpClient.delete(endpoint, config);
            return response.data;
        }
        catch (error) {
            throw this.handleAxiosError(error);
        }
    }
    async patch(endpoint, data, config) {
        try {
            const response = await this.httpClient.patch(endpoint, data, config);
            return response.data;
        }
        catch (error) {
            throw this.handleAxiosError(error);
        }
    }
    websocket(path = '/ws', handlers) {
        const isDefaultPath = path === '/ws';
        try {
            if (isDefaultPath && this.socket?.connected) {
                this.log('WebSocket already connected');
                if (handlers) {
                    this.socket.off('connect');
                    this.socket.off('disconnect');
                    this.socket.off('error');
                    this.socket.off('message');
                    this.socket.off('reconnect');
                    this.attachHandlers(this.socket, handlers);
                }
                return this.socket;
            }
            this.log('Connecting WebSocket:', { url: this.wsUrl + path });
            const socket = (0, socket_io_client_1.io)(this.wsUrl + path, {
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionAttempts: 5,
                reconnectionDelay: 1000,
                reconnectionDelayMax: 5000,
                timeout: 10000,
                auth: this.config.apiKey ? { apiKey: this.config.apiKey } : undefined,
            });
            if (isDefaultPath) {
                this.socket = socket;
            }
            if (handlers) {
                this.attachHandlers(socket, handlers);
            }
            return socket;
        }
        catch (error) {
            this.log('Failed to initialize WebSocket:', error);
            throw (0, errors_1.createIntegrationError)('backend', {
                message: 'Failed to initialize WebSocket connection',
                originalError: error
            });
        }
    }
    attachHandlers(socket, handlers) {
        socket.on('connect', () => {
            this.log('WebSocket connected');
            handlers.onConnect?.();
        });
        socket.on('disconnect', (reason) => {
            this.log('WebSocket disconnected:', { reason });
            handlers.onDisconnect?.(reason);
        });
        socket.on('error', (error) => {
            this.log('WebSocket error:', error);
            handlers.onError?.(error);
        });
        socket.on('message', (message) => {
            this.log('WebSocket message:', message);
            handlers.onMessage?.(message);
        });
        socket.on('reconnect', (attemptNumber) => {
            this.log('WebSocket reconnected:', { attemptNumber });
            handlers.onReconnect?.(attemptNumber);
        });
    }
    disconnectWebSocket() {
        if (this.socket) {
            this.log('Disconnecting WebSocket');
            this.socket.disconnect();
            this.socket = null;
        }
    }
    isWebSocketConnected() {
        return this.socket?.connected || false;
    }
    async ping() {
        try {
            const start = Date.now();
            await this.get('/ping');
            const duration = Date.now() - start;
            this.log('Ping successful:', { duration: `${duration}ms` });
            return true;
        }
        catch (error) {
            this.log('Ping failed:', error);
            return false;
        }
    }
    async getStatus() {
        try {
            const status = await this.get('/status');
            this.log('Backend status:', status);
            return status;
        }
        catch (error) {
            throw this.handleAxiosError(error);
        }
    }
    async getHealth() {
        try {
            const health = await this.get('/health');
            this.log('Backend health:', health);
            return health;
        }
        catch (error) {
            throw this.handleAxiosError(error);
        }
    }
    async getVersion() {
        try {
            const { version } = await this.get('/version');
            this.log('Backend version:', version);
            return version;
        }
        catch (error) {
            throw this.handleAxiosError(error);
        }
    }
    cancelAllRequests() {
        this.globalAbortController.abort();
        this.globalAbortController = new AbortController();
        this.httpClient.defaults.signal = this.globalAbortController.signal;
        this.log('All requests cancelled');
    }
    updateConfig(config) {
        this.config = { ...this.config, ...config };
        if (config.timeout) {
            this.httpClient.defaults.timeout = config.timeout;
        }
        if (config.headers) {
            this.httpClient.defaults.headers = {
                ...this.httpClient.defaults.headers,
                ...config.headers,
            };
        }
        if (config.apiKey) {
            this.httpClient.defaults.headers['Authorization'] = `Bearer ${config.apiKey}`;
        }
        this.log('Configuration updated:', config);
    }
    log(message, data) {
        if (this.debug) {
            if (data) {
                console.log(`[BackendClient] ${message}`, data);
            }
            else {
                console.log(`[BackendClient] ${message}`);
            }
        }
    }
    getHttpClient() {
        return this.httpClient;
    }
    getSocket() {
        return this.socket;
    }
}
exports.BackendClient = BackendClient;
function createBackendClient(baseUrl) {
    return new BackendClient({
        baseUrl,
        timeout: 30000,
        debug: false,
    });
}
//# sourceMappingURL=backend.js.map