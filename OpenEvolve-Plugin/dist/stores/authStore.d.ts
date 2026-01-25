/**
 * User interface
 */
export interface User {
    user_id: string;
    email: string;
    username: string;
    full_name?: string;
    role: string;
    created_at: string;
    preferences?: {
        theme: 'light' | 'dark';
        language: string;
    };
}
/**
 * Authentication state interface
 */
interface AuthState {
    user: User | null;
    token: string | null;
    refreshToken: string | null;
    isAuthenticated: boolean;
    isLoading: boolean;
    error: string | null;
    login: (email: string, password: string) => Promise<void>;
    logout: () => Promise<void>;
    register: (email: string, password: string, username: string, full_name?: string) => Promise<void>;
    refreshAuthToken: () => Promise<void>;
    updateUser: (updates: Partial<User>) => void;
    clearError: () => void;
    setLoading: (loading: boolean) => void;
}
/**
 * Authentication store with persistence
 */
export declare const useAuthStore: import('zustand').UseBoundStore<Omit<import('zustand').StoreApi<AuthState>, "persist"> & {
    persist: {
        setOptions: (options: Partial<import('zustand/middleware').PersistOptions<AuthState, {
            user: User;
            token: string;
            refreshToken: string;
            isAuthenticated: boolean;
        }>>) => void;
        clearStorage: () => void;
        rehydrate: () => Promise<void> | void;
        hasHydrated: () => boolean;
        onHydrate: (fn: (state: AuthState) => void) => () => void;
        onFinishHydration: (fn: (state: AuthState) => void) => () => void;
        getOptions: () => Partial<import('zustand/middleware').PersistOptions<AuthState, {
            user: User;
            token: string;
            refreshToken: string;
            isAuthenticated: boolean;
        }>>;
    };
}>;
export {};
