import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { errorLogger } from '@/utils';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';

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

  // Actions
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
export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
          set({ isLoading: true, error: null });

          const response = await fetch('/api/v1/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.message || `Login failed with status ${response.status}`;
            throw new Error(errorMessage);
          }

          const data = await response.json();
          set({
            user: data.user || null, // API might not return user in login response
            token: data.access_token,
            refreshToken: data.refresh_token,
            isAuthenticated: true,
            isLoading: false,
          });

          // Fetch user profile if not returned in login response
          if (!data.user) {
            try {
              const profileResponse = await fetch('/api/v1/users/me', {
                headers: {
                  'Authorization': `Bearer ${data.access_token}`,
                },
              });

              if (profileResponse.ok) {
                const userProfile = await profileResponse.json();
                set({ user: userProfile });
              }
            } catch (profileError) {
              errorLogger.logError(
                profileError instanceof Error ? profileError : new Error(String(profileError)),
                'error',
                { component: 'AuthStore', function: 'login', additionalData: { email } }
              );
              // Continue with login even if profile fetch fails
            }
          }

          return data;
        }, {
          strategy: 'retry',
          maxRetries: 3,
          retryDelay: 1000,
          showUserNotification: true,
          logError: true,
          context: {
            component: 'AuthStore',
            function: 'login',
            operation: 'USER_LOGIN',
            additionalData: { email }
          }
        });

        if (!result.success) {
          const errorMessage = result.error?.message || 'Login failed';
          set({
            error: errorMessage,
            isLoading: false,
            isAuthenticated: false,
          });
          throw result.error || new Error('Login failed');
        }
      },

      logout: async () => {
        await gracefulErrorHandler.executeWithErrorHandling(async () => {
          set({ isLoading: true });
          const token = get().token;
          if (token) {
            try {
              await fetch('/api/v1/auth/logout', {
                method: 'POST',
                headers: {
                  'Authorization': `Bearer ${token}`,
                },
              });
            } catch (logoutError) {
              errorLogger.logError(
                logoutError instanceof Error ? logoutError : new Error(String(logoutError)),
                'error',
                { component: 'AuthStore', function: 'logout' }
              );
              // Continue with logout even if API call fails
            }
          }
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          context: {
            component: 'AuthStore',
            function: 'logout',
            operation: 'USER_LOGOUT',
          }
        }).catch(error => {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AuthStore', function: 'logout' }
          );
        });

        // Always clear the state regardless of API call success/failure
        set({
          user: null,
          token: null,
          refreshToken: null,
          isAuthenticated: false,
          isLoading: false,
        });
      },

      register: async (email: string, password: string, username: string, full_name?: string) => {
        const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
          set({ isLoading: true, error: null });

          const response = await fetch('/api/v1/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password, username, full_name }),
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.message || `Registration failed with status ${response.status}`;
            throw new Error(errorMessage);
          }

          set({ isLoading: false });
          return true;
        }, {
          strategy: 'retry',
          maxRetries: 3,
          retryDelay: 1000,
          showUserNotification: true,
          logError: true,
          context: {
            component: 'AuthStore',
            function: 'register',
            operation: 'USER_REGISTRATION',
            additionalData: { email, username }
          }
        });

        if (!result.success) {
          const errorMessage = result.error?.message || 'Registration failed';
          set({
            error: errorMessage,
            isLoading: false,
          });
          throw result.error || new Error('Registration failed');
        }
      },

      refreshAuthToken: async () => {
        const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
          const refreshTokenValue = get().refreshToken;
          if (!refreshTokenValue) {
            throw new Error('No refresh token available');
          }

          const response = await fetch('/api/v1/auth/refresh', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: refreshTokenValue }),
          });

          if (!response.ok) {
            throw new Error(`Token refresh failed with status ${response.status}`);
          }

          const data = await response.json();
          set({
            token: data.access_token,
          });

          return data;
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 1000,
          showUserNotification: false, // Don't show notification for automatic token refresh
          logError: true,
          context: {
            component: 'AuthStore',
            function: 'refreshAuthToken',
            operation: 'TOKEN_REFRESH',
          }
        });

        if (!result.success) {
          errorLogger.logError(
            result.error instanceof Error ? result.error : new Error(String(result.error)),
            'error',
            { component: 'AuthStore', function: 'refreshAuthToken' }
          );
          // If refresh fails, logout user
          await get().logout();
          throw result.error || new Error('Token refresh failed');
        }
      },

      updateUser: (updates: Partial<User>) => {
        gracefulErrorHandler.executeWithErrorHandling(() => {
          const currentUser = get().user;
          if (currentUser) {
            set({ user: { ...currentUser, ...updates } });
          }
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          context: {
            component: 'AuthStore',
            function: 'updateUser',
            operation: 'UPDATE_USER_PROFILE',
            additionalData: { updates }
          }
        }).catch(error => {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AuthStore', function: 'updateUser', additionalData: { updates } }
          );
        });
      },

      clearError: () => {
        gracefulErrorHandler.executeWithErrorHandling(() => {
          set({ error: null });
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          context: {
            component: 'AuthStore',
            function: 'clearError',
            operation: 'CLEAR_ERROR',
          }
        }).catch(error => {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AuthStore', function: 'clearError' }
          );
        });
      },

      setLoading: (loading: boolean) => {
        gracefulErrorHandler.executeWithErrorHandling(() => {
          set({ isLoading: loading });
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          context: {
            component: 'AuthStore',
            function: 'setLoading',
            operation: 'SET_LOADING_STATE',
            additionalData: { loading }
          }
        }).catch(error => {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AuthStore', function: 'setLoading', additionalData: { loading } }
          );
        });
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        refreshToken: state.refreshToken,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
