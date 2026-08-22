/**
 * OpenEvolve Security / API Keys
 *
 * Manages the engine's API keys, roles and audit trail entirely from BubbleLab.
 * Talks to the `OPENEVOLVE_API_BASE_URL` proxy (`:8000`) which forwards the
 * `/api/security/*` routes to the security service:
 *  - `openevolveApi.listApiKeys()`   -> provisioned keys (never the raw secret)
 *  - `openevolveApi.createApiKey()`  -> provisions a key, RAW secret returned ONCE
 *  - `openevolveApi.revokeApiKey()`  -> revokes a key by `key_id`
 *  - `openevolveApi.listRoles()`     -> read-only role/permission matrix
 *  - `openevolveApi.createRole()`    -> optional role creation
 *  - `openevolveApi.getAuditLogs()`  -> read-only audit trail
 *
 * Creating keys requires the ACTIVE key to hold the `admin` role. The active key
 * is whatever `getOpenEvolveApiKey()` (`src/lib/api.ts`) resolves per request:
 * `VITE_OPENEVOLVE_API_KEY` first, else `localStorage['openevolve_api_key']`.
 * "Use this key" therefore only writes that localStorage entry — the shared
 * `ApiClient` picks it up on the next request with no client re-wiring.
 *
 * Error handling: the shared `ApiClient` already toasts HTTP/network failures, so
 * this page surfaces contextual inline banners instead of duplicating those
 * toasts, and toasts successes (plus the 404 case the client stays silent about).
 */

import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { createFileRoute } from '@tanstack/react-router';
import { Tab } from '@headlessui/react';
import { toast } from 'react-toastify';
import {
  AlertTriangle,
  KeyRound,
  RefreshCw,
  ScrollText,
  ShieldCheck,
} from 'lucide-react';
import { ApiHttpError } from '@/lib/api';
import { openevolveApi } from '@/services/openevolveApi';
import { CopyButton } from '@/components/common/CopyButton';
import type {
  ApiKeyCreateResponse,
  ApiKeyListItem,
  AuditLogEntry,
  SecurityRole,
} from '@/types/openevolve';

export const Route = createFileRoute('/openevolve/security')({
  component: OpenEvolveSecurityPage,
});

/**
 * Same key as `OpenEvolveControlPanel` and `getOpenEvolveApiKey()` in
 * `src/lib/api.ts` (neither exports it, so it is mirrored here on purpose).
 */
const OPENEVOLVE_API_KEY_STORAGE = 'openevolve_api_key';

const ASSIGNABLE_ROLES = ['admin', 'user', 'readonly'] as const;
const DEFAULT_ROLES: string[] = ['user'];
const AUDIT_LIMIT_OPTIONS = [50, 100, 200, 500];

// ============================ Styling helpers ============================

const inputClass =
  'w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100 placeholder:text-gray-600 focus:outline-none focus:ring-1 focus:ring-blue-500';
const labelClass = 'mb-1 block text-xs font-medium text-gray-400';
const sectionClass = 'space-y-4 rounded-xl border border-[#2a2a2a] bg-[#111111] p-4';
const thClass =
  'px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wider text-gray-500';
const tdClass = 'px-3 py-2 align-top text-sm text-gray-300';

// ============================== Utilities ===============================

const describeError = (err: unknown): string => {
  if (err instanceof ApiHttpError) {
    const data = err.data;
    if (typeof data === 'string' && data.trim().length > 0) return data;
    if (data && typeof data === 'object') {
      const detail = (data as { detail?: unknown }).detail;
      if (typeof detail === 'string' && detail.trim().length > 0) return detail;
      const message = (data as { message?: unknown }).message;
      if (typeof message === 'string' && message.trim().length > 0) return message;
    }
    return `Request failed with HTTP ${err.status}`;
  }
  if (err instanceof Error) return err.message;
  return 'Unknown error';
};

/** A 503 from `/api/security/*` means the engine has no RBAC subsystem loaded. */
const isRbacUnavailableError = (err: unknown): boolean =>
  err instanceof ApiHttpError && err.status === 503;

/**
 * `ApiClient` toasts every failure except 404s, so only fill that gap here to
 * avoid stacking two toasts on the same error.
 */
const notifySilentFailure = (context: string, err: unknown): void => {
  if (err instanceof ApiHttpError && err.status === 404) {
    toast.error(`${context}: ${describeError(err)}`);
  }
};

const readActiveKey = (): string => {
  try {
    return globalThis.localStorage?.getItem(OPENEVOLVE_API_KEY_STORAGE) ?? '';
  } catch {
    // ignore localStorage access errors (private mode / SSR)
    return '';
  }
};

const maskKey = (key: string): string =>
  key.length <= 8 ? '••••' : `${key.slice(0, 4)}…${key.slice(-4)}`;

const formatTimestamp = (value: unknown): string => {
  if (value === undefined || value === null || value === '') return '—';
  if (typeof value === 'number') {
    // Engine timestamps may be epoch seconds or milliseconds.
    const ms = value < 1e12 ? value * 1000 : value;
    const asDate = new Date(ms);
    return Number.isNaN(asDate.getTime()) ? String(value) : asDate.toLocaleString();
  }
  if (typeof value === 'string') {
    const asDate = new Date(value);
    return Number.isNaN(asDate.getTime()) ? value : asDate.toLocaleString();
  }
  return String(value);
};

const readCell = (entry: AuditLogEntry, key: string): string => {
  const value = entry[key];
  if (value === undefined || value === null || value === '') return '—';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

// ============================ Small components ==========================

function Badge({
  children,
  tone = 'gray',
}: {
  children: React.ReactNode;
  tone?: 'gray' | 'green' | 'red' | 'blue' | 'yellow';
}) {
  const tones: Record<string, string> = {
    gray: 'bg-[#1b1b1b] text-gray-300 border-[#303030]',
    green: 'bg-emerald-950/40 text-emerald-300 border-emerald-900/60',
    red: 'bg-red-950/40 text-red-300 border-red-900/60',
    blue: 'bg-blue-950/40 text-blue-300 border-blue-900/60',
    yellow: 'bg-yellow-950/40 text-yellow-300 border-yellow-900/60',
  };
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

function InlineError({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-red-900/60 bg-red-950/30 px-3 py-2 text-sm text-red-300">
      {children}
    </div>
  );
}

function RbacUnavailableBanner() {
  return (
    <div className="flex items-start gap-2 rounded-md border border-yellow-900/60 bg-yellow-950/30 px-3 py-2 text-sm text-yellow-300">
      <AlertTriangle className="mt-0.5 h-4 w-4 flex-none" />
      <div>
        <p className="font-medium">RBAC subsystem unavailable on the engine</p>
        <p className="mt-0.5 text-xs text-yellow-400/80">
          Key, role and audit data cannot be managed until the engine loads its RBAC
          subsystem. Existing keys keep working, but this page is read-only for now.
        </p>
      </div>
    </div>
  );
}

function TabButton({ label, icon: Icon }: { label: string; icon: typeof KeyRound }) {
  return (
    <Tab className="flex items-center gap-2 rounded-md px-3 py-2 text-xs font-medium text-gray-400 hover:text-gray-200 ui-selected:bg-[#1b1b1b] ui-selected:text-white">
      <Icon className="h-4 w-4" />
      {label}
    </Tab>
  );
}

function RefreshButton({
  onClick,
  busy,
  label = 'Refresh',
}: {
  onClick: () => void;
  busy: boolean;
  label?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      className="inline-flex items-center gap-2 rounded-md border border-[#303030] bg-[#1b1b1b] px-3 py-2 text-xs font-medium text-gray-300 hover:bg-[#232323] disabled:opacity-50"
    >
      <RefreshCw className={`h-3.5 w-3.5 ${busy ? 'animate-spin' : ''}`} />
      {busy ? 'Loading…' : label}
    </button>
  );
}

// ================================ Page ==================================

function OpenEvolveSecurityPage() {
  // --- API keys -----------------------------------------------------------
  const [keys, setKeys] = useState<ApiKeyListItem[]>([]);
  const [keysLoading, setKeysLoading] = useState(true);
  const [keysError, setKeysError] = useState<string | null>(null);
  const [rbacAvailable, setRbacAvailable] = useState(true);

  // --- Create-key form ----------------------------------------------------
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [selectedRoles, setSelectedRoles] = useState<string[]>(DEFAULT_ROLES);
  const [creating, setCreating] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [createdKey, setCreatedKey] = useState<ApiKeyCreateResponse | null>(null);

  // --- Revoke -------------------------------------------------------------
  const [pendingRevokeId, setPendingRevokeId] = useState<string | null>(null);
  const [revokingId, setRevokingId] = useState<string | null>(null);

  // --- Active key ---------------------------------------------------------
  const [activeKey, setActiveKey] = useState<string>(() => readActiveKey());

  // --- Roles --------------------------------------------------------------
  const [roles, setRoles] = useState<SecurityRole[]>([]);
  const [rolesLoading, setRolesLoading] = useState(true);
  const [rolesError, setRolesError] = useState<string | null>(null);
  const [roleName, setRoleName] = useState('');
  const [roleDescription, setRoleDescription] = useState('');
  const [rolePermissions, setRolePermissions] = useState('');
  const [creatingRole, setCreatingRole] = useState(false);
  const [roleFormError, setRoleFormError] = useState<string | null>(null);

  // --- Audit logs ---------------------------------------------------------
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [auditSource, setAuditSource] = useState<string | null>(null);
  const [auditLimit, setAuditLimit] = useState(200);
  const [auditLoading, setAuditLoading] = useState(true);
  const [auditError, setAuditError] = useState<string | null>(null);

  const loadKeys = useCallback(async () => {
    setKeysLoading(true);
    setKeysError(null);
    try {
      const response = await openevolveApi.listApiKeys();
      setKeys(response.api_keys ?? []);
      setRbacAvailable(response.rbac_available !== false);
    } catch (err) {
      if (isRbacUnavailableError(err)) {
        setKeys([]);
        setRbacAvailable(false);
      } else {
        setKeysError(describeError(err));
        notifySilentFailure('Failed to load API keys', err);
      }
    } finally {
      setKeysLoading(false);
    }
  }, []);

  const loadRoles = useCallback(async () => {
    setRolesLoading(true);
    setRolesError(null);
    try {
      const response = await openevolveApi.listRoles();
      setRoles(response.roles ?? []);
      if (response.rbac_available === false) setRbacAvailable(false);
    } catch (err) {
      if (isRbacUnavailableError(err)) {
        setRoles([]);
        setRbacAvailable(false);
      } else {
        setRolesError(describeError(err));
        notifySilentFailure('Failed to load roles', err);
      }
    } finally {
      setRolesLoading(false);
    }
  }, []);

  const loadAuditLogs = useCallback(async () => {
    setAuditLoading(true);
    setAuditError(null);
    try {
      const response = await openevolveApi.getAuditLogs(auditLimit);
      setAuditLogs(response.audit_logs ?? []);
      setAuditSource(response.source ?? null);
    } catch (err) {
      if (isRbacUnavailableError(err)) {
        setAuditLogs([]);
        setRbacAvailable(false);
      } else {
        setAuditError(describeError(err));
        notifySilentFailure('Failed to load audit logs', err);
      }
    } finally {
      setAuditLoading(false);
    }
  }, [auditLimit]);

  useEffect(() => {
    void loadKeys();
    void loadRoles();
  }, [loadKeys, loadRoles]);

  useEffect(() => {
    void loadAuditLogs();
  }, [loadAuditLogs]);

  const toggleRole = (role: string) =>
    setSelectedRoles((current) =>
      current.includes(role) ? current.filter((r) => r !== role) : [...current, role]
    );

  const activateKey = (key: string, label?: string) => {
    try {
      globalThis.localStorage?.setItem(OPENEVOLVE_API_KEY_STORAGE, key);
      setActiveKey(key);
      toast.success(
        `Active OpenEvolve key set${label ? ` to ${label}` : ''} — used for subsequent requests.`
      );
    } catch {
      toast.error('Could not store the key in this browser (localStorage blocked).');
    }
  };

  const handleCreateKey = async (event: FormEvent) => {
    event.preventDefault();
    const trimmedUsername = username.trim();
    if (trimmedUsername.length === 0) {
      setFormError('Username is required.');
      return;
    }
    if (trimmedUsername.length > 256) {
      setFormError('Username must be 256 characters or fewer.');
      return;
    }
    if (selectedRoles.length === 0) {
      setFormError('Select at least one role.');
      return;
    }

    setFormError(null);
    setCreating(true);
    try {
      const trimmedEmail = email.trim();
      const created = await openevolveApi.createApiKey({
        username: trimmedUsername,
        email: trimmedEmail.length > 0 ? trimmedEmail : null,
        roles: selectedRoles,
      });
      setCreatedKey(created);
      setUsername('');
      setEmail('');
      setSelectedRoles(DEFAULT_ROLES);
      toast.success(`API key created for ${created.username ?? trimmedUsername}.`);
      await loadKeys();
    } catch (err) {
      setFormError(describeError(err));
      notifySilentFailure('Failed to create API key', err);
    } finally {
      setCreating(false);
    }
  };

  const handleRevokeKey = async (keyId: string) => {
    setRevokingId(keyId);
    try {
      const response = await openevolveApi.revokeApiKey(keyId);
      toast.success(`Key ${response.key_id || keyId} ${response.status || 'revoked'}.`);
      setPendingRevokeId(null);
      await loadKeys();
    } catch (err) {
      setKeysError(describeError(err));
      notifySilentFailure('Failed to revoke API key', err);
    } finally {
      setRevokingId(null);
    }
  };

  const handleCreateRole = async (event: FormEvent) => {
    event.preventDefault();
    const trimmedName = roleName.trim();
    const permissions = rolePermissions
      .split(',')
      .map((permission) => permission.trim())
      .filter((permission) => permission.length > 0);

    if (trimmedName.length === 0) {
      setRoleFormError('Role name is required.');
      return;
    }
    if (permissions.length === 0) {
      setRoleFormError('Provide at least one permission.');
      return;
    }

    setRoleFormError(null);
    setCreatingRole(true);
    try {
      const trimmedDescription = roleDescription.trim();
      const created = await openevolveApi.createRole({
        name: trimmedName,
        ...(trimmedDescription.length > 0 ? { description: trimmedDescription } : {}),
        permissions,
      });
      toast.success(`Role "${created.name ?? trimmedName}" created.`);
      setRoleName('');
      setRoleDescription('');
      setRolePermissions('');
      await loadRoles();
    } catch (err) {
      setRoleFormError(describeError(err));
      notifySilentFailure('Failed to create role', err);
    } finally {
      setCreatingRole(false);
    }
  };

  const activeKeys = keys.filter((key) => key.revoked !== true).length;

  return (
    <div className="space-y-6 p-6">
      {/* ============================= Header ============================= */}
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold text-white">
            <KeyRound className="h-5 w-5 text-blue-400" />
            API Keys
          </h1>
          <p className="mt-1 text-sm text-gray-400">
            Provision, inspect and revoke OpenEvolve engine keys, review RBAC roles and
            the security audit trail.
          </p>
          <p className="mt-1 text-xs text-gray-500">
            Active key:{' '}
            {activeKey ? (
              <span className="font-mono text-gray-300">{maskKey(activeKey)}</span>
            ) : (
              <span className="text-gray-500">
                none stored in this browser (falls back to VITE_OPENEVOLVE_API_KEY)
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone={rbacAvailable ? 'green' : 'yellow'}>
            RBAC {rbacAvailable ? 'available' : 'unavailable'}
          </Badge>
          <Badge tone="blue">
            {activeKeys} active / {keys.length} total
          </Badge>
        </div>
      </header>

      {!rbacAvailable && <RbacUnavailableBanner />}

      {/* ====================== One-time secret alert ====================== */}
      {createdKey && (
        <div className="space-y-3 rounded-xl border border-emerald-900/60 bg-emerald-950/25 p-4">
          <div className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 h-5 w-5 flex-none text-emerald-300" />
            <div>
              <p className="text-sm font-semibold text-emerald-200">
                Copy this key now — it is shown only once
              </p>
              <p className="mt-0.5 text-xs text-emerald-300/80">
                {createdKey.warning ??
                  'The engine never returns the raw secret again. Store it somewhere safe before leaving this page.'}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2">
            <code className="flex-1 break-all font-mono text-sm text-emerald-200">
              {createdKey.api_key}
            </code>
            <CopyButton
              text={createdKey.api_key}
              className="rounded-md border border-[#303030] bg-[#1b1b1b]"
            />
          </div>

          <dl className="grid grid-cols-1 gap-2 text-xs text-gray-400 sm:grid-cols-3">
            <div>
              <dt className="text-gray-500">Key ID</dt>
              <dd className="font-mono text-gray-300">{createdKey.key_id}</dd>
            </div>
            <div>
              <dt className="text-gray-500">Username</dt>
              <dd className="text-gray-300">{createdKey.username ?? '—'}</dd>
            </div>
            <div>
              <dt className="text-gray-500">Roles</dt>
              <dd className="text-gray-300">
                {createdKey.roles?.length ? createdKey.roles.join(', ') : '—'}
              </dd>
            </div>
          </dl>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() =>
                activateKey(createdKey.api_key, createdKey.username ?? createdKey.key_id)
              }
              className="rounded-md bg-emerald-600 px-3 py-2 text-xs font-medium text-white hover:bg-emerald-500"
            >
              Use this key
            </button>
            <button
              type="button"
              onClick={() => setCreatedKey(null)}
              className="rounded-md border border-[#303030] bg-[#1b1b1b] px-3 py-2 text-xs font-medium text-gray-300 hover:bg-[#232323]"
            >
              I saved it — dismiss
            </button>
          </div>
        </div>
      )}

      {/* ============================== Tabs ============================== */}
      <Tab.Group>
        <Tab.List className="flex flex-wrap gap-1 rounded-lg bg-[#0d0d0d] p-2">
          <TabButton label="API Keys" icon={KeyRound} />
          <TabButton label="Roles" icon={ShieldCheck} />
          <TabButton label="Audit Logs" icon={ScrollText} />
        </Tab.List>

        <Tab.Panels className="mt-4">
          {/* ---------------------------- API keys --------------------------- */}
          <Tab.Panel className="space-y-4">
            <form onSubmit={handleCreateKey} className={sectionClass}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-sm font-semibold text-white">Create key</h2>
                <p className="text-xs text-gray-500">
                  Requires the active key to hold the <code>admin</code> role.
                </p>
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <label className="block">
                  <span className={labelClass}>Username *</span>
                  <input
                    type="text"
                    className={inputClass}
                    value={username}
                    maxLength={256}
                    placeholder="service-account"
                    onChange={(event) => setUsername(event.target.value)}
                  />
                </label>
                <label className="block">
                  <span className={labelClass}>Email (optional)</span>
                  <input
                    type="email"
                    className={inputClass}
                    value={email}
                    placeholder="owner@example.com"
                    onChange={(event) => setEmail(event.target.value)}
                  />
                </label>
              </div>

              <fieldset>
                <legend className={labelClass}>Roles</legend>
                <div className="flex flex-wrap gap-2">
                  {ASSIGNABLE_ROLES.map((role) => {
                    const checked = selectedRoles.includes(role);
                    return (
                      <label
                        key={role}
                        className={`flex cursor-pointer items-center gap-2 rounded-md border px-3 py-2 text-xs font-medium transition-colors ${
                          checked
                            ? 'border-blue-700 bg-blue-950/40 text-blue-200'
                            : 'border-[#303030] bg-[#0f0f0f] text-gray-400 hover:text-gray-200'
                        }`}
                      >
                        <input
                          type="checkbox"
                          className="h-3.5 w-3.5 accent-blue-500"
                          checked={checked}
                          onChange={() => toggleRole(role)}
                        />
                        {role}
                      </label>
                    );
                  })}
                </div>
              </fieldset>

              {formError && <InlineError>{formError}</InlineError>}

              <div className="flex items-center gap-2">
                <button
                  type="submit"
                  disabled={creating}
                  className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
                >
                  {creating ? 'Creating…' : 'Create key'}
                </button>
                <RefreshButton onClick={() => void loadKeys()} busy={keysLoading} />
              </div>
            </form>

            <section className={sectionClass}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-sm font-semibold text-white">Provisioned keys</h2>
                <p className="text-xs text-gray-500">
                  Raw secrets are never returned by the engine after creation.
                </p>
              </div>

              {keysError && <InlineError>{keysError}</InlineError>}

              {keysLoading && keys.length === 0 ? (
                <p className="text-sm text-gray-500">Loading API keys…</p>
              ) : keys.length === 0 ? (
                <p className="text-sm text-gray-500">
                  {rbacAvailable
                    ? 'No API keys have been provisioned yet.'
                    : 'No API keys available while RBAC is unavailable.'}
                </p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-[#242424]">
                    <thead>
                      <tr>
                        <th className={thClass}>Key ID</th>
                        <th className={thClass}>Username</th>
                        <th className={thClass}>Roles</th>
                        <th className={thClass}>Created</th>
                        <th className={thClass}>Status</th>
                        <th className={thClass}>Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1d1d1d]">
                      {keys.map((key) => {
                        const revoked = key.revoked === true;
                        const confirming = pendingRevokeId === key.key_id;
                        return (
                          <tr key={key.key_id}>
                            <td className={`${tdClass} font-mono text-xs`}>{key.key_id}</td>
                            <td className={tdClass}>{key.username ?? '—'}</td>
                            <td className={tdClass}>
                              <div className="flex flex-wrap gap-1">
                                {key.roles?.length ? (
                                  key.roles.map((role) => (
                                    <Badge key={role} tone="blue">
                                      {role}
                                    </Badge>
                                  ))
                                ) : (
                                  <span className="text-gray-500">—</span>
                                )}
                              </div>
                              {key.permissions?.length ? (
                                <p className="mt-1 text-[11px] text-gray-500">
                                  {key.permissions.join(', ')}
                                </p>
                              ) : null}
                            </td>
                            <td className={`${tdClass} whitespace-nowrap text-xs`}>
                              {formatTimestamp(key.created_at)}
                              {key.created_by && (
                                <span className="block text-[11px] text-gray-500">
                                  by {key.created_by}
                                </span>
                              )}
                            </td>
                            <td className={tdClass}>
                              <Badge tone={revoked ? 'red' : 'green'}>
                                {revoked ? 'revoked' : 'active'}
                              </Badge>
                            </td>
                            <td className={tdClass}>
                              {revoked ? (
                                <span className="text-xs text-gray-500">—</span>
                              ) : confirming ? (
                                <span className="flex items-center gap-2">
                                  <button
                                    type="button"
                                    onClick={() => void handleRevokeKey(key.key_id)}
                                    disabled={revokingId === key.key_id}
                                    className="rounded-md bg-red-600 px-2 py-1 text-xs font-medium text-white hover:bg-red-500 disabled:opacity-50"
                                  >
                                    {revokingId === key.key_id ? 'Revoking…' : 'Confirm'}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => setPendingRevokeId(null)}
                                    className="rounded-md border border-[#303030] bg-[#1b1b1b] px-2 py-1 text-xs font-medium text-gray-300 hover:bg-[#232323]"
                                  >
                                    Cancel
                                  </button>
                                </span>
                              ) : (
                                <button
                                  type="button"
                                  onClick={() => setPendingRevokeId(key.key_id)}
                                  className="rounded-md border border-red-900/60 bg-red-950/30 px-2 py-1 text-xs font-medium text-red-300 hover:bg-red-950/60"
                                >
                                  Revoke
                                </button>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </Tab.Panel>

          {/* ------------------------------ Roles ---------------------------- */}
          <Tab.Panel className="space-y-4">
            <section className={sectionClass}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-sm font-semibold text-white">Roles &amp; permissions</h2>
                <RefreshButton onClick={() => void loadRoles()} busy={rolesLoading} />
              </div>

              {rolesError && <InlineError>{rolesError}</InlineError>}

              {rolesLoading && roles.length === 0 ? (
                <p className="text-sm text-gray-500">Loading roles…</p>
              ) : roles.length === 0 ? (
                <p className="text-sm text-gray-500">No roles reported by the engine.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-[#242424]">
                    <thead>
                      <tr>
                        <th className={thClass}>Name</th>
                        <th className={thClass}>Description</th>
                        <th className={thClass}>Permissions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1d1d1d]">
                      {roles.map((role) => (
                        <tr key={role.name}>
                          <td className={`${tdClass} whitespace-nowrap font-medium text-gray-200`}>
                            {role.name}
                          </td>
                          <td className={tdClass}>{role.description || '—'}</td>
                          <td className={tdClass}>
                            <div className="flex flex-wrap gap-1">
                              {role.permissions?.length ? (
                                role.permissions.map((permission) => (
                                  <Badge key={permission}>{permission}</Badge>
                                ))
                              ) : (
                                <span className="text-gray-500">—</span>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            <form onSubmit={handleCreateRole} className={sectionClass}>
              <h2 className="text-sm font-semibold text-white">Create role</h2>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <label className="block">
                  <span className={labelClass}>Name *</span>
                  <input
                    type="text"
                    className={inputClass}
                    value={roleName}
                    placeholder="auditor"
                    onChange={(event) => setRoleName(event.target.value)}
                  />
                </label>
                <label className="block">
                  <span className={labelClass}>Description</span>
                  <input
                    type="text"
                    className={inputClass}
                    value={roleDescription}
                    placeholder="Read-only auditing access"
                    onChange={(event) => setRoleDescription(event.target.value)}
                  />
                </label>
                <label className="block">
                  <span className={labelClass}>Permissions (comma separated) *</span>
                  <input
                    type="text"
                    className={inputClass}
                    value={rolePermissions}
                    placeholder="read:workflows, read:audit"
                    onChange={(event) => setRolePermissions(event.target.value)}
                  />
                </label>
              </div>

              {roleFormError && <InlineError>{roleFormError}</InlineError>}

              <button
                type="submit"
                disabled={creatingRole}
                className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
              >
                {creatingRole ? 'Creating…' : 'Create role'}
              </button>
            </form>
          </Tab.Panel>

          {/* ---------------------------- Audit logs -------------------------- */}
          <Tab.Panel className="space-y-4">
            <section className={sectionClass}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="text-sm font-semibold text-white">Audit logs</h2>
                  <p className="mt-0.5 text-xs text-gray-500">
                    {auditLogs.length} entr{auditLogs.length === 1 ? 'y' : 'ies'}
                    {auditSource ? ` · source: ${auditSource}` : ''}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-2 text-xs text-gray-400">
                    Limit
                    <select
                      value={auditLimit}
                      onChange={(event) => setAuditLimit(Number(event.target.value))}
                      className="rounded-md border border-[#303030] bg-[#0f0f0f] px-2 py-1 text-xs text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    >
                      {AUDIT_LIMIT_OPTIONS.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  </label>
                  <RefreshButton onClick={() => void loadAuditLogs()} busy={auditLoading} />
                </div>
              </div>

              {auditError && <InlineError>{auditError}</InlineError>}

              {auditLoading && auditLogs.length === 0 ? (
                <p className="text-sm text-gray-500">Loading audit logs…</p>
              ) : auditLogs.length === 0 ? (
                <p className="text-sm text-gray-500">No audit entries recorded.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-[#242424]">
                    <thead>
                      <tr>
                        <th className={thClass}>Timestamp</th>
                        <th className={thClass}>Operation</th>
                        <th className={thClass}>Resource</th>
                        <th className={thClass}>Resource ID</th>
                        <th className={thClass}>Result</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1d1d1d]">
                      {auditLogs.map((entry, index) => {
                        const success = entry.success;
                        return (
                          <tr key={`${readCell(entry, 'timestamp')}-${index}`}>
                            <td className={`${tdClass} whitespace-nowrap text-xs`}>
                              {formatTimestamp(entry.timestamp)}
                            </td>
                            <td className={tdClass}>{readCell(entry, 'operation')}</td>
                            <td className={tdClass}>{readCell(entry, 'resource')}</td>
                            <td className={`${tdClass} font-mono text-xs`}>
                              {readCell(entry, 'resource_id')}
                            </td>
                            <td className={tdClass}>
                              {typeof success === 'boolean' ? (
                                <Badge tone={success ? 'green' : 'red'}>
                                  {success ? 'success' : 'failed'}
                                </Badge>
                              ) : (
                                <span className="text-gray-500">—</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </div>
  );
}
