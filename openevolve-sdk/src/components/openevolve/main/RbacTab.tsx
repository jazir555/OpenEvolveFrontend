import React, { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

interface RbacUser {
  user_id: string;
  username: string;
  email: string;
  password: string;
  full_name?: string;
  role_names: string[];
  is_superuser?: boolean;
}

interface RbacRole {
  name: string;
  description: string;
  permissions: string[];
  system_role?: boolean;
}

interface RbacAuditLog {
  action: string;
  user_id: string;
  timestamp: string;
  success: boolean;
  details?: Record<string, unknown>;
}

interface RbacState {
  users: Record<string, RbacUser>;
  roles: Record<string, RbacRole>;
  logs: RbacAuditLog[];
  current_user?: RbacUser;
}

const DEFAULT_STATE: RbacState = {
  users: {},
  roles: {
    admin: {
      name: "admin",
      description: "System administrator",
      permissions: ["manage_users", "manage_roles", "view_audit"],
      system_role: true,
    },
    analyst: {
      name: "analyst",
      description: "Read-only analyst",
      permissions: ["view_audit"],
    },
  },
  logs: [],
};

const readState = (): RbacState => {
  try {
    const raw = globalThis.localStorage?.getItem("openevolve_rbac_state");
    if (raw) {
      return JSON.parse(raw) as RbacState;
    }
  } catch {
    // ignore
  }
  return DEFAULT_STATE;
};

const persistState = (state: RbacState) => {
  try {
    globalThis.localStorage?.setItem("openevolve_rbac_state", JSON.stringify(state));
  } catch {
    // ignore
  }
};

export const RbacTab: React.FC = () => {
  const [state, setState] = useState<RbacState>(readState);
  const [loginUsername, setLoginUsername] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [newUser, setNewUser] = useState({
    username: "",
    email: "",
    password: "",
    full_name: "",
    roles: "",
    is_superuser: false,
  });
  const [newRole, setNewRole] = useState({
    name: "",
    description: "",
    permissions: "",
  });

  const currentUser = state.current_user;
  const userList = Object.values(state.users);
  const roleList = Object.values(state.roles);

  const addLog = (log: RbacAuditLog) => {
    const next = { ...state, logs: [log, ...state.logs].slice(0, 200) };
    setState(next);
    persistState(next);
  };

  const handleLogin = () => {
    setErrorMessage(null);
    const match = userList.find(
      (user) => user.username === loginUsername && user.password === loginPassword,
    );
    if (!match) {
      setErrorMessage("Invalid username or password.");
      addLog({
        action: "LOGIN",
        user_id: loginUsername,
        timestamp: new Date().toISOString(),
        success: false,
      });
      return;
    }
    const next = { ...state, current_user: match };
    setState(next);
    persistState(next);
    addLog({
      action: "LOGIN",
      user_id: match.user_id,
      timestamp: new Date().toISOString(),
      success: true,
    });
    setMessage(`Welcome ${match.username}.`);
  };

  const handleLogout = () => {
    const next = { ...state };
    delete next.current_user;
    setState(next);
    persistState(next);
  };

  const handleCreateUser = () => {
    if (!newUser.username.trim()) {
      setErrorMessage("Username is required.");
      return;
    }
    const userId = `user_${Math.random().toString(36).slice(2, 8)}`;
    const roles = newUser.roles
      .split(",")
      .map((role) => role.trim())
      .filter(Boolean);
    const user: RbacUser = {
      user_id: userId,
      username: newUser.username.trim(),
      email: newUser.email.trim(),
      password: newUser.password,
      full_name: newUser.full_name.trim(),
      role_names: roles,
      is_superuser: newUser.is_superuser,
    };
    const next = {
      ...state,
      users: {
        ...state.users,
        [userId]: user,
      },
    };
    setState(next);
    persistState(next);
    addLog({
      action: "CREATE_USER",
      user_id: currentUser?.user_id ?? "system",
      timestamp: new Date().toISOString(),
      success: true,
      details: { created_user: user.username },
    });
    setNewUser({ username: "", email: "", password: "", full_name: "", roles: "", is_superuser: false });
    setMessage("User created.");
  };

  const handleDeleteUser = (userId: string) => {
    const nextUsers = { ...state.users };
    const deleted = nextUsers[userId];
    delete nextUsers[userId];
    const next = { ...state, users: nextUsers };
    setState(next);
    persistState(next);
    addLog({
      action: "DELETE_USER",
      user_id: currentUser?.user_id ?? "system",
      timestamp: new Date().toISOString(),
      success: true,
      details: { deleted_user: deleted?.username },
    });
  };

  const handleCreateRole = () => {
    if (!newRole.name.trim()) {
      setErrorMessage("Role name is required.");
      return;
    }
    const role: RbacRole = {
      name: newRole.name.trim(),
      description: newRole.description.trim(),
      permissions: newRole.permissions
        .split(",")
        .map((perm) => perm.trim())
        .filter(Boolean),
    };
    const next = { ...state, roles: { ...state.roles, [role.name]: role } };
    setState(next);
    persistState(next);
    addLog({
      action: "CREATE_ROLE",
      user_id: currentUser?.user_id ?? "system",
      timestamp: new Date().toISOString(),
      success: true,
      details: { role: role.name },
    });
    setNewRole({ name: "", description: "", permissions: "" });
    setMessage("Role created.");
  };

  const handleDeleteRole = (roleName: string) => {
    const nextRoles = { ...state.roles };
    delete nextRoles[roleName];
    const next = { ...state, roles: nextRoles };
    setState(next);
    persistState(next);
    addLog({
      action: "DELETE_ROLE",
      user_id: currentUser?.user_id ?? "system",
      timestamp: new Date().toISOString(),
      success: true,
      details: { role: roleName },
    });
  };

  const auditLog = useMemo(() => state.logs.slice(0, 50), [state.logs]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>RBAC Management</CardTitle>
          <CardDescription>Manage local RBAC users, roles, and audit logs.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!currentUser ? (
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <Label>Username</Label>
                <Input value={loginUsername} onChange={(event) => setLoginUsername(event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Password</Label>
                <Input
                  type="password"
                  value={loginPassword}
                  onChange={(event) => setLoginPassword(event.target.value)}
                />
              </div>
              <div className="flex items-end">
                <Button onClick={handleLogin}>Login</Button>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-between">
              <div>
                Logged in as <strong>{currentUser.username}</strong>
              </div>
              <Button variant="outline" onClick={handleLogout}>
                Logout
              </Button>
            </div>
          )}

          {message ? <div className="text-sm text-green-600">{message}</div> : null}
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Users</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>New User</Label>
              <Input
                placeholder="Username"
                value={newUser.username}
                onChange={(event) => setNewUser({ ...newUser, username: event.target.value })}
              />
              <Input
                placeholder="Email"
                value={newUser.email}
                onChange={(event) => setNewUser({ ...newUser, email: event.target.value })}
              />
              <Input
                type="password"
                placeholder="Password"
                value={newUser.password}
                onChange={(event) => setNewUser({ ...newUser, password: event.target.value })}
              />
              <Input
                placeholder="Full name"
                value={newUser.full_name}
                onChange={(event) => setNewUser({ ...newUser, full_name: event.target.value })}
              />
              <Input
                placeholder="Roles (comma-separated)"
                value={newUser.roles}
                onChange={(event) => setNewUser({ ...newUser, roles: event.target.value })}
              />
              <Button onClick={handleCreateUser}>Create User</Button>
            </div>
            <Separator />
            <div className="space-y-2">
              {userList.length === 0 && (
                <div className="text-sm text-muted-foreground">No users found.</div>
              )}
              {userList.map((user) => (
                <div key={user.user_id} className="rounded border p-3 space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="font-semibold">{user.username}</div>
                    <Button size="sm" variant="destructive" onClick={() => handleDeleteUser(user.user_id)}>
                      Delete
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground">{user.email}</div>
                  <div className="flex flex-wrap gap-2">
                    {user.role_names.map((role) => (
                      <Badge key={role} variant="secondary">
                        {role}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Roles</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>New Role</Label>
              <Input
                placeholder="Role name"
                value={newRole.name}
                onChange={(event) => setNewRole({ ...newRole, name: event.target.value })}
              />
              <Input
                placeholder="Description"
                value={newRole.description}
                onChange={(event) => setNewRole({ ...newRole, description: event.target.value })}
              />
              <Textarea
                placeholder="Permissions (comma-separated)"
                value={newRole.permissions}
                onChange={(event) => setNewRole({ ...newRole, permissions: event.target.value })}
                rows={3}
              />
              <Button onClick={handleCreateRole}>Create Role</Button>
            </div>
            <Separator />
            <div className="space-y-2">
              {roleList.map((role) => (
                <div key={role.name} className="rounded border p-3 space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="font-semibold">{role.name}</div>
                    <Button size="sm" variant="destructive" onClick={() => handleDeleteRole(role.name)}>
                      Delete
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground">{role.description}</div>
                  <div className="flex flex-wrap gap-2">
                    {role.permissions.map((perm) => (
                      <Badge key={perm} variant="outline">
                        {perm}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Audit Logs</CardTitle>
          <CardDescription>Recent RBAC actions.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {auditLog.length === 0 && (
            <div className="text-sm text-muted-foreground">No audit logs.</div>
          )}
          {auditLog.map((log, index) => (
            <div key={`${log.timestamp}-${index}`} className="rounded border p-3 space-y-1 text-sm">
              <div className="flex items-center justify-between">
                <div className="font-semibold">{log.action}</div>
                <Badge variant={log.success ? "secondary" : "destructive"}>
                  {log.success ? "Success" : "Failed"}
                </Badge>
              </div>
              <div className="text-xs text-muted-foreground">
                {log.user_id} · {new Date(log.timestamp).toLocaleString()}
              </div>
              {log.details ? (
                <pre className="text-xs whitespace-pre-wrap rounded border p-2">
                  {JSON.stringify(log.details, null, 2)}
                </pre>
              ) : null}
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
};
