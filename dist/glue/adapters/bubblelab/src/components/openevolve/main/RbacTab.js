"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.RbacTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const separator_1 = require("@/components/ui/separator");
const DEFAULT_STATE = {
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
const readState = () => {
    try {
        const raw = globalThis.localStorage?.getItem("openevolve_rbac_state");
        if (raw) {
            return JSON.parse(raw);
        }
    }
    catch {
        // ignore
    }
    return DEFAULT_STATE;
};
const persistState = (state) => {
    try {
        globalThis.localStorage?.setItem("openevolve_rbac_state", JSON.stringify(state));
    }
    catch {
        // ignore
    }
};
const RbacTab = () => {
    const [state, setState] = (0, react_1.useState)(readState);
    const [loginUsername, setLoginUsername] = (0, react_1.useState)("");
    const [loginPassword, setLoginPassword] = (0, react_1.useState)("");
    const [message, setMessage] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [newUser, setNewUser] = (0, react_1.useState)({
        username: "",
        email: "",
        password: "",
        full_name: "",
        roles: "",
        is_superuser: false,
    });
    const [newRole, setNewRole] = (0, react_1.useState)({
        name: "",
        description: "",
        permissions: "",
    });
    const currentUser = state.current_user;
    const userList = Object.values(state.users);
    const roleList = Object.values(state.roles);
    const addLog = (log) => {
        const next = { ...state, logs: [log, ...state.logs].slice(0, 200) };
        setState(next);
        persistState(next);
    };
    const handleLogin = () => {
        setErrorMessage(null);
        const match = userList.find((user) => user.username === loginUsername && user.password === loginPassword);
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
        const user = {
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
    const handleDeleteUser = (userId) => {
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
        const role = {
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
    const handleDeleteRole = (roleName) => {
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
    const auditLog = (0, react_1.useMemo)(() => state.logs.slice(0, 50), [state.logs]);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>RBAC Management</card_1.CardTitle>
          <card_1.CardDescription>Manage local RBAC users, roles, and audit logs.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          {!currentUser ? (<div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <label_1.Label>Username</label_1.Label>
                <input_1.Input value={loginUsername} onChange={(event) => setLoginUsername(event.target.value)}/>
              </div>
              <div className="space-y-2">
                <label_1.Label>Password</label_1.Label>
                <input_1.Input type="password" value={loginPassword} onChange={(event) => setLoginPassword(event.target.value)}/>
              </div>
              <div className="flex items-end">
                <button_1.Button onClick={handleLogin}>Login</button_1.Button>
              </div>
            </div>) : (<div className="flex items-center justify-between">
              <div>
                Logged in as <strong>{currentUser.username}</strong>
              </div>
              <button_1.Button variant="outline" onClick={handleLogout}>
                Logout
              </button_1.Button>
            </div>)}

          {message ? <div className="text-sm text-green-600">{message}</div> : null}
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
        </card_1.CardContent>
      </card_1.Card>

      <div className="grid gap-6 md:grid-cols-2">
        <card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="text-base">Users</card_1.CardTitle>
          </card_1.CardHeader>
          <card_1.CardContent className="space-y-4">
            <div className="space-y-2">
              <label_1.Label>New User</label_1.Label>
              <input_1.Input placeholder="Username" value={newUser.username} onChange={(event) => setNewUser({ ...newUser, username: event.target.value })}/>
              <input_1.Input placeholder="Email" value={newUser.email} onChange={(event) => setNewUser({ ...newUser, email: event.target.value })}/>
              <input_1.Input type="password" placeholder="Password" value={newUser.password} onChange={(event) => setNewUser({ ...newUser, password: event.target.value })}/>
              <input_1.Input placeholder="Full name" value={newUser.full_name} onChange={(event) => setNewUser({ ...newUser, full_name: event.target.value })}/>
              <input_1.Input placeholder="Roles (comma-separated)" value={newUser.roles} onChange={(event) => setNewUser({ ...newUser, roles: event.target.value })}/>
              <button_1.Button onClick={handleCreateUser}>Create User</button_1.Button>
            </div>
            <separator_1.Separator />
            <div className="space-y-2">
              {userList.length === 0 && (<div className="text-sm text-muted-foreground">No users found.</div>)}
              {userList.map((user) => (<div key={user.user_id} className="rounded border p-3 space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="font-semibold">{user.username}</div>
                    <button_1.Button size="sm" variant="destructive" onClick={() => handleDeleteUser(user.user_id)}>
                      Delete
                    </button_1.Button>
                  </div>
                  <div className="text-xs text-muted-foreground">{user.email}</div>
                  <div className="flex flex-wrap gap-2">
                    {user.role_names.map((role) => (<badge_1.Badge key={role} variant="secondary">
                        {role}
                      </badge_1.Badge>))}
                  </div>
                </div>))}
            </div>
          </card_1.CardContent>
        </card_1.Card>

        <card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="text-base">Roles</card_1.CardTitle>
          </card_1.CardHeader>
          <card_1.CardContent className="space-y-4">
            <div className="space-y-2">
              <label_1.Label>New Role</label_1.Label>
              <input_1.Input placeholder="Role name" value={newRole.name} onChange={(event) => setNewRole({ ...newRole, name: event.target.value })}/>
              <input_1.Input placeholder="Description" value={newRole.description} onChange={(event) => setNewRole({ ...newRole, description: event.target.value })}/>
              <textarea_1.Textarea placeholder="Permissions (comma-separated)" value={newRole.permissions} onChange={(event) => setNewRole({ ...newRole, permissions: event.target.value })} rows={3}/>
              <button_1.Button onClick={handleCreateRole}>Create Role</button_1.Button>
            </div>
            <separator_1.Separator />
            <div className="space-y-2">
              {roleList.map((role) => (<div key={role.name} className="rounded border p-3 space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="font-semibold">{role.name}</div>
                    <button_1.Button size="sm" variant="destructive" onClick={() => handleDeleteRole(role.name)}>
                      Delete
                    </button_1.Button>
                  </div>
                  <div className="text-xs text-muted-foreground">{role.description}</div>
                  <div className="flex flex-wrap gap-2">
                    {role.permissions.map((perm) => (<badge_1.Badge key={perm} variant="outline">
                        {perm}
                      </badge_1.Badge>))}
                  </div>
                </div>))}
            </div>
          </card_1.CardContent>
        </card_1.Card>
      </div>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle className="text-base">Audit Logs</card_1.CardTitle>
          <card_1.CardDescription>Recent RBAC actions.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-2">
          {auditLog.length === 0 && (<div className="text-sm text-muted-foreground">No audit logs.</div>)}
          {auditLog.map((log, index) => (<div key={`${log.timestamp}-${index}`} className="rounded border p-3 space-y-1 text-sm">
              <div className="flex items-center justify-between">
                <div className="font-semibold">{log.action}</div>
                <badge_1.Badge variant={log.success ? "secondary" : "destructive"}>
                  {log.success ? "Success" : "Failed"}
                </badge_1.Badge>
              </div>
              <div className="text-xs text-muted-foreground">
                {log.user_id} · {new Date(log.timestamp).toLocaleString()}
              </div>
              {log.details ? (<pre className="text-xs whitespace-pre-wrap rounded border p-2">
                  {JSON.stringify(log.details, null, 2)}
                </pre>) : null}
            </div>))}
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.RbacTab = RbacTab;
//# sourceMappingURL=RbacTab.js.map