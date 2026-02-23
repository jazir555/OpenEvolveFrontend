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
exports.NotificationsTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const select_1 = require("@/components/ui/select");
const STORAGE_KEY = "openevolve_notifications";
const readStorage = () => {
    try {
        const raw = globalThis.localStorage?.getItem(STORAGE_KEY);
        if (!raw)
            return [];
        return JSON.parse(raw);
    }
    catch {
        return [];
    }
};
const writeStorage = (notifications) => {
    try {
        globalThis.localStorage?.setItem(STORAGE_KEY, JSON.stringify(notifications));
    }
    catch {
        // ignore storage errors
    }
};
const createId = () => {
    if (crypto?.randomUUID) {
        return crypto.randomUUID();
    }
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};
const NotificationsTab = () => {
    const [notifications, setNotifications] = (0, react_1.useState)(readStorage);
    const [recipient, setRecipient] = (0, react_1.useState)("all");
    const [sender, setSender] = (0, react_1.useState)("Current User");
    const [message, setMessage] = (0, react_1.useState)("");
    const [type, setType] = (0, react_1.useState)("info");
    const [filterType, setFilterType] = (0, react_1.useState)("All");
    const [filterRead, setFilterRead] = (0, react_1.useState)("All");
    const persist = (next) => {
        setNotifications(next);
        writeStorage(next);
    };
    const handleSend = () => {
        if (!recipient.trim() || !message.trim())
            return;
        const newNotification = {
            id: createId(),
            sender: sender || "System",
            recipient,
            message,
            type,
            timestamp: new Date().toISOString(),
            read: false,
        };
        persist([newNotification, ...notifications]);
        setMessage("");
    };
    const filteredNotifications = (0, react_1.useMemo)(() => {
        let results = notifications.filter((notification) => notification.recipient.toLowerCase() === "all" ||
            notification.recipient.toLowerCase() === recipient.toLowerCase() ||
            recipient.toLowerCase() === "all");
        if (filterType !== "All") {
            results = results.filter((notification) => notification.type === filterType);
        }
        if (filterRead === "Unread") {
            results = results.filter((notification) => !notification.read);
        }
        if (filterRead === "Read") {
            results = results.filter((notification) => notification.read);
        }
        return results;
    }, [notifications, recipient, filterType, filterRead]);
    const markAllRead = () => {
        const next = notifications.map((notification) => ({ ...notification, read: true }));
        persist(next);
    };
    const toggleRead = (id) => {
        const next = notifications.map((notification) => notification.id === id ? { ...notification, read: !notification.read } : notification);
        persist(next);
    };
    const stats = (0, react_1.useMemo)(() => {
        const total = notifications.length;
        const unread = notifications.filter((notification) => !notification.read).length;
        return { total, unread };
    }, [notifications]);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Notifications Center</card_1.CardTitle>
          <card_1.CardDescription>Send and manage collaboration alerts.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Send Notification</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <label_1.Label>Recipient</label_1.Label>
                  <input_1.Input value={recipient} onChange={(event) => setRecipient(event.target.value)}/>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Type</label_1.Label>
                  <select_1.Select value={type} onValueChange={(value) => setType(value)}>
                    <select_1.SelectTrigger>
                      <select_1.SelectValue />
                    </select_1.SelectTrigger>
                    <select_1.SelectContent>
                      <select_1.SelectItem value="info">info</select_1.SelectItem>
                      <select_1.SelectItem value="warning">warning</select_1.SelectItem>
                      <select_1.SelectItem value="error">error</select_1.SelectItem>
                      <select_1.SelectItem value="success">success</select_1.SelectItem>
                    </select_1.SelectContent>
                  </select_1.Select>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Sender</label_1.Label>
                  <input_1.Input value={sender} onChange={(event) => setSender(event.target.value)}/>
                </div>
              </div>
              <div className="space-y-2">
                <label_1.Label>Message</label_1.Label>
                <textarea_1.Textarea value={message} onChange={(event) => setMessage(event.target.value)}/>
              </div>
              <button_1.Button onClick={handleSend}>Send Notification</button_1.Button>
            </card_1.CardContent>
          </card_1.Card>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Filters</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="grid gap-3 md:grid-cols-2">
              <div className="space-y-2">
                <label_1.Label>Type</label_1.Label>
                <select_1.Select value={filterType} onValueChange={setFilterType}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue />
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    <select_1.SelectItem value="All">All</select_1.SelectItem>
                    <select_1.SelectItem value="info">info</select_1.SelectItem>
                    <select_1.SelectItem value="warning">warning</select_1.SelectItem>
                    <select_1.SelectItem value="error">error</select_1.SelectItem>
                    <select_1.SelectItem value="success">success</select_1.SelectItem>
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
              <div className="space-y-2">
                <label_1.Label>Read Status</label_1.Label>
                <select_1.Select value={filterRead} onValueChange={setFilterRead}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue />
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    <select_1.SelectItem value="All">All</select_1.SelectItem>
                    <select_1.SelectItem value="Unread">Unread</select_1.SelectItem>
                    <select_1.SelectItem value="Read">Read</select_1.SelectItem>
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
            </card_1.CardContent>
          </card_1.Card>

          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              Total: {stats.total} · Unread: {stats.unread}
            </div>
            <button_1.Button variant="outline" onClick={markAllRead}>
              Mark All Read
            </button_1.Button>
          </div>

          <div className="space-y-2">
            {filteredNotifications.length === 0 && (<div className="text-sm text-muted-foreground">No notifications.</div>)}
            {filteredNotifications.map((notification) => (<div key={notification.id} className="rounded border p-3 text-sm space-y-2">
                <div className="flex items-center justify-between">
                  <div className="font-semibold">{notification.sender}</div>
                  <badge_1.Badge variant={notification.read ? "secondary" : "default"}>
                    {notification.read ? "Read" : "Unread"}
                  </badge_1.Badge>
                </div>
                <div className="text-xs text-muted-foreground">
                  {notification.type} · {new Date(notification.timestamp).toLocaleString()}
                </div>
                <div>{notification.message}</div>
                <button_1.Button size="sm" variant="outline" onClick={() => toggleRead(notification.id)}>
                  {notification.read ? "Mark Unread" : "Mark Read"}
                </button_1.Button>
              </div>))}
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.NotificationsTab = NotificationsTab;
//# sourceMappingURL=NotificationsTab.js.map