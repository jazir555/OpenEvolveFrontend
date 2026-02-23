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
exports.CollaborationTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const CollaborationTab = () => {
    const [roomId, setRoomId] = (0, react_1.useState)("default-room");
    const [userId, setUserId] = (0, react_1.useState)(`user-${Math.floor(Math.random() * 1000)}`);
    const [username, setUsername] = (0, react_1.useState)("Operator");
    const [status, setStatus] = (0, react_1.useState)("disconnected");
    const [presence, setPresence] = (0, react_1.useState)({});
    const [sharedContent, setSharedContent] = (0, react_1.useState)("");
    const [cursorLine, setCursorLine] = (0, react_1.useState)("1");
    const [cursorColumn, setCursorColumn] = (0, react_1.useState)("1");
    const [events, setEvents] = (0, react_1.useState)([]);
    const socketRef = (0, react_1.useRef)(null);
    const wsBase = (0, react_1.useMemo)(() => {
        const direct = globalThis?.OPENEVOLVE_API_BASE;
        if (direct) {
            return direct.replace(/^http/, "ws");
        }
        try {
            const stored = globalThis.localStorage?.getItem("openevolve_api_base") || "";
            return stored ? stored.replace(/^http/, "ws") : "";
        }
        catch {
            return "";
        }
    }, []);
    const appendEvent = (event) => {
        setEvents((prev) => [event, ...prev].slice(0, 30));
    };
    const connect = () => {
        if (!wsBase) {
            setStatus("missing-api-base");
            return;
        }
        if (socketRef.current) {
            socketRef.current.close();
        }
        const wsUrl = `${wsBase}/ws/collaboration/${encodeURIComponent(roomId)}?user_id=${encodeURIComponent(userId)}&username=${encodeURIComponent(username)}`;
        const socket = new WebSocket(wsUrl);
        socketRef.current = socket;
        socket.onopen = () => {
            setStatus("connected");
            appendEvent({ type: "connected" });
        };
        socket.onclose = () => {
            setStatus("disconnected");
            appendEvent({ type: "disconnected" });
        };
        socket.onerror = () => {
            setStatus("error");
            appendEvent({ type: "error" });
        };
        socket.onmessage = (message) => {
            try {
                const payload = JSON.parse(message.data);
                appendEvent(payload);
                if (payload.type === "user_joined") {
                    const data = payload.data || {};
                    setPresence((prev) => ({
                        ...prev,
                        [String(data.user_id ?? "unknown")]: String(data.username ?? data.user_id ?? "user"),
                    }));
                }
                if (payload.type === "user_left") {
                    const data = payload.data || {};
                    setPresence((prev) => {
                        const next = { ...prev };
                        delete next[String(data.user_id ?? "unknown")];
                        return next;
                    });
                }
                if (payload.type === "content_update") {
                    const data = payload.data || {};
                    setSharedContent(String(data.content ?? ""));
                }
            }
            catch {
                appendEvent({ type: "message", data: { raw: message.data } });
            }
        };
    };
    const disconnect = () => {
        socketRef.current?.close();
        socketRef.current = null;
        setStatus("disconnected");
    };
    const sendContentUpdate = () => {
        if (!socketRef.current)
            return;
        socketRef.current.send(JSON.stringify({
            type: "content_update",
            content: sharedContent,
        }));
    };
    const sendCursorUpdate = () => {
        if (!socketRef.current)
            return;
        socketRef.current.send(JSON.stringify({
            type: "cursor_update",
            position: {
                line: Number(cursorLine),
                column: Number(cursorColumn),
            },
        }));
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Collaboration</card_1.CardTitle>
          <card_1.CardDescription>Real-time collaboration via WebSocket rooms.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <label_1.Label>Room ID</label_1.Label>
              <input_1.Input value={roomId} onChange={(event) => setRoomId(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>User ID</label_1.Label>
              <input_1.Input value={userId} onChange={(event) => setUserId(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Username</label_1.Label>
              <input_1.Input value={username} onChange={(event) => setUsername(event.target.value)}/>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <button_1.Button variant="outline" onClick={connect}>
              Connect
            </button_1.Button>
            <button_1.Button variant="outline" onClick={disconnect}>
              Disconnect
            </button_1.Button>
            <badge_1.Badge variant={status === "connected" ? "default" : "secondary"}>{status}</badge_1.Badge>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle className="text-sm">Shared Document</card_1.CardTitle>
          <card_1.CardDescription>Broadcast content updates to the room.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-3">
          <textarea_1.Textarea value={sharedContent} onChange={(event) => setSharedContent(event.target.value)} rows={6}/>
          <button_1.Button onClick={sendContentUpdate}>Send Content Update</button_1.Button>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle className="text-sm">Cursor Updates</card_1.CardTitle>
          <card_1.CardDescription>Share cursor positions with other collaborators.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="grid gap-3 md:grid-cols-3">
          <div className="space-y-2">
            <label_1.Label>Line</label_1.Label>
            <input_1.Input value={cursorLine} onChange={(event) => setCursorLine(event.target.value)}/>
          </div>
          <div className="space-y-2">
            <label_1.Label>Column</label_1.Label>
            <input_1.Input value={cursorColumn} onChange={(event) => setCursorColumn(event.target.value)}/>
          </div>
          <div className="flex items-end">
            <button_1.Button variant="outline" onClick={sendCursorUpdate}>
              Broadcast Cursor
            </button_1.Button>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <div className="grid gap-4 md:grid-cols-2">
        <card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="text-sm">Presence</card_1.CardTitle>
            <card_1.CardDescription>Active users in the room.</card_1.CardDescription>
          </card_1.CardHeader>
          <card_1.CardContent className="space-y-2 text-sm">
            {Object.keys(presence).length === 0 ? (<div className="text-muted-foreground">No users connected.</div>) : (Object.entries(presence).map(([id, name]) => (<div key={id} className="flex items-center justify-between">
                  <span>{name}</span>
                  <badge_1.Badge variant="secondary">{id}</badge_1.Badge>
                </div>)))}
          </card_1.CardContent>
        </card_1.Card>
        <card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="text-sm">Event Log</card_1.CardTitle>
            <card_1.CardDescription>Recent collaboration events.</card_1.CardDescription>
          </card_1.CardHeader>
          <card_1.CardContent className="space-y-2 text-xs">
            {events.length === 0 ? (<div className="text-muted-foreground">No events yet.</div>) : (events.map((event, index) => (<div key={`event-${index}`} className="rounded border p-2">
                  <div className="font-semibold">{event.type}</div>
                  <div className="text-muted-foreground">
                    {event.timestamp ?? ""} {event.data ? JSON.stringify(event.data) : ""}
                  </div>
                </div>)))}
          </card_1.CardContent>
        </card_1.Card>
      </div>
    </div>);
};
exports.CollaborationTab = CollaborationTab;
//# sourceMappingURL=CollaborationTab.js.map