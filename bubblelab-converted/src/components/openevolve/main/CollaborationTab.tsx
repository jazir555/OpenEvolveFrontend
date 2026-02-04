import React, { useMemo, useRef, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";

interface CollaborationEvent {
  type: string;
  data?: Record<string, unknown>;
  timestamp?: string;
}

export const CollaborationTab: React.FC = () => {
  const [roomId, setRoomId] = useState("default-room");
  const [userId, setUserId] = useState(`user-${Math.floor(Math.random() * 1000)}`);
  const [username, setUsername] = useState("Operator");
  const [status, setStatus] = useState("disconnected");
  const [presence, setPresence] = useState<Record<string, string>>({});
  const [sharedContent, setSharedContent] = useState("");
  const [cursorLine, setCursorLine] = useState("1");
  const [cursorColumn, setCursorColumn] = useState("1");
  const [events, setEvents] = useState<CollaborationEvent[]>([]);
  const socketRef = useRef<WebSocket | null>(null);

  const wsBase = useMemo(() => {
    const direct = (globalThis as any)?.OPENEVOLVE_API_BASE as string | undefined;
    if (direct) {
      return direct.replace(/^http/, "ws");
    }
    try {
      const stored = globalThis.localStorage?.getItem("openevolve_api_base") || "";
      return stored ? stored.replace(/^http/, "ws") : "";
    } catch {
      return "";
    }
  }, []);

  const appendEvent = (event: CollaborationEvent) => {
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

    const wsUrl = `${wsBase}/ws/collaboration/${encodeURIComponent(roomId)}?user_id=${encodeURIComponent(
      userId,
    )}&username=${encodeURIComponent(username)}`;
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
        const payload = JSON.parse(message.data) as CollaborationEvent;
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
      } catch {
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
    if (!socketRef.current) return;
    socketRef.current.send(
      JSON.stringify({
        type: "content_update",
        content: sharedContent,
      }),
    );
  };

  const sendCursorUpdate = () => {
    if (!socketRef.current) return;
    socketRef.current.send(
      JSON.stringify({
        type: "cursor_update",
        position: {
          line: Number(cursorLine),
          column: Number(cursorColumn),
        },
      }),
    );
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Collaboration</CardTitle>
          <CardDescription>Real-time collaboration via WebSocket rooms.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Room ID</Label>
              <Input value={roomId} onChange={(event) => setRoomId(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>User ID</Label>
              <Input value={userId} onChange={(event) => setUserId(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Username</Label>
              <Input value={username} onChange={(event) => setUsername(event.target.value)} />
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={connect}>
              Connect
            </Button>
            <Button variant="outline" onClick={disconnect}>
              Disconnect
            </Button>
            <Badge variant={status === "connected" ? "default" : "secondary"}>{status}</Badge>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Shared Document</CardTitle>
          <CardDescription>Broadcast content updates to the room.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={sharedContent}
            onChange={(event) => setSharedContent(event.target.value)}
            rows={6}
          />
          <Button onClick={sendContentUpdate}>Send Content Update</Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Cursor Updates</CardTitle>
          <CardDescription>Share cursor positions with other collaborators.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-3">
          <div className="space-y-2">
            <Label>Line</Label>
            <Input value={cursorLine} onChange={(event) => setCursorLine(event.target.value)} />
          </div>
          <div className="space-y-2">
            <Label>Column</Label>
            <Input value={cursorColumn} onChange={(event) => setCursorColumn(event.target.value)} />
          </div>
          <div className="flex items-end">
            <Button variant="outline" onClick={sendCursorUpdate}>
              Broadcast Cursor
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Presence</CardTitle>
            <CardDescription>Active users in the room.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {Object.keys(presence).length === 0 ? (
              <div className="text-muted-foreground">No users connected.</div>
            ) : (
              Object.entries(presence).map(([id, name]) => (
                <div key={id} className="flex items-center justify-between">
                  <span>{name}</span>
                  <Badge variant="secondary">{id}</Badge>
                </div>
              ))
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Event Log</CardTitle>
            <CardDescription>Recent collaboration events.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            {events.length === 0 ? (
              <div className="text-muted-foreground">No events yet.</div>
            ) : (
              events.map((event, index) => (
                <div key={`event-${index}`} className="rounded border p-2">
                  <div className="font-semibold">{event.type}</div>
                  <div className="text-muted-foreground">
                    {event.timestamp ?? ""} {event.data ? JSON.stringify(event.data) : ""}
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
