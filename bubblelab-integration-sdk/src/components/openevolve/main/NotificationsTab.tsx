import React, { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type NotificationItem = {
  id: string;
  sender: string;
  recipient: string;
  message: string;
  type: "info" | "warning" | "error" | "success";
  timestamp: string;
  read: boolean;
};

const STORAGE_KEY = "openevolve_notifications";

const readStorage = (): NotificationItem[] => {
  try {
    const raw = globalThis.localStorage?.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as NotificationItem[];
  } catch {
    return [];
  }
};

const writeStorage = (notifications: NotificationItem[]) => {
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, JSON.stringify(notifications));
  } catch {
    // ignore storage errors
  }
};

const createId = () => {
  if (crypto?.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

export const NotificationsTab: React.FC = () => {
  const [notifications, setNotifications] = useState<NotificationItem[]>(readStorage);
  const [recipient, setRecipient] = useState("all");
  const [sender, setSender] = useState("Current User");
  const [message, setMessage] = useState("");
  const [type, setType] = useState<NotificationItem["type"]>("info");
  const [filterType, setFilterType] = useState("All");
  const [filterRead, setFilterRead] = useState("All");

  const persist = (next: NotificationItem[]) => {
    setNotifications(next);
    writeStorage(next);
  };

  const handleSend = () => {
    if (!recipient.trim() || !message.trim()) return;
    const newNotification: NotificationItem = {
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

  const filteredNotifications = useMemo(() => {
    let results = notifications.filter(
      (notification) =>
        notification.recipient.toLowerCase() === "all" ||
        notification.recipient.toLowerCase() === recipient.toLowerCase() ||
        recipient.toLowerCase() === "all",
    );
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

  const toggleRead = (id: string) => {
    const next = notifications.map((notification) =>
      notification.id === id ? { ...notification, read: !notification.read } : notification,
    );
    persist(next);
  };

  const stats = useMemo(() => {
    const total = notifications.length;
    const unread = notifications.filter((notification) => !notification.read).length;
    return { total, unread };
  }, [notifications]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Notifications Center</CardTitle>
          <CardDescription>Send and manage collaboration alerts.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Send Notification</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Recipient</Label>
                  <Input value={recipient} onChange={(event) => setRecipient(event.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label>Type</Label>
                  <Select value={type} onValueChange={(value) => setType(value as NotificationItem["type"])}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="info">info</SelectItem>
                      <SelectItem value="warning">warning</SelectItem>
                      <SelectItem value="error">error</SelectItem>
                      <SelectItem value="success">success</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Sender</Label>
                  <Input value={sender} onChange={(event) => setSender(event.target.value)} />
                </div>
              </div>
              <div className="space-y-2">
                <Label>Message</Label>
                <Textarea value={message} onChange={(event) => setMessage(event.target.value)} />
              </div>
              <Button onClick={handleSend}>Send Notification</Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Filters</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-3 md:grid-cols-2">
              <div className="space-y-2">
                <Label>Type</Label>
                <Select value={filterType} onValueChange={setFilterType}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="All">All</SelectItem>
                    <SelectItem value="info">info</SelectItem>
                    <SelectItem value="warning">warning</SelectItem>
                    <SelectItem value="error">error</SelectItem>
                    <SelectItem value="success">success</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Read Status</Label>
                <Select value={filterRead} onValueChange={setFilterRead}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="All">All</SelectItem>
                    <SelectItem value="Unread">Unread</SelectItem>
                    <SelectItem value="Read">Read</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              Total: {stats.total} · Unread: {stats.unread}
            </div>
            <Button variant="outline" onClick={markAllRead}>
              Mark All Read
            </Button>
          </div>

          <div className="space-y-2">
            {filteredNotifications.length === 0 && (
              <div className="text-sm text-muted-foreground">No notifications.</div>
            )}
            {filteredNotifications.map((notification) => (
              <div key={notification.id} className="rounded border p-3 text-sm space-y-2">
                <div className="flex items-center justify-between">
                  <div className="font-semibold">{notification.sender}</div>
                  <Badge variant={notification.read ? "secondary" : "default"}>
                    {notification.read ? "Read" : "Unread"}
                  </Badge>
                </div>
                <div className="text-xs text-muted-foreground">
                  {notification.type} · {new Date(notification.timestamp).toLocaleString()}
                </div>
                <div>{notification.message}</div>
                <Button size="sm" variant="outline" onClick={() => toggleRead(notification.id)}>
                  {notification.read ? "Mark Unread" : "Mark Read"}
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
