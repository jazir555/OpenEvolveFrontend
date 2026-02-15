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
import type { TaskItem, TaskStatus } from "../../../lib/types";

const TASK_STORAGE_KEY = "openevolve_tasks";
const STATUS_OPTIONS: TaskStatus[] = ["To Do", "In Progress", "On Hold", "Completed"];

const readStorage = <T,>(key: string, fallback: T): T => {
  try {
    const raw = globalThis.localStorage?.getItem(key);
    if (!raw) {
      return fallback;
    }
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

const writeStorage = (key: string, value: unknown) => {
  try {
    globalThis.localStorage?.setItem(key, JSON.stringify(value));
  } catch {
    // ignore storage errors
  }
};

const generateId = () => {
  if (crypto?.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

export const TasksTab: React.FC = () => {
  const [tasks, setTasks] = useState<TaskItem[]>(() => readStorage<TaskItem[]>(TASK_STORAGE_KEY, []));
  const [title, setTitle] = useState("");
  const [assignee, setAssignee] = useState("");
  const [description, setDescription] = useState("");
  const [dueDate, setDueDate] = useState("");
  const [assigneeFilter, setAssigneeFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");

  const persistTasks = (next: TaskItem[]) => {
    setTasks(next);
    writeStorage(TASK_STORAGE_KEY, next);
  };

  const createTask = () => {
    if (!title.trim()) {
      return;
    }
    const newTask: TaskItem = {
      id: generateId(),
      title: title.trim(),
      description: description.trim() || null,
      assignee: assignee.trim() || null,
      status: "To Do",
      due_date: dueDate ? new Date(dueDate).toISOString() : null,
      created_at: new Date().toISOString(),
    };
    persistTasks([newTask, ...tasks]);
    setTitle("");
    setAssignee("");
    setDescription("");
    setDueDate("");
  };

  const updateTask = (taskId: string, updates: Partial<TaskItem>) => {
    const next = tasks.map((task) => (task.id === taskId ? { ...task, ...updates } : task));
    persistTasks(next);
  };

  const deleteTask = (taskId: string) => {
    persistTasks(tasks.filter((task) => task.id !== taskId));
  };

  const filteredTasks = useMemo(() => {
    return tasks.filter((task) => {
      const matchesAssignee = assigneeFilter
        ? (task.assignee || "").toLowerCase().includes(assigneeFilter.toLowerCase())
        : true;
      const matchesStatus = statusFilter === "All" ? true : task.status === statusFilter;
      return matchesAssignee && matchesStatus;
    });
  }, [tasks, assigneeFilter, statusFilter]);

  const groupedTasks = useMemo(() => {
    const groups: Record<TaskStatus, TaskItem[]> = {
      "To Do": [],
      "In Progress": [],
      "On Hold": [],
      "Completed": [],
    };
    filteredTasks.forEach((task) => groups[task.status].push(task));
    return groups;
  }, [filteredTasks]);

  const stats = useMemo(() => {
    const total = tasks.length;
    const completed = tasks.filter((task) => task.status === "Completed").length;
    const inProgress = tasks.filter((task) => task.status === "In Progress").length;
    const overdue = tasks.filter((task) => {
      if (!task.due_date || task.status === "Completed") return false;
      return new Date(task.due_date) < new Date();
    }).length;
    return { total, completed, inProgress, overdue };
  }, [tasks]);

  const completionPercent = stats.total ? (stats.completed / stats.total) * 100 : 0;

  const markAllCompleted = () => {
    const next = tasks.map((task) => ({ ...task, status: "Completed" as TaskStatus }));
    persistTasks(next);
  };

  const clearCompleted = () => {
    persistTasks(tasks.filter((task) => task.status !== "Completed"));
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Task Management</CardTitle>
          <CardDescription>Create, filter, and track execution tasks.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Task Title</Label>
              <Input value={title} onChange={(event) => setTitle(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Assignee</Label>
              <Input value={assignee} onChange={(event) => setAssignee(event.target.value)} />
            </div>
            <div className="space-y-2 md:col-span-2">
              <Label>Description</Label>
              <Textarea
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                className="min-h-[100px]"
              />
            </div>
            <div className="space-y-2">
              <Label>Due Date</Label>
              <Input
                type="date"
                value={dueDate}
                onChange={(event) => setDueDate(event.target.value)}
              />
            </div>
            <div className="flex items-end">
              <Button onClick={createTask}>Create Task</Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Task List</CardTitle>
          <CardDescription>Filter and manage active tasks.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Filter by Assignee</Label>
              <Input
                value={assigneeFilter}
                onChange={(event) => setAssigneeFilter(event.target.value)}
                placeholder="All assignees"
              />
            </div>
            <div className="space-y-2">
              <Label>Status</Label>
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="All">All</SelectItem>
                  {STATUS_OPTIONS.map((status) => (
                    <SelectItem key={status} value={status}>
                      {status}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {STATUS_OPTIONS.map((status) => (
              <Card key={status}>
                <CardHeader>
                  <CardTitle className="text-sm">
                    {status} <Badge variant="secondary">{groupedTasks[status].length}</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {groupedTasks[status].length === 0 && (
                    <div className="text-sm text-muted-foreground">No tasks.</div>
                  )}
                  {groupedTasks[status].map((task) => (
                    <div key={task.id} className="rounded border p-3 text-sm space-y-2">
                      <div className="font-semibold">{task.title}</div>
                      {task.description ? (
                        <div className="text-xs text-muted-foreground">{task.description}</div>
                      ) : null}
                      <div className="text-xs text-muted-foreground">
                        Assignee: {task.assignee || "Unassigned"}
                      </div>
                      {task.due_date ? (
                        <div className="text-xs text-muted-foreground">
                          Due: {new Date(task.due_date).toLocaleDateString()}
                        </div>
                      ) : null}
                      <div className="flex items-center gap-2">
                        <Select
                          value={task.status}
                          onValueChange={(value) =>
                            updateTask(task.id, { status: value as TaskStatus })
                          }
                        >
                          <SelectTrigger className="h-8">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {STATUS_OPTIONS.map((option) => (
                              <SelectItem key={option} value={option}>
                                {option}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <Button variant="ghost" onClick={() => deleteTask(task.id)}>
                          Delete
                        </Button>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Task Statistics</CardTitle>
          <CardDescription>Completion progress and overdue tracking.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-4 text-sm">
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Total Tasks</div>
              <div className="text-lg font-semibold">{stats.total}</div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Completed</div>
              <div className="text-lg font-semibold">{stats.completed}</div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">In Progress</div>
              <div className="text-lg font-semibold">{stats.inProgress}</div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Overdue</div>
              <div className="text-lg font-semibold">{stats.overdue}</div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-xs text-muted-foreground">
              Completion: {completionPercent.toFixed(1)}%
            </div>
            <div className="h-2 w-full rounded bg-muted">
              <div
                className="h-2 rounded bg-green-500"
                style={{ width: `${completionPercent}%` }}
              />
            </div>
          </div>

          <div className="flex gap-2">
            <Button onClick={markAllCompleted}>Mark All Completed</Button>
            <Button variant="outline" onClick={clearCompleted}>
              Clear Completed
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
