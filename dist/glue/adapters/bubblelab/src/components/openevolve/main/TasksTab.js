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
exports.TasksTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const select_1 = require("@/components/ui/select");
const TASK_STORAGE_KEY = "openevolve_tasks";
const STATUS_OPTIONS = ["To Do", "In Progress", "On Hold", "Completed"];
const readStorage = (key, fallback) => {
    try {
        const raw = globalThis.localStorage?.getItem(key);
        if (!raw) {
            return fallback;
        }
        return JSON.parse(raw);
    }
    catch {
        return fallback;
    }
};
const writeStorage = (key, value) => {
    try {
        globalThis.localStorage?.setItem(key, JSON.stringify(value));
    }
    catch {
        // ignore storage errors
    }
};
const generateId = () => {
    if (crypto?.randomUUID) {
        return crypto.randomUUID();
    }
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};
const TasksTab = () => {
    const [tasks, setTasks] = (0, react_1.useState)(() => readStorage(TASK_STORAGE_KEY, []));
    const [title, setTitle] = (0, react_1.useState)("");
    const [assignee, setAssignee] = (0, react_1.useState)("");
    const [description, setDescription] = (0, react_1.useState)("");
    const [dueDate, setDueDate] = (0, react_1.useState)("");
    const [assigneeFilter, setAssigneeFilter] = (0, react_1.useState)("");
    const [statusFilter, setStatusFilter] = (0, react_1.useState)("All");
    const persistTasks = (next) => {
        setTasks(next);
        writeStorage(TASK_STORAGE_KEY, next);
    };
    const createTask = () => {
        if (!title.trim()) {
            return;
        }
        const newTask = {
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
    const updateTask = (taskId, updates) => {
        const next = tasks.map((task) => (task.id === taskId ? { ...task, ...updates } : task));
        persistTasks(next);
    };
    const deleteTask = (taskId) => {
        persistTasks(tasks.filter((task) => task.id !== taskId));
    };
    const filteredTasks = (0, react_1.useMemo)(() => {
        return tasks.filter((task) => {
            const matchesAssignee = assigneeFilter
                ? (task.assignee || "").toLowerCase().includes(assigneeFilter.toLowerCase())
                : true;
            const matchesStatus = statusFilter === "All" ? true : task.status === statusFilter;
            return matchesAssignee && matchesStatus;
        });
    }, [tasks, assigneeFilter, statusFilter]);
    const groupedTasks = (0, react_1.useMemo)(() => {
        const groups = {
            "To Do": [],
            "In Progress": [],
            "On Hold": [],
            "Completed": [],
        };
        filteredTasks.forEach((task) => groups[task.status].push(task));
        return groups;
    }, [filteredTasks]);
    const stats = (0, react_1.useMemo)(() => {
        const total = tasks.length;
        const completed = tasks.filter((task) => task.status === "Completed").length;
        const inProgress = tasks.filter((task) => task.status === "In Progress").length;
        const overdue = tasks.filter((task) => {
            if (!task.due_date || task.status === "Completed")
                return false;
            return new Date(task.due_date) < new Date();
        }).length;
        return { total, completed, inProgress, overdue };
    }, [tasks]);
    const completionPercent = stats.total ? (stats.completed / stats.total) * 100 : 0;
    const markAllCompleted = () => {
        const next = tasks.map((task) => ({ ...task, status: "Completed" }));
        persistTasks(next);
    };
    const clearCompleted = () => {
        persistTasks(tasks.filter((task) => task.status !== "Completed"));
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Task Management</card_1.CardTitle>
          <card_1.CardDescription>Create, filter, and track execution tasks.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>Task Title</label_1.Label>
              <input_1.Input value={title} onChange={(event) => setTitle(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Assignee</label_1.Label>
              <input_1.Input value={assignee} onChange={(event) => setAssignee(event.target.value)}/>
            </div>
            <div className="space-y-2 md:col-span-2">
              <label_1.Label>Description</label_1.Label>
              <textarea_1.Textarea value={description} onChange={(event) => setDescription(event.target.value)} className="min-h-[100px]"/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Due Date</label_1.Label>
              <input_1.Input type="date" value={dueDate} onChange={(event) => setDueDate(event.target.value)}/>
            </div>
            <div className="flex items-end">
              <button_1.Button onClick={createTask}>Create Task</button_1.Button>
            </div>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Task List</card_1.CardTitle>
          <card_1.CardDescription>Filter and manage active tasks.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>Filter by Assignee</label_1.Label>
              <input_1.Input value={assigneeFilter} onChange={(event) => setAssigneeFilter(event.target.value)} placeholder="All assignees"/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Status</label_1.Label>
              <select_1.Select value={statusFilter} onValueChange={setStatusFilter}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue />
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  <select_1.SelectItem value="All">All</select_1.SelectItem>
                  {STATUS_OPTIONS.map((status) => (<select_1.SelectItem key={status} value={status}>
                      {status}
                    </select_1.SelectItem>))}
                </select_1.SelectContent>
              </select_1.Select>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {STATUS_OPTIONS.map((status) => (<card_1.Card key={status}>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">
                    {status} <badge_1.Badge variant="secondary">{groupedTasks[status].length}</badge_1.Badge>
                  </card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2">
                  {groupedTasks[status].length === 0 && (<div className="text-sm text-muted-foreground">No tasks.</div>)}
                  {groupedTasks[status].map((task) => (<div key={task.id} className="rounded border p-3 text-sm space-y-2">
                      <div className="font-semibold">{task.title}</div>
                      {task.description ? (<div className="text-xs text-muted-foreground">{task.description}</div>) : null}
                      <div className="text-xs text-muted-foreground">
                        Assignee: {task.assignee || "Unassigned"}
                      </div>
                      {task.due_date ? (<div className="text-xs text-muted-foreground">
                          Due: {new Date(task.due_date).toLocaleDateString()}
                        </div>) : null}
                      <div className="flex items-center gap-2">
                        <select_1.Select value={task.status} onValueChange={(value) => updateTask(task.id, { status: value })}>
                          <select_1.SelectTrigger className="h-8">
                            <select_1.SelectValue />
                          </select_1.SelectTrigger>
                          <select_1.SelectContent>
                            {STATUS_OPTIONS.map((option) => (<select_1.SelectItem key={option} value={option}>
                                {option}
                              </select_1.SelectItem>))}
                          </select_1.SelectContent>
                        </select_1.Select>
                        <button_1.Button variant="ghost" onClick={() => deleteTask(task.id)}>
                          Delete
                        </button_1.Button>
                      </div>
                    </div>))}
                </card_1.CardContent>
              </card_1.Card>))}
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Task Statistics</card_1.CardTitle>
          <card_1.CardDescription>Completion progress and overdue tracking.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
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
              <div className="h-2 rounded bg-green-500" style={{ width: `${completionPercent}%` }}/>
            </div>
          </div>

          <div className="flex gap-2">
            <button_1.Button onClick={markAllCompleted}>Mark All Completed</button_1.Button>
            <button_1.Button variant="outline" onClick={clearCompleted}>
              Clear Completed
            </button_1.Button>
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.TasksTab = TasksTab;
//# sourceMappingURL=TasksTab.js.map