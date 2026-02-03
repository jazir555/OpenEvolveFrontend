import { getAutoRefineEnabled } from '../Routing';

type RefinementNeededEvent = {
    reason?: string;
    overall_score?: number;
    weaknesses?: string[];
    friction_points?: string[];
    auto_refine?: boolean;
};

type RewardCalibrationRequest = {
    request_id?: string;
    option_a: string;
    option_b: string;
    confidence?: number;
    prompt?: string;
};

const DEFAULT_API_BASE = 'http://127.0.0.1:8000';
const EVENT_POLL_INTERVAL_MS = 4000;

function getApiBase(): string {
    return (window as any).__ICR_API_BASE || DEFAULT_API_BASE;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T | null> {
    try {
        const response = await fetch(url, init);
        if (!response.ok) return null;
        return await response.json() as T;
    } catch {
        return null;
    }
}

async function pollRefinementEvents() {
    const base = getApiBase();
    const events = await fetchJson<RefinementNeededEvent[]>(`${base}/icr/events/refinement-needed?limit=5`);
    if (!events || events.length === 0) return;

    events.forEach((event) => {
        window.dispatchEvent(new CustomEvent('icr:refinement-needed', { detail: event }));
    });
}

async function pollRewardCalibration() {
    const base = getApiBase();
    const request = await fetchJson<RewardCalibrationRequest>(`${base}/icr/reward-calibration/next`);
    if (!request || !request.option_a || !request.option_b) return;

    window.dispatchEvent(new CustomEvent('icr:reward-calibration', {
        detail: {
            requestId: request.request_id,
            optionA: request.option_a,
            optionB: request.option_b,
            confidence: request.confidence,
            prompt: request.prompt
        }
    }));
}

async function sendCalibrationResponse(detail: { requestId?: string; choice: string }) {
    const base = getApiBase();
    await fetchJson(`${base}/icr/reward-calibration/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            request_id: detail.requestId,
            choice: detail.choice
        })
    });
}

export function startIcrEventBridge(): void {
    setInterval(() => {
        const autoRefineEnabled = getAutoRefineEnabled();
        if (autoRefineEnabled) {
            pollRefinementEvents();
        }
        pollRewardCalibration();
    }, EVENT_POLL_INTERVAL_MS);

    window.addEventListener('icr:reward-calibration-response', (event: Event) => {
        const custom = event as CustomEvent<{ requestId?: string; choice: string }>;
        if (!custom.detail?.choice) return;
        sendCalibrationResponse(custom.detail);
    });
}
