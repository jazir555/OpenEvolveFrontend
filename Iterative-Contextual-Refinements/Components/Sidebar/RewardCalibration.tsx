import React, { useEffect, useState } from 'react';

type CalibrationChoice = 'A' | 'B' | 'defer';

interface CalibrationRequest {
    requestId?: string;
    optionA: string;
    optionB: string;
    confidence?: number;
    prompt?: string;
}

/**
 * Reward Calibration component
 * Shows preference selection when reward model confidence is low.
 */
export const RewardCalibration: React.FC = () => {
    const [request, setRequest] = useState<CalibrationRequest | null>(null);

    useEffect(() => {
        const handleRequest = (event: Event) => {
            const custom = event as CustomEvent<CalibrationRequest>;
            if (!custom.detail || !custom.detail.optionA || !custom.detail.optionB) return;
            setRequest(custom.detail);
        };

        window.addEventListener('icr:reward-calibration', handleRequest as EventListener);
        return () => window.removeEventListener('icr:reward-calibration', handleRequest as EventListener);
    }, []);

    const emitResponse = (choice: CalibrationChoice) => {
        if (!request) return;
        const detail = {
            requestId: request.requestId,
            choice,
            confidence: request.confidence ?? null
        };
        window.dispatchEvent(new CustomEvent('icr:reward-calibration-response', { detail }));
        setRequest(null);
    };

    return (
        <details className="sidebar-section" open={Boolean(request)}>
            <summary className="sidebar-section-header">Reward Calibration</summary>
            <div className="sidebar-section-content">
                {!request && (
                    <div className="reward-calibration-panel">
                        <span className="input-hint">
                            No calibration requests. This will appear when the reward model confidence is low.
                        </span>
                    </div>
                )}
                {request && (
                    <div className="reward-calibration-panel">
                        {request.prompt && (
                            <div className="reward-calibration-prompt">
                                <span className="input-label">Prompt</span>
                                <div className="reward-calibration-prompt-text">{request.prompt}</div>
                            </div>
                        )}
                        <div className="reward-calibration-meta">
                            <span className="reward-calibration-badge">Confidence: {request.confidence ?? 'unknown'}</span>
                        </div>
                        <div className="reward-calibration-options">
                            <div className="reward-calibration-option">
                                <span className="input-label">Option A</span>
                                <textarea
                                    className="input-base"
                                    value={request.optionA}
                                    readOnly
                                    rows={6}
                                />
                            </div>
                            <div className="reward-calibration-option">
                                <span className="input-label">Option B</span>
                                <textarea
                                    className="input-base"
                                    value={request.optionB}
                                    readOnly
                                    rows={6}
                                />
                            </div>
                        </div>
                        <div className="reward-calibration-actions">
                            <button type="button" className="button" onClick={() => emitResponse('A')}>
                                Prefer A
                            </button>
                            <button type="button" className="button" onClick={() => emitResponse('B')}>
                                Prefer B
                            </button>
                            <button type="button" className="button" onClick={() => emitResponse('defer')}>
                                Defer
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </details>
    );
};

export default RewardCalibration;
