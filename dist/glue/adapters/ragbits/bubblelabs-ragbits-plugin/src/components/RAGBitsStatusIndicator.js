"use strict";
// RAGBits Status Indicator Component
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RAGBitsStatusIndicator = void 0;
const react_1 = __importDefault(require("react"));
const lucide_react_1 = require("lucide-react");
const RAGBitsStatusIndicator = ({ status, className = '', showDetails = false }) => {
    const getStatusConfig = () => {
        switch (status) {
            case 'idle':
                return {
                    icon: lucide_react_1.Clock,
                    color: 'gray',
                    label: 'Idle'
                };
            case 'initializing':
                return {
                    icon: lucide_react_1.Clock,
                    color: 'blue',
                    label: 'Initializing'
                };
            case 'ready':
                return {
                    icon: lucide_react_1.CheckCircle,
                    color: 'green',
                    label: 'Ready'
                };
            case 'error':
                return {
                    icon: lucide_react_1.XCircle,
                    color: 'red',
                    label: 'Error'
                };
            case 'busy':
                return {
                    icon: lucide_react_1.AlertCircle,
                    color: 'yellow',
                    label: 'Busy'
                };
            default:
                return {
                    icon: lucide_react_1.Clock,
                    color: 'gray',
                    label: 'Unknown'
                };
        }
    };
    const config = getStatusConfig();
    const Icon = config.icon;
    return (<div className={`ragbits-status-indicator ${className}`} data-status={status}>
      <Icon className={`status-icon status-${config.color}`}/>
      {showDetails && (<span className="status-label">{config.label}</span>)}
    </div>);
};
exports.RAGBitsStatusIndicator = RAGBitsStatusIndicator;
//# sourceMappingURL=RAGBitsStatusIndicator.js.map