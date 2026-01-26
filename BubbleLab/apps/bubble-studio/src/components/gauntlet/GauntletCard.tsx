/**
 * GauntletCard Component
 * Display gauntlet configuration summary
 */

interface GauntletRound {
  name: string;
  quorum: number;
  confidence_threshold: number;
}

interface GauntletCardProps {
  gauntlet: {
    id: string;
    name: string;
    description?: string;
    rounds: GauntletRound[];
    created_at: string;
  };
  onEdit?: (id: string) => void;
  onDelete?: (id: string) => void;
}

export function GauntletCard({ gauntlet, onEdit, onDelete }: GauntletCardProps) {
  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {gauntlet.name}
          </h3>
          {gauntlet.description && (
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {gauntlet.description}
            </p>
          )}
        </div>
      </div>

      {/* Rounds Preview */}
      <div className="mb-3">
        <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Rounds ({gauntlet.rounds.length})
        </p>
        <div className="space-y-1">
          {gauntlet.rounds.slice(0, 2).map((round, index) => (
            <div
              key={index}
              className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400"
            >
              <span>{round.name}</span>
              <span>
                Q: {round.quorum} • C: {(round.confidence_threshold * 100).toFixed(0)}%
              </span>
            </div>
          ))}
          {gauntlet.rounds.length > 2 && (
            <p className="text-xs text-gray-500 dark:text-gray-500">
              +{gauntlet.rounds.length - 2} more rounds
            </p>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-200 dark:border-gray-700">
        <span className="text-sm text-gray-500 dark:text-gray-400">
          {gauntlet.rounds.length} {gauntlet.rounds.length === 1 ? 'round' : 'rounds'}
        </span>
        <div className="flex items-center gap-2">
          {onEdit && (
            <button
              onClick={() => onEdit(gauntlet.id)}
              className="px-3 py-1 text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
            >
              Edit
            </button>
          )}
          {onDelete && (
            <button
              onClick={() => onDelete(gauntlet.id)}
              className="px-3 py-1 text-sm font-medium text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
            >
              Delete
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
