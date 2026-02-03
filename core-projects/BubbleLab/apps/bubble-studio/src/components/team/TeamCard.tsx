/**
 * TeamCard Component
 * Display team configuration summary
 */

interface TeamMember {
  name: string;
  model: string;
  temperature: number;
}

interface TeamCardProps {
  team: {
    id: string;
    name: string;
    description?: string;
    members: TeamMember[];
    created_at: string;
  };
  onEdit?: (id: string) => void;
  onDelete?: (id: string) => void;
}

export function TeamCard({ team, onEdit, onDelete }: TeamCardProps) {
  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {team.name}
          </h3>
          {team.description && (
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {team.description}
            </p>
          )}
        </div>
      </div>

      {/* Members Preview */}
      <div className="mb-3">
        <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Members ({team.members.length})
        </p>
        <div className="flex flex-wrap gap-2">
          {team.members.slice(0, 3).map((member, index) => (
            <span
              key={index}
              className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300"
            >
              {member.name}
            </span>
          ))}
          {team.members.length > 3 && (
            <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300">
              +{team.members.length - 3} more
            </span>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-200 dark:border-gray-700">
        <span className="text-sm text-gray-500 dark:text-gray-400">
          {team.members.length} {team.members.length === 1 ? 'member' : 'members'}
        </span>
        <div className="flex items-center gap-2">
          {onEdit && (
            <button
              onClick={() => onEdit(team.id)}
              className="px-3 py-1 text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
            >
              Edit
            </button>
          )}
          {onDelete && (
            <button
              onClick={() => onDelete(team.id)}
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
