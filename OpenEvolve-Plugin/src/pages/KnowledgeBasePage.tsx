/**
 * OpenEvolve Knowledge Base Component for BubbleLab
 * 
 * This component replaces the Streamlit-based knowledge base UI
 * with a React-based component for the BubbleLab plugin system.
 */

import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  Search,
  Plus,
  Edit3,
  Trash2,
  Settings,
  RotateCcw,
  BarChart3,
  Users,
  GitBranch,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Tag,
  Folder,
  FileText,
  Database
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend
} from 'recharts';
import { BubbleButton, BubbleCard, BubbleInput, BubbleSelect, BubbleTabs, BubbleTab } from '../components/bubblelab';

// Mock data interfaces - these would come from the actual OpenEvolve API
interface KnowledgeEntry {
  id: string;
  title: string;
  content: string;
  tags: string[];
  category: string;
  createdAt: Date;
  updatedAt: Date;
  author: string;
  status: 'draft' | 'published' | 'archived';
}

interface KnowledgeCategory {
  id: string;
  name: string;
  description: string;
  count: number;
}

const KnowledgeBasePage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('browse');

  // Pie chart colors
  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#6366F1'];

  const [entries, setEntries] = useState<KnowledgeEntry[]>([
    {
      id: 'kb-1',
      title: 'Evolution Algorithm Fundamentals',
      content: 'Evolutionary algorithms are population-based optimization techniques inspired by biological evolution...',
      tags: ['evolution', 'optimization', 'algorithms'],
      category: 'algorithms',
      createdAt: new Date(Date.now() - 86400000 * 7), // 7 days ago
      updatedAt: new Date(Date.now() - 3600000 * 2), // 2 hours ago
      author: 'Dr. Jane Smith',
      status: 'published'
    },
    {
      id: 'kb-2',
      title: 'Adversarial Testing Strategies',
      content: 'Adversarial testing involves creating inputs designed to fool machine learning models...',
      tags: ['adversarial', 'testing', 'security'],
      category: 'security',
      createdAt: new Date(Date.now() - 86400000 * 3), // 3 days ago
      updatedAt: new Date(Date.now() - 3600000 * 1), // 1 hour ago
      author: 'John Doe',
      status: 'published'
    },
    {
      id: 'kb-3',
      title: 'Decomposition Techniques',
      content: 'Problem decomposition involves breaking complex problems into smaller, manageable sub-problems...',
      tags: ['decomposition', 'problem-solving', 'techniques'],
      category: 'methods',
      createdAt: new Date(Date.now() - 86400000 * 1), // 1 day ago
      updatedAt: new Date(Date.now() - 3600000 * 1), // 1 hour ago
      author: 'Alice Johnson',
      status: 'published'
    }
  ]);
  
  const [categories, setCategories] = useState<KnowledgeCategory[]>([
    { id: 'cat-1', name: 'Algorithms', description: 'Algorithmic approaches and techniques', count: 12 },
    { id: 'cat-2', name: 'Security', description: 'Security and adversarial considerations', count: 8 },
    { id: 'cat-3', name: 'Methods', description: 'Problem-solving methodologies', count: 15 },
    { id: 'cat-4', name: 'Best Practices', description: 'Recommended practices and guidelines', count: 7 }
  ]);
  
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [newEntry, setNewEntry] = useState({
    title: '',
    content: '',
    category: '',
    tags: ''
  });
  const [isCreating, setIsCreating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Get all unique tags
  const allTags = Array.from(
    new Set(entries.flatMap(entry => entry.tags))
  ).sort();

  // Filter entries based on search, category, and tags
  const filteredEntries = entries.filter(entry => {
    const matchesSearch = entry.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         entry.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         entry.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesCategory = selectedCategory === 'all' || entry.category === selectedCategory;
    
    const matchesTags = selectedTags.length === 0 || 
                       selectedTags.every(tag => entry.tags.includes(tag));
    
    return matchesSearch && matchesCategory && matchesTags;
  });

  const handleCreateEntry = () => {
    if (!newEntry.title.trim() || !newEntry.content.trim()) {
      alert('Title and content are required');
      return;
    }

    const newEntryObj: KnowledgeEntry = {
      id: `kb-${Date.now()}`,
      title: newEntry.title,
      content: newEntry.content,
      tags: newEntry.tags.split(',').map(tag => tag.trim()).filter(tag => tag),
      category: newEntry.category,
      createdAt: new Date(),
      updatedAt: new Date(),
      author: 'Current User',
      status: 'published'
    };

    setEntries([newEntryObj, ...entries]);
    setNewEntry({ title: '', content: '', category: '', tags: '' });
    setIsCreating(false);
  };

  const handleDeleteEntry = (entryId: string) => {
    setEntries(entries.filter(entry => entry.id !== entryId));
  };

  const handleTagClick = (tag: string) => {
    if (selectedTags.includes(tag)) {
      setSelectedTags(selectedTags.filter(t => t !== tag));
    } else {
      setSelectedTags([...selectedTags, tag]);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'published': return 'bg-green-500';
      case 'draft': return 'bg-yellow-500';
      case 'archived': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'published': return 'Published';
      case 'draft': return 'Draft';
      case 'archived': return 'Archived';
      default: return status;
    }
  };

  return (
    <div className="knowledge-base-page p-6 bg-white dark:bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <BookOpen className="w-8 h-8 text-blue-600 dark:text-blue-400 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              OpenEvolve Knowledge Base
            </h1>
          </div>
          <div className="flex items-center space-x-3">
            <BubbleButton
              variant="outline"
              size="sm"
              className="flex items-center"
            >
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </BubbleButton>
          </div>
        </div>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Access and manage knowledge resources for OpenEvolve workflows
        </p>
      </div>

      {/* Search and Filters */}
      <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-gray-400" />
            </div>
            <BubbleInput
              type="text"
              placeholder="Search knowledge base..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 w-full"
            />
          </div>
        </div>
        
        <div>
          <BubbleSelect
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="w-full"
          >
            <option value="all">All Categories</option>
            {categories.map(category => (
              <option key={category.id} value={category.name.toLowerCase()}>
                {category.name}
              </option>
            ))}
          </BubbleSelect>
        </div>
        
        <div>
          <BubbleButton
            variant="default"
            onClick={() => setIsCreating(true)}
            className="w-full flex items-center justify-center"
          >
            <Plus className="w-4 h-4 mr-2" />
            New Entry
          </BubbleButton>
        </div>
      </div>

      {/* Tabs */}
      <BubbleTabs value={activeTab} onValueChange={setActiveTab}>
        <BubbleTab value="browse" label="Browse">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Tags Sidebar */}
            <div className="lg:col-span-1">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Tag className="w-5 h-5 mr-2" />
                Tags
              </h3>
              <div className="space-y-2">
                {allTags.map(tag => (
                  <button
                    key={tag}
                    onClick={() => handleTagClick(tag)}
                    className={`flex items-center w-full px-3 py-2 text-sm rounded-md transition-colors ${
                      selectedTags.includes(tag)
                        ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                        : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    <Tag className="w-4 h-4 mr-2" />
                    {tag}
                  </button>
                ))}
              </div>
            </div>

            {/* Entries List */}
            <div className="lg:col-span-3">
              {isCreating ? (
                <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Create New Knowledge Entry
                  </h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Title
                      </label>
                      <BubbleInput
                        type="text"
                        value={newEntry.title}
                        onChange={(e) => setNewEntry({...newEntry, title: e.target.value})}
                        placeholder="Enter entry title"
                        className="w-full"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Category
                      </label>
                      <BubbleSelect
                        value={newEntry.category}
                        onChange={(e) => setNewEntry({...newEntry, category: e.target.value})}
                        className="w-full"
                      >
                        <option value="">Select a category</option>
                        {categories.map(category => (
                          <option key={category.id} value={category.name.toLowerCase()}>
                            {category.name}
                          </option>
                        ))}
                      </BubbleSelect>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Tags (comma separated)
                      </label>
                      <BubbleInput
                        type="text"
                        value={newEntry.tags}
                        onChange={(e) => setNewEntry({...newEntry, tags: e.target.value})}
                        placeholder="e.g., algorithm, optimization, technique"
                        className="w-full"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Content
                      </label>
                      <textarea
                        value={newEntry.content}
                        onChange={(e) => setNewEntry({...newEntry, content: e.target.value})}
                        rows={8}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        placeholder="Enter the knowledge content here..."
                      />
                    </div>
                    
                    <div className="flex space-x-3 pt-4">
                      <BubbleButton
                        variant="default"
                        onClick={handleCreateEntry}
                        className="flex items-center"
                      >
                        <Plus className="w-4 h-4 mr-2" />
                        Create Entry
                      </BubbleButton>
                      
                      <BubbleButton
                        variant="outline"
                        onClick={() => setIsCreating(false)}
                        className="flex items-center"
                      >
                        Cancel
                      </BubbleButton>
                    </div>
                  </div>
                </div>
              ) : null}
              
              <div className="space-y-4">
                {filteredEntries.length > 0 ? (
                  filteredEntries.map(entry => (
                    <div 
                      key={entry.id} 
                      className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow"
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                            {entry.title}
                          </h3>
                          <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mb-2">
                            <span>{entry.author}</span>
                            <span className="mx-2">•</span>
                            <span>{entry.createdAt.toLocaleDateString()}</span>
                            <span className="mx-2">•</span>
                            <span className={`inline-flex items-center px-2 py-1 text-xs font-semibold rounded-full ${entry.status === 'published' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : entry.status === 'draft' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' : 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'}`}>
                              {getStatusText(entry.status)}
                            </span>
                          </div>
                        </div>
                        <div className="flex space-x-2">
                          <button className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300">
                            <Edit3 className="w-5 h-5" />
                          </button>
                          <button 
                            className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"
                            onClick={() => handleDeleteEntry(entry.id)}
                          >
                            <Trash2 className="w-5 h-5" />
                          </button>
                        </div>
                      </div>
                      
                      <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-2">
                        {entry.content.substring(0, 200)}...
                      </p>
                      
                      <div className="flex flex-wrap gap-2 mb-4">
                        {entry.tags.map(tag => (
                          <span 
                            key={tag} 
                            className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                      
                      <div className="flex justify-between items-center text-sm text-gray-500 dark:text-gray-400">
                        <span className="flex items-center">
                          <Folder className="w-4 h-4 mr-1" />
                          {entry.category}
                        </span>
                        <span>Last updated: {entry.updatedAt.toLocaleDateString()}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-12">
                    <FileText className="mx-auto h-12 w-12 text-gray-400" />
                    <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">No entries found</h3>
                    <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                      {searchTerm || selectedTags.length > 0 || selectedCategory !== 'all'
                        ? "No entries match your filters. Try adjusting your search."
                        : "Get started by creating a new knowledge entry."}
                    </p>
                    <div className="mt-6">
                      <BubbleButton
                        variant="default"
                        onClick={() => setIsCreating(true)}
                        className="flex items-center justify-center"
                      >
                        <Plus className="w-4 h-4 mr-2" />
                        Create Entry
                      </BubbleButton>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </BubbleTab>

        <BubbleTab value="categories" label="Categories">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {categories.map(category => (
              <div 
                key={category.id} 
                className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow"
              >
                <div className="flex items-center mb-4">
                  <Folder className="w-8 h-8 text-blue-500 mr-3" />
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {category.name}
                  </h3>
                </div>
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  {category.description}
                </p>
                <div className="flex justify-between items-center">
                  <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                    {category.count} entries
                  </span>
                  <button className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300 text-sm font-medium">
                    View
                  </button>
                </div>
              </div>
            ))}
          </div>
        </BubbleTab>

        <BubbleTab value="analytics" label="Analytics">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <BubbleCard className="p-5">
              <div className="flex items-center">
                <BookOpen className="w-8 h-8 text-blue-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Entries</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">{entries.length}</p>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-5">
              <div className="flex items-center">
                <Database className="w-8 h-8 text-green-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Categories</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">{categories.length}</p>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-5">
              <div className="flex items-center">
                <Tag className="w-8 h-8 text-purple-500 mr-3" />
              <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Unique Tags</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">{allTags.length}</p>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-5">
              <div className="flex items-center">
                <Users className="w-8 h-8 text-yellow-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Contributors</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {Array.from(new Set(entries.map(e => e.author))).length}
                  </p>
                </div>
              </div>
            </BubbleCard>
          </div>

          {/* Knowledge Distribution Chart */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
              <BarChart3 className="w-5 h-5 mr-2" />
              Knowledge Distribution
            </h3>
            <ResponsiveContainer width="100%" height={320}>
              <PieChart>
                <Pie
                  data={categories.map(cat => ({
                    name: cat.name,
                    value: cat.count
                  }))}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {categories.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgb(31 41 55)',
                    border: '1px solid rgb(75 85 99)',
                    borderRadius: '0.5rem',
                    color: '#F9FAFB'
                  }}
                  formatter={(value: number, name: string) => [`${value} entries`, name]}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </BubbleTab>
      </BubbleTabs>

      {/* Action Buttons */}
      <div className="mt-8 flex justify-end space-x-3">
        <BubbleButton
          variant="outline"
          className="flex items-center"
        >
          <RotateCcw className="w-4 h-4 mr-2" />
          Refresh
        </BubbleButton>
        <BubbleButton
          variant="default"
          onClick={() => setIsCreating(true)}
          className="flex items-center"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Entry
        </BubbleButton>
      </div>
    </div>
  );
};

export default KnowledgeBasePage;