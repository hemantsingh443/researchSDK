export const formatBytes = (bytes: number, decimals = 2): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
  
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
};

export const formatDate = (dateValue: string | number): string => {
  const date = typeof dateValue === 'string' ? new Date(dateValue) : new Date(dateValue * 1000);
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
};

// Check if a filename is a markdown file
export const isMarkdownFile = (filename: string): boolean => {
  if (!filename) return false;
  const ext = filename.split('.').pop()?.toLowerCase();
  return ['md', 'markdown', 'mdown', 'mkd', 'mkdn'].includes(ext || '');
};

// Check if a file is a text-based file that could be previewed
export const isTextFile = (filename: string): boolean => {
  if (!filename) return false;
  const ext = filename.split('.').pop()?.toLowerCase();
  const textExtensions = [
    'txt', 'md', 'markdown', 'mdown', 'mkd', 'mkdn',
    'json', 'yaml', 'yml', 'xml', 'html', 'htm', 'css', 'scss', 'sass',
    'js', 'ts', 'jsx', 'tsx', 'py', 'java', 'c', 'cpp', 'h', 'hpp', 'cs',
    'go', 'rb', 'php', 'sh', 'bash', 'zsh', 'fish', 'ps1', 'bat', 'cmd',
    'log', 'ini', 'conf', 'config', 'toml', 'env', 'gitignore', 'dockerignore',
    'editorconfig', 'eslintrc', 'prettierrc', 'babelrc', 'jsonc', 'json5'
  ];
  return textExtensions.includes(ext || '');
};
