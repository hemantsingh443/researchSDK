import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  return (
    <div className={`prose max-w-none ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          // Customize heading styles
          h1: ({ node, ...props }) => <h1 className="text-2xl font-bold my-4" {...props} />,
          h2: ({ node, ...props }) => <h2 className="text-xl font-semibold my-3" {...props} />,
          h3: ({ node, ...props }) => <h3 className="text-lg font-medium my-2" {...props} />,
          
          // Style links
          a: ({ node, ...props }) => (
            <a 
              className="text-blue-500 hover:text-blue-600 underline" 
              target="_blank" 
              rel="noopener noreferrer"
              {...props} 
            />
          ),
          
          // Style code blocks
          code: ({ node, className, children, ...props }: any) => {
            const isInline = !className;
            if (isInline) {
              return (
                <code className="bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded text-sm" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto my-4">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            );
          },
          
          // Style blockquotes
          blockquote: ({ node, ...props }) => (
            <blockquote 
              className="border-l-4 border-gray-300 dark:border-gray-600 pl-4 my-4 text-gray-700 dark:text-gray-300" 
              {...props} 
            />
          ),
          
          // Style lists
          ul: ({ node, ...props }) => (
            <ul className="list-disc pl-6 my-2 space-y-1" {...props} />
          ),
          ol: ({ node, ...props }) => (
            <ol className="list-decimal pl-6 my-2 space-y-1" {...props} />
          ),
          
          // Style tables
          table: ({ node, ...props }) => (
            <div className="overflow-x-auto my-4">
              <table className="min-w-full border-collapse" {...props} />
            </div>
          ),
          thead: ({ node, ...props }) => (
            <thead className="bg-gray-100 dark:bg-gray-700" {...props} />
          ),
          th: ({ node, ...props }) => (
            <th 
              className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left" 
              {...props} 
            />
          ),
          td: ({ node, ...props }) => (
            <td 
              className="border border-gray-300 dark:border-gray-600 px-4 py-2" 
              {...props} 
            />
          ),
          
          // Style images
          img: ({ node, ...props }) => (
            <div className="my-4 flex justify-center">
              <img 
                className="max-w-full h-auto rounded-lg shadow-md" 
                alt={props.alt || ''} 
                {...props} 
              />
            </div>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
