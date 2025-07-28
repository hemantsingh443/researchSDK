import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

// Generate a simple client ID for WebSocket connection
const clientId = Math.random().toString(36).substring(2, 15);

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
};

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addMessage = useCallback((text: string, sender: 'user' | 'bot') => {
    const newMessage: Message = {
      id: Date.now().toString(),
      text,
      sender,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newMessage]);
  }, []);

  // Initialize WebSocket connection with reconnection
  const connectWebSocket = useCallback(() => {
    // Use wss:// for production, ws:// for development
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/${clientId}`;
    
    console.log('Connecting to WebSocket:', wsUrl);
    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      addMessage('Connected to Paper Agent SDK', 'bot');
    };

    ws.current.onmessage = (event) => {
      console.log('WebSocket message received:', event.data);
      try {
        const data = JSON.parse(event.data);
        
        // Handle different message types from the backend
        if (data.type === 'task_update') {
          if (data.status === 'in_progress' && data.step) {
            addMessage(data.step.content, 'bot');
          } else if (data.status === 'completed' && data.result) {
            addMessage(data.result, 'bot');
            // You could also handle artifacts here if needed
          } else if (data.artifacts && data.artifacts.length > 0) {
            // Handle any artifacts if needed
            console.log('Artifacts generated:', data.artifacts);
          }
        } else if (data.type === 'status') {
          addMessage(data.message, 'bot');
        } else if (data.type === 'error') {
          addMessage(`Error: ${data.error || data.message || 'Unknown error'}`, 'bot');
          setIsLoading(false);
        } else {
          console.log('Unknown message type:', data);
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error, event.data);
        addMessage('Error processing server response', 'bot');
        setIsLoading(false);
      }
    };

    ws.current.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      setIsConnected(false);
      if (event.code !== 1000) { // Only show reconnecting message if not a normal close
        addMessage('Disconnected from server. Reconnecting...', 'bot');
        
        // Attempt to reconnect after a delay
        setTimeout(() => {
          console.log('Attempting to reconnect...');
          connectWebSocket();
        }, 3000);
      }
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      addMessage('Connection error', 'bot');
      setIsLoading(false);
    };
  }, [addMessage]);

  // Initialize WebSocket connection
  useEffect(() => {
    connectWebSocket();
    
    // Cleanup function
    return () => {
      if (ws.current) {
        console.log('Cleaning up WebSocket');
        ws.current.close(1000, 'Component unmounting');
      }
    };
  }, [connectWebSocket]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !isConnected || !ws.current) return;

    // Add user message to chat
    addMessage(input, 'user');
    
    // Send message to server
    ws.current.send(JSON.stringify({
      type: 'new_query',
      query: input,
    }));

    setIsLoading(true);
    setInput('');
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <header className="bg-white shadow-sm p-4">
        <h1 className="text-xl font-semibold text-gray-800">Paper Agent</h1>
        <div className="flex items-center text-sm text-gray-500 mt-1">
          <div className={`w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </header>

      <main className="flex-1 p-4 overflow-y-auto">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs md:max-w-md lg:max-w-lg xl:max-w-xl rounded-lg px-4 py-2 ${
                  message.sender === 'user'
                    ? 'bg-blue-500 text-white rounded-br-none'
                    : 'bg-white text-gray-800 shadow rounded-bl-none'
                }`}
              >
                <p className="whitespace-pre-wrap break-words">{message.text}</p>
                <p className="text-xs opacity-70 mt-1">
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white text-gray-800 shadow rounded-lg px-4 py-2 rounded-bl-none">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="bg-white border-t p-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={!isConnected || isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || !isConnected || isLoading}
            className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </form>
      </footer>
    </div>
  )
}

export default App
