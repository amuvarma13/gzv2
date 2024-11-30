import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ReactMediaRecorder } from 'react-media-recorder';

const fetchConversationData = (conversationId) => {
  const conversation = [
    {
      user: "Hello, can you help me with something?",
      assistant: "Of course! I'm here to help. What would you like to know?"
    },
    {
      user: "How do I learn React?",
      assistant: "React is a JavaScript library for building user interfaces. The best way to learn is through practice and building projects like this one."
    },
    {
      user: "Could you explain components?",
      assistant: "Components are the building blocks of React applications. They let you split the UI into independent, reusable pieces that can be handled separately."
    }
  ];

  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(conversation);
    }, 500);
  });
};

const Message = ({ message, onStop }) => (
  <ReactMediaRecorder
    audio
    onStop={onStop}
    render={({ status, startRecording, stopRecording }) => (
      <div className="p-4 bg-gray-50 border rounded-lg">
        <div className="flex items-center justify-between">
          <div>{message}</div>
          <button
            onClick={status === 'recording' ? stopRecording : startRecording}
            className="p-2"
            aria-label={status === 'recording' ? "Stop Recording" : "Start Recording"}
          >
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              width="20" 
              height="20" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke={status === 'recording' ? "#EF4444" : "currentColor"}
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
              className="hover:scale-110 transition-transform"
            >
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
              <line x1="12" x2="12" y1="19" y2="22" />
            </svg>
          </button>
        </div>
      </div>
    )}
  />
);

const Conversation = ({ conversationId }) => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const loadConversation = async () => {
      setLoading(true);
      const data = await fetchConversationData(conversationId);
      setMessages(data);
      setLoading(false);
    };

    loadConversation();
  }, [conversationId]);

  const handleStop = (blobUrl, blob) => {
    console.log('Recording stopped:', { blobUrl, blob });
    // Handle the recorded audio here
  };

  const handleNext = () => {
    navigate('/conversations/456');
  };

  if (loading) return <div className="text-center p-4">Loading...</div>;

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6">
            <div className="space-y-6">
              {messages.map((message, index) => (
                <div key={index} className="space-y-2">
                  <div className="text-sm text-gray-600">
                    {message.user}
                  </div>
                  <Message 
                    message={message.assistant} 
                    onStop={handleStop}
                  />
                </div>
              ))}
            </div>
          </div>
          <div className="border-t p-4 flex justify-end">
            <button 
              onClick={handleNext}
              className="px-4 py-2 border rounded-md hover:bg-gray-50 flex items-center"
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Conversation;