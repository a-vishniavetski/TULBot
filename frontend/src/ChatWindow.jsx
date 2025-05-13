import React, { useState } from 'react';
import './ChatWindow.css';
import {BsX} from 'react-icons/bs';
import apiService from './services/api';

const ChatWindow = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  // Toggle chatbot window visibility
  const toggleChatWindow = () => {
    setIsOpen(!isOpen);
  };

  // Handle message submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    setMessages([...messages, { text: input, sender: 'user' }]);

    try {
      const data = await apiService.sendQuery(input);

      // Add bot response to the chat
      setMessages(prev => [...prev, {
        text: data.answer, 
        sender: 'bot',
        sources: data.sources
      }]);
      
    } catch (error) {
      console.error('Error fetching data:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: 'Error fetching data. Please try again.', sender: 'bot' },
      ]);
    }

    setInput('');
  };

  return (
    <div className="chatbot-container">
      {/* Chatbot toggle button */}
      {!isOpen && (
      <button onClick={toggleChatWindow} className={isOpen ? "chatbot-button-opened" : "chatbot-button-closed"} >
        <span>?</span>
      </button>
      )}

      {/* Chatbot window */}
      {isOpen && (
        <div className="chatbot-window">
          <img src="/src/assets/polibuda_logo.svg" alt="Logo" className="chatbot-bg-logo" />
          <div className="chatbot-header">
            <h2 className="relative">Chatbot</h2>
            <a onClick={toggleChatWindow} className={isOpen ? "chatbot-button-opened" : "chatbot-button-closed"}>
              <BsX className="text-3xl font-bold" />
            </a>
          </div>
          <div className="chatbot-messages">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
              >
                {message.text}
              </div>
            ))}
          </div>

          <form className="chatbot-input" onSubmit={handleSubmit}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type a message..."
            />
            <button classname="submit-button" type="submit">↑</button>
          </form>
        </div>
      )}
    </div>
  );
};

export default ChatWindow;