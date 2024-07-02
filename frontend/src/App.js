import React from 'react';
import { Chatbot } from 'react-chatbot-kit';
import 'react-chatbot-kit/build/main.css'

import './styles/ChatBot.css';

import MessageParser from './messageParser';
import ActionProvider from './actionProvider';
import config from './config';

import './App.css';

function App() {
  return (
    <div className="App">
      <div></div>
      <header className="App-header">
      
        <Chatbot config={config} messageParser={MessageParser} actionProvider={ActionProvider} />
      </header>
    </div>
  );
}

export default App;
