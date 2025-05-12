import { useState } from 'react'
import './App.css'
import ChatWindow from "./ChatWindow.jsx";

function App() {

  return (
    <>
      {/*<Header />*/}
      <div className="content">
        <ChatWindow />
      </div>
    </>
  )
}

export default App
