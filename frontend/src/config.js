// Config starter code
import React from "react";
import { createChatBotMessage } from "react-chatbot-kit";
import BotAvatar from "./botAvatar";
import Buttons from "./buttons";

const config = {
    botName: "VedicGuru",
    initialMessages: [
        createChatBotMessage(`Hello, how may I help you today?`)
    ],
    customComponents: {
        botAvatar: (props) => <BotAvatar {...props} />,
    },
    customStyles: {
        botMessageBox: {
          backgroundColor: "#9db5db",
        },
        chatButton: {
          backgroundColor: "#4f5f78",
        },
    },
    widgets: [
      {
        widgetName: "Feedback",
        widgetFunc: (props) => <Buttons {...props} />
      }
    ]
}

export default config
