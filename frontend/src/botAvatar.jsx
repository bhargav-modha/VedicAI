import React from "react";
import "./styles/BotAvatar.css";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faRobot } from "@fortawesome/free-solid-svg-icons";

// overwrite the existing avatar

const BotAvatar = () => {
  return <div className="chatbot-avatar"> <FontAwesomeIcon icon={faRobot} /> </div>;
};

export default BotAvatar;
