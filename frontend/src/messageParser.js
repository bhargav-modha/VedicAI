// MessageParser starter code
class MessageParser {
    constructor(actionProvider, state) {
      this.actionProvider = actionProvider;
      this.state = state;
    }
  
    parse(message) {
        const msg = message.toLowerCase();

        this.actionProvider.msgHandler(msg);

        // if(msg.includes("hello")) {
            // this.actionProvider.helloWorldHandler(msg);
        // }
    }
  }
  
  export default MessageParser;