// ActionProvider starter code
class ActionProvider {
    constructor(
        createChatBotMessage,
        setStateFunc,
        createClientMessage,
        stateRef,
        createCustomMessage,
        ...rest
    ) {
        this.createChatBotMessage = createChatBotMessage;
        this.setState = setStateFunc;
        this.createClientMessage = createClientMessage;
        this.stateRef = stateRef;
        this.createCustomMessage = createCustomMessage;
    }

    helloWorldHandler = (que) => {
        // console.log("https://8714-34-87-94-191.ngrok.io/api/" + que);
        fetch("https://127.0.0.1:8000/api/" + que)
            .then((res) => res.json())
            .then((data) => {
                console.log(data.answer);

                const message = this.createChatBotMessage(data.answer, { widget: "Feedback" });
                this.setChatBotMessage(message);
            });

        // console.log(this.stateRef);
        // const message = this.createChatBotMessage("hello how are you?", { widget: "Feedback" });
        // this.setChatBotMessage(message);
    };

    msgHandler = (que) => {
        fetch("http://10.1.19.16:8001/api/" + que)
            .then((res) => res.json())
            .then((data) => {
                console.log(data.answer);

                const message = this.createChatBotMessage(data.answer, { widget: "Feedback" });
                this.setChatBotMessage(message);
            });
        // console.log(this.stateRef);
        // const message = this.createChatBotMessage(
        //     <div>
        //     Seeking medical advice or consulting a healthcare professional for irregular periods is important if you experience any of the following symptoms: <br/>
        //     * Abnormal bleeding or spotting, such as heavy or prolonged menstrual bleeding, irregular intervals between periods, or absence of periods for more than three months <br/>
        //     * Heavy or prolonged menstrual bleeding that interferes with daily activities or causes anemia <br/>
        //     * Changes in menstrual cycle length or duration, such as shorter or longer cycles than usual <br/>
        //     * Other symptoms like pelvic pain, breast tenderness, mood swings, or acne <br/>
        //     It is essential to consult a healthcare professional if you have not yet reached menopause but are experiencing irregular periods, as it could indicate an underlying hormonal imbalance or another condition that requires proper evaluation and management. Early diagnosis and treatment can help alleviate symptoms, improve quality of life, and prevent potential complications associated with PCOS. <br/>
        //     <b>Note: This is an AI generated response. For more clarifications, request you to please seek medical advice.</b>
        //     </div>, { 
        //     widget: "Feedback",
        // });
        // this.setChatBotMessage(message);
    }

    setChatBotMessage = (message) => {
        this.setState(state => ({ ...state, messages: [...state.messages, message] }));
    };
}
 
 export default ActionProvider;
