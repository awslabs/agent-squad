import {  ChatHistory, ConversationMessage, ParticipantRole } from "../types";
import { Agent, AgentOptions } from "./agent";
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";

export interface LambdaAgentOptions extends AgentOptions {
    functionName: string;
    functionRegion: string;
    inputPayloadEncoder?: (inputText: string, ...additionalParams: any) => any; 
    outputPayloadDecoder?: (response: any) => ConversationMessage;
}

export class LambdaAgent extends Agent {
    private options: LambdaAgentOptions;
    private lambdaClient: LambdaClient;

    constructor(options: LambdaAgentOptions) {
        super(options);
        this.options = options;
        this.lambdaClient = new LambdaClient({region:this.options.functionRegion});
    }

    private defaultInputPayloadEncoder(inputText: string, chatHistory: ConversationMessage[], userId: string, sessionId:string, additionalParams?: Record<string, string>):string {
        return JSON.stringify({
            query: inputText,
            chatHistory: chatHistory,
            additionalParams: additionalParams,
            userId: userId,
            sessionId: sessionId,
        });
    }

    private defaultOutputPayloaderDecoder(response: any): ConversationMessage {
        const decodedResponse = JSON.parse(JSON.parse(new TextDecoder("utf-8").decode(response.Payload)).body).response;
        const message: ConversationMessage = {
            role: ParticipantRole.ASSISTANT,
            content: [{ text: `${decodedResponse}` }]
        };
        return message;
    }

    async processRequest(
        inputText: string,
        userId: string,
        sessionId: string,
        chatHistory: ChatHistory,
        additionalParams?: Record<string, string>
      ): Promise<ConversationMessage>{

        const payload = this.options.inputPayloadEncoder ? this.options.inputPayloadEncoder(inputText, chatHistory, userId, sessionId, additionalParams) : this.defaultInputPayloadEncoder(inputText, chatHistory.messages, userId, sessionId, additionalParams);
        const invokeParams = {
            FunctionName: this.options.functionName,
            Payload: payload,
        };
        
        const response = await this.lambdaClient.send(new InvokeCommand(invokeParams));
        
        return new Promise((resolve) => {
            const message = this.options.outputPayloadDecoder ? this.options.outputPayloadDecoder(response) : this.defaultOutputPayloaderDecoder(response);
            resolve(message);
          });
      }
}
